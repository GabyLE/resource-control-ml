from pathlib import Path
from collections import Counter, defaultdict
import polars as pl

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
# Support running this script either from the project root or from /scripts.
PROJECT_ROOT = next(
    (candidate for candidate in (SCRIPT_DIR, SCRIPT_DIR.parent)
     if (candidate / "data" / "processed").is_dir()),
    SCRIPT_DIR,
)
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

STORAGE_FILE = PROCESSED_DIR / "storage_state.parquet"
CANDIDATES_FILE = PROCESSED_DIR / "candidates.parquet"
DECISIONS_FILE = PROCESSED_DIR / "decisions.parquet"

# Expected physical positions in each box
EXPECTED_BOX_POSITIONS = set(range(1, 9))

# Column names used in the datasets
DECISION_COL = "decision_id"
RUN_COL = "run_id"
BOX_COL = "BoxID"
STORAGE_POSITION_COL = "BufPos"
BOX_POSITION_COL = "BoxPos"


def require_columns(df: pl.DataFrame, required: set[str], dataset_name: str):
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing columns in {dataset_name}: {sorted(missing)}\n"
            f"Available columns: {df.columns}"
        )


def valid_box_id(value):
    return value is not None and value > 0


def main():
    for path in (STORAGE_FILE, CANDIDATES_FILE):
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

    storage = pl.read_parquet(STORAGE_FILE)
    candidates = pl.read_parquet(CANDIDATES_FILE)

    require_columns(
        storage,
        {DECISION_COL, BOX_COL, STORAGE_POSITION_COL},
        "storage_state.parquet",
    )
    require_columns(
        candidates,
        {DECISION_COL, BOX_COL, BOX_POSITION_COL},
        "candidates.parquet",
    )

    # Bring run_id into the report from decisions.parquet when available.
    run_by_decision = {}
    if DECISIONS_FILE.exists():
        decisions = pl.read_parquet(DECISIONS_FILE)
        if DECISION_COL in decisions.columns and RUN_COL in decisions.columns:
            run_by_decision = dict(
                decisions.select(DECISION_COL, RUN_COL)
                .unique(subset=[DECISION_COL])
                .iter_rows()
            )

    storage_by_decision = defaultdict(list)
    for row in storage.select(DECISION_COL, BOX_COL, STORAGE_POSITION_COL).iter_rows(named=True):
        storage_by_decision[row[DECISION_COL]].append(row)

    candidates_by_decision = defaultdict(list)
    for row in candidates.select(DECISION_COL, BOX_COL, BOX_POSITION_COL).iter_rows(named=True):
        candidates_by_decision[row[DECISION_COL]].append(row)

    all_decision_ids = sorted(set(storage_by_decision) | set(candidates_by_decision))

    report_rows = []
    duplicate_storage_rows = []
    invalid_storage_rows = []
    missing_boxes_rows = []
    extra_boxes_rows = []
    missing_positions_rows = []
    duplicate_candidate_positions = []
    unexpected_positions_rows = []

    for decision_id in all_decision_ids:
        s_rows = storage_by_decision.get(decision_id, [])
        c_rows = candidates_by_decision.get(decision_id, [])
        run_id = run_by_decision.get(decision_id, "unknown")

        storage_ids = [r[BOX_COL] for r in s_rows if valid_box_id(r[BOX_COL])]
        storage_counts = Counter(storage_ids)
        storage_box_ids = set(storage_counts)

        candidate_ids = [r[BOX_COL] for r in c_rows if valid_box_id(r[BOX_COL])]
        candidate_box_ids = set(candidate_ids)

        duplicate_ids = {box_id: count for box_id, count in storage_counts.items() if count > 1}
        for box_id, count in duplicate_ids.items():
            duplicate_storage_rows.append({
                "run_id": run_id,
                "decision_id": decision_id,
                "BoxID": box_id,
                "storage_rows_for_box": count,
            })

        invalid_storage_count = sum(not valid_box_id(r[BOX_COL]) for r in s_rows)
        if invalid_storage_count:
            invalid_storage_rows.append({
                "run_id": run_id,
                "decision_id": decision_id,
                "invalid_or_empty_BoxID_rows": invalid_storage_count,
            })

        missing_boxes = storage_box_ids - candidate_box_ids
        extra_boxes = candidate_box_ids - storage_box_ids

        for box_id in sorted(missing_boxes):
            missing_boxes_rows.append({"run_id": run_id, "decision_id": decision_id, "BoxID": box_id})
        for box_id in sorted(extra_boxes):
            extra_boxes_rows.append({"run_id": run_id, "decision_id": decision_id, "BoxID": box_id})

        # Validate positions independently for every box represented in candidates.
        positions_by_box = defaultdict(list)
        for row in c_rows:
            box_id = row[BOX_COL]
            box_pos = row[BOX_POSITION_COL]
            if valid_box_id(box_id):
                positions_by_box[box_id].append(box_pos)

        for box_id in sorted(storage_box_ids | candidate_box_ids):
            observed = positions_by_box.get(box_id, [])
            observed_valid = {p for p in observed if p is not None}
            missing_positions = EXPECTED_BOX_POSITIONS - observed_valid
            unexpected_positions = observed_valid - EXPECTED_BOX_POSITIONS
            duplicates = [p for p, count in Counter(observed).items() if count > 1]

            if missing_positions:
                missing_positions_rows.append({
                    "run_id": run_id,
                    "decision_id": decision_id,
                    "BoxID": box_id,
                    "missing_BoxPos": sorted(missing_positions),
                    "observed_positions": sorted(observed_valid),
                })
            if unexpected_positions:
                unexpected_positions_rows.append({
                    "run_id": run_id,
                    "decision_id": decision_id,
                    "BoxID": box_id,
                    "unexpected_BoxPos": sorted(unexpected_positions),
                })
            if duplicates:
                duplicate_candidate_positions.append({
                    "run_id": run_id,
                    "decision_id": decision_id,
                    "BoxID": box_id,
                    "duplicated_BoxPos": duplicates,
                })

        expected_candidate_rows = len(storage_box_ids) * len(EXPECTED_BOX_POSITIONS)
        report_rows.append({
            "run_id": run_id,
            "decision_id": decision_id,
            "storage_rows": len(s_rows),
            "unique_storage_boxes": len(storage_box_ids),
            "duplicate_storage_BoxIDs": len(duplicate_ids),
            "invalid_storage_BoxID_rows": invalid_storage_count,
            "candidate_rows": len(c_rows),
            "unique_candidate_boxes": len(candidate_box_ids),
            "expected_candidate_rows": expected_candidate_rows,
            "candidate_row_difference": len(c_rows) - expected_candidate_rows,
            "missing_boxes_in_candidates": len(missing_boxes),
            "extra_boxes_in_candidates": len(extra_boxes),
        })

    report = pl.DataFrame(report_rows).sort(["run_id", "decision_id"])

    print("\n" + "=" * 90)
    print("STORAGE ↔ CANDIDATES INTEGRITY CHECK")
    print("=" * 90)
    print(f"Storage rows:    {storage.height:,}")
    print(f"Candidate rows:  {candidates.height:,}")
    print(f"Decisions checked: {len(all_decision_ids):,}")

    print("\n1) SUMMARY BY RUN")
    print(
        report.group_by("run_id").agg(
            pl.len().alias("decisions"),
            pl.col("storage_rows").sum().alias("storage_rows"),
            pl.col("unique_storage_boxes").sum().alias("storage_box_decision_pairs"),
            pl.col("candidate_rows").sum().alias("candidate_rows"),
            pl.col("expected_candidate_rows").sum().alias("expected_candidate_rows"),
            pl.col("duplicate_storage_BoxIDs").sum().alias("duplicate_box_occurrences"),
            pl.col("missing_boxes_in_candidates").sum().alias("missing_box_pairs"),
            pl.col("extra_boxes_in_candidates").sum().alias("extra_box_pairs"),
        ).sort("run_id")
    )

    print("\n2) DECISIONS WHERE STORAGE DOES NOT CONTAIN 20 DISTINCT BOXES")
    print(
        report.filter(pl.col("unique_storage_boxes") != 20).select(
            "run_id", "decision_id", "storage_rows", "unique_storage_boxes",
            "duplicate_storage_BoxIDs", "invalid_storage_BoxID_rows"
        )
    )

    print("\n3) DECISIONS WHERE CANDIDATE ROW COUNT DOES NOT MATCH 8 × UNIQUE STORAGE BOXES")
    print(
        report.filter(pl.col("candidate_row_difference") != 0).select(
            "run_id", "decision_id", "unique_storage_boxes", "candidate_rows",
            "expected_candidate_rows", "candidate_row_difference"
        )
    )

    def print_detail(title, rows):
        print(f"\n{title}")
        if rows:
            print(pl.DataFrame(rows))
        else:
            print("None")

    print_detail("4) DUPLICATE BoxIDs IN STORAGE", duplicate_storage_rows)
    print_detail("5) EMPTY / INVALID BoxIDs IN STORAGE", invalid_storage_rows)
    print_detail("6) STORAGE BOXES MISSING FROM CANDIDATES", missing_boxes_rows)
    print_detail("7) BOXES IN CANDIDATES BUT NOT IN STORAGE", extra_boxes_rows)
    print_detail("8) BOXES WITH MISSING BoxPos VALUES", missing_positions_rows)
    print_detail("9) DUPLICATE (BoxID, BoxPos) CANDIDATE ROWS", duplicate_candidate_positions)
    print_detail("10) UNEXPECTED BoxPos VALUES (EXPECTED 1–8)", unexpected_positions_rows)

    # Save detailed per-decision report for filtering/sorting in a spreadsheet.
    output_path = PROCESSED_DIR / "storage_candidate_integrity_report.parquet"
    report.write_parquet(output_path)
    print(f"\nDetailed per-decision report saved to:\n{output_path}")

    # Overall interpretation
    print("\n11) QUICK INTERPRETATION")
    if duplicate_storage_rows:
        print("- At least one BoxID occupies multiple Storage rows for the same decision.")
    if invalid_storage_rows:
        print("- At least one Storage row has an empty/invalid BoxID.")
    if missing_boxes_rows:
        print("- At least one Storage box has no corresponding candidate rows.")
    if missing_positions_rows:
        print("- At least one box does not have all eight candidate positions.")
    if extra_boxes_rows:
        print("- At least one candidate box is not present in the Storage snapshot.")
    if not any([
        duplicate_storage_rows, invalid_storage_rows, missing_boxes_rows,
        missing_positions_rows, extra_boxes_rows, duplicate_candidate_positions,
        unexpected_positions_rows,
    ]):
        print("- No structural inconsistencies were detected.")


if __name__ == "__main__":
    main()
