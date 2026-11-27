from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
DATA_DIR = ROOT / "data" / "processed"

RESOURCE_ID = 17
BOX_TYPE_PNO = 29
RELEVANT_BUFFERS = [1, 3]  # 1 = Storage, 3 = Pick & Sort
EVENT_WINDOW_MINUTES = 5


# ============================================================
# HELPERS
# ============================================================

def load_table(run_id: str, table_name: str) -> pl.DataFrame:
    """Load all parquet parts for one run and attach run_id."""
    folder = RAW_DIR / run_id / table_name
    files = sorted(folder.rglob("*.parquet"))

    if not files:
        raise FileNotFoundError(f"No Parquet files found in {folder}")

    frames = []
    for file in files:
        frame = pl.read_parquet(file)
        if "run_id" not in frame.columns:
            frame = frame.with_columns(pl.lit(run_id).alias("run_id"))
        frames.append(frame)

    return pl.concat(frames, how="diagonal_relaxed")


def parse_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value

    value = str(value).strip().replace("Z", "+00:00")
    if value.endswith("+0000"):
        value = value[:-5] + "+00:00"

    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def section(title: str) -> None:
    print(f"\n{'=' * 88}\n{title}\n{'=' * 88}")


def existing_columns(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    return df.select([column for column in columns if column in df.columns])


# ============================================================
# LOAD GENERATED DATASETS
# ============================================================

decisions = pl.read_parquet(DATA_DIR / "decisions.parquet")
candidates = pl.read_parquet(DATA_DIR / "candidates.parquet")
storage = pl.read_parquet(DATA_DIR / "storage_state.parquet")

print(f"Project root: {ROOT}")
print(f"Processed data: {DATA_DIR}")

section("1. DATASET OVERVIEW")
print(f"Decisions:  {decisions.shape}")
print(f"Candidates: {candidates.shape}")
print(f"Storage:    {storage.shape}")


# ============================================================
# CANDIDATE COUNTS PER DECISION
# ============================================================

section("2. CANDIDATE COUNTS PER DECISION")

candidate_counts = candidates.group_by("decision_id").agg([
    pl.len().alias("candidate_position_rows"),
    pl.col("is_occupied").sum().alias("occupied_candidates"),
    pl.col("BoxID").n_unique().alias("candidate_boxes"),
])

decision_columns = [
    "decision_id", "run_id", "decision_timestamp", "required_PNo",
    "candidate_count", "valid_for_training",
]

print(
    existing_columns(decisions, decision_columns)
    .join(candidate_counts, on="decision_id", how="left")
    .with_columns([
        pl.col("candidate_position_rows").fill_null(0),
        pl.col("occupied_candidates").fill_null(0),
        pl.col("candidate_boxes").fill_null(0),
    ])
    .sort(["run_id", "decision_timestamp"])
)


# ============================================================
# CANDIDATE COUNTS BY REQUIRED PART
# ============================================================

section("3. CANDIDATE COUNTS BY REQUIRED PART")

print(
    candidates.group_by("required_PNo").agg([
        pl.len().alias("candidate_position_rows"),
        pl.col("is_occupied").sum().alias("occupied_candidates"),
        pl.col("decision_id").n_unique().alias("decisions"),
        pl.col("BoxID").n_unique().alias("unique_boxes_across_decisions"),
    ]).sort("required_PNo")
)


# ============================================================
# STORAGE SNAPSHOT INTEGRITY
# ============================================================

section("4. STORAGE SNAPSHOT INTEGRITY")

print("Storage boxes and rows per decision:")
print(
    storage.group_by("decision_id").agg([
        pl.col("BoxID").n_unique().alias("unique_boxes"),
        pl.len().alias("storage_rows"),
    ]).sort("decision_id")
)

duplicates = (
    storage.group_by(["decision_id", "BoxID"]).agg([
        pl.len().alias("rows"),
        pl.col("BufPos").n_unique().alias("unique_positions"),
    ])
    .filter((pl.col("rows") > 1) | (pl.col("unique_positions") > 1))
    .sort(["decision_id", "BoxID"])
)

print("\nBoxes appearing more than once in a Storage snapshot:")
print("None" if duplicates.is_empty() else duplicates)


# ============================================================
# SELECTED MES ACTION VERIFICATION
# ============================================================

section("5. SELECTED MES ACTION VERIFICATION")

selected_counts = (
    candidates.filter(pl.col("selected"))
    .group_by("decision_id")
    .agg([
        pl.len().alias("selected_candidate_rows"),
        pl.col("is_occupied").sum().alias("selected_occupied_rows"),
    ])
)

verification_columns = [
    "decision_id", "run_id", "decision_timestamp", "required_PNo",
    "selected_BoxID", "selected_BoxPos", "selected_box_in_storage",
    "selected_candidate_exists", "selected_position_PNo",
    "selected_position_occupied", "valid_for_training",
]

verification = (
    existing_columns(decisions, verification_columns)
    .join(selected_counts, on="decision_id", how="left")
    .with_columns([
        pl.col("selected_candidate_rows").fill_null(0),
        pl.col("selected_occupied_rows").fill_null(0),
    ])
    .sort(["run_id", "decision_timestamp"])
)

print(verification)

invalid = decisions.filter(~pl.col("valid_for_training"))
print(f"\nInvalid decisions: {invalid.height}")

if invalid.is_empty():
    print("All decisions have a valid selected candidate.")
else:
    print(existing_columns(invalid, verification_columns))


# ============================================================
# RAW BUFFER EVENT HISTORY FOR INVALID DECISIONS
# ============================================================

section("6. RAW STORAGE / PICK & SORT EVENTS FOR INVALID DECISIONS")

if invalid.is_empty():
    print("No invalid decisions; raw event inspection skipped.")
else:
    run_ids = sorted(set(invalid.get_column("run_id").to_list()))
    raw_buffers = {
        run_id: load_table(run_id, "tblbufferpos")
        for run_id in run_ids
    }

    for decision in invalid.sort(
        ["run_id", "decision_timestamp"]
    ).to_dicts():

        run_id = decision["run_id"]
        selected_box = int(decision["selected_BoxID"])
        decision_ts = parse_timestamp(decision["decision_timestamp"])

        section(
            f"{decision['decision_id']} | "
            f"selected BoxID={selected_box}, "
            f"BoxPos={decision['selected_BoxPos']}"
        )
        print(f"Decision timestamp: {decision['decision_timestamp']}")

        if decision_ts is None:
            print("Could not parse decision timestamp.")
            continue

        raw = raw_buffers[run_id]
        relevant = raw.filter(
            (pl.col("ResourceId").cast(pl.Int64, strict=False) == RESOURCE_ID)
            & (pl.col("BoxID").cast(pl.Int64, strict=False) == selected_box)
            & (pl.col("BufNo").cast(pl.Int64, strict=False).is_in(RELEVANT_BUFFERS))
            & (pl.col("PNo").cast(pl.Int64, strict=False) == BOX_TYPE_PNO)
        )

        lower = decision_ts - timedelta(minutes=EVENT_WINDOW_MINUTES)
        upper = decision_ts + timedelta(minutes=EVENT_WINDOW_MINUTES)
        event_rows = []

        for row in relevant.to_dicts():
            event_ts = parse_timestamp(row.get("timestamp"))
            if event_ts is None or not (lower <= event_ts <= upper):
                continue

            event_rows.append({
                "timestamp": event_ts.isoformat(),
                "seconds_from_decision": round(
                    (event_ts - decision_ts).total_seconds(), 3
                ),
                "action": row.get("action"),
                "BufNo": row.get("BufNo"),
                "BufPos": row.get("BufPos"),
                "BoxID": row.get("BoxID"),
                "PNo": row.get("PNo"),
            })

        if event_rows:
            print(pl.DataFrame(event_rows).sort("timestamp"))
        else:
            print(
                f"No matching buffer events within "
                f"+/- {EVENT_WINDOW_MINUTES} minutes."
            )


# ============================================================
# FINAL SUMMARY
# ============================================================

section("7. FINAL QUALITY SUMMARY")
print(f"Total decisions: {decisions.height}")
print(f"Valid decisions: {decisions.filter(pl.col('valid_for_training')).height}")
print(f"Invalid decisions: {invalid.height}")
print(f"Candidate position rows: {candidates.height}")
print(f"Occupied candidate rows: {candidates.filter(pl.col('is_occupied')).height}")
print(f"Storage snapshot rows: {storage.height}")

print("\nThis script only diagnoses the datasets and raw buffer events.")
print("It does not modify or regenerate any dataset.")