
from pathlib import Path
import polars as pl

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DECISIONS_PATH = PROJECT_ROOT / "data/processed/decisions.parquet"
STORAGE_PATH = PROJECT_ROOT / "data/processed/storage_state.parquet"

# Define manually if automatic discovery does not find the file.
BUFFERPOS_PATH = None

TARGET_BOXES = {
    "20261016": [4],
    "20261020": [14],
}

RESOURCE_ID = 17
BOX_TYPE_PNO = 29

STORAGE_BUF_NO = 1
PICK_SORT_BUF_NO = 3


# ============================================================
# LOAD DATA
# ============================================================

def find_bufferpos_file():
    if BUFFERPOS_PATH is not None:
        path = Path(BUFFERPOS_PATH)
        if not path.is_absolute():
            path = PROJECT_ROOT / path

        if not path.exists():
            raise FileNotFoundError(path)

        return path

    matches = [
        p for p in PROJECT_ROOT.rglob("*tblbufferpos*")
        if p.is_file()
        and p.suffix.lower() in [".parquet", ".csv"]
        and "processed" not in p.parts
    ]

    if len(matches) == 0:
        raise FileNotFoundError(
            "No encontré tblbufferpos. "
            "Define BUFFERPOS_PATH manualmente."
        )

    if len(matches) > 1:
        print("Encontré varios archivos:")
        for p in matches:
            print(p)

        raise RuntimeError(
            "Define BUFFERPOS_PATH con el archivo correcto."
        )

    return matches[0]


def read_table(path):
    if path.suffix.lower() == ".parquet":
        return pl.read_parquet(path)

    return pl.read_csv(path, try_parse_dates=True)


# ============================================================
# MAIN
# ============================================================

def main():

    bufferpos_path = find_bufferpos_file()

    decisions = pl.read_parquet(DECISIONS_PATH)
    storage = pl.read_parquet(STORAGE_PATH)
    events = read_table(bufferpos_path)

    required_columns = [
        "timestamp",
        "action",
        "ResourceId",
        "BufNo",
        "BufPos",
        "BoxID",
        "PNo",
    ]

    missing = [
        c for c in required_columns
        if c not in events.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns in tblbufferpos: {missing}"
        )

    print("=" * 90)
    print("ORIGINAL TBLBUFFERPOS EVENT INSPECTION")
    print("=" * 90)

    print(f"Source: {bufferpos_path}")
    print(f"Raw rows: {len(events)}")

    # Only actual box records from the relevant resource.
    events = (
        events
        .filter(
            (pl.col("ResourceId") == RESOURCE_ID)
            & (pl.col("PNo") == BOX_TYPE_PNO)
            & pl.col("BufNo").is_in([
                STORAGE_BUF_NO,
                PICK_SORT_BUF_NO,
            ])
        )
        .with_columns(
            pl.col("timestamp")
            .cast(pl.String)
            .alias("_timestamp_str")
        )
    )

    # ========================================================
    # 1. EVENT HISTORY FOR EACH BOX
    # ========================================================

    for run_id, box_ids in TARGET_BOXES.items():

        run_date = (
            f"{run_id[:4]}-{run_id[4:6]}-{run_id[6:8]}"
        )

        print("\n" + "=" * 90)
        print(f"RUN {run_id}")
        print("=" * 90)

        for box_id in box_ids:

            print("\n" + "-" * 70)
            print(f"BOXID {box_id}")
            print("-" * 70)

            box_events = (
                events
                .filter(
                    (pl.col("BoxID") == box_id)
                    & pl.col("_timestamp_str").str.contains(run_date)
                )
                .sort(["timestamp", "BufNo", "BufPos"])
                .select([
                    "timestamp",
                    "action",
                    "ResourceId",
                    "BufNo",
                    "BufPos",
                    "BoxID",
                ])
            )

            if len(box_events) == 0:
                print("No events found.")
            else:
                print(box_events)

        # ====================================================
        # 2. RECONSTRUCTED STORAGE SNAPSHOTS
        # ====================================================

        print("\nRECONSTRUCTED STORAGE SNAPSHOTS")

        run_storage = (
            storage
            .filter(
                (pl.col("run_id") == run_id)
                & pl.col("BoxID").is_in(box_ids)
            )
            .sort(["decision_timestamp", "BoxID", "BufPos"])
        )

        print(
            run_storage if len(run_storage) > 0
            else "No matching storage rows."
        )

        # ====================================================
        # 3. DECISION TIMESTAMPS
        # ====================================================

        print("\nDECISION TIMESTAMPS")

        run_decisions = (
            decisions
            .filter(pl.col("run_id") == run_id)
            .sort("decision_timestamp")
        )

        print(
            run_decisions.select([
                "decision_id",
                "decision_timestamp",
                "MainONo",
                "required_PNo",
                "selected_BoxID",
                "selected_BoxPos",
            ])
        )

    # ========================================================
    # 4. ALL EVENTS FOR THE TWO BOXES
    # ========================================================

    print("\n" + "=" * 90)
    print("EVENTS IN STORAGE AND PICK & SORT")
    print("=" * 90)

    for run_id, box_ids in TARGET_BOXES.items():

        run_date = (
            f"{run_id[:4]}-{run_id[4:6]}-{run_id[6:8]}"
        )

        subset = (
            events
            .filter(
                pl.col("BoxID").is_in(box_ids)
                & pl.col("_timestamp_str").str.contains(run_date)
            )
            .sort(["timestamp", "BufNo", "BufPos"])
            .select([
                "timestamp",
                "action",
                "ResourceId",
                "BufNo",
                "BufPos",
                "BoxID",
            ])
        )

        print(f"\nRUN {run_id}")
        print(subset)

    print("\nInspection finished.")


if __name__ == "__main__":
    main()
