
from __future__ import annotations

from pathlib import Path

import polars as pl


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"

DATASETS = {
    "DECISIONS": "decisions.parquet",
    "CANDIDATES": "candidates.parquet",
    "STORAGE_STATE": "storage_state.parquet",
}

SAMPLE_SIZE = 5
MAX_DISTINCT_VALUES = 20


# ============================================================
# HELPERS
# ============================================================

def section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def compact_value(value, max_length: int = 100) -> str:
    text = str(value)

    if len(text) > max_length:
        return text[:max_length - 3] + "..."

    return text


# ============================================================
# DATASET INSPECTION
# ============================================================

def inspect_dataset(name: str, filename: str) -> pl.DataFrame:

    path = DATA_DIR / filename

    section(f"{name} | {filename}")

    if not path.exists():
        print(f"ERROR: File not found: {path}")
        return pl.DataFrame()

    df = pl.read_parquet(path)

    # --------------------------------------------------------
    # 1. Basic information
    # --------------------------------------------------------

    print("\n[1] BASIC INFORMATION")
    print(f"Rows:    {df.height:,}")
    print(f"Columns: {df.width}")

    # --------------------------------------------------------
    # 2. Complete schema
    # --------------------------------------------------------

    print("\n[2] SCHEMA")

    null_counts = df.null_count().row(0, named=True)

    print(
        f"{'COLUMN':35} {'TYPE':20} {'NULLS':>10} "
        f"{'NULL %':>10} {'UNIQUE':>10}"
    )

    print("-" * 90)

    for column, dtype in df.schema.items():

        series = df.get_column(column)

        nulls = null_counts[column]

        null_pct = (
            nulls / df.height * 100
            if df.height > 0
            else 0
        )

        print(
            f"{column[:34]:35} "
            f"{str(dtype):20} "
            f"{nulls:10,} "
            f"{null_pct:9.2f}% "
            f"{series.n_unique():10,}"
        )

    # --------------------------------------------------------
    # 3. Representative rows
    # --------------------------------------------------------

    print(f"\n[3] FIRST {SAMPLE_SIZE} ROWS")

    print(
        df.head(SAMPLE_SIZE).to_dicts()
    )

    # --------------------------------------------------------
    # 4. Distinct values for categorical columns
    # --------------------------------------------------------

    print("\n[4] DISTINCT VALUES (LOW CARDINALITY)")

    for column, dtype in df.schema.items():

        series = df.get_column(column)

        if series.n_unique() <= MAX_DISTINCT_VALUES:

            values = series.unique().sort().to_list()

            print(
                f"{column}: "
                f"{compact_value(values, 250)}"
            )

    # --------------------------------------------------------
    # 5. Numeric summaries
    # --------------------------------------------------------

    numeric_columns = [
        column
        for column, dtype in df.schema.items()
        if dtype.is_numeric()
    ]

    if numeric_columns:

        print("\n[5] NUMERIC SUMMARY")

        summary = df.select(
            numeric_columns
        ).describe()

        print(summary)

    return df


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("\nDATASET INSPECTION")
    print(f"Data directory: {DATA_DIR}")

    datasets = {}

    for name, filename in DATASETS.items():

        datasets[name] = inspect_dataset(
            name,
            filename,
        )

    # --------------------------------------------------------
    # 6. Dataset overview
    # --------------------------------------------------------

    section("DATASET OVERVIEW")

    for name, df in datasets.items():

        if not df.is_empty():

            print(
                f"{name:20} "
                f"Rows: {df.height:>10,} | "
                f"Columns: {df.width:>3}"
            )

    section("INSPECTION FINISHED")

if __name__ == "__main__":
    main()