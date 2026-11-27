from pathlib import Path
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"

RUN_IDS = ["20261016", "20261020"]
TABLES = [
    "tblorderpos",
    "tblboxpos",
    "tblbufferpos",
    "tblstepparameter"
]

for run_id in RUN_IDS:
    print(f"\n{'='*60}")
    print(f"RUN: {run_id}")
    print(f"{'='*60}")

    for table in TABLES:
        path = DATA_DIR / run_id / table
        files = list(path.rglob("*.parquet"))

        df = pl.read_parquet(files)

        print(f"\n--- {table} ---")
        print(df.head(5))