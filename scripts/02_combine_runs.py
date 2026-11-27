from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from resource_control.io.loaders import load_runs


DATA_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "data" / "interim" / "combined"

# Replace with the actual folder names of your two reliable runs.
RUN_IDS = [
    "20261020",
    "20261016",
]


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    data = load_runs(
        data_dir=DATA_DIR,
        run_ids=RUN_IDS,
    )

    for table_name, df in data.items():

        output_path = OUTPUT_DIR / f"{table_name}.parquet"

        df.write_parquet(output_path)

        print(f"\n{table_name}")
        print(f"Rows: {df.height}")
        print(f"Columns: {len(df.columns)}")
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()