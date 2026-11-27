from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from resource_control.io.loaders import (
    TABLES,
    discover_runs,
    load_table,
)

DATA_DIR = ROOT / "data" / "raw"


def main():

    runs = discover_runs(DATA_DIR)

    print(f"Total runs found: {len(runs)}")

    for run_path in runs:

        print("\n" + "=" * 60)
        print(f"RUN: {run_path.name}")
        print("=" * 60)

        for table_name in TABLES:

            print(f"\n--- {table_name} ---")

            try:
                df = load_table(run_path, table_name)

                print(f"Rows: {df.height}")
                print(f"Columns: {df.columns}")
                print(f"Size: {df.estimated_size('mb'):.2f} MB")

            except FileNotFoundError as error:
                print(f"WARNING: {error}")


if __name__ == "__main__":
    main()