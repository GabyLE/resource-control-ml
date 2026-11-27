from pathlib import Path
import re
from sys import path

import polars as pl

TABLES = [
    "tblorderpos",
    "tblboxpos",
    "tblbufferpos",
    "tblstepparameter"
]

RUN_PATTERN = re.compile(r"^\d{8}(?:_\d+)?$") # format 8 digits plus if more runs in that date

def discover_runs(data_dir: Path) -> list[Path]:
    """
    Find run folders named DDMMYYYY or DDMMYYYY_N
    :param data_dir:
    :return: list runs
    """
    runs = [
        path
        for path in data_dir.iterdir()
        if path.is_dir() and RUN_PATTERN.match(path.name)
    ]

    return sorted(runs, key = lambda p: p.name)

def load_table(run_path: Path, table_name: str) -> pl.DataFrame:
    """
    Load all Parquet batches of one table from a single run.
    Preserve run ID and source file.
    :param run_path:
    :param table_name:
    :return: dataframe
    """
    table_path = run_path / table_name

    if not table_path.exists():
        raise FileNotFoundError(f"Table {table_path} not found")

    files = sorted(table_path.rglob("*.parquet"))

    if not files:
        raise FileNotFoundError(f"No Parquet filse found in {table_path}")

    frames = []

    for file in files:
        df = pl.read_parquet(file)

        df = df.with_columns(
            pl.lit(run_path.name).alias("run_id"),
            pl.lit(str(file.relative_to(run_path))).alias("source_file"),
        )

        frames.append(df)

    return pl.concat(frames, how= "diagonal_relaxed")

def load_runs(
    data_dir: Path,
    run_ids: list[str],
    table_names: list[str] = TABLES,
) -> dict[str, pl.DataFrame]:
    """
    Load selected runs and combine each table across runs.
    """
    combined = {}

    for table_name in table_names:
        frames = []

        for run_id in run_ids:
            run_path = data_dir / run_id

            if not run_path.is_dir():
                raise FileNotFoundError(
                    f"Run folder not found: {run_path}"
                )

            df = load_table(run_path, table_name)
            frames.append(df)

        combined[table_name] = pl.concat(
            frames,
            how="diagonal_relaxed",
        )

    return combined