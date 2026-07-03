#!/usr/bin/env python3
"""Lightweight data-readiness smoke test for the Hedge Fund TS Kaggle repo.

This script does not train LightGBM. It validates that train/test parquet files
exist and contain the columns expected by the notebooks.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


EXPECTED_BASE_COLUMNS = {"code", "sub_code", "sub_category", "ts_index", "horizon"}


def inspect_parquet(path: Path, name: str) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")

    df = pd.read_parquet(path)
    print(f"{name}: {df.shape[0]:,} rows × {df.shape[1]:,} columns")
    return set(df.columns)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=".", help="Folder containing train.parquet and test.parquet")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    print("Hedge Fund TS smoke test")
    print(f"Data dir: {data_dir.resolve()}")

    train_cols = inspect_parquet(data_dir / "train.parquet", "train.parquet")
    test_cols = inspect_parquet(data_dir / "test.parquet", "test.parquet")

    for name, cols in [("train", train_cols), ("test", test_cols)]:
        missing = sorted(EXPECTED_BASE_COLUMNS - cols)
        if missing:
            print(f"\n{name} is missing expected columns: {missing}")
            return 1

    if "y_target" not in train_cols:
        print("\nWarning: train.parquet does not contain y_target. Check competition column naming.")

    print("\nSmoke test passed: parquet files are readable and core columns exist.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
