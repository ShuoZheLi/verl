#!/usr/bin/env python3

import argparse
import pandas as pd
from pathlib import Path


def filter_level_one(input_path: str, output_path: str):
    """
    Load a parquet file, keep only rows where level == "Level 1",
    and save to a new parquet file.
    """

    input_path = Path(input_path)
    output_path = Path(output_path)

    print(f"Loading parquet file: {input_path}")
    df = pd.read_parquet(input_path)

    if "level" not in df.columns:
        raise ValueError("Column 'level' not found in parquet file.")

    print("Filtering rows where level == 'Level 1'")
    filtered_df = df[df["level"] == "Level 1"]

    print(f"Original rows: {len(df)}")
    print(f"Filtered rows: {len(filtered_df)}")

    print(f"Saving filtered parquet to: {output_path}")
    filtered_df.to_parquet(output_path, index=False)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="/data/shuozhe/saved_dataset/lighteval-MATH-preprocessed/train.parquet", help="Input parquet file path")
    parser.add_argument("--output", type=str, default="/data/shuozhe/saved_dataset/level_1_only/train.parquet", help="Output parquet file path")
    args = parser.parse_args()

    filter_level_one(args.input, args.output)