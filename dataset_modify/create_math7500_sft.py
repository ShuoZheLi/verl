#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_INPUT = Path("/data/shuozhe/saved_dataset/MATH_7500_5000/MATH_dataset_train.json")
DEFAULT_OUTPUT = Path("/data/shuozhe/saved_dataset/MetaMathQA-math-500/math7500_sft.parquet")
DATASET_SOURCE = "DigitalLearningGmbH/MATH-lighteval"
PROMPT_SUFFIX = " Let's think step by step and output the final answer within \\boxed{}."
MALFORMED_BOXED_RE = re.compile(r"\\boxed\s+([^$.,;\s]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert MATH_7500_5000 JSONL rows to an SFT messages parquet."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} in {path}.") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected object on line {line_number} in {path}.")
            rows.append(obj)
    return rows


def require_string(row: dict[str, Any], key: str, index: int) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Row {index} has missing or invalid {key!r}.")
    return value


def normalize_solution(solution: str, index: int) -> str:
    if "\\boxed{" in solution:
        return solution

    normalized, replacements = MALFORMED_BOXED_RE.subn(r"\\boxed{\1}", solution)
    if replacements and "\\boxed{" in normalized:
        return normalized

    raise ValueError(f"Row {index} solution does not contain \\boxed{{...}}.")


def build_sft_row(row: dict[str, Any], index: int) -> dict[str, Any]:
    problem = require_string(row, "problem", index).rstrip()
    solution = normalize_solution(require_string(row, "solution", index), index)

    return {
        "messages": [
            {
                "content": f"{problem}{PROMPT_SUFFIX}",
                "role": "user",
            },
            {
                "content": solution,
                "role": "assistant",
            },
        ],
        "dataset_source": DATASET_SOURCE,
        "level": row.get("level"),
        "type": row.get("type"),
        "answer": row.get("answer"),
        "id": row.get("id"),
    }


def write_parquet(rows: list[dict[str, Any]], output: Path) -> None:
    schema = pa.schema(
        [
            pa.field(
                "messages",
                pa.list_(
                    pa.struct(
                        [
                            pa.field("content", pa.string()),
                            pa.field("role", pa.string()),
                        ]
                    )
                ),
            ),
            pa.field("dataset_source", pa.string()),
            pa.field("level", pa.string()),
            pa.field("type", pa.string()),
            pa.field("answer", pa.string()),
            pa.field("id", pa.int64()),
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, output)


def main() -> None:
    args = parse_args()
    source_rows = read_jsonl(args.input)
    sft_rows = [build_sft_row(row, index) for index, row in enumerate(source_rows)]
    write_parquet(sft_rows, args.output)
    print(f"Wrote {len(sft_rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
