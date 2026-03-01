#!/usr/bin/env python3
"""
Inspect correctness scoring and answer extraction for debug_critic_values_all outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from verl.utils.reward_score import default_compute_score
from verl.utils.reward_score import math_reward

MATH_DATA_SOURCES = {
    "lighteval/MATH",
    "DigitalLearningGmbH/MATH-lighteval",
    "HuggingFaceH4/MATH-500",
}


def _iter_jsonl(paths: list[Path]):
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                yield path, json.loads(line)


def _extract_math_answer(response: str) -> str | None:
    boxed = math_reward.last_boxed_only_string(response)
    if boxed is None:
        return None
    try:
        return math_reward.remove_boxed(boxed)
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Debug answer extraction and correctness scoring.")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/nfs/shuozhe/verl/outputs/critic_debug_all",
        help="Directory containing responses_rank*.jsonl or responses.jsonl.",
    )
    parser.add_argument("--limit", type=int, default=5, help="Number of examples to print.")
    parser.add_argument(
        "--data_source",
        type=str,
        default=None,
        help="Fallback data_source if not present in the jsonl record.",
    )
    parser.add_argument("--score_threshold", type=float, default=1.0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    paths = sorted(out_dir.glob("responses_rank*.jsonl"))
    if not paths:
        paths = [out_dir / "responses.jsonl"]

    shown = 0
    total = 0
    correct = 0
    for path, rec in _iter_jsonl(paths):
        data_source = rec.get("data_source") or args.data_source
        response = rec.get("response", "")
        reference = rec.get("reference")

        if data_source and reference is not None:
            score = default_compute_score(data_source, response, reference)
            if score >= args.score_threshold:
                correct += 1
        total += 1

        if shown < args.limit:
            extracted = None
            if data_source in MATH_DATA_SOURCES:
                extracted = _extract_math_answer(response)
                equiv = math_reward.is_equiv(extracted, reference)
            else:
                equiv = None
            print(f"\n[{path.name}] index={rec.get('index')}")
            print(f"data_source: {data_source}")
            print(f"reference: {reference}")
            if extracted is not None:
                print(f"extracted: {extracted}")
            if equiv is not None:
                print(f"is_equiv: {equiv}")
            if data_source and reference is not None:
                print(f"score: {default_compute_score(data_source, response, reference)}")
            shown += 1

    print(f"\nsummary: correct={correct} / total={total}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
