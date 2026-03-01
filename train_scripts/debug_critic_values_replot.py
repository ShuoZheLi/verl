#!/usr/bin/env python3
"""
Recompute correctness from existing jsonl files and plot curves.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from verl.utils.reward_score import default_compute_score


def _normalize_text_for_match(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _is_correct(
    response: str,
    reference: str | None,
    mode: str,
    data_source: str | None = None,
    score_threshold: float = 1.0,
    record_correct: bool | None = None,
) -> bool:
    if reference is None:
        return False
    if mode == "record":
        return bool(record_correct)
    if mode == "exact":
        return _normalize_text_for_match(response) == _normalize_text_for_match(reference)
    if mode == "contains":
        return _normalize_text_for_match(reference) in _normalize_text_for_match(response)
    if mode == "regex":
        try:
            return re.search(reference, response, flags=re.DOTALL) is not None
        except re.error:
            return False
    if mode == "verl":
        if data_source is None:
            raise ValueError("correct_match=verl requires data_source")
        score = default_compute_score(data_source, response, reference)
        return score >= score_threshold
    raise ValueError(f"Unknown correct_match mode: {mode}")


def _accumulate_bins(values_list, num_bins, sum_arr, cnt_arr):
    n = len(values_list)
    if n == 0:
        return
    for i, v in enumerate(values_list):
        b = int((i / n) * num_bins)
        if b >= num_bins:
            b = num_bins - 1
        sum_arr[b] += float(v)
        cnt_arr[b] += 1


def _finalize_curve(sum_arr, cnt_arr):
    curve = np.full_like(sum_arr, np.nan, dtype=np.float64)
    mask = cnt_arr > 0
    curve[mask] = sum_arr[mask] / cnt_arr[mask]
    return curve.tolist()


def main() -> int:
    parser = argparse.ArgumentParser(description="Replot curves from saved jsonl responses.")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/nfs/shuozhe/verl/outputs/critic_debug_all",
        help="Directory containing responses_rank*.jsonl or responses.jsonl.",
    )
    parser.add_argument("--num_bins", type=int, default=100)
    parser.add_argument(
        "--correct_match",
        type=str,
        default="verl",
        choices=["record", "exact", "contains", "regex", "verl"],
    )
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

    correct_sum = np.zeros(args.num_bins, dtype=np.float64)
    correct_cnt = np.zeros(args.num_bins, dtype=np.int64)
    wrong_sum = np.zeros(args.num_bins, dtype=np.float64)
    wrong_cnt = np.zeros(args.num_bins, dtype=np.int64)
    num_correct = 0
    num_wrong = 0

    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                response = rec.get("response", "")
                reference = rec.get("reference")
                data_source = rec.get("data_source") or args.data_source
                values_list = rec.get("values", [])
                record_correct = rec.get("correct")

                correct = _is_correct(
                    response,
                    reference,
                    mode=args.correct_match,
                    data_source=data_source,
                    score_threshold=args.score_threshold,
                    record_correct=record_correct,
                )
                if correct:
                    num_correct += 1
                    _accumulate_bins(values_list, args.num_bins, correct_sum, correct_cnt)
                else:
                    num_wrong += 1
                    _accumulate_bins(values_list, args.num_bins, wrong_sum, wrong_cnt)

    correct_curve = _finalize_curve(correct_sum, correct_cnt)
    wrong_curve = _finalize_curve(wrong_sum, wrong_cnt)

    curves = {
        "num_bins": args.num_bins,
        "correct_curve": correct_curve,
        "wrong_curve": wrong_curve,
        "num_correct": num_correct,
        "num_wrong": num_wrong,
        "correct_match": args.correct_match,
        "score_threshold": args.score_threshold,
    }

    curves_path = out_dir / "curves.json"
    with curves_path.open("w", encoding="utf-8") as cf:
        json.dump(curves, cf, ensure_ascii=True, indent=2)

    try:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(correct_curve, label=f"correct (n={num_correct})", linewidth=1.5)
        ax.plot(wrong_curve, label=f"wrong (n={num_wrong})", linewidth=1.5)
        ax.set_title("Average Critic Values Over Normalized Response Position")
        ax.set_xlabel("Normalized token position (bins)")
        ax.set_ylabel("Value")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "curves.png", dpi=150)
        plt.close(fig)
    except Exception as exc:
        print(f"[warn] Failed to write curves plot: {exc}")

    print(f"[saved] {curves_path}")
    print(f"[saved] {out_dir / 'curves.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
