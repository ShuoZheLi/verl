#!/usr/bin/env python3
"""
Replot curves and final-value distribution from saved responses jsonl files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _find_jsonl_paths(out_dir: Path) -> list[Path]:
    paths = sorted(out_dir.glob("responses_rank*.jsonl"))
    if paths:
        return paths
    single = out_dir / "responses.jsonl"
    if single.exists():
        return [single]
    raise FileNotFoundError(f"No responses_rank*.jsonl or responses.jsonl found in {out_dir}")


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


def _summarize_distribution(values):
    if len(values) == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "p25": None,
            "median": None,
            "p75": None,
        }

    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
    }


def _build_histogram(values, bins):
    if len(values) == 0:
        return [0.0 for _ in range(len(bins) - 1)]
    hist, _ = np.histogram(values, bins=bins, density=True)
    return hist.astype(np.float64).tolist()


def _save_curves_plot(out_dir: Path, correct_curve, wrong_curve, num_correct: int, num_wrong: int):
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


def _save_final_value_distribution(
    out_dir: Path,
    correct_final_values,
    wrong_final_values,
    num_bins: int,
):
    all_values = list(correct_final_values) + list(wrong_final_values)
    bins = None
    distribution = {
        "num_bins": int(num_bins),
        "correct_final_value_stats": _summarize_distribution(correct_final_values),
        "wrong_final_value_stats": _summarize_distribution(wrong_final_values),
        "note": "Final value means critic value at the last generated response token.",
    }

    if all_values:
        all_arr = np.asarray(all_values, dtype=np.float64)
        vmin = float(all_arr.min())
        vmax = float(all_arr.max())
        if np.isclose(vmin, vmax):
            span = max(abs(vmin) * 1e-3, 1e-6)
            bins = np.linspace(vmin - span, vmax + span, num_bins + 1)
        else:
            bins = np.linspace(vmin, vmax, num_bins + 1)
        distribution["bin_edges"] = bins.tolist()
        distribution["correct_density"] = _build_histogram(correct_final_values, bins)
        distribution["wrong_density"] = _build_histogram(wrong_final_values, bins)

    dist_path = out_dir / "final_value_distribution.json"
    with dist_path.open("w", encoding="utf-8") as df:
        json.dump(distribution, df, ensure_ascii=True, indent=2)

    try:
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(9, 4))
        ax = fig.add_subplot(1, 1, 1)

        has_any = False
        if len(correct_final_values) > 0:
            ax.hist(
                correct_final_values,
                bins=bins if bins is not None else num_bins,
                density=True,
                alpha=0.45,
                label=f"correct (n={len(correct_final_values)})",
            )
            has_any = True
        if len(wrong_final_values) > 0:
            ax.hist(
                wrong_final_values,
                bins=bins if bins is not None else num_bins,
                density=True,
                alpha=0.45,
                label=f"wrong (n={len(wrong_final_values)})",
            )
            has_any = True
        if has_any:
            ax.legend()

        ax.set_title("Distribution of Final Response-Token Critic Value")
        ax.set_xlabel("Final response-token value")
        ax.set_ylabel("Density")
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(out_dir / "final_value_distribution.png", dpi=150)
        plt.close(fig)
    except Exception as exc:
        print(f"[warn] Failed to write final value distribution plot: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Replot critic curves and final-value distributions from jsonl.")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/data/shuozhe/verl/critic_debug/critic_debug_all_gs400",
        help="Directory containing responses_rank*.jsonl or responses.jsonl.",
    )
    parser.add_argument("--num_bins", type=int, default=100, help="Bins for normalized position curves.")
    parser.add_argument("--dist_bins", type=int, default=80, help="Bins for final-value distribution histogram.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    paths = _find_jsonl_paths(out_dir)

    correct_sum = np.zeros(args.num_bins, dtype=np.float64)
    correct_cnt = np.zeros(args.num_bins, dtype=np.int64)
    wrong_sum = np.zeros(args.num_bins, dtype=np.float64)
    wrong_cnt = np.zeros(args.num_bins, dtype=np.int64)

    num_correct = 0
    num_wrong = 0
    correct_final_values = []
    wrong_final_values = []

    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                values = rec.get("values", [])
                correct = bool(rec.get("correct", False))

                _accumulate_bins(
                    values_list=values,
                    num_bins=args.num_bins,
                    sum_arr=correct_sum if correct else wrong_sum,
                    cnt_arr=correct_cnt if correct else wrong_cnt,
                )

                final_val = rec.get("final_response_value")
                if final_val is None and len(values) > 0:
                    final_val = values[-1]
                if final_val is not None:
                    if correct:
                        correct_final_values.append(float(final_val))
                    else:
                        wrong_final_values.append(float(final_val))

                if correct:
                    num_correct += 1
                else:
                    num_wrong += 1

    correct_curve = _finalize_curve(correct_sum, correct_cnt)
    wrong_curve = _finalize_curve(wrong_sum, wrong_cnt)

    curves = {
        "num_bins": args.num_bins,
        "correct_curve": correct_curve,
        "wrong_curve": wrong_curve,
        "num_correct": num_correct,
        "num_wrong": num_wrong,
    }

    curves_path = out_dir / "curves.json"
    with curves_path.open("w", encoding="utf-8") as cf:
        json.dump(curves, cf, ensure_ascii=True, indent=2)

    _save_curves_plot(
        out_dir=out_dir,
        correct_curve=correct_curve,
        wrong_curve=wrong_curve,
        num_correct=num_correct,
        num_wrong=num_wrong,
    )
    _save_final_value_distribution(
        out_dir=out_dir,
        correct_final_values=correct_final_values,
        wrong_final_values=wrong_final_values,
        num_bins=args.dist_bins,
    )

    print(f"[saved] {curves_path}")
    print(f"[saved] {out_dir / 'curves.png'}")
    print(f"[saved] {out_dir / 'final_value_distribution.json'}")
    print(f"[saved] {out_dir / 'final_value_distribution.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
