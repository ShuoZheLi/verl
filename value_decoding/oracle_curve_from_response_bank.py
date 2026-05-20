from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - plotting is optional at runtime
    plt = None
    MATPLOTLIB_IMPORT_ERROR = exc
else:
    MATPLOTLIB_IMPORT_ERROR = None


DEFAULT_BUDGETS = (1, 2, 4, 8, 16)


@dataclass(frozen=True)
class BankSpec:
    name: str
    path: Path


@dataclass(frozen=True)
class PromptScores:
    prompt_index: int
    scores: tuple[float, ...]

    @property
    def num_successes(self) -> int:
        return sum(1 for score in self.scores if score > 0.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute oracle best-of-N curves from saved actor proposal response banks. "
            "The exact-subset curve reports the expected oracle score for a uniformly random N-subset "
            "without replacement from each prompt's full bank."
        )
    )
    parser.add_argument(
        "--banks",
        nargs="+",
        required=True,
        help="Response banks as NAME=PATH entries, e.g. step100=/path/response_bank.jsonl.",
    )
    parser.add_argument("--budgets", nargs="+", type=int, default=list(DEFAULT_BUDGETS))
    parser.add_argument("--score_key", type=str, default="accuracy", help="Score field to maximize; falls back to task_score.")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--allow_missing_budgets",
        action="store_true",
        help="Skip budgets larger than a bank size instead of raising an error.",
    )
    parser.add_argument("--skip_plot", action="store_true")
    return parser.parse_args()


def parse_bank_specs(raw_specs: Sequence[str]) -> list[BankSpec]:
    specs: list[BankSpec] = []
    seen_names: set[str] = set()
    for raw_spec in raw_specs:
        if "=" not in raw_spec:
            raise ValueError(f"Bank spec must be NAME=PATH, got: {raw_spec}")
        name, raw_path = raw_spec.split("=", 1)
        name = name.strip()
        path = Path(raw_path).expanduser()
        if not name:
            raise ValueError(f"Bank name cannot be empty in spec: {raw_spec}")
        if name in seen_names:
            raise ValueError(f"Duplicate bank name: {name}")
        if not path.is_file():
            raise FileNotFoundError(f"Response bank does not exist: {path}")
        seen_names.add(name)
        specs.append(BankSpec(name=name, path=path))
    return specs


def _score_from_row(row: dict[str, Any], score_key: str) -> float:
    if score_key in row:
        return float(row[score_key])
    if "task_score" in row:
        return float(row["task_score"])
    raise KeyError(f"Row has neither requested score_key={score_key!r} nor fallback 'task_score'.")


def load_prompt_scores(path: Path, score_key: str) -> tuple[list[PromptScores], dict[str, Any]]:
    rows_by_prompt: dict[int, list[tuple[int, float]]] = defaultdict(list)
    actor_names: set[str] = set()
    example_ids_by_prompt: dict[int, int] = {}
    num_rows = 0

    with path.open("r", encoding="utf-8") as response_file:
        for line_number, line in enumerate(response_file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            prompt_index = int(row["prompt_index"])
            sample_index = int(row["sample_index"])
            score = _score_from_row(row, score_key)
            if not math.isfinite(score):
                raise ValueError(f"Non-finite score at {path}:{line_number}: {score}")
            rows_by_prompt[prompt_index].append((sample_index, score))
            if "actor_name" in row:
                actor_names.add(str(row["actor_name"]))
            if "example_id" in row:
                example_id = int(row["example_id"])
                previous_example_id = example_ids_by_prompt.setdefault(prompt_index, example_id)
                if previous_example_id != example_id:
                    raise ValueError(
                        f"Prompt {prompt_index} has inconsistent example_id values "
                        f"{previous_example_id} and {example_id}."
                    )
            num_rows += 1

    if not rows_by_prompt:
        raise ValueError(f"Response bank is empty: {path}")

    prompt_scores: list[PromptScores] = []
    bank_sizes: set[int] = set()
    for prompt_index, indexed_scores in sorted(rows_by_prompt.items()):
        indexed_scores = sorted(indexed_scores, key=lambda item: item[0])
        sample_indices = [sample_index for sample_index, _ in indexed_scores]
        if len(set(sample_indices)) != len(sample_indices):
            raise ValueError(f"Prompt {prompt_index} has duplicate sample_index values in {path}.")
        expected_sample_indices = list(range(len(indexed_scores)))
        if sample_indices != expected_sample_indices:
            raise ValueError(
                f"Prompt {prompt_index} sample_index values are {sample_indices[:20]}...; "
                f"expected contiguous 0..{len(indexed_scores) - 1}."
            )
        scores = tuple(score for _, score in indexed_scores)
        prompt_scores.append(PromptScores(prompt_index=prompt_index, scores=scores))
        bank_sizes.add(len(scores))

    metadata = {
        "path": str(path),
        "num_rows": int(num_rows),
        "num_prompts": int(len(prompt_scores)),
        "bank_sizes": sorted(int(size) for size in bank_sizes),
        "actor_names": sorted(actor_names),
    }
    return prompt_scores, metadata


def oracle_first_n(scores: Sequence[float], budget: int) -> float:
    if budget <= 0:
        raise ValueError(f"Budget must be positive, got {budget}")
    if budget > len(scores):
        raise ValueError(f"Budget {budget} exceeds bank size {len(scores)}")
    return float(max(scores[:budget]))


def oracle_exact_random_subset_binary(scores: Sequence[float], budget: int) -> float:
    """Expected oracle success over all uniformly random subsets of size budget.

    For binary success scores, the oracle succeeds unless the subset contains only failures.
    With M total samples and F failures, P(no success) = C(F, budget) / C(M, budget), with
    the convention C(F, budget)=0 when budget > F.
    """

    if budget <= 0:
        raise ValueError(f"Budget must be positive, got {budget}")
    bank_size = len(scores)
    if budget > bank_size:
        raise ValueError(f"Budget {budget} exceeds bank size {bank_size}")
    num_successes = sum(1 for score in scores if score > 0.0)
    num_failures = bank_size - num_successes
    if num_successes == 0:
        return 0.0
    if budget > num_failures:
        return 1.0
    return float(1.0 - (math.comb(num_failures, budget) / math.comb(bank_size, budget)))


def oracle_exact_random_subset_general(scores: Sequence[float], budget: int) -> float:
    """Expected max score over all uniformly random subsets of size budget.

    If sorted scores are x_1 <= ... <= x_M, the probability x_i is the subset maximum is
    C(i, budget - 1) / C(M, budget) using zero-based i, because the subset must include x_i
    and choose the other budget-1 elements from the i lower-or-equal earlier elements. This
    handles ties correctly because tied values contribute the same score across their positions.
    """

    if budget <= 0:
        raise ValueError(f"Budget must be positive, got {budget}")
    bank_size = len(scores)
    if budget > bank_size:
        raise ValueError(f"Budget {budget} exceeds bank size {bank_size}")
    sorted_scores = sorted(float(score) for score in scores)
    denominator = math.comb(bank_size, budget)
    expected_value = 0.0
    for index, score in enumerate(sorted_scores):
        if index >= budget - 1:
            expected_value += score * (math.comb(index, budget - 1) / denominator)
    return float(expected_value)


def scores_are_binary(prompt_scores: Iterable[PromptScores]) -> bool:
    for prompt in prompt_scores:
        for score in prompt.scores:
            if score not in (0.0, 1.0):
                return False
    return True


def summarize_bank(
    *,
    name: str,
    prompt_scores: Sequence[PromptScores],
    budgets: Sequence[int],
    allow_missing_budgets: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not prompt_scores:
        raise ValueError("Cannot summarize an empty bank.")
    bank_sizes = {len(prompt.scores) for prompt in prompt_scores}
    max_bank_size = min(bank_sizes)
    if len(bank_sizes) != 1:
        print(f"Warning: {name} has variable per-prompt bank sizes: {sorted(bank_sizes)}")

    valid_budgets: list[int] = []
    for budget in budgets:
        if budget <= 0:
            raise ValueError(f"Budgets must be positive, got {budget}")
        if budget > max_bank_size:
            if allow_missing_budgets:
                continue
            raise ValueError(f"Budget {budget} exceeds smallest bank size {max_bank_size} for {name}.")
        valid_budgets.append(int(budget))
    if not valid_budgets:
        raise ValueError(f"No valid budgets for {name} with smallest bank size {max_bank_size}.")

    is_binary = scores_are_binary(prompt_scores)
    random_subset_fn = oracle_exact_random_subset_binary if is_binary else oracle_exact_random_subset_general
    rows: list[dict[str, Any]] = []
    for budget in valid_budgets:
        first_n_values = [oracle_first_n(prompt.scores, budget) for prompt in prompt_scores]
        exact_subset_values = [random_subset_fn(prompt.scores, budget) for prompt in prompt_scores]
        rows.append(
            {
                "bank_name": name,
                "budget": int(budget),
                "num_prompts": int(len(prompt_scores)),
                "min_bank_size": int(min(bank_sizes)),
                "max_bank_size": int(max(bank_sizes)),
                "first_n_oracle_accuracy": float(sum(first_n_values) / len(first_n_values)),
                "exact_random_subset_oracle_accuracy": float(sum(exact_subset_values) / len(exact_subset_values)),
                "mean_sample_accuracy": float(
                    sum(sum(prompt.scores) / len(prompt.scores) for prompt in prompt_scores) / len(prompt_scores)
                ),
                "binary_scores": bool(is_binary),
            }
        )

    summary = {
        "bank_name": name,
        "num_prompts": int(len(prompt_scores)),
        "bank_sizes": sorted(int(size) for size in bank_sizes),
        "binary_scores": bool(is_binary),
        "budgets": valid_budgets,
        "full_bank_oracle_accuracy": float(
            sum(max(prompt.scores) for prompt in prompt_scores) / len(prompt_scores)
        ),
        "mean_sample_accuracy": rows[0]["mean_sample_accuracy"],
    }
    return rows, summary


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty CSV.")
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_pairwise_deltas(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows_by_bank_and_budget = {
        (str(row["bank_name"]), int(row["budget"])): dict(row) for row in rows
    }
    bank_names = sorted({str(row["bank_name"]) for row in rows})
    budgets = sorted({int(row["budget"]) for row in rows})
    deltas: list[dict[str, Any]] = []
    for left_index, left_name in enumerate(bank_names):
        for right_name in bank_names[left_index + 1 :]:
            for budget in budgets:
                left_row = rows_by_bank_and_budget.get((left_name, budget))
                right_row = rows_by_bank_and_budget.get((right_name, budget))
                if left_row is None or right_row is None:
                    continue
                deltas.append(
                    {
                        "comparison": f"{left_name}_minus_{right_name}",
                        "left_bank_name": left_name,
                        "right_bank_name": right_name,
                        "budget": int(budget),
                        "first_n_oracle_accuracy_diff": float(
                            left_row["first_n_oracle_accuracy"] - right_row["first_n_oracle_accuracy"]
                        ),
                        "exact_random_subset_oracle_accuracy_diff": float(
                            left_row["exact_random_subset_oracle_accuracy"]
                            - right_row["exact_random_subset_oracle_accuracy"]
                        ),
                    }
                )
    return deltas


def plot_curves(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if plt is None:
        raise RuntimeError(f"matplotlib import failed: {MATPLOTLIB_IMPORT_ERROR}")
    rows_by_bank: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_bank[str(row["bank_name"])].append(dict(row))

    fig, ax = plt.subplots(figsize=(8, 5))
    for bank_name, bank_rows in rows_by_bank.items():
        bank_rows = sorted(bank_rows, key=lambda row: int(row["budget"]))
        budgets = [int(row["budget"]) for row in bank_rows]
        first_n = [float(row["first_n_oracle_accuracy"]) for row in bank_rows]
        exact_subset = [float(row["exact_random_subset_oracle_accuracy"]) for row in bank_rows]
        ax.plot(budgets, first_n, marker="o", label=f"{bank_name} first-N")
        ax.plot(budgets, exact_subset, marker="s", linestyle="--", label=f"{bank_name} random subset")

    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted({int(row["budget"]) for row in rows}))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Search budget N")
    ax.set_ylabel("Oracle best-of-N accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    specs = parse_bank_specs(args.banks)
    budgets = sorted(dict.fromkeys(int(budget) for budget in args.budgets))
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for spec in specs:
        prompt_scores, metadata = load_prompt_scores(spec.path, args.score_key)
        rows, summary = summarize_bank(
            name=spec.name,
            prompt_scores=prompt_scores,
            budgets=budgets,
            allow_missing_budgets=bool(args.allow_missing_budgets),
        )
        all_rows.extend(rows)
        summaries[spec.name] = {**metadata, **summary}

    csv_path = output_dir / "oracle_curves.csv"
    deltas = build_pairwise_deltas(all_rows)
    deltas_csv_path = output_dir / "oracle_curve_pairwise_deltas.csv"
    json_path = output_dir / "oracle_curves_summary.json"
    write_csv(csv_path, all_rows)
    if deltas:
        write_csv(deltas_csv_path, deltas)
    with json_path.open("w", encoding="utf-8") as json_file:
        json.dump(
            {
                "score_key": args.score_key,
                "requested_budgets": budgets,
                "curve_definitions": {
                    "first_n_oracle_accuracy": "Mean over prompts of max(score[0:N]).",
                    "exact_random_subset_oracle_accuracy": (
                        "Mean over prompts of the exact expected max score for a uniformly random N-subset "
                        "without replacement from the full per-prompt bank."
                    ),
                },
                "banks": summaries,
                "rows": all_rows,
                "pairwise_deltas": deltas,
            },
            json_file,
            indent=2,
            sort_keys=True,
        )

    plot_path = output_dir / "oracle_curves.png"
    wrote_plot = False
    if not args.skip_plot:
        try:
            plot_curves(plot_path, all_rows)
        except Exception as exc:
            print(f"Warning: skipped plot because plotting failed: {exc}")
        else:
            wrote_plot = True

    print(f"Wrote {csv_path}")
    if deltas:
        print(f"Wrote {deltas_csv_path}")
    print(f"Wrote {json_path}")
    if wrote_plot:
        print(f"Wrote {plot_path}")
    for row in all_rows:
        print(
            f"{row['bank_name']} N={row['budget']:>2}: "
            f"first_N={row['first_n_oracle_accuracy']:.6f}, "
            f"random_subset_exact={row['exact_random_subset_oracle_accuracy']:.6f}"
        )
    for delta in deltas:
        print(
            f"{delta['comparison']} N={delta['budget']:>2}: "
            f"first_N_diff={delta['first_n_oracle_accuracy_diff']:+.6f}, "
            f"random_subset_exact_diff={delta['exact_random_subset_oracle_accuracy_diff']:+.6f}"
        )


if __name__ == "__main__":
    main()
