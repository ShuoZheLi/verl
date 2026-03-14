#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot checkpoint evaluation accuracy from summary.csv")
    parser.add_argument(
        "--summary",
        default="/data/shuozhe/verl/train_log/job_05b_vh_init_e5_new_lvl45/eval_ckpt_val_only/summary.csv",
        help="Path to summary.csv produced by eval_05b_vh_init_load_freeze_ckpts.py",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output image path. Default: <summary_dir>/accuracy_vs_step.png",
    )
    return parser.parse_args()


def parse_core_metric(core_metric: str) -> tuple[str, float]:
    if ":" not in core_metric:
        raise ValueError(f"Invalid core_metric format: {core_metric}")
    metric_name, value = core_metric.rsplit(":", 1)
    return metric_name, float(value)


def load_accuracy(summary_path: Path) -> tuple[list[int], list[float], str]:
    steps: list[int] = []
    accs: list[float] = []
    metric_name = ""

    with summary_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") != "ok":
                continue
            core_metric = (row.get("core_metric") or "").strip()
            if not core_metric:
                continue
            step = int(row["step"])
            m_name, value = parse_core_metric(core_metric)
            metric_name = m_name
            steps.append(step)
            accs.append(value)

    if not steps:
        raise ValueError(f"No valid 'ok' rows with core_metric found in {summary_path}")

    pairs = sorted(zip(steps, accs), key=lambda x: x[0])
    steps = [p[0] for p in pairs]
    accs = [p[1] for p in pairs]
    return steps, accs, metric_name


def save_curve_csv(out_csv: Path, steps: list[int], accs: list[float], metric_name: str) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "accuracy", "metric_name"])
        for step, acc in zip(steps, accs):
            writer.writerow([step, acc, metric_name])


def save_svg_plot(
    out_svg: Path,
    steps: list[int],
    accs: list[float],
    metric_name: str,
    best_step: int,
    best_acc: float,
) -> None:
    width, height = 1000, 560
    ml, mr, mt, mb = 80, 30, 55, 75
    pw, ph = width - ml - mr, height - mt - mb

    min_step, max_step = min(steps), max(steps)
    min_acc = min(accs)
    max_acc = max(accs)

    # Keep zero baseline when range is small and accuracy is positive.
    y0 = min(0.0, min_acc * 0.98)
    y1 = max_acc * 1.02 if max_acc > 0 else 1.0
    if abs(y1 - y0) < 1e-9:
        y1 = y0 + 1.0

    def sx(x: float) -> float:
        if max_step == min_step:
            return ml + pw / 2.0
        return ml + (x - min_step) / (max_step - min_step) * pw

    def sy(y: float) -> float:
        return mt + (y1 - y) / (y1 - y0) * ph

    points = " ".join(f"{sx(s):.2f},{sy(a):.2f}" for s, a in zip(steps, accs))
    best_x, best_y = sx(best_step), sy(best_acc)

    # y ticks
    y_ticks = 6
    y_lines = []
    y_labels = []
    for i in range(y_ticks + 1):
        v = y0 + (y1 - y0) * i / y_ticks
        y = sy(v)
        y_lines.append(
            f'<line x1="{ml}" y1="{y:.2f}" x2="{width-mr}" y2="{y:.2f}" stroke="#e6e6e6" stroke-width="1"/>'
        )
        y_labels.append(
            f'<text x="{ml-10}" y="{y+4:.2f}" font-size="12" text-anchor="end" fill="#444">{v:.3f}</text>'
        )

    # x ticks
    x_ticks = min(8, len(steps))
    x_lines = []
    x_labels = []
    for i in range(x_ticks + 1):
        if x_ticks == 0:
            break
        step_v = min_step + (max_step - min_step) * i / x_ticks
        x = sx(step_v)
        x_lines.append(
            f'<line x1="{x:.2f}" y1="{mt}" x2="{x:.2f}" y2="{height-mb}" stroke="#f1f1f1" stroke-width="1"/>'
        )
        x_labels.append(
            f'<text x="{x:.2f}" y="{height-mb+22}" font-size="12" text-anchor="middle" fill="#444">{int(round(step_v))}</text>'
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect x="0" y="0" width="{width}" height="{height}" fill="white"/>
  <text x="{width/2:.1f}" y="30" text-anchor="middle" font-size="22" fill="#222">Validation Accuracy vs Checkpoint Step</text>
  {''.join(x_lines)}
  {''.join(y_lines)}
  <line x1="{ml}" y1="{height-mb}" x2="{width-mr}" y2="{height-mb}" stroke="#444" stroke-width="1.5"/>
  <line x1="{ml}" y1="{mt}" x2="{ml}" y2="{height-mb}" stroke="#444" stroke-width="1.5"/>
  <polyline fill="none" stroke="#1666d9" stroke-width="2.5" points="{points}"/>
  <circle cx="{best_x:.2f}" cy="{best_y:.2f}" r="5" fill="#d91c1c"/>
  <text x="{min(best_x+12, width-240):.2f}" y="{max(best_y-10, mt+14):.2f}" font-size="12" fill="#d91c1c">
    best: step={best_step}, acc={best_acc:.4f}
  </text>
  {''.join(y_labels)}
  {''.join(x_labels)}
  <text x="{width/2:.1f}" y="{height-18}" text-anchor="middle" font-size="14" fill="#222">Checkpoint Step</text>
  <text transform="translate(22 {height/2:.1f}) rotate(-90)" text-anchor="middle" font-size="14" fill="#222">Accuracy</text>
  <text x="{width-mr}" y="{height-18}" text-anchor="end" font-size="11" fill="#666">{metric_name}</text>
</svg>
"""

    out_svg.parent.mkdir(parents=True, exist_ok=True)
    out_svg.write_text(svg, encoding="utf-8")


def main() -> int:
    args = parse_args()
    summary_path = Path(args.summary).resolve()
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.csv not found: {summary_path}")

    if args.out is None:
        out_png = summary_path.parent / "accuracy_vs_step.png"
    else:
        out_png = Path(args.out).resolve()

    steps, accs, metric_name = load_accuracy(summary_path)
    best_idx = max(range(len(accs)), key=lambda i: accs[i])
    best_step = steps[best_idx]
    best_acc = accs[best_idx]

    curve_csv = summary_path.parent / "accuracy_curve.csv"
    save_curve_csv(curve_csv, steps, accs, metric_name)
    svg_path = summary_path.parent / "accuracy_vs_step.svg"
    save_svg_plot(svg_path, steps, accs, metric_name, best_step, best_acc)

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print("matplotlib is not available, wrote dependency-free SVG + parsed values:")
        print(f"  {svg_path}")
        print(f"  {curve_csv}")
        print(f"Best: step={best_step}, accuracy={best_acc:.6f}")
        print(f"Reason: {e}")
        return 0

    plt.figure(figsize=(9, 5))
    plt.plot(steps, accs, marker="o", linewidth=2)
    plt.title("Validation Accuracy vs Checkpoint Step")
    plt.xlabel("Checkpoint Step")
    plt.ylabel("Accuracy")
    plt.grid(True, alpha=0.3)

    # Highlight best checkpoint
    plt.scatter([best_step], [best_acc], s=80)
    plt.annotate(
        f"best: step={best_step}, acc={best_acc:.4f}",
        xy=(best_step, best_acc),
        xytext=(8, 8),
        textcoords="offset points",
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)

    print(f"Saved plot: {out_png}")
    print(f"Saved SVG: {svg_path}")
    print(f"Saved parsed curve: {curve_csv}")
    print(f"Metric: {metric_name}")
    print(f"Best: step={best_step}, accuracy={best_acc:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
