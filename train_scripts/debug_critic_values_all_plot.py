import json
from pathlib import Path
import numpy as np

out_dir = Path("/nfs/shuozhe/verl/outputs/critic_debug_all")
paths = sorted(out_dir.glob("responses_rank*.jsonl"))
if not paths:
    paths = [out_dir / "responses.jsonl"]

num_bins = 100  # set this to the same --num_bins you used

def accumulate(values, sum_arr, cnt_arr):
    n = len(values)
    if n == 0:
        return
    for i, v in enumerate(values):
        b = int((i / n) * num_bins)
        if b >= num_bins:
            b = num_bins - 1
        sum_arr[b] += float(v)
        cnt_arr[b] += 1

correct_sum = np.zeros(num_bins, dtype=np.float64)
correct_cnt = np.zeros(num_bins, dtype=np.int64)
wrong_sum = np.zeros(num_bins, dtype=np.float64)
wrong_cnt = np.zeros(num_bins, dtype=np.int64)
num_correct = 0
num_wrong = 0

for p in paths:
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            values = rec.get("values", [])
            if rec.get("correct", False):
                num_correct += 1
                accumulate(values, correct_sum, correct_cnt)
            else:
                num_wrong += 1
                accumulate(values, wrong_sum, wrong_cnt)

def finalize(sum_arr, cnt_arr):
    curve = np.full_like(sum_arr, np.nan, dtype=np.float64)
    mask = cnt_arr > 0
    curve[mask] = sum_arr[mask] / cnt_arr[mask]
    return curve.tolist()

correct_curve = finalize(correct_sum, correct_cnt)
wrong_curve = finalize(wrong_sum, wrong_cnt)

curves = {
    "num_bins": num_bins,
    "correct_curve": correct_curve,
    "wrong_curve": wrong_curve,
    "num_correct": num_correct,
    "num_wrong": num_wrong,
}
(out_dir / "curves.json").write_text(json.dumps(curves, indent=2), encoding="utf-8")

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
    print("Saved", out_dir / "curves.png")
except Exception as exc:
    print("Plot skipped:", exc)

print("Saved", out_dir / "curves.json")
