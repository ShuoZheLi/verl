#!/bin/bash
#SBATCH --job-name=collect_search_data
#SBATCH --account=ECS26006
#SBATCH --partition=gh
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:20:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

# -----------------------------
# Environment setup
# -----------------------------
module reset
module load nvidia/25.9

VENV="/work/09576/shuozhe/verl_setup_tacc/.venv"
source "${VENV}/bin/activate"

UV_CACHE_DIR="${SCRATCH}/.cache/uv"
HF_HOME="${SCRATCH}/.cache/huggingface"
TIKTOKEN_ENCODINGS_BASE="${SCRATCH}/data/embeddings"

mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$TIKTOKEN_ENCODINGS_BASE"

export UV_CACHE_DIR
export HF_HOME
export TIKTOKEN_ENCODINGS_BASE
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=true
export VLLM_USE_V1=1

echo "Activated environment"
echo "Python: $(which python3)"
python3 -V

# -----------------------------
# Run identity
# -----------------------------
RUN_NAME="search_induced_critic_data"
RUN_ID="${RUN_NAME}_${SLURM_JOB_ID}"

# -----------------------------
# Paths
# -----------------------------
# Edit these for the actor/collector pair you want to collect from.
ACTOR_CHECKPOINT_DIR="/scratch/10587/npg493/verl_runs/low_ent_critic_training_ckpt_750_actor_697767/train_log/global_step_600"
COLLECTOR_CRITIC_CHECKPOINT_DIR="/scratch/10587/npg493/verl_runs/low_ent_critic_training_ckpt_750_actor_697767/train_log/global_step_600"
DATASET_PATH="/work2/09576/shuozhe/saved_dataset/MetaMathQA-math-500/train.parquet"
WORK_DIR="/work2/09576/shuozhe/verl"
export PYTHONPATH="${WORK_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

ARCHIVE_ROOT="/work2/09576/shuozhe/verl/value_decoding/output_archive"
ARCHIVE_DIR="${ARCHIVE_ROOT}/${RUN_ID}"

SCRATCH_ROOT="${SCRATCH}/value_decoding_runs"
RUN_DIR="${SCRATCH_ROOT}/${RUN_ID}"
LOG_DIR="${RUN_DIR}/logs"
SHARD_ROOT="${RUN_DIR}/shards"
OUTPUT_DIR="${RUN_DIR}/search_induced_critic_data"

mkdir -p "$LOG_DIR" "$SHARD_ROOT" "$ARCHIVE_ROOT" "$OUTPUT_DIR"

# Optional override directories for HF config/tokenizer metadata used during FSDP merge.
# Leave empty for normal VERL checkpoint layouts.
ACTOR_HF_SOURCE_DIR=""
CRITIC_HF_SOURCE_DIR=""

# -----------------------------
# Collection config
# -----------------------------
PROMPT_KEY="prompt"
RESPONSE_KEY=""
START_INDEX=0
MAX_PROMPTS=2000
SHUFFLE_PROMPTS=0

CHUNK_SIZE=128
NUM_CHUNK_CANDIDATES=8
NUM_SEARCH_STEPS_PER_PROMPT=20
COMPLETION_MAX_NEW_TOKENS=2048
COLLECTOR_SELECTION_MODE="argmax"  # argmax, epsilon_greedy, softmax_value, actor_logprob, random
COLLECTOR_EPSILON=0.0
COLLECTOR_VALUE_TEMPERATURE=1.0

ACTOR_TEMPERATURE=1.0
ACTOR_TOP_P=1.0
ACTOR_TOP_K=0
ACTOR_BATCH_SIZE=8

MAX_PROMPT_LENGTH=2048
DTYPE="bf16"
SEED=42
GENERATION_BACKEND="vllm"
CRITIC_BATCH_SIZE=8

# vLLM shares the node GPU with the critic in this single-process collector.
# Keep this conservative if critic memory is tight.
VLLM_GPU_MEMORY_UTILIZATION=0.60
VLLM_TENSOR_PARALLEL_SIZE=1
VLLM_MAX_MODEL_LEN=""
VLLM_ENFORCE_EAGER=0

SAVE_FULL_TEXT=1
SAVE_TOKEN_IDS=1
SAVE_COMPLETED_RESPONSES=0
TRUST_REMOTE_CODE=1
SKIP_MERGE=0

# If >0, overrides MAX_PROMPTS for a tiny debug run.
DEBUG_NUM_PROMPTS=""

# -----------------------------
# Helpers
# -----------------------------
nodes_array=()
SCRIPT_PATH="${WORK_DIR}/value_decoding/script/$(basename "${BASH_SOURCE[0]}")"

sync_to_work() {
  echo "Syncing run directory back to WORK..."
  mkdir -p "$ARCHIVE_DIR"
  rsync -a "$RUN_DIR"/ "$ARCHIVE_DIR"/ || true
  echo "Archived run to: $ARCHIVE_DIR"
}

cleanup() {
  sync_to_work
}
trap cleanup EXIT

validate_component_checkpoint() {
  local checkpoint_dir="$1"
  local component="$2"
  python3 - "$checkpoint_dir" "$component" <<'PY'
import sys
from pathlib import Path

from value_decoding.checkpointing import (
    find_missing_hf_weight_files,
    has_complete_hf_checkpoint,
    has_fsdp_checkpoint_shards,
    has_hf_config,
    resolve_component_checkpoint_dir,
)

checkpoint_dir = Path(sys.argv[1])
component = sys.argv[2]
component_dir = resolve_component_checkpoint_dir(checkpoint_dir, component=component)

if has_complete_hf_checkpoint(component_dir):
    print(f"{component}: detected complete Hugging Face checkpoint at {component_dir}")
    raise SystemExit(0)

if has_fsdp_checkpoint_shards(component_dir):
    print(f"{component}: detected raw FSDP checkpoint at {component_dir}")
    raise SystemExit(0)

if has_hf_config(component_dir):
    missing_files = find_missing_hf_weight_files(component_dir)
    missing_preview = ", ".join(path.name for path in missing_files[:5]) or "unknown weight shards"
    if len(missing_files) > 5:
        missing_preview += ", ..."
    raise SystemExit(
        f"{component}: incomplete Hugging Face checkpoint at {component_dir}. "
        f"Missing files referenced by the index: {missing_preview}"
    )

raise SystemExit(f"{component}: unsupported checkpoint layout at {component_dir}")
PY
}

count_dataset_rows() {
  python3 - "$DATASET_PATH" <<'PY'
import sys
import pandas as pd
print(len(pd.read_parquet(sys.argv[1], columns=[])))
PY
}

merge_shards() {
  python3 - "$SHARD_ROOT" "$OUTPUT_DIR" "$RUN_ID" <<'PY'
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

shard_root = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
run_id = sys.argv[3]
output_dir.mkdir(parents=True, exist_ok=True)

candidate_out = output_dir / "search_induced_candidates.jsonl"
prompt_out = output_dir / "prompt_summaries.jsonl"

candidate_rows = []
prompt_rows = []
shard_dirs = sorted(path for path in shard_root.glob("shard_*") if path.is_dir())
for shard_dir in shard_dirs:
    candidate_path = shard_dir / "search_induced_candidates.jsonl"
    prompt_path = shard_dir / "prompt_summaries.jsonl"
    if candidate_path.exists():
        with candidate_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    candidate_rows.append(json.loads(line))
    if prompt_path.exists():
        with prompt_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    prompt_rows.append(json.loads(line))

with candidate_out.open("w", encoding="utf-8") as handle:
    for row in candidate_rows:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
with prompt_out.open("w", encoding="utf-8") as handle:
    for row in prompt_rows:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")

first_config = None
for shard_dir in shard_dirs:
    config_path = shard_dir / "config.json"
    if config_path.exists():
        first_config = json.loads(config_path.read_text(encoding="utf-8"))
        break
if first_config is not None:
    merged_config = dict(first_config)
    merged_config["run_id"] = run_id
    merged_config["shard_root"] = str(shard_root)
    merged_config["output_dir"] = str(output_dir)
    merged_config["num_shards"] = len(shard_dirs)
    (output_dir / "config.json").write_text(json.dumps(merged_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

candidate_rewards = [float(row["mc_reward"]) for row in candidate_rows]
candidate_values = [float(row["collector_value"]) for row in candidate_rows]

def mean(values):
    values = list(values)
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))

def corr(x, y, spearman=False):
    if len(x) != len(y) or len(x) < 2:
        return None
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if spearman:
        x = rankdata(x)
        y = rankdata(y)
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])

def rankdata(values):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return ranks

groups = {}
for row in candidate_rows:
    groups.setdefault(row["candidate_group_id"], []).append(row)
group_summaries = []
for rows in groups.values():
    first = rows[0]
    group_summaries.append(
        {
            "group_success_rate": float(first["group_success_rate"]),
            "group_oracle_reward": float(first["group_oracle_reward"]),
            "group_random_reward_mean": float(first["group_random_reward_mean"]),
            "group_collector_top1_reward": float(first["group_collector_top1_reward"]),
            "group_false_high_selected": bool(first["group_false_high_selected"]),
            "pairwise": first.get("collector_pairwise_ranking_accuracy_in_group"),
        }
    )
rankable = [group for group in group_summaries if group["pairwise"] is not None]
oracle_positive = [group for group in group_summaries if group["group_oracle_reward"] == 1.0]

summary = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "run_id": run_id,
    "shard_root": str(shard_root),
    "output_dir": str(output_dir),
    "num_shards": len(shard_dirs),
    "num_prompts": len(prompt_rows),
    "num_candidate_examples": len(candidate_rows),
    "num_candidate_groups": len(group_summaries),
    "candidate_level": {
        "mean_mc_reward": mean(candidate_rewards),
        "mean_collector_value": mean(candidate_values),
        "pearson_collector_value_vs_reward": corr(candidate_values, candidate_rewards),
        "spearman_collector_value_vs_reward": corr(candidate_values, candidate_rewards, spearman=True),
    },
    "group_level": {
        "fraction_rankable_groups": float(len(rankable) / len(group_summaries)) if group_summaries else None,
        "mean_group_success_rate": mean(group["group_success_rate"] for group in group_summaries),
        "mean_group_oracle_reward": mean(group["group_oracle_reward"] for group in group_summaries),
        "mean_group_random_reward": mean(group["group_random_reward_mean"] for group in group_summaries),
        "mean_group_collector_top1_reward": mean(group["group_collector_top1_reward"] for group in group_summaries),
        "collector_recovery_rate": (
            mean(group["group_collector_top1_reward"] == 1.0 for group in oracle_positive)
            if oracle_positive
            else None
        ),
        "false_high_selected_rate": (
            mean(group["group_false_high_selected"] for group in oracle_positive) if oracle_positive else None
        ),
        "collector_pairwise_ranking_accuracy": mean(group["pairwise"] for group in rankable),
    },
    "committed_path": {
        "mean_num_steps_collected_per_prompt": mean(row["num_search_steps_collected"] for row in prompt_rows),
        "mean_final_committed_prefix_length": mean(row["final_committed_prefix_length"] for row in prompt_rows),
        "fraction_prompts_stopped_by_eos": mean(row["committed_prefix_has_eos"] for row in prompt_rows),
    },
}
(output_dir / "summary_metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
(output_dir / "README.md").write_text(
    "# Merged Search-Induced Critic Data\n\n"
    f"Merged from `{len(shard_dirs)}` shard directories under `{shard_root}`.\n\n"
    "Shard-level outputs are preserved under the shard root; this directory contains concatenated JSONL files "
    "and aggregate metrics recomputed from all candidate rows.\n",
    encoding="utf-8",
)
print(f"Merged {len(candidate_rows)} candidate rows and {len(prompt_rows)} prompt summaries into {output_dir}")
PY
}

# -----------------------------
# Validate config
# -----------------------------
if [[ "$MAX_PROMPTS" -le 0 ]]; then
  echo "MAX_PROMPTS must be positive." >&2
  exit 1
fi
if [[ "$NUM_CHUNK_CANDIDATES" -le 0 || "$NUM_SEARCH_STEPS_PER_PROMPT" -le 0 ]]; then
  echo "NUM_CHUNK_CANDIDATES and NUM_SEARCH_STEPS_PER_PROMPT must be positive." >&2
  exit 1
fi
if [[ "$SHUFFLE_PROMPTS" != "0" ]]; then
  echo "SHUFFLE_PROMPTS is disabled in this sharded launcher because per-shard shuffling would overlap prompts." >&2
  echo "Shuffle the parquet upstream or run the single-node launcher if you need script-level shuffling." >&2
  exit 1
fi

# -----------------------------
# Debug info
# -----------------------------
echo "Job ID: $SLURM_JOB_ID"
echo "Run ID: $RUN_ID"
echo "SLURM nodes: $SLURM_JOB_NODELIST"
echo "SCRATCH: $SCRATCH"
echo "RUN_DIR: $RUN_DIR"
echo "SHARD_ROOT: $SHARD_ROOT"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "ARCHIVE_DIR: $ARCHIVE_DIR"

echo "Checking inputs..."
ls -ld "$WORK_DIR"
ls -lh "$DATASET_PATH"
ls -ld "$ACTOR_CHECKPOINT_DIR"
ls -ld "$COLLECTOR_CRITIC_CHECKPOINT_DIR"
validate_component_checkpoint "$ACTOR_CHECKPOINT_DIR" actor
validate_component_checkpoint "$COLLECTOR_CRITIC_CHECKPOINT_DIR" critic
cp "$SCRIPT_PATH" "$LOG_DIR/$(basename "$SCRIPT_PATH")"

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
NUM_SHARDS=${#nodes_array[@]}
DATASET_ROWS=$(count_dataset_rows)
AVAILABLE_ROWS=$((DATASET_ROWS - START_INDEX))
if [[ "$AVAILABLE_ROWS" -le 0 ]]; then
  echo "START_INDEX=${START_INDEX} is outside dataset with ${DATASET_ROWS} rows." >&2
  exit 1
fi
TOTAL_PROMPTS=$MAX_PROMPTS
if [[ "$TOTAL_PROMPTS" -gt "$AVAILABLE_ROWS" ]]; then
  TOTAL_PROMPTS=$AVAILABLE_ROWS
fi
if [[ -n "$DEBUG_NUM_PROMPTS" && "$DEBUG_NUM_PROMPTS" -gt 0 && "$DEBUG_NUM_PROMPTS" -lt "$TOTAL_PROMPTS" ]]; then
  TOTAL_PROMPTS=$DEBUG_NUM_PROMPTS
fi
PROMPTS_PER_SHARD=$(((TOTAL_PROMPTS + NUM_SHARDS - 1) / NUM_SHARDS))

echo "Dataset rows: $DATASET_ROWS"
echo "Total prompts requested after bounds/debug: $TOTAL_PROMPTS"
echo "Num shards: $NUM_SHARDS"
echo "Prompts per shard: $PROMPTS_PER_SHARD"

# -----------------------------
# Run collection shards
# -----------------------------
cd "$WORK_DIR"

pids=()
for shard_index in "${!nodes_array[@]}"; do
  node="${nodes_array[$shard_index]}"
  shard_start=$((START_INDEX + shard_index * PROMPTS_PER_SHARD))
  remaining=$((START_INDEX + TOTAL_PROMPTS - shard_start))
  if [[ "$remaining" -le 0 ]]; then
    echo "Skipping shard $shard_index on $node: no prompts assigned."
    continue
  fi
  shard_max=$PROMPTS_PER_SHARD
  if [[ "$shard_max" -gt "$remaining" ]]; then
    shard_max=$remaining
  fi

  shard_dir="${SHARD_ROOT}/shard_$(printf '%03d' "$shard_index")"
  shard_log="${LOG_DIR}/collector_shard_$(printf '%03d' "$shard_index").log"
  mkdir -p "$shard_dir"

  CMD=(
    python3 value_decoding/collect_search_induced_critic_data.py
    --actor_checkpoint_dir "$ACTOR_CHECKPOINT_DIR"
    --collector_critic_checkpoint_dir "$COLLECTOR_CRITIC_CHECKPOINT_DIR"
    --dataset_path "$DATASET_PATH"
    --output_dir "$shard_dir"
    --prompt_key "$PROMPT_KEY"
    --start_index "$shard_start"
    --max_prompts "$shard_max"
    --chunk_size "$CHUNK_SIZE"
    --num_chunk_candidates "$NUM_CHUNK_CANDIDATES"
    --num_search_steps_per_prompt "$NUM_SEARCH_STEPS_PER_PROMPT"
    --completion_max_new_tokens "$COMPLETION_MAX_NEW_TOKENS"
    --collector_selection_mode "$COLLECTOR_SELECTION_MODE"
    --collector_epsilon "$COLLECTOR_EPSILON"
    --collector_value_temperature "$COLLECTOR_VALUE_TEMPERATURE"
    --actor_temperature "$ACTOR_TEMPERATURE"
    --actor_top_p "$ACTOR_TOP_P"
    --actor_top_k "$ACTOR_TOP_K"
    --actor_batch_size "$ACTOR_BATCH_SIZE"
    --max_prompt_length "$MAX_PROMPT_LENGTH"
    --dtype "$DTYPE"
    --seed "$((SEED + shard_index))"
    --generation_backend "$GENERATION_BACKEND"
    --critic_device cuda:0
    --merged_root "$shard_dir/merged_hf"
    --critic_batch_size "$CRITIC_BATCH_SIZE"
    --vllm_gpu_memory_utilization "$VLLM_GPU_MEMORY_UTILIZATION"
    --vllm_tensor_parallel_size "$VLLM_TENSOR_PARALLEL_SIZE"
  )

  [[ -n "$RESPONSE_KEY" ]] && CMD+=(--response_key "$RESPONSE_KEY")
  [[ -n "$ACTOR_HF_SOURCE_DIR" ]] && CMD+=(--actor_hf_source_dir "$ACTOR_HF_SOURCE_DIR")
  [[ -n "$CRITIC_HF_SOURCE_DIR" ]] && CMD+=(--critic_hf_source_dir "$CRITIC_HF_SOURCE_DIR")
  [[ -n "$VLLM_MAX_MODEL_LEN" ]] && CMD+=(--vllm_max_model_len "$VLLM_MAX_MODEL_LEN")
  [[ "$VLLM_ENFORCE_EAGER" != "0" ]] && CMD+=(--vllm_enforce_eager true)
  [[ "$SAVE_FULL_TEXT" == "0" ]] && CMD+=(--save_full_text false)
  [[ "$SAVE_TOKEN_IDS" == "0" ]] && CMD+=(--save_token_ids false)
  [[ "$SAVE_COMPLETED_RESPONSES" != "0" ]] && CMD+=(--save_completed_responses true)
  [[ "$TRUST_REMOTE_CODE" != "0" ]] && CMD+=(--trust_remote_code true)
  [[ "$SKIP_MERGE" != "0" ]] && CMD+=(--skip_merge true)

  shard_runner="${LOG_DIR}/run_collector_shard_$(printf '%03d' "$shard_index").sh"
  {
    echo "#!/bin/bash"
    echo "set -euo pipefail"
    printf 'source %q\n' "${VENV}/bin/activate"
    printf 'cd %q\n' "$WORK_DIR"
    printf 'exec'
    printf ' %q' "${CMD[@]}"
    printf '\n'
  } > "$shard_runner"
  chmod +x "$shard_runner"

  {
    echo "Shard $shard_index on $node"
    echo "Shard start: $shard_start"
    echo "Shard max prompts: $shard_max"
    printf 'Command:'
    printf ' %q' "${CMD[@]}"
    printf '\n'
    echo "Runner: $shard_runner"
  } > "$shard_log"

  echo "Launching shard $shard_index on $node: start=$shard_start max=$shard_max"
  srun --exclusive --nodes=1 --ntasks=1 -w "$node" bash "$shard_runner" >> "$shard_log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" != "0" ]]; then
  echo "At least one collection shard failed. See $LOG_DIR/collector_shard_*.log" >&2
  exit 1
fi

# -----------------------------
# Merge shard outputs
# -----------------------------
merge_shards

echo "Search-induced critic data collection finished successfully."
echo "Merged output: $OUTPUT_DIR"
