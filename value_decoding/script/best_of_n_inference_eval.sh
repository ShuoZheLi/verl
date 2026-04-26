#!/usr/bin/env bash

set -eo pipefail

# =============================================================================
# STAGE 2 BEST-OF-N INFERENCE EVALUATION
# Shared actor response bank, then response-level best-of-N selection.
#
# Important selectors in the output:
# - oracle_best_in_bank:
#     argmax over task_score, where task_score is computed from ground truth via
#     value_decoding.data.score_response -> verl.utils.reward_score.default_compute_score
# - best_of_n_old_critic / best_of_n_new_critic:
#     both selectors are backed by the same critic checkpoint in this launcher,
#     and score critic(prefix + full_response)[last_token_position]
# =============================================================================

# -----------------------------
# Checkpoints
# -----------------------------
ACTOR_CHECKPOINT_DIR="/data/shuozhe/verl/train_log/job_05b_vh_init_e5_metamath/global_step_800"
CRITIC_CHECKPOINT_DIR="/data/shuozhe/verl/train_log/job_policy_gs800_dsk_1d5b_critic/global_step_750"
ACTOR_HF_SOURCE_DIR=""
OLD_CRITIC_HF_SOURCE_DIR=""
NEW_CRITIC_HF_SOURCE_DIR=""

ACTOR_MERGED_ROOT=""
CRITIC_MERGED_ROOT=""

# -----------------------------
# Data
# -----------------------------
DATASET_PATH="/data/shuozhe/saved_dataset/MetaMathQA-math-500/test.parquet"
OUTPUT_DIR="/data/shuozhe/verl/value_decoding/output/stage2_best_of_n_inference_eval"
PROMPT_KEY="prompt"

# Leave RESPONSE_KEY empty for MetaMathQA-math-500:
# ground truth is read from reward_model.ground_truth.
RESPONSE_KEY=""

START_INDEX=0
MAX_EXAMPLES=500
SHUFFLE_EXAMPLES=0

# -----------------------------
# Bank / N sweep
# -----------------------------
N_VALUES="2 4 8"

# Leave empty to use max(N_VALUES).
MAX_BANK_SIZE=""

MAX_PROMPT_LENGTH=2048
MAX_NEW_TOKENS=2048
CRITIC_SCORE_BATCH_SIZE=8
BOOTSTRAP_SAMPLES=2000
DTYPE="bf16"

# -----------------------------
# Devices
# -----------------------------
# Examples:
#   single-worker:
#     DEVICE=""
#     ACTOR_DEVICE="cuda:0"
#     CRITIC_DEVICE="cuda:1"
#
#   prompt-sharded (each worker loads its own actor + critic, so check GPU memory first):
#     WORKER_LAYOUTS="cuda:0,cuda:1,cuda:1 cuda:2,cuda:3,cuda:3"

DEVICE=""
ACTOR_DEVICE="cuda:0"
CRITIC_DEVICE="cuda:1"
WORKER_LAYOUTS=""

# -----------------------------
# Actor sampling
# -----------------------------
ACTOR_SAMPLING_MODE="sample"
ACTOR_TEMPERATURE=1.0
ACTOR_TOP_P=1.0
ACTOR_TOP_K=0
SEED="42"

# -----------------------------
# Validation / plots
# -----------------------------
REFERENCE_STAGE1_TRAJECTORY_BANK=""
SKIP_PLOTS=0
PLOT_DPI=160

# -----------------------------
# Misc
# -----------------------------
TRUST_REMOTE_CODE=0
SKIP_MERGE=0
DISABLE_ACTOR_CACHE=0

source /data/shuozhe/miniconda3/etc/profile.d/conda.sh
conda activate verl
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
EXTRA_ARGS=("$@")

validate_component_checkpoint() {
  local checkpoint_dir="$1"
  local component="$2"
  python - "$checkpoint_dir" "$component" <<'PY'
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

read -r -a N_VALUES_ARR <<< "$N_VALUES"
read -r -a WORKER_LAYOUTS_ARR <<< "$WORKER_LAYOUTS"
read -r -a SEED_ARR <<< "$SEED"

validate_component_checkpoint "$ACTOR_CHECKPOINT_DIR" actor
validate_component_checkpoint "$CRITIC_CHECKPOINT_DIR" critic

mkdir -p "$OUTPUT_DIR"

if [[ ${#SEED_ARR[@]} -eq 0 ]]; then
  echo "SEED must contain at least one value." >&2
  exit 1
fi

if [[ ${#SEED_ARR[@]} -gt 1 ]]; then
  for extra_arg in "${EXTRA_ARGS[@]}"; do
    case "$extra_arg" in
      --seed|--seed=*|--output_dir|--output_dir=*)
        echo "When SEED contains multiple values, do not override --seed or --output_dir via extra CLI args." >&2
        exit 1
        ;;
    esac
  done
fi

seed_to_id() {
  local seed_value="$1"
  local seed_id="${seed_value//-/m}"
  seed_id="${seed_id//./p}"
  printf '%s' "$seed_id"
}

run_one_seed() {
  local seed_value="$1"
  local output_dir_for_seed="$2"

  CMD=(
    python -m value_decoding.best_of_n_inference_eval
    --actor_checkpoint_dir      "$ACTOR_CHECKPOINT_DIR"
    --old_critic_checkpoint_dir "$CRITIC_CHECKPOINT_DIR"
    --new_critic_checkpoint_dir "$CRITIC_CHECKPOINT_DIR"
    --dataset_path              "$DATASET_PATH"
    --output_dir                "$output_dir_for_seed"
    --prompt_key                "$PROMPT_KEY"
    --start_index               "$START_INDEX"
    --max_prompt_length         "$MAX_PROMPT_LENGTH"
    --max_new_tokens            "$MAX_NEW_TOKENS"
    --critic_score_batch_size   "$CRITIC_SCORE_BATCH_SIZE"
    --bootstrap_samples         "$BOOTSTRAP_SAMPLES"
    --dtype                     "$DTYPE"
    --seed                      "$seed_value"
    --actor_sampling_mode       "$ACTOR_SAMPLING_MODE"
    --actor_temperature         "$ACTOR_TEMPERATURE"
    --actor_top_p               "$ACTOR_TOP_P"
    --actor_top_k               "$ACTOR_TOP_K"
    --n_values                  "${N_VALUES_ARR[@]}"
    --plot_dpi                  "$PLOT_DPI"
  )

  [[ -n "$ACTOR_MERGED_ROOT" ]] && CMD+=(--actor_merged_root "$ACTOR_MERGED_ROOT")
  [[ -n "$CRITIC_MERGED_ROOT" ]] && CMD+=(--old_critic_merged_root "$CRITIC_MERGED_ROOT" --new_critic_merged_root "$CRITIC_MERGED_ROOT")
  [[ -n "$ACTOR_HF_SOURCE_DIR" ]] && CMD+=(--actor_hf_source_dir "$ACTOR_HF_SOURCE_DIR")
  [[ -n "$OLD_CRITIC_HF_SOURCE_DIR" ]] && CMD+=(--old_critic_hf_source_dir "$OLD_CRITIC_HF_SOURCE_DIR")
  [[ -n "$NEW_CRITIC_HF_SOURCE_DIR" ]] && CMD+=(--new_critic_hf_source_dir "$NEW_CRITIC_HF_SOURCE_DIR")
  [[ -n "$RESPONSE_KEY" ]] && CMD+=(--response_key "$RESPONSE_KEY")
  [[ -n "$MAX_EXAMPLES" ]] && CMD+=(--max_examples "$MAX_EXAMPLES")
  [[ -n "$MAX_BANK_SIZE" ]] && CMD+=(--max_bank_size "$MAX_BANK_SIZE")
  [[ -n "$DEVICE" ]] && CMD+=(--device "$DEVICE")
  [[ -n "$ACTOR_DEVICE" ]] && CMD+=(--actor_device "$ACTOR_DEVICE")
  [[ -n "$CRITIC_DEVICE" ]] && CMD+=(--old_critic_device "$CRITIC_DEVICE" --new_critic_device "$CRITIC_DEVICE")
  [[ ${#WORKER_LAYOUTS_ARR[@]} -gt 0 ]] && CMD+=(--worker_layouts "${WORKER_LAYOUTS_ARR[@]}")
  [[ -n "$REFERENCE_STAGE1_TRAJECTORY_BANK" ]] && CMD+=(--reference_stage1_trajectory_bank "$REFERENCE_STAGE1_TRAJECTORY_BANK")
  [[ "$SHUFFLE_EXAMPLES" != "0" ]] && CMD+=(--shuffle_examples)
  [[ "$TRUST_REMOTE_CODE" != "0" ]] && CMD+=(--trust_remote_code)
  [[ "$SKIP_MERGE" != "0" ]] && CMD+=(--skip_merge)
  [[ "$DISABLE_ACTOR_CACHE" != "0" ]] && CMD+=(--disable_actor_cache)
  [[ "$SKIP_PLOTS" != "0" ]] && CMD+=(--skip_plots)
  [[ ${#EXTRA_ARGS[@]} -gt 0 ]] && CMD+=("${EXTRA_ARGS[@]}")

  mkdir -p "$output_dir_for_seed"
  (cd "$REPO_DIR" && "${CMD[@]}")
}

if [[ ${#SEED_ARR[@]} -eq 1 ]]; then
  run_one_seed "${SEED_ARR[0]}" "$OUTPUT_DIR"
  exit 0
fi

SUMMARY_PATHS=()
SEED_OUTPUT_DIRS=()
for seed_value in "${SEED_ARR[@]}"; do
  seed_id="$(seed_to_id "$seed_value")"
  seed_output_dir="${OUTPUT_DIR}/seed_${seed_id}"
  run_one_seed "$seed_value" "$seed_output_dir"
  SUMMARY_PATHS+=("${seed_output_dir}/summary_metrics.json")
  SEED_OUTPUT_DIRS+=("${seed_output_dir}")
done

python -m value_decoding.multi_seed_summary \
  --output_path "${OUTPUT_DIR}/summary_metrics.json" \
  --source_script "$SCRIPT_PATH" \
  --seed_values "${SEED_ARR[@]}" \
  --summary_paths "${SUMMARY_PATHS[@]}" \
  --seed_output_dirs "${SEED_OUTPUT_DIRS[@]}"
