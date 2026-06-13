#!/usr/bin/env bash

set -euo pipefail

# =============================================================================
# TRAJECTORY VALUE BASELINE EVAL - LOCAL
# Compare critic trajectory-value MSE against constant policy-accuracy baseline.
# =============================================================================

# --- Checkpoints --------------------------------------------------------------
ACTOR_CHECKPOINT_DIR="/data/shuozhe/verl/train_log/job_05b_vh_init_e5_metamath/global_step_800"
CRITIC_CHECKPOINT_DIRS="/data/shuozhe/verl/train_log/job_05b_vh_init_e5_metamath/global_step_800"

# --- Data ---------------------------------------------------------------------
DATASET_PATH="/data/shuozhe/saved_dataset/MetaMathQA-math-500/test.parquet"
OUTPUT_DIR="/data/shuozhe/verl/value_decoding/output/trajectory_value_baseline_1d5_critic"
PROMPT_KEY="prompt"
RESPONSE_KEY="ground_truth"
START_INDEX=0
MAX_EXAMPLES=500
SHUFFLE=0

# Optional existing bank. If non-empty, skips actor generation and scores rows.
RESPONSE_BANK_PATH=""

# --- Generation ---------------------------------------------------------------
NUM_SAMPLES_PER_PROMPT=1
MAX_PROMPT_LENGTH=2048
MAX_NEW_TOKENS=2048
TEMPERATURE=1.0
TOP_P=1.0
TOP_K=0
BATCH_SIZE=128
GENERATION_BACKEND="vllm"  # vllm accelerates actor generation; use "torch" to disable.
VLLM_GPU_MEMORY_UTILIZATION=0.45
VLLM_TENSOR_PARALLEL_SIZE=1
VLLM_MAX_NUM_SEQS=""
VLLM_ENFORCE_EAGER=1

# --- Critic readout -----------------------------------------------------------
# Recommended: pre_eos. Also useful: tail_mean_8 tail_mean_16.
VALUE_POSITIONS="pre_eos"

# --- Runtime ------------------------------------------------------------------
DTYPE="bf16"
DEVICE="cuda:0"
CRITIC_DEVICE="cuda:0"
TRUST_REMOTE_CODE=0
MATH_DAPO_BINARY_REWARD=1
SEED=42

source /data/shuozhe/miniconda3/etc/profile.d/conda.sh
conda activate verl

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

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

validate_component_checkpoint "${ACTOR_CHECKPOINT_DIR}" actor
read -r -a CRITIC_CHECKPOINT_DIRS_ARR <<< "${CRITIC_CHECKPOINT_DIRS}"
if [[ ${#CRITIC_CHECKPOINT_DIRS_ARR[@]} -eq 0 ]]; then
  echo "CRITIC_CHECKPOINT_DIRS must contain at least one checkpoint." >&2
  exit 1
fi
for critic_checkpoint_dir in "${CRITIC_CHECKPOINT_DIRS_ARR[@]}"; do
  validate_component_checkpoint "${critic_checkpoint_dir}" critic
done
mkdir -p "${OUTPUT_DIR}"

read -r -a VALUE_POSITIONS_ARR <<< "${VALUE_POSITIONS}"
if [[ ${#VALUE_POSITIONS_ARR[@]} -eq 0 ]]; then
  echo "VALUE_POSITIONS must contain at least one value." >&2
  exit 1
fi

run_eval() {
  local log_path="${OUTPUT_DIR}/eval.log"

  CMD=(
    python -m value_decoding.eval_trajectory_value_baseline
    --actor_checkpoint_dir "${ACTOR_CHECKPOINT_DIR}"
    --critic_checkpoint_dir "${CRITIC_CHECKPOINT_DIRS_ARR[@]}"
    --dataset_path "${DATASET_PATH}"
    --output_dir "${OUTPUT_DIR}"
    --prompt_key "${PROMPT_KEY}"
    --response_key "${RESPONSE_KEY}"
    --start_index "${START_INDEX}"
    --max_examples "${MAX_EXAMPLES}"
    --num_samples_per_prompt "${NUM_SAMPLES_PER_PROMPT}"
    --max_prompt_length "${MAX_PROMPT_LENGTH}"
    --max_new_tokens "${MAX_NEW_TOKENS}"
    --temperature "${TEMPERATURE}"
    --top_p "${TOP_P}"
    --top_k "${TOP_K}"
    --value_position "${VALUE_POSITIONS_ARR[@]}"
    --seed "${SEED}"
    --dtype "${DTYPE}"
    --batch_size "${BATCH_SIZE}"
    --generation_backend "${GENERATION_BACKEND}"
    --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}"
    --vllm_tensor_parallel_size "${VLLM_TENSOR_PARALLEL_SIZE}"
  )

  [[ -n "${DEVICE}" ]] && CMD+=(--device "${DEVICE}")
  [[ -n "${CRITIC_DEVICE}" ]] && CMD+=(--critic_device "${CRITIC_DEVICE}")
  [[ -n "${RESPONSE_BANK_PATH}" ]] && CMD+=(--response_bank_path "${RESPONSE_BANK_PATH}")
  [[ -n "${VLLM_MAX_NUM_SEQS}" ]] && CMD+=(--vllm_max_num_seqs "${VLLM_MAX_NUM_SEQS}")
  [[ "${VLLM_ENFORCE_EAGER}" != "0" ]] && CMD+=(--vllm_enforce_eager)
  [[ "${VLLM_ENFORCE_EAGER}" == "0" ]] && CMD+=(--no-vllm_enforce_eager)
  [[ "${SHUFFLE}" != "0" ]] && CMD+=(--shuffle)
  [[ "${TRUST_REMOTE_CODE}" != "0" ]] && CMD+=(--trust_remote_code)
  if [[ "${MATH_DAPO_BINARY_REWARD}" != "0" ]]; then
    CMD+=(--math_dapo_binary_reward)
  else
    CMD+=(--no-math_dapo_binary_reward)
  fi

  printf 'Running trajectory baseline eval with %d critic(s) and %d value position(s)\n' \
    "${#CRITIC_CHECKPOINT_DIRS_ARR[@]}" "${#VALUE_POSITIONS_ARR[@]}"
  printf ' %q' "${CMD[@]}"
  printf '\n'
  (cd "${REPO_DIR}" && "${CMD[@]}") 2>&1 | tee "${log_path}"
}

run_eval

echo "Trajectory value baseline eval finished. Outputs: ${OUTPUT_DIR}"
