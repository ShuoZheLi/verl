#!/usr/bin/env bash

set -eo pipefail

# =============================================================================
# CHUNK-LEVEL GUIDANCE EXPERIMENT
# Frozen actor + new critic, chunk-level candidate competition.
# =============================================================================

# --- Checkpoints --------------------------------------------------------------
ACTOR_CHECKPOINT_DIR="/data/shuozhe/verl/train_log/job_05b_vh_init_e5_metamath/global_step_800"
CRITIC_CHECKPOINT_DIR="/data/shuozhe/verl/train_log/job_policy_gs800_dsk_1d5b_critic/global_step_750"
ACTOR_HF_SOURCE_DIR=""
CRITIC_HF_SOURCE_DIR=""

# --- Data ---------------------------------------------------------------------
DATASET_PATH="/data/shuozhe/saved_dataset/MetaMathQA-math-500/test.parquet"
OUTPUT_DIR="/data/shuozhe/verl/value_decoding/output/chunk_guidance_eval_512_1024_ds_1d5_critic"
PROMPT_KEY="prompt"
RESPONSE_KEY=""            # Leave empty if unused
START_INDEX=0
MAX_EXAMPLES="500"
SHUFFLE_EXAMPLES=0

# --- Generation ---------------------------------------------------------------
MAX_PROMPT_LENGTH=2048
MAX_NEW_TOKENS=2048
DTYPE="bf16"

# --- 8-GPU Prompt Sharding ----------------------------------------------------
# Recommended default on an 8-GPU machine:
#   worker 0: actor=cuda:0, critic=cuda:1
#   worker 1: actor=cuda:2, critic=cuda:3
#   worker 2: actor=cuda:4, critic=cuda:5
#   worker 3: actor=cuda:6, critic=cuda:7
DEVICE=""
ACTOR_DEVICE=""
CRITIC_DEVICE=""
# WORKER_PAIRS="cuda:0,cuda:1 cuda:2,cuda:3 cuda:4,cuda:5 cuda:6,cuda:7"
WORKER_PAIRS="cuda:0,cuda:1 cuda:2,cuda:3"

# --- Actor Sampling -----------------------------------------------------------
ACTOR_SAMPLING_MODE="sample"
ACTOR_TEMPERATURE=1.0
ACTOR_TOP_P=1.0
ACTOR_TOP_K=0

# --- Chunk Grid ---------------------------------------------------------------
CHUNK_SIZES="512 1024"
NUM_CHUNK_CANDIDATES_VALUES="2 4 8"
BETAS="0"
VALUE_REDUCERS="end"       # Add "mean" for the ablation.
INCLUDE_CRITIC_ONLY=1      # 1 = add optional critic-only chunk rerank configs
INCLUDE_UNCERTAINTY_ONLY=0 # 1 = add actor-uncertainty chunk rerank configs
ONLY_CRITIC_ONLY=1         # 1 = run only critic-only chunk rerank configs

# --- Misc ---------------------------------------------------------------------
NORMALIZATION_EPS=1e-6
SEED="42"
SKIP_MERGE=0
DISABLE_ACTOR_CACHE=0
DEBUG_FULL_CHUNK_CANDIDATES=0

# Leave empty for ordinary single-node execution.
# Use the dedicated multi-node launcher for Ray-based runs so a stale shell
# RAY_ADDRESS does not silently switch this script into cross-node mode.
RAY_ADDRESS=""
RAY_NUM_CPUS_PER_WORKER="1"

source /data/shuozhe/miniconda3/bin/activate verl
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"

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
)

checkpoint_dir = Path(sys.argv[1])
component = sys.argv[2]
component_dir = checkpoint_dir / component

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

read -r -a CHUNK_SIZES_ARR <<< "${CHUNK_SIZES}"
read -r -a NUM_CHUNK_CANDIDATES_VALUES_ARR <<< "${NUM_CHUNK_CANDIDATES_VALUES}"
read -r -a BETAS_ARR <<< "${BETAS}"
read -r -a VALUE_REDUCERS_ARR <<< "${VALUE_REDUCERS}"
read -r -a WORKER_PAIRS_ARR <<< "${WORKER_PAIRS}"
read -r -a SEED_ARR <<< "${SEED}"

validate_component_checkpoint "${ACTOR_CHECKPOINT_DIR}" actor
validate_component_checkpoint "${CRITIC_CHECKPOINT_DIR}" critic

mkdir -p "${OUTPUT_DIR}"

if [[ ${#SEED_ARR[@]} -eq 0 ]]; then
  echo "SEED must contain at least one value." >&2
  exit 1
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
    python -m value_decoding.chunk_guidance_eval
    --actor_checkpoint_dir       "${ACTOR_CHECKPOINT_DIR}"
    --critic_checkpoint_dir      "${CRITIC_CHECKPOINT_DIR}"
    --dataset_path               "${DATASET_PATH}"
    --output_dir                 "${output_dir_for_seed}"
    --prompt_key                 "${PROMPT_KEY}"
    --start_index                "${START_INDEX}"
    --max_prompt_length          "${MAX_PROMPT_LENGTH}"
    --max_new_tokens             "${MAX_NEW_TOKENS}"
    --dtype                      "${DTYPE}"
    --normalization_eps          "${NORMALIZATION_EPS}"
    --seed                       "${seed_value}"
    --actor_sampling_mode        "${ACTOR_SAMPLING_MODE}"
    --actor_temperature          "${ACTOR_TEMPERATURE}"
    --actor_top_p                "${ACTOR_TOP_P}"
    --actor_top_k                "${ACTOR_TOP_K}"
    --chunk_sizes                "${CHUNK_SIZES_ARR[@]}"
    --num_chunk_candidates_values "${NUM_CHUNK_CANDIDATES_VALUES_ARR[@]}"
    --betas                      "${BETAS_ARR[@]}"
    --value_reducers             "${VALUE_REDUCERS_ARR[@]}"
  )

  [[ -n "${RESPONSE_KEY}" ]] && CMD+=(--response_key "${RESPONSE_KEY}")
  [[ -n "${MAX_EXAMPLES}" ]] && CMD+=(--max_examples "${MAX_EXAMPLES}")
  [[ -n "${DEVICE}" ]] && CMD+=(--device "${DEVICE}")
  [[ -n "${ACTOR_DEVICE}" ]] && CMD+=(--actor_device "${ACTOR_DEVICE}")
  [[ -n "${CRITIC_DEVICE}" ]] && CMD+=(--critic_device "${CRITIC_DEVICE}")
  [[ -n "${ACTOR_HF_SOURCE_DIR}" ]] && CMD+=(--actor_hf_source_dir "${ACTOR_HF_SOURCE_DIR}")
  [[ -n "${CRITIC_HF_SOURCE_DIR}" ]] && CMD+=(--critic_hf_source_dir "${CRITIC_HF_SOURCE_DIR}")
  [[ ${#WORKER_PAIRS_ARR[@]} -gt 0 ]] && CMD+=(--worker_pairs "${WORKER_PAIRS_ARR[@]}")
  [[ -n "${RAY_ADDRESS}" ]] && CMD+=(--ray_address "${RAY_ADDRESS}" --ray_num_cpus_per_worker "${RAY_NUM_CPUS_PER_WORKER}")
  [[ "${SHUFFLE_EXAMPLES}" != "0" ]] && CMD+=(--shuffle_examples)
  [[ "${SKIP_MERGE}" != "0" ]] && CMD+=(--skip_merge)
  [[ "${DISABLE_ACTOR_CACHE}" != "0" ]] && CMD+=(--disable_actor_cache)
  [[ "${INCLUDE_CRITIC_ONLY}" != "0" ]] && CMD+=(--include_critic_only)
  [[ "${INCLUDE_UNCERTAINTY_ONLY}" != "0" ]] && CMD+=(--include_uncertainty_only)
  [[ "${ONLY_CRITIC_ONLY}" != "0" ]] && CMD+=(--only_critic_only)
  [[ "${DEBUG_FULL_CHUNK_CANDIDATES}" != "0" ]] && CMD+=(--debug_full_chunk_candidates)

  mkdir -p "${output_dir_for_seed}"
  (cd "${REPO_DIR}" && "${CMD[@]}")
}

if [[ ${#SEED_ARR[@]} -eq 1 ]]; then
  run_one_seed "${SEED_ARR[0]}" "${OUTPUT_DIR}"
  exit 0
fi

SUMMARY_PATHS=()
SEED_OUTPUT_DIRS=()
for seed_value in "${SEED_ARR[@]}"; do
  seed_id="$(seed_to_id "${seed_value}")"
  seed_output_dir="${OUTPUT_DIR}/seed_${seed_id}"
  run_one_seed "${seed_value}" "${seed_output_dir}"
  SUMMARY_PATHS+=("${seed_output_dir}/summary_metrics.json")
  SEED_OUTPUT_DIRS+=("${seed_output_dir}")
done

python -m value_decoding.multi_seed_summary \
  --output_path "${OUTPUT_DIR}/summary_metrics.json" \
  --source_script "${SCRIPT_PATH}" \
  --seed_values "${SEED_ARR[@]}" \
  --summary_paths "${SUMMARY_PATHS[@]}" \
  --seed_output_dirs "${SEED_OUTPUT_DIRS[@]}"
