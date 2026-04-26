#!/usr/bin/env bash

set -eo pipefail

# =============================================================================
# DELAYED-ONSET TOKEN-LEVEL VALUE GUIDANCE
# Frozen actor + frozen critic. One shared reference rollout per prompt/seed,
# then matched actor-only and delayed value-guided continuations from multiple
# prefix start positions.
# =============================================================================

# --- Checkpoints --------------------------------------------------------------
ACTOR_CHECKPOINT_DIR="/data/shuozhe/verl/train_log/job_05b_vh_init_e5_metamath/global_step_800"
CRITIC_CHECKPOINT_DIR="/data/shuozhe/verl/train_log/job_policy_gs800_dsk_1d5b_critic/global_step_750"
ACTOR_HF_SOURCE_DIR=""
CRITIC_HF_SOURCE_DIR=""

# --- Data ---------------------------------------------------------------------
DATASET_PATH="/data/shuozhe/saved_dataset/MetaMathQA-math-500/test.parquet"
OUTPUT_DIR="/data/shuozhe/verl/value_decoding/output/delayed_value_guidance_eval_ds_1d5_critic"
PROMPT_KEY="prompt"
RESPONSE_KEY=""  # Leave empty if unused.
START_INDEX=0
MAX_EXAMPLES="500"
SHUFFLE_EXAMPLES=0

# --- Generation ---------------------------------------------------------------
MAX_PROMPT_LENGTH=2048
MAX_NEW_TOKENS=2048
DTYPE="bf16"

# --- Multi-GPU Layout ---------------------------------------------------------
# Example 4-GPU layout:
#   worker 0: actor=cuda:0, critic=cuda:1
#   worker 1: actor=cuda:2, critic=cuda:3
DEVICE=""
ACTOR_DEVICE=""
CRITIC_DEVICE=""
WORKER_PAIRS="cuda:1,cuda:0 cuda:3,cuda:2"

# --- Delayed Value Guidance ---------------------------------------------------
SEEDS="42"
START_FRACTIONS="75"
CANDIDATE_SIZE="8"

ACTOR_SAMPLING_MODE="sample"
ACTOR_TEMPERATURE=1.0
ACTOR_TOP_P=1.0
ACTOR_TOP_K=0

# --- Misc ---------------------------------------------------------------------
SKIP_MERGE=0
DISABLE_ACTOR_CACHE=0
SKIP_PLOTS=0
TRUST_REMOTE_CODE=0

# Leave empty for ordinary single-node execution.
RAY_ADDRESS=""
RAY_NUM_CPUS_PER_WORKER="1"

source /data/shuozhe/miniconda3/bin/activate verl
set -u

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

read -r -a WORKER_PAIRS_ARR <<< "${WORKER_PAIRS}"
read -r -a SEED_ARR <<< "${SEEDS}"
read -r -a START_FRACTIONS_ARR <<< "${START_FRACTIONS}"

validate_component_checkpoint "${ACTOR_CHECKPOINT_DIR}" actor
validate_component_checkpoint "${CRITIC_CHECKPOINT_DIR}" critic

mkdir -p "${OUTPUT_DIR}"

if [[ ${#SEED_ARR[@]} -eq 0 ]]; then
  echo "SEEDS must contain at least one value." >&2
  exit 1
fi

CMD=(
  python -m value_decoding.delayed_value_guidance_eval
  --actor_checkpoint_dir "${ACTOR_CHECKPOINT_DIR}"
  --critic_checkpoint_dir "${CRITIC_CHECKPOINT_DIR}"
  --dataset_path "${DATASET_PATH}"
  --output_dir "${OUTPUT_DIR}"
  --prompt_key "${PROMPT_KEY}"
  --start_index "${START_INDEX}"
  --max_prompt_length "${MAX_PROMPT_LENGTH}"
  --max_new_tokens "${MAX_NEW_TOKENS}"
  --dtype "${DTYPE}"
  --seeds "${SEED_ARR[@]}"
  --start_fractions "${START_FRACTIONS_ARR[@]}"
  --candidate_size "${CANDIDATE_SIZE}"
  --actor_sampling_mode "${ACTOR_SAMPLING_MODE}"
  --actor_temperature "${ACTOR_TEMPERATURE}"
  --actor_top_p "${ACTOR_TOP_P}"
  --actor_top_k "${ACTOR_TOP_K}"
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
[[ "${SKIP_PLOTS}" != "0" ]] && CMD+=(--skip_plots)
[[ "${TRUST_REMOTE_CODE}" != "0" ]] && CMD+=(--trust_remote_code)

(cd "${REPO_DIR}" && "${CMD[@]}")
