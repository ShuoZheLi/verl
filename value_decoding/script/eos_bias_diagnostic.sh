#!/usr/bin/env bash

set -eo pipefail

# =============================================================================
# EOS BIAS DIAGNOSTIC
# Analysis-only experiment. No training or checkpoint modification.
# =============================================================================

# --- Paths -------------------------------------------------------------------
CHECKPOINT_DIR="/data/shuozhe/verl/train_log/job_05b_vh_init_e5_metamath/global_step_800"
DATASET_PATH="/data/shuozhe/saved_dataset/MetaMathQA-math-500/test.parquet"
OUTPUT_DIR="/data/shuozhe/verl/value_decoding/output/eos_bias_diag"

# --- Data Keys ---------------------------------------------------------------
PROMPT_KEY="prompt"
RESPONSE_KEY=""          # Leave empty if unused

# --- Data Selection ----------------------------------------------------------
START_INDEX=0
MAX_EXAMPLES=""          # Leave empty to use all examples
SHUFFLE_EXAMPLES=0       # 1 = shuffle, 0 = no shuffle

# --- Generation Settings -----------------------------------------------------
MAX_PROMPT_LENGTH=2048
MAX_NEW_TOKENS=2048
DTYPE="bf16"             # Options: bf16, fp16, fp32

# --- Device Assignment -------------------------------------------------------
DEVICE=""
ACTOR_DEVICE=""
CRITIC_DEVICE=""
WORKER_PAIRS="cuda:0,cuda:1 cuda:2,cuda:3"

# --- Actor Rollout Settings --------------------------------------------------
ACTOR_SAMPLING_MODE="sample"  # Options: greedy, sample
ACTOR_TEMPERATURE=1.0
ACTOR_TOP_P=1.0
ACTOR_TOP_K=0

# --- Misc --------------------------------------------------------------------
SEED=42
SKIP_MERGE=0             # 1 = skip merging checkpoints
DISABLE_ACTOR_CACHE=0    # 1 = disable actor KV cache
SKIP_FINAL_EOS_CHECK=0   # 1 = skip the optional append-final-EOS check

# =============================================================================
# DO NOT EDIT BELOW THIS LINE (unless you know what you're doing)
# =============================================================================

source /data/shuozhe/miniconda3/bin/activate verl
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

read -r -a WORKER_PAIRS_ARR <<< "${WORKER_PAIRS}"

mkdir -p "${OUTPUT_DIR}"

CMD=(
  python -m value_decoding.eos_bias_diagnostic
  --checkpoint_dir      "${CHECKPOINT_DIR}"
  --dataset_path        "${DATASET_PATH}"
  --output_dir          "${OUTPUT_DIR}"
  --prompt_key          "${PROMPT_KEY}"
  --start_index         "${START_INDEX}"
  --max_prompt_length   "${MAX_PROMPT_LENGTH}"
  --max_new_tokens      "${MAX_NEW_TOKENS}"
  --dtype               "${DTYPE}"
  --seed                "${SEED}"
  --actor_sampling_mode "${ACTOR_SAMPLING_MODE}"
  --actor_temperature   "${ACTOR_TEMPERATURE}"
  --actor_top_p         "${ACTOR_TOP_P}"
  --actor_top_k         "${ACTOR_TOP_K}"
)

[[ -n "${RESPONSE_KEY}"          ]] && CMD+=(--response_key          "${RESPONSE_KEY}")
[[ -n "${MAX_EXAMPLES}"          ]] && CMD+=(--max_examples          "${MAX_EXAMPLES}")
[[ -n "${DEVICE}"                ]] && CMD+=(--device                "${DEVICE}")
[[ -n "${ACTOR_DEVICE}"          ]] && CMD+=(--actor_device          "${ACTOR_DEVICE}")
[[ -n "${CRITIC_DEVICE}"         ]] && CMD+=(--critic_device         "${CRITIC_DEVICE}")
[[ ${#WORKER_PAIRS_ARR[@]} -gt 0 ]] && CMD+=(--worker_pairs         "${WORKER_PAIRS_ARR[@]}")
[[ "${SHUFFLE_EXAMPLES}"    != "0" ]] && CMD+=(--shuffle_examples)
[[ "${SKIP_MERGE}"          != "0" ]] && CMD+=(--skip_merge)
[[ "${DISABLE_ACTOR_CACHE}" != "0" ]] && CMD+=(--disable_actor_cache)
[[ "${SKIP_FINAL_EOS_CHECK}" != "0" ]] && CMD+=(--skip_final_eos_check)

(cd "${REPO_DIR}" && "${CMD[@]}")
