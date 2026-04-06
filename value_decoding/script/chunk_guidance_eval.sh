#!/usr/bin/env bash

set -eo pipefail

# =============================================================================
# CHUNK-LEVEL GUIDANCE EXPERIMENT
# Frozen actor + new critic, chunk-level candidate competition.
# =============================================================================

# --- Checkpoints --------------------------------------------------------------
ACTOR_CHECKPOINT_DIR="/data/shuozhe/verl/train_log/job_05b_vh_init_e5_metamath/global_step_800"
CRITIC_CHECKPOINT_DIR="/data/shuozhe/verl/train_log/job_policy_gs800_dsk_1d5b_critic/global_step_750"

# --- Data ---------------------------------------------------------------------
DATASET_PATH="/data/shuozhe/saved_dataset/MetaMathQA-math-500/test.parquet"
OUTPUT_DIR="/data/shuozhe/verl/value_decoding/output/chunk_guidance_eval"
PROMPT_KEY="prompt"
RESPONSE_KEY=""            # Leave empty if unused
START_INDEX=0
MAX_EXAMPLES="500"
SHUFFLE_EXAMPLES=0

# --- Generation ---------------------------------------------------------------
MAX_PROMPT_LENGTH=2048
MAX_NEW_TOKENS=2048
DTYPE="bf16"

# --- 4-GPU Prompt Sharding ----------------------------------------------------
# Recommended default on a 4-GPU machine:
#   worker 0: actor=cuda:0, critic=cuda:1
#   worker 1: actor=cuda:2, critic=cuda:3
DEVICE=""
ACTOR_DEVICE=""
CRITIC_DEVICE=""
WORKER_PAIRS="cuda:0,cuda:1 cuda:2,cuda:3"

# --- Actor Sampling -----------------------------------------------------------
ACTOR_SAMPLING_MODE="sample"
ACTOR_TEMPERATURE=1.0
ACTOR_TOP_P=1.0
ACTOR_TOP_K=0

# --- Chunk Grid ---------------------------------------------------------------
CHUNK_SIZES="16"
NUM_CHUNK_CANDIDATES_VALUES="2"
BETAS="0 0.05 0.1 0.25"
VALUE_REDUCERS="end"       # Add "mean" for the ablation.
INCLUDE_CRITIC_ONLY=1      # 1 = add optional critic-only chunk rerank configs

# --- Misc ---------------------------------------------------------------------
NORMALIZATION_EPS=1e-6
SEED=42
SKIP_MERGE=0
DISABLE_ACTOR_CACHE=0
DEBUG_FULL_CHUNK_CANDIDATES=0

source /data/shuozhe/miniconda3/bin/activate verl
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

read -r -a CHUNK_SIZES_ARR <<< "${CHUNK_SIZES}"
read -r -a NUM_CHUNK_CANDIDATES_VALUES_ARR <<< "${NUM_CHUNK_CANDIDATES_VALUES}"
read -r -a BETAS_ARR <<< "${BETAS}"
read -r -a VALUE_REDUCERS_ARR <<< "${VALUE_REDUCERS}"
read -r -a WORKER_PAIRS_ARR <<< "${WORKER_PAIRS}"

mkdir -p "${OUTPUT_DIR}"

CMD=(
  python -m value_decoding.chunk_guidance_eval
  --actor_checkpoint_dir       "${ACTOR_CHECKPOINT_DIR}"
  --critic_checkpoint_dir      "${CRITIC_CHECKPOINT_DIR}"
  --dataset_path               "${DATASET_PATH}"
  --output_dir                 "${OUTPUT_DIR}"
  --prompt_key                 "${PROMPT_KEY}"
  --start_index                "${START_INDEX}"
  --max_prompt_length          "${MAX_PROMPT_LENGTH}"
  --max_new_tokens             "${MAX_NEW_TOKENS}"
  --dtype                      "${DTYPE}"
  --normalization_eps          "${NORMALIZATION_EPS}"
  --seed                       "${SEED}"
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
[[ ${#WORKER_PAIRS_ARR[@]} -gt 0 ]] && CMD+=(--worker_pairs "${WORKER_PAIRS_ARR[@]}")
[[ "${SHUFFLE_EXAMPLES}" != "0" ]] && CMD+=(--shuffle_examples)
[[ "${SKIP_MERGE}" != "0" ]] && CMD+=(--skip_merge)
[[ "${DISABLE_ACTOR_CACHE}" != "0" ]] && CMD+=(--disable_actor_cache)
[[ "${INCLUDE_CRITIC_ONLY}" != "0" ]] && CMD+=(--include_critic_only)
[[ "${DEBUG_FULL_CHUNK_CANDIDATES}" != "0" ]] && CMD+=(--debug_full_chunk_candidates)

(cd "${REPO_DIR}" && "${CMD[@]}")
