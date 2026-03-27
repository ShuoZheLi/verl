#!/usr/bin/env bash

set -eo pipefail

# =============================================================================
# VALUE DECODING EVALUATION SCRIPT
# Edit the parameters below to configure your run.
# =============================================================================

# --- Paths -------------------------------------------------------------------
CHECKPOINT_DIR="/data/shuozhe/verl/train_log/job_05b_vh_init_e5_metamath/global_step_800"
DATASET_PATH="/data/shuozhe/saved_dataset/MetaMathQA-math-500/test.parquet"
OUTPUT_DIR="/data/shuozhe/verl/value_decoding/output/actor_critic_rerank_beta_1d5_2d0"

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
# DEVICE is a fallback used when ACTOR_DEVICE / CRITIC_DEVICE are not set.
# WORKER_PAIRS lets you run multiple actor+critic pairs in parallel,
# formatted as "actor_device,critic_device" entries, e.g. "cuda:0,cuda:1 cuda:2,cuda:3"
# Leave WORKER_PAIRS empty to disable parallel workers.
DEVICE=""
ACTOR_DEVICE=""
CRITIC_DEVICE=""
WORKER_PAIRS="cuda:0,cuda:1 cuda:2,cuda:3"          # e.g. "cuda:0,cuda:1 cuda:2,cuda:3"

# --- Actor Sampling ----------------------------------------------------------
ACTOR_SAMPLING_MODE="sample"  # Options: greedy, sample
ACTOR_TEMPERATURE=1.0
ACTOR_TOP_P=1.0
ACTOR_TOP_K=0

# --- Evaluation Modes --------------------------------------------------------
# Space-separated list. Available options:
#   actor_only  critic_only_rerank  actor_critic_rerank  actor_critic_soft_rerank
MODES="actor_critic_rerank"

# --- Candidate Configuration -------------------------------------------------
CANDIDATE_BUILDERS="top_k"   # Space-separated, e.g. "top_k sampled"
CANDIDATE_SIZES="8"          # Space-separated ints, e.g. "4 8 16"
BETAS="1.5 2.0"                  # Space-separated floats, e.g. "0.5 1.0 2.0"
NORMALIZATIONS="zscore"      # Space-separated, e.g. "zscore minmax"
RANK_TEMPERATURES="0.5"      # Space-separated floats, e.g. "0.5 1.0"

# --- Misc --------------------------------------------------------------------
NORMALIZATION_EPS=1e-6
SEED=42
SKIP_MERGE=0             # 1 = skip merging checkpoints
DISABLE_ACTOR_CACHE=0    # 1 = disable actor KV cache
DEBUG_FULL_CANDIDATES=0  # 1 = log full candidate set for debugging

# --- Self-Check (quick sanity run before full eval) --------------------------
RUN_SELF_CHECK=1         # 1 = run self-check, 0 = skip
CHECK_TWO_GPU=0          # 1 = verify two-GPU setup during self-check

# =============================================================================
# DO NOT EDIT BELOW THIS LINE (unless you know what you're doing)
# =============================================================================

source /data/shuozhe/miniconda3/bin/activate verl
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

read -r -a MODES_ARR              <<< "${MODES}"
read -r -a CANDIDATE_BUILDERS_ARR <<< "${CANDIDATE_BUILDERS}"
read -r -a CANDIDATE_SIZES_ARR    <<< "${CANDIDATE_SIZES}"
read -r -a BETAS_ARR              <<< "${BETAS}"
read -r -a NORMALIZATIONS_ARR     <<< "${NORMALIZATIONS}"
read -r -a RANK_TEMPERATURES_ARR  <<< "${RANK_TEMPERATURES}"
read -r -a WORKER_PAIRS_ARR       <<< "${WORKER_PAIRS}"

mkdir -p "${OUTPUT_DIR}"

# Merge HF weights if needed (required by both self-check and the main run).
if [[ "${SKIP_MERGE}" == "0" ]]; then
  (
    cd "${REPO_DIR}"
    CHECKPOINT_DIR="${CHECKPOINT_DIR}" python -c "
import os
from pathlib import Path
from value_decoding.checkpointing import ensure_merged_checkpoints
ensure_merged_checkpoints(Path(os.environ['CHECKPOINT_DIR']), skip_merge=False)
"
  )
fi

# --- Self-check --------------------------------------------------------------
if [[ "${RUN_SELF_CHECK}" != "0" ]]; then
  # Device priority: explicit ACTOR/CRITIC_DEVICE > first WORKER_PAIRS entry > DEVICE > unset
  SC_ACTOR_DEVICE="${ACTOR_DEVICE}"
  SC_CRITIC_DEVICE="${CRITIC_DEVICE}"
  if [[ -z "${SC_ACTOR_DEVICE}" && ${#WORKER_PAIRS_ARR[@]} -gt 0 ]]; then
    FIRST_PAIR="${WORKER_PAIRS_ARR[0]}"
    SC_ACTOR_DEVICE="${FIRST_PAIR%%,*}"
    SC_CRITIC_DEVICE="${FIRST_PAIR#*,}"  # equals actor device if no comma in pair
  fi
  [[ -z "${SC_ACTOR_DEVICE}"  && -n "${DEVICE}" ]] && SC_ACTOR_DEVICE="${DEVICE}"
  [[ -z "${SC_CRITIC_DEVICE}" && -n "${DEVICE}" ]] && SC_CRITIC_DEVICE="${DEVICE}"

  SELF_CHECK_CMD=(
    python -m value_decoding.self_check
    --actor_dir         "${CHECKPOINT_DIR}/merged_hf/actor"
    --critic_dir        "${CHECKPOINT_DIR}/merged_hf/critic"
    --dataset_path      "${DATASET_PATH}"
    --dtype             "${DTYPE}"
    --max_examples      1
    --max_new_tokens    8
    --max_prompt_length "${MAX_PROMPT_LENGTH}"
    --seed              "${SEED}"
  )
  [[ -n "${SC_ACTOR_DEVICE}"   ]] && SELF_CHECK_CMD+=(--actor_device  "${SC_ACTOR_DEVICE}")
  [[ -n "${SC_CRITIC_DEVICE}"  ]] && SELF_CHECK_CMD+=(--critic_device "${SC_CRITIC_DEVICE}")
  [[ "${CHECK_TWO_GPU}" != "0" ]] && SELF_CHECK_CMD+=(--check_two_gpu)

  (cd "${REPO_DIR}" && "${SELF_CHECK_CMD[@]}")
fi

# --- Main command ------------------------------------------------------------
CMD=(
  python -m value_decoding
  --checkpoint_dir      "${CHECKPOINT_DIR}"
  --dataset_path        "${DATASET_PATH}"
  --output_dir          "${OUTPUT_DIR}"
  --prompt_key          "${PROMPT_KEY}"
  --start_index         "${START_INDEX}"
  --max_prompt_length   "${MAX_PROMPT_LENGTH}"
  --max_new_tokens      "${MAX_NEW_TOKENS}"
  --dtype               "${DTYPE}"
  --normalization_eps   "${NORMALIZATION_EPS}"
  --seed                "${SEED}"
  --actor_sampling_mode "${ACTOR_SAMPLING_MODE}"
  --actor_temperature   "${ACTOR_TEMPERATURE}"
  --actor_top_p         "${ACTOR_TOP_P}"
  --actor_top_k         "${ACTOR_TOP_K}"
  --modes               "${MODES_ARR[@]}"
  --candidate_builders  "${CANDIDATE_BUILDERS_ARR[@]}"
  --candidate_sizes     "${CANDIDATE_SIZES_ARR[@]}"
  --betas               "${BETAS_ARR[@]}"
  --normalizations      "${NORMALIZATIONS_ARR[@]}"
  --rank_temperatures   "${RANK_TEMPERATURES_ARR[@]}"
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
[[ "${DEBUG_FULL_CANDIDATES}" != "0" ]] && CMD+=(--debug_full_candidates)

(cd "${REPO_DIR}" && "${CMD[@]}")