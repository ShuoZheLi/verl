#!/usr/bin/env bash

set -eo pipefail

# =============================================================================
# ENTROPY-BOUNDARY VALUE-GUIDED CHUNK DECODING INSPECTION
# =============================================================================

# --- Checkpoint / data --------------------------------------------------------
CHECKPOINT_DIR="/data/shuozhe/verl/train_log/job_05b_vh_init_e5_metamath/global_step_800"
# CHECKPOINT_DIR="/data/shuozhe/saved_model/DeepSeek-R1-Distill-Qwen-1.5B"
CRITIC_CHECKPOINT_DIR=""    # Leave empty for actor-only HF models or VERL checkpoint_dir/critic.
DATASET_PATH="/data/shuozhe/saved_dataset/MetaMathQA-math-500/test.parquet"
OUTPUT_DIR="/data/shuozhe/verl/step_boundary_detect/output_2"
PROMPT_KEY="prompt"
RESPONSE_KEY=""            # Leave empty if unused.
START_INDEX=270
MAX_SCAN_EXAMPLES=128
NUM_CORRECT=1

# --- Generation ---------------------------------------------------------------
MAX_PROMPT_LENGTH=2048
MAX_NEW_TOKENS=2048
DTYPE="bf16"

# --- Entropy boundary hyper-parameters ---------------------------------------
NUM_CANDIDATES=1           # Set to 4/8 with sample mode for K-way value-guided chunk selection.
MIN_CHUNK_LENGTH=4
MAX_CHUNK_LENGTH=1024
ENTROPY_PERCENTILE=0.97
WARMUP_SIZE=32
BOOTSTRAP_CURRENT_CHUNK_ENTROPY=1

# --- Actor sampling -----------------------------------------------------------
ACTOR_SAMPLING_MODE="greedy"   # "sample" greedy is useful when NUM_CANDIDATES > 1.
ACTOR_TEMPERATURE=1.0
ACTOR_TOP_P=1.0
ACTOR_TOP_K=0
SELECTION_MODE="auto"      # auto: critic value if available, actor_logprob otherwise.

# --- Multi-GPU worker layout --------------------------------------------------
# One entry launches one independent worker over a disjoint dataset shard.
# Use "cuda:N" to place actor+critic on the same GPU.
WORKER_PAIRS="cuda:0 cuda:1 cuda:2 cuda:3"

# If actor+critic do not fit together on one GPU, use paired devices instead.
# This uses four GPUs as two workers:
# WORKER_PAIRS="cuda:0,cuda:1 cuda:2,cuda:3"
#
# On 8 GPUs, either use 8 one-GPU workers:
# WORKER_PAIRS="cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7"
# or 4 split actor/critic workers:
# WORKER_PAIRS="cuda:0,cuda:1 cuda:2,cuda:3 cuda:4,cuda:5 cuda:6,cuda:7"

# Used only when WORKER_PAIRS is empty.
DEVICE=""
ACTOR_DEVICE=""
CRITIC_DEVICE=""

# --- Misc ---------------------------------------------------------------------
SEED=42
SKIP_MERGE=1              # Use 1 for plain HF dirs or checkpoints that already have merged_hf.
DISABLE_CRITIC_MODEL=0    # Set 1 to force actor-only selection.
DISABLE_ACTOR_CACHE=0
TRUST_REMOTE_CODE=0

source /data/shuozhe/miniconda3/etc/profile.d/conda.sh
conda activate verl
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

read -r -a WORKER_PAIRS_ARR <<< "${WORKER_PAIRS}"

mkdir -p "${OUTPUT_DIR}"

CMD=(
  python step_boundary_detect/detect.py
  --checkpoint_dir       "${CHECKPOINT_DIR}"
  --dataset_path         "${DATASET_PATH}"
  --output_dir           "${OUTPUT_DIR}"
  --prompt_key           "${PROMPT_KEY}"
  --start_index          "${START_INDEX}"
  --max_scan_examples    "${MAX_SCAN_EXAMPLES}"
  --num_correct          "${NUM_CORRECT}"
  --max_prompt_length    "${MAX_PROMPT_LENGTH}"
  --max_new_tokens       "${MAX_NEW_TOKENS}"
  --num_candidates       "${NUM_CANDIDATES}"
  --min_chunk_length     "${MIN_CHUNK_LENGTH}"
  --max_chunk_length     "${MAX_CHUNK_LENGTH}"
  --entropy_percentile   "${ENTROPY_PERCENTILE}"
  --warmup_size          "${WARMUP_SIZE}"
  --dtype                "${DTYPE}"
  --actor_sampling_mode  "${ACTOR_SAMPLING_MODE}"
  --actor_temperature    "${ACTOR_TEMPERATURE}"
  --actor_top_p          "${ACTOR_TOP_P}"
  --actor_top_k          "${ACTOR_TOP_K}"
  --selection_mode       "${SELECTION_MODE}"
  --seed                 "${SEED}"
)

[[ -n "${CRITIC_CHECKPOINT_DIR}" ]] && CMD+=(--critic_checkpoint_dir "${CRITIC_CHECKPOINT_DIR}")
[[ -n "${RESPONSE_KEY}" ]] && CMD+=(--response_key "${RESPONSE_KEY}")
[[ -n "${DEVICE}" ]] && CMD+=(--device "${DEVICE}")
[[ -n "${ACTOR_DEVICE}" ]] && CMD+=(--actor_device "${ACTOR_DEVICE}")
[[ -n "${CRITIC_DEVICE}" ]] && CMD+=(--critic_device "${CRITIC_DEVICE}")
[[ ${#WORKER_PAIRS_ARR[@]} -gt 0 ]] && CMD+=(--worker_pairs "${WORKER_PAIRS_ARR[@]}")
[[ "${BOOTSTRAP_CURRENT_CHUNK_ENTROPY}" != "0" ]] && CMD+=(--bootstrap_current_chunk_entropy)
[[ "${SKIP_MERGE}" != "0" ]] && CMD+=(--skip_merge)
[[ "${DISABLE_CRITIC_MODEL}" != "0" ]] && CMD+=(--disable_critic_model)
[[ "${DISABLE_ACTOR_CACHE}" != "0" ]] && CMD+=(--disable_actor_cache)
[[ "${TRUST_REMOTE_CODE}" != "0" ]] && CMD+=(--trust_remote_code)

cd "${REPO_DIR}"
PYTHONUNBUFFERED=1 "${CMD[@]}"
