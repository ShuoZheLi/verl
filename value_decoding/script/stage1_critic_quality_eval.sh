#!/usr/bin/env bash

set -eo pipefail

# =============================================================================
# STAGE 1 CRITIC-QUALITY COMPARISON
# Shared response bank evaluation only:
# - frozen actor samples N full responses once per prompt
# - old critic and new critic score the exact same completed trajectories
# - no training
# - no online critic-guided generation
# =============================================================================

# --- Checkpoints --------------------------------------------------------------
ACTOR_CHECKPOINT_DIR="/data/shuozhe/verl/train_log/job_05b_vh_init_e5_metamath/global_step_800"
OLD_CRITIC_CHECKPOINT_DIR="/data/shuozhe/verl/train_log/job_05b_vh_init_e5_metamath/global_step_800"
NEW_CRITIC_CHECKPOINT_DIR="/data/shuozhe/verl/train_log/job_policy_gs800_dsk_1d5b_critic/global_step_750"

# --- Data ---------------------------------------------------------------------
DATASET_PATH="/data/shuozhe/saved_dataset/MetaMathQA-math-500/test.parquet"
OUTPUT_DIR="/data/shuozhe/verl/value_decoding/output/stage1_critic_quality_eval"
PROMPT_KEY="prompt"
RESPONSE_KEY=""           # Leave empty if unused
START_INDEX=0
MAX_EXAMPLES="500"        # Recommended first pass: 500. Leave empty to use all.
SHUFFLE_EXAMPLES=0

# --- Shared Response Bank -----------------------------------------------------
NUM_SAMPLES_PER_PROMPT=8  # Recommended first pass: 8. Stronger pass: 16.
MAX_PROMPT_LENGTH=2048
MAX_NEW_TOKENS=2048
DTYPE="bf16"

# --- Devices ------------------------------------------------------------------
# Recommended layout if you have 3 GPUs:
#   ACTOR_DEVICE=cuda:0
#   OLD_CRITIC_DEVICE=cuda:1
#   NEW_CRITIC_DEVICE=cuda:2
DEVICE=""
ACTOR_DEVICE="cuda:0"
OLD_CRITIC_DEVICE="cuda:1"
NEW_CRITIC_DEVICE="cuda:2"

# --- Actor Sampling -----------------------------------------------------------
ACTOR_SAMPLING_MODE="sample"  # Options: greedy, sample
ACTOR_TEMPERATURE=1.0
ACTOR_TOP_P=1.0
ACTOR_TOP_K=0

# --- Analysis -----------------------------------------------------------------
CRITIC_SCORE_BATCH_SIZE=8
BOOTSTRAP_SAMPLES=2000
CALIBRATION_BINS=10
SEED=42

# --- Misc ---------------------------------------------------------------------
SKIP_MERGE=0
DISABLE_ACTOR_CACHE=0

source /data/shuozhe/miniconda3/bin/activate verl
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

mkdir -p "${OUTPUT_DIR}"

CMD=(
  python -m value_decoding.critic_quality_eval
  --actor_checkpoint_dir      "${ACTOR_CHECKPOINT_DIR}"
  --old_critic_checkpoint_dir "${OLD_CRITIC_CHECKPOINT_DIR}"
  --new_critic_checkpoint_dir "${NEW_CRITIC_CHECKPOINT_DIR}"
  --dataset_path              "${DATASET_PATH}"
  --output_dir                "${OUTPUT_DIR}"
  --prompt_key                "${PROMPT_KEY}"
  --start_index               "${START_INDEX}"
  --max_prompt_length         "${MAX_PROMPT_LENGTH}"
  --max_new_tokens            "${MAX_NEW_TOKENS}"
  --num_samples_per_prompt    "${NUM_SAMPLES_PER_PROMPT}"
  --critic_score_batch_size   "${CRITIC_SCORE_BATCH_SIZE}"
  --bootstrap_samples         "${BOOTSTRAP_SAMPLES}"
  --calibration_bins          "${CALIBRATION_BINS}"
  --dtype                     "${DTYPE}"
  --seed                      "${SEED}"
  --actor_sampling_mode       "${ACTOR_SAMPLING_MODE}"
  --actor_temperature         "${ACTOR_TEMPERATURE}"
  --actor_top_p               "${ACTOR_TOP_P}"
  --actor_top_k               "${ACTOR_TOP_K}"
)

[[ -n "${RESPONSE_KEY}"      ]] && CMD+=(--response_key      "${RESPONSE_KEY}")
[[ -n "${MAX_EXAMPLES}"      ]] && CMD+=(--max_examples      "${MAX_EXAMPLES}")
[[ -n "${DEVICE}"            ]] && CMD+=(--device            "${DEVICE}")
[[ -n "${ACTOR_DEVICE}"      ]] && CMD+=(--actor_device      "${ACTOR_DEVICE}")
[[ -n "${OLD_CRITIC_DEVICE}" ]] && CMD+=(--old_critic_device "${OLD_CRITIC_DEVICE}")
[[ -n "${NEW_CRITIC_DEVICE}" ]] && CMD+=(--new_critic_device "${NEW_CRITIC_DEVICE}")
[[ "${SHUFFLE_EXAMPLES}"  != "0" ]] && CMD+=(--shuffle_examples)
[[ "${SKIP_MERGE}"        != "0" ]] && CMD+=(--skip_merge)
[[ "${DISABLE_ACTOR_CACHE}" != "0" ]] && CMD+=(--disable_actor_cache)

(cd "${REPO_DIR}" && "${CMD[@]}")
