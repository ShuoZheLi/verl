#!/usr/bin/env bash

set -eo pipefail

# =============================================================================
# STAGE 2 BEST-OF-N INFERENCE EVALUATION
# Shared full-response bank, then prompt-level selection by multiple selectors.
# =============================================================================

# --- Checkpoints --------------------------------------------------------------
ACTOR_CHECKPOINT_DIR="/data/shuozhe/verl/train_log/job_05b_vh_init_e5_metamath/global_step_800"
OLD_CRITIC_CHECKPOINT_DIR="/data/shuozhe/verl/train_log/job_05b_vh_init_e5_metamath/global_step_800"
NEW_CRITIC_CHECKPOINT_DIR="/data/shuozhe/verl/train_log/job_policy_gs800_dsk_1d5b_critic/global_step_750"

# --- Data ---------------------------------------------------------------------
DATASET_PATH="/data/shuozhe/saved_dataset/MetaMathQA-math-500/test.parquet"
OUTPUT_DIR="/data/shuozhe/verl/value_decoding/output/stage2_best_of_n_inference_eval"
PROMPT_KEY="prompt"
RESPONSE_KEY=""            # Leave empty if unused
START_INDEX=0
MAX_EXAMPLES="500"
SHUFFLE_EXAMPLES=0

# --- Bank / N Sweep -----------------------------------------------------------
N_VALUES="8"
MAX_BANK_SIZE=""           # Leave empty to use max(N_VALUES)
MAX_PROMPT_LENGTH=2048
MAX_NEW_TOKENS=2048
CRITIC_SCORE_BATCH_SIZE=8  # Kept for interface compatibility; completed trajectories are scored exactly one by one.
BOOTSTRAP_SAMPLES=2000
DTYPE="bf16"

# --- Devices ------------------------------------------------------------------
# Recommended layout if you have 3 GPUs:
#   ACTOR_DEVICE=cuda:0
#   OLD_CRITIC_DEVICE=cuda:1
#   NEW_CRITIC_DEVICE=cuda:2
# Recommended prompt-sharded layout for this 4-GPU machine when actor sampling is the bottleneck:
#   WORKER_LAYOUTS="cuda:0,cuda:1,cuda:2 cuda:3,cuda:1,cuda:2"
DEVICE=""
ACTOR_DEVICE="cuda:0"
OLD_CRITIC_DEVICE="cuda:1"
NEW_CRITIC_DEVICE="cuda:2"
WORKER_LAYOUTS="cuda:0,cuda:1,cuda:2 cuda:3,cuda:1,cuda:2"

# --- Actor Sampling -----------------------------------------------------------
ACTOR_SAMPLING_MODE="sample"
ACTOR_TEMPERATURE=1.0
ACTOR_TOP_P=1.0
ACTOR_TOP_K=0
SEED=42

# --- Validation / Plots -------------------------------------------------------
# Optional: strict overlap check against the Stage 1 bank for shared samples.
REFERENCE_STAGE1_TRAJECTORY_BANK=""
SKIP_PLOTS=0
PLOT_DPI=160

# --- Misc ---------------------------------------------------------------------
SKIP_MERGE=0
DISABLE_ACTOR_CACHE=0

source /data/shuozhe/miniconda3/bin/activate verl
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

read -r -a N_VALUES_ARR <<< "${N_VALUES}"
read -r -a WORKER_LAYOUTS_ARR <<< "${WORKER_LAYOUTS}"

mkdir -p "${OUTPUT_DIR}"

CMD=(
  python -m value_decoding.best_of_n_inference_eval
  --actor_checkpoint_dir      "${ACTOR_CHECKPOINT_DIR}"
  --old_critic_checkpoint_dir "${OLD_CRITIC_CHECKPOINT_DIR}"
  --new_critic_checkpoint_dir "${NEW_CRITIC_CHECKPOINT_DIR}"
  --dataset_path              "${DATASET_PATH}"
  --output_dir                "${OUTPUT_DIR}"
  --prompt_key                "${PROMPT_KEY}"
  --start_index               "${START_INDEX}"
  --max_prompt_length         "${MAX_PROMPT_LENGTH}"
  --max_new_tokens            "${MAX_NEW_TOKENS}"
  --critic_score_batch_size   "${CRITIC_SCORE_BATCH_SIZE}"
  --bootstrap_samples         "${BOOTSTRAP_SAMPLES}"
  --dtype                     "${DTYPE}"
  --seed                      "${SEED}"
  --actor_sampling_mode       "${ACTOR_SAMPLING_MODE}"
  --actor_temperature         "${ACTOR_TEMPERATURE}"
  --actor_top_p               "${ACTOR_TOP_P}"
  --actor_top_k               "${ACTOR_TOP_K}"
  --n_values                  "${N_VALUES_ARR[@]}"
  --plot_dpi                  "${PLOT_DPI}"
)

[[ -n "${RESPONSE_KEY}" ]] && CMD+=(--response_key "${RESPONSE_KEY}")
[[ -n "${MAX_EXAMPLES}" ]] && CMD+=(--max_examples "${MAX_EXAMPLES}")
[[ -n "${MAX_BANK_SIZE}" ]] && CMD+=(--max_bank_size "${MAX_BANK_SIZE}")
[[ -n "${DEVICE}" ]] && CMD+=(--device "${DEVICE}")
[[ -n "${ACTOR_DEVICE}" ]] && CMD+=(--actor_device "${ACTOR_DEVICE}")
[[ -n "${OLD_CRITIC_DEVICE}" ]] && CMD+=(--old_critic_device "${OLD_CRITIC_DEVICE}")
[[ -n "${NEW_CRITIC_DEVICE}" ]] && CMD+=(--new_critic_device "${NEW_CRITIC_DEVICE}")
[[ ${#WORKER_LAYOUTS_ARR[@]} -gt 0 ]] && CMD+=(--worker_layouts "${WORKER_LAYOUTS_ARR[@]}")
[[ -n "${REFERENCE_STAGE1_TRAJECTORY_BANK}" ]] && CMD+=(--reference_stage1_trajectory_bank "${REFERENCE_STAGE1_TRAJECTORY_BANK}")
[[ "${SHUFFLE_EXAMPLES}" != "0" ]] && CMD+=(--shuffle_examples)
[[ "${SKIP_MERGE}" != "0" ]] && CMD+=(--skip_merge)
[[ "${DISABLE_ACTOR_CACHE}" != "0" ]] && CMD+=(--disable_actor_cache)
[[ "${SKIP_PLOTS}" != "0" ]] && CMD+=(--skip_plots)

(cd "${REPO_DIR}" && "${CMD[@]}")
