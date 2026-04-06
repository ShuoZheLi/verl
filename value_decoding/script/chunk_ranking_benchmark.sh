#!/usr/bin/env bash

set -eo pipefail

# =============================================================================
# CHUNK-RANKING BENCHMARK
# Shared prefix states, shared chunk banks, shared completions, old-vs-new
# critic comparison on the local chunk decision problem.
# =============================================================================

# --- Checkpoints --------------------------------------------------------------
ACTOR_CHECKPOINT_DIR="/data/shuozhe/verl/train_log/job_05b_vh_init_e5_metamath/global_step_800"
OLD_CRITIC_CHECKPOINT_DIR="/data/shuozhe/verl/train_log/job_05b_vh_init_e5_metamath/global_step_800"
NEW_CRITIC_CHECKPOINT_DIR="/data/shuozhe/verl/train_log/job_policy_gs800_dsk_1d5b_critic/global_step_750"

# --- Data ---------------------------------------------------------------------
DATASET_PATH="/data/shuozhe/saved_dataset/MetaMathQA-math-500/test.parquet"
OUTPUT_DIR="/data/shuozhe/verl/value_decoding/output/chunk_ranking_benchmark"
PROMPT_KEY="prompt"
RESPONSE_KEY=""            # Leave empty if unused
START_INDEX=0
MAX_EXAMPLES="500"
SHUFFLE_EXAMPLES=0

# --- Benchmark Config ---------------------------------------------------------
MAX_PROMPT_LENGTH=2048
MAX_NEW_TOKENS=2048
CHUNK_SIZE=32
NUM_CHUNK_CANDIDATES=8
BOOTSTRAP_SAMPLES=2000
DTYPE="bf16"

# --- Devices ------------------------------------------------------------------
# Recommended 4-GPU prompt-sharded layout for two critics, if GPU memory is
# sufficient for duplicated critic replicas across worker processes:
#   worker 0: actor=cuda:0, old=cuda:1, new=cuda:2
#   worker 1: actor=cuda:3, old=cuda:1, new=cuda:2
# This duplicates the actor across two prompt shards and also loads one critic
# replica per worker on the critic GPUs.
DEVICE=""
ACTOR_DEVICE=""
OLD_CRITIC_DEVICE=""
NEW_CRITIC_DEVICE=""
WORKER_LAYOUTS="cuda:0,cuda:1,cuda:2 cuda:3,cuda:1,cuda:2"

# --- Actor Sampling -----------------------------------------------------------
ACTOR_SAMPLING_MODE="sample"
ACTOR_TEMPERATURE=1.0
ACTOR_TOP_P=1.0
ACTOR_TOP_K=0
SEED=42

# --- Misc ---------------------------------------------------------------------
SKIP_MERGE=0
DISABLE_ACTOR_CACHE=0

source /data/shuozhe/miniconda3/bin/activate verl
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

read -r -a WORKER_LAYOUTS_ARR <<< "${WORKER_LAYOUTS}"

mkdir -p "${OUTPUT_DIR}"

CMD=(
  python -m value_decoding.chunk_ranking_benchmark
  --actor_checkpoint_dir      "${ACTOR_CHECKPOINT_DIR}"
  --old_critic_checkpoint_dir "${OLD_CRITIC_CHECKPOINT_DIR}"
  --new_critic_checkpoint_dir "${NEW_CRITIC_CHECKPOINT_DIR}"
  --dataset_path              "${DATASET_PATH}"
  --output_dir                "${OUTPUT_DIR}"
  --prompt_key                "${PROMPT_KEY}"
  --start_index               "${START_INDEX}"
  --max_prompt_length         "${MAX_PROMPT_LENGTH}"
  --max_new_tokens            "${MAX_NEW_TOKENS}"
  --chunk_size                "${CHUNK_SIZE}"
  --num_chunk_candidates      "${NUM_CHUNK_CANDIDATES}"
  --bootstrap_samples         "${BOOTSTRAP_SAMPLES}"
  --dtype                     "${DTYPE}"
  --seed                      "${SEED}"
  --actor_sampling_mode       "${ACTOR_SAMPLING_MODE}"
  --actor_temperature         "${ACTOR_TEMPERATURE}"
  --actor_top_p               "${ACTOR_TOP_P}"
  --actor_top_k               "${ACTOR_TOP_K}"
)

[[ -n "${RESPONSE_KEY}" ]] && CMD+=(--response_key "${RESPONSE_KEY}")
[[ -n "${MAX_EXAMPLES}" ]] && CMD+=(--max_examples "${MAX_EXAMPLES}")
[[ -n "${DEVICE}" ]] && CMD+=(--device "${DEVICE}")
[[ -n "${ACTOR_DEVICE}" ]] && CMD+=(--actor_device "${ACTOR_DEVICE}")
[[ -n "${OLD_CRITIC_DEVICE}" ]] && CMD+=(--old_critic_device "${OLD_CRITIC_DEVICE}")
[[ -n "${NEW_CRITIC_DEVICE}" ]] && CMD+=(--new_critic_device "${NEW_CRITIC_DEVICE}")
[[ ${#WORKER_LAYOUTS_ARR[@]} -gt 0 ]] && CMD+=(--worker_layouts "${WORKER_LAYOUTS_ARR[@]}")
[[ "${SHUFFLE_EXAMPLES}" != "0" ]] && CMD+=(--shuffle_examples)
[[ "${SKIP_MERGE}" != "0" ]] && CMD+=(--skip_merge)
[[ "${DISABLE_ACTOR_CACHE}" != "0" ]] && CMD+=(--disable_actor_cache)

(cd "${REPO_DIR}" && "${CMD[@]}")
