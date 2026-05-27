#!/bin/bash
#SBATCH --job-name=collect_search_data
#SBATCH --account=ECS26006
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

# -----------------------------
# Environment setup
# -----------------------------
module reset
module load nvidia/25.9

VENV="/work/09576/shuozhe/verl_setup_tacc/.venv"
source "${VENV}/bin/activate"

UV_CACHE_DIR="${SCRATCH}/.cache/uv"
HF_HOME="${SCRATCH}/.cache/huggingface"
TIKTOKEN_ENCODINGS_BASE="${SCRATCH}/data/embeddings"

mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$TIKTOKEN_ENCODINGS_BASE"

export UV_CACHE_DIR
export HF_HOME
export TIKTOKEN_ENCODINGS_BASE
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=true
export VLLM_USE_V1=1

echo "Activated environment"
echo "Python: $(which python3)"
python3 -V

# -----------------------------
# Run identity
# -----------------------------
RUN_NAME="search_induced_critic_data"
RUN_ID="${RUN_NAME}_${SLURM_JOB_ID}"

# -----------------------------
# Paths
# -----------------------------
ACTOR_CHECKPOINT_DIR="/work2/09576/shuozhe/saved_model/Prathyusha101/Qwen2.5_7b_750_ckpt_global_step_750"
COLLECTOR_CRITIC_CHECKPOINT_DIR="/work2/09576/shuozhe/saved_model/Prathyusha101/Qwen2.5_7b_750_ckpt_global_step_750"
DATASET_PATH="/work2/09576/shuozhe/saved_dataset/MetaMathQA-math-500/train.parquet"
WORK_DIR="/work2/09576/shuozhe/verl"
export PYTHONPATH="${WORK_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

ARCHIVE_ROOT="/work2/09576/shuozhe/verl/value_decoding/output_archive"
ARCHIVE_DIR="${ARCHIVE_ROOT}/${RUN_ID}"

SCRATCH_ROOT="${SCRATCH}/value_decoding_runs"
RUN_DIR="${SCRATCH_ROOT}/${RUN_ID}"
LOG_DIR="${RUN_DIR}/logs"
OUTPUT_DIR="${RUN_DIR}/search_induced_critic_data"
MERGED_ROOT="${RUN_DIR}/merged_hf"

mkdir -p "$LOG_DIR" "$ARCHIVE_ROOT" "$OUTPUT_DIR" "$MERGED_ROOT"

ACTOR_HF_SOURCE_DIR=""
CRITIC_HF_SOURCE_DIR=""

# -----------------------------
# Collection config
# -----------------------------
PROMPT_KEY="prompt"
RESPONSE_KEY=""
START_INDEX=0
MAX_PROMPTS=2000
SHUFFLE_PROMPTS=0

CHUNK_SIZE=256
NUM_CHUNK_CANDIDATES=8
NUM_SEARCH_STEPS_PER_PROMPT=4
COMPLETION_MAX_NEW_TOKENS=2048
COLLECTOR_SELECTION_MODE="argmax"
COLLECTOR_EPSILON=0.1
COLLECTOR_VALUE_TEMPERATURE=1.0

ACTOR_TEMPERATURE=1.0
ACTOR_TOP_P=1.0
ACTOR_TOP_K=0
ACTOR_BATCH_SIZE=8

MAX_PROMPT_LENGTH=2048
DTYPE="bf16"
SEED=42
GENERATION_BACKEND="vllm"
CRITIC_BATCH_SIZE=8
VLLM_GPU_MEMORY_UTILIZATION=0.60
VLLM_TENSOR_PARALLEL_SIZE=1
VLLM_MAX_MODEL_LEN=""
VLLM_ENFORCE_EAGER=0

SAVE_FULL_TEXT=1
SAVE_TOKEN_IDS=1
SAVE_COMPLETED_RESPONSES=0
TRUST_REMOTE_CODE=0
SKIP_MERGE=0
DEBUG_NUM_PROMPTS=""

# -----------------------------
# Helpers
# -----------------------------
SCRIPT_PATH="${WORK_DIR}/value_decoding/script/$(basename "${BASH_SOURCE[0]}")"

sync_to_work() {
  echo "Syncing run directory back to WORK..."
  mkdir -p "$ARCHIVE_DIR"
  rsync -a \
    --exclude='merged_hf/' \
    --exclude='merged_hf/***' \
    --exclude='shards/*/merged_hf/' \
    --exclude='shards/*/merged_hf/***' \
    "$RUN_DIR"/ "$ARCHIVE_DIR"/ || true
  echo "Archived run to: $ARCHIVE_DIR"
}
trap sync_to_work EXIT

validate_component_checkpoint() {
  local checkpoint_dir="$1"
  local component="$2"
  python3 - "$checkpoint_dir" "$component" <<'PY'
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

# -----------------------------
# Debug info
# -----------------------------
echo "Job ID: $SLURM_JOB_ID"
echo "Run ID: $RUN_ID"
echo "RUN_DIR: $RUN_DIR"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "ARCHIVE_DIR: $ARCHIVE_DIR"

echo "Checking inputs..."
ls -ld "$WORK_DIR"
ls -lh "$DATASET_PATH"
ls -ld "$ACTOR_CHECKPOINT_DIR"
ls -ld "$COLLECTOR_CRITIC_CHECKPOINT_DIR"
validate_component_checkpoint "$ACTOR_CHECKPOINT_DIR" actor
validate_component_checkpoint "$COLLECTOR_CRITIC_CHECKPOINT_DIR" critic
cp "$SCRIPT_PATH" "$LOG_DIR/$(basename "$SCRIPT_PATH")"

# -----------------------------
# Run collector
# -----------------------------
cd "$WORK_DIR"

CMD=(
  python3 value_decoding/collect_search_induced_critic_data.py
  --actor_checkpoint_dir "$ACTOR_CHECKPOINT_DIR"
  --collector_critic_checkpoint_dir "$COLLECTOR_CRITIC_CHECKPOINT_DIR"
  --dataset_path "$DATASET_PATH"
  --output_dir "$OUTPUT_DIR"
  --prompt_key "$PROMPT_KEY"
  --start_index "$START_INDEX"
  --max_prompts "$MAX_PROMPTS"
  --chunk_size "$CHUNK_SIZE"
  --num_chunk_candidates "$NUM_CHUNK_CANDIDATES"
  --num_search_steps_per_prompt "$NUM_SEARCH_STEPS_PER_PROMPT"
  --completion_max_new_tokens "$COMPLETION_MAX_NEW_TOKENS"
  --collector_selection_mode "$COLLECTOR_SELECTION_MODE"
  --collector_epsilon "$COLLECTOR_EPSILON"
  --collector_value_temperature "$COLLECTOR_VALUE_TEMPERATURE"
  --actor_temperature "$ACTOR_TEMPERATURE"
  --actor_top_p "$ACTOR_TOP_P"
  --actor_top_k "$ACTOR_TOP_K"
  --actor_batch_size "$ACTOR_BATCH_SIZE"
  --max_prompt_length "$MAX_PROMPT_LENGTH"
  --dtype "$DTYPE"
  --seed "$SEED"
  --generation_backend "$GENERATION_BACKEND"
  --critic_device cuda:0
  --merged_root "$MERGED_ROOT"
  --critic_batch_size "$CRITIC_BATCH_SIZE"
  --vllm_gpu_memory_utilization "$VLLM_GPU_MEMORY_UTILIZATION"
  --vllm_tensor_parallel_size "$VLLM_TENSOR_PARALLEL_SIZE"
)

[[ -n "$RESPONSE_KEY" ]] && CMD+=(--response_key "$RESPONSE_KEY")
[[ -n "$ACTOR_HF_SOURCE_DIR" ]] && CMD+=(--actor_hf_source_dir "$ACTOR_HF_SOURCE_DIR")
[[ -n "$CRITIC_HF_SOURCE_DIR" ]] && CMD+=(--critic_hf_source_dir "$CRITIC_HF_SOURCE_DIR")
[[ -n "$VLLM_MAX_MODEL_LEN" ]] && CMD+=(--vllm_max_model_len "$VLLM_MAX_MODEL_LEN")
[[ -n "$DEBUG_NUM_PROMPTS" ]] && CMD+=(--debug_num_prompts "$DEBUG_NUM_PROMPTS")
[[ "$SHUFFLE_PROMPTS" != "0" ]] && CMD+=(--shuffle_prompts true)
[[ "$VLLM_ENFORCE_EAGER" != "0" ]] && CMD+=(--vllm_enforce_eager true)
[[ "$SAVE_FULL_TEXT" == "0" ]] && CMD+=(--save_full_text false)
[[ "$SAVE_TOKEN_IDS" == "0" ]] && CMD+=(--save_token_ids false)
[[ "$SAVE_COMPLETED_RESPONSES" != "0" ]] && CMD+=(--save_completed_responses true)
[[ "$TRUST_REMOTE_CODE" != "0" ]] && CMD+=(--trust_remote_code true)
[[ "$SKIP_MERGE" != "0" ]] && CMD+=(--skip_merge true)

printf 'Running command:\n'
printf ' %q' "${CMD[@]}"
printf '\n'
"${CMD[@]}" 2>&1 | tee "$LOG_DIR/collect_search_induced_critic_data.log"

echo "Search-induced critic data collection finished successfully."
echo "Output: $OUTPUT_DIR"
