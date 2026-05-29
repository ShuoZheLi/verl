#!/bin/bash
set -eo pipefail

# Local 4-GPU launcher for search-induced critic training.
# Override most knobs by exporting the variable before running this script, e.g.:
#   MAX_TRAIN_STEPS=2 MAX_EVAL_EXAMPLES=128 USE_WANDB=0 bash value_decoding/script/train_search_induced_critic_local_4gpu.sh

# -----------------------------
# Environment setup
# -----------------------------
source /data/shuozhe/miniconda3/etc/profile.d/conda.sh
conda activate verl
set -u

WORK_DIR="${WORK_DIR:-/data/shuozhe/verl}"
export PYTHONPATH="${WORK_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HOME="${HF_HOME:-/data/shuozhe/.cache/huggingface}"
export TIKTOKEN_ENCODINGS_BASE="${TIKTOKEN_ENCODINGS_BASE:-/data/shuozhe/data/embeddings}"
mkdir -p "$HF_HOME" "$TIKTOKEN_ENCODINGS_BASE"

# -----------------------------
# Run identity and paths
# -----------------------------
RUN_NAME="${RUN_NAME:-search_induced_critic_training_local_4gpu}"
RUN_ID="${RUN_ID:-${RUN_NAME}_$(date +%Y%m%d_%H%M%S)}"

INIT_CRITIC_CHECKPOINT_DIR="${INIT_CRITIC_CHECKPOINT_DIR:-/data/shuozhe/verl/train_log/job_05b_vh_init_e5_metamath/global_step_800}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-/data/shuozhe/verl/value_decoding/local_runs/search_induced_critic_data_local_4gpu_20260528_025542/search_induced_critic_data/search_induced_candidates.jsonl}"
EVAL_DATA_PATH="${EVAL_DATA_PATH:-$TRAIN_DATA_PATH}"

OUTPUT_ARCHIVE_ROOT="${OUTPUT_ARCHIVE_ROOT:-${WORK_DIR}/value_decoding/output_archive}"
RUN_ROOT="${RUN_ROOT:-${WORK_DIR}/value_decoding/local_runs}"
RUN_DIR="${RUN_DIR:-${RUN_ROOT}/${RUN_ID}}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_DIR}/search_induced_critic_training}"
LOG_DIR="${RUN_DIR}/logs"
CHECKPOINT_DIR="${OUTPUT_DIR}/checkpoints"
MERGED_ROOT="${MERGED_ROOT:-${RUN_DIR}/merged_hf}"
CACHE_ROOT="${CACHE_ROOT:-${RUN_DIR}/cache}"
ARCHIVE_DIR="${ARCHIVE_DIR:-${OUTPUT_ARCHIVE_ROOT}/${RUN_ID}}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR" "$CHECKPOINT_DIR" "$MERGED_ROOT" "$CACHE_ROOT" "$OUTPUT_ARCHIVE_ROOT"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_ROOT}/xdg}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${CACHE_ROOT}/torch_inductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${CACHE_ROOT}/triton}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${CACHE_ROOT}/torch_extensions}"
export DO_NOT_TRACK="${DO_NOT_TRACK:-1}"
CUDA_RUNTIME_LIB_DIR="${CUDA_RUNTIME_LIB_DIR:-/data/shuozhe/miniconda3/envs/verl/lib}"
CUDA_RUNTIME_LIB64_DIR="${CACHE_ROOT}/cuda_runtime/lib64"
mkdir -p "$CUDA_RUNTIME_LIB64_DIR" "$XDG_CACHE_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"
ln -sf "${CUDA_RUNTIME_LIB_DIR}/libcudart.so" "$CUDA_RUNTIME_LIB64_DIR/libcudart.so"
export CUDA_HOME="${CUDA_HOME:-/data/shuozhe/miniconda3/envs/verl}"
export LD_LIBRARY_PATH="${CUDA_RUNTIME_LIB64_DIR}:${CUDA_RUNTIME_LIB_DIR}:${CUDA_HOME}/lib:${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export LIBRARY_PATH="${CUDA_RUNTIME_LIB64_DIR}:${CUDA_RUNTIME_LIB_DIR}:${CUDA_HOME}/lib:${CUDA_HOME}/lib64${LIBRARY_PATH:+:${LIBRARY_PATH}}"
export LDFLAGS="-L${CUDA_RUNTIME_LIB64_DIR} -L${CUDA_RUNTIME_LIB_DIR} -L${CUDA_HOME}/lib -L${CUDA_HOME}/lib64${LDFLAGS:+ ${LDFLAGS}}"

# Optional override directory for HF config/tokenizer metadata used during FSDP merge.
CRITIC_HF_SOURCE_DIR="${CRITIC_HF_SOURCE_DIR:-}"

# -----------------------------
# Training config
# -----------------------------
LOSS_TYPE="${LOSS_TYPE:-hybrid}"              # mse, bce, pairwise, hybrid
RANK_LOSS_WEIGHT="${RANK_LOSS_WEIGHT:-0.1}"
BATCH_SAMPLING_MODE="${BATCH_SAMPLING_MODE:-mixed}"     # uniform, prompt_balanced, rankable_prioritized, mixed
BATCH_SIZE="${BATCH_SIZE:-32}"                # per GPU/rank; effective batch = BATCH_SIZE * GRAD_ACCUM_STEPS * NUM_LOCAL_GPUS
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-64}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-4}"
LR="${LR:-5e-7}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
ADAM_EPS="${ADAM_EPS:-1e-5}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-1024}"
TRAINABLE_SCOPE="${TRAINABLE_SCOPE:-all}"     # all, value_head. FSDP requires all.
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
DTYPE="${DTYPE:-bf16}"
SEED="${SEED:-42}"

MAX_EXAMPLES_PER_PROMPT_PER_BATCH="${MAX_EXAMPLES_PER_PROMPT_PER_BATCH:-4}"
POSITIVE_FRACTION="${POSITIVE_FRACTION:-}"    # empty uses trainer default for mixed mode
RANKABLE_GROUP_FRACTION="${RANKABLE_GROUP_FRACTION:-0.5}"

EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-100}"
SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-100}"
EVAL_AT_START="${EVAL_AT_START:-1}"
MAX_EVAL_EXAMPLES="${MAX_EVAL_EXAMPLES:-}"    # set for fast local smoke tests, e.g. 128
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-}"        # set for debug, e.g. 2
NUM_WORKERS="${NUM_WORKERS:-0}"
DEVICE="${DEVICE:-cuda:0}"                    # ignored in FSDP mode; each rank uses cuda:${LOCAL_RANK}
DISTRIBUTED_BACKEND="${DISTRIBUTED_BACKEND:-fsdp}"
FSDP_CPU_OFFLOAD="${FSDP_CPU_OFFLOAD:-0}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
SKIP_MERGE="${SKIP_MERGE:-0}"
NO_PLOTS="${NO_PLOTS:-0}"
DRY_RUN="${DRY_RUN:-0}"

# Local distributed launch. NUM_LOCAL_GPUS defaults to the number of visible GPU ids.
CUDA_DEVICES_CSV="${CUDA_DEVICES_CSV:-0,1,2,3}"
if [[ -z "${NUM_LOCAL_GPUS:-}" ]]; then
  IFS=',' read -r -a CUDA_DEVICE_IDS <<< "$CUDA_DEVICES_CSV"
  NUM_LOCAL_GPUS="${#CUDA_DEVICE_IDS[@]}"
  unset CUDA_DEVICE_IDS
fi
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29517}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
TORCHRUN_STANDALONE="${TORCHRUN_STANDALONE:-1}"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES_CSV"
export MASTER_ADDR MASTER_PORT OMP_NUM_THREADS

# Weights & Biases logging. Disabled by default for local smoke tests.
USE_WANDB="${USE_WANDB:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-value-decoding-train}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_RUN_NAME="${WANDB_RUN_NAME:-$RUN_ID}"
WANDB_GROUP="${WANDB_GROUP:-search_induced_critic_local}"
WANDB_MODE="${WANDB_MODE:-offline}"       # online, offline, disabled
WANDB_TAGS=("search_induced" "critic_training" "local_4gpu" "$LOSS_TYPE")

# Optional expensive end-to-end chunk-guidance eval.
RUN_END_TO_END_CHUNK_EVAL="${RUN_END_TO_END_CHUNK_EVAL:-0}"
CHUNK_EVAL_SCRIPT_PATH="${CHUNK_EVAL_SCRIPT_PATH:-${WORK_DIR}/value_decoding/chunk_guidance_eval.py}"
CHUNK_EVAL_ACTOR_CHECKPOINT_DIR="${CHUNK_EVAL_ACTOR_CHECKPOINT_DIR:-}"
CHUNK_EVAL_DATASET_PATH="${CHUNK_EVAL_DATASET_PATH:-}"
CHUNK_EVAL_MAX_EXAMPLES="${CHUNK_EVAL_MAX_EXAMPLES:-}"
CHUNK_EVAL_NUM_SEEDS="${CHUNK_EVAL_NUM_SEEDS:-3}"
CHUNK_EVAL_GENERATION_BACKEND="${CHUNK_EVAL_GENERATION_BACKEND:-vllm}"

# -----------------------------
# Helpers
# -----------------------------
SCRIPT_PATH="${WORK_DIR}/value_decoding/script/$(basename "${BASH_SOURCE[0]}")"

sync_to_archive() {
  echo "Syncing run directory to archive..."
  mkdir -p "$ARCHIVE_DIR"
  rsync -a \
    --exclude='merged_hf/' \
    --exclude='merged_hf/***' \
    --exclude='cache/' \
    --exclude='cache/***' \
    --exclude='search_induced_critic_training/checkpoints/' \
    --exclude='search_induced_critic_training/checkpoints/***' \
    "$RUN_DIR"/ "$ARCHIVE_DIR"/ || true
  echo "Archived run to: $ARCHIVE_DIR"
}
trap sync_to_archive EXIT

archive_submitted_script() {
  local saved_path="${LOG_DIR}/$(basename "${SCRIPT_PATH}")"
  local candidate
  for candidate in "$SCRIPT_PATH" "${BASH_SOURCE[0]}"; do
    if [[ -f "$candidate" ]]; then
      cp "$candidate" "$saved_path" || true
      echo "Saved launcher script snapshot to: $saved_path"
      return 0
    fi
  done
  echo "Warning: could not find launcher script to archive; continuing." >&2
}

validate_critic_checkpoint() {
  local checkpoint_dir="$1"
  python3 - "$checkpoint_dir" <<'PY'
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
component_dir = resolve_component_checkpoint_dir(checkpoint_dir, component="critic")
if has_complete_hf_checkpoint(component_dir):
    print(f"critic: detected complete Hugging Face checkpoint at {component_dir}")
    raise SystemExit(0)
if has_fsdp_checkpoint_shards(component_dir):
    print(f"critic: detected raw FSDP checkpoint at {component_dir}")
    raise SystemExit(0)
if has_hf_config(component_dir):
    missing_files = find_missing_hf_weight_files(component_dir)
    missing_preview = ", ".join(path.name for path in missing_files[:5]) or "unknown weight shards"
    if len(missing_files) > 5:
        missing_preview += ", ..."
    raise SystemExit(
        f"critic: incomplete Hugging Face checkpoint at {component_dir}. "
        f"Missing files referenced by the index: {missing_preview}"
    )
raise SystemExit(f"critic: unsupported checkpoint layout at {component_dir}")
PY
}

validate_jsonl() {
  local path="$1"
  python3 - "$path" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"JSONL file not found: {path}")
num_rows = 0
num_rankable = 0
with path.open("r", encoding="utf-8") as handle:
    for line in handle:
        if not line.strip():
            continue
        row = json.loads(line)
        if "mc_reward" not in row:
            raise SystemExit(f"Missing mc_reward in {path}")
        num_rows += 1
        num_rankable += int(bool(row.get("group_has_mixed_rewards", False)))
print(f"{path}: rows={num_rows}, rows_marked_rankable={num_rankable}")
if num_rows <= 0:
    raise SystemExit(f"No rows in {path}")
PY
}

# -----------------------------
# Validation and launch
# -----------------------------
echo "Activated environment"
echo "Python: $(which python3)"
python3 -V
nvidia-smi || true

echo "Run ID: $RUN_ID"
echo "Output dir: $OUTPUT_DIR"
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo "Archive dir: $ARCHIVE_DIR"
echo "Initial critic checkpoint: $INIT_CRITIC_CHECKPOINT_DIR"
echo "Train data: $TRAIN_DATA_PATH"
echo "Eval data: $EVAL_DATA_PATH"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "NUM_LOCAL_GPUS: $NUM_LOCAL_GPUS"
echo "Effective batch size: $((BATCH_SIZE * GRAD_ACCUM_STEPS * NUM_LOCAL_GPUS))"

if [[ "$NUM_LOCAL_GPUS" -le 0 ]]; then
  echo "NUM_LOCAL_GPUS must be positive, got: $NUM_LOCAL_GPUS" >&2
  exit 1
fi
if [[ "$DISTRIBUTED_BACKEND" == "fsdp" && "$TRAINABLE_SCOPE" != "all" ]]; then
  echo "DISTRIBUTED_BACKEND=fsdp requires TRAINABLE_SCOPE=all for this trainer." >&2
  exit 1
fi
if [[ "$DISTRIBUTED_BACKEND" == "none" && "$NUM_LOCAL_GPUS" != "1" ]]; then
  echo "DISTRIBUTED_BACKEND=none would duplicate training under torchrun; set NUM_LOCAL_GPUS=1 or use fsdp." >&2
  exit 1
fi

cd "$WORK_DIR"
validate_critic_checkpoint "$INIT_CRITIC_CHECKPOINT_DIR"
validate_jsonl "$TRAIN_DATA_PATH"
validate_jsonl "$EVAL_DATA_PATH"
archive_submitted_script

CMD=(
  value_decoding/train_search_induced_critic.py
  --init_critic_checkpoint_dir "$INIT_CRITIC_CHECKPOINT_DIR"
  --train_data_path "$TRAIN_DATA_PATH"
  --eval_data_path "$EVAL_DATA_PATH"
  --output_dir "$OUTPUT_DIR"
  --loss_type "$LOSS_TYPE"
  --rank_loss_weight "$RANK_LOSS_WEIGHT"
  --batch_sampling_mode "$BATCH_SAMPLING_MODE"
  --batch_size "$BATCH_SIZE"
  --eval_batch_size "$EVAL_BATCH_SIZE"
  --grad_accum_steps "$GRAD_ACCUM_STEPS"
  --num_train_epochs "$NUM_TRAIN_EPOCHS"
  --lr "$LR"
  --weight_decay "$WEIGHT_DECAY"
  --adam_eps "$ADAM_EPS"
  --max_seq_length "$MAX_SEQ_LENGTH"
  --trainable_scope "$TRAINABLE_SCOPE"
  --eval_every_steps "$EVAL_EVERY_STEPS"
  --save_every_steps "$SAVE_EVERY_STEPS"
  --dtype "$DTYPE"
  --seed "$SEED"
  --max_examples_per_prompt_per_batch "$MAX_EXAMPLES_PER_PROMPT_PER_BATCH"
  --rankable_group_fraction "$RANKABLE_GROUP_FRACTION"
  --device "$DEVICE"
  --merged_root "$MERGED_ROOT"
  --num_workers "$NUM_WORKERS"
  --distributed_backend "$DISTRIBUTED_BACKEND"
)

[[ -n "$POSITIVE_FRACTION" ]] && CMD+=(--positive_fraction "$POSITIVE_FRACTION")
[[ -n "$MAX_EVAL_EXAMPLES" ]] && CMD+=(--max_eval_examples "$MAX_EVAL_EXAMPLES")
[[ -n "$MAX_TRAIN_STEPS" ]] && CMD+=(--max_train_steps "$MAX_TRAIN_STEPS")
[[ "$EVAL_AT_START" != "0" ]] && CMD+=(--eval_at_start)
[[ -n "$CRITIC_HF_SOURCE_DIR" ]] && CMD+=(--critic_hf_source_dir "$CRITIC_HF_SOURCE_DIR")
[[ "$TRUST_REMOTE_CODE" != "0" ]] && CMD+=(--trust_remote_code)
[[ "$SKIP_MERGE" != "0" ]] && CMD+=(--skip_merge)
[[ "$NO_PLOTS" != "0" ]] && CMD+=(--no_plots)
[[ "$GRADIENT_CHECKPOINTING" != "0" ]] && CMD+=(--gradient_checkpointing)
[[ "$FSDP_CPU_OFFLOAD" != "0" ]] && CMD+=(--fsdp_cpu_offload)

if [[ "$USE_WANDB" != "0" ]]; then
  CMD+=(
    --use_wandb
    --wandb_project "$WANDB_PROJECT"
    --wandb_run_name "$WANDB_RUN_NAME"
    --wandb_group "$WANDB_GROUP"
    --wandb_mode "$WANDB_MODE"
  )
  [[ -n "$WANDB_ENTITY" ]] && CMD+=(--wandb_entity "$WANDB_ENTITY")
  if [[ "${#WANDB_TAGS[@]}" -gt 0 ]]; then
    CMD+=(--wandb_tags "${WANDB_TAGS[@]}")
  fi
fi

if [[ "$RUN_END_TO_END_CHUNK_EVAL" != "0" ]]; then
  CMD+=(
    --run_end_to_end_chunk_eval
    --chunk_eval_script_path "$CHUNK_EVAL_SCRIPT_PATH"
    --chunk_eval_actor_checkpoint_dir "$CHUNK_EVAL_ACTOR_CHECKPOINT_DIR"
    --chunk_eval_dataset_path "$CHUNK_EVAL_DATASET_PATH"
    --chunk_eval_generation_backend "$CHUNK_EVAL_GENERATION_BACKEND"
    --chunk_eval_num_seeds "$CHUNK_EVAL_NUM_SEEDS"
  )
  [[ -n "$CHUNK_EVAL_MAX_EXAMPLES" ]] && CMD+=(--chunk_eval_max_examples "$CHUNK_EVAL_MAX_EXAMPLES")
fi

LAUNCH_CMD=(torchrun --nproc_per_node="$NUM_LOCAL_GPUS" --master_addr="$MASTER_ADDR" --master_port="$MASTER_PORT")
if [[ "$TORCHRUN_STANDALONE" != "0" ]]; then
  LAUNCH_CMD=(torchrun --standalone --nproc_per_node="$NUM_LOCAL_GPUS")
fi

printf 'Launch command:' | tee "${LOG_DIR}/train_command.log"
printf ' %q' "${LAUNCH_CMD[@]}" "${CMD[@]}" | tee -a "${LOG_DIR}/train_command.log"
printf '\n' | tee -a "${LOG_DIR}/train_command.log"

if [[ "$DRY_RUN" != "0" ]]; then
  echo "DRY_RUN=1, not launching training."
  exit 0
fi

"${LAUNCH_CMD[@]}" "${CMD[@]}" 2>&1 | tee "${LOG_DIR}/train.log"

echo "Search-induced critic training finished successfully."
echo "Output: $OUTPUT_DIR"
echo "Checkpoints: $CHECKPOINT_DIR"
echo "Archive: $ARCHIVE_DIR"
