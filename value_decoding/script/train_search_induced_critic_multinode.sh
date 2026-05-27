#!/bin/bash
#SBATCH --job-name=train_simc_critic
#SBATCH --account=ECS26006
#SBATCH --partition=gh
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:20:00
#SBATCH --output=slurm-%j_train_search_induced_critic_low_ent_128.out
#SBATCH --error=slurm-%j_train_search_induced_critic_low_ent_128.err

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
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# -----------------------------
# Run identity
# -----------------------------
RUN_NAME="search_induced_critic_training"
RUN_ID="${RUN_NAME}_${SLURM_JOB_ID}"

# -----------------------------
# Paths: edit these for your run
# -----------------------------
INIT_CRITIC_CHECKPOINT_DIR="/scratch/10587/npg493/verl_runs/low_ent_critic_training_ckpt_750_actor_697767/train_log/global_step_600"
TRAIN_DATA_PATH="/work2/09576/shuozhe/verl/value_decoding/output_archive/search_induced_critic_data_722759/search_induced_critic_data/search_induced_candidates.jsonl"
EVAL_DATA_PATH="/work2/09576/shuozhe/verl/value_decoding/output_archive/search_induced_critic_data_723857/search_induced_critic_data/search_induced_candidates.jsonl"
WORK_DIR="/work2/09576/shuozhe/verl"
export PYTHONPATH="${WORK_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

ARCHIVE_ROOT="${WORK_DIR}/value_decoding/output_archive"
ARCHIVE_DIR="${ARCHIVE_ROOT}/${RUN_ID}"
SCRATCH_ROOT="${SCRATCH}/value_decoding_runs"
RUN_DIR="${SCRATCH_ROOT}/${RUN_ID}"
OUTPUT_DIR="${RUN_DIR}/search_induced_critic_training"
LOG_DIR="${RUN_DIR}/logs"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR" "$ARCHIVE_ROOT"

# Optional override directory for HF config/tokenizer metadata used during FSDP merge.
CRITIC_HF_SOURCE_DIR=""
MERGED_ROOT="${RUN_DIR}/merged_hf"

# -----------------------------
# Training config
# -----------------------------
LOSS_TYPE="hybrid"              # mse, bce, pairwise, hybrid
RANK_LOSS_WEIGHT=0.1
BATCH_SAMPLING_MODE="mixed"     # uniform, prompt_balanced, rankable_prioritized, mixed
BATCH_SIZE=32
EVAL_BATCH_SIZE=64
GRAD_ACCUM_STEPS=1
NUM_TRAIN_EPOCHS=4
LR="1e-6"
WEIGHT_DECAY="0.0"
ADAM_EPS="1e-5"
MAX_SEQ_LENGTH=1024
TRAINABLE_SCOPE="all"   # all, value_head. Full finetune of this critic OOMs on one GH200.
GRADIENT_CHECKPOINTING=1        # set 1 if TRAINABLE_SCOPE="all" and memory is tight.
DTYPE="bf16"      # FSDP uses fp32 master params with bf16 forward/reduce where supported.
SEED=42

MAX_EXAMPLES_PER_PROMPT_PER_BATCH=4
POSITIVE_FRACTION=""            # empty uses trainer default for mixed mode
RANKABLE_GROUP_FRACTION=0.5

EVAL_EVERY_STEPS=100
SAVE_EVERY_STEPS=500
EVAL_AT_START=0
MAX_EVAL_EXAMPLES=""        # empty for full eval; finite keeps rank-0 eval affordable
MAX_TRAIN_STEPS=""              # set for debug, e.g. 2
NUM_WORKERS=0
DEVICE="cuda:0"  # ignored in distributed mode; each rank uses cuda:${SLURM_LOCALID}
DISTRIBUTED_BACKEND="fsdp"
FSDP_CPU_OFFLOAD=0
# Vista gh nodes already expose the allocated GPU to the task; requesting --gres=gpu:1
# can be invalid on this partition. Leave empty unless your cluster requires it.
SRUN_GPU_ARGS=()
# Examples for other clusters:
# SRUN_GPU_ARGS=(--gres=gpu:1)
# SRUN_GPU_ARGS=(--gpus-per-task=1)
TRUST_REMOTE_CODE=1
SKIP_MERGE=0
NO_PLOTS=0

# Weights & Biases logging. Set USE_WANDB=1 after `wandb login` or with WANDB_API_KEY in the job env.
USE_WANDB=1
WANDB_PROJECT="value-decoding-train"
WANDB_ENTITY=""
WANDB_RUN_NAME="${RUN_ID}"
WANDB_GROUP="search_induced_critic"
WANDB_MODE="online"       # online, offline, disabled
WANDB_TAGS=("search_induced" "critic_training" "$LOSS_TYPE")

# Optional expensive end-to-end chunk-guidance eval.
RUN_END_TO_END_CHUNK_EVAL=0
CHUNK_EVAL_SCRIPT_PATH="${WORK_DIR}/value_decoding/chunk_guidance_eval.py"
CHUNK_EVAL_ACTOR_CHECKPOINT_DIR=""
CHUNK_EVAL_DATASET_PATH=""
CHUNK_EVAL_MAX_EXAMPLES=""
CHUNK_EVAL_NUM_SEEDS=3
CHUNK_EVAL_GENERATION_BACKEND="vllm"

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
    "$RUN_DIR"/ "$ARCHIVE_DIR"/ || true
  echo "Archived run to: $ARCHIVE_DIR"
}

cleanup() {
  sync_to_work
}
trap cleanup EXIT

archive_submitted_script() {
  local archive_dir="${OUTPUT_DIR}/submitted_script"
  mkdir -p "$archive_dir"
  local saved_path="${archive_dir}/$(basename "${BASH_SOURCE[0]}")"
  local candidates=("${BASH_SOURCE[0]}" "$0" "$SCRIPT_PATH")
  for candidate in "${candidates[@]}"; do
    if [[ -n "$candidate" && -f "$candidate" ]]; then
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
echo "Archive dir: $ARCHIVE_DIR"
echo "Initial critic checkpoint: $INIT_CRITIC_CHECKPOINT_DIR"
echo "Train data: $TRAIN_DATA_PATH"
echo "Eval data: $EVAL_DATA_PATH"

cd "$WORK_DIR"
validate_critic_checkpoint "$INIT_CRITIC_CHECKPOINT_DIR"
validate_jsonl "$TRAIN_DATA_PATH"
validate_jsonl "$EVAL_DATA_PATH"
archive_submitted_script

CMD=(
  python3 value_decoding/train_search_induced_critic.py
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

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 10000))
NUM_TRAIN_TASKS="${SLURM_NTASKS:-$SLURM_NNODES}"
export MASTER_ADDR MASTER_PORT
export WORLD_SIZE="$NUM_TRAIN_TASKS"

echo "Distributed launch: MASTER_ADDR=${MASTER_ADDR} MASTER_PORT=${MASTER_PORT} WORLD_SIZE=${WORLD_SIZE} TASKS=${NUM_TRAIN_TASKS}" | tee "${LOG_DIR}/distributed_env.log"

printf 'Command:' | tee "${LOG_DIR}/train_command.log"
printf ' %q' "${CMD[@]}" | tee -a "${LOG_DIR}/train_command.log"
printf '\n' | tee -a "${LOG_DIR}/train_command.log"

srun --nodes="${SLURM_NNODES}" --ntasks="$NUM_TRAIN_TASKS" "${SRUN_GPU_ARGS[@]}" "${CMD[@]}" 2>&1 | tee "${LOG_DIR}/train.log"

echo "Search-induced critic training finished successfully."
echo "Output: $OUTPUT_DIR"
echo "Archive: $ARCHIVE_DIR"
