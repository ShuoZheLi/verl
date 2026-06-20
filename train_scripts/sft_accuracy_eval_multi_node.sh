#!/bin/bash
#SBATCH --job-name=sft_qwen25_3b
#SBATCH --account=ASC24079
#SBATCH --partition=gh
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:20:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

# -----------------------------
# Environment setup
# -----------------------------
module reset
module load nvidia/25.9

VENV="${VENV:-/work/09576/shuozhe/verl_setup_tacc/.venv}"
source "${VENV}/bin/activate"

UV_CACHE_DIR="${UV_CACHE_DIR:-${SCRATCH}/.cache/uv}"
HF_HOME="${HF_HOME:-${SCRATCH}/.cache/huggingface}"
TIKTOKEN_ENCODINGS_BASE="${TIKTOKEN_ENCODINGS_BASE:-${SCRATCH}/data/embeddings}"
TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${SCRATCH}/.cache/torch_extensions}"

mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$TIKTOKEN_ENCODINGS_BASE" "$TORCH_EXTENSIONS_DIR"

export UV_CACHE_DIR
export HF_HOME
export TIKTOKEN_ENCODINGS_BASE
export TORCH_EXTENSIONS_DIR
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=true
export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-0}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export WANDB_PROJECT=${WANDB_PROJECT:-SFT_midi}

# Avoid inheriting incompatible local launch variables.
unset MASTER_ADDR MASTER_PORT WORLD_SIZE RANK LOCAL_RANK GROUP_RANK ROLE_RANK ROLE_NAME TORCHELASTIC_RUN_ID || true

echo "Activated environment"
echo "Python: $(which python3)"
python3 -V

# -----------------------------
# Run identity and paths
# -----------------------------
RUN_NAME="${RUN_NAME:-qwen2.5-3b-math7500-sft}"
REAL_SLURM_JOB_ID="${SLURM_JOB_ID:-manual}"
RUN_ID="${RUN_NAME}_${REAL_SLURM_JOB_ID}"

HF_DATASETS_CACHE_ROOT="${HF_DATASETS_CACHE:-}"
HF_MODULES_CACHE_ROOT="${HF_MODULES_CACHE:-}"
export HF_DATASETS_CACHE_ROOT
export HF_MODULES_CACHE_ROOT

WORK_DIR="${WORK_DIR:-/work2/09576/shuozhe/verl}"
export PYTHONPATH="${WORK_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

MODEL_INIT_CKPT="${MODEL_INIT_CKPT:-/work2/09576/shuozhe/saved_model/Qwen2.5-3B}"
TRAIN_FILE="${TRAIN_FILE:-/work2/09576/shuozhe/saved_dataset/MetaMathQA-math-500/math7500_sft.parquet}"
VAL_FILE="${VAL_FILE:-/work2/09576/shuozhe/saved_dataset/MetaMathQA-math-500/test.parquet}"

SCRATCH_ROOT="${SCRATCH_ROOT:-${SCRATCH}/verl_runs}"
RUN_DIR="${SCRATCH_ROOT}/${RUN_ID}"
LOG_DIR="${RUN_DIR}/logs"
TRAIN_LOG_DIR="${RUN_DIR}/train_log"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-/work2/09576/shuozhe/verl/train_log_archive}"
ARCHIVE_DIR="${ARCHIVE_ROOT}/${RUN_ID}"
TRAIN_STDOUT_LOG="${TRAIN_LOG_DIR}/job_${RUN_ID}.txt"

mkdir -p "$LOG_DIR" "$TRAIN_LOG_DIR" "$ARCHIVE_ROOT"

# -----------------------------
# SFT training defaults
# -----------------------------
# This is a small-data full fine-tune. Defaults are conservative for Qwen2.5-3B.
train_batch_size=${train_batch_size:-32}
micro_batch_size_per_gpu=${micro_batch_size_per_gpu:-1}
max_length=${max_length:-4096}
max_token_len_per_gpu=${max_token_len_per_gpu:-4096}
lr=${lr:-5e-6}
total_epochs=${total_epochs:-3}
save_freq=${save_freq:-100}

# -----------------------------
# Eval options
# -----------------------------
# eval_method: loss, generation_reward, or both.
eval_method=${eval_method:-generation_reward}
eval_before_train=${eval_before_train:-True}
eval_freq=${eval_freq:--1}
loss_eval_freq=${loss_eval_freq:-50}
generation_eval_freq=${generation_eval_freq:-50}

# loss_eval_files must be SFT messages format; generation_eval_files must be PPO prompt+reward_model format.
loss_eval_files=${loss_eval_files:-${TRAIN_FILE}}
generation_eval_files=${generation_eval_files:-${VAL_FILE}}

# Generation accuracy eval can be memory-heavy; start small unless you know memory is safe.
generation_eval_batch_size=${generation_eval_batch_size:-32}
generation_max_new_tokens=${generation_max_new_tokens:-2048}
generation_do_sample=${generation_do_sample:-False}
generation_temperature=${generation_temperature:-1.0}
generation_top_p=${generation_top_p:-1.0}
generation_top_k=${generation_top_k:-null}
generation_num_samples=${generation_num_samples:-1}
generation_dtype=${generation_dtype:-null}

generation_backend=${generation_backend:-vllm}
generation_vllm_gpu_memory_utilization=${generation_vllm_gpu_memory_utilization:-0.3}
generation_vllm_host_ip=${generation_vllm_host_ip:-127.0.0.1}
generation_vllm_enforce_eager=${generation_vllm_enforce_eager:-True}
generation_vllm_sync_weights=${generation_vllm_sync_weights:-True}
generation_vllm_enable_multiprocessing=${generation_vllm_enable_multiprocessing:-False}

# -----------------------------
# Multi-node torchrun config
# -----------------------------
GPUS_PER_NODE="${GPUS_PER_NODE:-1}"
NNODES="${SLURM_JOB_NUM_NODES}"
RDZV_PORT="${RDZV_PORT:-29500}"

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node="${nodes_array[0]}"
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

resolved_head_node_ip=""
for candidate_ip in $head_node_ip; do
  if [[ "$candidate_ip" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]]; then
    resolved_head_node_ip="$candidate_ip"
    break
  fi
done
if [[ -z "$resolved_head_node_ip" ]]; then
  for candidate_ip in $head_node_ip; do
    resolved_head_node_ip="$candidate_ip"
    break
  done
fi
if [[ -z "$resolved_head_node_ip" ]]; then
  echo "Failed to resolve a usable IP address for torchrun head node $head_node." >&2
  exit 1
fi
MASTER_ADDR="$resolved_head_node_ip"
MASTER_PORT="$RDZV_PORT"
RDZV_ENDPOINT="${MASTER_ADDR}:${MASTER_PORT}"

# -----------------------------
# Helpers
# -----------------------------
describe_path() {
  local label="$1"
  local path="$2"

  echo "$label: $path"
  if [[ "$path" == *"://"* ]]; then
    echo "  non-local URI path"
  elif [[ -d "$path" || -f "$path" ]]; then
    ls -ld "$path"
  elif [[ "$path" = /* || "$path" == ./* || "$path" == ../* || "$path" == ~* ]]; then
    echo "  local path not found"
  else
    echo "  passthrough model identifier"
  fi
}

sync_to_work() {
  echo "Syncing run directory back to WORK..."
  mkdir -p "$ARCHIVE_DIR"
  rsync -a "$RUN_DIR"/ "$ARCHIVE_DIR"/ || true
  echo "Archived run to: $ARCHIVE_DIR"
}

cleanup() {
  sync_to_work
}
trap cleanup EXIT

# -----------------------------
# Debug info and input checks
# -----------------------------
echo "Job ID: ${SLURM_JOB_ID}"
echo "Run ID: ${RUN_ID}"
echo "SLURM nodes: ${SLURM_JOB_NODELIST}"
echo "Head node: ${head_node}"
echo "Rendezvous endpoint: ${RDZV_ENDPOINT}"
echo "GPUS_PER_NODE: ${GPUS_PER_NODE}"
echo "NNODES: ${NNODES}"
echo "SCRATCH: ${SCRATCH}"
echo "RUN_DIR: ${RUN_DIR}"
echo "LOG_DIR: ${LOG_DIR}"
echo "TRAIN_LOG_DIR: ${TRAIN_LOG_DIR}"
describe_path "WORK_DIR" "$WORK_DIR"
describe_path "MODEL_INIT_CKPT" "$MODEL_INIT_CKPT"
describe_path "TRAIN_FILE" "$TRAIN_FILE"
describe_path "VAL_FILE" "$VAL_FILE"
describe_path "loss_eval_files" "$loss_eval_files"
describe_path "generation_eval_files" "$generation_eval_files"

ls -ld "$WORK_DIR"
ls -lh "$TRAIN_FILE"
ls -lh "$VAL_FILE"

cd "$WORK_DIR"

# -----------------------------
# Run SFT training
# -----------------------------
# One Slurm task launches one torchrun agent per node. torchrun then launches GPUS_PER_NODE
# local trainer processes on that node. SLURM_PROCID is the node_rank because ntasks-per-node=1.
srun --nodes="$NNODES" --ntasks="$NNODES" --ntasks-per-node=1 \
  bash -c '
    set -euo pipefail
    source "'"${VENV}"'/bin/activate"
    cd "'"${WORK_DIR}"'"
    export PYTHONPATH="'"${WORK_DIR}"'${PYTHONPATH:+:${PYTHONPATH}}"
    export UV_CACHE_DIR="'"${UV_CACHE_DIR}"'"
    export HF_HOME="'"${HF_HOME}"'"
    node_cache_root="${TMPDIR:-/tmp}/verl_'"${RUN_ID}"'_node_${SLURM_PROCID}"
    export HF_DATASETS_CACHE="${HF_DATASETS_CACHE_ROOT:-${node_cache_root}/huggingface_datasets}"
    export HF_MODULES_CACHE="${HF_MODULES_CACHE_ROOT:-${node_cache_root}/huggingface_modules}"
    export TIKTOKEN_ENCODINGS_BASE="'"${TIKTOKEN_ENCODINGS_BASE}"'"
    export TORCH_EXTENSIONS_DIR="'"${TORCH_EXTENSIONS_DIR}"'/node_${SLURM_PROCID}"
    mkdir -p "$HF_DATASETS_CACHE" "$HF_MODULES_CACHE" "$TORCH_EXTENSIONS_DIR"
    export PYTHONUNBUFFERED=1
    export TOKENIZERS_PARALLELISM=true
    export HYDRA_FULL_ERROR="'"${HYDRA_FULL_ERROR}"'"
    export NCCL_DEBUG="'"${NCCL_DEBUG}"'"
    export WANDB_PROJECT="'"${WANDB_PROJECT}"'"

    torchrun \
      --nnodes="'"${NNODES}"'" \
      --nproc_per_node="'"${GPUS_PER_NODE}"'" \
      --node_rank="${SLURM_PROCID}" \
      --rdzv_id="'"${RUN_ID}"'" \
      --rdzv_backend=c10d \
      --rdzv_endpoint="'"${RDZV_ENDPOINT}"'" \
      -m verl.trainer.sft_trainer \
      data.train_files="'"${TRAIN_FILE}"'" \
      data.val_files="'"${VAL_FILE}"'" \
      data.loss_val_files="'"${loss_eval_files}"'" \
      data.generation_eval_files="'"${generation_eval_files}"'" \
      data.generation_eval_batch_size="'"${generation_eval_batch_size}"'" \
      data.messages_key=messages \
      data.train_batch_size="'"${train_batch_size}"'" \
      data.micro_batch_size_per_gpu="'"${micro_batch_size_per_gpu}"'" \
      data.max_length="'"${max_length}"'" \
      data.max_token_len_per_gpu="'"${max_token_len_per_gpu}"'" \
      data.num_workers=8 \
      optim.lr="'"${lr}"'" \
      engine=fsdp \
      model.path="'"${MODEL_INIT_CKPT}"'" \
      trainer.default_local_dir="'"${TRAIN_LOG_DIR}"'" \
      trainer.project_name=math_7500 \
      trainer.experiment_name="'"${RUN_ID}"'" \
      trainer.total_epochs="'"${total_epochs}"'" \
      trainer.save_freq="'"${save_freq}"'" \
      trainer.test_freq="'"${eval_freq}"'" \
      trainer.loss_test_freq="'"${loss_eval_freq}"'" \
      trainer.eval_method="'"${eval_method}"'" \
      trainer.val_before_train="'"${eval_before_train}"'" \
      trainer.generation_eval.test_freq="'"${generation_eval_freq}"'" \
      trainer.generation_eval.max_new_tokens="'"${generation_max_new_tokens}"'" \
      trainer.generation_eval.do_sample="'"${generation_do_sample}"'" \
      trainer.generation_eval.temperature="'"${generation_temperature}"'" \
      trainer.generation_eval.top_p="'"${generation_top_p}"'" \
      trainer.generation_eval.top_k="'"${generation_top_k}"'" \
      trainer.generation_eval.n="'"${generation_num_samples}"'" \
      trainer.generation_eval.dtype="'"${generation_dtype}"'" \
      trainer.generation_eval.backend="'"${generation_backend}"'" \
      trainer.generation_eval.vllm_gpu_memory_utilization="'"${generation_vllm_gpu_memory_utilization}"'" \
      trainer.generation_eval.vllm_host_ip="'"${generation_vllm_host_ip}"'" \
      trainer.generation_eval.vllm_enforce_eager="'"${generation_vllm_enforce_eager}"'" \
      trainer.generation_eval.vllm_sync_weights="'"${generation_vllm_sync_weights}"'" \
      trainer.generation_eval.vllm_enable_multiprocessing="'"${generation_vllm_enable_multiprocessing}"'" \
      trainer.logger="[\"console\",\"wandb\"]" \
      "$@"
  ' _ "$@" 2>&1 | tee "$TRAIN_STDOUT_LOG"
