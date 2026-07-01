#!/bin/bash
#SBATCH --job-name=grpo_qwen3_8b_wanda_d1
#SBATCH --account=ASC24079
#SBATCH --partition=gh-dev
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=1:00:00
#SBATCH --output=slurm-%j_grpo_qwen3_8b_wanda_d1.out
#SBATCH --error=slurm-%j_grpo_qwen3_8b_wanda_d1.err

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
export HYDRA_FULL_ERROR=0

# You had both 0 and 1 before; the later one wins anyway.
# Keep only one to avoid confusion.
export VLLM_USE_V1=1

export WANDB_PROJECT="prune_for_post_train"
WANDB_PROJECT="prune_for_post_train"

echo "Activated environment"
echo "Python: $(which python3)"
echo "Ray: $(which ray)"
python3 -V

# -----------------------------
# Run identity
# -----------------------------
RUN_NAME="grpo_qwen3_8b_wanda_sparse_kept_sparsity_d1"
REAL_SLURM_JOB_ID="${SLURM_JOB_ID}"
RUN_ID="${RUN_NAME}_${REAL_SLURM_JOB_ID}"

# -----------------------------
# Training config
# -----------------------------
# When true, math_dapo incorrect answers get reward 0.0 instead of -1.0.
MATH_DAPO_BINARY_REWARD=true
WORK_DIR="${WORK_DIR:-/work2/09576/shuozhe/verl}"
POLICY_INIT_CKPT="${POLICY_INIT_CKPT:-/work2/09576/shuozhe/saved_model/Qwen3-8B-Base}"
TRAIN_FILE="${TRAIN_FILE:-/work2/09576/shuozhe/saved_dataset/MetaMathQA-math-500/train.parquet}"
VAL_FILE="${VAL_FILE:-/work2/09576/shuozhe/saved_dataset/MetaMathQA-math-500/test.parquet}"

# Allow the same script to be syntax-tested or dry-run locally under /data/shuozhe.
if [[ ! -d "$WORK_DIR" && -d "/work2/09576/shuozhe/verl" ]]; then
  WORK_DIR="/work2/09576/shuozhe/verl"
fi
if [[ ! -d "$POLICY_INIT_CKPT" && -d "/work2/09576/shuozhe/saved_model/Qwen3-8B-Base" ]]; then
  POLICY_INIT_CKPT="/work2/09576/shuozhe/saved_model/Qwen3-8B-Base"
fi
if [[ ! -f "$TRAIN_FILE" && -f TRAIN_FILE="/work2/09576/shuozhe/saved_dataset/MetaMathQA-math-500/math7500.parquet" ]]; then
  TRAIN_FILE=TRAIN_FILE="/work2/09576/shuozhe/saved_dataset/MetaMathQA-math-500/math7500.parquet"
fi
if [[ ! -f "$VAL_FILE" && -f "/work2/09576/shuozhe/saved_dataset/MetaMathQA-math-500/test.parquet" ]]; then
  VAL_FILE="/work2/09576/shuozhe/saved_dataset/MetaMathQA-math-500/test.parquet"
fi
export PYTHONPATH="${WORK_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

# -----------------------------
# Output paths
# -----------------------------
SCRATCH_ROOT="${SCRATCH}/verl_runs"
RUN_DIR="${SCRATCH_ROOT}/${RUN_ID}"
LOG_DIR="${RUN_DIR}/logs"
TRAIN_LOG_DIR="${RUN_DIR}/train_log"
ARCHIVE_ROOT="/work2/09576/shuozhe/verl/train_log_archive"
ARCHIVE_DIR="${ARCHIVE_ROOT}/${RUN_ID}"

mkdir -p "$LOG_DIR" "$TRAIN_LOG_DIR" "$ARCHIVE_ROOT"

TRAIN_STDOUT_LOG="${TRAIN_LOG_DIR}/job_${RUN_ID}.txt"

# -----------------------------
# WANDA sparse update config
# -----------------------------
# Sparse-update mask semantics in VERL: True means trainable/kept, False means frozen.
# WANDA score semantics: higher score means more important to keep.
# Therefore, sparsity=s trains exactly the top (1-s) WANDA-score entries per target linear weight.
SPARSE_UPDATE_MODE="wanda_top"
SPARSE_UPDATE_SPARSITY="${SPARSE_UPDATE_SPARSITY:-0.1}"
SPARSE_UPDATE_KEEP_FRACTION="${SPARSE_UPDATE_KEEP_FRACTION:-}"
WANDA_SCORE_DIR="${WANDA_SCORE_DIR:-/work2/09576/shuozhe/saved_score/qwen3_8b_base_wanda_scores}"
if [[ ! -d "$WANDA_SCORE_DIR" && -d "/work2/09576/shuozhe/saved_score/qwen3_8b_base_wanda_scores" ]]; then
  WANDA_SCORE_DIR="/work2/09576/shuozhe/saved_score/qwen3_8b_base_wanda_scores"
fi
SPARSE_UPDATE_MASK_PATH="${SPARSE_UPDATE_MASK_PATH:-${TRAIN_LOG_DIR}/wanda_top_sparsity_${SPARSE_UPDATE_SPARSITY}_mask.pt}"
SPARSE_UPDATE_VERIFY="${SPARSE_UPDATE_VERIFY:-true}"
SPARSE_UPDATE_VERIFY_INTERVAL="${SPARSE_UPDATE_VERIFY_INTERVAL:-200}"
# false: dense forward with frozen entries restored to pretrained values.
# true: actually pruned forward/training with frozen entries set to and kept at zero.
SPARSE_UPDATE_ZERO_FROZEN_PARAMS="${SPARSE_UPDATE_ZERO_FROZEN_PARAMS:-true}"
SPARSE_UPDATE_TARGET_MODULES="${SPARSE_UPDATE_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}"
SPARSE_UPDATE_EXCLUDE_KEYWORDS="${SPARSE_UPDATE_EXCLUDE_KEYWORDS:-embed,lm_head,norm,layernorm,rmsnorm}"

SPARSE_UPDATE_MASK_OVERRIDES=(
  actor_rollout_ref.actor.sparse_update.enabled=true
  actor_rollout_ref.actor.sparse_update.mode="${SPARSE_UPDATE_MODE}"
  actor_rollout_ref.actor.sparse_update.mask_path="${SPARSE_UPDATE_MASK_PATH}"
  actor_rollout_ref.actor.sparse_update.build_mask_on_init=false
  actor_rollout_ref.actor.sparse_update.restore_frozen_after_step=true
  actor_rollout_ref.actor.sparse_update.zero_frozen_params="${SPARSE_UPDATE_ZERO_FROZEN_PARAMS}"
  actor_rollout_ref.actor.sparse_update.mask_optimizer_state=true
  actor_rollout_ref.actor.sparse_update.verify_frozen_weights="${SPARSE_UPDATE_VERIFY}"
  actor_rollout_ref.actor.sparse_update.verification_interval="${SPARSE_UPDATE_VERIFY_INTERVAL}"
  actor_rollout_ref.actor.fsdp_config.use_orig_params=True
)

# -----------------------------
# Helpers
# -----------------------------
nodes_array=()
MODEL_PATH_RESOLVER="${WORK_DIR}/tools/resolve_model_init_path.py"

resolve_model_init_path() {
  local raw_path="$1"
  local role="$2"

  if [[ ! -f "$MODEL_PATH_RESOLVER" ]]; then
    echo "Missing model path resolver: $MODEL_PATH_RESOLVER" >&2
    return 1
  fi

  python3 "$MODEL_PATH_RESOLVER" \
    --path "$raw_path" \
    --role "$role" \
    --log-dir "$LOG_DIR"
}

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

stop_ray_all_nodes() {
  if [[ ${#nodes_array[@]} -eq 0 ]]; then
    return 0
  fi
  for node in "${nodes_array[@]}"; do
    srun --nodes=1 --ntasks=1 -w "$node" \
      bash -c "source '${VENV}/bin/activate' && ray stop --force || true" \
      >> "$LOG_DIR/ray_stop_${node}.log" 2>&1 || true
  done
}

count_alive_ray_nodes() {
  local ray_address="$1"
  python3 - "$ray_address" <<'PY'
import logging
import sys

address = sys.argv[1]

import ray

try:
    ray.init(address=address, logging_level=logging.ERROR)
    alive_nodes = sum(1 for node in ray.nodes() if node.get("Alive"))
    print(alive_nodes)
finally:
    if ray.is_initialized():
        ray.shutdown()
PY
}

cleanup() {
  echo "Stopping Ray on all nodes..."
  stop_ray_all_nodes || true
  sync_to_work
}
trap cleanup EXIT

POLICY_MODEL_PATH="$(resolve_model_init_path "$POLICY_INIT_CKPT" actor)"

# -----------------------------
# Build WANDA kept-parameter mask
# -----------------------------
# Build the mask before Ray startup so failures happen early and deterministically.
mkdir -p "$(dirname "$SPARSE_UPDATE_MASK_PATH")"
if [[ ! -f "$SPARSE_UPDATE_MASK_PATH" ]]; then
  echo "Building WANDA sparse-update mask at $SPARSE_UPDATE_MASK_PATH"
  mask_args=(
    --model_name_or_path "$POLICY_MODEL_PATH"
    --output_path "$SPARSE_UPDATE_MASK_PATH"
    --wanda_score_dir "$WANDA_SCORE_DIR"
    --target_modules "$SPARSE_UPDATE_TARGET_MODULES"
    --exclude_keywords "$SPARSE_UPDATE_EXCLUDE_KEYWORDS"
  )
  if [[ -n "$SPARSE_UPDATE_KEEP_FRACTION" ]]; then
    mask_args+=(--keep_fraction "$SPARSE_UPDATE_KEEP_FRACTION")
  else
    mask_args+=(--sparsity "$SPARSE_UPDATE_SPARSITY")
  fi
  python3 "$WORK_DIR/tools/build_sparse_update_mask.py" "${mask_args[@]}" | tee "$LOG_DIR/build_wanda_sparse_update_mask.log"
else
  echo "Using existing WANDA sparse-update mask: $SPARSE_UPDATE_MASK_PATH"
fi

# -----------------------------
# Debug info
# -----------------------------
echo "Job ID: $SLURM_JOB_ID"
echo "Run ID: $RUN_ID"
echo "SLURM nodes: $SLURM_JOB_NODELIST"
echo "SCRATCH: $SCRATCH"
echo "RUN_DIR: $RUN_DIR"
echo "LOG_DIR: $LOG_DIR"
describe_path "POLICY_INIT_CKPT" "$POLICY_INIT_CKPT"
describe_path "POLICY_MODEL_PATH" "$POLICY_MODEL_PATH"
echo "SPARSE_UPDATE_MODE: $SPARSE_UPDATE_MODE"
echo "SPARSE_UPDATE_SPARSITY: $SPARSE_UPDATE_SPARSITY"
echo "SPARSE_UPDATE_KEEP_FRACTION: ${SPARSE_UPDATE_KEEP_FRACTION:-1 - sparsity}"
echo "SPARSE_UPDATE_ZERO_FROZEN_PARAMS: $SPARSE_UPDATE_ZERO_FROZEN_PARAMS"
echo "WANDA_SCORE_DIR: $WANDA_SCORE_DIR"
echo "SPARSE_UPDATE_MASK_PATH: $SPARSE_UPDATE_MASK_PATH"

echo "Checking inputs..."
ls -ld "$WORK_DIR"
ls -lh "$TRAIN_FILE"
ls -lh "$VAL_FILE"
ls -ld "$WANDA_SCORE_DIR"
if [[ ! -f "$WANDA_SCORE_DIR/metadata.json" ]]; then
  echo "Missing WANDA score metadata: $WANDA_SCORE_DIR/metadata.json" >&2
  exit 1
fi

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
  echo "Failed to resolve a usable IP address for Ray head node $head_node." >&2
  exit 1
fi
head_node_ip="$resolved_head_node_ip"

port=6379
ip_head="${head_node_ip}:${port}"
export RAY_ADDRESS="$ip_head"

echo "Head node: $head_node"
echo "Head IP: $ip_head"

# -----------------------------
# Ray cluster config
# -----------------------------
# This version assumes 1 GPU per node, matching your example multi-node script.
# If you really have 4 GPUs per node available to the job, see notes below.
RAY_GPUS_PER_NODE=1

# -----------------------------
# Start Ray head
# -----------------------------
echo "Starting Ray head..."
srun --nodes=1 --ntasks=1 -w "$head_node" \
  bash -c "source '${VENV}/bin/activate' && \
           ray start --head \
           --node-ip-address='${head_node_ip}' \
           --port='${port}' \
           --num-cpus='${SLURM_CPUS_PER_TASK}' \
           --num-gpus='${RAY_GPUS_PER_NODE}'" \
  > "$LOG_DIR/ray_head.log" 2>&1 &

sleep 10

echo "Waiting for Ray head..."
head_ready=0
for i in {1..30}; do
  if ray status --address="$ip_head" > /dev/null 2>&1; then
    echo "Ray head is ready."
    head_ready=1
    break
  fi
  sleep 2
done
if [[ "$head_ready" != "1" ]]; then
  echo "Ray head failed to become ready at $ip_head." >&2
  echo "See $LOG_DIR/ray_head.log for details." >&2
  exit 1
fi

# -----------------------------
# Start Ray workers
# -----------------------------
worker_num=$((SLURM_JOB_NUM_NODES - 1))
for ((i = 1; i <= worker_num; i++)); do
  node_i="${nodes_array[$i]}"
  echo "Starting worker on $node_i"

  srun --nodes=1 --ntasks=1 -w "$node_i" \
    bash -c "source '${VENV}/bin/activate' && \
             ray start --address '${ip_head}' \
             --num-cpus='${SLURM_CPUS_PER_TASK}' \
             --num-gpus='${RAY_GPUS_PER_NODE}'" \
    > "$LOG_DIR/ray_worker_${i}.log" 2>&1 &
done

wait

echo "Waiting for all Ray nodes to register..."
alive_nodes=0
all_nodes_ready=0
RAY_NODE_PROBE_LOG="$LOG_DIR/ray_node_probe.log"
: > "$RAY_NODE_PROBE_LOG"

for i in {1..30}; do
  if alive_nodes="$(count_alive_ray_nodes "$ip_head" 2>>"$RAY_NODE_PROBE_LOG")"; then
    :
  else
    alive_nodes=0
  fi

  if [[ "$alive_nodes" -ge "$SLURM_JOB_NUM_NODES" ]]; then
    echo "All $alive_nodes Ray nodes are registered."
    all_nodes_ready=1
    break
  fi

  echo "Currently registered Ray nodes: ${alive_nodes}/${SLURM_JOB_NUM_NODES}"
  sleep 5
done

if [[ "$all_nodes_ready" != "1" ]]; then
  echo "Expected $SLURM_JOB_NUM_NODES Ray nodes, but only $alive_nodes registered." >&2
  if [[ -s "$RAY_NODE_PROBE_LOG" ]]; then
    echo "Recent Ray probe errors:" >&2
    tail -n 20 "$RAY_NODE_PROBE_LOG" >&2 || true
  fi
  ray status --address="$ip_head" || true
  exit 1
fi

echo "Ray cluster status:"
ray status --address="$ip_head" || true

# -----------------------------
# Run GRPO training
# -----------------------------
cd "$WORK_DIR"

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  data.prompt_key=prompt \
  data.train_batch_size=32 \
  data.max_prompt_length=2048 \
  data.max_response_length=2048 \
  actor_rollout_ref.model.path="$POLICY_MODEL_PATH" \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.actor.calculate_sum_pi_squared=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  "${SPARSE_UPDATE_MASK_OVERRIDES[@]}" \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
  actor_rollout_ref.hybrid_engine=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  +reward.reward_kwargs.math_dapo_binary_reward="${MATH_DAPO_BINARY_REWARD}" \
  critic.enable=False \
  trainer.critic_warmup=0 \
  trainer.val_before_train=true \
  trainer.n_gpus_per_node="${RAY_GPUS_PER_NODE}" \
  trainer.nnodes="${SLURM_JOB_NUM_NODES}" \
  trainer.test_freq=50 \
  trainer.save_freq=50 \
  trainer.total_epochs=5 \
  trainer.logger='["console","wandb"]' \
  trainer.project_name="${WANDB_PROJECT}" \
  trainer.experiment_name="${RUN_ID}" \
  trainer.default_local_dir="${TRAIN_LOG_DIR}" \
  "$@" \
  2>&1 | tee "$TRAIN_STDOUT_LOG"
