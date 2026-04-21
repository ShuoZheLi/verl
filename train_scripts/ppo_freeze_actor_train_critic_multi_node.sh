#!/bin/bash
#SBATCH --job-name=ppo_metamath_multinode
#SBATCH --account=DBS24009
#SBATCH --partition=gh
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=00:10:00
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
export HYDRA_FULL_ERROR=0

# You had both 0 and 1 before; the later one wins anyway.
# Keep only one to avoid confusion.
export VLLM_USE_V1=1

export WANDB_PROJECT="PPO_midi"

echo "Activated environment"
echo "Python: $(which python3)"
echo "Ray: $(which ray)"
python3 -V

# -----------------------------
# Run identity
# -----------------------------
RUN_NAME="policy_gs800_dsk_1d5b_critic"
RUN_ID="${RUN_NAME}_${SLURM_JOB_ID}"
export SLURM_JOB_ID="${RUN_ID}"

# -----------------------------
# Training config
# -----------------------------
MATH_DAPO_BINARY_REWARD=true
POLICY_INIT_CKPT="/work2/09576/shuozhe/saved_model/Qwen2.5_7B_PPO_global_step_1000"
CRITIC_INIT_CKPT="/work2/09576/shuozhe/saved_model/DeepSeek-R1-Distill-Qwen-7B"
CRITIC_ONLY_STEPS=1000000000

TRAIN_FILE="/work2/09576/shuozhe/saved_dataset/MetaMathQA-math-500/train.parquet"
VAL_FILE="/work2/09576/shuozhe/saved_dataset/MetaMathQA-math-500/test.parquet"

WORK_DIR="/work2/09576/shuozhe/verl"
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
# Helpers
# -----------------------------
nodes_array=()

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

# -----------------------------
# Debug info
# -----------------------------
echo "Job ID: $SLURM_JOB_ID"
echo "Run ID: $RUN_ID"
echo "SLURM nodes: $SLURM_JOB_NODELIST"
echo "SCRATCH: $SCRATCH"
echo "RUN_DIR: $RUN_DIR"
echo "LOG_DIR: $LOG_DIR"

echo "Checking inputs..."
ls -ld "$WORK_DIR"
ls -ld "$POLICY_INIT_CKPT"
ls -ld "$CRITIC_INIT_CKPT"
ls -lh "$TRAIN_FILE"
ls -lh "$VAL_FILE"

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
# Run PPO training
# -----------------------------
cd "$WORK_DIR"

python3 -m verl.trainer.main_ppo \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  data.prompt_key=prompt \
  +data.response_key=ground_truth \
  data.train_batch_size=32 \
  data.max_prompt_length=2048 \
  data.max_response_length=2048 \
  actor_rollout_ref.model.path="${POLICY_INIT_CKPT}" \
  actor_rollout_ref.actor.optim.lr=0.0 \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.actor.calculate_sum_pi_squared=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
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
  critic.optim.lr=1e-5 \
  critic.model.path="${CRITIC_INIT_CKPT}" \
  critic.model.external_lib=trl \
  critic.model.value_head_init_mean=0.0 \
  critic.model.value_head_init_std=0.00001 \
  critic.model.fsdp_config.param_offload=False \
  critic.ppo_micro_batch_size_per_gpu=4 \
  +reward.reward_kwargs.math_dapo_binary_reward="${MATH_DAPO_BINARY_REWARD}" \
  trainer.resume_mode=disable \
  trainer.critic_warmup="${CRITIC_ONLY_STEPS}" \
  trainer.val_before_train=True \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes="${SLURM_JOB_NUM_NODES}" \
  trainer.test_freq=50 \
  trainer.save_freq=50 \
  trainer.total_epochs=5 \
  trainer.logger='["console","wandb"]' \
  trainer.project_name="PPO_metamath" \
  trainer.experiment_name="qwen2.5_0.5B_critic_only_from_policy_${SLURM_JOB_ID}" \
  trainer.default_local_dir="${TRAIN_LOG_DIR}" \
  2>&1 | tee "$TRAIN_STDOUT_LOG"