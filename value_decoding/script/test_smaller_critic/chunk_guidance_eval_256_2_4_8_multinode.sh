#!/bin/bash
#SBATCH --job-name=chunk_guidance_7b_seed_42
#SBATCH --account=ECS26006
#SBATCH --partition=gh
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=10:00:00
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

echo "Activated environment"
echo "Python: $(which python3)"
echo "Ray: $(which ray)"
python3 -V

# -----------------------------
# Run identity
# -----------------------------
RUN_NAME="smaller_critic_chunk_guidance_eval_256_2_4_8_multinode"
RUN_ID="${RUN_NAME}_${SLURM_JOB_ID}"

# -----------------------------
# Paths
# -----------------------------
ACTOR_CHECKPOINT_DIR="/scratch/09576/shuozhe/verl_runs/7b_actor_1d5_dsk_critic_676393/train_log/global_step_950"
CRITIC_CHECKPOINT_DIR="/scratch/09576/shuozhe/verl_runs/7b_actor_1d5_dsk_critic_676393/train_log/global_step_950"
DATASET_PATH="/work2/09576/shuozhe/saved_dataset/MetaMathQA-math-500/test.parquet"
WORK_DIR="/work2/09576/shuozhe/verl"
export PYTHONPATH="${WORK_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

ARCHIVE_ROOT="/work2/09576/shuozhe/verl/value_decoding/output_archive"
ARCHIVE_DIR="${ARCHIVE_ROOT}/${RUN_ID}"

SCRATCH_ROOT="${SCRATCH}/value_decoding_runs"
RUN_DIR="${SCRATCH_ROOT}/${RUN_ID}"
LOG_DIR="${RUN_DIR}/logs"
OUTPUT_DIR="${RUN_DIR}/chunk_guidance_eval_7b_seed_42"
ACTOR_MERGED_ROOT="${RUN_DIR}/merged_actor_hf"
CRITIC_MERGED_ROOT="${RUN_DIR}/merged_critic_hf"
# Optional override directories for HF config/tokenizer metadata used during FSDP merge.
# Set these if the raw checkpoint was copied without actor/critic/huggingface.
ACTOR_HF_SOURCE_DIR=""
CRITIC_HF_SOURCE_DIR=""

mkdir -p "$LOG_DIR" "$ARCHIVE_ROOT" "$OUTPUT_DIR"

# -----------------------------
# Chunk-guidance config
# -----------------------------
PROMPT_KEY="prompt"
RESPONSE_KEY=""
START_INDEX=0
MAX_EXAMPLES=500
SHUFFLE_EXAMPLES=0

MAX_PROMPT_LENGTH=2048
MAX_NEW_TOKENS=2048
DTYPE="bf16"

# This is the per-node local layout.
# In the current 1-GPU-per-node Slurm setup, keep this as cuda:0.
# If you later move to multi-GPU nodes, update both WORKER_PAIRS and RAY_GPUS_PER_NODE.
WORKER_PAIRS="cuda:0"

ACTOR_SAMPLING_MODE="sample"
ACTOR_TEMPERATURE=1.0
ACTOR_TOP_P=1.0
ACTOR_TOP_K=0

CHUNK_SIZES="256"
NUM_CHUNK_CANDIDATES_VALUES="4 8"
BETAS="0"
VALUE_REDUCERS="end"
INCLUDE_CRITIC_ONLY=1
INCLUDE_UNCERTAINTY_ONLY=0
ONLY_CRITIC_ONLY=1

NORMALIZATION_EPS="1e-6"
SEED="42"
SKIP_MERGE=0
DISABLE_ACTOR_CACHE=0
DEBUG_FULL_CHUNK_CANDIDATES=0
TRUST_REMOTE_CODE=0
RAY_NUM_CPUS_PER_WORKER=1
RAY_GPUS_PER_NODE=1

# -----------------------------
# Helpers
# -----------------------------
nodes_array=()
SCRIPT_PATH="${WORK_DIR}/value_decoding/script/$(basename "${BASH_SOURCE[0]}")"

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
)

checkpoint_dir = Path(sys.argv[1])
component = sys.argv[2]
component_dir = checkpoint_dir / component

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
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "ARCHIVE_DIR: $ARCHIVE_DIR"

echo "Checking inputs..."
ls -ld "$WORK_DIR"
ls -ld "$ACTOR_CHECKPOINT_DIR"
ls -ld "$CRITIC_CHECKPOINT_DIR"
ls -lh "$DATASET_PATH"
validate_component_checkpoint "$ACTOR_CHECKPOINT_DIR" actor
validate_component_checkpoint "$CRITIC_CHECKPOINT_DIR" critic

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
# Run chunk-guidance eval
# -----------------------------
cd "$WORK_DIR"

read -r -a CHUNK_SIZES_ARR <<< "$CHUNK_SIZES"
read -r -a NUM_CHUNK_CANDIDATES_VALUES_ARR <<< "$NUM_CHUNK_CANDIDATES_VALUES"
read -r -a BETAS_ARR <<< "$BETAS"
read -r -a VALUE_REDUCERS_ARR <<< "$VALUE_REDUCERS"
read -r -a WORKER_PAIRS_ARR <<< "$WORKER_PAIRS"
read -r -a SEED_ARR <<< "$SEED"

if [[ ${#SEED_ARR[@]} -eq 0 ]]; then
  echo "SEED must contain at least one value." >&2
  exit 1
fi

seed_to_id() {
  local seed_value="$1"
  local seed_id="${seed_value//-/m}"
  seed_id="${seed_id//./p}"
  printf '%s' "$seed_id"
}

run_one_seed() {
  local seed_value="$1"
  local output_dir_for_seed="$2"
  local log_path="$3"

  CMD=(
    python3 -m value_decoding.chunk_guidance_eval
    --actor_checkpoint_dir "$ACTOR_CHECKPOINT_DIR"
    --critic_checkpoint_dir "$CRITIC_CHECKPOINT_DIR"
    --dataset_path "$DATASET_PATH"
    --output_dir "$output_dir_for_seed"
    --actor_merged_root "$ACTOR_MERGED_ROOT"
    --critic_merged_root "$CRITIC_MERGED_ROOT"
    --prompt_key "$PROMPT_KEY"
    --start_index "$START_INDEX"
    --max_prompt_length "$MAX_PROMPT_LENGTH"
    --max_new_tokens "$MAX_NEW_TOKENS"
    --dtype "$DTYPE"
    --normalization_eps "$NORMALIZATION_EPS"
    --seed "$seed_value"
    --actor_sampling_mode "$ACTOR_SAMPLING_MODE"
    --actor_temperature "$ACTOR_TEMPERATURE"
    --actor_top_p "$ACTOR_TOP_P"
    --actor_top_k "$ACTOR_TOP_K"
    --chunk_sizes "${CHUNK_SIZES_ARR[@]}"
    --num_chunk_candidates_values "${NUM_CHUNK_CANDIDATES_VALUES_ARR[@]}"
    --betas "${BETAS_ARR[@]}"
    --value_reducers "${VALUE_REDUCERS_ARR[@]}"
    --ray_address auto
    --ray_num_cpus_per_worker "$RAY_NUM_CPUS_PER_WORKER"
  )

  [[ -n "$RESPONSE_KEY" ]] && CMD+=(--response_key "$RESPONSE_KEY")
  [[ -n "$MAX_EXAMPLES" ]] && CMD+=(--max_examples "$MAX_EXAMPLES")
  [[ -n "$ACTOR_HF_SOURCE_DIR" ]] && CMD+=(--actor_hf_source_dir "$ACTOR_HF_SOURCE_DIR")
  [[ -n "$CRITIC_HF_SOURCE_DIR" ]] && CMD+=(--critic_hf_source_dir "$CRITIC_HF_SOURCE_DIR")
  [[ ${#WORKER_PAIRS_ARR[@]} -gt 0 ]] && CMD+=(--worker_pairs "${WORKER_PAIRS_ARR[@]}")
  [[ "$SHUFFLE_EXAMPLES" != "0" ]] && CMD+=(--shuffle_examples)
  [[ "$SKIP_MERGE" != "0" ]] && CMD+=(--skip_merge)
  [[ "$DISABLE_ACTOR_CACHE" != "0" ]] && CMD+=(--disable_actor_cache)
  [[ "$INCLUDE_CRITIC_ONLY" != "0" ]] && CMD+=(--include_critic_only)
  [[ "$INCLUDE_UNCERTAINTY_ONLY" != "0" ]] && CMD+=(--include_uncertainty_only)
  [[ "$ONLY_CRITIC_ONLY" != "0" ]] && CMD+=(--only_critic_only)
  [[ "$DEBUG_FULL_CHUNK_CANDIDATES" != "0" ]] && CMD+=(--debug_full_chunk_candidates)
  [[ "$TRUST_REMOTE_CODE" != "0" ]] && CMD+=(--trust_remote_code)

  mkdir -p "$output_dir_for_seed"
  printf 'Running command for seed %s:\n' "$seed_value"
  printf ' %q' "${CMD[@]}"
  printf '\n'
  "${CMD[@]}" 2>&1 | tee "$log_path"
}

if [[ ${#SEED_ARR[@]} -eq 1 ]]; then
  run_one_seed "${SEED_ARR[0]}" "$OUTPUT_DIR" "$LOG_DIR/chunk_guidance_eval.log"
  echo "Chunk-guidance eval finished successfully."
  exit 0
fi

SUMMARY_PATHS=()
SEED_OUTPUT_DIRS=()
for seed_value in "${SEED_ARR[@]}"; do
  seed_id="$(seed_to_id "$seed_value")"
  seed_output_dir="${OUTPUT_DIR}/seed_${seed_id}"
  seed_log_path="$LOG_DIR/chunk_guidance_eval__seed_${seed_id}.log"
  run_one_seed "$seed_value" "$seed_output_dir" "$seed_log_path"
  SUMMARY_PATHS+=("${seed_output_dir}/summary_metrics.json")
  SEED_OUTPUT_DIRS+=("${seed_output_dir}")
done

python3 -m value_decoding.multi_seed_summary \
  --output_path "${OUTPUT_DIR}/summary_metrics.json" \
  --source_script "$SCRIPT_PATH" \
  --seed_values "${SEED_ARR[@]}" \
  --summary_paths "${SUMMARY_PATHS[@]}" \
  --seed_output_dirs "${SEED_OUTPUT_DIRS[@]}"

echo "Chunk-guidance eval finished successfully."
