#!/bin/bash
#SBATCH --job-name=actor_greedy_eval
#SBATCH --account=ECS26006
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=01:00:00
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
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

echo "Activated environment"
echo "Python: $(which python3)"
echo "Ray: $(which ray)"
python3 -V

# -----------------------------
# Run identity
# -----------------------------
RUN_NAME="actor_greedy_eval"
RUN_ID="${RUN_NAME}_${SLURM_JOB_ID}"

# -----------------------------
# Paths
# -----------------------------
ACTOR_CHECKPOINT_DIRS=(
  "/work2/09576/shuozhe/saved_model/Prathyusha101/Qwen2.5_7b_100_ckpt_global_step_100"
  "/work2/09576/shuozhe/saved_model/Prathyusha101/Qwen2.5_7b_750_ckpt_global_step_750"
)
ACTOR_NAMES=(
  "Qwen2.5_7b_step_100"
  "Qwen2.5_7b_step_750"
)

DATASET_PATH="/work2/09576/shuozhe/saved_dataset/MetaMathQA-math-500/gsm8k_test.parquet"
WORK_DIR="/work2/09576/shuozhe/verl"
export PYTHONPATH="${WORK_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

ARCHIVE_ROOT="/work2/09576/shuozhe/verl/value_decoding/output_archive"
ARCHIVE_DIR="${ARCHIVE_ROOT}/${RUN_ID}"

SCRATCH_ROOT="${SCRATCH}/value_decoding_runs"
RUN_DIR="${SCRATCH_ROOT}/${RUN_ID}"
LOG_DIR="${RUN_DIR}/logs"
OUTPUT_DIR="${RUN_DIR}/actor_greedy_eval"

mkdir -p "$LOG_DIR" "$ARCHIVE_ROOT" "$OUTPUT_DIR"

# If you leave ACTOR_MERGED_ROOTS empty, the script will create one merged root
# per actor under the run directory so merges do not collide.
ACTOR_MERGED_ROOTS=()
ACTOR_HF_SOURCE_DIRS=()

# -----------------------------
# Evaluation config
# -----------------------------
PROMPT_KEY="prompt"
RESPONSE_KEY=""
START_INDEX=0
MAX_EXAMPLES=500
SHUFFLE_EXAMPLES=0

MAX_PROMPT_LENGTH=2048
MAX_NEW_TOKENS=2048
DTYPE="bf16"
SEED=42

# Per-node local worker layout. In the current 1-GPU-per-node Slurm setup, keep
# this as `cuda:0`. For multi-GPU nodes, set e.g. `cuda:0 cuda:1` and update
# RAY_GPUS_PER_NODE.
WORKER_DEVICES="cuda:0"

OMIT_PROMPT_TEXT=0
OMIT_RESPONSE_TOKEN_IDS=0

TRUST_REMOTE_CODE=0
SKIP_MERGE=0
DISABLE_ACTOR_CACHE=0
RAY_NUM_CPUS_PER_WORKER=1
RAY_GPUS_PER_NODE=1

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
# Validate config
# -----------------------------
if [[ ${#ACTOR_CHECKPOINT_DIRS[@]} -ne ${#ACTOR_NAMES[@]} ]]; then
  echo "ACTOR_CHECKPOINT_DIRS and ACTOR_NAMES must have the same length." >&2
  exit 1
fi

if [[ ${#ACTOR_HF_SOURCE_DIRS[@]} -gt 0 && ${#ACTOR_HF_SOURCE_DIRS[@]} -ne ${#ACTOR_CHECKPOINT_DIRS[@]} ]]; then
  echo "If ACTOR_HF_SOURCE_DIRS is set, it must have one entry per actor." >&2
  exit 1
fi

if [[ ${#ACTOR_MERGED_ROOTS[@]} -eq 0 ]]; then
  for actor_name in "${ACTOR_NAMES[@]}"; do
    ACTOR_MERGED_ROOTS+=("${RUN_DIR}/merged_actor_hf/${actor_name}")
  done
fi

if [[ ${#ACTOR_MERGED_ROOTS[@]} -ne ${#ACTOR_CHECKPOINT_DIRS[@]} ]]; then
  echo "ACTOR_MERGED_ROOTS must have one entry per actor." >&2
  exit 1
fi

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
ls -lh "$DATASET_PATH"
for actor_dir in "${ACTOR_CHECKPOINT_DIRS[@]}"; do
  ls -ld "$actor_dir"
  validate_component_checkpoint "$actor_dir" actor
done

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
# Run evaluation
# -----------------------------
cd "$WORK_DIR"

read -r -a WORKER_DEVICES_ARR <<< "$WORKER_DEVICES"

for actor_index in "${!ACTOR_CHECKPOINT_DIRS[@]}"; do
  actor_checkpoint_dir="${ACTOR_CHECKPOINT_DIRS[$actor_index]}"
  actor_name="${ACTOR_NAMES[$actor_index]}"
  actor_output_dir="${OUTPUT_DIR}/${actor_name}"
  actor_log_path="${LOG_DIR}/actor_greedy_eval_${actor_name}.log"
  mkdir -p "$actor_output_dir"

  CMD=(
    python3 -m value_decoding.actor_greedy_eval
    --actor_checkpoint_dir "$actor_checkpoint_dir"
    --actor_name "$actor_name"
    --dataset_path "$DATASET_PATH"
    --output_dir "$actor_output_dir"
    --prompt_key "$PROMPT_KEY"
    --start_index "$START_INDEX"
    --max_examples "$MAX_EXAMPLES"
    --max_prompt_length "$MAX_PROMPT_LENGTH"
    --max_new_tokens "$MAX_NEW_TOKENS"
    --dtype "$DTYPE"
    --seed "$SEED"
    --actor_merged_root "${ACTOR_MERGED_ROOTS[$actor_index]}"
    --ray_address auto
    --ray_num_cpus_per_worker "$RAY_NUM_CPUS_PER_WORKER"
  )

  [[ -n "$RESPONSE_KEY" ]] && CMD+=(--response_key "$RESPONSE_KEY")
  [[ ${#ACTOR_HF_SOURCE_DIRS[@]} -gt 0 && -n "${ACTOR_HF_SOURCE_DIRS[$actor_index]}" ]] && \
    CMD+=(--actor_hf_source_dir "${ACTOR_HF_SOURCE_DIRS[$actor_index]}")
  [[ ${#WORKER_DEVICES_ARR[@]} -gt 0 ]] && CMD+=(--worker_devices "${WORKER_DEVICES_ARR[@]}")
  [[ "$SHUFFLE_EXAMPLES" != "0" ]] && CMD+=(--shuffle_examples)
  [[ "$TRUST_REMOTE_CODE" != "0" ]] && CMD+=(--trust_remote_code)
  [[ "$SKIP_MERGE" != "0" ]] && CMD+=(--skip_merge)
  [[ "$OMIT_PROMPT_TEXT" != "0" ]] && CMD+=(--omit_prompt_text)
  [[ "$OMIT_RESPONSE_TOKEN_IDS" != "0" ]] && CMD+=(--omit_response_token_ids)
  [[ "$DISABLE_ACTOR_CACHE" != "0" ]] && CMD+=(--disable_actor_cache)

  printf 'Running greedy evaluation for actor %s (%s/%s):\n' "$actor_name" "$((actor_index + 1))" "${#ACTOR_CHECKPOINT_DIRS[@]}"
  printf ' %q' "${CMD[@]}"
  printf '\n'
  "${CMD[@]}" 2>&1 | tee "$actor_log_path"
done

python3 - "$OUTPUT_DIR" "${ACTOR_NAMES[@]}" <<'PY'
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
actor_names = sys.argv[2:]
summary = {}
for actor_name in actor_names:
    summary_path = output_dir / actor_name / "summary_metrics.json"
    if summary_path.exists():
        summary[actor_name] = json.loads(summary_path.read_text(encoding="utf-8"))
(output_dir / "summary_metrics.json").write_text(
    json.dumps({"actor_metrics": summary}, ensure_ascii=True, indent=2),
    encoding="utf-8",
)
PY

echo "Actor greedy evaluation finished successfully."
