#!/usr/bin/env bash
#SBATCH --job-name=traj_value_baseline
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --output=/data/shuozhe/verl/value_decoding/output/trajectory_value_baseline_multinode_%j/slurm.out
#SBATCH --error=/data/shuozhe/verl/value_decoding/output/trajectory_value_baseline_multinode_%j/slurm.err

set -euo pipefail

# =============================================================================
# TRAJECTORY VALUE BASELINE EVAL - SLURM MULTINODE
# Runs one independent shard per node/GPU slot, then aggregates global metrics.
# =============================================================================

# --- Checkpoints --------------------------------------------------------------
ACTOR_CHECKPOINT_DIR="/data/shuozhe/verl/train_log/job_05b_vh_init_e5_metamath/global_step_800"
CRITIC_CHECKPOINT_DIR="/data/shuozhe/verl/train_log/job_05b_vh_init_e5_metamath/global_step_800"

# --- Data ---------------------------------------------------------------------
DATASET_PATH="/data/shuozhe/saved_dataset/MetaMathQA-math-500/test.parquet"
ARCHIVE_DIR="/data/shuozhe/verl/value_decoding/output/trajectory_value_baseline_1d5_critic_multinode/${SLURM_JOB_ID:-manual}"
PROMPT_KEY="prompt"
RESPONSE_KEY="ground_truth"
START_INDEX=0
MAX_EXAMPLES=500
SHUFFLE=0
RESPONSE_BANK_PATH=""

# --- Generation ---------------------------------------------------------------
NUM_SAMPLES_PER_PROMPT=1
MAX_PROMPT_LENGTH=2048
MAX_NEW_TOKENS=2048
TEMPERATURE=1.0
TOP_P=1.0
TOP_K=0
BATCH_SIZE=1

# --- Critic readout -----------------------------------------------------------
VALUE_POSITIONS="pre_eos tail_mean_8 tail_mean_16"

# --- Runtime ------------------------------------------------------------------
DTYPE="bf16"
TRUST_REMOTE_CODE=0
MATH_DAPO_BINARY_REWARD=1
SEED=42
GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-4}"
SHARDS_PER_NODE="${GPUS_PER_NODE}"

source /data/shuozhe/miniconda3/etc/profile.d/conda.sh
conda activate verl

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
LOG_DIR="${ARCHIVE_DIR}/logs"
mkdir -p "${ARCHIVE_DIR}" "${LOG_DIR}"

validate_component_checkpoint() {
  local checkpoint_dir="$1"
  local component="$2"
  python - "$checkpoint_dir" "$component" <<'PY'
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

validate_component_checkpoint "${ACTOR_CHECKPOINT_DIR}" actor
validate_component_checkpoint "${CRITIC_CHECKPOINT_DIR}" critic

read -r -a VALUE_POSITIONS_ARR <<< "${VALUE_POSITIONS}"
if [[ ${#VALUE_POSITIONS_ARR[@]} -eq 0 ]]; then
  echo "VALUE_POSITIONS must contain at least one value." >&2
  exit 1
fi
if [[ "${SHARDS_PER_NODE}" -le 0 ]]; then
  echo "SHARDS_PER_NODE must be positive." >&2
  exit 1
fi

NODES=$(scontrol show hostnames "${SLURM_JOB_NODELIST:-$(hostname)}")
mapfile -t NODES_ARRAY <<< "${NODES}"
NUM_NODES="${#NODES_ARRAY[@]}"
TOTAL_SHARDS=$((NUM_NODES * SHARDS_PER_NODE))

if [[ "${TOTAL_SHARDS}" -le 0 ]]; then
  echo "No shards to run: NUM_NODES=${NUM_NODES}, SHARDS_PER_NODE=${SHARDS_PER_NODE}" >&2
  exit 1
fi

echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Nodes (${NUM_NODES}): ${NODES_ARRAY[*]}"
echo "SHARDS_PER_NODE=${SHARDS_PER_NODE}; TOTAL_SHARDS=${TOTAL_SHARDS}"
echo "ARCHIVE_DIR=${ARCHIVE_DIR}"

run_shard_command() {
  local value_position="$1"
  local shard_id="$2"
  local local_gpu_id="$3"
  local output_dir_for_shard="$4"

  CMD=(
    python -m value_decoding.eval_trajectory_value_baseline
    --actor_checkpoint_dir "${ACTOR_CHECKPOINT_DIR}"
    --critic_checkpoint_dir "${CRITIC_CHECKPOINT_DIR}"
    --dataset_path "${DATASET_PATH}"
    --output_dir "${output_dir_for_shard}"
    --prompt_key "${PROMPT_KEY}"
    --response_key "${RESPONSE_KEY}"
    --start_index "${START_INDEX}"
    --max_examples "${MAX_EXAMPLES}"
    --num_samples_per_prompt "${NUM_SAMPLES_PER_PROMPT}"
    --max_prompt_length "${MAX_PROMPT_LENGTH}"
    --max_new_tokens "${MAX_NEW_TOKENS}"
    --temperature "${TEMPERATURE}"
    --top_p "${TOP_P}"
    --top_k "${TOP_K}"
    --value_position "${value_position}"
    --seed "${SEED}"
    --dtype "${DTYPE}"
    --batch_size "${BATCH_SIZE}"
    --device "cuda:0"
    --critic_device "cuda:0"
    --num_shards "${TOTAL_SHARDS}"
    --shard_id "${shard_id}"
  )

  [[ -n "${RESPONSE_BANK_PATH}" ]] && CMD+=(--response_bank_path "${RESPONSE_BANK_PATH}")
  [[ "${SHUFFLE}" != "0" ]] && CMD+=(--shuffle)
  [[ "${TRUST_REMOTE_CODE}" != "0" ]] && CMD+=(--trust_remote_code)
  if [[ "${MATH_DAPO_BINARY_REWARD}" != "0" ]]; then
    CMD+=(--math_dapo_binary_reward)
  else
    CMD+=(--no-math_dapo_binary_reward)
  fi

  printf '%q ' "${CMD[@]}"
}

for value_position in "${VALUE_POSITIONS_ARR[@]}"; do
  position_dir="${ARCHIVE_DIR}/${value_position}"
  mkdir -p "${position_dir}"
  echo "Running shards for value_position=${value_position}"

  for ((node_rank = 0; node_rank < NUM_NODES; node_rank++)); do
    node_name="${NODES_ARRAY[$node_rank]}"
    for ((local_rank = 0; local_rank < SHARDS_PER_NODE; local_rank++)); do
      shard_id=$((node_rank * SHARDS_PER_NODE + local_rank))
      shard_dir="${position_dir}/shard_${shard_id}"
      shard_log="${LOG_DIR}/${value_position}_shard_${shard_id}.log"
      mkdir -p "${shard_dir}"
      shard_command="$(run_shard_command "${value_position}" "${shard_id}" "${local_rank}" "${shard_dir}")"
      echo "Launching ${value_position} shard ${shard_id}/${TOTAL_SHARDS} on ${node_name} cuda:${local_rank}"
      srun --nodes=1 --ntasks=1 -w "${node_name}" --cpus-per-task=$((SLURM_CPUS_PER_TASK / SHARDS_PER_NODE)) \
        bash -lc "export CUDA_VISIBLE_DEVICES=${local_rank}; source /data/shuozhe/miniconda3/etc/profile.d/conda.sh && conda activate verl && cd '${REPO_DIR}' && ${shard_command}" \
        > "${shard_log}" 2>&1 &
    done
  done

  wait

  aggregate_dir="${position_dir}/aggregate"
  aggregate_log="${LOG_DIR}/${value_position}_aggregate.log"
  aggregate_inputs=()
  for ((shard_id = 0; shard_id < TOTAL_SHARDS; shard_id++)); do
    shard_file="${position_dir}/shard_${shard_id}/trajectory_values.jsonl"
    if [[ ! -s "${shard_file}" ]]; then
      echo "Missing or empty shard output: ${shard_file}" >&2
      exit 1
    fi
    aggregate_inputs+=("${position_dir}/shard_${shard_id}")
  done

  echo "Aggregating value_position=${value_position} into ${aggregate_dir}"
  python -m value_decoding.eval_trajectory_value_baseline \
    --actor_checkpoint_dir "${ACTOR_CHECKPOINT_DIR}" \
    --critic_checkpoint_dir "${CRITIC_CHECKPOINT_DIR}" \
    --dataset_path "${DATASET_PATH}" \
    --output_dir "${aggregate_dir}" \
    --value_position "${value_position}" \
    --aggregate_input_dirs "${aggregate_inputs[@]}" \
    2>&1 | tee "${aggregate_log}"
done

echo "Trajectory value baseline multinode eval finished. Outputs: ${ARCHIVE_DIR}"
