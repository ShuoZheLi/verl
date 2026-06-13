#!/bin/bash
#SBATCH --job-name=ppo_metamath_multinode
#SBATCH --account=ECS26006
#SBATCH --partition=gh
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=10:00:00
#SBATCH --output=slurm-%j_eval_trajectory_value_baseline_critic_multinode_1d5b_actor.out
#SBATCH --error=slurm-%j_eval_trajectory_value_baseline_critic_multinode_1d5b_actor.err

set -euo pipefail

# -----------------------------
# Cluster environment setup
# -----------------------------
module reset
module load nvidia/25.9

VENV="/work/09576/shuozhe/verl_setup_tacc/.venv"
source "${VENV}/bin/activate"

WORK_DIR="/work2/09576/shuozhe/verl"
export PYTHONPATH="${WORK_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

UV_CACHE_DIR="${SCRATCH}/.cache/uv"
HF_HOME="${SCRATCH}/.cache/huggingface"
TIKTOKEN_ENCODINGS_BASE="${SCRATCH}/data/embeddings"
mkdir -p "${UV_CACHE_DIR}" "${HF_HOME}" "${TIKTOKEN_ENCODINGS_BASE}"
export UV_CACHE_DIR HF_HOME TIKTOKEN_ENCODINGS_BASE

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=true
export VLLM_USE_V1=0
unset TORCH_LOGS
unset TORCH_LOGS_OUT
export TORCHDYNAMO_VERBOSE=0
export PYTHONWARNINGS="ignore::FutureWarning"

echo "Activated environment"
echo "Python: $(which python)"
python -V

# =============================================================================
# TRAJECTORY VALUE BASELINE EVAL - SLURM MULTINODE
# Runs one independent shard per node/GPU slot, then aggregates global metrics.
# =============================================================================

# --- Checkpoints --------------------------------------------------------------
CHECKPOINT_ROOT="/scratch/09576/shuozhe/verl_runs/7b_testset_752950/train_log/"
ACTOR_CHECKPOINT_DIR="${CHECKPOINT_ROOT}/global_step_100"
CRITIC_CHECKPOINT_DIRS=""
for step in $(seq 100 100 1000); do
  CRITIC_CHECKPOINT_DIRS+=" ${CHECKPOINT_ROOT}/global_step_${step}"
done
CRITIC_CHECKPOINT_DIRS="${CRITIC_CHECKPOINT_DIRS# }"

# --- Data ---------------------------------------------------------------------
DATASET_PATH="/work2/09576/shuozhe/saved_dataset/MetaMathQA-math-500/test.parquet"
ARCHIVE_DIR="/work2/09576/shuozhe/verl/value_decoding/output/trajectory_value_baseline_7b_testset_752950/${SLURM_JOB_ID:-manual}"
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
BATCH_SIZE=128
GENERATION_BACKEND="torch"  # vllm accelerates actor generation; use "torch" to disable.
VLLM_GPU_MEMORY_UTILIZATION=0.45
VLLM_TENSOR_PARALLEL_SIZE=1
VLLM_MAX_NUM_SEQS=""
VLLM_ENFORCE_EAGER=0

# --- Critic readout -----------------------------------------------------------
VALUE_POSITIONS="pre_eos"

# --- Runtime ------------------------------------------------------------------
DTYPE="bf16"
TRUST_REMOTE_CODE=0
MATH_DAPO_BINARY_REWARD=1
SEED=42
GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-1}"
# Run one shard per node by default, matching the training scripts
# that expose one GPU per Ray worker. Increase only if your Slurm
# allocation actually provides multiple visible GPUs per node.
SHARDS_PER_NODE="${SHARDS_PER_NODE:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${WORK_DIR}"
LOG_DIR="${ARCHIVE_DIR}/logs"
MERGE_TMP_ROOT="${SCRATCH}/verl_eval_merge_cache/${SLURM_JOB_ID:-manual}"
mkdir -p "${ARCHIVE_DIR}" "${LOG_DIR}" "${MERGE_TMP_ROOT}"

cleanup_merge_cache() {
  if [[ -n "${MERGE_TMP_ROOT:-}" && -d "${MERGE_TMP_ROOT}" ]]; then
    echo "Removing temporary merge cache: ${MERGE_TMP_ROOT}"
    rm -rf "${MERGE_TMP_ROOT}" || true
  fi
}
trap cleanup_merge_cache EXIT


describe_path() {
  local label="$1"
  local path="$2"
  echo "$label: $path"
  if [[ -d "$path" || -f "$path" ]]; then
    ls -ld "$path"
  else
    echo "  local path not found"
  fi
}

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


echo "Checking cluster paths..."
describe_path "WORK_DIR" "${WORK_DIR}"
describe_path "DATASET_PATH" "${DATASET_PATH}"
describe_path "ACTOR_CHECKPOINT_DIR" "${ACTOR_CHECKPOINT_DIR}"
validate_component_checkpoint "${ACTOR_CHECKPOINT_DIR}" actor
read -r -a CRITIC_CHECKPOINT_DIRS_ARR <<< "${CRITIC_CHECKPOINT_DIRS}"
if [[ ${#CRITIC_CHECKPOINT_DIRS_ARR[@]} -eq 0 ]]; then
  echo "CRITIC_CHECKPOINT_DIRS must contain at least one checkpoint." >&2
  exit 1
fi
for critic_checkpoint_dir in "${CRITIC_CHECKPOINT_DIRS_ARR[@]}"; do
  validate_component_checkpoint "${critic_checkpoint_dir}" critic
done

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
  local shard_id="$1"
  local local_gpu_id="$2"
  local output_dir_for_shard="$3"

  CMD=(
    python -m value_decoding.eval_trajectory_value_baseline
    --actor_checkpoint_dir "${ACTOR_CHECKPOINT_DIR}"
    --critic_checkpoint_dir "${CRITIC_CHECKPOINT_DIRS_ARR[@]}"
    --dataset_path "${DATASET_PATH}"
    --output_dir "${output_dir_for_shard}"
    --actor_merged_root "${MERGE_TMP_ROOT}/shard_${shard_id}/actor"
    --critic_merged_root "${MERGE_TMP_ROOT}/shard_${shard_id}/critics"
    --delete_merged_critics_after_load
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
    --value_position "${VALUE_POSITIONS_ARR[@]}"
    --seed "${SEED}"
    --dtype "${DTYPE}"
    --batch_size "${BATCH_SIZE}"
    --generation_backend "${GENERATION_BACKEND}"
    --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}"
    --vllm_tensor_parallel_size "${VLLM_TENSOR_PARALLEL_SIZE}"
    --device "cuda:0"
    --critic_device "cuda:0"
    --num_shards "${TOTAL_SHARDS}"
    --shard_id "${shard_id}"
    --always_write_run_subdirs
  )

  [[ -n "${RESPONSE_BANK_PATH}" ]] && CMD+=(--response_bank_path "${RESPONSE_BANK_PATH}")
  [[ -n "${VLLM_MAX_NUM_SEQS}" ]] && CMD+=(--vllm_max_num_seqs "${VLLM_MAX_NUM_SEQS}")
  [[ "${VLLM_ENFORCE_EAGER}" != "0" ]] && CMD+=(--vllm_enforce_eager)
  [[ "${VLLM_ENFORCE_EAGER}" == "0" ]] && CMD+=(--no-vllm_enforce_eager)
  [[ "${SHUFFLE}" != "0" ]] && CMD+=(--shuffle)
  [[ "${TRUST_REMOTE_CODE}" != "0" ]] && CMD+=(--trust_remote_code)
  if [[ "${MATH_DAPO_BINARY_REWARD}" != "0" ]]; then
    CMD+=(--math_dapo_binary_reward)
  else
    CMD+=(--no-math_dapo_binary_reward)
  fi

  printf '%q ' "${CMD[@]}"
}

position_dir="${ARCHIVE_DIR}/all_values"
mkdir -p "${position_dir}"
echo "Running shards for VALUE_POSITIONS=${VALUE_POSITIONS} and ${#CRITIC_CHECKPOINT_DIRS_ARR[@]} critic(s)"

SHARD_PIDS=()
SHARD_LOGS=()

for ((node_rank = 0; node_rank < NUM_NODES; node_rank++)); do
  node_name="${NODES_ARRAY[$node_rank]}"
  for ((local_rank = 0; local_rank < SHARDS_PER_NODE; local_rank++)); do
    shard_id=$((node_rank * SHARDS_PER_NODE + local_rank))
    shard_dir="${position_dir}/shard_${shard_id}"
    shard_log="${LOG_DIR}/shard_${shard_id}.log"
    mkdir -p "${shard_dir}"
    shard_command="$(run_shard_command "${shard_id}" "${local_rank}" "${shard_dir}")"
    echo "Launching shard ${shard_id}/${TOTAL_SHARDS} on ${node_name} cuda:${local_rank}"
    SRUN_GPU_ARGS=()
    if [[ -n "${SLURM_GPUS_ON_NODE:-}" ]]; then
      SRUN_GPU_ARGS+=(--gres="gpu:1")
    fi
    srun --nodes=1 --ntasks=1 -w "${node_name}" --cpus-per-task=$((SLURM_CPUS_PER_TASK / SHARDS_PER_NODE)) "${SRUN_GPU_ARGS[@]}" \
      bash -lc "export CUDA_VISIBLE_DEVICES=${local_rank}; source '${VENV}/bin/activate' && cd '${REPO_DIR}' && ${shard_command}" \
      > "${shard_log}" 2>&1 &
    SHARD_PIDS+=("$!")
    SHARD_LOGS+=("${shard_log}")
  done
done

shard_failures=0
for shard_index in "${!SHARD_PIDS[@]}"; do
  if ! wait "${SHARD_PIDS[$shard_index]}"; then
    shard_failures=$((shard_failures + 1))
    echo "Shard process failed; log: ${SHARD_LOGS[$shard_index]}" >&2
    tail -n 40 "${SHARD_LOGS[$shard_index]}" >&2 || true
  fi
done
if [[ "${shard_failures}" -ne 0 ]]; then
  echo "${shard_failures} shard process(es) failed. See ${LOG_DIR}." >&2
  exit 1
fi

aggregate_inputs=()
for ((shard_id = 0; shard_id < TOTAL_SHARDS; shard_id++)); do
  shard_dir="${position_dir}/shard_${shard_id}"
  if [[ ! -s "${shard_dir}/trajectory_bank.jsonl" ]]; then
    echo "Missing or empty shard trajectory bank: ${shard_dir}/trajectory_bank.jsonl" >&2
    shard_log="${LOG_DIR}/shard_${shard_id}.log"
    if [[ -f "${shard_log}" ]]; then
      echo "Last 120 lines from ${shard_log}:" >&2
      tail -n 60 "${shard_log}" >&2 || true
    fi
    exit 1
  fi
  aggregate_inputs+=("${shard_dir}")
done

# Aggregate each critic/value-position run separately.
for critic_label_dir in "${position_dir}/shard_0"/*; do
  [[ -d "${critic_label_dir}" ]] || continue
  critic_label="$(basename "${critic_label_dir}")"
  for value_position in "${VALUE_POSITIONS_ARR[@]}"; do
    per_run_inputs=()
    for ((shard_id = 0; shard_id < TOTAL_SHARDS; shard_id++)); do
      shard_run_dir="${position_dir}/shard_${shard_id}/${critic_label}/${value_position}"
      if [[ ! -s "${shard_run_dir}/trajectory_values.jsonl" ]]; then
        echo "Missing or empty shard run output: ${shard_run_dir}/trajectory_values.jsonl" >&2
        exit 1
      fi
      per_run_inputs+=("${shard_run_dir}")
    done
    aggregate_dir="${ARCHIVE_DIR}/aggregate/${critic_label}/${value_position}"
    aggregate_log="${LOG_DIR}/aggregate_${critic_label}_${value_position}.log"
    echo "Aggregating ${critic_label}/${value_position} into ${aggregate_dir}"
    python -m value_decoding.eval_trajectory_value_baseline \
      --actor_checkpoint_dir "${ACTOR_CHECKPOINT_DIR}" \
      --critic_checkpoint_dir "${CRITIC_CHECKPOINT_DIRS_ARR[0]}" \
      --dataset_path "${DATASET_PATH}" \
      --output_dir "${aggregate_dir}" \
      --value_position "${value_position}" \
      --aggregate_input_dirs "${per_run_inputs[@]}" \
      2>&1 | tee "${aggregate_log}"
  done
done

SUMMARY_PATHS=()
for summary_path in "${ARCHIVE_DIR}/aggregate"/*/*/summary_metrics.json; do
  [[ -f "${summary_path}" ]] || continue
  SUMMARY_PATHS+=("${summary_path}")
done
if [[ ${#SUMMARY_PATHS[@]} -gt 0 ]]; then
  python - "${ARCHIVE_DIR}/comparison_summary.json" "${SUMMARY_PATHS[@]}" <<'PYINNER'
import json
import sys
from pathlib import Path

output_path = Path(sys.argv[1])
summaries = [json.loads(Path(path).read_text(encoding="utf-8")) for path in sys.argv[2:]]
payload = {
    "num_runs": len(summaries),
    "runs": summaries,
    "best_by_critic_mse": min(summaries, key=lambda item: float(item["critic_mse"])),
    "best_by_relative_mse_improvement": max(
        summaries,
        key=lambda item: float("-inf") if item.get("relative_mse_improvement") is None else float(item["relative_mse_improvement"]),
    ),
}
output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PYINNER
fi

echo "Trajectory value baseline multinode eval finished. Outputs: ${ARCHIVE_DIR}"
