#!/bin/bash
#SBATCH --job-name=critic_test_metrics
#SBATCH --account=ECS26006
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --time=10:20:00
#SBATCH --output=slurm-%j_critic_test_metrics.out
#SBATCH --error=slurm-%j_critic_test_metrics.err

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
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

# Reduce harmless Transformers warnings for greedy generation.
export TRANSFORMERS_VERBOSITY=error

echo "Activated environment"
echo "Python: $(which python3)"
python3 -V

# -----------------------------
# Run identity
# -----------------------------
RUN_NAME="critic_test_metrics_math500"
RUN_ID="${RUN_NAME}_${SLURM_JOB_ID}"

# -----------------------------
# Paths
# -----------------------------
# Add one or more checkpoint roots here. Each root may be either:
#   1. a PPO export containing actor/ and critic/ subdirectories, or
#   2. a plain HF actor checkpoint, in which case only accuracy is logged.
CHECKPOINT_PATHS=(
  "/scratch/10587/npg493/verl_runs/qwen2.5_7B_ppo_664998/global_step_100"
  "/scratch/10587/npg493/verl_runs/qwen2.5_7B_ppo_664998/global_step_200"
  "/scratch/10587/npg493/verl_runs/qwen2.5_7B_ppo_664998/global_step_300"
  "/scratch/10587/npg493/verl_runs/qwen2.5_7B_ppo_664998/global_step_400"
  "/scratch/10587/npg493/verl_runs/qwen2.5_7B_ppo_664998/global_step_500"
  "/scratch/10587/npg493/verl_runs/qwen2.5_7B_ppo_664998/global_step_600"
  "/scratch/10587/npg493/verl_runs/qwen2.5_7B_ppo_664998/global_step_700"
  "/scratch/10587/npg493/verl_runs/qwen2.5_7B_ppo_664998/global_step_800"
)

if [[ -n "${CHECKPOINT_PATHS_OVERRIDE:-}" ]]; then
  # Optional submit-time override:
  #   sbatch --export=ALL,CHECKPOINT_PATHS_OVERRIDE="/path/ckpt1:/path/ckpt2" ...
  IFS=':' read -r -a CHECKPOINT_PATHS <<< "$CHECKPOINT_PATHS_OVERRIDE"
fi

DATASET_PATH="${DATASET_PATH:-/work2/09576/shuozhe/saved_dataset/MetaMathQA-math-500/test.parquet}"
WORK_DIR="${WORK_DIR:-/work2/09576/shuozhe/verl}"
export PYTHONPATH="${WORK_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

ARCHIVE_ROOT="${ARCHIVE_ROOT:-/work2/09576/shuozhe/verl/value_decoding/output_archive}"
ARCHIVE_DIR="${ARCHIVE_ROOT}/${RUN_ID}"

SCRATCH_ROOT="${SCRATCH}/value_decoding_runs"
RUN_DIR="${SCRATCH_ROOT}/${RUN_ID}"
LOG_DIR="${RUN_DIR}/logs"
OUTPUT_DIR="${RUN_DIR}/critic_test_metrics_math500"

mkdir -p "$LOG_DIR" "$ARCHIVE_ROOT" "$OUTPUT_DIR"

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
BATCH_SIZE=4
ACTOR_MICRO_BATCH_SIZE=1
DTYPE="bf16"
DEVICE="cuda:0"

# Match default validation/pass@1 unless changed.
GENERATION_BACKEND="vllm"
ACTOR_SAMPLING_MODE="greedy"
ACTOR_TEMPERATURE=1.0
ACTOR_TOP_P=1.0
ACTOR_TOP_K=0
VLLM_GPU_MEMORY_UTILIZATION=0.90
VLLM_TENSOR_PARALLEL_SIZE=1
VLLM_MAX_MODEL_LEN=4096
VLLM_MAX_NUM_SEQS=""
VLLM_ENFORCE_EAGER=1

# PPO critic target/loss defaults.
GAMMA=1.0
LAM=1.0
CLIPRANGE_VALUE=0.5
LOSS_AGG_MODE="token-mean"

SEED=42
TRUST_REMOTE_CODE=0
SKIP_MERGE=0
REQUIRE_CRITIC=0
SAVE_TRAJECTORIES=0
HF_SOURCE_DIR=""

# -----------------------------
# Helpers
# -----------------------------
SCRIPT_PATH="${WORK_DIR}/value_decoding/script/$(basename "${BASH_SOURCE[0]}")"

sync_to_work() {
  echo "Syncing run directory back to WORK..."
  mkdir -p "$ARCHIVE_DIR"
  rsync -a "$RUN_DIR"/ "$ARCHIVE_DIR"/ || true
  echo "Archived run to: $ARCHIVE_DIR"
}
trap sync_to_work EXIT

validate_checkpoint() {
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
if not checkpoint_dir.exists():
    raise SystemExit(f"checkpoint path not found: {checkpoint_dir}")

for component in ("actor", "critic"):
    component_dir = resolve_component_checkpoint_dir(checkpoint_dir, component=component)
    if has_complete_hf_checkpoint(component_dir):
        print(f"{checkpoint_dir} {component}: complete Hugging Face checkpoint at {component_dir}")
        continue
    if has_fsdp_checkpoint_shards(component_dir):
        print(f"{checkpoint_dir} {component}: raw FSDP checkpoint at {component_dir}")
        continue
    if has_hf_config(component_dir):
        missing_files = find_missing_hf_weight_files(component_dir)
        if missing_files:
            preview = ", ".join(path.name for path in missing_files[:5])
            if len(missing_files) > 5:
                preview += ", ..."
            raise SystemExit(f"{checkpoint_dir} {component}: incomplete HF checkpoint; missing {preview}")
        print(f"{checkpoint_dir} {component}: HF config found at {component_dir}")
        continue
    if component == "critic":
        print(f"{checkpoint_dir} {component}: not found; evaluator will log accuracy only unless --require_critic is set")
        continue
    raise SystemExit(f"{checkpoint_dir} {component}: unsupported checkpoint layout at {component_dir}")
PY
}

# -----------------------------
# Debug info
# -----------------------------
echo "Job ID: $SLURM_JOB_ID"
echo "Run ID: $RUN_ID"
echo "SLURM nodes: ${SLURM_JOB_NODELIST:-unknown}"
echo "SCRATCH: $SCRATCH"
echo "RUN_DIR: $RUN_DIR"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "ARCHIVE_DIR: $ARCHIVE_DIR"
echo "SCRIPT_PATH: $SCRIPT_PATH"

echo "Checking inputs..."
ls -ld "$WORK_DIR"
ls -lh "$DATASET_PATH"
for checkpoint_path in "${CHECKPOINT_PATHS[@]}"; do
  echo "Checkpoint: $checkpoint_path"
  ls -ld "$checkpoint_path"
  validate_checkpoint "$checkpoint_path"
done

cd "$WORK_DIR"

CMD=(
  python3 -m value_decoding.critic_test_metrics_eval
  --dataset_path "$DATASET_PATH"
  --checkpoint_paths "${CHECKPOINT_PATHS[@]}"
  --output_dir "$OUTPUT_DIR"
  --prompt_key "$PROMPT_KEY"
  --start_index "$START_INDEX"
  --max_examples "$MAX_EXAMPLES"
  --max_prompt_length "$MAX_PROMPT_LENGTH"
  --max_new_tokens "$MAX_NEW_TOKENS"
  --batch_size "$BATCH_SIZE"
  --actor_micro_batch_size "$ACTOR_MICRO_BATCH_SIZE"
  --dtype "$DTYPE"
  --device "$DEVICE"
  --generation_backend "$GENERATION_BACKEND"
  --actor_sampling_mode "$ACTOR_SAMPLING_MODE"
  --actor_temperature "$ACTOR_TEMPERATURE"
  --actor_top_p "$ACTOR_TOP_P"
  --actor_top_k "$ACTOR_TOP_K"
  --vllm_gpu_memory_utilization "$VLLM_GPU_MEMORY_UTILIZATION"
  --vllm_tensor_parallel_size "$VLLM_TENSOR_PARALLEL_SIZE"
  --gamma "$GAMMA"
  --lam "$LAM"
  --cliprange_value "$CLIPRANGE_VALUE"
  --loss_agg_mode "$LOSS_AGG_MODE"
  --seed "$SEED"
)

[[ -n "$RESPONSE_KEY" ]] && CMD+=(--response_key "$RESPONSE_KEY")
[[ "$SHUFFLE_EXAMPLES" != "0" ]] && CMD+=(--shuffle_examples)
[[ "$TRUST_REMOTE_CODE" != "0" ]] && CMD+=(--trust_remote_code)
[[ "$SKIP_MERGE" != "0" ]] && CMD+=(--skip_merge)
[[ "$REQUIRE_CRITIC" != "0" ]] && CMD+=(--require_critic)
[[ "$SAVE_TRAJECTORIES" != "0" ]] && CMD+=(--save_trajectories)
[[ "$VLLM_ENFORCE_EAGER" != "0" ]] && CMD+=(--vllm_enforce_eager)
[[ -n "$VLLM_MAX_MODEL_LEN" ]] && CMD+=(--vllm_max_model_len "$VLLM_MAX_MODEL_LEN")
[[ -n "$VLLM_MAX_NUM_SEQS" ]] && CMD+=(--vllm_max_num_seqs "$VLLM_MAX_NUM_SEQS")
[[ -n "$HF_SOURCE_DIR" ]] && CMD+=(--hf_source_dir "$HF_SOURCE_DIR")

printf 'Running command:\n'
printf ' %q' "${CMD[@]}"
printf '\n'
"${CMD[@]}" 2>&1 | tee "$LOG_DIR/critic_test_metrics_eval.log"

echo "Critic test metrics evaluation finished successfully."
