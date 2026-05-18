#!/bin/bash
#SBATCH --job-name=critic_bank_bon_hi_ent
#SBATCH --account=ECS26006
#SBATCH --partition=gh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j_critic_bank_bon_hi_ent.out
#SBATCH --error=slurm-%j_critic_bank_bon_hi_ent.err

set -euo pipefail

# -----------------------------
# Environment setup
# -----------------------------
if command -v module >/dev/null 2>&1; then
  module reset
  module load nvidia/25.9
fi

VENV="${VENV:-}"
CONDA_SH="${CONDA_SH:-/data/shuozhe/miniconda3/etc/profile.d/conda.sh}"
CONDA_ENV="${CONDA_ENV:-verl}"
if [[ -n "$VENV" && -f "${VENV}/bin/activate" ]]; then
  source "${VENV}/bin/activate"
elif [[ -f "$CONDA_SH" ]]; then
  source "$CONDA_SH"
  conda activate "$CONDA_ENV"
else
  echo "Warning: no venv or conda setup found; using current Python environment."
fi

UV_CACHE_DIR="${SCRATCH:-/tmp}/.cache/uv"
HF_HOME="${SCRATCH:-/tmp}/.cache/huggingface"
TIKTOKEN_ENCODINGS_BASE="${SCRATCH:-/tmp}/data/embeddings"

mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$TIKTOKEN_ENCODINGS_BASE"

export UV_CACHE_DIR
export HF_HOME
export TIKTOKEN_ENCODINGS_BASE
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=true

echo "Activated environment"
echo "Python: $(which python3)"
python3 -V

# -----------------------------
# Run identity
# -----------------------------
RUN_NAME="critic_bank_bon_hi_ent"
JOB_ID="${SLURM_JOB_ID:-manual}"
RUN_ID="${RUN_NAME}_${JOB_ID}"

# -----------------------------
# Paths
# -----------------------------
WORK_DIR="/data/shuozhe/verl"
CRITIC_CHECKPOINT_DIR="/data/shuozhe/saved_model/Prathyusha101/high_ent_critic_training_qwen7b_global_step_340"
RESPONSE_BANK_PATH="/data/shuozhe/verl/outputs/response_bank_hi_ent.jsonl"
export PYTHONPATH="${WORK_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

ARCHIVE_ROOT="${WORK_DIR}/value_decoding/output_archive"
ARCHIVE_DIR="${ARCHIVE_ROOT}/${RUN_ID}"

SCRATCH_ROOT="${SCRATCH:-${WORK_DIR}/value_decoding/tmp}/value_decoding_runs"
RUN_DIR="${SCRATCH_ROOT}/${RUN_ID}"
LOG_DIR="${RUN_DIR}/logs"
OUTPUT_DIR="${RUN_DIR}/critic_bank_best_of_n_hi_ent"
CRITIC_MERGED_ROOT="${RUN_DIR}/merged_critic_hf"

mkdir -p "$LOG_DIR" "$ARCHIVE_ROOT" "$OUTPUT_DIR"

# -----------------------------
# Best-of-N config
# -----------------------------
N_VALUES="1 2 4 8 16"
MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_TOKENS=2048
DTYPE="bf16"
DEVICE="cuda:0"
SKIP_MERGE=0
TRUST_REMOTE_CODE=0
RETOKENIZE_RESPONSES=0
KEEP_ALL_ORIGINAL_FIELDS=0

# -----------------------------
# Helpers
# -----------------------------
sync_to_work() {
  echo "Syncing run directory back to WORK..."
  mkdir -p "$ARCHIVE_DIR"
  rsync -a "$RUN_DIR"/ "$ARCHIVE_DIR"/ || true
  echo "Archived run to: $ARCHIVE_DIR"
}

trap sync_to_work EXIT

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
    missing_preview = ", ".join(path.name for path in missing_files[:5])
    if len(missing_files) > 5:
        missing_preview += ", ..."
    raise SystemExit(
        f"{component}: found config/tokenizer metadata at {component_dir}, but model weights are incomplete. "
        f"Missing referenced files: {missing_preview or 'weight files'}"
    )

raise SystemExit(
    f"{component}: checkpoint is neither complete HF nor raw FSDP at {component_dir}"
)
PY
}

validate_response_bank() {
  local bank_path="$1"
  python3 - "$bank_path" <<'PY'
import json
import sys
from collections import defaultdict
from pathlib import Path

path = Path(sys.argv[1])
if not path.exists():
    raise SystemExit(f"Response bank does not exist: {path}")

counts = defaultdict(int)
num_rows = 0
with path.open("r", encoding="utf-8") as input_file:
    for line_number, line in enumerate(input_file, start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        for key in ("prompt", "response", "response_token_ids", "prompt_index", "sample_index"):
            if key not in row:
                raise SystemExit(f"Missing {key!r} at {path}:{line_number}")
        counts[int(row["prompt_index"])] += 1
        num_rows += 1

if not counts:
    raise SystemExit(f"Response bank is empty: {path}")

print(
    f"Response bank OK: {num_rows} rows, {len(counts)} prompts, "
    f"bank sizes={sorted(set(counts.values()))}"
)
PY
}

echo "Validating inputs..."
validate_component_checkpoint "$CRITIC_CHECKPOINT_DIR" critic
validate_response_bank "$RESPONSE_BANK_PATH"

cd "$WORK_DIR"

read -r -a N_VALUES_ARR <<< "$N_VALUES"

CMD=(
  python3 -m value_decoding.critic_bank_best_of_n_eval
  --critic_checkpoint_dir "$CRITIC_CHECKPOINT_DIR"
  --response_bank_path "$RESPONSE_BANK_PATH"
  --output_dir "$OUTPUT_DIR"
  --critic_merged_root "$CRITIC_MERGED_ROOT"
  --max_prompt_length "$MAX_PROMPT_LENGTH"
  --max_response_tokens "$MAX_RESPONSE_TOKENS"
  --n_values "${N_VALUES_ARR[@]}"
  --dtype "$DTYPE"
  --device "$DEVICE"
)

[[ "$SKIP_MERGE" != "0" ]] && CMD+=(--skip_merge)
[[ "$TRUST_REMOTE_CODE" != "0" ]] && CMD+=(--trust_remote_code)
[[ "$RETOKENIZE_RESPONSES" != "0" ]] && CMD+=(--retokenize_responses)
[[ "$KEEP_ALL_ORIGINAL_FIELDS" != "0" ]] && CMD+=(--keep_all_original_fields)

printf 'Running command:\n'
printf ' %q' "${CMD[@]}"
printf '\n'

"${CMD[@]}" 2>&1 | tee "$LOG_DIR/critic_bank_best_of_n_eval.log"

echo "Critic value-estimation Best-of-N finished successfully."
