#!/usr/bin/env bash

set -euo pipefail

# =============================================================================
# ACTOR GREEDY EVALUATION
#
# For each frozen actor checkpoint:
# - deterministically decode one greedy response per prompt
# - score each response with the task scorer
# - write per-actor prediction rows and summary metrics
#
# Important:
# - This launcher is actor-only. It does not load or use a critic.
# - Each worker device loads a full copy of the actor, so size the worker list
#   to match available GPU memory.
# =============================================================================

# -----------------------------
# Actors
# -----------------------------
ACTOR_CHECKPOINT_DIRS=(
  "/data/shuozhe/verl/train_log/job_05b_vh_init_e5_metamath/global_step_800"
)
ACTOR_NAMES=(
  "actor_gs800"
)

# Optional per-actor merged HF roots. Leave empty to use each checkpoint's
# default `merged_hf/actor` location or merge target.
ACTOR_MERGED_ROOTS=()

# Optional per-actor HF config/tokenizer source dirs. Useful when a raw actor
# checkpoint was copied without its huggingface metadata.
ACTOR_HF_SOURCE_DIRS=()

# -----------------------------
# Data
# -----------------------------
DATASET_PATH="/data/shuozhe/saved_dataset/MetaMathQA-math-500/test.parquet"
OUTPUT_DIR="/data/shuozhe/verl/value_decoding/output/actor_greedy_eval"
PROMPT_KEY="prompt"
RESPONSE_KEY=""
START_INDEX=0
MAX_EXAMPLES=500
SHUFFLE_EXAMPLES=0

# -----------------------------
# Decoding
# -----------------------------
MAX_PROMPT_LENGTH=2048
MAX_NEW_TOKENS=2048
DTYPE="bf16"
SEED=42
DEVICE="cuda:0"

# For local multi-worker execution, use one device entry per worker. With one
# GPU, keep this empty and use DEVICE above.
WORKER_DEVICES=()

# -----------------------------
# Output / diagnostics
# -----------------------------
OMIT_PROMPT_TEXT=0
OMIT_RESPONSE_TOKEN_IDS=0

# -----------------------------
# Misc
# -----------------------------
TRUST_REMOTE_CODE=0
SKIP_MERGE=0
DISABLE_ACTOR_CACHE=0

source /data/shuozhe/miniconda3/etc/profile.d/conda.sh
conda activate verl

if [[ ${#ACTOR_CHECKPOINT_DIRS[@]} -ne ${#ACTOR_NAMES[@]} ]]; then
  echo "ACTOR_CHECKPOINT_DIRS and ACTOR_NAMES must have the same length." >&2
  exit 1
fi

if [[ ${#ACTOR_MERGED_ROOTS[@]} -gt 0 && ${#ACTOR_MERGED_ROOTS[@]} -ne ${#ACTOR_CHECKPOINT_DIRS[@]} ]]; then
  echo "If ACTOR_MERGED_ROOTS is set, it must have one entry per actor." >&2
  exit 1
fi

if [[ ${#ACTOR_HF_SOURCE_DIRS[@]} -gt 0 && ${#ACTOR_HF_SOURCE_DIRS[@]} -ne ${#ACTOR_CHECKPOINT_DIRS[@]} ]]; then
  echo "If ACTOR_HF_SOURCE_DIRS is set, it must have one entry per actor." >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

for actor_index in "${!ACTOR_CHECKPOINT_DIRS[@]}"; do
  actor_checkpoint_dir="${ACTOR_CHECKPOINT_DIRS[$actor_index]}"
  actor_name="${ACTOR_NAMES[$actor_index]}"
  actor_output_dir="${OUTPUT_DIR}/${actor_name}"
  mkdir -p "$actor_output_dir"

  CMD=(
    python -m value_decoding.actor_greedy_eval
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
    --device "$DEVICE"
    --seed "$SEED"
  )

  [[ -n "$RESPONSE_KEY" ]] && CMD+=(--response_key "$RESPONSE_KEY")
  [[ ${#ACTOR_MERGED_ROOTS[@]} -gt 0 && -n "${ACTOR_MERGED_ROOTS[$actor_index]}" ]] && \
    CMD+=(--actor_merged_root "${ACTOR_MERGED_ROOTS[$actor_index]}")
  [[ ${#ACTOR_HF_SOURCE_DIRS[@]} -gt 0 && -n "${ACTOR_HF_SOURCE_DIRS[$actor_index]}" ]] && \
    CMD+=(--actor_hf_source_dir "${ACTOR_HF_SOURCE_DIRS[$actor_index]}")
  [[ ${#WORKER_DEVICES[@]} -gt 0 ]] && CMD+=(--worker_devices "${WORKER_DEVICES[@]}")
  [[ "$SHUFFLE_EXAMPLES" == "1" ]] && CMD+=(--shuffle_examples)
  [[ "$TRUST_REMOTE_CODE" == "1" ]] && CMD+=(--trust_remote_code)
  [[ "$SKIP_MERGE" == "1" ]] && CMD+=(--skip_merge)
  [[ "$OMIT_PROMPT_TEXT" == "1" ]] && CMD+=(--omit_prompt_text)
  [[ "$OMIT_RESPONSE_TOKEN_IDS" == "1" ]] && CMD+=(--omit_response_token_ids)
  [[ "$DISABLE_ACTOR_CACHE" == "1" ]] && CMD+=(--disable_actor_cache)

  printf 'Running greedy evaluation for actor %s (%s/%s):\n  ' "$actor_name" "$((actor_index + 1))" "${#ACTOR_CHECKPOINT_DIRS[@]}"
  printf '%q ' "${CMD[@]}"
  printf '\n'

  "${CMD[@]}" "$@"
done

python - "$OUTPUT_DIR" "${ACTOR_NAMES[@]}" <<'PY'
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
