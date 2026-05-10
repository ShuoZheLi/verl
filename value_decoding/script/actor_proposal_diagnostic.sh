#!/usr/bin/env bash

set -euo pipefail

# =============================================================================
# ACTOR PROPOSAL QUALITY DIAGNOSTIC
#
# For each frozen actor checkpoint:
# - sample a bank of responses per prompt
# - score each response with the task scorer
# - measure search headroom, entropy, diversity, and within-prompt variance
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
OUTPUT_DIR="/data/shuozhe/verl/value_decoding/output/actor_proposal_diagnostic"
PROMPT_KEY="prompt"
RESPONSE_KEY=""
START_INDEX=0
MAX_EXAMPLES=500
SHUFFLE_EXAMPLES=0

# -----------------------------
# Sampling
# -----------------------------
NUM_SAMPLES_PER_PROMPT=16
MAX_PROMPT_LENGTH=2048
MAX_NEW_TOKENS=2048
TEMPERATURE=1.0
TOP_P=1.0
TOP_K=0
DTYPE="bf16"
SEED=42
BOOTSTRAP_SAMPLES=2000

# Each worker loads the full actor. Use one device entry per worker.
WORKER_DEVICES=(
  "cuda:0"
)

# -----------------------------
# Output / diagnostics
# -----------------------------
SKIP_PLOTS=0
PLOT_DPI=160
OMIT_PROMPT_TEXT=0
OMIT_RESPONSE_TOKEN_IDS=0
STORE_TOKEN_ENTROPIES=0

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

CMD=(
  python value_decoding/actor_proposal_diagnostic.py
  --actor_checkpoint_dirs "${ACTOR_CHECKPOINT_DIRS[@]}"
  --actor_names "${ACTOR_NAMES[@]}"
  --dataset_path "$DATASET_PATH"
  --output_dir "$OUTPUT_DIR"
  --prompt_key "$PROMPT_KEY"
  --start_index "$START_INDEX"
  --max_examples "$MAX_EXAMPLES"
  --num_samples_per_prompt "$NUM_SAMPLES_PER_PROMPT"
  --max_prompt_length "$MAX_PROMPT_LENGTH"
  --max_new_tokens "$MAX_NEW_TOKENS"
  --temperature "$TEMPERATURE"
  --top_p "$TOP_P"
  --top_k "$TOP_K"
  --dtype "$DTYPE"
  --seed "$SEED"
  --bootstrap_samples "$BOOTSTRAP_SAMPLES"
  --plot_dpi "$PLOT_DPI"
)

[[ -n "$RESPONSE_KEY" ]] && CMD+=(--response_key "$RESPONSE_KEY")
[[ "$SHUFFLE_EXAMPLES" == "1" ]] && CMD+=(--shuffle_examples)
[[ "$TRUST_REMOTE_CODE" == "1" ]] && CMD+=(--trust_remote_code)
[[ "$SKIP_MERGE" == "1" ]] && CMD+=(--skip_merge)
[[ "$SKIP_PLOTS" == "1" ]] && CMD+=(--skip_plots)
[[ "$OMIT_PROMPT_TEXT" == "1" ]] && CMD+=(--omit_prompt_text)
[[ "$OMIT_RESPONSE_TOKEN_IDS" == "1" ]] && CMD+=(--omit_response_token_ids)
[[ "$STORE_TOKEN_ENTROPIES" == "1" ]] && CMD+=(--store_token_entropies)
[[ "$DISABLE_ACTOR_CACHE" == "1" ]] && CMD+=(--disable_actor_cache)
[[ ${#WORKER_DEVICES[@]} -gt 0 ]] && CMD+=(--worker_devices "${WORKER_DEVICES[@]}")
[[ ${#ACTOR_MERGED_ROOTS[@]} -gt 0 ]] && CMD+=(--actor_merged_roots "${ACTOR_MERGED_ROOTS[@]}")
[[ ${#ACTOR_HF_SOURCE_DIRS[@]} -gt 0 ]] && CMD+=(--actor_hf_source_dirs "${ACTOR_HF_SOURCE_DIRS[@]}")

printf 'Running command:\n  '
printf '%q ' "${CMD[@]}"
printf '\n'

"${CMD[@]}" "$@"
