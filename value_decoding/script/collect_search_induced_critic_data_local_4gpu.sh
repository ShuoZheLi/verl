#!/bin/bash
set -eo pipefail

# Local 4-GPU launcher for search-induced critic data collection.
# Override most knobs by exporting the variable before running this script, e.g.:
#   MAX_PROMPTS=16 DEBUG_NUM_PROMPTS=4 bash value_decoding/script/collect_search_induced_critic_data_local_4gpu.sh

# -----------------------------
# Environment setup
# -----------------------------
source /data/shuozhe/miniconda3/etc/profile.d/conda.sh
conda activate verl
set -u

WORK_DIR="${WORK_DIR:-/data/shuozhe/verl}"
export PYTHONPATH="${WORK_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export HF_HOME="${HF_HOME:-/data/shuozhe/.cache/huggingface}"
export TIKTOKEN_ENCODINGS_BASE="${TIKTOKEN_ENCODINGS_BASE:-/data/shuozhe/data/embeddings}"
mkdir -p "$HF_HOME" "$TIKTOKEN_ENCODINGS_BASE"

# -----------------------------
# Run identity and paths
# -----------------------------
RUN_NAME="${RUN_NAME:-search_induced_critic_data_local_4gpu}"
RUN_ID="${RUN_ID:-${RUN_NAME}_$(date +%Y%m%d_%H%M%S)}"

ACTOR_CHECKPOINT_DIR="${ACTOR_CHECKPOINT_DIR:-/data/shuozhe/verl/train_log/job_05b_vh_init_e5_metamath/global_step_800}"
COLLECTOR_CRITIC_CHECKPOINT_DIR="${COLLECTOR_CRITIC_CHECKPOINT_DIR:-/data/shuozhe/verl/train_log/job_05b_vh_init_e5_metamath/global_step_800}"
DATASET_PATH="${DATASET_PATH:-/data/shuozhe/saved_dataset/MetaMathQA-math-500/math7500.parquet}"

OUTPUT_ARCHIVE_ROOT="${OUTPUT_ARCHIVE_ROOT:-${WORK_DIR}/value_decoding/output_archive}"
RUN_ROOT="${RUN_ROOT:-${WORK_DIR}/value_decoding/local_runs}"
RUN_DIR="${RUN_DIR:-${RUN_ROOT}/${RUN_ID}}"
LOG_DIR="${RUN_DIR}/logs"
SHARD_ROOT="${RUN_DIR}/shards"
MERGED_ROOT="${RUN_DIR}/merged_hf"
CACHE_ROOT="${RUN_DIR}/cache"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_DIR}/search_induced_critic_data}"
ARCHIVE_DIR="${ARCHIVE_DIR:-${OUTPUT_ARCHIVE_ROOT}/${RUN_ID}}"

mkdir -p "$LOG_DIR" "$SHARD_ROOT" "$MERGED_ROOT" "$CACHE_ROOT" "$OUTPUT_DIR" "$OUTPUT_ARCHIVE_ROOT"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_ROOT}/xdg}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${CACHE_ROOT}/torch_inductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${CACHE_ROOT}/triton}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${CACHE_ROOT}/torch_extensions}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${CACHE_ROOT}/vllm}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-${CACHE_ROOT}/flashinfer_workspace}"
export DO_NOT_TRACK="${DO_NOT_TRACK:-1}"
export VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"
CUDA_RUNTIME_LIB_DIR="${CUDA_RUNTIME_LIB_DIR:-/data/shuozhe/miniconda3/envs/verl/lib}"
CUDA_RUNTIME_LIB64_DIR="${CACHE_ROOT}/cuda_runtime/lib64"
mkdir -p "$CUDA_RUNTIME_LIB64_DIR"
ln -sf "${CUDA_RUNTIME_LIB_DIR}/libcudart.so" "$CUDA_RUNTIME_LIB64_DIR/libcudart.so"
export CUDA_HOME="${CUDA_HOME:-/data/shuozhe/miniconda3/envs/verl}"
export LD_LIBRARY_PATH="${CUDA_RUNTIME_LIB64_DIR}:${CUDA_RUNTIME_LIB_DIR}:${CUDA_HOME}/lib:${CUDA_HOME}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export LIBRARY_PATH="${CUDA_RUNTIME_LIB64_DIR}:${CUDA_RUNTIME_LIB_DIR}:${CUDA_HOME}/lib:${CUDA_HOME}/lib64${LIBRARY_PATH:+:${LIBRARY_PATH}}"
export LDFLAGS="-L${CUDA_RUNTIME_LIB64_DIR} -L${CUDA_RUNTIME_LIB_DIR} -L${CUDA_HOME}/lib -L${CUDA_HOME}/lib64${LDFLAGS:+ ${LDFLAGS}}"
mkdir -p "$XDG_CACHE_HOME" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$VLLM_CACHE_ROOT" "$FLASHINFER_WORKSPACE_BASE"

# Optional override directories for HF config/tokenizer metadata used during FSDP merge.
ACTOR_HF_SOURCE_DIR="${ACTOR_HF_SOURCE_DIR:-}"
CRITIC_HF_SOURCE_DIR="${CRITIC_HF_SOURCE_DIR:-}"

# -----------------------------
# Collection config
# -----------------------------
PROMPT_KEY="${PROMPT_KEY:-prompt}"
RESPONSE_KEY="${RESPONSE_KEY:-}"
START_INDEX="${START_INDEX:-0}"
MAX_PROMPTS="${MAX_PROMPTS:-7500}"
SHUFFLE_PROMPTS="${SHUFFLE_PROMPTS:-0}"

CHUNK_SIZE="${CHUNK_SIZE:-128}"
NUM_CHUNK_CANDIDATES="${NUM_CHUNK_CANDIDATES:-8}"
NUM_SEARCH_STEPS_PER_PROMPT="${NUM_SEARCH_STEPS_PER_PROMPT:-20}"
COMPLETION_MAX_NEW_TOKENS="${COMPLETION_MAX_NEW_TOKENS:-2048}"
COLLECTOR_SELECTION_MODE="${COLLECTOR_SELECTION_MODE:-argmax}"
COLLECTOR_EPSILON="${COLLECTOR_EPSILON:-0.0}"
COLLECTOR_VALUE_TEMPERATURE="${COLLECTOR_VALUE_TEMPERATURE:-1.0}"

ACTOR_TEMPERATURE="${ACTOR_TEMPERATURE:-1.0}"
ACTOR_TOP_P="${ACTOR_TOP_P:-1.0}"
ACTOR_TOP_K="${ACTOR_TOP_K:-0}"
ACTOR_BATCH_SIZE="${ACTOR_BATCH_SIZE:-8}"

MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
DTYPE="${DTYPE:-bf16}"
SEED="${SEED:-42}"
GENERATION_BACKEND="${GENERATION_BACKEND:-vllm}"
CRITIC_BATCH_SIZE="${CRITIC_BATCH_SIZE:-8}"

# Each shard sees exactly one GPU through CUDA_VISIBLE_DEVICES, so vLLM TP must stay 1.
NUM_LOCAL_SHARDS="${NUM_LOCAL_SHARDS:-4}"
CUDA_DEVICES_CSV="${CUDA_DEVICES_CSV:-0,1,2,3}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.60}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-}"
VLLM_ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-0}"

SAVE_FULL_TEXT="${SAVE_FULL_TEXT:-1}"
SAVE_TOKEN_IDS="${SAVE_TOKEN_IDS:-1}"
SAVE_COMPLETED_RESPONSES="${SAVE_COMPLETED_RESPONSES:-0}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
SKIP_MERGE="${SKIP_MERGE:-0}"
DEBUG_NUM_PROMPTS="${DEBUG_NUM_PROMPTS:-}"
DRY_RUN="${DRY_RUN:-0}"

SCRIPT_PATH="${WORK_DIR}/value_decoding/script/$(basename "${BASH_SOURCE[0]}")"

sync_to_archive() {
  echo "Syncing run directory to archive..."
  mkdir -p "$ARCHIVE_DIR"
  rsync -a \
    --exclude='merged_hf/' \
    --exclude='merged_hf/***' \
    --exclude='cache/' \
    --exclude='cache/***' \
    --exclude='shards/*/merged_hf/' \
    --exclude='shards/*/merged_hf/***' \
    "$RUN_DIR"/ "$ARCHIVE_DIR"/ || true
  echo "Archived run to: $ARCHIVE_DIR"
}
trap sync_to_archive EXIT

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

ensure_shared_merged_checkpoints() {
  python3 - "$ACTOR_CHECKPOINT_DIR" "$COLLECTOR_CRITIC_CHECKPOINT_DIR" "$MERGED_ROOT" "$ACTOR_HF_SOURCE_DIR" "$CRITIC_HF_SOURCE_DIR" "$SKIP_MERGE" <<'PY'
import sys
from pathlib import Path

from value_decoding.checkpointing import ensure_merged_component_checkpoint

actor_checkpoint = Path(sys.argv[1])
critic_checkpoint = Path(sys.argv[2])
merged_root = Path(sys.argv[3])
actor_hf_source = None if not sys.argv[4] else Path(sys.argv[4])
critic_hf_source = None if not sys.argv[5] else Path(sys.argv[5])
skip_merge = sys.argv[6] not in {"0", "false", "False", ""}

actor_hf = ensure_merged_component_checkpoint(
    actor_checkpoint,
    component="actor",
    merged_root=merged_root,
    hf_source_dir=actor_hf_source,
    skip_merge=skip_merge,
)
critic_hf = ensure_merged_component_checkpoint(
    critic_checkpoint,
    component="critic",
    merged_root=merged_root,
    hf_source_dir=critic_hf_source,
    skip_merge=skip_merge,
)
print(f"actor HF checkpoint ready: {actor_hf}")
print(f"critic HF checkpoint ready: {critic_hf}")
PY
}

count_dataset_rows() {
  python3 - "$DATASET_PATH" <<'PY'
import sys
path = sys.argv[1]
try:
    import pyarrow.parquet as pq
    print(pq.ParquetFile(path).metadata.num_rows)
except Exception:
    import pandas as pd
    print(len(pd.read_parquet(path)))
PY
}

ensure_conda_cuda_runtime_link() {
  local conda_lib="/data/shuozhe/miniconda3/envs/verl/lib"
  local conda_lib64="/data/shuozhe/miniconda3/envs/verl/lib64"
  mkdir -p "$conda_lib64"
  if [[ -f "${conda_lib}/libcudart.so" && ! -e "${conda_lib64}/libcudart.so" ]]; then
    ln -s "${conda_lib}/libcudart.so" "${conda_lib64}/libcudart.so"
  fi
  if [[ ! -e "${conda_lib64}/libcudart.so" ]]; then
    echo "Missing libcudart.so in ${conda_lib64}; FlashInfer/vLLM JIT may fail to link." >&2
    return 1
  fi
}

check_cuda_devices() {
  python3 - "$NUM_LOCAL_SHARDS" "$CUDA_DEVICES_CSV" <<'PY'
import sys
import torch

num_shards = int(sys.argv[1])
requested = [item.strip() for item in sys.argv[2].split(",") if item.strip()]
if len(requested) != num_shards:
    raise SystemExit(f"CUDA_DEVICES_CSV must list exactly {num_shards} devices, got {requested}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in the active conda environment")
count = torch.cuda.device_count()
missing = [dev for dev in requested if not dev.isdigit() or int(dev) < 0 or int(dev) >= count]
if missing:
    raise SystemExit(f"Requested CUDA devices {missing} are not available; torch sees {count} CUDA devices")
print(f"CUDA devices available: {count}; using physical devices: {','.join(requested)}")
PY
}

merge_shards() {
  python3 - "$SHARD_ROOT" "$OUTPUT_DIR" "$RUN_ID" <<'PY'
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

shard_root = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
run_id = sys.argv[3]
output_dir.mkdir(parents=True, exist_ok=True)

candidate_out = output_dir / "search_induced_candidates.jsonl"
prompt_out = output_dir / "prompt_summaries.jsonl"

candidate_rows = []
prompt_rows = []
shard_dirs = sorted(path for path in shard_root.glob("shard_*") if path.is_dir())
for shard_dir in shard_dirs:
    candidate_path = shard_dir / "search_induced_candidates.jsonl"
    prompt_path = shard_dir / "prompt_summaries.jsonl"
    if candidate_path.exists():
        with candidate_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    candidate_rows.append(json.loads(line))
    if prompt_path.exists():
        with prompt_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    prompt_rows.append(json.loads(line))

with candidate_out.open("w", encoding="utf-8") as handle:
    for row in candidate_rows:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
with prompt_out.open("w", encoding="utf-8") as handle:
    for row in prompt_rows:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")

first_config = None
for shard_dir in shard_dirs:
    config_path = shard_dir / "config.json"
    if config_path.exists():
        first_config = json.loads(config_path.read_text(encoding="utf-8"))
        break
if first_config is not None:
    merged_config = dict(first_config)
    merged_config["run_id"] = run_id
    merged_config["shard_root"] = str(shard_root)
    merged_config["output_dir"] = str(output_dir)
    merged_config["num_shards"] = len(shard_dirs)
    (output_dir / "config.json").write_text(json.dumps(merged_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

candidate_rewards = [float(row["mc_reward"]) for row in candidate_rows]
candidate_values = [float(row["collector_value"]) for row in candidate_rows]

def mean(values):
    values = list(values)
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))

def rankdata(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end
    return ranks

def corr(x, y, spearman=False):
    if len(x) != len(y) or len(x) < 2:
        return None
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if spearman:
        x = rankdata(x)
        y = rankdata(y)
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])

groups = {}
for row in candidate_rows:
    groups.setdefault(row["candidate_group_id"], []).append(row)
group_summaries = []
for rows in groups.values():
    first = rows[0]
    group_summaries.append(
        {
            "group_success_rate": float(first["group_success_rate"]),
            "group_oracle_reward": float(first["group_oracle_reward"]),
            "group_random_reward_mean": float(first["group_random_reward_mean"]),
            "group_collector_top1_reward": float(first["group_collector_top1_reward"]),
            "group_false_high_selected": bool(first["group_false_high_selected"]),
            "pairwise": first.get("collector_pairwise_ranking_accuracy_in_group"),
        }
    )
rankable = [group for group in group_summaries if group["pairwise"] is not None]
oracle_positive = [group for group in group_summaries if group["group_oracle_reward"] == 1.0]

summary = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "run_id": run_id,
    "shard_root": str(shard_root),
    "output_dir": str(output_dir),
    "num_shards": len(shard_dirs),
    "num_prompts": len(prompt_rows),
    "num_candidate_examples": len(candidate_rows),
    "num_candidate_groups": len(group_summaries),
    "candidate_level": {
        "mean_mc_reward": mean(candidate_rewards),
        "mean_collector_value": mean(candidate_values),
        "pearson_collector_value_vs_reward": corr(candidate_values, candidate_rewards),
        "spearman_collector_value_vs_reward": corr(candidate_values, candidate_rewards, spearman=True),
    },
    "group_level": {
        "fraction_rankable_groups": float(len(rankable) / len(group_summaries)) if group_summaries else None,
        "mean_group_success_rate": mean(group["group_success_rate"] for group in group_summaries),
        "mean_group_oracle_reward": mean(group["group_oracle_reward"] for group in group_summaries),
        "mean_group_random_reward": mean(group["group_random_reward_mean"] for group in group_summaries),
        "mean_group_collector_top1_reward": mean(group["group_collector_top1_reward"] for group in group_summaries),
        "collector_recovery_rate": mean(group["group_collector_top1_reward"] == 1.0 for group in oracle_positive) if oracle_positive else None,
        "false_high_selected_rate": mean(group["group_false_high_selected"] for group in oracle_positive) if oracle_positive else None,
        "collector_pairwise_ranking_accuracy": mean(group["pairwise"] for group in rankable),
    },
}
(output_dir / "summary_metrics.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
}

validate_merged_outputs() {
  python3 - "$OUTPUT_DIR" "$TOTAL_PROMPTS" "$NUM_CHUNK_CANDIDATES" <<'PY'
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
expected_prompts = int(sys.argv[2])
num_candidates = int(sys.argv[3])
candidate_path = output_dir / "search_induced_candidates.jsonl"
prompt_path = output_dir / "prompt_summaries.jsonl"
summary_path = output_dir / "summary_metrics.json"
for path in (candidate_path, prompt_path, summary_path):
    if not path.is_file():
        raise SystemExit(f"Missing expected output file: {path}")

def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows

candidate_rows = read_jsonl(candidate_path)
prompt_rows = read_jsonl(prompt_path)
if len(prompt_rows) != expected_prompts:
    raise SystemExit(f"Expected {expected_prompts} prompt summaries, found {len(prompt_rows)}")
expected_min_candidates = expected_prompts * num_candidates
if len(candidate_rows) < expected_min_candidates:
    raise SystemExit(
        f"Expected at least {expected_min_candidates} candidate rows "
        f"({expected_prompts} prompts * {num_candidates} candidates), found {len(candidate_rows)}"
    )
for row in candidate_rows[: min(100, len(candidate_rows))]:
    if not (0 <= int(row["candidate_index"]) < num_candidates):
        raise SystemExit(f"Bad candidate_index in row: {row.get('example_id')}")
    if "mc_reward" not in row or "collector_value" not in row:
        raise SystemExit(f"Missing reward/value fields in row: {row.get('example_id')}")
print(
    f"Validated merged outputs: prompts={len(prompt_rows)}, "
    f"candidate_rows={len(candidate_rows)}, expected_min_candidates={expected_min_candidates}"
)
PY
}

# -----------------------------
# Preflight checks
# -----------------------------
echo "Run ID: $RUN_ID"
echo "Python: $(which python3)"
python3 -V
echo "WORK_DIR: $WORK_DIR"
echo "RUN_DIR: $RUN_DIR"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "MERGED_ROOT: $MERGED_ROOT"
echo "CACHE_ROOT: $CACHE_ROOT"
echo "ARCHIVE_DIR: $ARCHIVE_DIR"

cd "$WORK_DIR"
ls -ld "$WORK_DIR"
ls -lh "$DATASET_PATH"
ls -ld "$ACTOR_CHECKPOINT_DIR"
ls -ld "$COLLECTOR_CRITIC_CHECKPOINT_DIR"
check_cuda_devices
ensure_conda_cuda_runtime_link
validate_component_checkpoint "$ACTOR_CHECKPOINT_DIR" actor
validate_component_checkpoint "$COLLECTOR_CRITIC_CHECKPOINT_DIR" critic
if [[ -f "$SCRIPT_PATH" ]]; then
  cp "$SCRIPT_PATH" "$LOG_DIR/$(basename "$SCRIPT_PATH")"
fi

if [[ "$VLLM_TENSOR_PARALLEL_SIZE" != "1" ]]; then
  echo "This launcher runs one process per visible GPU; forcing VLLM_TENSOR_PARALLEL_SIZE=1." >&2
  VLLM_TENSOR_PARALLEL_SIZE=1
fi

DATASET_ROWS="$(count_dataset_rows)"
if [[ "$START_INDEX" -lt 0 || "$START_INDEX" -ge "$DATASET_ROWS" ]]; then
  echo "START_INDEX=$START_INDEX is outside dataset row count $DATASET_ROWS" >&2
  exit 1
fi
TOTAL_PROMPTS="$MAX_PROMPTS"
if [[ -n "$DEBUG_NUM_PROMPTS" && "$DEBUG_NUM_PROMPTS" -gt 0 && "$DEBUG_NUM_PROMPTS" -lt "$TOTAL_PROMPTS" ]]; then
  TOTAL_PROMPTS="$DEBUG_NUM_PROMPTS"
fi
MAX_AVAILABLE=$((DATASET_ROWS - START_INDEX))
if [[ "$TOTAL_PROMPTS" -gt "$MAX_AVAILABLE" ]]; then
  TOTAL_PROMPTS="$MAX_AVAILABLE"
fi
if [[ "$TOTAL_PROMPTS" -le 0 ]]; then
  echo "No prompts selected after bounds/debug handling." >&2
  exit 1
fi
PROMPTS_PER_SHARD=$(((TOTAL_PROMPTS + NUM_LOCAL_SHARDS - 1) / NUM_LOCAL_SHARDS))

IFS=',' read -r -a CUDA_DEVICES <<< "$CUDA_DEVICES_CSV"

echo "Dataset rows: $DATASET_ROWS"
echo "Total prompts requested after bounds/debug: $TOTAL_PROMPTS"
echo "Num local shards: $NUM_LOCAL_SHARDS"
echo "Prompts per shard: $PROMPTS_PER_SHARD"
echo "CUDA devices: ${CUDA_DEVICES[*]}"

if [[ "$DRY_RUN" != "0" ]]; then
  echo "DRY_RUN=1 set; preflight succeeded without launching collectors."
  exit 0
fi

echo "Preparing shared merged Hugging Face checkpoints..."
ensure_shared_merged_checkpoints

# -----------------------------
# Run collection shards
# -----------------------------
pids=()
for shard_index in $(seq 0 $((NUM_LOCAL_SHARDS - 1))); do
  shard_start=$((START_INDEX + shard_index * PROMPTS_PER_SHARD))
  remaining=$((START_INDEX + TOTAL_PROMPTS - shard_start))
  if [[ "$remaining" -le 0 ]]; then
    echo "Skipping shard $shard_index: no prompts assigned."
    continue
  fi
  shard_max=$PROMPTS_PER_SHARD
  if [[ "$shard_max" -gt "$remaining" ]]; then
    shard_max=$remaining
  fi

  gpu_id="${CUDA_DEVICES[$shard_index]}"
  shard_dir="${SHARD_ROOT}/shard_$(printf '%03d' "$shard_index")"
  shard_log="${LOG_DIR}/collector_shard_$(printf '%03d' "$shard_index").log"
  mkdir -p "$shard_dir"

  CMD=(
    python3 value_decoding/collect_search_induced_critic_data.py
    --actor_checkpoint_dir "$ACTOR_CHECKPOINT_DIR"
    --collector_critic_checkpoint_dir "$COLLECTOR_CRITIC_CHECKPOINT_DIR"
    --dataset_path "$DATASET_PATH"
    --output_dir "$shard_dir"
    --prompt_key "$PROMPT_KEY"
    --start_index "$shard_start"
    --max_prompts "$shard_max"
    --chunk_size "$CHUNK_SIZE"
    --num_chunk_candidates "$NUM_CHUNK_CANDIDATES"
    --num_search_steps_per_prompt "$NUM_SEARCH_STEPS_PER_PROMPT"
    --completion_max_new_tokens "$COMPLETION_MAX_NEW_TOKENS"
    --collector_selection_mode "$COLLECTOR_SELECTION_MODE"
    --collector_epsilon "$COLLECTOR_EPSILON"
    --collector_value_temperature "$COLLECTOR_VALUE_TEMPERATURE"
    --actor_temperature "$ACTOR_TEMPERATURE"
    --actor_top_p "$ACTOR_TOP_P"
    --actor_top_k "$ACTOR_TOP_K"
    --actor_batch_size "$ACTOR_BATCH_SIZE"
    --max_prompt_length "$MAX_PROMPT_LENGTH"
    --dtype "$DTYPE"
    --seed "$((SEED + shard_index))"
    --generation_backend "$GENERATION_BACKEND"
    --critic_device cuda:0
    --merged_root "$MERGED_ROOT"
    --critic_batch_size "$CRITIC_BATCH_SIZE"
    --vllm_gpu_memory_utilization "$VLLM_GPU_MEMORY_UTILIZATION"
    --vllm_tensor_parallel_size "$VLLM_TENSOR_PARALLEL_SIZE"
  )

  [[ -n "$RESPONSE_KEY" ]] && CMD+=(--response_key "$RESPONSE_KEY")
  [[ -n "$ACTOR_HF_SOURCE_DIR" ]] && CMD+=(--actor_hf_source_dir "$ACTOR_HF_SOURCE_DIR")
  [[ -n "$CRITIC_HF_SOURCE_DIR" ]] && CMD+=(--critic_hf_source_dir "$CRITIC_HF_SOURCE_DIR")
  [[ -n "$VLLM_MAX_MODEL_LEN" ]] && CMD+=(--vllm_max_model_len "$VLLM_MAX_MODEL_LEN")
  [[ "$SHUFFLE_PROMPTS" != "0" ]] && CMD+=(--shuffle_prompts true)
  [[ "$VLLM_ENFORCE_EAGER" != "0" ]] && CMD+=(--vllm_enforce_eager true)
  [[ "$SAVE_FULL_TEXT" == "0" ]] && CMD+=(--save_full_text false)
  [[ "$SAVE_TOKEN_IDS" == "0" ]] && CMD+=(--save_token_ids false)
  [[ "$SAVE_COMPLETED_RESPONSES" != "0" ]] && CMD+=(--save_completed_responses true)
  [[ "$TRUST_REMOTE_CODE" != "0" ]] && CMD+=(--trust_remote_code true)
  [[ "$SKIP_MERGE" != "0" ]] && CMD+=(--skip_merge true)

  shard_runner="${LOG_DIR}/run_collector_shard_$(printf '%03d' "$shard_index").sh"
  {
    echo "#!/bin/bash"
    echo "set -eo pipefail"
    printf 'source %q\n' "/data/shuozhe/miniconda3/etc/profile.d/conda.sh"
    printf 'conda activate %q\n' "verl"
    echo "set -u"
    printf 'cd %q\n' "$WORK_DIR"
    printf 'export CUDA_VISIBLE_DEVICES=%q\n' "$gpu_id"
    printf 'export PYTHONPATH=%q\n' "$PYTHONPATH"
    printf 'export PYTHONUNBUFFERED=%q\n' "$PYTHONUNBUFFERED"
    printf 'export TOKENIZERS_PARALLELISM=%q\n' "$TOKENIZERS_PARALLELISM"
    printf 'export VLLM_USE_V1=%q\n' "$VLLM_USE_V1"
    printf 'export HF_HOME=%q\n' "$HF_HOME"
    printf 'export TIKTOKEN_ENCODINGS_BASE=%q\n' "$TIKTOKEN_ENCODINGS_BASE"
    printf 'export XDG_CACHE_HOME=%q\n' "$XDG_CACHE_HOME"
    printf 'export TORCHINDUCTOR_CACHE_DIR=%q\n' "$TORCHINDUCTOR_CACHE_DIR"
    printf 'export TRITON_CACHE_DIR=%q\n' "$TRITON_CACHE_DIR"
    printf 'export TORCH_EXTENSIONS_DIR=%q\n' "$TORCH_EXTENSIONS_DIR"
    printf 'export VLLM_CACHE_ROOT=%q\n' "$VLLM_CACHE_ROOT"
    printf 'export FLASHINFER_WORKSPACE_BASE=%q\n' "$FLASHINFER_WORKSPACE_BASE"
    printf 'export DO_NOT_TRACK=%q\n' "$DO_NOT_TRACK"
    printf 'export VLLM_NO_USAGE_STATS=%q\n' "$VLLM_NO_USAGE_STATS"
    printf 'export CUDA_HOME=%q\n' "$CUDA_HOME"
    printf 'export LD_LIBRARY_PATH=%q\n' "$LD_LIBRARY_PATH"
    printf 'export LIBRARY_PATH=%q\n' "$LIBRARY_PATH"
    printf 'export LDFLAGS=%q\n' "$LDFLAGS"
    printf 'exec'
    printf ' %q' "${CMD[@]}"
    printf '\n'
  } > "$shard_runner"
  chmod +x "$shard_runner"

  {
    echo "Shard $shard_index on physical GPU $gpu_id"
    echo "Shard start: $shard_start"
    echo "Shard max prompts: $shard_max"
    echo "CUDA_VISIBLE_DEVICES=$gpu_id"
    printf 'Command:'
    printf ' %q' "${CMD[@]}"
    printf '\n'
    echo "Runner: $shard_runner"
  } > "$shard_log"

  echo "Launching shard $shard_index on GPU $gpu_id: start=$shard_start max=$shard_max"
  bash "$shard_runner" >> "$shard_log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" != "0" ]]; then
  echo "At least one collection shard failed. See $LOG_DIR/collector_shard_*.log" >&2
  exit 1
fi

# -----------------------------
# Merge and validate shard outputs
# -----------------------------
merge_shards
validate_merged_outputs

echo "Search-induced critic data collection finished successfully."
echo "Merged output: $OUTPUT_DIR"
