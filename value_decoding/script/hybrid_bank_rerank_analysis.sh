#!/usr/bin/env bash

set -eo pipefail

# =============================================================================
# HYBRID TRAJECTORY RERANKING ON EXISTING STAGE 2 BANK
# Pure post-hoc analysis:
# - reuses the saved Stage 2 shared response bank
# - performs no new actor generation
# - performs no new critic scoring
# =============================================================================

# --- Input Files --------------------------------------------------------------
STAGE2_OUTPUT_DIR="/data/shuozhe/verl/value_decoding/output/stage2_best_of_n_inference_eval"
TRAJECTORY_BANK_PATH="${STAGE2_OUTPUT_DIR}/trajectory_bank.jsonl"
PROMPT_SUMMARY_PATH="${STAGE2_OUTPUT_DIR}/prompt_level_summary.jsonl"
SUMMARY_METRICS_PATH="${STAGE2_OUTPUT_DIR}/summary_metrics.json"

# --- Output -------------------------------------------------------------------
OUTPUT_DIR="/data/shuozhe/verl/value_decoding/output/hybrid_bank_rerank_analysis"

# --- Analysis Config ----------------------------------------------------------
# Leave BANK_SIZE empty to infer it from the saved bank.
BANK_SIZE=""
NORMALIZATION="zscore"    # Options: zscore, rank, minmax
NORMALIZATION_EPS=1e-6
LAMBDAS="0.0 0.1 0.25 0.5 1.0 2.0"
INCLUDE_NEGATIVE_LAMBDAS=0
BOOTSTRAP_SAMPLES=2000

# --- Validation / Plots -------------------------------------------------------
# Leave SEED empty to inherit the Stage 2 seed from summary_metrics.json.
SEED=""
SKIP_PLOTS=0
PLOT_DPI=160

source /data/shuozhe/miniconda3/bin/activate verl
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

read -r -a LAMBDAS_ARR <<< "${LAMBDAS}"

mkdir -p "${OUTPUT_DIR}"

CMD=(
  python -m value_decoding.hybrid_bank_rerank_analysis
  --trajectory_bank_path "${TRAJECTORY_BANK_PATH}"
  --output_dir           "${OUTPUT_DIR}"
  --normalization        "${NORMALIZATION}"
  --eps                  "${NORMALIZATION_EPS}"
  --bootstrap_samples    "${BOOTSTRAP_SAMPLES}"
  --plot_dpi             "${PLOT_DPI}"
  --lambdas              "${LAMBDAS_ARR[@]}"
)

[[ -n "${PROMPT_SUMMARY_PATH}" ]] && CMD+=(--prompt_summary_path "${PROMPT_SUMMARY_PATH}")
[[ -n "${SUMMARY_METRICS_PATH}" ]] && CMD+=(--summary_metrics_path "${SUMMARY_METRICS_PATH}")
[[ -n "${BANK_SIZE}" ]] && CMD+=(--bank_size "${BANK_SIZE}")
[[ -n "${SEED}" ]] && CMD+=(--seed "${SEED}")
[[ "${INCLUDE_NEGATIVE_LAMBDAS}" != "0" ]] && CMD+=(--include_negative_lambdas)
[[ "${SKIP_PLOTS}" != "0" ]] && CMD+=(--skip_plots)

(cd "${REPO_DIR}" && "${CMD[@]}")
