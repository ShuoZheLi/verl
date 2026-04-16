#!/usr/bin/env bash

set -euo pipefail

REPO_DIR="/data/shuozhe/verl"
ACTIVATE_SCRIPT="/data/shuozhe/miniconda3/bin/activate"
TARGET_SCRIPT="${REPO_DIR}/train_scripts/ppo_05b_prompt_baseline.sh"
LOG_DIR="${REPO_DIR}/train_log"
LAUNCHER_LOG="${LOG_DIR}/auto_exp_launcher.log"
SESSION_NAME="${AUTO_EXP_TMUX_SESSION:-auto_exp}"
WAIT_SECONDS=$((4 * 60 * 60))
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"

timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is not available on PATH." >&2
  exit 1
fi

if [[ ! -f "${ACTIVATE_SCRIPT}" ]]; then
  echo "Missing conda activate script: ${ACTIVATE_SCRIPT}" >&2
  exit 1
fi

if [[ ! -f "${TARGET_SCRIPT}" ]]; then
  echo "Missing target script: ${TARGET_SCRIPT}" >&2
  exit 1
fi

mkdir -p "${LOG_DIR}"

# Re-launch under nohup so the scheduler keeps running even if the current
# terminal or tmux server goes away before the delayed start happens.
if [[ -z "${AUTO_EXP_FOREGROUND:-}" ]]; then
  nohup env AUTO_EXP_FOREGROUND=1 bash "${SCRIPT_PATH}" >> "${LAUNCHER_LOG}" 2>&1 &
  echo "Background launcher started as PID $!. Logs: ${LAUNCHER_LOG}"
  exit 0
fi

echo "[$(timestamp)] Waiting 4 hours before restarting tmux and launching training."
sleep "${WAIT_SECONDS}"

echo "[$(timestamp)] Killing existing tmux server."
tmux kill-server >/dev/null 2>&1 || true

echo "[$(timestamp)] Starting tmux session '${SESSION_NAME}'."
tmux new-session -d -s "${SESSION_NAME}" \
  "bash -lc 'cd \"${REPO_DIR}\" && source \"${ACTIVATE_SCRIPT}\" verl && bash \"${TARGET_SCRIPT}\"'"

echo "[$(timestamp)] Training launched in tmux session '${SESSION_NAME}'."
echo "[$(timestamp)] Attach with: tmux attach -t ${SESSION_NAME}"
