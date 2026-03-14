#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export VLLM_USE_V1=1
export HYDRA_FULL_ERROR=0
export WANDB_MODE=disabled

CHECKPOINT_ROOT=/data/shuozhe/verl/train_log/job_05b_vh_init_e5_new_lvl45
TRAIN_FILE=/data/shuozhe/saved_dataset/verl_math_7500_500_5000_level_4_5/train.parquet
VAL_FILE=/data/shuozhe/saved_dataset/verl_math_7500_500_5000_level_5/test_5000.parquet
MODEL_PATH=/data/shuozhe/saved_model/Qwen2.5-0.5B

N_GPUS=4
NNODES=1
TP_SIZE=1
GPU_MEM_UTIL=0.4

TRAIN_BATCH_SIZE=32
MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=2048
ROLLOUT_N=8
VAL_N=1
VAL_DO_SAMPLE=false

START_STEP=0
END_STEP=999999999
SKIP_EXISTING=true
CONTINUE_ON_ERROR=true

EVAL_ROOT=${CHECKPOINT_ROOT}/eval_ckpt_val_only
PROJECT_NAME=PPO_eval
EXPERIMENT_PREFIX=job_05b_vh_init_load_c1100_e5_freeze_critic_lvl45

python3 /data/shuozhe/verl/eval_scripts/eval_05b_vh_init_load_freeze_ckpts.py \
  --checkpoint-root="${CHECKPOINT_ROOT}" \
  --train-file="${TRAIN_FILE}" \
  --val-file="${VAL_FILE}" \
  --model-path="${MODEL_PATH}" \
  --n-gpus="${N_GPUS}" \
  --nnodes="${NNODES}" \
  --tp-size="${TP_SIZE}" \
  --gpu-mem-util="${GPU_MEM_UTIL}" \
  --train-batch-size="${TRAIN_BATCH_SIZE}" \
  --max-prompt-length="${MAX_PROMPT_LENGTH}" \
  --max-response-length="${MAX_RESPONSE_LENGTH}" \
  --rollout-n="${ROLLOUT_N}" \
  --val-n="${VAL_N}" \
  --val-do-sample="${VAL_DO_SAMPLE}" \
  --start-step="${START_STEP}" \
  --end-step="${END_STEP}" \
  --skip-existing="${SKIP_EXISTING}" \
  --continue-on-error="${CONTINUE_ON_ERROR}" \
  --eval-root="${EVAL_ROOT}" \
  --project-name="${PROJECT_NAME}" \
  --experiment-prefix="${EXPERIMENT_PREFIX}" \
  "$@"
