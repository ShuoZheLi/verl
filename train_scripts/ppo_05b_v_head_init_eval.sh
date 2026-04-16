#!/usr/bin/env bash
set -euo pipefail

source /data/shuozhe/miniconda3/etc/profile.d/conda.sh
conda activate verl

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=0
export VLLM_USE_V1=1


# This will try to reproduce the result from wandb curve

# -----------------------------
# User config
# -----------------------------

# Checkpoint selection:
TRAIN_JOB_DIR="/data/shuozhe/verl/train_log/job_05b_critic_dsk-1d5b"
# Set either CKPT_PATH directly, or set CKPT_STEP, or leave both empty
# to use TRAIN_JOB_DIR/latest_checkpointed_iteration.txt
CKPT_PATH=""
CKPT_STEP=""

# Dataset:
TRAIN_FILE="/data/shuozhe/saved_dataset/MetaMathQA-math-500/train.parquet"
VAL_FILE="/data/shuozhe/saved_dataset/MetaMathQA-math-500/test.parquet"

# Base models for the actor and critic. The checkpoint will be loaded on top of these.
BASE_ACTOR_MODEL="/data/shuozhe/saved_model/Qwen2.5-0.5B"
BASE_CRITIC_MODEL="/data/shuozhe/saved_model/DeepSeek-R1-Distill-Qwen-1.5B"

N_GPUS_PER_NODE=4
NNODES=1
MATH_DAPO_BINARY_REWARD=true


# Store the eval result here!!!!!!!!!!!!
EVAL_TAG="metamath500_ckpt_eval"
EVAL_LOG_DIR="${TRAIN_JOB_DIR}/standalone_eval"
VALIDATION_DATA_DIR="${EVAL_LOG_DIR}/validation_data"
LOG_FILE=""

# -----------------------------
# Resolve checkpoint
# -----------------------------
if [[ -n "$CKPT_PATH" && -n "$CKPT_STEP" ]]; then
  echo "Set only one of CKPT_PATH or CKPT_STEP." >&2
  exit 1
fi

if [[ -z "$CKPT_PATH" ]]; then
  if [[ -n "$CKPT_STEP" ]]; then
    CKPT_PATH="${TRAIN_JOB_DIR}/global_step_${CKPT_STEP}"
  else
    TRACKER_FILE="${TRAIN_JOB_DIR}/latest_checkpointed_iteration.txt"
    if [[ ! -f "$TRACKER_FILE" ]]; then
      echo "Could not find ${TRACKER_FILE}. Set CKPT_PATH or CKPT_STEP explicitly." >&2
      exit 1
    fi
    CKPT_STEP="$(tr -d '[:space:]' < "$TRACKER_FILE")"
    CKPT_PATH="${TRAIN_JOB_DIR}/global_step_${CKPT_STEP}"
  fi
fi

CKPT_BASENAME="$(basename "$CKPT_PATH")"
if [[ ! "$CKPT_BASENAME" =~ ^global_step_[0-9]+$ ]]; then
  echo "Checkpoint path must end with global_step_<step>: ${CKPT_PATH}" >&2
  exit 1
fi

if [[ -z "$CKPT_STEP" ]]; then
  CKPT_STEP="${CKPT_BASENAME#global_step_}"
fi

if [[ ! -d "$CKPT_PATH" ]]; then
  echo "Checkpoint directory does not exist: ${CKPT_PATH}" >&2
  exit 1
fi

if [[ ! -d "${CKPT_PATH}/actor" ]]; then
  echo "Missing actor checkpoint under ${CKPT_PATH}" >&2
  exit 1
fi

if [[ ! -d "${CKPT_PATH}/critic" ]]; then
  echo "Missing critic checkpoint under ${CKPT_PATH}" >&2
  exit 1
fi

EXPECTED_WORLD_SIZE=$((N_GPUS_PER_NODE * NNODES))
ACTOR_WORLD_SIZE_FILE="$(find "${CKPT_PATH}/actor" -maxdepth 1 -type f -name 'model_world_size_*_rank_0.pt' | head -n 1)"

if [[ -n "$ACTOR_WORLD_SIZE_FILE" ]]; then
  CKPT_WORLD_SIZE="$(basename "$ACTOR_WORLD_SIZE_FILE" | sed -E 's/model_world_size_([0-9]+)_rank_0\.pt/\1/')"
  if [[ "$CKPT_WORLD_SIZE" != "$EXPECTED_WORLD_SIZE" ]]; then
    echo "Checkpoint world size is ${CKPT_WORLD_SIZE}, but N_GPUS_PER_NODE * NNODES is ${EXPECTED_WORLD_SIZE}." >&2
    echo "Set N_GPUS_PER_NODE and NNODES to match the checkpoint shard count." >&2
    exit 1
  fi
fi

if [[ -z "$LOG_FILE" ]]; then
  LOG_FILE="${EVAL_LOG_DIR}/${EVAL_TAG}_step_${CKPT_STEP}.log"
fi

mkdir -p "$EVAL_LOG_DIR"
mkdir -p "$VALIDATION_DATA_DIR"

echo "Evaluating checkpoint: $CKPT_PATH"
echo "Validation file: $VAL_FILE"
echo "Validation generations will be written to: $VALIDATION_DATA_DIR"
echo "Console log will be written to: $LOG_FILE"

python3 -m verl.trainer.main_ppo \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$VAL_FILE" \
  data.prompt_key=prompt \
  +data.response_key=ground_truth \
  data.train_batch_size=32 \
  data.validation_shuffle=False \
  data.max_prompt_length=2048 \
  data.max_response_length=2048 \
  actor_rollout_ref.model.path="$BASE_ACTOR_MODEL" \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.actor.calculate_sum_pi_squared=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
  actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=False \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
  actor_rollout_ref.hybrid_engine=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  critic.optim.lr=1e-5 \
  critic.model.path="$BASE_CRITIC_MODEL" \
  critic.model.external_lib=trl \
  critic.model.value_head_init_mean=0.0 \
  critic.model.value_head_init_std=0.00001 \
  critic.model.fsdp_config.param_offload=False \
  critic.ppo_micro_batch_size_per_gpu=4 \
  +reward.reward_kwargs.math_dapo_binary_reward="$MATH_DAPO_BINARY_REWARD" \
  trainer.resume_mode=resume_path \
  trainer.resume_from_path="$CKPT_PATH" \
  trainer.val_before_train=True \
  trainer.val_only=True \
  trainer.n_gpus_per_node="$N_GPUS_PER_NODE" \
  trainer.nnodes="$NNODES" \
  trainer.test_freq=-1 \
  trainer.save_freq=-1 \
  trainer.total_epochs=1 \
  trainer.logger='["console"]' \
  trainer.project_name="PPO_metamath_eval" \
  trainer.experiment_name="qwen2.5_0.5B_ppo_valuehead_05b_critic_dsk-1d5b_eval" \
  trainer.default_local_dir="$TRAIN_JOB_DIR" \
  trainer.validation_data_dir="$VALIDATION_DATA_DIR" \
  2>&1 | tee "$LOG_FILE"