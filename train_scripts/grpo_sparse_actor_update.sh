#!/usr/bin/env bash

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=0
export VLLM_USE_V1=1
export WANDB_PROJECT="GRPO_midi"
export SLURM_JOB_ID="05b_grpo_sparse_safe_svd_lowmag"
TRAIN_LOG_DIR="${TRAIN_LOG_DIR:-/data/shuozhe/verl/train_log}"

SPARSE_UPDATE_MODE="${SPARSE_UPDATE_MODE:-safe_svd_lowmag}"
SPARSE_UPDATE_RANK_K="${SPARSE_UPDATE_RANK_K:-128}"
SPARSE_UPDATE_ALPHA_PRINC="${SPARSE_UPDATE_ALPHA_PRINC:-0.5}"
SPARSE_UPDATE_ALPHA_LOW="${SPARSE_UPDATE_ALPHA_LOW:-0.5}"
SPARSE_UPDATE_MASK_PATH="${SPARSE_UPDATE_MASK_PATH:-}"
SPARSE_UPDATE_SAVE_MASK_PATH="${SPARSE_UPDATE_SAVE_MASK_PATH:-${TRAIN_LOG_DIR}/job_${SLURM_JOB_ID}/sparse_update_mask.pt}"
SPARSE_UPDATE_VERIFY="${SPARSE_UPDATE_VERIFY:-true}"
SPARSE_UPDATE_VERIFY_INTERVAL="${SPARSE_UPDATE_VERIFY_INTERVAL:-100}"

SPARSE_UPDATE_MASK_OVERRIDES=(
  actor_rollout_ref.actor.sparse_update.enabled=true
  actor_rollout_ref.actor.sparse_update.mode="${SPARSE_UPDATE_MODE}"
  actor_rollout_ref.actor.sparse_update.rank_k="${SPARSE_UPDATE_RANK_K}"
  actor_rollout_ref.actor.sparse_update.alpha_princ="${SPARSE_UPDATE_ALPHA_PRINC}"
  actor_rollout_ref.actor.sparse_update.alpha_low="${SPARSE_UPDATE_ALPHA_LOW}"
  actor_rollout_ref.actor.sparse_update.restore_frozen_after_step=true
  actor_rollout_ref.actor.sparse_update.mask_optimizer_state=true
  actor_rollout_ref.actor.sparse_update.verify_frozen_weights="${SPARSE_UPDATE_VERIFY}"
  actor_rollout_ref.actor.sparse_update.verification_interval="${SPARSE_UPDATE_VERIFY_INTERVAL}"
  actor_rollout_ref.actor.fsdp_config.use_orig_params=True
)

if [[ -n "${SPARSE_UPDATE_MASK_PATH}" ]]; then
  SPARSE_UPDATE_MASK_OVERRIDES+=(
    actor_rollout_ref.actor.sparse_update.mask_path="${SPARSE_UPDATE_MASK_PATH}"
  )
else
  SPARSE_UPDATE_MASK_OVERRIDES+=(
    actor_rollout_ref.actor.sparse_update.build_mask_on_init=true
    actor_rollout_ref.actor.sparse_update.save_mask_path="${SPARSE_UPDATE_SAVE_MASK_PATH}"
  )
fi

# When true, math_dapo incorrect answers get reward 0.0 instead of -1.0.
MATH_DAPO_BINARY_REWARD=true
  # data.train_files=/data/shuozhe/saved_dataset/verl_math_7500_500_5000/train.parquet \
  # data.val_files=/data/shuozhe/saved_dataset/verl_math_7500_500_5000/test.parquet \

mkdir -p "${TRAIN_LOG_DIR}"

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files=/data/shuozhe/saved_dataset/MetaMathQA-math-500/train.parquet \
  data.val_files=/data/shuozhe/saved_dataset/MetaMathQA-math-500/test.parquet \
  data.prompt_key=prompt \
  data.train_batch_size=32 \
  data.max_prompt_length=2048 \
  data.max_response_length=2048 \
  actor_rollout_ref.model.path=/data/shuozhe/saved_model/Qwen2.5-0.5B \
  actor_rollout_ref.model.use_remove_padding=False \
  +actor_rollout_ref.model.override_config.attn_implementation=eager \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.actor.calculate_sum_pi_squared=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  "${SPARSE_UPDATE_MASK_OVERRIDES[@]}" \
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
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
  actor_rollout_ref.hybrid_engine=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  +reward.reward_kwargs.math_dapo_binary_reward=${MATH_DAPO_BINARY_REWARD} \
  critic.enable=False \
  trainer.critic_warmup=0 \
  trainer.val_before_train=True \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  trainer.test_freq=50 \
  trainer.save_freq=50 \
  trainer.total_epochs=5 \
  trainer.logger='["console","wandb"]' \
  trainer.project_name="GRPO_metamath" \
  trainer.experiment_name="qwen2.5_0.5B_grpo_${SLURM_JOB_ID}" \
  trainer.default_local_dir="${TRAIN_LOG_DIR}/job_${SLURM_JOB_ID}" \
  "$@" \
  2>&1 | tee "${TRAIN_LOG_DIR}/job_${SLURM_JOB_ID}.txt"
