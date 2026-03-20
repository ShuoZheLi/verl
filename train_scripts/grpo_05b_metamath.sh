#!/usr/bin/env bash

# source /data/shuozhe/miniconda3/bin/activate verl

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=0
export VLLM_USE_V1=1
export WANDB_PROJECT="GRPO_metamath"
export SLURM_JOB_ID="${SLURM_JOB_ID:-05b_grpo_init_e5}"

# When true, math_dapo incorrect answers get reward 0.0 instead of -1.0.
MATH_DAPO_BINARY_REWARD=true

# data.train_files=/data/shuozhe/saved_dataset/verl_math_7500_500_5000/train.parquet \
# data.val_files=/data/shuozhe/saved_dataset/verl_math_7500_500_5000/test.parquet \

mkdir -p /data/shuozhe/verl/train_log

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=/data/shuozhe/saved_dataset/MetaMathQA-math-500/train.parquet \
  data.val_files=/data/shuozhe/saved_dataset/MetaMathQA-math-500/test.parquet \
  data.prompt_key=prompt \
  +data.response_key=ground_truth \
  data.train_batch_size=32 \
  data.max_prompt_length=2048 \
  data.max_response_length=2048 \
  actor_rollout_ref.model.path=/data/shuozhe/saved_model/Qwen2.5-0.5B \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.actor.calculate_sum_pi_squared=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
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
  algorithm.use_kl_in_reward=False \
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
  trainer.default_local_dir="/data/shuozhe/verl/train_log/job_${SLURM_JOB_ID}" \
  "$@" 2>&1 | tee /data/shuozhe/verl/train_log/job_${SLURM_JOB_ID}.txt
