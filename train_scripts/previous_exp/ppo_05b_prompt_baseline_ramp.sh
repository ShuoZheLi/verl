export PYTHONUNBUFFERED=1
export VLLM_USE_V1=0
export HYDRA_FULL_ERROR=0
export VLLM_USE_V1=1
export WANDB_PROJECT="PPO_midi"
export SLURM_JOB_ID="05b_prompt_residual_ramp"

# When true, math_dapo incorrect answers get reward 0.0 instead of -1.0.
MATH_DAPO_BINARY_REWARD=true
# When true, responses that only stop because they hit max response length are
# fully masked out of PPO/critic optimization.
OVERLONG_FILTERING=false
# RayPPOTrainer starts counting global PPO steps from 1 and begins actor
# updates once global_steps >= trainer.critic_warmup.
# Use +1 here to get exactly 50 critic-only optimization steps.
CRITIC_ONLY_STEPS=25
PROMPT_RESIDUAL_RAMP_STEPS=100
  # trainer.critic_warmup=$((CRITIC_ONLY_STEPS + 1)) \

python3 -m verl.trainer.main_ppo \
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
  algorithm.adv_estimator=prompt_residual_baseline_ramp \
  algorithm.gamma=1.0 \
  algorithm.lam=1.0 \
  ++algorithm.prompt_residual_alpha=1.0 \
  ++algorithm.prompt_residual_alpha_ramp_steps=${PROMPT_RESIDUAL_RAMP_STEPS} \
  critic.enable=True \
  critic.optim.lr=1e-5 \
  critic.model.path=/data/shuozhe/saved_model/Qwen2.5-0.5B \
  critic.model.external_lib=trl \
  critic.model.value_head_init_mean=0.0 \
  critic.model.value_head_init_std=0.00001 \
  critic.model.fsdp_config.param_offload=False \
  critic.prompt_residual_prompt_loss_weight=1.0 \
  critic.prompt_residual_residual_loss_weight=1.0 \
  critic.ppo_micro_batch_size_per_gpu=4 \
  +reward.reward_kwargs.math_dapo_binary_reward=${MATH_DAPO_BINARY_REWARD} \
  trainer.use_legacy_worker_impl=enable \
  trainer.val_before_train=True \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  trainer.test_freq=50 \
  trainer.save_freq=50 \
  trainer.total_epochs=5 \
  trainer.overlong_filtering=${OVERLONG_FILTERING} \
  trainer.logger='["console","wandb"]' \
  trainer.project_name="PPO_metamath" \
  trainer.experiment_name="qwen2.5_0.5B_prompt_residual_${SLURM_JOB_ID}" \
  trainer.default_local_dir="/data/shuozhe/verl/train_log/job_${SLURM_JOB_ID}" \
  2>&1 | tee /data/shuozhe/verl/train_log/job_${SLURM_JOB_ID}.txt
