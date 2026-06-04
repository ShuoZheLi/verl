export PYTHONUNBUFFERED=1
export VLLM_USE_V1=0
export HYDRA_FULL_ERROR=0
export VLLM_USE_V1=1
export WANDB_PROJECT="PPO_midi"

# Keep Python tempfile, Ray session files, and Ray object spilling off /tmp.
# /tmp is often a small local partition on shared GPU nodes.
export TMPDIR=/data/shuozhe/tmp
export TEMP=${TMPDIR}
export TMP=${TMPDIR}
export RAY_TMPDIR=${TMPDIR}/ray
export RAY_OBJECT_SPILLING_DIR=${TMPDIR}/ray_spilled_objects
mkdir -p "${TMPDIR}" "${RAY_TMPDIR}" "${RAY_OBJECT_SPILLING_DIR}"
export SLURM_JOB_ID="qwen3-0.6B_critic_only_from_policy"

# When true, math_dapo incorrect answers get reward 0.0 instead of -1.0.
MATH_DAPO_BINARY_REWARD=true
POLICY_INIT_CKPT=/data/shuozhe/saved_model/Qwen3-0.6B
CRITIC_INIT_CKPT=/data/shuozhe/saved_model/Qwen3-0.6B
CRITIC_ONLY_STEPS=1000000000

python3 -m verl.trainer.main_ppo \
  +ray_kwargs.ray_init._temp_dir=${RAY_TMPDIR} \
  +ray_kwargs.ray_init.object_spilling_directory=${RAY_OBJECT_SPILLING_DIR} \
  +ray_kwargs.ray_init.runtime_env.env_vars.TMPDIR=${TMPDIR} \
  +ray_kwargs.ray_init.runtime_env.env_vars.TEMP=${TEMP} \
  +ray_kwargs.ray_init.runtime_env.env_vars.TMP=${TMP} \
  data.train_files=/data/shuozhe/saved_dataset/MetaMathQA-math-500/train.parquet \
  data.val_files=/data/shuozhe/saved_dataset/MetaMathQA-math-500/test.parquet \
  data.prompt_key=prompt \
  +data.response_key=ground_truth \
  data.train_batch_size=64 \
  data.max_prompt_length=2048 \
  data.max_response_length=2048 \
  actor_rollout_ref.model.path=${POLICY_INIT_CKPT} \
  actor_rollout_ref.rollout.max_model_len=4096 \
  actor_rollout_ref.actor.optim.lr=0.0 \
  actor_rollout_ref.actor.ppo_mini_batch_size=32 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.actor.calculate_sum_pi_squared=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.free_cache_engine=True \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
  actor_rollout_ref.hybrid_engine=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
  algorithm.adv_estimator=token_success_bce \
  algorithm.gamma=1.0 \
  algorithm.lam=1.0 \
  critic.optim.lr=1e-5 \
  critic.model.path=${CRITIC_INIT_CKPT} \
  critic.model.external_lib=trl \
  critic.model.fsdp_config.param_offload=False \
  critic.ppo_micro_batch_size_per_gpu=4 \
  +reward.reward_kwargs.math_dapo_binary_reward=${MATH_DAPO_BINARY_REWARD} \
  trainer.resume_mode=disable \
  trainer.critic_warmup=${CRITIC_ONLY_STEPS} \
  trainer.val_before_train=True \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  trainer.test_freq=50 \
  trainer.save_freq=50 \
  trainer.total_epochs=5 \
  trainer.logger='["console","wandb"]' \
  trainer.project_name="PPO_metamath" \
  trainer.experiment_name="${SLURM_JOB_ID}" \
  trainer.default_local_dir="/data/shuozhe/verl/train_log/${SLURM_JOB_ID}" \
  2>&1 | tee /data/shuozhe/verl/train_log/${SLURM_JOB_ID}.txt
