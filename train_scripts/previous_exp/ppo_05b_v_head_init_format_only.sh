export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=0
export VLLM_USE_V1=1
export WANDB_PROJECT="${WANDB_PROJECT:-PPO_midi}"
export SLURM_JOB_ID="${SLURM_JOB_ID:-05b_vh_init_format_only_s10}"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REWARD_FN_PATH="${PROJECT_DIR}/train_scripts/reward_math_boxed_format_only.py"

TRAIN_FILE="${TRAIN_FILE:-/data/shuozhe/saved_dataset/MetaMathQA-math-500/train.parquet}"
VAL_FILE="${VAL_FILE:-/data/shuozhe/saved_dataset/MetaMathQA-math-500/test.parquet}"
MODEL_PATH="${MODEL_PATH:-/data/shuozhe/saved_model/Qwen2.5-0.5B}"
FORMAT_REWARD="${FORMAT_REWARD:-1.0}"
MISSING_REWARD="${MISSING_REWARD:-0.0}"
REQUIRE_NONEMPTY_BOX="${REQUIRE_NONEMPTY_BOX:-true}"

# levave here for later
  # trainer.critic_warmup=101 \

python3 -m verl.trainer.main_ppo \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${VAL_FILE}" \
  data.prompt_key=prompt \
  +data.response_key=ground_truth \
  data.train_batch_size=32 \
  data.max_prompt_length=2048 \
  data.max_response_length=2048 \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
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
  critic.optim.lr=1e-5 \
  critic.model.path="${MODEL_PATH}" \
  critic.model.external_lib=trl \
  critic.model.value_head_init_mean=0.0 \
  critic.model.value_head_init_std=0.00001 \
  critic.model.fsdp_config.param_offload=False \
  critic.ppo_micro_batch_size_per_gpu=4 \
  reward.custom_reward_function.path="${REWARD_FN_PATH}" \
  reward.custom_reward_function.name=compute_score \
  +reward.custom_reward_function.reward_kwargs.format_reward=${FORMAT_REWARD} \
  +reward.custom_reward_function.reward_kwargs.missing_reward=${MISSING_REWARD} \
  +reward.custom_reward_function.reward_kwargs.require_nonempty_box=${REQUIRE_NONEMPTY_BOX} \
  trainer.val_before_train=True \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  trainer.test_freq=10 \
  trainer.save_freq=10 \
  trainer.total_epochs=5 \
  trainer.logger='["console","wandb"]' \
  trainer.project_name="PPO_metamath" \
  trainer.experiment_name="qwen2.5_0.5B_ppo_valuehead_${SLURM_JOB_ID}" \
  trainer.default_local_dir="/data/shuozhe/verl/train_log/job_${SLURM_JOB_ID}" \
  2>&1 | tee /data/shuozhe/verl/train_log/job_${SLURM_JOB_ID}.txt
