python3 train_scripts/debug_critic_values.py \
  --checkpoint_dir /data/shuozhe/verl/train_log/job_level_1_only_05b/global_step_60 \
  --dataset_path /data/shuozhe/saved_dataset/level_1_only/test.parquet \
  --out_dir /data/shuozhe/verl/critic_debug \
  --max_new_tokens 2048 \
  --sample_index 1



# /nfs/shuozhe/saved_dataset/math-500/data/test-00000-of-00001_verl.parquet
# /nfs/shuozhe/saved_dataset/math-500/data/train-00000-of-00001_verl.parquet