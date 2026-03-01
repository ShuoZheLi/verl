python3 train_scripts/debug_critic_values.py \
  --checkpoint_dir /nfs/shuozhe/verl/train_log/max_response_length=2048/global_step_200 \
  --dataset_path /nfs/shuozhe/saved_dataset/math-500/data/test-00000-of-00001_verl.parquet \
  --max_new_tokens 1024 \
  --sample_index 17


# /nfs/shuozhe/saved_dataset/math-500/data/test-00000-of-00001_verl.parquet
# /nfs/shuozhe/saved_dataset/math-500/data/train-00000-of-00001_verl.parquet