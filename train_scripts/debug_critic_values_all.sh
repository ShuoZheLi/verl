CUDA_VISIBLE_DEVICES=1,2,3 \
torchrun --standalone --nproc_per_node=3 train_scripts/debug_critic_values_all.py \
  --checkpoint_dir /nfs/shuozhe/verl/train_log/max_response_length=2048/global_step_200 \
  --dataset_path /nfs/shuozhe/saved_dataset/math-500/data/train-00000-of-00001_verl.parquet \
  --max_new_tokens 2048 \
  --start_index 3000 \
  --end_index 3500 \
  --correct_match verl \