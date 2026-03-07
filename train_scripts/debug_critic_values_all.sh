CUDA_VISIBLE_DEVICES=0,2,3 \
torchrun --standalone --nproc_per_node=3 train_scripts/debug_critic_values_all.py \
  --checkpoint_dir /data/shuozhe/verl/train_log/job_05b_vh_init_e5/global_step_320 \
  --dataset_path /data/shuozhe/saved_dataset/level_1_only/test.parquet \
  --correct_match verl \
  --max_new_tokens 2048 \
  --out_dir /data/shuozhe/verl/critic_debug/05b_vh_init_e5_test_level_1 \
  # --start_index 3000 \
  # --end_index 3500 \