# Value-Guided Decoding

This package runs inference-time decoding experiments with a VERL actor checkpoint and its paired critic checkpoint.

Implemented modes:

- `actor_only`
- `critic_only_rerank`
- `actor_critic_rerank`
- `actor_critic_soft_rerank`

Implemented outputs:

- `summary_metrics.json`
- `per_example_results.jsonl`
- `step_level_minimal.jsonl`
- `main_results.csv`

Additional Stage 1 critic-comparison outputs:

- `trajectory_bank.jsonl`
- `prompt_level_summary.jsonl`
- `summary_metrics.json`
- `main_results.csv`
- `README.md`

Additional Stage 2 best-of-N inference outputs:

- `trajectory_bank.jsonl`
- `prompt_level_summary.jsonl`
- `summary_metrics.json`
- `main_results.csv`
- `accuracy_vs_n.png`
- `conditional_success_recovery_vs_n.png`
- `mean_selected_response_length_vs_n.png`
- `README.md`

Additional hybrid bank-reranking outputs:

- `hybrid_prompt_level_summary.jsonl`
- `hybrid_summary_metrics.json`
- `hybrid_main_results.csv`
- `hybrid_comparisons.json`
- `hybrid_disagreement_analysis.json`
- `accuracy_vs_lambda.png`
- `conditional_recovery_vs_lambda.png`
- `mean_selected_response_length_vs_lambda.png`
- `fraction_changed_vs_lambda.png`
- `README.md`

Additional margin-gated reranking outputs:

- `gated_hybrid_prompt_level_summary.jsonl`
- `gated_hybrid_summary_metrics.json`
- `gated_hybrid_main_results.csv`
- `accuracy_vs_tau.png`
- `conditional_recovery_vs_tau.png`
- `fraction_gated_prompts_vs_tau.png`
- `gated_subset_accuracy_delta_vs_tau.png`
- `mean_selected_response_length_vs_tau.png`
- `README.md`

Additional chunk-guidance outputs:

- `summary_metrics.json`
- `main_results.csv`
- `per_example_results.jsonl`
- `chunk_decision_results.jsonl`
- `README.md`

Additional chunk-ranking benchmark outputs:

- `chunk_benchmark_candidates.jsonl`
- `chunk_benchmark_prefix_summary.jsonl`
- `chunk_benchmark_summary_metrics.json`
- `chunk_benchmark_main_results.csv`
- `chunk_benchmark_trace_manifest.json`
- `README.md`

## What It Does

For reranking modes, the runner:

1. Gets actor logits at the current prefix.
2. Builds a candidate set from the actor distribution.
3. Appends each candidate token to the prefix.
4. Runs the critic on the full child sequence.
5. Uses the critic value at the last token position for ranking.

It also logs the full-trajectory value:

- `trajectory_value = critic(prefix + full_generated_response)[last_position]`

## Usage

Activate the environment first:

```bash
source /data/shuozhe/miniconda3/bin/activate verl
```

Smoke test on a tiny subset:

```bash
python -m value_decoding \
  --checkpoint_dir /data/shuozhe/verl/train_log/job_05b_vh_init_e5_metamath/global_step_800 \
  --dataset_path /data/shuozhe/saved_dataset/MetaMathQA-math-500/test.parquet \
  --output_dir /data/shuozhe/verl/value_decoding/out_smoke \
  --max_examples 2 \
  --max_new_tokens 128 \
  --modes actor_only critic_only_rerank actor_critic_rerank \
  --candidate_sizes 4 \
  --betas 1.0
```

Example fuller run on the provided MetaMath test split:

```bash
python -m value_decoding \
  --checkpoint_dir /data/shuozhe/verl/train_log/job_05b_vh_init_e5_metamath/global_step_800 \
  --dataset_path /data/shuozhe/saved_dataset/MetaMathQA-math-500/test.parquet \
  --output_dir /data/shuozhe/verl/value_decoding/out_job_05b_vh_init_e5_metamath_gs800 \
  --max_new_tokens 2048 \
  --modes actor_only critic_only_rerank actor_critic_rerank actor_critic_soft_rerank \
  --candidate_builders top_k \
  --candidate_sizes 4 8 \
  --betas 0.5 1.0 2.0 \
  --rank_temperatures 0.5 1.0 \
  --actor_sampling_mode greedy
```

Split actor and critic across two GPUs:

```bash
python -m value_decoding \
  --checkpoint_dir /data/shuozhe/verl/train_log/job_05b_vh_init_e5_metamath/global_step_800 \
  --dataset_path /data/shuozhe/saved_dataset/MetaMathQA-math-500/test.parquet \
  --output_dir /data/shuozhe/verl/value_decoding/out_two_gpu \
  --max_examples 8 \
  --max_new_tokens 256 \
  --modes actor_only critic_only_rerank actor_critic_rerank \
  --candidate_sizes 4 \
  --betas 1.0 \
  --actor_device cuda:0 \
  --critic_device cuda:1
```

Run a single experiment across multiple workers:

```bash
WORKER_PAIRS="cuda:0,cuda:1 cuda:2,cuda:3" \
RUN_SELF_CHECK=0 \
MAX_EXAMPLES=500 \
MAX_NEW_TOKENS=2048 \
OUTPUT_DIR=/data/shuozhe/verl/value_decoding/out_multi_worker \
bash /data/shuozhe/verl/value_decoding/run_value_guided_experiment.sh
```

Run built-in invariance checks:

```bash
python -m value_decoding.self_check \
  --actor_dir /data/shuozhe/verl/train_log/job_05b_vh_init_e5_metamath/global_step_800/merged_hf/actor \
  --critic_dir /data/shuozhe/verl/train_log/job_05b_vh_init_e5_metamath/global_step_800/merged_hf/critic \
  --dataset_path /data/shuozhe/saved_dataset/MetaMathQA-math-500/test.parquet \
  --check_two_gpu
```

Run the Stage 1 old-vs-new critic comparison on a shared actor response bank:

```bash
python -m value_decoding.critic_quality_eval \
  --actor_checkpoint_dir /data/shuozhe/verl/train_log/job_05b_vh_init_e5_metamath/global_step_800 \
  --old_critic_checkpoint_dir /data/shuozhe/verl/train_log/job_05b_vh_init_e5_metamath/global_step_800 \
  --new_critic_checkpoint_dir /data/shuozhe/verl/train_log/job_policy_gs800_dsk_1d5b_critic/global_step_750 \
  --dataset_path /data/shuozhe/saved_dataset/MetaMathQA-math-500/test.parquet \
  --output_dir /data/shuozhe/verl/value_decoding/out_stage1_critic_quality \
  --max_examples 500 \
  --num_samples_per_prompt 8 \
  --actor_sampling_mode sample \
  --actor_temperature 1.0 \
  --actor_top_p 1.0 \
  --actor_top_k 0
```

Or use the wrapper script with the same defaults:

```bash
bash /data/shuozhe/verl/value_decoding/script/stage1_critic_quality_eval.sh
```

Run the Stage 2 end-to-end best-of-N inference evaluation:

```bash
python -m value_decoding.best_of_n_inference_eval \
  --actor_checkpoint_dir /data/shuozhe/verl/train_log/job_05b_vh_init_e5_metamath/global_step_800 \
  --old_critic_checkpoint_dir /data/shuozhe/verl/train_log/job_05b_vh_init_e5_metamath/global_step_800 \
  --new_critic_checkpoint_dir /data/shuozhe/verl/train_log/job_policy_gs800_dsk_1d5b_critic/global_step_750 \
  --dataset_path /data/shuozhe/saved_dataset/MetaMathQA-math-500/test.parquet \
  --output_dir /data/shuozhe/verl/value_decoding/out_stage2_best_of_n \
  --max_examples 500 \
  --n_values 1 2 4 8 16 \
  --actor_sampling_mode sample \
  --actor_temperature 1.0 \
  --actor_top_p 1.0 \
  --actor_top_k 0
```

Or use the wrapper script:

```bash
bash /data/shuozhe/verl/value_decoding/script/stage2_best_of_n_inference_eval.sh
```

The Stage 2 wrapper also supports prompt-sharded multi-worker inference via:

```bash
WORKER_LAYOUTS="cuda:0,cuda:1,cuda:2 cuda:3,cuda:1,cuda:2"
```

This duplicates the actor across two prompt shards and reuses the old/new critic GPUs, which can help when actor sampling is the main bottleneck.

Run the post-hoc hybrid reranking analysis on an existing Stage 2 bank:

```bash
python -m value_decoding.hybrid_bank_rerank_analysis \
  --trajectory_bank_path /data/shuozhe/verl/value_decoding/output/stage2_best_of_n_inference_eval/trajectory_bank.jsonl \
  --prompt_summary_path /data/shuozhe/verl/value_decoding/output/stage2_best_of_n_inference_eval/prompt_level_summary.jsonl \
  --summary_metrics_path /data/shuozhe/verl/value_decoding/output/stage2_best_of_n_inference_eval/summary_metrics.json \
  --output_dir /data/shuozhe/verl/value_decoding/output/hybrid_bank_rerank_analysis \
  --normalization zscore \
  --lambdas 0.0 0.1 0.25 0.5 1.0 2.0
```

Or use the wrapper script:

```bash
bash /data/shuozhe/verl/value_decoding/script/hybrid_bank_rerank_analysis.sh
```

Run the margin-gated reranking analysis on the saved Stage 2 bank:

```bash
python -m value_decoding.gated_margin_rerank_analysis \
  --trajectory_bank_path /data/shuozhe/verl/value_decoding/output/stage2_best_of_n_inference_eval/trajectory_bank.jsonl \
  --prompt_summary_path /data/shuozhe/verl/value_decoding/output/stage2_best_of_n_inference_eval/prompt_level_summary.jsonl \
  --summary_metrics_path /data/shuozhe/verl/value_decoding/output/stage2_best_of_n_inference_eval/summary_metrics.json \
  --output_dir /data/shuozhe/verl/value_decoding/output/gated_margin_rerank_analysis \
  --normalization zscore \
  --taus 0.0 0.05 0.1 0.2 0.3 0.5 \
  --local_hybrid_lambdas 0.1 0.25 0.5
```

Or use the wrapper script:

```bash
bash /data/shuozhe/verl/value_decoding/script/gated_margin_rerank_analysis.sh
```

Run the chunk-level guidance experiment:

```bash
python -m value_decoding.chunk_guidance_eval \
  --actor_checkpoint_dir /data/shuozhe/verl/train_log/job_05b_vh_init_e5_metamath/global_step_800 \
  --critic_checkpoint_dir /data/shuozhe/verl/train_log/job_policy_gs800_dsk_1d5b_critic/global_step_750 \
  --dataset_path /data/shuozhe/saved_dataset/MetaMathQA-math-500/test.parquet \
  --output_dir /data/shuozhe/verl/value_decoding/output/chunk_guidance_eval \
  --chunk_sizes 2 4 \
  --num_chunk_candidates_values 2 \
  --betas 0 0.05 0.1 0.25 \
  --value_reducers end
```

Or use the wrapper script:

```bash
bash /data/shuozhe/verl/value_decoding/script/chunk_guidance_eval.sh
```

Run the offline chunk-ranking benchmark:

```bash
python -m value_decoding.chunk_ranking_benchmark \
  --actor_checkpoint_dir /data/shuozhe/verl/train_log/job_05b_vh_init_e5_metamath/global_step_800 \
  --old_critic_checkpoint_dir /data/shuozhe/verl/train_log/job_05b_vh_init_e5_metamath/global_step_800 \
  --new_critic_checkpoint_dir /data/shuozhe/verl/train_log/job_policy_gs800_dsk_1d5b_critic/global_step_750 \
  --dataset_path /data/shuozhe/saved_dataset/MetaMathQA-math-500/test.parquet \
  --output_dir /data/shuozhe/verl/value_decoding/output/chunk_ranking_benchmark \
  --max_examples 500 \
  --chunk_size 32 \
  --num_chunk_candidates 8 \
  --bootstrap_samples 2000 \
  --actor_sampling_mode sample \
  --actor_temperature 1.0 \
  --actor_top_p 1.0 \
  --actor_top_k 0
```

On a 4-GPU machine, prompt-shard the actor across two workers while reusing the old/new critic GPUs:

```bash
python -m value_decoding.chunk_ranking_benchmark \
  --actor_checkpoint_dir /data/shuozhe/verl/train_log/job_05b_vh_init_e5_metamath/global_step_800 \
  --old_critic_checkpoint_dir /data/shuozhe/verl/train_log/job_05b_vh_init_e5_metamath/global_step_800 \
  --new_critic_checkpoint_dir /data/shuozhe/verl/train_log/job_policy_gs800_dsk_1d5b_critic/global_step_750 \
  --dataset_path /data/shuozhe/saved_dataset/MetaMathQA-math-500/test.parquet \
  --output_dir /data/shuozhe/verl/value_decoding/output/chunk_ranking_benchmark \
  --chunk_size 32 \
  --num_chunk_candidates 8 \
  --worker_layouts cuda:0,cuda:1,cuda:2 cuda:3,cuda:1,cuda:2
```

This prompt-sharded layout launches one worker process per shard, so each worker loads its own actor and critic copies. Use it only if the critic GPUs have enough memory for duplicated replicas.

Or use the wrapper script:

```bash
bash /data/shuozhe/verl/value_decoding/script/chunk_ranking_benchmark.sh
```

Later, rescore the saved trace bank with another critic without rerunning the actor:

```bash
python -m value_decoding.chunk_ranking_benchmark \
  --existing_candidate_bank /data/shuozhe/verl/value_decoding/output/chunk_ranking_benchmark/chunk_benchmark_candidates.jsonl \
  --existing_trace_manifest /data/shuozhe/verl/value_decoding/output/chunk_ranking_benchmark/chunk_benchmark_trace_manifest.json \
  --critic chunk_trained /path/to/new/critic_checkpoint \
  --critic_device chunk_trained cuda:0 \
  --output_dir /data/shuozhe/verl/value_decoding/output/chunk_ranking_benchmark_rescored
```

The saved candidate bank now includes prompt-token, prefix-token, chunk-token, and prefix-plus-chunk token traces so later critics can be benchmarked offline without actor regeneration.

The default chunk-guidance wrapper uses 4 GPUs as two prompt-sharded workers:

```bash
WORKER_PAIRS="cuda:0,cuda:1 cuda:2,cuda:3"
```

## Notes

- `actor_only` still logs critic values for the chosen path, so all modes can be compared with the same metrics.
- `candidate_builders sampled` samples `K` unique tokens from the actor distribution instead of taking the top-`K`.
- `--debug_full_candidates` adds candidate ids, log-probs, values, and scores into the step-level log.
- Multi-GPU support is model-split rather than data-parallel: the actor and critic can run on different devices via `--actor_device` and `--critic_device`.
- Multi-worker support is explicit via `--worker_pairs` or `WORKER_PAIRS="actor_dev,critic_dev actor_dev,critic_dev ..."`.

## Important Off-Policy Caveat

The critic estimates:

- `V^pi(s) = E[R | s, follow training policy]`

Value-guided decoding changes the inference policy.

Therefore:

- value-guided decoding is off-policy usage of the critic
- improvements are empirical, not guaranteed
