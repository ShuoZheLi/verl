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
