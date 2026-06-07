#!/usr/bin/env python3
"""
Submit 18 SLURM jobs: 2 actors x 3 critics x 3 seeds.
Each job evaluates one (actor, critic, seed) combination with chunk sizes 128 & 256 and K=8.

Usage:
    python submit_jobs.py            # dry run: print sbatch commands
    python submit_jobs.py --submit   # actually submit to SLURM
"""

import argparse
import subprocess
import sys
import textwrap
from pathlib import Path

# ---------------------------------------------------------------------------
# Experiment matrix
# ---------------------------------------------------------------------------

ACTORS = {
    "low_ent": "/scratch/10587/npg493/verl_runs/low_ent_critic_training_ckpt_750_actor_697767/train_log/global_step_600",
    "high_ent": "/scratch/10587/npg493/verl_runs/high_ent_critic_training_ckpt_100_actor_697771/train_log/global_step_600",
}

CRITICS = {
    "low_ent": [
        "/scratch/10587/npg493/verl_runs/low_ent_critic_training_ckpt_750_actor_697767/train_log/global_step_200",
        "/scratch/10587/npg493/verl_runs/low_ent_critic_training_ckpt_750_actor_697767/train_log/global_step_400",
        "/scratch/10587/npg493/verl_runs/low_ent_critic_training_ckpt_750_actor_697767/train_log/global_step_600",
    ],
    "high_ent": [
        "/scratch/10587/npg493/verl_runs/high_ent_critic_training_ckpt_100_actor_697771/train_log/global_step_200",
        "/scratch/10587/npg493/verl_runs/high_ent_critic_training_ckpt_100_actor_697771/train_log/global_step_400",
        "/scratch/10587/npg493/verl_runs/high_ent_critic_training_ckpt_100_actor_697771/train_log/global_step_600",
    ],
}

SEEDS = [42, 137, 919]

# ---------------------------------------------------------------------------
# Fixed config (unchanged from the template)
# ---------------------------------------------------------------------------

VENV            = "/work/10587/npg493/vista/verl/.venv"
DATASET_PATH    = "/scratch/10587/npg493/dataset/MetaMathQA-math-500/test.parquet"
WORK_DIR        = "/work/10587/npg493/vista/verl"
ARCHIVE_ROOT    = "/work/10587/npg493/vista/value_decoding/output_archive"

CHUNK_SIZES              = "128 256"
NUM_CHUNK_CANDIDATES     = "8"
BETAS                    = "0"
VALUE_REDUCERS           = "end"
INCLUDE_CRITIC_ONLY      = 1
INCLUDE_UNCERTAINTY_ONLY = 0
ONLY_CRITIC_ONLY         = 1

MAX_PROMPT_LENGTH        = 2048
MAX_NEW_TOKENS           = 2048
MAX_EXAMPLES             = 500
START_INDEX              = 0
DTYPE                    = "bf16"
NORMALIZATION_EPS        = "1e-6"
ACTOR_SAMPLING_MODE      = "sample"
ACTOR_TEMPERATURE        = 1.0
ACTOR_TOP_P              = 1.0
ACTOR_TOP_K              = 0
GENERATION_BACKEND       = "vllm"
VLLM_GPU_MEMORY_UTIL     = 0.6
VLLM_ENFORCE_EAGER       = 0
WORKER_PAIRS             = "cuda:0"
RAY_NUM_CPUS_PER_WORKER  = 1
RAY_GPUS_PER_NODE        = 1
SHUFFLE_EXAMPLES         = 0
SKIP_MERGE               = 0
DISABLE_ACTOR_CACHE      = 0
DEBUG_FULL_CHUNK         = 0
TRUST_REMOTE_CODE        = 0


def make_run_name(actor_key: str, critic_step: int, seed: int) -> str:
    return f"{actor_key}_critic_step{critic_step}_seed{seed}"


def critic_step(critic_path: str) -> int:
    """Extract the global_step number from the critic path."""
    return int(critic_path.split("global_step_")[-1])


def render_script(
    run_name: str,
    actor_dir: str,
    critic_dir: str,
    seed: int,
    output_subdir: str,
) -> str:
    """Return a complete sbatch script as a string."""

    flag_lines = []
    if INCLUDE_CRITIC_ONLY:
        flag_lines.append("    --include_critic_only \\")
    if ONLY_CRITIC_ONLY:
        flag_lines.append("    --only_critic_only \\")
    if INCLUDE_UNCERTAINTY_ONLY:
        flag_lines.append("    --include_uncertainty_only \\")
    flags = "\n".join(flag_lines)

    script = textwrap.dedent(f"""\
        #!/bin/bash
        #SBATCH --job-name={run_name}
        #SBATCH --account=ASC26008
        #SBATCH --partition=gh
        #SBATCH --nodes=8
        #SBATCH --ntasks-per-node=1
        #SBATCH --cpus-per-task=72
        #SBATCH --time=10:00:00
        #SBATCH --output=slurm-%j.out
        #SBATCH --error=slurm-%j.err

        set -euo pipefail

        # -----------------------------
        # Environment setup
        # -----------------------------
        module reset
        module load nvidia/25.9

        VENV="{VENV}"
        source "${{VENV}}/bin/activate"

        UV_CACHE_DIR="${{SCRATCH}}/.cache/uv"
        HF_HOME="${{SCRATCH}}/.cache/huggingface"
        TIKTOKEN_ENCODINGS_BASE="${{SCRATCH}}/data/embeddings"

        mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$TIKTOKEN_ENCODINGS_BASE"

        export UV_CACHE_DIR HF_HOME TIKTOKEN_ENCODINGS_BASE
        export PYTHONUNBUFFERED=1
        export TOKENIZERS_PARALLELISM=true

        echo "Activated environment"
        echo "Python: $(which python3)"
        echo "Ray: $(which ray)"
        python3 -V

        # -----------------------------
        # Run identity
        # -----------------------------
        RUN_NAME="{run_name}"
        RUN_ID="${{RUN_NAME}}_${{SLURM_JOB_ID}}"

        # -----------------------------
        # Paths
        # -----------------------------
        ACTOR_CHECKPOINT_DIR="{actor_dir}"
        CRITIC_CHECKPOINT_DIR="{critic_dir}"
        DATASET_PATH="{DATASET_PATH}"
        WORK_DIR="{WORK_DIR}"
        export PYTHONPATH="${{WORK_DIR}}${{PYTHONPATH:+:${{PYTHONPATH}}}}"

        ARCHIVE_ROOT="{ARCHIVE_ROOT}"
        ARCHIVE_DIR="${{ARCHIVE_ROOT}}/${{RUN_ID}}"

        SCRATCH_ROOT="${{SCRATCH}}/value_decoding_runs"
        RUN_DIR="${{SCRATCH_ROOT}}/${{RUN_ID}}"
        LOG_DIR="${{RUN_DIR}}/logs"
        OUTPUT_DIR="${{RUN_DIR}}/{output_subdir}"
        ACTOR_MERGED_ROOT="${{RUN_DIR}}/merged_actor_hf"
        CRITIC_MERGED_ROOT="${{RUN_DIR}}/merged_critic_hf"
        ACTOR_HF_SOURCE_DIR=""
        CRITIC_HF_SOURCE_DIR=""

        mkdir -p "$LOG_DIR" "$ARCHIVE_ROOT" "$OUTPUT_DIR"

        # -----------------------------
        # Chunk-guidance config
        # -----------------------------
        PROMPT_KEY="prompt"
        RESPONSE_KEY=""
        START_INDEX={START_INDEX}
        MAX_EXAMPLES={MAX_EXAMPLES}
        SHUFFLE_EXAMPLES={SHUFFLE_EXAMPLES}

        MAX_PROMPT_LENGTH={MAX_PROMPT_LENGTH}
        MAX_NEW_TOKENS={MAX_NEW_TOKENS}
        DTYPE="{DTYPE}"

        WORKER_PAIRS="{WORKER_PAIRS}"

        ACTOR_SAMPLING_MODE="{ACTOR_SAMPLING_MODE}"
        ACTOR_TEMPERATURE={ACTOR_TEMPERATURE}
        ACTOR_TOP_P={ACTOR_TOP_P}
        ACTOR_TOP_K={ACTOR_TOP_K}
        GENERATION_BACKEND="{GENERATION_BACKEND}"
        VLLM_GPU_MEMORY_UTILIZATION={VLLM_GPU_MEMORY_UTIL}
        VLLM_ENFORCE_EAGER={VLLM_ENFORCE_EAGER}

        CHUNK_SIZES="{CHUNK_SIZES}"
        NUM_CHUNK_CANDIDATES_VALUES="{NUM_CHUNK_CANDIDATES}"
        BETAS="{BETAS}"
        VALUE_REDUCERS="{VALUE_REDUCERS}"
        INCLUDE_CRITIC_ONLY={INCLUDE_CRITIC_ONLY}
        INCLUDE_UNCERTAINTY_ONLY={INCLUDE_UNCERTAINTY_ONLY}
        ONLY_CRITIC_ONLY={ONLY_CRITIC_ONLY}

        NORMALIZATION_EPS="{NORMALIZATION_EPS}"
        SEED="{seed}"
        SKIP_MERGE={SKIP_MERGE}
        DISABLE_ACTOR_CACHE={DISABLE_ACTOR_CACHE}
        DEBUG_FULL_CHUNK_CANDIDATES={DEBUG_FULL_CHUNK}
        TRUST_REMOTE_CODE={TRUST_REMOTE_CODE}
        RAY_NUM_CPUS_PER_WORKER={RAY_NUM_CPUS_PER_WORKER}
        RAY_GPUS_PER_NODE={RAY_GPUS_PER_NODE}

        # -----------------------------
        # Helpers
        # -----------------------------
        nodes_array=()
        SCRIPT_PATH="${{WORK_DIR}}/value_decoding/script/$(basename "${{BASH_SOURCE[0]}}")"

        sync_to_work() {{
          echo "Syncing run directory back to WORK..."
          mkdir -p "$ARCHIVE_DIR"
          rsync -a "$RUN_DIR"/ "$ARCHIVE_DIR"/ || true
          echo "Archived run to: $ARCHIVE_DIR"
        }}

        stop_ray_all_nodes() {{
          if [[ ${{#nodes_array[@]}} -eq 0 ]]; then return 0; fi
          for node in "${{nodes_array[@]}}"; do
            srun --nodes=1 --ntasks=1 -w "$node" \\
              bash -c "source '${{VENV}}/bin/activate' && ray stop --force || true" \\
              >> "$LOG_DIR/ray_stop_${{node}}.log" 2>&1 || true
          done
        }}

        validate_component_checkpoint() {{
          local checkpoint_dir="$1"
          local component="$2"
          python3 - "$checkpoint_dir" "$component" <<'PY'
        import sys
        from pathlib import Path
        from value_decoding.checkpointing import (
            find_missing_hf_weight_files, has_complete_hf_checkpoint,
            has_fsdp_checkpoint_shards, has_hf_config,
            resolve_component_checkpoint_dir,
        )
        checkpoint_dir = Path(sys.argv[1])
        component = sys.argv[2]
        component_dir = resolve_component_checkpoint_dir(checkpoint_dir, component=component)
        if has_complete_hf_checkpoint(component_dir):
            print(f"{{component}}: complete HF checkpoint at {{component_dir}}")
            raise SystemExit(0)
        if has_fsdp_checkpoint_shards(component_dir):
            print(f"{{component}}: raw FSDP checkpoint at {{component_dir}}")
            raise SystemExit(0)
        if has_hf_config(component_dir):
            missing = find_missing_hf_weight_files(component_dir)
            preview = ", ".join(p.name for p in missing[:5]) or "unknown"
            raise SystemExit(f"{{component}}: incomplete HF checkpoint. Missing: {{preview}}")
        raise SystemExit(f"{{component}}: unsupported checkpoint layout at {{component_dir}}")
        PY
        }}

        count_alive_ray_nodes() {{
          local ray_address="$1"
          python3 - "$ray_address" <<'PY'
        import logging, sys, ray
        address = sys.argv[1]
        try:
            ray.init(address=address, logging_level=logging.ERROR)
            print(sum(1 for n in ray.nodes() if n.get("Alive")))
        finally:
            ray.is_initialized() and ray.shutdown()
        PY
        }}

        cleanup() {{
          echo "Stopping Ray on all nodes..."
          stop_ray_all_nodes || true
          sync_to_work
        }}
        trap cleanup EXIT

        # -----------------------------
        # Debug info
        # -----------------------------
        echo "Job ID: $SLURM_JOB_ID"
        echo "Run ID: $RUN_ID"
        echo "SLURM nodes: $SLURM_JOB_NODELIST"
        echo "ACTOR_CHECKPOINT_DIR: $ACTOR_CHECKPOINT_DIR"
        echo "CRITIC_CHECKPOINT_DIR: $CRITIC_CHECKPOINT_DIR"
        echo "SEED: $SEED"

        echo "Checking inputs..."
        ls -ld "$WORK_DIR"
        ls -ld "$ACTOR_CHECKPOINT_DIR"
        ls -ld "$CRITIC_CHECKPOINT_DIR"
        ls -lh "$DATASET_PATH"
        validate_component_checkpoint "$ACTOR_CHECKPOINT_DIR" actor
        validate_component_checkpoint "$CRITIC_CHECKPOINT_DIR" critic

        nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
        nodes_array=($nodes)

        head_node="${{nodes_array[0]}}"
        head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
        IP_REGEX='^([0-9]+\\.?){{3}}[0-9]+$'
        resolved_head_node_ip=""
        for candidate_ip in $head_node_ip; do
          if [[ "$candidate_ip" =~ $IP_REGEX ]]; then
            resolved_head_node_ip="$candidate_ip"; break
          fi
        done
        if [[ -z "$resolved_head_node_ip" ]]; then
          for candidate_ip in $head_node_ip; do resolved_head_node_ip="$candidate_ip"; break; done
        fi
        if [[ -z "$resolved_head_node_ip" ]]; then
          echo "Failed to resolve head node IP." >&2; exit 1
        fi
        head_node_ip="$resolved_head_node_ip"

        port=6379
        ip_head="${{head_node_ip}}:${{port}}"
        export RAY_ADDRESS="$ip_head"

        echo "Head node: $head_node"
        echo "Head IP: $ip_head"

        # -----------------------------
        # Start Ray head
        # -----------------------------
        echo "Starting Ray head..."
        srun --nodes=1 --ntasks=1 -w "$head_node" \\
          bash -c "source '${{VENV}}/bin/activate' && \\
                   ray start --head \\
                   --node-ip-address='${{head_node_ip}}' \\
                   --port='${{port}}' \\
                   --num-cpus='${{SLURM_CPUS_PER_TASK}}' \\
                   --num-gpus='${{RAY_GPUS_PER_NODE}}'" \\
          > "$LOG_DIR/ray_head.log" 2>&1 &

        sleep 10

        echo "Waiting for Ray head..."
        head_ready=0
        for i in {{1..30}}; do
          if ray status --address="$ip_head" > /dev/null 2>&1; then
            echo "Ray head is ready."; head_ready=1; break
          fi
          sleep 2
        done
        if [[ "$head_ready" != "1" ]]; then
          echo "Ray head failed to become ready." >&2; exit 1
        fi

        # -----------------------------
        # Start Ray workers
        # -----------------------------
        worker_num=$(($SLURM_JOB_NUM_NODES - 1))
        for ((i = 1; i <= worker_num; i++)); do
          node_i="${{nodes_array[$i]}}"
          echo "Starting worker on $node_i"
          srun --nodes=1 --ntasks=1 -w "$node_i" \\
            bash -c "source '${{VENV}}/bin/activate' && \\
                     ray start --address '${{ip_head}}' \\
                     --num-cpus='${{SLURM_CPUS_PER_TASK}}' \\
                     --num-gpus='${{RAY_GPUS_PER_NODE}}'" \\
            > "$LOG_DIR/ray_worker_${{i}}.log" 2>&1 &
        done

        wait

        echo "Waiting for all Ray nodes to register..."
        alive_nodes=0
        all_nodes_ready=0
        RAY_NODE_PROBE_LOG="$LOG_DIR/ray_node_probe.log"
        : > "$RAY_NODE_PROBE_LOG"
        for i in {{1..30}}; do
          alive_nodes="$(count_alive_ray_nodes "$ip_head" 2>>"$RAY_NODE_PROBE_LOG")" || alive_nodes=0
          if [[ "$alive_nodes" -ge "$SLURM_JOB_NUM_NODES" ]]; then
            echo "All $alive_nodes Ray nodes registered."; all_nodes_ready=1; break
          fi
          echo "Registered: ${{alive_nodes}}/${{SLURM_JOB_NUM_NODES}}"
          sleep 5
        done
        if [[ "$all_nodes_ready" != "1" ]]; then
          echo "Only $alive_nodes/$SLURM_JOB_NUM_NODES nodes registered." >&2
          ray status --address="$ip_head" || true; exit 1
        fi

        echo "Ray cluster status:"
        ray status --address="$ip_head" || true

        # -----------------------------
        # Run chunk-guidance eval
        # -----------------------------
        cd "$WORK_DIR"

        read -r -a CHUNK_SIZES_ARR        <<< "$CHUNK_SIZES"
        read -r -a NUM_CHUNK_ARR          <<< "$NUM_CHUNK_CANDIDATES_VALUES"
        read -r -a BETAS_ARR              <<< "$BETAS"
        read -r -a VALUE_REDUCERS_ARR     <<< "$VALUE_REDUCERS"
        read -r -a WORKER_PAIRS_ARR       <<< "$WORKER_PAIRS"
        read -r -a SEED_ARR               <<< "$SEED"

        seed_to_id() {{
          local s="${{1//-/m}}"; s="${{s//./p}}"; printf '%s' "$s"
        }}

        run_one_seed() {{
          local seed_value="$1"
          local output_dir_for_seed="$2"
          local log_path="$3"

          CMD=(
            python3 -m value_decoding.chunk_guidance_eval
            --actor_checkpoint_dir  "$ACTOR_CHECKPOINT_DIR"
            --critic_checkpoint_dir "$CRITIC_CHECKPOINT_DIR"
            --dataset_path          "$DATASET_PATH"
            --output_dir            "$output_dir_for_seed"
            --actor_merged_root     "$ACTOR_MERGED_ROOT"
            --critic_merged_root    "$CRITIC_MERGED_ROOT"
            --prompt_key            "$PROMPT_KEY"
            --start_index           "$START_INDEX"
            --max_prompt_length     "$MAX_PROMPT_LENGTH"
            --max_new_tokens        "$MAX_NEW_TOKENS"
            --dtype                 "$DTYPE"
            --normalization_eps     "$NORMALIZATION_EPS"
            --seed                  "$seed_value"
            --actor_sampling_mode   "$ACTOR_SAMPLING_MODE"
            --actor_temperature     "$ACTOR_TEMPERATURE"
            --actor_top_p           "$ACTOR_TOP_P"
            --actor_top_k           "$ACTOR_TOP_K"
            --generation_backend    "$GENERATION_BACKEND"
            --vllm_gpu_memory_utilization "$VLLM_GPU_MEMORY_UTILIZATION"
            --chunk_sizes           "${{CHUNK_SIZES_ARR[@]}}"
            --num_chunk_candidates_values "${{NUM_CHUNK_ARR[@]}}"
            --betas                 "${{BETAS_ARR[@]}}"
            --value_reducers        "${{VALUE_REDUCERS_ARR[@]}}"
            --ray_address           auto
            --ray_num_cpus_per_worker "$RAY_NUM_CPUS_PER_WORKER"
          )

          [[ -n "$RESPONSE_KEY"        ]] && CMD+=(--response_key        "$RESPONSE_KEY")
          [[ -n "$MAX_EXAMPLES"        ]] && CMD+=(--max_examples         "$MAX_EXAMPLES")
          [[ -n "$ACTOR_HF_SOURCE_DIR" ]] && CMD+=(--actor_hf_source_dir "$ACTOR_HF_SOURCE_DIR")
          [[ -n "$CRITIC_HF_SOURCE_DIR" ]] && CMD+=(--critic_hf_source_dir "$CRITIC_HF_SOURCE_DIR")
          [[ "${{#WORKER_PAIRS_ARR[@]}}" -gt 0 ]] && CMD+=(--worker_pairs  "${{WORKER_PAIRS_ARR[@]}}")
          [[ "$SHUFFLE_EXAMPLES"       != "0" ]] && CMD+=(--shuffle_examples)
          [[ "$VLLM_ENFORCE_EAGER"     != "0" ]] && CMD+=(--vllm_enforce_eager)
          [[ "$SKIP_MERGE"             != "0" ]] && CMD+=(--skip_merge)
          [[ "$DISABLE_ACTOR_CACHE"    != "0" ]] && CMD+=(--disable_actor_cache)
          [[ "$INCLUDE_CRITIC_ONLY"    != "0" ]] && CMD+=(--include_critic_only)
          [[ "$INCLUDE_UNCERTAINTY_ONLY" != "0" ]] && CMD+=(--include_uncertainty_only)
          [[ "$ONLY_CRITIC_ONLY"       != "0" ]] && CMD+=(--only_critic_only)
          [[ "$DEBUG_FULL_CHUNK_CANDIDATES" != "0" ]] && CMD+=(--debug_full_chunk_candidates)
          [[ "$TRUST_REMOTE_CODE"      != "0" ]] && CMD+=(--trust_remote_code)

          mkdir -p "$output_dir_for_seed"
          printf 'Running command for seed %s:\\n' "$seed_value"
          printf ' %q' "${{CMD[@]}}"; printf '\\n'
          "${{CMD[@]}}" 2>&1 | tee "$log_path"
        }}

        if [[ ${{#SEED_ARR[@]}} -eq 1 ]]; then
          run_one_seed "${{SEED_ARR[0]}}" "$OUTPUT_DIR" "$LOG_DIR/chunk_guidance_eval.log"
          echo "Chunk-guidance eval finished successfully."
          exit 0
        fi

        SUMMARY_PATHS=()
        SEED_OUTPUT_DIRS=()
        for seed_value in "${{SEED_ARR[@]}}"; do
          seed_id="$(seed_to_id "$seed_value")"
          seed_output_dir="${{OUTPUT_DIR}}/seed_${{seed_id}}"
          seed_log_path="$LOG_DIR/chunk_guidance_eval__seed_${{seed_id}}.log"
          run_one_seed "$seed_value" "$seed_output_dir" "$seed_log_path"
          SUMMARY_PATHS+=("${{seed_output_dir}}/summary_metrics.json")
          SEED_OUTPUT_DIRS+=("${{seed_output_dir}}")
        done

        python3 -m value_decoding.multi_seed_summary \\
          --output_path "${{OUTPUT_DIR}}/summary_metrics.json" \\
          --source_script "$SCRIPT_PATH" \\
          --seed_values "${{SEED_ARR[@]}}" \\
          --summary_paths "${{SUMMARY_PATHS[@]}}" \\
          --seed_output_dirs "${{SEED_OUTPUT_DIRS[@]}}"

        echo "Chunk-guidance eval finished successfully."
    """)
    return script


def build_jobs():
    """Yield (run_name, actor_key, actor_dir, critic_dir, seed) for all 18 jobs."""
    for actor_key, actor_dir in ACTORS.items():
        for critic_dir in CRITICS[actor_key]:
            step = critic_step(critic_dir)
            for seed in SEEDS:
                run_name = make_run_name(actor_key, step, seed)
                yield run_name, actor_key, actor_dir, critic_dir, seed


def main():
    parser = argparse.ArgumentParser(description="Submit chunk-guidance eval jobs to SLURM.")
    parser.add_argument("--submit", action="store_true", help="Actually submit via sbatch (default: dry run).")
    parser.add_argument("--script-dir", default="slurm_scripts", help="Directory to write generated scripts.")
    args = parser.parse_args()

    script_dir = Path(args.script_dir)
    script_dir.mkdir(parents=True, exist_ok=True)

    jobs = list(build_jobs())
    print(f"{'=' * 60}")
    print(f"Total jobs: {len(jobs)}  (2 actors × 3 critics × 3 seeds)")
    print(f"{'=' * 60}\n")

    submitted = []
    for run_name, actor_key, actor_dir, critic_dir, seed in jobs:
        step = critic_step(critic_dir)
        output_subdir = f"{actor_key}_critic{step}_seed{seed}"
        script_content = render_script(run_name, actor_dir, critic_dir, seed, output_subdir)

        script_path = script_dir / f"{run_name}.sh"
        script_path.write_text(script_content)
        script_path.chmod(0o755)

        if args.submit:
            result = subprocess.run(
                ["sbatch", str(script_path)],
                capture_output=True, text=True
            )
            job_id = result.stdout.strip()
            status = "✓ submitted" if result.returncode == 0 else f"✗ FAILED: {result.stderr.strip()}"
            print(f"[{run_name}] {status}  →  {job_id}")
            submitted.append((run_name, job_id, result.returncode))
        else:
            print(f"[DRY RUN] Would submit: {script_path}")

    print(f"\nScripts written to: {script_dir}/")

    if args.submit:
        n_ok  = sum(1 for _, _, rc in submitted if rc == 0)
        n_err = len(submitted) - n_ok
        print(f"\nSubmitted {n_ok}/{len(submitted)} jobs successfully.")
        if n_err:
            print(f"  {n_err} job(s) failed — check output above.")
            sys.exit(1)
    else:
        print("\nRe-run with --submit to actually submit to SLURM.")


if __name__ == "__main__":
    main()
