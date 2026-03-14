#!/usr/bin/env python3
import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path


ANSI_RE = re.compile(r"\x1B\[[0-9;]*[mK]")


def str2bool(value: str) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {value}")


def bool_to_hydra(value: bool) -> str:
    return "True" if value else "False"


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Evaluate all VERL checkpoints with main_ppo val_only mode.",
    )
    parser.add_argument(
        "--checkpoint-root",
        default="/data/shuozhe/verl/train_log/job_05b_vh_init_load_c1100_e5_freeze_critic_lvl45",
    )
    parser.add_argument(
        "--train-file",
        default="/data/shuozhe/saved_dataset/verl_math_7500_500_5000_level_4_5/train.parquet",
    )
    parser.add_argument(
        "--val-file",
        default="/data/shuozhe/saved_dataset/verl_math_7500_500_5000_level_4_5/test_5000_500.parquet",
    )
    parser.add_argument("--model-path", default="/data/shuozhe/saved_model/Qwen2.5-0.5B")

    parser.add_argument("--n-gpus", type=int, default=4)
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--gpu-mem-util", type=float, default=0.4)

    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--max-prompt-length", type=int, default=2048)
    parser.add_argument("--max-response-length", type=int, default=2048)
    parser.add_argument("--rollout-n", type=int, default=8)
    parser.add_argument("--val-n", type=int, default=1)
    parser.add_argument("--val-do-sample", type=str2bool, default=False)

    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--end-step", type=int, default=10**18)
    parser.add_argument("--skip-existing", type=str2bool, default=True)
    parser.add_argument("--continue-on-error", type=str2bool, default=True)

    parser.add_argument("--eval-root", default=None)
    parser.add_argument("--project-name", default="PPO_eval")
    parser.add_argument(
        "--experiment-prefix",
        default="job_05b_vh_init_load_c1100_e5_freeze_critic_lvl45",
    )
    parser.add_argument("--python-bin", default="python3")

    args, extra_hydra_overrides = parser.parse_known_args()
    return args, extra_hydra_overrides


def find_checkpoints(checkpoint_root: Path) -> list[tuple[int, Path]]:
    checkpoints: list[tuple[int, Path]] = []
    for child in checkpoint_root.iterdir():
        if not child.is_dir():
            continue
        match = re.fullmatch(r"global_step_(\d+)", child.name)
        if not match:
            continue
        checkpoints.append((int(match.group(1)), child))
    checkpoints.sort(key=lambda x: x[0])
    return checkpoints


def ensure_summary_csv(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "status", "core_metric", "log_file"])


def append_summary(path: Path, step: int, status: str, core_metric: str, log_file: Path) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([step, status, core_metric, str(log_file)])


def parse_core_metric(log_file: Path) -> str:
    if not log_file.exists():
        return ""
    last_metric = ""
    with log_file.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = ANSI_RE.sub("", raw_line)
            if "val-core/" not in line:
                continue
            for token in line.strip().split():
                if token.startswith("val-core/") and ":" in token:
                    last_metric = token.rstrip(",")
    return last_metric


def run_and_tee(cmd: list[str], log_file: Path) -> int:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("w", encoding="utf-8") as f:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            f.write(line)
        return process.wait()


def build_hydra_overrides(
    args: argparse.Namespace,
    checkpoint_path: Path,
    val_dump_dir: Path,
    step: int,
    extra_overrides: list[str],
) -> list[str]:
    overrides = [
        f"data.train_files={args.train_file}",
        f"data.val_files={args.val_file}",
        "data.prompt_key=prompt",
        "+data.response_key=ground_truth",
        f"data.train_batch_size={args.train_batch_size}",
        f"data.max_prompt_length={args.max_prompt_length}",
        f"data.max_response_length={args.max_response_length}",
        f"actor_rollout_ref.model.path={args.model_path}",
        "actor_rollout_ref.actor.optim.lr=1e-6",
        "actor_rollout_ref.actor.ppo_mini_batch_size=32",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4",
        "actor_rollout_ref.actor.fsdp_config.param_offload=False",
        "actor_rollout_ref.actor.use_kl_loss=False",
        "actor_rollout_ref.ref.fsdp_config.param_offload=True",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={args.tp_size}",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={args.gpu_mem_util}",
        "actor_rollout_ref.rollout.enforce_eager=True",
        "actor_rollout_ref.rollout.free_cache_engine=True",
        "actor_rollout_ref.rollout.enable_chunked_prefill=True",
        f"actor_rollout_ref.rollout.n={args.rollout_n}",
        f"actor_rollout_ref.rollout.val_kwargs.n={args.val_n}",
        f"actor_rollout_ref.rollout.val_kwargs.do_sample={bool_to_hydra(args.val_do_sample)}",
        "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096",
        "actor_rollout_ref.hybrid_engine=True",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1",
        "critic.enable=False",
        "trainer.resume_mode=resume_path",
        f"trainer.resume_from_path={checkpoint_path}",
        "trainer.val_before_train=True",
        "trainer.val_only=True",
        f"trainer.n_gpus_per_node={args.n_gpus}",
        f"trainer.nnodes={args.nnodes}",
        "trainer.test_freq=-1",
        "trainer.save_freq=-1",
        "trainer.total_epochs=1",
        "trainer.logger=[\"console\"]",
        "trainer.log_val_generations=0",
        f"trainer.validation_data_dir={val_dump_dir}",
        f"trainer.project_name={args.project_name}",
        f"trainer.experiment_name={args.experiment_prefix}_step_{step}",
        f"trainer.default_local_dir={args.checkpoint_root}",
    ]
    overrides.extend(extra_overrides)
    return overrides


def main() -> int:
    args, extra_hydra_overrides = parse_args()

    checkpoint_root = Path(args.checkpoint_root).resolve()
    train_file = Path(args.train_file).resolve()
    val_file = Path(args.val_file).resolve()
    if args.eval_root:
        eval_root = Path(args.eval_root).resolve()
    else:
        eval_root = checkpoint_root / "eval_ckpt_val_only"

    if not checkpoint_root.is_dir():
        print(f"Checkpoint root does not exist: {checkpoint_root}", file=sys.stderr)
        return 1
    if not train_file.is_file():
        print(f"Train parquet not found: {train_file}", file=sys.stderr)
        return 1
    if not val_file.is_file():
        print(f"Val parquet not found: {val_file}", file=sys.stderr)
        return 1

    log_dir = eval_root / "logs"
    val_gen_dir = eval_root / "val_generations"
    summary_csv = eval_root / "summary.csv"
    log_dir.mkdir(parents=True, exist_ok=True)
    val_gen_dir.mkdir(parents=True, exist_ok=True)
    ensure_summary_csv(summary_csv)

    checkpoints = find_checkpoints(checkpoint_root)
    if not checkpoints:
        print(f"No checkpoints found under {checkpoint_root}", file=sys.stderr)
        return 1

    print(f"Found {len(checkpoints)} checkpoints under {checkpoint_root}")
    print(f"Evaluation output: {eval_root}")

    matched_steps = 0
    for step, ckpt_path in checkpoints:
        if step < args.start_step or step > args.end_step:
            continue
        matched_steps += 1

        log_file = log_dir / f"global_step_{step}.log"
        val_dump_dir = val_gen_dir / f"global_step_{step}"
        val_dump_dir.mkdir(parents=True, exist_ok=True)

        if args.skip_existing and log_file.exists():
            core_metric = parse_core_metric(log_file)
            if core_metric:
                append_summary(summary_csv, step, "skipped", core_metric, log_file)
                print(f"[skip] global_step_{step} already evaluated")
                continue

        print(f"[run ] Evaluating global_step_{step}")
        overrides = build_hydra_overrides(
            args=args,
            checkpoint_path=ckpt_path,
            val_dump_dir=val_dump_dir,
            step=step,
            extra_overrides=extra_hydra_overrides,
        )
        cmd = [args.python_bin, "-m", "verl.trainer.main_ppo", *overrides]
        status = run_and_tee(cmd, log_file)

        core_metric = parse_core_metric(log_file)
        if status == 0:
            append_summary(summary_csv, step, "ok", core_metric, log_file)
            print(f"[done] global_step_{step} {core_metric}")
        else:
            append_summary(summary_csv, step, "failed", core_metric, log_file)
            print(f"[fail] global_step_{step} (exit={status})", file=sys.stderr)
            if not args.continue_on_error:
                return status

    if matched_steps == 0:
        print(
            f"No checkpoints matched START_STEP={args.start_step}, END_STEP={args.end_step}",
            file=sys.stderr,
        )

    print(f"All done. Summary: {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
