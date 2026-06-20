from __future__ import annotations

import argparse
import importlib.util
import json
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

LLM = None
SamplingParams = None
TokensPrompt = None
_VLLM_IMPORT_ERROR: Exception | None = None


def ensure_vllm_imported() -> None:
    global LLM, SamplingParams, TokensPrompt, _VLLM_IMPORT_ERROR
    if LLM is not None and SamplingParams is not None:
        return
    if _VLLM_IMPORT_ERROR is not None:
        raise ImportError("vLLM is required for --generation_backend=vllm, but it is not installed.") from _VLLM_IMPORT_ERROR
    try:
        from vllm import LLM as imported_llm, SamplingParams as imported_sampling_params
        try:
            from vllm.inputs.data import TokensPrompt as imported_tokens_prompt
        except Exception:
            imported_tokens_prompt = None
    except Exception as exc:
        _VLLM_IMPORT_ERROR = exc
        raise ImportError("vLLM is required for --generation_backend=vllm, but it is not installed.") from exc
    LLM = imported_llm
    SamplingParams = imported_sampling_params
    TokensPrompt = imported_tokens_prompt


# DEFAULT_CHECKPOINT = "/data/shuozhe/verl/train_log/job_05b_vh_init_e5_metamath/global_step_800"
DEFAULT_CHECKPOINT = "/data/shuozhe/saved_model/DeepSeek-R1-Distill-Qwen-1.5B"
DEFAULT_DATASET = "/data/shuozhe/saved_dataset/MetaMathQA-math-500/test.parquet"
DEFAULT_OUTPUT = "/data/shuozhe/verl/value_decoding/output/dsk_1d5.jsonl"
DEFAULT_OUTPUT_DATASET = "/data/shuozhe/verl/value_decoding/output/dsk_1d5.parquet"

WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin", "model.safetensors.index.json", "pytorch_model.bin.index.json")


@dataclass(frozen=True)
class ExampleRecord:
    example_id: int
    prompt_text: str
    data_source: str
    ground_truth: Any


def has_hf_checkpoint(path: Path) -> bool:
    return (path / "config.json").is_file() and any((path / name).is_file() for name in WEIGHT_FILES)


def resolve_actor_hf_dir(checkpoint_dir: str | Path, *, skip_merge: bool = False) -> Path:
    checkpoint_dir = Path(checkpoint_dir).expanduser().resolve()
    candidates = [
        checkpoint_dir / "merged_hf" / "actor",
        checkpoint_dir / "actor",
        checkpoint_dir,
    ]
    for candidate in candidates:
        if has_hf_checkpoint(candidate):
            return candidate

    actor_fsdp_dir = checkpoint_dir / "actor"
    if not any(actor_fsdp_dir.glob("model_world_size_*_rank_*.pt")):
        tried = "\n".join(str(path) for path in candidates)
        raise FileNotFoundError(f"No actor HF checkpoint or FSDP shards found. Tried:\n{tried}")
    if skip_merge:
        raise FileNotFoundError(f"Actor checkpoint needs merging, but --skip_merge was set: {actor_fsdp_dir}")

    target_dir = checkpoint_dir / "merged_hf" / "actor"
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "verl.model_merger",
        "merge",
        "--backend",
        "fsdp",
        "--local_dir",
        str(actor_fsdp_dir),
        "--target_dir",
        str(target_dir),
    ]
    hf_config_dir = actor_fsdp_dir / "huggingface"
    if (hf_config_dir / "config.json").is_file():
        cmd.extend(["--hf_model_config_path", str(hf_config_dir)])
    subprocess.run(cmd, check=True)
    if not has_hf_checkpoint(target_dir):
        raise RuntimeError(f"Merge completed but no HF actor checkpoint was found at {target_dir}")
    return target_dir


def resolve_dtype(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def resolve_vllm_dtype(name: str) -> str:
    return {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}[name]


def vllm_output_token_ids(output: Any) -> list[int]:
    token_ids = getattr(output, "token_ids", None)
    if token_ids is None:
        token_ids = getattr(output, "output_token_ids", None)
    if token_ids is None:
        return []
    return [int(token_id) for token_id in token_ids]


def make_vllm_prompt(prompt_token_ids: list[int]) -> Any:
    if TokensPrompt is not None:
        return TokensPrompt(prompt_token_ids=prompt_token_ids)
    return {"prompt_token_ids": prompt_token_ids}


def normalize_prompt(prompt: Any, tokenizer) -> str:
    if isinstance(prompt, np.ndarray):
        prompt = prompt.tolist()
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, dict):
        if "messages" in prompt:
            return normalize_prompt(prompt["messages"], tokenizer)
        for key in ("prompt", "text", "content"):
            if key in prompt:
                return str(prompt[key])
        return json.dumps(prompt, ensure_ascii=True)
    if isinstance(prompt, list):
        if not prompt:
            return ""
        if all(isinstance(item, dict) for item in prompt) and hasattr(tokenizer, "apply_chat_template"):
            try:
                return tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            except Exception:
                pass
        if all(isinstance(item, dict) for item in prompt):
            return "\n".join(f"{item.get('role', 'user')}: {item.get('content', '')}" for item in prompt)
        if all(isinstance(item, str) for item in prompt):
            return "\n".join(prompt)
        return "\n".join(str(item) for item in prompt)
    return str(prompt)


def _is_missing(value: Any) -> bool:
    try:
        result = pd.isna(value)
    except Exception:
        return False

    if isinstance(result, (bool, np.bool_)):
        return bool(result)
    return False


def extract_ground_truth(row: pd.Series, response_key: str | None) -> Any:
    if response_key and response_key in row and not _is_missing(row[response_key]):
        return row[response_key]

    reward_model = row.get("reward_model")
    if isinstance(reward_model, dict):
        return reward_model.get("ground_truth")

    return None


def _load_reward_module(module_name: str):
    module_path = Path(__file__).resolve().parents[1] / "verl" / "utils" / "reward_score" / f"{module_name}.py"
    spec = importlib.util.spec_from_file_location(f"_minimal_reward_{module_name}", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load reward module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _compute_score_with_local_reward_module(data_source: str, response_text: str, ground_truth: Any) -> Any:
    if data_source == "openai/gsm8k":
        return _load_reward_module("gsm8k").compute_score(response_text, ground_truth)
    if data_source in {"lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval", "HuggingFaceH4/MATH-500", "math_500"}:
        return _load_reward_module("math_reward").compute_score(response_text, ground_truth)
    if data_source in {"math_dapo", "math", "math_dapo_reasoning"} or data_source.startswith("aime"):
        return _load_reward_module("math_dapo").compute_score(response_text, ground_truth, incorrect_reward=0.0)
    raise NotImplementedError(f"Reward function is not implemented for data_source={data_source!r}")


def score_response(example: ExampleRecord, response_text: str) -> float:
    score = _compute_score_with_local_reward_module(example.data_source, response_text, example.ground_truth)
    if isinstance(score, dict):
        for key in ("score", "reward", "accuracy", "acc"):
            if key in score:
                return float(score[key])
        raise ValueError(f"Cannot scalarize score dictionary: {score}")
    return float(score)


def load_examples(
    path: str | Path,
    tokenizer,
    *,
    prompt_key: str,
    response_key: str | None,
    start_index: int,
    max_examples: int,
    shuffle: bool,
    seed: int,
) -> list[ExampleRecord]:
    dataframe = pd.read_parquet(Path(path).expanduser())
    indices = list(range(len(dataframe)))
    if start_index:
        indices = indices[start_index:]
    if shuffle:
        random.Random(seed).shuffle(indices)
    if max_examples >= 0:
        indices = indices[:max_examples]

    examples: list[ExampleRecord] = []
    for index in indices:
        row = dataframe.iloc[index]
        data_source = row.get("data_source", "")
        data_source = "" if _is_missing(data_source) else str(data_source)
        examples.append(
            ExampleRecord(
                example_id=int(index),
                prompt_text=normalize_prompt(row[prompt_key], tokenizer),
                data_source=data_source,
                ground_truth=extract_ground_truth(row, response_key=response_key),
            )
        )
    return examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load one actor checkpoint and generate responses for a parquet dataset.")
    parser.add_argument("--checkpoint_dir", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset_path", default=DEFAULT_DATASET)
    parser.add_argument("--output_path", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--output_dataset_path",
        default=DEFAULT_OUTPUT_DATASET,
        help="Path for a new parquet dataset copy with generated trajectory columns. Use empty string to skip.",
    )
    parser.add_argument(
        "--trajectory_text_key",
        default="prompt_generated_trajectory_text",
        help="Column name for prompt + generated trajectory text in the output dataset.",
    )
    parser.add_argument(
        "--trajectory_ids_key",
        default="prompt_generated_trajectory_ids",
        help="Column name for tokenized prompt + generated trajectory ids in the output dataset.",
    )
    parser.add_argument("--prompt_key", default="prompt")
    parser.add_argument("--response_key", default=None, help="Optional dataset column containing ground-truth answers.")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_examples", type=int, default=500, help="Use -1 for all examples.")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--generation_backend",
        choices=("vllm", "torch"),
        default="vllm",
        help="Use vLLM batched generation by default; use torch to keep the old per-example generate path.",
    )
    parser.add_argument("--batch_size", type=int, default=128, help="Number of prompts per vLLM/torch generation batch.")
    parser.add_argument(
        "--vllm_gpu_memory_utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization passed to vLLM.",
    )
    parser.add_argument(
        "--vllm_tensor_parallel_size",
        type=int,
        default=4,
        help="Tensor parallel size passed to vLLM. Defaults to 4 to use all GPUs on this machine.",
    )
    parser.add_argument(
        "--vllm_max_model_len",
        type=int,
        default=None,
        help=(
            "max_model_len passed to vLLM. Defaults to --max_prompt_length + --max_new_tokens "
            "instead of the checkpoint maximum to avoid oversized KV/cache warmup."
        ),
    )
    parser.add_argument("--vllm_max_num_seqs", type=int, default=None, help="Optional max_num_seqs passed to vLLM.")
    parser.add_argument(
        "--vllm_enforce_eager",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass enforce_eager to vLLM. Defaults to true to avoid CUDA graph warmup OOM; use --no-vllm_enforce_eager for max throughput.",
    )
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--skip_merge", action="store_true")
    return parser.parse_args()


def cuda_device_index(device_name: str) -> int | None:
    if device_name == "cuda":
        return 0
    if device_name.startswith("cuda:"):
        return int(device_name.split(":", 1)[1])
    return None


def build_vllm_engine(args: argparse.Namespace, actor_dir: Path) -> Any:
    ensure_vllm_imported()
    device_index = cuda_device_index(args.device)
    if device_index is None:
        raise ValueError("--generation_backend=vllm requires --device cuda[:index].")
    visible_device_count = torch.cuda.device_count()
    tensor_parallel_size = int(args.vllm_tensor_parallel_size)
    if tensor_parallel_size <= 0:
        raise ValueError("--vllm_tensor_parallel_size must be positive.")
    if not torch.cuda.is_available() or visible_device_count <= device_index:
        raise ValueError(f"--generation_backend=vllm requires visible CUDA device {args.device!r}.")
    if visible_device_count < tensor_parallel_size:
        raise ValueError(
            "--generation_backend=vllm requires at least "
            f"{tensor_parallel_size} visible CUDA devices, but only {visible_device_count} are visible."
        )
    if not (0.0 < args.vllm_gpu_memory_utilization <= 1.0):
        raise ValueError("--vllm_gpu_memory_utilization must be in (0, 1].")

    max_model_len = (
        int(args.vllm_max_model_len)
        if args.vllm_max_model_len is not None
        else int(args.max_prompt_length) + int(args.max_new_tokens)
    )
    if max_model_len <= 0:
        raise ValueError("vLLM max_model_len must be positive.")

    kwargs = {
        "model": str(actor_dir),
        "tokenizer": str(actor_dir),
        "dtype": resolve_vllm_dtype(args.dtype),
        "trust_remote_code": args.trust_remote_code,
        "tensor_parallel_size": tensor_parallel_size,
        "gpu_memory_utilization": float(args.vllm_gpu_memory_utilization),
        "enforce_eager": bool(args.vllm_enforce_eager),
        "max_model_len": max_model_len,
    }
    if args.vllm_max_num_seqs is not None:
        kwargs["max_num_seqs"] = int(args.vllm_max_num_seqs)

    previous_device = torch.cuda.current_device()
    torch.cuda.set_device(device_index)
    try:
        return LLM(**kwargs)
    finally:
        torch.cuda.set_device(previous_device)


def make_sampling_params(args: argparse.Namespace, tokenizer) -> Any:
    ensure_vllm_imported()
    stop_token_ids = [] if tokenizer.eos_token_id is None else [int(tokenizer.eos_token_id)]
    return SamplingParams(
        n=1,
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        top_k=0 if int(args.top_k) <= 0 else int(args.top_k),
        max_tokens=int(args.max_new_tokens),
        seed=int(args.seed),
        stop_token_ids=stop_token_ids,
        skip_special_tokens=True,
    )


def iter_generations_with_vllm(
    *,
    llm: Any,
    tokenizer,
    examples: list[ExampleRecord],
    args: argparse.Namespace,
):
    sampling_params = make_sampling_params(args, tokenizer)
    batch_size = max(1, int(args.batch_size))

    for batch_start in tqdm(range(0, len(examples), batch_size), desc="Generating"):
        batch = examples[batch_start : batch_start + batch_size]
        prompt_token_id_batches: list[list[int]] = []
        prompts: list[Any] = []
        for example in batch:
            prompt_token_ids = tokenizer(
                example.prompt_text,
                truncation=True,
                max_length=args.max_prompt_length,
                return_token_type_ids=False,
            )["input_ids"]
            prompt_token_ids = [int(token_id) for token_id in prompt_token_ids]
            prompt_token_id_batches.append(prompt_token_ids)
            prompts.append(make_vllm_prompt(prompt_token_ids))

        request_outputs = llm.generate(prompts=prompts, sampling_params=sampling_params, use_tqdm=False)
        for example, prompt_token_ids, request_output in zip(batch, prompt_token_id_batches, request_outputs, strict=True):
            output = request_output.outputs[0]
            response_ids = vllm_output_token_ids(output)
            response = getattr(output, "text", None)
            if response is None:
                response = tokenizer.decode(response_ids, skip_special_tokens=True)
            trajectory_ids = prompt_token_ids + response_ids
            trajectory_text = tokenizer.decode(trajectory_ids, skip_special_tokens=True)
            yield {
                "example": example,
                "response": response,
                "trajectory_ids": trajectory_ids,
                "trajectory_text": trajectory_text,
            }


def iter_generations_with_torch(
    *,
    model,
    tokenizer,
    device: torch.device,
    examples: list[ExampleRecord],
    args: argparse.Namespace,
):
    do_sample = args.temperature > 0
    batch_size = max(1, int(args.batch_size))

    for batch_start in tqdm(range(0, len(examples), batch_size), desc="Generating"):
        batch = examples[batch_start : batch_start + batch_size]
        inputs = tokenizer(
            [example.prompt_text for example in batch],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_prompt_length,
            return_token_type_ids=False,
        ).to(device)
        generation_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs.update(temperature=args.temperature, top_p=args.top_p, top_k=args.top_k)

        with torch.inference_mode():
            generated = model.generate(**inputs, **generation_kwargs)
        input_width = int(inputs["input_ids"].shape[1])
        prompt_token_id_batches = [
            tokenizer(
                example.prompt_text,
                truncation=True,
                max_length=args.max_prompt_length,
                return_token_type_ids=False,
            )["input_ids"]
            for example in batch
        ]
        for example, generated_ids, prompt_token_ids in zip(
            batch,
            generated,
            prompt_token_id_batches,
            strict=True,
        ):
            generated_ids_list = generated_ids.detach().cpu().tolist()
            response_ids = generated_ids_list[input_width:]
            prompt_ids = [int(token_id) for token_id in prompt_token_ids]
            trajectory_ids = prompt_ids + response_ids
            response = tokenizer.decode(response_ids, skip_special_tokens=True)
            trajectory_text = tokenizer.decode(trajectory_ids, skip_special_tokens=True)
            yield {
                "example": example,
                "response": response,
                "trajectory_ids": trajectory_ids,
                "trajectory_text": trajectory_text,
            }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    actor_dir = resolve_actor_hf_dir(args.checkpoint_dir, skip_merge=args.skip_merge)
    tokenizer = AutoTokenizer.from_pretrained(actor_dir, trust_remote_code=args.trust_remote_code)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token


    examples = load_examples(
        args.dataset_path,
        tokenizer,
        prompt_key=args.prompt_key,
        response_key=args.response_key,
        start_index=args.start_index,
        max_examples=args.max_examples,
        shuffle=args.shuffle,
        seed=args.seed,
    )
    if not examples:
        raise ValueError("No examples were loaded. Check --dataset_path and slicing arguments.")

    output_path = Path(args.output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(args.dataset_path).expanduser()
    output_dataset_path = Path(args.output_dataset_path).expanduser() if args.output_dataset_path else None
    if output_dataset_path is not None:
        if output_dataset_path.resolve() == dataset_path.resolve():
            raise ValueError("--output_dataset_path must be different from --dataset_path to avoid modifying the input dataset.")
        output_dataset_path.parent.mkdir(parents=True, exist_ok=True)
        output_dataframe = pd.read_parquet(dataset_path).copy()
        output_dataframe[args.trajectory_text_key] = None
        output_dataframe[args.trajectory_ids_key] = None
    else:
        output_dataframe = None

    if args.generation_backend == "vllm":
        llm = build_vllm_engine(args, actor_dir)
        generations = iter_generations_with_vllm(llm=llm, tokenizer=tokenizer, examples=examples, args=args)
    else:
        device = torch.device(args.device)
        model = AutoModelForCausalLM.from_pretrained(
            actor_dir,
            dtype=resolve_dtype(args.dtype),
            trust_remote_code=args.trust_remote_code,
        ).to(device)
        model.eval()
        generations = iter_generations_with_torch(
            model=model,
            tokenizer=tokenizer,
            device=device,
            examples=examples,
            args=args,
        )

    scores: list[float] = []
    correct: list[bool] = []
    num_unscored = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for generation in generations:
            example = generation["example"]
            response = generation["response"]
            trajectory_ids = generation["trajectory_ids"]
            trajectory_text = generation["trajectory_text"]
            if output_dataframe is not None:
                output_dataframe.iat[example.example_id, output_dataframe.columns.get_loc(args.trajectory_text_key)] = trajectory_text
                output_dataframe.iat[example.example_id, output_dataframe.columns.get_loc(args.trajectory_ids_key)] = trajectory_ids
            row = {"example_id": example.example_id, "prompt": example.prompt_text, "response": response}
            if example.ground_truth is None:
                row["task_score"] = None
                num_unscored += 1
            else:
                score = score_response(example, response)
                is_correct = bool(score == 1.0)
                scores.append(score)
                correct.append(is_correct)
                row["task_score"] = score
                row["is_correct"] = is_correct
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()

    print(f"Wrote {len(examples)} responses to {output_path}")
    if output_dataframe is not None and output_dataset_path is not None:
        output_dataframe.to_parquet(output_dataset_path, index=False)
        print(f"Wrote dataset with generated trajectory columns to {output_dataset_path}")
    if scores:
        pass_at_1 = float(np.mean(correct))
        mean_score = float(np.mean(scores))
        print(
            json.dumps(
                {
                    "num_examples": len(examples),
                    "num_scored": len(scores),
                    "num_unscored": num_unscored,
                    "pass@1": pass_at_1,
                    "accuracy": pass_at_1,
                    "mean_score": mean_score,
                    "score_sum": float(np.sum(scores)),
                    "num_correct": int(np.sum(correct)),
                },
                indent=2,
            )
        )
    else:
        print("pass@1 was not computed because no ground-truth answers were found.")


if __name__ == "__main__":
    main()
