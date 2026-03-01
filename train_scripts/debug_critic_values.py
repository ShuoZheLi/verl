#!/usr/bin/env python3
"""
Debug critic values for a PPO checkpoint.

Workflow:
1) Merge FSDP checkpoints (actor/critic) into HuggingFace format (if not already merged).
2) Load actor and critic models.
3) Sample a prompt from the training set, generate a response with the actor.
4) Run the critic over prompt+response and plot per-token values for the response tokens.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


def _has_hf_weights(model_dir: Path) -> bool:
    if not model_dir.exists():
        return False
    # HuggingFace weights can be sharded or single-file.
    for name in ["model.safetensors", "pytorch_model.bin"]:
        if (model_dir / name).exists():
            return True
    # Sharded safetensors or pytorch shards.
    for pattern in ["model-*.safetensors", "pytorch_model-*.bin"]:
        if list(model_dir.glob(pattern)):
            return True
    return False


def _merge_fsdp_checkpoint(local_dir: Path, target_dir: Path) -> None:
    cmd = [
        sys.executable,
        "-m",
        "verl.model_merger",
        "merge",
        "--backend",
        "fsdp",
        "--local_dir",
        str(local_dir),
        "--target_dir",
        str(target_dir),
    ]
    print(f"[merge] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def _load_policy(model_dir: Path, dtype: torch.dtype, device: torch.device, trust_remote_code: bool):
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
    )
    model.to(device)
    model.eval()
    return model


def _load_critic(model_dir: Path, dtype: torch.dtype, device: torch.device, trust_remote_code: bool):
    from transformers import AutoConfig

    try:
        from verl.utils.model import load_valuehead_model

        config = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=trust_remote_code)
        config.num_labels = 1
        model = load_valuehead_model(
            str(model_dir),
            torch_dtype=dtype,
            model_config=config,
            trust_remote_code=trust_remote_code,
        )
    except Exception:
        try:
            from trl import AutoModelForCausalLMWithValueHead
        except Exception as exc:  # pragma: no cover - runtime dependency
            raise RuntimeError("trl is required to load value-head critic models") from exc

        model = AutoModelForCausalLMWithValueHead.from_pretrained(
            str(model_dir),
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
    model.to(device)
    model.eval()
    return model


def _load_prompt(dataset_path: str, prompt_key: str, response_key: str | None, sample_index: int | None, seed: int):
    ds = load_dataset("parquet", data_files=dataset_path, split="train")
    if sample_index is None:
        rng = random.Random(seed)
        sample_index = rng.randrange(len(ds))
    row = ds[int(sample_index)]
    prompt = row[prompt_key]
    response = row[response_key] if response_key and response_key in row else None
    return prompt, response, sample_index


def _stringify_chat_messages(messages) -> str:
    parts = []
    for msg in messages:
        if not isinstance(msg, dict):
            parts.append(str(msg))
            continue
        role = msg.get("role", "user")
        content = msg.get("content", "")
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


def _normalize_prompt(prompt, tokenizer) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, dict):
        # Common dataset formats: {"messages": [...]}, {"prompt": "..."}
        if "messages" in prompt:
            messages = prompt["messages"]
            if hasattr(tokenizer, "apply_chat_template"):
                return tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            return _stringify_chat_messages(messages)
        for key in ("prompt", "text", "content"):
            if key in prompt:
                return str(prompt[key])
        return json.dumps(prompt, ensure_ascii=True)
    if isinstance(prompt, list):
        # List[str] or list[dict]
        if len(prompt) == 0:
            return ""
        if all(isinstance(x, dict) for x in prompt):
            if hasattr(tokenizer, "apply_chat_template"):
                return tokenizer.apply_chat_template(
                    prompt, tokenize=False, add_generation_prompt=True
                )
            return _stringify_chat_messages(prompt)
        if all(isinstance(x, str) for x in prompt):
            return "\n".join(prompt)
        return "\n".join(str(x) for x in prompt)
    return str(prompt)


def _get_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str == "bf16":
        return torch.bfloat16
    if dtype_str == "fp16":
        return torch.float16
    if dtype_str == "fp32":
        return torch.float32
    raise ValueError(f"Unknown dtype: {dtype_str}")


def _prepare_tokenizer(model_dir: Path, trust_remote_code: bool):
    tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=trust_remote_code)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def _get_generation_config(model_dir: Path) -> GenerationConfig | None:
    try:
        return GenerationConfig.from_pretrained(str(model_dir))
    except Exception:
        return None


def _run_generation(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    gen_model_dir: Path,
    max_prompt_len: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_prompt_len,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    gen_config = _get_generation_config(gen_model_dir)

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    if gen_config is not None:
        gen_kwargs["generation_config"] = gen_config

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    prompt_len = inputs["input_ids"].shape[1]
    response_ids = output_ids[:, prompt_len:]
    response_text = tokenizer.decode(response_ids[0], skip_special_tokens=True)
    return output_ids, response_ids, response_text, prompt_len


def _compute_critic_values(critic, input_ids: torch.Tensor, response_len: int) -> torch.Tensor:
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        outputs = critic(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

    # TRL AutoModelForCausalLMWithValueHead returns values at index 2
    if hasattr(critic, "v_head"):
        values = outputs[2]
    else:
        values = outputs.logits

    if values.dim() == 3:
        values = values.squeeze(-1)

    # Align with training behavior: use values for response tokens only.
    if response_len > 0:
        values = values[:, -response_len - 1 : -1]
    return values


def _save_outputs(
    out_dir: Path,
    prompt: str,
    response_text: str,
    response_ids: torch.Tensor,
    response_values: torch.Tensor,
    tokenizer,
    sample_index: int,
    ref_response,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    tokens = tokenizer.convert_ids_to_tokens(response_ids[0].tolist())
    values = response_values[0].detach().cpu().tolist()

    rows = []
    for i, (tok, val, tid) in enumerate(zip(tokens, values, response_ids[0].tolist())):
        rows.append({"index": i, "token_id": tid, "token": tok, "value": float(val)})

    data_path = out_dir / "critic_values.jsonl"
    with data_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    summary = {
        "sample_index": sample_index,
        "prompt": prompt,
        "response": response_text,
        "reference_response": None if ref_response is None else str(ref_response),
    }
    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2)

    try:
        import matplotlib.pyplot as plt  # noqa: WPS433

        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(values, linewidth=1.5)
        ax.set_title("Critic Values Over Response Tokens")
        ax.set_xlabel("Token index in response")
        ax.set_ylabel("Value")
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(out_dir / "critic_values.png", dpi=150)
        plt.close(fig)
    except Exception as exc:
        print(f"[warn] Failed to write plot: {exc}")

    print(f"[saved] {data_path}")
    print(f"[saved] {summary_path}")
    print(f"[saved] {out_dir / 'critic_values.png'}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Debug PPO critic values for a single prompt.")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="/nfs/shuozhe/verl/train_log/max_response_length=2048/global_step_200",
        help="Path to the PPO checkpoint directory (contains actor/critic).",
    )
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to training parquet file.")
    parser.add_argument("--prompt_key", type=str, default="prompt", help="Prompt column key in dataset.")
    parser.add_argument(
        "--response_key",
        type=str,
        default="ground_truth",
        help="Optional response column key for reference output.",
    )
    parser.add_argument("--sample_index", type=int, default=None, help="Row index to sample from dataset.")
    parser.add_argument("--seed", type=int, default=0, help="Seed when choosing a random row.")
    parser.add_argument("--max_prompt_len", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu. Default: auto")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/nfs/shuozhe/verl/outputs/critic_debug",
        help="Output directory for plots and token/value dumps.",
    )
    parser.add_argument(
        "--skip_merge",
        action="store_true",
        help="Assume HF-merged models already exist and skip merging.",
    )
    args = parser.parse_args()

    ckpt_dir = Path(args.checkpoint_dir)
    actor_ckpt = ckpt_dir / "actor"
    critic_ckpt = ckpt_dir / "critic"

    merged_root = ckpt_dir / "merged_hf"
    actor_hf = merged_root / "actor"
    critic_hf = merged_root / "critic"

    if not args.skip_merge:
        if not _has_hf_weights(actor_hf):
            _merge_fsdp_checkpoint(actor_ckpt, actor_hf)
        if not _has_hf_weights(critic_hf):
            _merge_fsdp_checkpoint(critic_ckpt, critic_hf)

    if not _has_hf_weights(actor_hf):
        raise FileNotFoundError(f"Actor HF weights not found in {actor_hf}")
    if not _has_hf_weights(critic_hf):
        raise FileNotFoundError(f"Critic HF weights not found in {critic_hf}")

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = _get_dtype(args.dtype)

    tokenizer = _prepare_tokenizer(actor_hf, trust_remote_code=args.trust_remote_code)
    actor = _load_policy(actor_hf, dtype=dtype, device=device, trust_remote_code=args.trust_remote_code)
    critic = _load_critic(critic_hf, dtype=dtype, device=device, trust_remote_code=args.trust_remote_code)

    prompt, ref_response, sample_index = _load_prompt(
        dataset_path=args.dataset_path,
        prompt_key=args.prompt_key,
        response_key=args.response_key,
        sample_index=args.sample_index,
        seed=args.seed,
    )
    prompt = _normalize_prompt(prompt, tokenizer)

    output_ids, response_ids, response_text, prompt_len = _run_generation(
        model=actor,
        tokenizer=tokenizer,
        prompt=prompt,
        device=device,
        gen_model_dir=actor_hf,
        max_prompt_len=args.max_prompt_len,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=args.do_sample,
    )

    response_len = response_ids.shape[1]
    if response_len == 0:
        print("[warn] Model generated zero tokens. Consider increasing max_new_tokens.")

    response_values = _compute_critic_values(critic, output_ids, response_len)

    print("\n=== Prompt ===\n")
    print(prompt)
    print("\n=== Policy Response ===\n")
    print(response_text)
    if ref_response is not None:
        print("\n=== Reference Response ===\n")
        print(ref_response)

    out_dir = Path(args.out_dir)
    _save_outputs(
        out_dir=out_dir,
        prompt=prompt,
        response_text=response_text,
        response_ids=response_ids,
        response_values=response_values,
        tokenizer=tokenizer,
        sample_index=sample_index,
        ref_response=ref_response,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
