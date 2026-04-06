from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from verl.utils.model import load_valuehead_model


def has_hf_weights(model_dir: Path) -> bool:
    if not model_dir.exists():
        return False

    exact_names = ("model.safetensors", "pytorch_model.bin")
    shard_patterns = ("model-*.safetensors", "pytorch_model-*.bin")
    for name in exact_names:
        if (model_dir / name).exists():
            return True
    for pattern in shard_patterns:
        if any(model_dir.glob(pattern)):
            return True
    return False


def merge_fsdp_checkpoint(local_dir: Path, target_dir: Path) -> None:
    target_dir.parent.mkdir(parents=True, exist_ok=True)
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
    subprocess.run(cmd, check=True)


def ensure_merged_component_checkpoint(
    checkpoint_dir: Path,
    *,
    component: str,
    merged_root: Path | None = None,
    skip_merge: bool = False,
) -> Path:
    if component not in {"actor", "critic"}:
        raise ValueError(f"Unsupported checkpoint component: {component}")

    merged_root = merged_root or (checkpoint_dir / "merged_hf")
    target_dir = merged_root / component

    if not skip_merge:
        local_dir = checkpoint_dir / component
        if not has_hf_weights(target_dir):
            merge_fsdp_checkpoint(local_dir, target_dir)

    if not has_hf_weights(target_dir):
        raise FileNotFoundError(f"{component.capitalize()} HF weights not found in {target_dir}")
    return target_dir


def ensure_merged_checkpoints(
    checkpoint_dir: Path,
    *,
    merged_root: Path | None = None,
    skip_merge: bool = False,
) -> tuple[Path, Path]:
    actor_hf = ensure_merged_component_checkpoint(
        checkpoint_dir,
        component="actor",
        merged_root=merged_root,
        skip_merge=skip_merge,
    )
    critic_hf = ensure_merged_component_checkpoint(
        checkpoint_dir,
        component="critic",
        merged_root=merged_root,
        skip_merge=skip_merge,
    )
    return actor_hf, critic_hf


def resolve_dtype(dtype_name: str) -> torch.dtype:
    name = dtype_name.lower()
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def resolve_device(device_name: str | None) -> torch.device:
    if device_name:
        return torch.device(device_name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_tokenizer(model_dir: Path, *, trust_remote_code: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=trust_remote_code)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError(f"Tokenizer at {model_dir} has no pad_token and no eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_actor_model(
    model_dir: Path,
    *,
    dtype: torch.dtype,
    device: torch.device,
    trust_remote_code: bool = False,
):
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    return model


def load_critic_model(
    model_dir: Path,
    *,
    dtype: torch.dtype,
    device: torch.device,
    trust_remote_code: bool = False,
):
    config = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=trust_remote_code)
    model = load_valuehead_model(
        str(model_dir),
        torch_dtype=dtype,
        model_config=config,
        trust_remote_code=trust_remote_code,
    )
    model.to(device)
    model.eval()
    return model


def resolve_eos_token_ids(model_dir: Path, tokenizer) -> tuple[int, ...]:
    eos_token_ids: list[int] = []
    generation_config = None
    try:
        generation_config = GenerationConfig.from_pretrained(str(model_dir))
    except Exception:
        generation_config = None

    for candidate in (getattr(generation_config, "eos_token_id", None), tokenizer.eos_token_id):
        if candidate is None:
            continue
        if isinstance(candidate, int):
            eos_token_ids.append(int(candidate))
            continue
        eos_token_ids.extend(int(token_id) for token_id in candidate if token_id is not None)

    return tuple(sorted(set(eos_token_ids)))
