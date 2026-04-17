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


def has_hf_config(model_dir: Path) -> bool:
    return model_dir.exists() and (model_dir / "config.json").is_file()


def has_complete_hf_checkpoint(model_dir: Path) -> bool:
    return has_hf_weights(model_dir) and has_hf_config(model_dir)


def _candidate_hf_source_dirs(
    checkpoint_dir: Path,
    *,
    component: str,
    hf_source_dir: Path | None,
) -> list[Path]:
    local_dir = checkpoint_dir / component
    candidates: list[Path] = []

    def add(candidate: Path | None) -> None:
        if candidate is None:
            return
        candidate = Path(candidate)
        if candidate not in candidates:
            candidates.append(candidate)

    add(hf_source_dir)
    add(checkpoint_dir / "merged_hf" / component)
    add(local_dir / "huggingface")
    add(local_dir)
    add(checkpoint_dir / "huggingface" / component)
    add(checkpoint_dir / "huggingface")
    return candidates


def resolve_hf_source_dir(
    checkpoint_dir: Path,
    *,
    component: str,
    hf_source_dir: Path | None = None,
) -> Path:
    candidates = _candidate_hf_source_dirs(
        checkpoint_dir,
        component=component,
        hf_source_dir=hf_source_dir,
    )
    for candidate in candidates:
        if has_hf_config(candidate):
            return candidate

    tried = "\n".join(f"- {candidate}" for candidate in candidates)
    component_flag = f"--{component}_hf_source_dir"
    raise FileNotFoundError(
        f"Unable to locate Hugging Face config metadata for {component!r} under checkpoint {checkpoint_dir}.\n"
        f"Tried:\n{tried}\n"
        f"If this checkpoint was copied without its per-component 'huggingface/' folder, pass {component_flag} "
        "to a directory containing the original config/tokenizer files."
    )


def merge_fsdp_checkpoint(local_dir: Path, target_dir: Path, *, hf_model_config_path: Path | None = None) -> None:
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
    if hf_model_config_path is not None:
        cmd.extend(["--hf_model_config_path", str(hf_model_config_path)])
    subprocess.run(cmd, check=True)


def ensure_merged_component_checkpoint(
    checkpoint_dir: Path,
    *,
    component: str,
    merged_root: Path | None = None,
    hf_source_dir: Path | None = None,
    skip_merge: bool = False,
) -> Path:
    if component not in {"actor", "critic"}:
        raise ValueError(f"Unsupported checkpoint component: {component}")

    merged_root = merged_root or (checkpoint_dir / "merged_hf")
    target_dir = merged_root / component
    local_dir = checkpoint_dir / component

    # Prefer any complete HF checkpoint that already exists so we do not
    # re-merge shards unnecessarily on every evaluation run.
    for existing_dir in (target_dir, checkpoint_dir / "merged_hf" / component, local_dir):
        if has_complete_hf_checkpoint(existing_dir):
            return existing_dir

    if not skip_merge and not has_complete_hf_checkpoint(target_dir):
        resolved_hf_source_dir = resolve_hf_source_dir(
            checkpoint_dir,
            component=component,
            hf_source_dir=hf_source_dir,
        )
        merge_fsdp_checkpoint(local_dir, target_dir, hf_model_config_path=resolved_hf_source_dir)

    if not has_complete_hf_checkpoint(target_dir):
        if skip_merge:
            raise FileNotFoundError(
                f"{component.capitalize()} HF checkpoint not found in {target_dir}. "
                "Disable --skip_merge or provide a checkpoint that already contains merged HF weights."
            )
        raise FileNotFoundError(
            f"{component.capitalize()} HF checkpoint is incomplete in {target_dir}. "
            "Expected merged weights plus config.json after the merge step."
        )
    return target_dir


def ensure_merged_checkpoints(
    checkpoint_dir: Path,
    *,
    merged_root: Path | None = None,
    actor_hf_source_dir: Path | None = None,
    critic_hf_source_dir: Path | None = None,
    skip_merge: bool = False,
) -> tuple[Path, Path]:
    actor_hf = ensure_merged_component_checkpoint(
        checkpoint_dir,
        component="actor",
        merged_root=merged_root,
        hf_source_dir=actor_hf_source_dir,
        skip_merge=skip_merge,
    )
    critic_hf = ensure_merged_component_checkpoint(
        checkpoint_dir,
        component="critic",
        merged_root=merged_root,
        hf_source_dir=critic_hf_source_dir,
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
