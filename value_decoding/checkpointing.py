from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.modeling_outputs import TokenClassifierOutput

from verl.utils.model import load_valuehead_model


HF_EXACT_WEIGHT_NAMES = ("model.safetensors", "pytorch_model.bin")
HF_INDEX_WEIGHT_NAMES = ("model.safetensors.index.json", "pytorch_model.bin.index.json")
HF_SHARD_PATTERNS = ("model-*.safetensors", "pytorch_model-*.bin")


def _read_hf_index_shard_names(index_path: Path) -> list[str]:
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, dict):
        return []

    shard_names: list[str] = []
    seen: set[str] = set()
    for candidate in weight_map.values():
        if not isinstance(candidate, str) or not candidate:
            continue
        if candidate in seen:
            continue
        seen.add(candidate)
        shard_names.append(candidate)
    return shard_names


def find_missing_hf_weight_files(model_dir: Path) -> list[Path]:
    if not model_dir.exists():
        return []

    for name in HF_EXACT_WEIGHT_NAMES:
        if (model_dir / name).is_file():
            return []

    for name in HF_INDEX_WEIGHT_NAMES:
        index_path = model_dir / name
        if not index_path.is_file():
            continue
        shard_names = _read_hf_index_shard_names(index_path)
        if not shard_names:
            return [index_path]
        return [model_dir / shard_name for shard_name in shard_names if not (model_dir / shard_name).is_file()]

    for pattern in HF_SHARD_PATTERNS:
        if any(model_dir.glob(pattern)):
            return []

    return []


def has_hf_weights(model_dir: Path) -> bool:
    if not model_dir.exists():
        return False

    if find_missing_hf_weight_files(model_dir):
        return False

    for name in HF_EXACT_WEIGHT_NAMES:
        if (model_dir / name).is_file():
            return True
    for name in HF_INDEX_WEIGHT_NAMES:
        if (model_dir / name).is_file():
            return True
    for pattern in HF_SHARD_PATTERNS:
        if any(model_dir.glob(pattern)):
            return True
    return False


def has_hf_config(model_dir: Path) -> bool:
    return model_dir.exists() and (model_dir / "config.json").is_file()


def has_complete_hf_checkpoint(model_dir: Path) -> bool:
    return has_hf_weights(model_dir) and has_hf_config(model_dir)


def has_fsdp_checkpoint_shards(model_dir: Path) -> bool:
    if not model_dir.exists():
        return False
    if (model_dir / "fsdp_config.json").is_file():
        return True
    return any(model_dir.glob("model_world_size_*_rank_*.pt"))


def resolve_component_checkpoint_dir(checkpoint_dir: Path, *, component: str) -> Path:
    if component not in {"actor", "critic"}:
        raise ValueError(f"Unsupported checkpoint component: {component}")

    component_dir = checkpoint_dir / component
    candidates = (component_dir, checkpoint_dir)
    for candidate in candidates:
        if has_complete_hf_checkpoint(candidate) or has_fsdp_checkpoint_shards(candidate) or has_hf_config(candidate):
            return candidate
    return component_dir


def _candidate_hf_source_dirs(
    checkpoint_dir: Path,
    *,
    component: str,
    hf_source_dir: Path | None,
) -> list[Path]:
    local_dir = resolve_component_checkpoint_dir(checkpoint_dir, component=component)
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
    add(checkpoint_dir)
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
    local_dir = resolve_component_checkpoint_dir(checkpoint_dir, component=component)

    # Prefer any complete HF checkpoint that already exists so we do not
    # re-merge shards unnecessarily on every evaluation run.
    for existing_dir in (target_dir, checkpoint_dir / "merged_hf" / component, local_dir):
        if has_complete_hf_checkpoint(existing_dir):
            return existing_dir

    if not skip_merge and not has_complete_hf_checkpoint(target_dir):
        if not has_fsdp_checkpoint_shards(local_dir):
            if has_hf_config(local_dir):
                missing_files = find_missing_hf_weight_files(local_dir)
                missing_preview = ", ".join(str(path.name) for path in missing_files[:5])
                if len(missing_files) > 5:
                    missing_preview += ", ..."
                raise FileNotFoundError(
                    f"{component.capitalize()} checkpoint at {local_dir} looks like a Hugging Face checkpoint "
                    "directory, but it does not contain a complete set of model weight files. "
                    "Expected one of: model.safetensors, pytorch_model.bin, "
                    "model.safetensors.index.json, pytorch_model.bin.index.json, or model-*.safetensors shards. "
                    + (
                        f"Missing files referenced by the index: {missing_preview}. "
                        if missing_preview
                        else ""
                    )
                    + "Please re-download the checkpoint or point the script at a complete local copy."
                )
            raise FileNotFoundError(
                f"{component.capitalize()} checkpoint at {local_dir} is neither a complete Hugging Face checkpoint "
                "nor a raw FSDP checkpoint (missing fsdp_config.json and model_world_size_*_rank_*.pt shards)."
            )
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


def _is_process_reward_model_config(config) -> bool:
    architectures = getattr(config, "architectures", None) or ()
    return any("ProcessRewardModel" in str(arch) for arch in architectures)


class PRMCriticAdapter(nn.Module):
    """Adapt a process reward model to the scalar-value critic interface used by value_decoding."""

    def __init__(
        self,
        model: nn.Module,
        *,
        separator_token_id: int,
        pad_token_id: int,
    ) -> None:
        super().__init__()
        self.prm_model = model
        self.config = model.config
        self.separator_token_id = int(separator_token_id)
        self.pad_token_id = int(pad_token_id)
        self.is_prm_critic_adapter = True
        self.positive_class_index = 1
        self.prefix_scoring_batch_size = 32

    def _normalize_attention_mask(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if attention_mask is None:
            return torch.ones_like(input_ids, device=input_ids.device, dtype=torch.long)
        return attention_mask.to(device=input_ids.device, dtype=torch.long)

    def _positive_class_probs(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.dim() != 3 or logits.shape[-1] <= self.positive_class_index:
            raise ValueError(
                "PRM critic adapter expected token-classification logits with a positive class at index "
                f"{self.positive_class_index}, but received shape {tuple(logits.shape)}."
            )
        return torch.softmax(logits.float(), dim=-1)[..., self.positive_class_index]

    def _append_step_separator(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        valid_lengths = attention_mask.long().sum(dim=-1)
        if torch.any(valid_lengths <= 0):
            raise ValueError("Each PRM-scored sequence must contain at least one unmasked token.")

        appended_input_ids = torch.full(
            (batch_size, seq_len + 1),
            self.pad_token_id,
            device=input_ids.device,
            dtype=input_ids.dtype,
        )
        appended_attention_mask = torch.zeros(
            (batch_size, seq_len + 1),
            device=input_ids.device,
            dtype=attention_mask.dtype,
        )

        for batch_idx in range(batch_size):
            valid_token_ids = input_ids[batch_idx][attention_mask[batch_idx].bool()]
            valid_len = int(valid_token_ids.numel())
            appended_input_ids[batch_idx, :valid_len] = valid_token_ids
            appended_input_ids[batch_idx, valid_len] = self.separator_token_id
            appended_attention_mask[batch_idx, : valid_len + 1] = 1

        return appended_input_ids, appended_attention_mask, valid_lengths

    def _exact_prefix_scores_for_sequence(
        self,
        token_ids: torch.Tensor,
        *,
        start_index: int = 0,
    ) -> torch.Tensor:
        if token_ids.dim() != 1:
            raise ValueError(f"Expected a 1D token sequence, got shape {tuple(token_ids.shape)}")
        sequence_length = int(token_ids.shape[0])
        if sequence_length <= 0:
            raise ValueError("PRM exact prefix scoring requires at least one token.")
        if start_index < 0 or start_index >= sequence_length:
            raise ValueError(
                f"start_index must be within [0, {sequence_length - 1}], but received {start_index}."
            )

        scores: list[torch.Tensor] = []
        batch_size = max(int(self.prefix_scoring_batch_size), 1)
        positions = list(range(start_index, sequence_length))
        for batch_start in range(0, len(positions), batch_size):
            batch_positions = positions[batch_start : batch_start + batch_size]
            max_prefix_length = batch_positions[-1] + 1
            batch_input_ids = torch.full(
                (len(batch_positions), max_prefix_length + 1),
                self.pad_token_id,
                device=token_ids.device,
                dtype=token_ids.dtype,
            )
            batch_attention_mask = torch.zeros(
                (len(batch_positions), max_prefix_length + 1),
                device=token_ids.device,
                dtype=torch.long,
            )
            separator_positions: list[int] = []

            for row_index, position in enumerate(batch_positions):
                prefix_length = position + 1
                batch_input_ids[row_index, :prefix_length] = token_ids[:prefix_length]
                batch_input_ids[row_index, prefix_length] = self.separator_token_id
                batch_attention_mask[row_index, : prefix_length + 1] = 1
                separator_positions.append(prefix_length)

            outputs = self.prm_model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                use_cache=False,
                return_dict=True,
            )
            probs = self._positive_class_probs(outputs.logits)
            separator_index_tensor = torch.tensor(
                separator_positions,
                device=probs.device,
                dtype=torch.long,
            )
            batch_scores = probs.gather(dim=1, index=separator_index_tensor[:, None]).squeeze(1)
            scores.append(batch_scores)

        return torch.cat(scores, dim=0)

    @torch.inference_mode()
    def continuation_values(
        self,
        *,
        prefix_ids: torch.Tensor,
        continuation_ids: torch.Tensor,
    ) -> torch.Tensor:
        if prefix_ids.dim() != 2 or prefix_ids.shape[0] != 1:
            raise ValueError(
                "PRM continuation scoring currently expects prefix_ids with shape (1, prefix_length), "
                f"but received {tuple(prefix_ids.shape)}."
            )
        if continuation_ids.dim() == 2:
            if continuation_ids.shape[0] != 1:
                raise ValueError(
                    "PRM continuation scoring currently expects continuation_ids with batch size 1, "
                    f"but received {tuple(continuation_ids.shape)}."
                )
            continuation_ids = continuation_ids.squeeze(0)
        if continuation_ids.dim() != 1:
            raise ValueError(
                "PRM continuation scoring expects continuation_ids to be 1D or shape (1, num_tokens), "
                f"but received {tuple(continuation_ids.shape)}."
            )
        if continuation_ids.numel() <= 0:
            raise ValueError("PRM continuation scoring requires at least one continuation token.")

        full_sequence = torch.cat([prefix_ids[0], continuation_ids], dim=0)
        return self._exact_prefix_scores_for_sequence(full_sequence, start_index=int(prefix_ids.shape[1]))

    @torch.inference_mode()
    def sequence_last_values(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        base_attention_mask = self._normalize_attention_mask(input_ids, attention_mask)
        appended_input_ids, appended_attention_mask, valid_lengths = self._append_step_separator(
            input_ids, base_attention_mask
        )
        outputs = self.prm_model(
            input_ids=appended_input_ids,
            attention_mask=appended_attention_mask,
            use_cache=False,
            return_dict=True,
        )
        probs = self._positive_class_probs(outputs.logits)
        separator_positions = valid_lengths.to(device=probs.device, dtype=torch.long)
        return probs.gather(dim=1, index=separator_positions[:, None]).squeeze(1)

    @torch.inference_mode()
    def sequence_values(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        base_attention_mask = self._normalize_attention_mask(input_ids, attention_mask)
        values = torch.zeros(
            input_ids.shape,
            device=input_ids.device,
            dtype=torch.float32,
        )
        for batch_index in range(input_ids.shape[0]):
            valid_token_ids = input_ids[batch_index][base_attention_mask[batch_index].bool()]
            if valid_token_ids.numel() == 0:
                raise ValueError("Each PRM-scored sequence must contain at least one unmasked token.")
            exact_scores = self._exact_prefix_scores_for_sequence(valid_token_ids, start_index=0)
            values[batch_index, : exact_scores.shape[0]] = exact_scores
        return values

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values=None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ):
        unsupported_args = {
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "inputs_embeds": inputs_embeds,
            "labels": labels,
        }
        provided_args = [name for name, value in unsupported_args.items() if value is not None]
        if provided_args:
            raise ValueError(
                "PRM critic adapter does not support forwarding the following arguments: "
                + ", ".join(provided_args)
            )
        if input_ids is None:
            raise ValueError("PRM critic adapter requires input_ids.")
        values = self.sequence_values(input_ids=input_ids, attention_mask=attention_mask)
        if return_dict is False:
            return (values,)
        return TokenClassifierOutput(
            logits=values,
            hidden_states=None if output_hidden_states is False else None,
            attentions=None if output_attentions is False else None,
        )


def _load_prm_critic_model(
    model_dir: Path,
    *,
    dtype: torch.dtype,
    device: torch.device,
    trust_remote_code: bool = False,
):
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=trust_remote_code)
    separator_token_ids = tokenizer.encode("<extra_0>", add_special_tokens=False)
    if len(separator_token_ids) != 1:
        raise ValueError(
            "PRM critic adapter expected '<extra_0>' to tokenize to a single token, "
            f"but received ids={separator_token_ids}."
        )
    separator_token_id = int(separator_token_ids[0])
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = separator_token_id

    # PRM checkpoints rely on their bundled model implementation under auto_map["AutoModel"].
    model = AutoModel.from_pretrained(
        str(model_dir),
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    adapter = PRMCriticAdapter(
        model,
        separator_token_id=separator_token_id,
        pad_token_id=pad_token_id,
    )
    adapter.to(device)
    adapter.eval()
    return adapter


def load_critic_model(
    model_dir: Path,
    *,
    dtype: torch.dtype,
    device: torch.device,
    trust_remote_code: bool = False,
):
    config = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=trust_remote_code)
    if _is_process_reward_model_config(config):
        model = _load_prm_critic_model(
            model_dir,
            dtype=dtype,
            device=device,
            trust_remote_code=trust_remote_code,
        )
        return model

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
