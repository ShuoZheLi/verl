import json
import logging
import os
from collections.abc import Mapping
from typing import Any, Optional

import torch
from torch import nn

logger = logging.getLogger(__name__)

SUPPORTED_MODES = {
    "safe_svd_lowmag",
    "non_principal",
    "low_magnitude",
    "principal",
    "random_same_density",
}
DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
DEFAULT_EXCLUDE_KEYWORDS = ["embed", "lm_head", "norm", "layernorm", "rmsnorm"]
FSDP_PREFIX = "_fsdp_wrapped_module."


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _as_list(value: Any, default: Optional[list[str]] = None) -> list[str]:
    if value is None:
        return list(default or [])
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    return list(value)


def _canonical_name(name: str) -> str:
    return name.replace(FSDP_PREFIX, "")


def _name_aliases(name: str) -> list[str]:
    canonical_name = _canonical_name(name)
    aliases = [canonical_name]
    if canonical_name.startswith("model."):
        aliases.append(canonical_name[len("model.") :])
    else:
        aliases.append(f"model.{canonical_name}")
    return list(dict.fromkeys(aliases))


def _validate_alpha(alpha: float, name: str) -> float:
    alpha = float(alpha)
    if alpha < 0 or alpha > 1:
        raise ValueError(f"{name} must be in [0, 1], got {alpha}")
    return alpha


def fraction_to_count(numel: int, fraction: float) -> int:
    fraction = _validate_alpha(fraction, "fraction")
    return int(round(numel * fraction))


def top_fraction_mask(scores: torch.Tensor, fraction: float) -> torch.BoolTensor:
    fraction = _validate_alpha(fraction, "fraction")
    flat = scores.reshape(-1)
    count = fraction_to_count(flat.numel(), fraction)
    mask = torch.zeros(flat.numel(), dtype=torch.bool, device=flat.device)
    if count <= 0:
        return mask.reshape_as(scores)
    if count >= flat.numel():
        return torch.ones_like(mask, dtype=torch.bool).reshape_as(scores)
    _, indices = torch.topk(flat, k=count, largest=True, sorted=False)
    mask[indices] = True
    return mask.reshape_as(scores)


def bottom_fraction_mask(scores: torch.Tensor, fraction: float) -> torch.BoolTensor:
    fraction = _validate_alpha(fraction, "fraction")
    flat = scores.reshape(-1)
    count = fraction_to_count(flat.numel(), fraction)
    mask = torch.zeros(flat.numel(), dtype=torch.bool, device=flat.device)
    if count <= 0:
        return mask.reshape_as(scores)
    if count >= flat.numel():
        return torch.ones_like(mask, dtype=torch.bool).reshape_as(scores)
    _, indices = torch.topk(flat, k=count, largest=False, sorted=False)
    mask[indices] = True
    return mask.reshape_as(scores)


def keep_fraction_mask_from_scores(scores: torch.Tensor, keep_fraction: float) -> torch.BoolTensor:
    """Return True for trainable/kept entries with highest pruning-importance scores."""
    return top_fraction_mask(scores.detach().float(), keep_fraction)


def _load_score_metadata(score_dir: str | os.PathLike[str]) -> tuple[str, dict[str, str], dict[str, Any]]:
    score_dir = os.fspath(score_dir)
    metadata_path = os.path.join(score_dir, "metadata.json")
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"WANDA score metadata not found: {metadata_path}")
    with open(metadata_path, encoding="utf-8") as handle:
        metadata = json.load(handle)
    modules = metadata.get("modules")
    if not isinstance(modules, Mapping) or not modules:
        raise ValueError(f"Invalid WANDA score metadata {metadata_path}: missing non-empty 'modules' mapping")
    score_key = str(metadata.get("score_key", "wanda"))
    return score_key, {str(name): str(file_name) for name, file_name in modules.items()}, metadata


def _load_module_score(score_dir: str | os.PathLike[str], file_name: str, score_key: str) -> torch.Tensor:
    score_path = os.path.join(os.fspath(score_dir), file_name)
    if not os.path.isfile(score_path):
        raise FileNotFoundError(f"WANDA score tensor not found: {score_path}")
    payload = torch.load(score_path, map_location="cpu", weights_only=False)
    if isinstance(payload, Mapping):
        if score_key not in payload:
            raise KeyError(f"Score file {score_path} does not contain key {score_key!r}; available keys={list(payload)}")
        payload = payload[score_key]
    if not torch.is_tensor(payload):
        raise ValueError(f"Score file {score_path} did not contain a tensor for key {score_key!r}")
    return payload.detach().cpu().float()


def _score_name_aliases(module_name: str) -> list[str]:
    canonical_name = _canonical_name(module_name).removesuffix(".weight")
    aliases = [canonical_name]
    if canonical_name.startswith("model."):
        aliases.append(canonical_name[len("model.") :])
    else:
        aliases.append(f"model.{canonical_name}")
    return list(dict.fromkeys(aliases))


def build_masks_from_wanda_scores(
    model: nn.Module,
    score_dir: str | os.PathLike[str],
    keep_fraction: float,
    config: Any = None,
) -> tuple[dict[str, torch.BoolTensor], dict[str, Any]]:
    """Build sparse-update masks from saved WANDA pruning scores.

    WANDA scores measure pruning importance: higher score means the parameter is more important to keep.
    The returned bool masks use VERL sparse-update semantics: True entries remain trainable, False
    entries are frozen/restored after optimizer steps. Therefore, for sparsity s, use
    keep_fraction = 1 - s and select the top keep_fraction scores per tensor.
    """
    keep_fraction = _validate_alpha(keep_fraction, "keep_fraction")
    target_modules = _as_list(_cfg_get(config, "target_modules", DEFAULT_TARGET_MODULES), DEFAULT_TARGET_MODULES)
    exclude_keywords = _as_list(_cfg_get(config, "exclude_keywords", DEFAULT_EXCLUDE_KEYWORDS), DEFAULT_EXCLUDE_KEYWORDS)
    apply_to_bias = bool(_cfg_get(config, "apply_to_bias", False))
    score_key, score_index, score_metadata = _load_score_metadata(score_dir)
    masks: dict[str, torch.BoolTensor] = {}
    missing_scores: list[str] = []
    shape_mismatches: list[dict[str, Any]] = []

    target_param_index = 0
    for name, param in model.named_parameters():
        canonical_name = _canonical_name(name)
        if not should_mask_param(canonical_name, param, target_modules, exclude_keywords, apply_to_bias):
            continue
        score_file = None
        for alias in _score_name_aliases(canonical_name):
            score_file = score_index.get(alias)
            if score_file is not None:
                break
        if score_file is None:
            missing_scores.append(canonical_name)
            continue
        target_param_index += 1
        if target_param_index == 1 or target_param_index % 25 == 0:
            logger.info("building WANDA sparse-update mask for target tensor %d: %s", target_param_index, canonical_name)
        scores = _load_module_score(score_dir, score_file, score_key)
        if tuple(scores.shape) != tuple(param.shape):
            shape_mismatches.append(
                {"name": canonical_name, "score_shape": list(scores.shape), "param_shape": list(param.shape)}
            )
            continue
        masks[canonical_name] = keep_fraction_mask_from_scores(scores, keep_fraction).cpu().bool()

    if missing_scores:
        raise ValueError(f"WANDA score directory is missing scores for target params: {missing_scores[:10]}")
    if shape_mismatches:
        raise ValueError(f"WANDA score tensors do not match model parameter shapes: {shape_mismatches[:10]}")
    metadata = sparse_mask_metadata(
        masks,
        {
            "mode": "wanda_top",
            "target_modules": target_modules,
            "exclude_keywords": exclude_keywords,
        },
        extra={
            "score_dir": os.fspath(score_dir),
            "score_key": score_key,
            "score_model_name": score_metadata.get("model_name"),
            "score_calibration_path": score_metadata.get("calibration_path"),
            "sparsity": 1.0 - keep_fraction,
            "keep_fraction": keep_fraction,
            "selection_rule": "top WANDA scores are trainable/kept; lower scores are frozen",
        },
    )
    return masks, metadata


def should_mask_param(
    name: str,
    param: torch.Tensor,
    target_modules: Optional[list[str]] = None,
    exclude_keywords: Optional[list[str]] = None,
    apply_to_bias: bool = False,
) -> bool:
    canonical_name = _canonical_name(name).lower()
    target_modules = _as_list(target_modules, DEFAULT_TARGET_MODULES)
    exclude_keywords = _as_list(exclude_keywords, DEFAULT_EXCLUDE_KEYWORDS)

    if param.is_meta:
        raise ValueError(f"Cannot build sparse-update mask for meta parameter {name}.")
    if any(keyword.lower() in canonical_name for keyword in exclude_keywords):
        return False
    if not apply_to_bias and canonical_name.endswith(".bias"):
        return False
    if param.ndim != 2:
        return False
    if target_modules and not any(f".{target.lower()}." in f".{canonical_name}" for target in target_modules):
        return False
    return True


def should_mask_param_name(
    name: str,
    target_modules: Optional[list[str]] = None,
    exclude_keywords: Optional[list[str]] = None,
    apply_to_bias: bool = False,
) -> bool:
    canonical_name = _canonical_name(name).lower()
    target_modules = _as_list(target_modules, DEFAULT_TARGET_MODULES)
    exclude_keywords = _as_list(exclude_keywords, DEFAULT_EXCLUDE_KEYWORDS)
    if any(keyword.lower() in canonical_name for keyword in exclude_keywords):
        return False
    if canonical_name.endswith(".bias"):
        return apply_to_bias and any(f".{target.lower()}." in f".{canonical_name}" for target in target_modules)
    if not canonical_name.endswith(".weight"):
        return False
    if target_modules and not any(f".{target.lower()}." in f".{canonical_name}" for target in target_modules):
        return False
    return True


def _rank_k_reconstruction(weight: torch.Tensor, rank_k: int) -> torch.Tensor:
    max_rank = min(weight.shape)
    if max_rank <= 0:
        raise ValueError(f"Cannot compute SVD for empty tensor with shape {tuple(weight.shape)}")
    clipped_rank = min(int(rank_k), max_rank)
    if clipped_rank <= 0:
        raise ValueError(f"rank_k must be positive after clipping, got rank_k={rank_k}, max_rank={max_rank}")
    if clipped_rank != int(rank_k):
        logger.warning("Clipped sparse-update SVD rank_k from %s to %s for shape %s", rank_k, clipped_rank, tuple(weight.shape))
    u, s, vh = torch.linalg.svd(weight.float(), full_matrices=False)
    return (u[:, :clipped_rank] * s[:clipped_rank]) @ vh[:clipped_rank, :]


def build_safe_svd_lowmag_mask_for_tensor(
    W: torch.Tensor,
    rank_k: int,
    alpha_princ: float,
    alpha_low: float,
    mode: str = "safe_svd_lowmag",
    svd_device: Optional[str] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.BoolTensor:
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"Unsupported sparse_update mode {mode!r}; expected one of {sorted(SUPPORTED_MODES)}")
    alpha_princ = _validate_alpha(alpha_princ, "alpha_princ")
    alpha_low = _validate_alpha(alpha_low, "alpha_low")
    device = torch.device(svd_device) if svd_device is not None else W.device
    weight = W.detach().to(device=device, dtype=torch.float32)

    principal_mask = None
    low_mask = None
    safe_mask = None
    if mode in {"safe_svd_lowmag", "non_principal", "principal", "random_same_density"}:
        reconstructed = _rank_k_reconstruction(weight, rank_k)
        principal_mask = top_fraction_mask(reconstructed.abs(), alpha_princ)
    if mode in {"safe_svd_lowmag", "low_magnitude", "random_same_density"}:
        low_mask = bottom_fraction_mask(weight.abs(), alpha_low)

    if mode == "safe_svd_lowmag":
        safe_mask = (~principal_mask) | low_mask
        train_mask = safe_mask
    elif mode == "non_principal":
        train_mask = ~principal_mask
    elif mode == "low_magnitude":
        train_mask = low_mask
    elif mode == "principal":
        train_mask = principal_mask
    else:
        safe_mask = (~principal_mask) | low_mask
        density = safe_mask.float().mean().item()
        train_mask = torch.rand(weight.shape, device=device, generator=generator) < density

    return train_mask.detach().cpu().bool()


def build_masks_from_model(model: nn.Module, config: Any) -> dict[str, torch.BoolTensor]:
    mode = _cfg_get(config, "mode", "safe_svd_lowmag")
    if mode == "wanda_top":
        raise ValueError("mode='wanda_top' requires build_masks_from_wanda_scores(...) or tools/build_sparse_update_mask.py --wanda_score_dir")
    if mode == "random_same_density" or _cfg_get(config, "random_mask_same_density", False):
        mode = "random_same_density"
    rank_k = int(_cfg_get(config, "rank_k", 128))
    alpha_princ = float(_cfg_get(config, "alpha_princ", 0.5))
    alpha_low = float(_cfg_get(config, "alpha_low", 0.5))
    target_modules = _as_list(_cfg_get(config, "target_modules", DEFAULT_TARGET_MODULES), DEFAULT_TARGET_MODULES)
    exclude_keywords = _as_list(_cfg_get(config, "exclude_keywords", DEFAULT_EXCLUDE_KEYWORDS), DEFAULT_EXCLUDE_KEYWORDS)
    apply_to_bias = bool(_cfg_get(config, "apply_to_bias", False))
    svd_device = _cfg_get(config, "svd_device", None)
    dry_run = bool(_cfg_get(config, "dry_run_log_only", False))
    seed = int(_cfg_get(config, "seed", 42))

    masks: dict[str, torch.BoolTensor] = {}
    skipped: dict[str, str] = {}
    for name, param in model.named_parameters():
        canonical_name = _canonical_name(name)
        if param.ndim == 1 and "flat_param" in canonical_name:
            raise ValueError(
                "sparse_update currently requires named unflattened parameters / use_orig_params=True for FSDP. "
                f"Found flattened parameter {name}."
            )
        if not should_mask_param(canonical_name, param, target_modules, exclude_keywords, apply_to_bias):
            if any(keyword.lower() in canonical_name.lower() for keyword in exclude_keywords):
                skipped[canonical_name] = "excluded_keyword"
            elif param.ndim != 2:
                skipped[canonical_name] = "not_2d"
            else:
                skipped[canonical_name] = "not_target_module"
            continue
        mask = build_safe_svd_lowmag_mask_for_tensor(
            param,
            rank_k=rank_k,
            alpha_princ=alpha_princ,
            alpha_low=alpha_low,
            mode=mode,
            svd_device=svd_device,
            generator=torch.Generator(device=torch.device(svd_device) if svd_device is not None else param.device).manual_seed(
                seed + sum(ord(ch) for ch in canonical_name)
            ),
        )
        masks[canonical_name] = mask
        logger.info("sparse_update mask %s trainable_fraction=%.6f", canonical_name, mask.float().mean().item())

    if dry_run:
        logger.info("sparse_update dry_run_log_only=True; built %s masks but runtime should not enforce them", len(masks))
    if target_modules and not masks:
        logger.warning("sparse_update found no target parameters for target_modules=%s", target_modules)
    return masks


def sparse_mask_metadata(masks: dict[str, torch.BoolTensor], config: Any, extra: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    per_param_density = {name: mask.float().mean().item() for name, mask in masks.items()}
    total_numel = sum(mask.numel() for mask in masks.values())
    total_trainable = sum(mask.count_nonzero().item() for mask in masks.values())
    metadata = {
        "mode": _cfg_get(config, "mode", "safe_svd_lowmag"),
        "rank_k": int(_cfg_get(config, "rank_k", 128)),
        "alpha_princ": float(_cfg_get(config, "alpha_princ", 0.5)),
        "alpha_low": float(_cfg_get(config, "alpha_low", 0.5)),
        "target_modules": _as_list(_cfg_get(config, "target_modules", DEFAULT_TARGET_MODULES), DEFAULT_TARGET_MODULES),
        "exclude_keywords": _as_list(_cfg_get(config, "exclude_keywords", DEFAULT_EXCLUDE_KEYWORDS), DEFAULT_EXCLUDE_KEYWORDS),
        "linear_trainable_fraction": (total_trainable / total_numel) if total_numel else 0.0,
        "per_param_density": per_param_density,
        "num_masked_tensors": len(masks),
    }
    if extra:
        metadata.update(extra)
    return metadata


def save_sparse_masks(path: str, masks: dict[str, torch.BoolTensor], metadata: Optional[dict[str, Any]] = None) -> None:
    dirname = os.path.dirname(os.path.abspath(path))
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    payload = {
        "masks": {name: mask.detach().cpu().bool() for name, mask in masks.items()},
        "metadata": metadata or {},
    }
    torch.save(payload, path)


def load_sparse_masks(path: str) -> tuple[dict[str, torch.BoolTensor], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or "masks" not in payload:
        raise ValueError(f"Invalid sparse-update mask file {path}: expected payload with a 'masks' field")
    masks = {str(name): tensor.detach().cpu().bool() for name, tensor in payload["masks"].items()}
    metadata = dict(payload.get("metadata", {}))
    return masks, metadata


class SparseUpdateMaskManager:
    def __init__(self, model: nn.Module, masks: dict[str, torch.BoolTensor], config: Any, metadata: Optional[dict[str, Any]] = None):
        self.model = model
        self.config = config
        self.enabled = bool(_cfg_get(config, "enabled", False)) and not bool(_cfg_get(config, "dry_run_log_only", False))
        self.restore_frozen_after_step_enabled = bool(_cfg_get(config, "restore_frozen_after_step", True))
        self.zero_frozen_params_enabled = bool(_cfg_get(config, "zero_frozen_params", False))
        self.mask_optimizer_state_enabled = bool(_cfg_get(config, "mask_optimizer_state", True))
        self.verify_frozen_weights_enabled = bool(_cfg_get(config, "verify_frozen_weights", False))
        self.verification_interval = int(_cfg_get(config, "verification_interval", 10))
        self.verification_tolerance = float(_cfg_get(config, "verification_tolerance", 1e-6))
        self.strict_load = bool(_cfg_get(config, "strict_load", True))
        self.masks = { _canonical_name(name): mask.detach().cpu().bool() for name, mask in masks.items() }
        self.metadata = metadata or sparse_mask_metadata(self.masks, config)
        self.original_params: dict[str, torch.Tensor] = {}
        self.local_masks: dict[str, torch.BoolTensor] = {}
        self.num_missing_masks = 0
        self.num_shape_mismatches = 0
        self._step = 0
        self._param_names_by_id: dict[int, str] = {}
        self._fsdp_shard_infos_by_name = self._collect_fsdp_shard_infos_by_name(model)
        if self.enabled:
            self._initialize_original_params()

    def _collect_fsdp_shard_infos_by_name(self, model: nn.Module) -> dict[str, Any]:
        shard_infos = {}
        handle_prefixes_by_id: dict[int, list[str]] = {}
        module_names_by_id: dict[int, list[str]] = {}
        handles = []
        all_handles = getattr(model, "_all_handles", None)
        if all_handles is not None:
            handles.extend(all_handles)

        for module_name, module in model.named_modules():
            canonical_module_name = _canonical_name(module_name).strip(".")
            module_names_by_id.setdefault(id(module), [])
            if canonical_module_name not in module_names_by_id[id(module)]:
                module_names_by_id[id(module)].append(canonical_module_name)
            handle = getattr(module, "_handle", None)
            if handle is not None:
                handles.append(handle)
                handle_prefixes_by_id.setdefault(id(handle), [])
                if canonical_module_name not in handle_prefixes_by_id[id(handle)]:
                    handle_prefixes_by_id[id(handle)].append(canonical_module_name)

        seen_handles = set()
        for handle in handles:
            if handle is None or id(handle) in seen_handles:
                continue
            seen_handles.add(id(handle))
            flat_param = getattr(handle, "flat_param", None)
            param_infos = getattr(flat_param, "_param_infos", None)
            shard_param_infos = getattr(flat_param, "_shard_param_infos", None)
            if param_infos is None or shard_param_infos is None:
                continue
            for param_info, shard_param_info in zip(param_infos, shard_param_infos):
                param_name = getattr(param_info, "param_name", param_info[0])
                param_module = getattr(param_info, "module", param_info[1])
                module_name = getattr(param_info, "module_name", param_info[2])
                relative_name = f"{module_name}.{param_name}" if module_name else param_name
                candidate_names = [relative_name]
                for module_path in module_names_by_id.get(id(param_module), []):
                    candidate_names.append(f"{module_path}.{param_name}" if module_path else param_name)
                for prefix in handle_prefixes_by_id.get(id(handle), []):
                    if prefix:
                        candidate_names.append(f"{prefix}.{relative_name}")
                for full_name in candidate_names:
                    shard_infos[_canonical_name(full_name)] = shard_param_info
        return shard_infos

    def _get_fsdp_shard_info(self, name: str) -> Any:
        for alias in _name_aliases(name):
            shard_info = self._fsdp_shard_infos_by_name.get(alias)
            if shard_info is not None:
                return shard_info
        return None

    def _get_mask(self, name: str) -> Optional[torch.BoolTensor]:
        for alias in _name_aliases(name):
            mask = self.masks.get(alias)
            if mask is not None:
                return mask
        return None

    @staticmethod
    def _has_seen_alias(name: str, seen: set[str]) -> bool:
        return any(alias in seen for alias in _name_aliases(name))

    def _iter_named_masked_params(self):
        for name, param in self.model.named_parameters():
            canonical_name = _canonical_name(name)
            if self._get_mask(canonical_name) is not None:
                yield canonical_name, param

    def _initialize_original_params(self) -> None:
        seen = set()
        target_param_names = set()
        named_param_names = {_canonical_name(name) for name, _ in self.model.named_parameters()}
        target_modules = _as_list(_cfg_get(self.config, "target_modules", DEFAULT_TARGET_MODULES), DEFAULT_TARGET_MODULES)
        exclude_keywords = _as_list(
            _cfg_get(self.config, "exclude_keywords", DEFAULT_EXCLUDE_KEYWORDS), DEFAULT_EXCLUDE_KEYWORDS
        )
        apply_to_bias = bool(_cfg_get(self.config, "apply_to_bias", False))
        for name in self._fsdp_shard_infos_by_name:
            matching_named_param = next((alias for alias in _name_aliases(name) if alias in named_param_names), None)
            if matching_named_param and should_mask_param_name(
                matching_named_param, target_modules, exclude_keywords, apply_to_bias
            ):
                target_param_names.add(matching_named_param)
        for name, param in self.model.named_parameters():
            canonical_name = _canonical_name(name)
            if param.ndim == 1 and "flat_param" in canonical_name:
                raise ValueError(
                    "sparse_update currently requires named unflattened parameters / use_orig_params=True for FSDP. "
                    f"Found flattened parameter {name}."
                )
            if should_mask_param(canonical_name, param, target_modules, exclude_keywords, apply_to_bias):
                target_param_names.add(canonical_name)
            mask = self._get_mask(canonical_name)
            if mask is None:
                continue
            local_mask = mask
            if tuple(mask.shape) != tuple(param.shape):
                local_mask = self._maybe_make_local_shard_mask(canonical_name, param, mask)
            if tuple(local_mask.shape) != tuple(param.shape):
                self.num_shape_mismatches += 1
                raise ValueError(
                    f"sparse_update mask shape mismatch for {canonical_name}: mask {tuple(mask.shape)} vs param {tuple(param.shape)}"
                )
            local_mask = local_mask.detach().cpu().bool()
            self.local_masks[canonical_name] = local_mask
            self.original_params[canonical_name] = self._make_original_snapshot(param, local_mask)
            self._param_names_by_id[id(param)] = canonical_name
            if self.zero_frozen_params_enabled:
                self._apply_param_mask_in_place(canonical_name, param)
            seen.add(canonical_name)
        missing = sorted(name for name in self.masks if not self._has_seen_alias(name, seen))
        uncovered = sorted(name for name in target_param_names if not self._has_seen_alias(name, seen))
        self.num_missing_masks = len(missing) + len(uncovered)
        if missing and self.strict_load:
            raise ValueError(
                "sparse_update masks do not match model parameters; "
                f"masks have no matching model params: {missing[:10]}"
            )
        if uncovered and self.strict_load:
            raise ValueError(
                "sparse_update mask_path is missing masks for target model parameters: "
                f"{uncovered[:10]}"
            )
        if missing:
            logger.warning("sparse_update loaded masks for missing params: %s", missing[:10])
        if uncovered:
            logger.warning("sparse_update missing masks for target params: %s", uncovered[:10])

    def _maybe_make_local_shard_mask(self, name: str, param: torch.Tensor, full_mask: torch.BoolTensor) -> torch.BoolTensor:
        shard_info = self._get_fsdp_shard_info(name)
        if shard_info is None:
            self.num_shape_mismatches += 1
            available_names = list(self._fsdp_shard_infos_by_name)[:10]
            raise ValueError(
                "sparse_update could not find FSDP shard metadata for parameter "
                f"{name} with local shape {tuple(param.shape)} and full mask shape {tuple(full_mask.shape)}. "
                f"Collected {len(self._fsdp_shard_infos_by_name)} FSDP shard metadata entries; "
                f"first entries: {available_names}."
            )
        if not shard_info.in_shard:
            if param.numel() != 0:
                raise ValueError(
                    f"sparse_update expected empty local shard for {name}, got shape {tuple(param.shape)}."
                )
            return torch.empty(0, dtype=torch.bool)
        start = shard_info.intra_param_start_idx
        end = shard_info.intra_param_end_idx
        if start is None or end is None:
            raise ValueError(f"sparse_update missing FSDP shard indices for {name}.")
        local_mask = full_mask.reshape(-1)[start : end + 1].contiguous()
        if local_mask.numel() != param.numel():
            raise ValueError(
                f"sparse_update local shard mask for {name} has {local_mask.numel()} entries, "
                f"but parameter local shard has {param.numel()} entries."
            )
        return local_mask

    def _mask_for_param(self, name: str, param: torch.Tensor) -> torch.Tensor:
        return self.local_masks[name].to(device=param.device, dtype=torch.bool)

    def _original_for_param(self, name: str, param: torch.Tensor) -> torch.Tensor:
        return self.original_params[name].to(device=param.device, dtype=param.dtype)

    def _make_original_snapshot(self, param: torch.Tensor, local_mask: torch.BoolTensor) -> torch.Tensor:
        snapshot = param.detach().cpu().clone()
        if self.zero_frozen_params_enabled:
            snapshot = torch.where(local_mask, snapshot, torch.zeros_like(snapshot))
        return snapshot

    @torch.no_grad()
    def _apply_param_mask_in_place(self, name: str, param: torch.Tensor) -> None:
        mask = self._mask_for_param(name, param)
        param.mul_(mask.to(dtype=param.dtype))

    def apply_grad_mask(self) -> None:
        if not self.enabled:
            return
        for name, param in self._iter_named_masked_params():
            if param.grad is None:
                continue
            mask = self._mask_for_param(name, param)
            if isinstance(param.grad, torch.Tensor) and tuple(param.grad.shape) == tuple(mask.shape):
                param.grad.mul_(mask.to(dtype=param.grad.dtype))

    def mask_optimizer_state(self, optimizer: torch.optim.Optimizer) -> None:
        if not self.enabled or not self.mask_optimizer_state_enabled:
            return
        state_names = {"exp_avg", "exp_avg_sq", "max_exp_avg_sq", "momentum_buffer"}
        for group in optimizer.param_groups:
            for param in group["params"]:
                name = self._param_names_by_id.get(id(param))
                if name is None or name not in self.local_masks:
                    continue
                mask = self._mask_for_param(name, param)
                state = optimizer.state.get(param, {})
                for state_name in state_names:
                    state_tensor = state.get(state_name, None)
                    if isinstance(state_tensor, torch.Tensor) and tuple(state_tensor.shape) == tuple(mask.shape):
                        state_tensor.mul_(mask.to(device=state_tensor.device, dtype=state_tensor.dtype))

    @torch.no_grad()
    def restore_frozen_params(self) -> None:
        if not self.enabled or not self.restore_frozen_after_step_enabled:
            return
        for name, param in self._iter_named_masked_params():
            mask = self._mask_for_param(name, param)
            original = self._original_for_param(name, param)
            param.copy_(torch.where(mask, param, original))

    @torch.no_grad()
    def verify_frozen_params(self, step: Optional[int] = None) -> dict[str, float]:
        metrics = {
            "sparse_update/frozen_max_abs_change": 0.0,
            "sparse_update/frozen_mean_abs_change": 0.0,
            "sparse_update/num_masked_tensors": float(len(self.masks)),
            "sparse_update/num_missing_masks": float(self.num_missing_masks),
            "sparse_update/num_shape_mismatches": float(self.num_shape_mismatches),
        }
        if not self.enabled:
            return metrics
        total_abs_change = 0.0
        total_frozen = 0
        max_abs_change = 0.0
        for name, param in self._iter_named_masked_params():
            mask = self._mask_for_param(name, param)
            original = self._original_for_param(name, param)
            frozen_change = (param.detach() - original).abs().masked_select(~mask)
            if frozen_change.numel() == 0:
                continue
            max_abs_change = max(max_abs_change, frozen_change.max().item())
            total_abs_change += frozen_change.sum().item()
            total_frozen += frozen_change.numel()
        metrics["sparse_update/frozen_max_abs_change"] = max_abs_change
        metrics["sparse_update/frozen_mean_abs_change"] = total_abs_change / total_frozen if total_frozen else 0.0
        if max_abs_change > self.verification_tolerance:
            raise RuntimeError(
                f"sparse_update frozen weights changed by {max_abs_change} at step {step}; "
                f"tolerance={self.verification_tolerance}"
            )
        return metrics

    def maybe_verify(self) -> dict[str, float]:
        self._step += 1
        base_metrics = self.metrics()
        if not self.verify_frozen_weights_enabled:
            return base_metrics
        if self.verification_interval <= 0 or self._step % self.verification_interval == 0:
            base_metrics.update(self.verify_frozen_params(step=self._step))
        return base_metrics

    def metrics(self) -> dict[str, float]:
        return {
            "sparse_update/trainable_fraction": float(self.metadata.get("linear_trainable_fraction", 0.0)),
            "sparse_update/num_masked_tensors": float(len(self.masks)),
            "sparse_update/num_missing_masks": float(self.num_missing_masks),
            "sparse_update/num_shape_mismatches": float(self.num_shape_mismatches),
            "sparse_update/zero_frozen_params": float(self.zero_frozen_params_enabled),
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "masks": {name: mask.detach().cpu().bool() for name, mask in self.masks.items()},
            "local_masks": {name: mask.detach().cpu().bool() for name, mask in self.local_masks.items()},
            "original_params": {name: tensor.detach().cpu() for name, tensor in self.original_params.items()},
            "metadata": self.metadata,
            "step": self._step,
            "zero_frozen_params_enabled": self.zero_frozen_params_enabled,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if "masks" in state:
            self.masks = {str(name): tensor.detach().cpu().bool() for name, tensor in state["masks"].items()}
        if "local_masks" in state:
            self.local_masks = {str(name): tensor.detach().cpu().bool() for name, tensor in state["local_masks"].items()}
        if "original_params" in state:
            self.original_params = {str(name): tensor.detach().cpu() for name, tensor in state["original_params"].items()}
        self.metadata = dict(state.get("metadata", self.metadata))
        self.zero_frozen_params_enabled = bool(state.get("zero_frozen_params_enabled", self.zero_frozen_params_enabled))
        self._step = int(state.get("step", self._step))
