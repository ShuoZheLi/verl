#!/usr/bin/env python3
"""Resolve actor/critic init paths to usable Hugging Face model directories.

This helper accepts several path styles used in training scripts:
- a local Hugging Face model directory
- a Hugging Face model ID (passed through unchanged)
- a non-local URI such as hdfs://... (passed through unchanged)
- a raw FSDP checkpoint component directory such as .../actor or .../critic
- a global_step_* checkpoint root containing actor/ and critic/
- a merged_hf container containing actor/ and critic/

For raw local FSDP checkpoints, the helper prefers an existing merged HF model
directory and otherwise runs ``python -m verl.model_merger merge`` once to
produce one.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

HF_WEIGHT_FILES = (
    "model.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
)
VALID_ROLES = ("actor", "critic")


@dataclass(frozen=True)
class ResolutionPlan:
    mode: str
    normalized_input: str
    resolved_path: str | None = None
    component_dir: str | None = None
    merged_dir: str | None = None
    reason: str = ""


def normalize_path(path: str) -> str:
    """Normalize local and URI-like paths while preserving roots."""
    if path is None:
        raise ValueError("Model init path cannot be None.")

    normalized = os.path.expanduser(path.strip())
    if not normalized:
        raise ValueError("Model init path cannot be empty.")

    if normalized == "/":
        return normalized

    scheme, sep, rest = normalized.partition("://")
    if sep:
        normalized_rest = rest.rstrip("/")
        return f"{scheme}://{normalized_rest}" if normalized_rest else normalized

    normalized = normalized.rstrip("/")
    return normalized or "/"


def is_uri(path: str) -> bool:
    return "://" in path


def has_hf_weights(path: Path) -> bool:
    if not path.is_dir():
        return False
    if not (path / "config.json").is_file():
        return False
    return any((path / name).is_file() for name in HF_WEIGHT_FILES)


def _make_merge_plan(raw_path: str, component_dir: Path, role: str, reason: str) -> ResolutionPlan:
    huggingface_dir = component_dir / "huggingface"
    merged_dir = component_dir.parent / "merged_hf" / role

    if has_hf_weights(huggingface_dir):
        return ResolutionPlan(
            mode="resolved",
            normalized_input=raw_path,
            resolved_path=str(huggingface_dir),
            reason=f"{reason}; using full weights from {huggingface_dir}",
        )

    if has_hf_weights(merged_dir):
        return ResolutionPlan(
            mode="resolved",
            normalized_input=raw_path,
            resolved_path=str(merged_dir),
            reason=f"{reason}; using existing merged weights from {merged_dir}",
        )

    if not (component_dir / "fsdp_config.json").is_file():
        raise ValueError(f"Unsupported raw checkpoint component without fsdp_config.json: {component_dir}")

    if not huggingface_dir.is_dir() or not (huggingface_dir / "config.json").is_file():
        raise ValueError(
            "Raw FSDP checkpoint is missing Hugging Face metadata needed for merging: "
            f"{huggingface_dir}. Expected at least config.json."
        )

    return ResolutionPlan(
        mode="merge",
        normalized_input=raw_path,
        component_dir=str(component_dir),
        merged_dir=str(merged_dir),
        reason=f"{reason}; merge raw FSDP checkpoint from {component_dir} to {merged_dir}",
    )


def plan_model_init_path(raw_path: str, role: str) -> ResolutionPlan:
    if role not in VALID_ROLES:
        raise ValueError(f"role must be one of {VALID_ROLES}, got {role!r}")

    normalized = normalize_path(raw_path)

    if is_uri(normalized):
        return ResolutionPlan(
            mode="resolved",
            normalized_input=normalized,
            resolved_path=normalized,
            reason="non-local URI path passed through unchanged",
        )

    path = Path(normalized)

    if not path.exists():
        parent_exists = path.parent != Path(".") and path.parent.exists()
        if path.is_absolute() or raw_path.startswith(".") or raw_path.startswith("~") or parent_exists:
            raise FileNotFoundError(f"Local model init path does not exist: {normalized}")
        return ResolutionPlan(
            mode="resolved",
            normalized_input=normalized,
            resolved_path=normalized,
            reason="non-local identifier or relative model name passed through unchanged",
        )

    if not path.is_dir():
        raise ValueError(
            f"Model init path must be a directory, URI, or Hugging Face model ID, but got file: {normalized}"
        )

    if has_hf_weights(path):
        return ResolutionPlan(
            mode="resolved",
            normalized_input=normalized,
            resolved_path=str(path),
            reason=f"found direct Hugging Face model directory at {path}",
        )

    role_child = path / role
    if has_hf_weights(role_child):
        return ResolutionPlan(
            mode="resolved",
            normalized_input=normalized,
            resolved_path=str(role_child),
            reason=f"found role-specific Hugging Face model directory at {role_child}",
        )

    merged_role = path / "merged_hf" / role
    if has_hf_weights(merged_role):
        return ResolutionPlan(
            mode="resolved",
            normalized_input=normalized,
            resolved_path=str(merged_role),
            reason=f"found existing merged_hf model directory at {merged_role}",
        )

    if path.name == "huggingface" and (path.parent / "fsdp_config.json").is_file():
        merge_role = path.parent.name if path.parent.name in VALID_ROLES else role
        return _make_merge_plan(
            normalized,
            path.parent,
            role=merge_role,
            reason="input is checkpoint metadata directory",
        )

    if (path / "fsdp_config.json").is_file():
        merge_role = path.name if path.name in VALID_ROLES else role
        return _make_merge_plan(normalized, path, role=merge_role, reason="input is raw FSDP component directory")

    if role_child.is_dir() and (role_child / "fsdp_config.json").is_file():
        return _make_merge_plan(normalized, role_child, role=role, reason="input is checkpoint root directory")

    if role_child.is_dir() and (role_child / "huggingface").is_dir():
        return _make_merge_plan(
            normalized,
            role_child,
            role=role,
            reason="input is checkpoint root directory with role metadata subdirectory",
        )

    raise ValueError(
        "Unsupported model init path layout. Expected one of: "
        "a Hugging Face model dir, a merged_hf container, a raw actor/critic FSDP checkpoint dir, "
        "or a global_step_* checkpoint root. Got "
        f"{normalized}"
    )


def ensure_model_init_path(
    raw_path: str,
    role: str,
    *,
    log_dir: str | None = None,
    python_executable: str | None = None,
) -> str:
    plan = plan_model_init_path(raw_path, role)

    if plan.mode == "resolved":
        return plan.resolved_path

    if plan.mode != "merge":
        raise RuntimeError(f"Unknown resolution mode: {plan.mode}")

    if not log_dir:
        raise ValueError("log_dir is required when resolving a raw FSDP checkpoint that needs merging.")

    python_executable = python_executable or sys.executable
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)
    log_path = log_dir_path / f"model_merge_{role}.log"

    command = [
        python_executable,
        "-m",
        "verl.model_merger",
        "merge",
        "--backend",
        "fsdp",
        "--local_dir",
        plan.component_dir,
        "--target_dir",
        plan.merged_dir,
        "--use_cpu_initialization",
    ]

    print(plan.reason, file=sys.stderr)
    print(f"Running: {' '.join(command)}", file=sys.stderr)
    print(f"Merge log: {log_path}", file=sys.stderr)

    with open(log_path, "w", encoding="utf-8") as log_file:
        subprocess.run(command, check=True, stdout=log_file, stderr=subprocess.STDOUT)

    merged_path = Path(plan.merged_dir)
    if not has_hf_weights(merged_path):
        raise RuntimeError(
            f"Merge command completed but did not produce a valid Hugging Face model directory at {merged_path}"
        )

    return str(merged_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", required=True, help="Input model/checkpoint path or model identifier.")
    parser.add_argument("--role", required=True, choices=VALID_ROLES, help="Whether to resolve actor or critic.")
    parser.add_argument("--log-dir", default=None, help="Directory for merger logs when raw FSDP checkpoints need merging.")
    parser.add_argument(
        "--python-exec",
        default=None,
        help="Python executable to use for merger subprocesses. Defaults to the current interpreter.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    resolved_path = ensure_model_init_path(
        args.path,
        args.role,
        log_dir=args.log_dir,
        python_executable=args.python_exec,
    )
    print(resolved_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
