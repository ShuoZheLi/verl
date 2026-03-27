from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from verl.utils.reward_score import default_compute_score


@dataclass(frozen=True)
class ExampleRecord:
    example_id: int
    prompt_text: str
    data_source: str
    ground_truth: Any
    prompt_token_ids: tuple[int, ...] | None = None


def _is_missing(value: Any) -> bool:
    try:
        result = pd.isna(value)
    except Exception:
        return False

    if isinstance(result, (bool, np.bool_)):
        return bool(result)
    return False


def _stringify_chat_messages(messages: list[Any]) -> str:
    parts: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            parts.append(str(message))
            continue
        role = message.get("role", "user")
        content = message.get("content", "")
        parts.append(f"{role}: {content}")
    return "\n".join(parts)


def _coerce_prompt(prompt: Any) -> Any:
    if isinstance(prompt, np.ndarray):
        return prompt.tolist()
    return prompt


def normalize_prompt(prompt: Any, tokenizer) -> str:
    prompt = _coerce_prompt(prompt)

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
        if all(isinstance(item, dict) for item in prompt):
            if hasattr(tokenizer, "apply_chat_template"):
                try:
                    return tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
                except Exception:
                    pass
            return _stringify_chat_messages(prompt)
        if all(isinstance(item, str) for item in prompt):
            return "\n".join(prompt)
        return "\n".join(str(item) for item in prompt)

    return str(prompt)


def extract_ground_truth(row: pd.Series, response_key: str | None) -> Any:
    if response_key and response_key in row and not _is_missing(row[response_key]):
        return row[response_key]

    reward_model = row.get("reward_model")
    if isinstance(reward_model, dict):
        return reward_model.get("ground_truth")

    return None


def load_examples(
    dataset_path: str | Path,
    *,
    tokenizer,
    prompt_key: str = "prompt",
    response_key: str | None = None,
    start_index: int = 0,
    max_examples: int | None = None,
    shuffle_examples: bool = False,
    seed: int = 0,
    pretokenize_max_length: int | None = None,
) -> list[ExampleRecord]:
    dataframe = pd.read_parquet(Path(dataset_path))
    indices = list(range(len(dataframe)))

    if start_index:
        indices = indices[start_index:]

    if shuffle_examples:
        random.Random(seed).shuffle(indices)

    if max_examples is not None:
        indices = indices[:max_examples]

    examples: list[ExampleRecord] = []
    for example_id in indices:
        row = dataframe.iloc[example_id]
        prompt_text = normalize_prompt(row[prompt_key], tokenizer)
        prompt_token_ids = None
        if pretokenize_max_length is not None:
            tokenized = tokenizer(
                prompt_text,
                truncation=True,
                max_length=pretokenize_max_length,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            prompt_token_ids = tuple(int(token_id) for token_id in tokenized["input_ids"])
        data_source = row.get("data_source", "")
        data_source = "" if _is_missing(data_source) else str(data_source)
        ground_truth = extract_ground_truth(row, response_key=response_key)
        examples.append(
            ExampleRecord(
                example_id=int(example_id),
                prompt_text=prompt_text,
                data_source=data_source,
                ground_truth=ground_truth,
                prompt_token_ids=prompt_token_ids,
            )
        )
    return examples


def score_response(example: ExampleRecord, response_text: str) -> float:
    score = default_compute_score(
        data_source=example.data_source,
        solution_str=response_text,
        ground_truth=example.ground_truth,
    )
    if isinstance(score, dict):
        for key in ("score", "reward", "accuracy", "acc"):
            if key in score:
                return float(score[key])
        raise ValueError(f"Cannot scalarize score dictionary: {score}")
    return float(score)
