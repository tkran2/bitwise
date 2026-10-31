"""Benchmark prompts: instructions from databricks-dolly-15k (CC BY-SA 3.0), chat-templated."""

from __future__ import annotations

import json
import random

from huggingface_hub import hf_hub_download


def _records():
    path = hf_hub_download("databricks/databricks-dolly-15k", "databricks-dolly-15k.jsonl", repo_type="dataset")
    with open(path) as f:
        return [json.loads(line) for line in f]


def encode(tok, text: str, system: str | None = None) -> list[int]:
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": text}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True)
    return list(ids["input_ids"] if hasattr(ids, "keys") else ids)


def load_prompts(tok, n: int, seed: int = 0, system: str | None = None, with_context: bool = False) -> list[list[int]]:
    """n chat-templated prompts. Without context they are short instructions (~20-60 tokens);
    with context, instructions carry a reference passage (~100-500 tokens)."""
    recs = [r for r in _records() if bool(r["context"]) == with_context and len(r["instruction"]) < 400]
    if with_context:
        recs = [r for r in recs if len(r["context"]) < 2000]
    rng = random.Random(seed)
    picked = rng.sample(recs, n)
    texts = [r["instruction"] + (("\n\n" + r["context"]) if with_context else "") for r in picked]
    return [encode(tok, t, system) for t in texts]
