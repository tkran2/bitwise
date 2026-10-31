"""Checks on real checkpoints. Skipped unless the converted models exist:

    python tools/convert.py Qwen/Qwen2.5-0.5B-Instruct models/qwen2.5-0.5b-instruct.bw
    python tools/convert.py HuggingFaceTB/SmolLM2-135M-Instruct models/smollm2-135m-instruct.bw
    python tools/convert.py HuggingFaceTB/SmolLM2-360M-Instruct models/smollm2-360m-instruct.bw
"""

from pathlib import Path

import numpy as np
import pytest

import bitwise as bw

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
QWEN = ("Qwen/Qwen2.5-0.5B-Instruct", MODELS / "qwen2.5-0.5b-instruct.bw")
SMOL = ("HuggingFaceTB/SmolLM2-135M-Instruct", MODELS / "smollm2-135m-instruct.bw")
SMOL360 = ("HuggingFaceTB/SmolLM2-360M-Instruct", MODELS / "smollm2-360m-instruct.bw")

PROMPTS = [
    "Explain why the sky is blue in two sentences.",
    "Write a haiku about a lighthouse.",
    "What is 17 times 23? Show your work.",
    "List three uses of a paperclip.",
    "Translate 'good morning, friends' into French and Spanish.",
    "Give a one-paragraph summary of how a hash table works.",
    "Name a prime number larger than 100 and explain how you know it is prime.",
    "Describe the taste of a lemon to someone who has never had one.",
]


def need(path):
    if not path.exists():
        pytest.skip(f"{path.name} not converted")


def encode(tok, text, system=None):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": text}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True)
    return list(ids["input_ids"] if isinstance(ids, dict) or hasattr(ids, "keys") else ids)


@pytest.fixture(scope="module")
def torch_threads():
    torch = pytest.importorskip("torch")
    torch.set_num_threads(4)
    return torch


@pytest.mark.parametrize("spec", [QWEN, SMOL], ids=["qwen2", "llama"])
def test_logits_match_transformers(spec, torch_threads):
    torch = torch_threads
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name, path = spec
    need(path)
    tok = AutoTokenizer.from_pretrained(name)
    ours = bw.Model.load(str(path))
    ref = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
    worst, agree, total = 0.0, 0, 0
    for p in PROMPTS[:4]:
        ids = encode(tok, p)
        a = ours.logits(ids)
        with torch.no_grad():
            b = ref(torch.tensor([ids])).logits[0].numpy()
        worst = max(worst, float(np.abs(a - b).max()))
        agree += int((a.argmax(-1) == b.argmax(-1)).sum())
        total += len(ids)
    assert worst < 2e-3
    assert agree / total > 0.99


def run(engine_cfg, target, prompts, params, draft=None):
    eng = bw.Engine(target, draft, engine_cfg)
    ids = [eng.add_request(p, sp) for p, sp in zip(prompts, params)]
    while not eng.idle():
        eng.step()
    return [eng.result(i) for i in ids], eng.stats


def test_engine_outputs_independent_of_batching():
    """Real weights: crowded continuous batching with shared prefixes, chunked prefill,
    tight memory and mixed greedy/sampled requests gives each request the same tokens
    and the same logits bits as running it alone."""
    from transformers import AutoTokenizer

    name, path = QWEN
    need(path)
    tok = AutoTokenizer.from_pretrained(name)
    model = bw.Model.load(str(path))
    system = "You are a careful assistant. Answer briefly and precisely, and never make up facts."
    prompts = [encode(tok, p, system) for p in PROMPTS]
    params = [bw.SamplingParams(temperature=0.0 if i % 2 == 0 else 0.9, seed=1000 + i, max_new_tokens=24)
              for i in range(len(prompts))]
    solo_cfg = bw.EngineConfig()
    solo_cfg.prefix_caching = False
    solo = [run(solo_cfg, model, [p], [sp])[0][0] for p, sp in zip(prompts, params)]

    cfg = bw.EngineConfig()
    cfg.max_batched_tokens = 48
    cfg.num_blocks = 24  # 384 tokens of cache for eight ~60-token prompts: forces preemption
    crowd, stats = run(cfg, model, prompts, params)
    assert stats.preemptions > 0 and stats.prefix_hit_tokens > 0
    for a, b in zip(solo, crowd):
        assert list(a.tokens) == list(b.tokens)
        assert a.logits_fingerprint == b.logits_fingerprint


def test_greedy_speculative_decoding_is_bit_identical():
    from transformers import AutoTokenizer

    need(SMOL[1])
    need(SMOL360[1])
    tok = AutoTokenizer.from_pretrained(SMOL360[0])
    target = bw.Model.load(str(SMOL360[1]))
    draft = bw.Model.load(str(SMOL[1]))
    prompts = [encode(tok, p) for p in PROMPTS[:4]]
    params = [bw.SamplingParams(max_new_tokens=40) for _ in prompts]
    base, _ = run(bw.EngineConfig(), target, prompts, params)
    cfg = bw.EngineConfig()
    cfg.spec_tokens = 4
    spec, stats = run(cfg, target, prompts, params, draft)
    assert stats.spec_accepted > 0
    for a, b in zip(base, spec):
        assert list(a.tokens) == list(b.tokens)
        assert a.logits_fingerprint == b.logits_fingerprint
