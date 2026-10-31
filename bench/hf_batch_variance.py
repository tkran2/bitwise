"""Does a request's output depend on what else is in the batch?

Runs the same greedy requests (a) one at a time and (b) together in one padded batch,
through Hugging Face transformers (PyTorch CPU kernels) and through Bitwise in both
kernel modes, and reports how many requests change.

Usage: python bench/hf_batch_variance.py [--n 16] [--tokens 64]
Writes results/batch_variance.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import bitwise as bw
from prompts import load_prompts

ROOT = Path(__file__).resolve().parents[1]


def hf_greedy(model, tok, batch: list[list[int]], max_new: int):
    """Greedy decode with left padding; returns generated ids and first-step logits per row."""
    width = max(len(p) for p in batch)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ids = torch.tensor([[pad] * (width - len(p)) + p for p in batch])
    mask = torch.tensor([[0] * (width - len(p)) + [1] * len(p) for p in batch])
    with torch.no_grad():
        out = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=max_new, do_sample=False,
                             output_logits=True, return_dict_in_generate=True, pad_token_id=pad)
    gen = out.sequences[:, width:].tolist()
    first = out.logits[0].float().numpy()
    return gen, first


def trim(seq, eos):
    return seq[: seq.index(eos) + 1] if eos in seq else seq


def first_divergence(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--bw-model", default=str(ROOT / "models" / "qwen2.5-0.5b-instruct.bw"))
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--tokens", type=int, default=64)
    args = ap.parse_args()
    torch.set_num_threads(4)

    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = load_prompts(tok, args.n)
    report = {"model": args.model, "requests": args.n, "max_new_tokens": args.tokens}

    hf = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
    t0 = time.time()
    solo = [hf_greedy(hf, tok, [p], args.tokens) for p in prompts]
    batched_gen, batched_first = hf_greedy(hf, tok, prompts, args.tokens)
    eos = tok.eos_token_id
    changed, divergences, bit_diff_first = 0, [], 0
    for i, (gen, first) in enumerate(solo):
        a, b = trim(gen[0], eos), trim(batched_gen[i], eos)
        bit_diff_first += int(not np.array_equal(first[0], batched_first[i]))
        if a != b:
            changed += 1
            divergences.append(first_divergence(a, b))
    report["transformers"] = {
        "first_token_logits_differ": bit_diff_first,
        "outputs_changed": changed,
        "divergence_positions": divergences,
        "seconds": time.time() - t0,
    }
    print("transformers:", report["transformers"], flush=True)
    del hf

    model = bw.Model.load(args.bw_model)
    for mode_name, mode in (("bitwise_invariant", bw.KernelMode.Invariant),
                            ("bitwise_shape_specialized", bw.KernelMode.ShapeSpecialized)):
        cfg = bw.EngineConfig()
        cfg.kernels = mode
        cfg.prefix_caching = False
        params = bw.SamplingParams(max_new_tokens=args.tokens)
        alone = []
        for p in prompts:
            e = bw.Engine(model, None, cfg)
            rid = e.add_request(p, params)
            while not e.idle():
                e.step()
            alone.append(e.result(rid))
        e = bw.Engine(model, None, cfg)
        ids = [e.add_request(p, params) for p in prompts]
        while not e.idle():
            e.step()
        together = [e.result(i) for i in ids]
        report[mode_name] = {
            "logits_fingerprints_differ": sum(a.logits_fingerprint != b.logits_fingerprint for a, b in zip(alone, together)),
            "outputs_changed": sum(list(a.tokens) != list(b.tokens) for a, b in zip(alone, together)),
            "divergence_positions": [first_divergence(list(a.tokens), list(b.tokens))
                                     for a, b in zip(alone, together) if list(a.tokens) != list(b.tokens)],
        }
        print(mode_name, report[mode_name], flush=True)

    out = ROOT / "results" / "batch_variance.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
