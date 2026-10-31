"""Serving benchmarks for Bitwise on CPU.

Experiments (each writes results/<name>.json):
  scheduling  sequential vs static vs continuous batching under Poisson arrivals
  prefix      a long shared system prompt with and without prefix caching
  spec        single-stream greedy decoding with speculative decoding, k = 0..6
  kernels     invariant vs shape-specialized kernels
  hf          transformers generate() at batch size 1, for reference

Usage: python bench/loadgen.py scheduling prefix spec kernels hf
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np

import bitwise as bw
from prompts import load_prompts

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
OUT = ROOT / "results"

SYSTEM = (
    "You are a support assistant for a company that sells outdoor equipment. Follow these rules. "
    "Answer in at most three sentences. Be polite and direct. If the question is about an order, ask "
    "for the order number before anything else. If the question is about returns, explain that unused "
    "items can be returned within 30 days with a receipt, that used items can be exchanged within 14 "
    "days for store credit, and that shipping costs for returns are covered only when the item was "
    "defective. If the question is about warranties, explain that tents, packs and sleeping bags carry "
    "a two-year warranty against manufacturing defects, that footwear carries a one-year warranty, and "
    "that the warranty does not cover normal wear, misuse or cosmetic damage. If the question is about "
    "product recommendations, ask about the customer's budget, the season and the kind of trip before "
    "recommending anything, and never recommend a product you have not been told about. If the "
    "customer is upset, acknowledge the problem first. Never promise refunds, discounts or delivery "
    "dates. Never share internal policies beyond what is written here. If you do not know the answer, "
    "say so and offer to connect the customer with a human agent during business hours, which are "
    "8am to 6pm Mountain Time, Monday through Saturday."
)


def percentile(xs, q):
    return float(np.percentile(xs, q)) if xs else float("nan")


def drive(engine: bw.Engine, prompts, params, arrivals):
    """Feed requests at their arrival times (seconds after start) and step until done."""
    order = sorted(range(len(prompts)), key=lambda i: arrivals[i])
    ids = [None] * len(prompts)
    nxt = 0
    t_start = engine.now()
    while nxt < len(order) or not engine.idle():
        now = engine.now() - t_start
        while nxt < len(order) and arrivals[order[nxt]] <= now:
            i = order[nxt]
            ids[i] = engine.add_request(prompts[i], params[i])
            nxt += 1
        if engine.idle():
            if nxt < len(order):
                time.sleep(max(0.0, arrivals[order[nxt]] - (engine.now() - t_start)))
            continue
        engine.step()
    makespan = engine.now() - t_start
    results = [engine.result(i) for i in ids]
    ttft = [r.first_token - r.arrival for r in results]
    e2e = [r.finish - r.arrival for r in results]
    tpot = [(r.finish - r.first_token) / (len(r.tokens) - 1) for r in results if len(r.tokens) > 1]
    gen = sum(len(r.tokens) for r in results)
    st = engine.stats
    return {
        "requests": len(prompts),
        "generated_tokens": gen,
        "makespan_s": makespan,
        "throughput_tok_s": gen / makespan,
        "ttft_p50_s": percentile(ttft, 50),
        "ttft_p99_s": percentile(ttft, 99),
        "tpot_p50_ms": 1000 * percentile(tpot, 50),
        "tpot_p99_ms": 1000 * percentile(tpot, 99),
        "e2e_p50_s": percentile(e2e, 50),
        "e2e_p99_s": percentile(e2e, 99),
        "steps": st.steps,
        "preemptions": st.preemptions,
        "prefix_hit_tokens": st.prefix_hit_tokens,
        "spec_proposed": st.spec_proposed,
        "spec_accepted": st.spec_accepted,
        "fingerprints": [r.logits_fingerprint for r in results],
    }


def poisson(n, rate, seed=0):
    if math.isinf(rate):
        return [0.0] * n
    rng = random.Random(seed)
    t, out = 0.0, []
    for _ in range(n):
        t += rng.expovariate(rate)
        out.append(t)
    return out


def tokenizer(name):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(name)


def save(name, obj):
    OUT.mkdir(exist_ok=True)
    for row in obj.get("runs", []):
        row.pop("fingerprints", None)
    (OUT / f"{name}.json").write_text(json.dumps(obj, indent=2))


def exp_scheduling(args):
    tok = tokenizer("Qwen/Qwen2.5-0.5B-Instruct")
    model = bw.Model.load(str(MODELS / "qwen2.5-0.5b-instruct.bw"))
    n = args.requests
    prompts = load_prompts(tok, n, seed=1)
    rng = random.Random(2)
    params = [bw.SamplingParams(max_new_tokens=rng.choice([32, 64, 96, 128]), ignore_eos=True) for _ in range(n)]
    runs = []
    for rate in (0.5, 1.0, 2.0, math.inf):
        arrivals = poisson(n, rate, seed=3)
        for policy in ("sequential", "static", "continuous"):
            cfg = bw.EngineConfig()
            cfg.num_blocks = 1024
            cfg.max_batched_tokens = 512
            cfg.max_num_seqs = 1 if policy == "sequential" else 16
            cfg.policy = bw.BatchPolicy.Static if policy == "static" else bw.BatchPolicy.Continuous
            cfg.fingerprint_logits = False
            r = drive(bw.Engine(model, None, cfg), prompts, params, arrivals)
            r.update(policy=policy, arrival_rate=rate if math.isfinite(rate) else "burst")
            runs.append(r)
            print(f"rate={rate:<4} {policy:10s} {r['throughput_tok_s']:7.1f} tok/s  ttft p50 {r['ttft_p50_s']:6.2f}s "
                  f"p99 {r['ttft_p99_s']:6.2f}s  tpot p50 {r['tpot_p50_ms']:6.1f}ms  e2e p99 {r['e2e_p99_s']:6.1f}s", flush=True)
    save("scheduling", {"model": "Qwen2.5-0.5B-Instruct", "threads": 4, "runs": runs})


def exp_prefix(args):
    tok = tokenizer("Qwen/Qwen2.5-0.5B-Instruct")
    model = bw.Model.load(str(MODELS / "qwen2.5-0.5b-instruct.bw"))
    n = args.requests
    prompts = load_prompts(tok, n, seed=4, system=SYSTEM)
    params = [bw.SamplingParams(max_new_tokens=32, ignore_eos=True) for _ in range(n)]
    arrivals = poisson(n, 1.0, seed=5)
    runs = []
    for prefix in (False, True):
        cfg = bw.EngineConfig()
        cfg.num_blocks = 2048
        cfg.max_batched_tokens = 512
        cfg.max_num_seqs = 16
        cfg.prefix_caching = prefix
        r = drive(bw.Engine(model, None, cfg), prompts, params, arrivals)
        r.update(prefix_caching=prefix, mean_prompt_tokens=float(np.mean([len(p) for p in prompts])))
        runs.append(r)
        print(f"prefix_caching={prefix!s:5s} ttft p50 {r['ttft_p50_s']:.3f}s p99 {r['ttft_p99_s']:.3f}s  "
              f"throughput {r['throughput_tok_s']:.1f} tok/s  hit tokens {r['prefix_hit_tokens']}", flush=True)
    same = runs[0]["fingerprints"] == runs[1]["fingerprints"]
    print("outputs bit-identical with and without prefix caching:", same)
    save("prefix", {"model": "Qwen2.5-0.5B-Instruct", "bit_identical": same, "runs": runs})


def exp_spec(args):
    out = {"runs": []}
    for target_name, target_file in (("SmolLM2-360M-Instruct", "smollm2-360m-instruct.bw"),
                                     ("SmolLM2-1.7B-Instruct", "smollm2-1.7b-instruct.bw")):
        path = MODELS / target_file
        if not path.exists():
            continue
        tok = tokenizer("HuggingFaceTB/" + target_name)
        target = bw.Model.load(str(path))
        draft = bw.Model.load(str(MODELS / "smollm2-135m-instruct.bw"))
        prompts = load_prompts(tok, args.spec_prompts, seed=6)
        base_fp = None
        for k in (0, 2, 4, 6):
            cfg = bw.EngineConfig()
            cfg.spec_tokens = k
            cfg.max_num_seqs = 1
            gen = 0
            fps = []
            t0 = time.time()
            accepted = proposed = 0
            for p in prompts:  # one stream at a time: the latency-bound case speculation targets
                e = bw.Engine(target, draft if k else None, cfg)
                rid = e.add_request(p, bw.SamplingParams(max_new_tokens=128))
                while not e.idle():
                    e.step()
                gen += len(e.result(rid).tokens)
                fps.append(e.result(rid).logits_fingerprint)
                accepted += e.stats.spec_accepted
                proposed += e.stats.spec_proposed
            dt = time.time() - t0
            if k == 0:
                base_fp = fps
            row = {"target": target_name, "draft": "SmolLM2-135M-Instruct", "k": k, "tokens": gen,
                   "tok_s": gen / dt, "acceptance": accepted / proposed if proposed else None,
                   "bit_identical_to_k0": fps == base_fp}
            out["runs"].append(row)
            print(row, flush=True)
    save("spec", out)


def exp_kernels(args):
    tok = tokenizer("Qwen/Qwen2.5-0.5B-Instruct")
    model = bw.Model.load(str(MODELS / "qwen2.5-0.5b-instruct.bw"))
    runs = []
    for batch in (1, 8):
        prompts = load_prompts(tok, batch, seed=7)
        params = [bw.SamplingParams(max_new_tokens=96, ignore_eos=True) for _ in prompts]
        for mode_name, mode in (("invariant", bw.KernelMode.Invariant), ("shape_specialized", bw.KernelMode.ShapeSpecialized)):
            best = None
            for _ in range(3):
                cfg = bw.EngineConfig()
                cfg.kernels = mode
                cfg.fingerprint_logits = False
                r = drive(bw.Engine(model, None, cfg), prompts, params, [0.0] * batch)
                best = r if best is None or r["throughput_tok_s"] > best["throughput_tok_s"] else best
            best.update(kernels=mode_name, batch=batch)
            runs.append(best)
            print(f"batch={batch} {mode_name:18s} {best['throughput_tok_s']:7.1f} tok/s  tpot p50 {best['tpot_p50_ms']:.1f}ms", flush=True)
    save("kernels", {"model": "Qwen2.5-0.5B-Instruct", "runs": runs})


def exp_hf(args):
    import torch
    from transformers import AutoModelForCausalLM

    torch.set_num_threads(4)
    name = "Qwen/Qwen2.5-0.5B-Instruct"
    tok = tokenizer(name)
    prompts = load_prompts(tok, 4, seed=8)
    hf = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
    gen, t0 = 0, time.time()
    for p in prompts:
        with torch.no_grad():
            out = hf.generate(torch.tensor([p]), max_new_tokens=96, min_new_tokens=96, do_sample=False)
        gen += out.shape[1] - len(p)
    hf_rate = gen / (time.time() - t0)
    model = bw.Model.load(str(MODELS / "qwen2.5-0.5b-instruct.bw"))
    cfg = bw.EngineConfig()
    cfg.fingerprint_logits = False
    gen, t0 = 0, time.time()
    for p in prompts:
        e = bw.Engine(model, None, cfg)
        rid = e.add_request(p, bw.SamplingParams(max_new_tokens=96, ignore_eos=True))
        while not e.idle():
            e.step()
        gen += len(e.result(rid).tokens)
    bw_rate = gen / (time.time() - t0)
    row = {"model": name, "batch": 1, "transformers_fp32_tok_s": hf_rate, "bitwise_tok_s": bw_rate}
    print(row)
    save("hf_baseline", row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiments", nargs="+", choices=["scheduling", "prefix", "spec", "kernels", "hf"])
    ap.add_argument("--requests", type=int, default=48)
    ap.add_argument("--spec-prompts", type=int, default=6)
    args = ap.parse_args()
    for name in args.experiments:
        globals()[f"exp_{name}"](args)


if __name__ == "__main__":
    main()
