# Benchmarks

Everything here runs on CPU against converted checkpoints in `models/`
(`python tools/convert.py <hf-model> models/<name>.bw`). Results land in `results/`.

| Command | What it measures | Output |
| --- | --- | --- |
| `python loadgen.py scheduling` | one-at-a-time vs static vs continuous batching under Poisson arrivals (Qwen2.5-0.5B, 48 requests) | `results/scheduling.json` |
| `python loadgen.py prefix` | time to first token with and without prefix caching behind a shared 254-token system prompt | `results/prefix.json` |
| `python loadgen.py spec` | speculative decoding with SmolLM2-135M drafting for 360M and 1.7B targets, k = 0, 2, 4, 6 | `results/spec.json` |
| `python loadgen.py kernels` | throughput of invariant vs shape-specialized kernels at batch 1 and 8 | `results/kernels.json` |
| `python loadgen.py hf` | single-stream decode vs Hugging Face transformers | `results/hf_baseline.json` |
| `python hf_batch_variance.py` | whether a request's logits change when it is batched, for transformers and both Bitwise kernel modes | `results/batch_variance.json` |
| `python plot.py` | the scheduling figure in the main README | `figures/scheduling.png` |

Run from this folder with `PYTHONPATH=../python`. Prompts come from databricks-dolly-15k
(`prompts.py`). Every engine run also records a logits fingerprint per request, which is
how the prefix-caching and speculative-decoding runs confirm their outputs are
bit-identical to the baseline.
