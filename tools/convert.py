"""Convert a Hugging Face Llama or Qwen2 checkpoint (safetensors) into a .bw file.

The .bw format is a small text header followed by raw tensors:

    BITWISE 1
    arch qwen2
    vocab_size 151936
    ...
    tensor model.embed_tokens.weight bf16 0 2 151936 896
    ...
    end

Tensor offsets are relative to the data region, which starts at the first multiple of
4096 bytes after the header. BF16 and F32 tensors are copied byte for byte; F16 is
widened to F32. No PyTorch needed.

Usage: python tools/convert.py Qwen/Qwen2.5-0.5B-Instruct models/qwen2.5-0.5b.bw
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

ARCHS = {"LlamaForCausalLM": "llama", "Qwen2ForCausalLM": "qwen2"}


def resolve(model: str) -> Path:
    p = Path(model)
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model, allow_patterns=["*.json", "*.safetensors"]))


def read_safetensors(path: Path):
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
        base = 8 + n
    header.pop("__metadata__", None)
    data = np.memmap(path, dtype=np.uint8, mode="r")
    for name, info in header.items():
        start, end = info["data_offsets"]
        yield name, info["dtype"], info["shape"], data[base + start: base + end]


def convert(model: str, out: str) -> None:
    src = resolve(model)
    cfg = json.loads((src / "config.json").read_text())
    arch = ARCHS.get(cfg["architectures"][0])
    if arch is None:
        raise SystemExit(f"unsupported architecture {cfg['architectures'][0]}")
    if cfg.get("rope_scaling"):
        raise SystemExit("rope scaling is not supported")
    eos = cfg.get("eos_token_id", -1)
    if isinstance(eos, list):
        eos = eos[0]
    heads = cfg["num_attention_heads"]
    meta = {
        "arch": arch,
        "vocab_size": cfg["vocab_size"],
        "hidden": cfg["hidden_size"],
        "layers": cfg["num_hidden_layers"],
        "heads": heads,
        "kv_heads": cfg.get("num_key_value_heads", heads),
        "head_dim": cfg.get("head_dim") or cfg["hidden_size"] // heads,
        "intermediate": cfg["intermediate_size"],
        "rope_theta": cfg.get("rope_theta", 10000.0),
        "rms_eps": cfg.get("rms_norm_eps", 1e-6),
        "qkv_bias": int(arch == "qwen2"),
        "tie_embeddings": int(cfg.get("tie_word_embeddings", False)),
        "bos_id": cfg.get("bos_token_id") if cfg.get("bos_token_id") is not None else -1,
        "eos_id": eos if eos is not None else -1,
    }

    tensors = []
    for f in sorted(src.glob("*.safetensors")):
        for name, dtype, shape, raw in read_safetensors(f):
            if dtype == "BF16":
                tensors.append((name, "bf16", shape, raw))
            elif dtype == "F32":
                tensors.append((name, "f32", shape, raw))
            elif dtype == "F16":
                tensors.append((name, "f32", shape, np.frombuffer(raw.tobytes(), np.float16).astype(np.float32).view(np.uint8)))
            else:
                raise SystemExit(f"unsupported dtype {dtype} for {name}")

    lines = ["BITWISE 1"] + [f"{k} {v}" for k, v in meta.items()]
    offset = 0
    placed = []
    for name, dt, shape, raw in tensors:
        offset = (offset + 63) // 64 * 64
        lines.append(f"tensor {name} {dt} {offset} {len(shape)} " + " ".join(str(d) for d in shape))
        placed.append((offset, raw))
        offset += raw.nbytes
    header = ("\n".join(lines) + "\nend\n").encode()
    data_start = (len(header) + 4095) // 4096 * 4096

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        f.write(header)
        f.write(b"\0" * (data_start - len(header)))
        for off, raw in placed:
            f.seek(data_start + off)
            f.write(raw.tobytes())
    print(f"wrote {out}: {len(tensors)} tensors, {(data_start + offset) / 1e6:.1f} MB, arch={arch}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    args = ap.parse_args()
    convert(args.model, args.out)
