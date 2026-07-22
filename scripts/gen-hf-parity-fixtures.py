#!/usr/bin/env python
"""HF transformers parity fixtures for the runnable lane (Phase 5 / Gate 5).

Reference: HuggingFace `transformers` loading the SAME GGUF the runnable lane runs
(`AutoModelForCausalLM.from_pretrained(gguf_file=...)`). transformers dequantizes the
Q8_0 weights to f32 — the same gguf dequant camelid is bit-exact against (Phase 2) —
and un-permutes the llama Q/K projections for its rotate-half RoPE. So HF runs the
identical numerical weights as the runnable lane: the logit difference is pure
graph-implementation fidelity (RoPE pairing, attention, norm order), not quant noise.

For each fixed prompt we record, under greedy (argmax) decoding with no chat template:
  - prompt token ids (from the GGUF tokenizer) — the runnable lane decodes the SAME ids
  - the greedy continuation token ids (the hard gate: exact-match)
  - HF's full first-step logits (f32 bit patterns) — for the reported max-abs-diff

Deterministic: fixed prompts, greedy, no sampling, CPU float32.

IMPORTANT: full python.exe path, never the `py` launcher. Memory-heavy (~4.4 GB f32
model) — run alone, it saves fixtures and exits before the Rust side runs.

Usage:  <python.exe> scripts/gen-hf-parity-fixtures.py
"""
import json
import os
import sys

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import importlib.metadata
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# transformers 5.12.1 mis-detects the installed gguf version as 'N/A' (its
# _is_package_available returns (True, 'N/A')), which crashes is_gguf_available's
# version parse. gguf 0.19.0 is in fact installed and importable; force the gate
# true in the module that both the model and tokenizer GGUF loaders go through.
import gguf  # noqa: F401  (confirm it really is importable)
import transformers.modeling_gguf_pytorch_utils as _mgpu

_mgpu.is_gguf_available = lambda *a, **k: True

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO, "tests", "fixtures", "hf_parity")
MODEL_DIR = os.path.join(REPO, "models")

# model key -> (gguf filename, output fixture filename)
TARGETS = {
    "tinyllama": ("tinyllama-1.1b-chat-v1.0.Q8_0.gguf", "tinyllama.json"),
    "qwen3": ("Qwen3-0.6B-Q8_0.gguf", "qwen3.json"),
    "gemma3": ("gemma-3-1b-it-Q8_0.gguf", "gemma3.json"),
    "phi3": ("Phi-3-mini-4k-instruct-Q8_0.gguf", "phi3.json"),
}
KEY = sys.argv[1] if len(sys.argv) > 1 else "tinyllama"
GGUF, OUT_NAME = TARGETS[KEY]

PROMPTS = [
    "The capital of France is",
    "Once upon a time",
    "The quick brown fox",
    "2 + 2 =",
]
MAX_NEW = 16

TVER = importlib.metadata.version("transformers")
PTVER = importlib.metadata.version("torch")


def f32_bits(arr):
    u = np.asarray(arr, dtype=np.float32).view(np.uint32)
    return [f"0x{int(v):08x}" for v in u]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(0)
    print(f"loading {GGUF} into HF transformers (f32, CPU)...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, gguf_file=GGUF)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, gguf_file=GGUF, torch_dtype=torch.float32, low_cpu_mem_usage=True
    )
    model.eval()
    print("loaded.", flush=True)

    fixtures = []
    with torch.no_grad():
        for text in PROMPTS:
            # GGUF tokenizer, no chat template, no extra specials beyond its default.
            enc = tok(text, return_tensors="pt", add_special_tokens=True)
            ids = enc.input_ids
            prompt_ids = ids[0].tolist()

            first_logits = None
            cur = ids
            greedy = []
            for step in range(MAX_NEW):
                out = model(cur)
                logits = out.logits[0, -1, :]
                if step == 0:
                    first_logits = logits.detach().cpu().numpy().astype(np.float32)
                nxt = int(torch.argmax(logits).item())
                greedy.append(nxt)
                cur = torch.cat([cur, torch.tensor([[nxt]])], dim=1)

            fixtures.append({
                "prompt_text": text,
                "prompt_ids": prompt_ids,
                "greedy_ids": greedy,
                "first_step_logits_bits": f32_bits(first_logits),
            })
            print(f"  {text!r}: prompt={len(prompt_ids)} tok, greedy={greedy}", flush=True)

    out = {
        "lane": "runnable",
        "reference": f"hf-transformers=={TVER} (torch=={PTVER})",
        "gguf": GGUF,
        "decode": "greedy/argmax, no chat template",
        "max_new": MAX_NEW,
        "fixtures": fixtures,
    }
    path = os.path.join(OUT_DIR, OUT_NAME)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(f"wrote {path} ({len(fixtures)} prompts)", flush=True)


if __name__ == "__main__":
    main()
