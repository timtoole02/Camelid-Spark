#!/usr/bin/env python
"""Generate HF-tokenizers reference fixtures for the runnable lane (Phase 3 / Gate 3).

The reference is HuggingFace `tokenizers` loading each model's genuine `tokenizer.json`
from the Hub. For each covered tokenizer family we emit string -> id-sequence fixtures
that the Rust side must reproduce exactly from the SAME model's GGUF metadata.

Families anchored:
  SPM  -> TinyLlama/TinyLlama-1.1B-Chat-v1.0  (matches models/tinyllama-...Q8_0.gguf)
  BPE  -> Qwen/Qwen3-0.6B                      (matches models/Qwen3-0.6B-Q8_0.gguf)

Comparison contract: HF `encode(text, add_special_tokens=False)` vs camelid
`encode(text, add_special=false, parse_special=false)` — core tokenization only, no
BOS/EOS, no chat template (per spec: chat-template handling is out of scope here).

IMPORTANT: invoke via full python.exe path, never the `py` launcher (it fork-bombs
on this box). See BACKEND_ASKS / memory.

Usage:  <python.exe> scripts/gen-tokenizer-fixtures.py
"""
import json
import os

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import importlib.metadata
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO, "tests", "fixtures", "tokenizer_hf")

TOK_VERSION = importlib.metadata.version("tokenizers")

# (family, hf_repo, local gguf filename it must agree with)
TARGETS = [
    ("spm", "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "tinyllama-1.1b-chat-v1.0.Q8_0.gguf"),
    ("bpe", "Qwen/Qwen3-0.6B", "Qwen3-0.6B-Q8_0.gguf"),
]

# Diverse corpus. Deliberately exercises the footguns: leading/internal/trailing
# whitespace, digit runs (llama3 groups 3 vs qwen2 single), multi-byte UTF-8
# (accents, CJK, emoji), byte-fallback territory, code, punctuation, mixed scripts.
CORPUS = [
    "Hello world",
    "Hello, world!",
    " leading space",
    "trailing space ",
    "double  space\tand\ttabs",
    "newline\nhere\nand\nhere",
    "Numbers: 0 1 12 123 1234 12345 007 3.14159",
    "MixedCase camelCase snake_case SCREAMING_CASE",
    "Punctuation?!... (parentheses) [brackets] {braces} <angle>",
    "Café déjà vu naïve résumé Zürich",
    "Цена в рублях — 1000₽",
    "日本語のテキストです。",
    "中文测试：你好世界",
    "한국어 텍스트입니다",
    "Emoji: 😀 🚀 👨‍👩‍👧‍👦 🇺🇸",
    "Math: ∑ ∫ √2 ≈ 1.41421 π≈3.14159",
    "def f(x):\n    return x ** 2 + 1  # comment",
    "https://example.com/path?q=1&r=2#frag",
    "email@example.com and user.name+tag@domain.co.uk",
    "Tab\tseparated\tvalues\there",
    "Repeated!!! characters??? ...dots and ---dashes",
    "A very long run of the same word word word word word word word",
    "Mixed: ASCII + 日本 + 123 + 😀 + ₽",
    "  many   spaces   between   words  ",
    "Quotes: \"double\" 'single' `backtick` «guillemets»",
    "",
    " ",
    "\n",
    "a",
    "🚀",
]


def gen(family, repo, gguf_name):
    path = hf_hub_download(repo, "tokenizer.json")
    tk = Tokenizer.from_file(path)
    corpus = []
    for text in CORPUS:
        ids = tk.encode(text, add_special_tokens=False).ids
        corpus.append({"text": text, "ids": ids})
    fixture = {
        "family": family,
        "hf_repo": repo,
        "reference": f"tokenizers=={TOK_VERSION} (HF tokenizer.json)",
        "vocab_size": tk.get_vocab_size(),
        "gguf": gguf_name,
        "add_special_tokens": False,
        "corpus": corpus,
    }
    out = os.path.join(OUT_DIR, f"{family}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(fixture, fh, ensure_ascii=False, indent=1)
    print(f"  wrote {family}: repo={repo} n={len(corpus)} vocab={tk.get_vocab_size()}")
    return fixture


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"generating tokenizer fixtures -> {OUT_DIR}  (tokenizers=={TOK_VERSION})")
    manifest = [gen(*t) for t in TARGETS]
    idx = {
        "reference": f"tokenizers=={TOK_VERSION}",
        "fixtures": [
            {k: m[k] for k in ("family", "hf_repo", "gguf", "vocab_size")}
            for m in manifest
        ],
    }
    with open(os.path.join(OUT_DIR, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(idx, fh, ensure_ascii=False, indent=1)
    print(f"done: {len(manifest)} fixtures + manifest.json")


if __name__ == "__main__":
    main()
