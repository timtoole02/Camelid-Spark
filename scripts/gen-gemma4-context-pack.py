#!/usr/bin/env python3
"""Generate a gemma4 bounded-context recall pack (camelid.gemma4.prompt-pack.v1).

Reproduces the committed 512/1024/2048 packs byte-exact (verified by --check)
and extrapolates new buckets. The reference_prompt_token_count field must be
filled from the pinned llama.cpp /tokenize result before committing.

Usage:
  gen-gemma4-context-pack.py --check                    # verify 512/1024/2048 reproduce
  gen-gemma4-context-pack.py --window 4096 --facts 333 --code-fact 200 \
      --ref-tokens <n>                                  # emit pack JSON to stdout
"""
import argparse
import json
import sys

SUBJECTS = [
    "llama", "vicuna", "guanaco", "camel", "dromedary",
    "tapir", "capybara", "ibex", "markhor", "saiga",
    "chamois", "serow", "goral", "takin", "argali",
    "urial", "mouflon", "addax", "oryx", "alpaca",
]
PREDICATES = [
    "records deterministic tokens", "checks bounded context",
    "preserves exact rows", "audits public summaries",
    "matches clean checkouts", "verifies wire checksums",
    "stores layer plans", "tracks greedy parity",
    "guards pinned oracles", "keeps blue evidence",
]


def prompt_text(window: int, n_facts: int, code_fact: int) -> str:
    parts = ["Context block start."]
    for n in range(1, n_facts + 1):
        if n == code_fact:
            parts.append(f"Fact {n:02d}: the audit code is CMLD-{window}.")
        else:
            subj = SUBJECTS[(n - 1) % len(SUBJECTS)]
            pred = PREDICATES[(n - 1) % len(PREDICATES)]
            parts.append(f"Fact {n:02d}: {subj} {pred}.")
    parts.append(
        f"Context block end. The audit code hidden in fact {code_fact} is the answer. "
        f"Question: respond with ONLY the audit code from fact {code_fact}, "
        f"formatted as CMLD-<number>. Answer: CMLD-"
    )
    return " ".join(parts)


def pack(window: int, n_facts: int, code_fact: int, ref_tokens: int) -> dict:
    depth_pct = code_fact / n_facts
    return {
        "schema": "camelid.gemma4.prompt-pack.v1",
        "pack_id": f"gemma4-context-{window}-v1",
        "description": (
            f"Bounded {window}-token context recall bucket: synthetic numbered facts "
            f"with one embedded audit code at ~60% depth; greedy completion must "
            f"recall it. Prompt sized by reference tokenization (BOS + plain text)."
        ),
        "decode": "greedy",
        "target_context_window": window,
        "reference_prompt_token_count": ref_tokens,
        "expected_code": f"CMLD-{window}",
        "prompts": [
            {
                "id": f"recall-{window}",
                "text": prompt_text(window, n_facts, code_fact),
                "max_new_tokens": 8,
            }
        ],
    }


def check() -> int:
    params = {512: (34, 20, 440), 1024: (79, 47, 953), 2048: (164, 98, 1980)}
    ok = True
    for window, (n_facts, code_fact, ref) in params.items():
        committed = json.load(open(f"qa/gemma4/prompt_packs/context_{window}_v1.json"))
        generated = pack(window, n_facts, code_fact, ref)
        if committed == generated:
            print(f"{window}: byte-exact reproduction OK")
        else:
            ok = False
            for key in committed:
                if committed[key] != generated.get(key):
                    print(f"{window}: MISMATCH in {key!r}")
                    if key == "prompts":
                        ct, gt = committed[key][0]["text"], generated[key][0]["text"]
                        for i, (a, b) in enumerate(zip(ct, gt)):
                            if a != b:
                                print(f"  first text diff at char {i}: {ct[i-40:i+40]!r} vs {gt[i-40:i+40]!r}")
                                break
                        print(f"  len committed={len(ct)} generated={len(gt)}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--window", type=int)
    ap.add_argument("--facts", type=int)
    ap.add_argument("--code-fact", type=int)
    ap.add_argument("--ref-tokens", type=int, default=0)
    ap.add_argument("--text-only", action="store_true",
                    help="print just the prompt text (for /tokenize sizing)")
    args = ap.parse_args()
    if args.check:
        return check()
    if not (args.window and args.facts and args.code_fact):
        ap.error("--window, --facts, --code-fact required (or --check)")
    if args.text_only:
        sys.stdout.write(prompt_text(args.window, args.facts, args.code_fact))
        return 0
    json.dump(pack(args.window, args.facts, args.code_fact, args.ref_tokens),
              sys.stdout, indent=1, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
