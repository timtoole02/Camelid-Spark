#!/usr/bin/env python3
"""DiffusionGemma lane Phase 2 control: compare two reference checkpoint dumps
(produced by scripts/dg-encoder-dump.cpp at DIFFERENT thread counts) to
measure the pinned llama.cpp build's OWN cross-run determinism envelope for
this graph. Thread count changes accumulation order only — both runs are the
same implementation on the same machine — so this is the noise floor any
independent implementation's checkpoints should be judged against.

Usage: dg-encoder-ref-self-compare.py <ref_dir_a> <ref_dir_b> <out.json>
"""
import json
import struct
import sys


def load_manifest(d):
    out = {}
    for line in open(f"{d}/manifest.json"):
        line = line.strip()
        if not line.startswith("{"):
            continue
        e = json.loads(line)
        out[e["name"]] = e
    return out


def load_f32(d, e):
    raw = open(f"{d}/{e['file']}", "rb").read()
    return struct.unpack(f"<{len(raw) // 4}f", raw)


def load_i32(d, e):
    raw = open(f"{d}/{e['file']}", "rb").read()
    return struct.unpack(f"<{len(raw) // 4}i", raw)


def main(dir_a, dir_b, out_path):
    ma, mb = load_manifest(dir_a), load_manifest(dir_b)
    assert set(ma) == set(mb), "checkpoint sets differ"
    rows = []
    topk_set_flips = 0
    topk_positions = 0
    for name in sorted(ma):
        ea, eb = ma[name], mb[name]
        if ea["type"] == "i32":
            a, b = load_i32(dir_a, ea), load_i32(dir_b, eb)
            k = ea["ne"][0]
            n = ea["ne"][1]
            stride = k if len(a) == n * k else (len(a) - k) // (n - 1)
            for pos in range(n):
                sa = sorted(a[pos * stride : pos * stride + k])
                sb = sorted(b[pos * stride : pos * stride + k])
                topk_positions += 1
                if sa != sb:
                    topk_set_flips += 1
            continue
        a, b = load_f32(dir_a, ea), load_f32(dir_b, eb)
        assert len(a) == len(b), name
        max_abs = 0.0
        sum_abs = 0.0
        for x, y in zip(a, b):
            d = abs(x - y)
            sum_abs += d
            if d > max_abs:
                max_abs = d
        rows.append(
            {
                "name": name,
                "values": len(a),
                "max_abs": max_abs,
                "mean_abs": sum_abs / max(len(a), 1),
            }
        )
    report = {
        "control": "pinned llama.cpp vs ITSELF (same build, same machine, different n_threads)",
        "dirs": [dir_a, dir_b],
        "topk_positions": topk_positions,
        "topk_set_flips": topk_set_flips,
        "checkpoints": rows,
    }
    with open(out_path, "w") as f:
        json.dump(report, f, indent=1)
    worst = sorted(rows, key=lambda r: -r["max_abs"])[:8]
    print(f"topk set flips: {topk_set_flips}/{topk_positions}")
    for r in worst:
        print(f"  {r['name']:22s} max_abs {r['max_abs']:.3e} mean {r['mean_abs']:.3e}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
