import fs from 'node:fs';

// Compute B = bytes read per decode step, exactly, from a camelid `inspect` JSON dump.
// Dense Llama with tied embeddings => every weight tensor is read once per decode step.
// token_embd.weight is the tied LM head (read FULLY as the output projection each step).

function analyze(path, label) {
  const j = JSON.parse(fs.readFileSync(path, 'utf8'));
  const tensors = j.tensors;
  const md = j.metadata;
  const hasSeparateOutput = tensors.some(t => t.name === 'output.weight');

  // Categorize each tensor by role in one decode step at ctx~0.
  const cat = {
    lm_head: 0n,          // token_embd (tied) OR output.weight — full read as output projection
    embd_lookup: 0n,      // only 1 row read if untied; ~negligible (not added to B)
    attn: 0n,             // q,k,v,output projections
    ffn: 0n,              // gate, up, down
    norms: 0n,            // attn_norm, ffn_norm, output_norm (F32, tiny)
    rope: 0n,             // rope_freqs (tiny table)
    excluded: 0n,         // tensors NOT read per decode step
  };
  const excludedNames = [];

  for (const t of tensors) {
    const n = BigInt(t.n_bytes);
    const name = t.name;
    if (name === 'token_embd.weight') {
      if (hasSeparateOutput) {
        // untied: embedding lookup reads ~1 row only; effectively excluded from B
        cat.embd_lookup += n; // tracked but NOT added to B
      } else {
        cat.lm_head += n;     // tied: full head read
      }
    } else if (name === 'output.weight') {
      cat.lm_head += n;
    } else if (/\.attn_(q|k|v|output)\.weight$/.test(name)) {
      cat.attn += n;
    } else if (/\.ffn_(gate|up|down)\.weight$/.test(name)) {
      cat.ffn += n;
    } else if (/(attn_norm|ffn_norm|output_norm)\.weight$/.test(name)) {
      cat.norms += n;
    } else if (name === 'rope_freqs.weight') {
      cat.rope += n;
    } else {
      cat.excluded += n;
      excludedNames.push(`${name} (${t.tensor_type}, ${t.n_bytes}B)`);
    }
  }

  // B = everything read per decode step. embd_lookup (untied) is ~1 row, excluded.
  const B = cat.lm_head + cat.attn + cat.ffn + cat.norms + cat.rope;
  const GB = Number(B) / 1e9;
  const GiB = Number(B) / (1024 ** 3);

  console.log(`\n===== ${label} =====`);
  console.log(`arch=${md['general.architecture']} file_type=${md['general.file_type']} ` +
    `layers=${md['llama.block_count']} d_model=${md['llama.embedding_length']} ` +
    `ffn=${md['llama.feed_forward_length']} vocab=${md['llama.vocab_size']} ` +
    `n_head=${md['llama.attention.head_count']} n_kv=${md['llama.attention.head_count_kv']}`);
  console.log(`tied_embeddings=${!hasSeparateOutput}`);
  const pct = v => (Number(v) / Number(B) * 100).toFixed(1);
  console.log(`  lm_head (tied token_embd): ${(Number(cat.lm_head)/1e6).toFixed(2)} MB  (${pct(cat.lm_head)}%)`);
  console.log(`  attn q/k/v/o           :   ${(Number(cat.attn)/1e6).toFixed(2)} MB  (${pct(cat.attn)}%)`);
  console.log(`  ffn gate/up/down       :   ${(Number(cat.ffn)/1e6).toFixed(2)} MB  (${pct(cat.ffn)}%)`);
  console.log(`  norms (F32)            :   ${(Number(cat.norms)/1e3).toFixed(2)} KB  (${pct(cat.norms)}%)`);
  console.log(`  rope_freqs             :   ${Number(cat.rope)} B`);
  if (excludedNames.length) console.log(`  EXCLUDED (not per-step): ${excludedNames.join(', ')}`);
  console.log(`  ----`);
  console.log(`  B (weights/decode step): ${B} bytes = ${GB.toFixed(4)} GB = ${GiB.toFixed(4)} GiB`);
  return { label, B: B.toString(), GB, GiB, breakdown: {
    lm_head: Number(cat.lm_head), attn: Number(cat.attn), ffn: Number(cat.ffn),
    norms: Number(cat.norms), rope: Number(cat.rope) }, tied: !hasSeparateOutput };
}

const results = [
  analyze(process.argv[2], 'P1  Llama-3.2-1B Q8_0'),
  analyze(process.argv[3], 'P2  Llama-3.2-3B Q8_0'),
];
fs.writeFileSync(process.argv[4], JSON.stringify(results, null, 2));
console.log(`\nWrote ${process.argv[4]}`);
