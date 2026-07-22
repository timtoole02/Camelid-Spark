//! Experimental DiffusionGemma runtime (recon/evidence lane only).
//!
//! Implements the ENCODER side of the block-diffusion model (a causal
//! prompt-prefill forward over the shared Gemma-4 backbone with the
//! encoder-mode per-layer output scalars — Phase 2) and the Phase 3 decode
//! surface (`unified_forward`: one zero-self-conditioning bidirectional
//! forward over `[prompt | canvas]` through the tied lm_head, plus
//! `eb_step`/`refrng`: one Entropy-Bound sampler step with the reference's
//! exact host RNG). See
//! `docs/recon/DIFFUSIONGEMMA_RECON.md` for the verified architecture facts
//! and the lane charter rules. Nothing here is a support claim; the public
//! posture stays "Active development — recon/evidence-only. Not supported."
//!
//! Correctness-first by construction:
//! - Weights are read lazily from the GGUF mmap and dotted with quantized
//!   activations mirroring the reference CPU kernels (Q8_0 activations for
//!   Q8_0/Q5_0 rows, Q8_K activations for Q4_K/Q6_K rows) — the same
//!   discipline that earned the gemma4 rows their parity.
//! - The forward emits full per-layer checkpoints (K/V, post-attention
//!   residual, MoE router logits, selected experts, scaled layer output) so
//!   the Phase 2 gate can compare layer-by-layer against the pinned
//!   llama.cpp reference (`scripts/dg-encoder-dump.cpp`).
//! - Anything the recon has not proven fails closed with a typed error.

use std::path::Path;

pub mod chat;
#[cfg(feature = "cuda")]
mod cuda;
mod reff16;
pub(crate) mod refmath;
pub mod refrng;

/// GPU soft-embedding matmul (`emb_t @ probs`) for the self-conditioning
/// signal — returns the `[c*hidden]` f32 buffer, or `None` to fall back to the
/// CPU path. A no-op (always `None`) when the `cuda` feature is off.
#[cfg(feature = "cuda")]
fn sc_soft_gpu(
    emb_t: &[u16],
    probs_f16: &[u16],
    c: usize,
    hidden: usize,
    n_vocab: usize,
    embed_scale: f32,
) -> Option<Vec<f32>> {
    cuda::sc_soft_embedding_gpu(emb_t, probs_f16, c, hidden, n_vocab, embed_scale)
}
#[cfg(not(feature = "cuda"))]
fn sc_soft_gpu(
    _emb_t: &[u16],
    _probs_f16: &[u16],
    _c: usize,
    _hidden: usize,
    _n_vocab: usize,
    _embed_scale: f32,
) -> Option<Vec<f32>> {
    None
}

/// GPU Q6_K lm_head over the canvas activations — returns `[c*n_vocab]` logits
/// (row-major per position) or `None` to fall back to the CPU matvec. A no-op
/// when the `cuda` feature is off. The Q6_K GEMV mirrors the CPU `q6_k_dot`
/// reduction (exact i64 dot + fused per-block f32 term), so it is bit-close /
/// bit-identical to the CPU path.
#[cfg(feature = "cuda")]
fn lm_head_gpu(wire: &DgWire, acts: &[DgActivation], c: usize, hidden: usize) -> Option<Vec<f32>> {
    // Opt-in: the GPU Q6_K GEMV is bit-identical to the CPU lm_head but the
    // current naive one-thread-per-output kernel is SLOWER than CPU in isolation
    // (lm_head is only ~9s on CPU). Kept as the validated building block for the
    // full-forward GPU path (where it avoids the per-stage round-trip); enable
    // with CAMELID_DG_CUDA_LMHEAD=1. Needs a coalesced/warp kernel to win solo.
    if std::env::var("CAMELID_DG_CUDA_LMHEAD").as_deref() != Ok("1") {
        return None;
    }
    if wire.format != DgFormat::Q6K {
        return None;
    }
    let bpr = hidden / 256;
    let rb = wire.row_bytes();
    let bytes = wire.mmap.bytes(wire.offset, wire.rows * rb).ok()?;
    let mut scales = vec![0f32; c * bpr];
    let mut quants = vec![0i8; c * bpr * 256];
    for (pos, a) in acts.iter().enumerate() {
        let blocks = a.q8_k.as_ref()?;
        if blocks.len() != bpr {
            return None;
        }
        for (b, blk) in blocks.iter().enumerate() {
            scales[pos * bpr + b] = blk.d;
            let off = (pos * bpr + b) * 256;
            quants[off..off + 256].copy_from_slice(&blk.qs);
        }
    }
    cuda::lm_head_q6k_gpu(bytes, wire.rows, bpr, &scales, &quants, c)
}
#[cfg(not(feature = "cuda"))]
fn lm_head_gpu(
    _wire: &DgWire,
    _acts: &[DgActivation],
    _c: usize,
    _hidden: usize,
) -> Option<Vec<f32>> {
    None
}

/// MoE expert row-range GEMV on the VRAM-resident expert pool — returns the
/// `n_rows` outputs or `None` to fall back to the CPU rows loop. A no-op when
/// the `cuda` feature is off. The GPU kernels mirror `q4_k_dot_scalar` /
/// `q0_pair_dot` exactly (bit-identical outputs); only Q4_K and Q8_0 expert
/// formats route here — anything else stays on CPU.
#[cfg(feature = "cuda")]
fn expert_rows_gpu(
    wire: &DgWire,
    first_row: usize,
    n_rows: usize,
    x: &DgActivation,
) -> Option<Vec<f32>> {
    let rb = wire.row_bytes();
    // Whole-tensor slice: creating it faults no pages; only the one-time
    // resident upload reads it.
    let tensor = wire.mmap.bytes(wire.offset, wire.rows * rb).ok()?;
    match wire.format {
        DgFormat::Q4K => {
            let blocks = x.q8_k.as_ref()?;
            let bpr = wire.in_dim / 256;
            if blocks.len() != bpr {
                return None;
            }
            let mut scales = vec![0f32; bpr];
            let mut quants = vec![0i8; bpr * 256];
            for (b, blk) in blocks.iter().enumerate() {
                scales[b] = blk.d;
                quants[b * 256..(b + 1) * 256].copy_from_slice(&blk.qs);
            }
            cuda::expert_rows_gemv_gpu(
                tensor,
                cuda::DgExpertKind::Q4K,
                first_row,
                n_rows,
                bpr,
                &scales,
                &quants,
            )
        }
        DgFormat::Q8_0 => {
            let nb = wire.in_dim / 32;
            if x.q8_0.len() != nb {
                return None;
            }
            let mut scales = vec![0f32; nb];
            let mut quants = vec![0i8; nb * 32];
            for (b, blk) in x.q8_0.iter().enumerate() {
                scales[b] = blk.scale;
                quants[b * 32..(b + 1) * 32].copy_from_slice(&blk.quants);
            }
            cuda::expert_rows_gemv_gpu(
                tensor,
                cuda::DgExpertKind::Q80,
                first_row,
                n_rows,
                nb,
                &scales,
                &quants,
            )
        }
        _ => None,
    }
}
#[cfg(not(feature = "cuda"))]
fn expert_rows_gpu(
    _wire: &DgWire,
    _first_row: usize,
    _n_rows: usize,
    _x: &DgActivation,
) -> Option<Vec<f32>> {
    None
}

/// FAST mode (`CAMELID_DG_FAST=1`, cuda builds): batch the per-step FFN+MoE
/// across ALL positions on the GPU so each weight is read once per step
/// instead of once per position. NOT bit-exact (GPU f32 accumulation, GEMM
/// batching) — the parity lane is the default; this exists for interactive
/// latency. Token-closeness is the bar, like the SC GPU stage.
fn dg_fast_enabled() -> bool {
    cfg!(feature = "cuda")
        && matches!(
            std::env::var("CAMELID_DG_FAST").as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
        )
}

/// FAST-mode whole-layer FFN+MoE across all positions. Mirrors the
/// per-position loop's math mechanically (same helpers, same accumulation
/// formulas, same region-aware output scalar) but runs the six heavy matmuls
/// as batched GPU GEMMs — dense gate/up/down over all `n` positions and the
/// expert gate_up/down over all `n×k` (position, expert) pairs — so each
/// weight tensor is read/streamed ONCE per step. Mutates `h` only after every
/// GPU call has succeeded; `None` leaves the state untouched for the CPU
/// fallback loop.
#[cfg(feature = "cuda")]
fn ffn_moe_layer_fast(
    rt: &DgEncoderRuntime,
    lw: &DgLayer,
    h: &mut [Vec<f32>],
    p: usize,
    eps: f32,
) -> Option<()> {
    use rayon::prelude::*;
    // Q4_K_M mixes the down-projection quants per layer (Q8_0 on some layers,
    // Q5_0 on others); both have batched kernels. The gate/up sides are Q4_K
    // on every layer of the tracked file.
    let down_kind = |f: DgFormat| -> Option<cuda::DgExpertKind> {
        match f {
            DgFormat::Q8_0 => Some(cuda::DgExpertKind::Q80),
            DgFormat::Q5_0 => Some(cuda::DgExpertKind::Q50),
            _ => None,
        }
    };
    if lw.ffn_gate.format != DgFormat::Q4K
        || lw.ffn_up.format != DgFormat::Q4K
        || lw.gate_up_exps.format != DgFormat::Q4K
    {
        return None;
    }
    let dense_down_kind = down_kind(lw.ffn_down.format)?;
    let exp_down_kind = down_kind(lw.down_exps.format)?;
    let hidden = h.first()?.len();
    if !hidden.is_multiple_of(256) {
        return None;
    }
    let n = h.len();
    let k = rt.n_expert_used;
    let n_ff_exp = rt.n_ff_exp;
    let two_nff = 2 * n_ff_exp;
    let ffn_dim = lw.ffn_gate.rows;
    let inv = 1.0f32 / (hidden as f32).sqrt();
    let prof = std::env::var("CAMELID_DG_STAGE_TIMINGS").is_ok();
    let dbg_bail = |st: &str| {
        if prof {
            eprintln!("[dg-fast] BAIL {st}");
        }
    };
    let t0 = std::time::Instant::now();
    macro_rules! mark {
        ($label:expr, $t:ident) => {
            let now = std::time::Instant::now();
            if prof {
                eprint!("[dg-fast] {}={}ms ", $label, ($t.elapsed()).as_millis());
            }
            #[allow(unused)]
            let $t = now;
        };
    }
    let t = t0;

    // ---- per-position prep (parallel; no h mutation): dense input act,
    // router selection + normalized weights, MoE input act ----
    struct Prep {
        xq: DgActivation,
        cur_q: DgActivation,
        idx: Vec<usize>,
        w_norm: Vec<f32>,
    }
    let preps: Vec<Prep> = h
        .par_iter()
        .map(|hp| {
            let xn = refmath::rms_norm(hp, Some(&lw.ffn_norm), eps);
            let xq = DgActivation::new(&xn);
            let mut r = refmath::rms_norm(hp, None, eps);
            for (rv, sv) in r.iter_mut().zip(&lw.gate_inp_scale) {
                *rv = *rv * inv * sv;
            }
            let logits: Vec<f32> = (0..rt.n_expert)
                .map(|e| refmath::vec_dot_f32(&lw.gate_inp[e * hidden..(e + 1) * hidden], &r))
                .collect();
            let mut probs = logits.clone();
            refmath::softmax_row(&mut probs);
            let order = argsort_desc_experts(&logits);
            let idx: Vec<usize> = order[..k].iter().map(|&e| e as usize).collect();
            let selected: Vec<f32> = idx.iter().map(|&e| probs[e]).collect();
            let mut wsum = refmath::vec_sum_f32(&selected);
            wsum = wsum.max(f32::from_bits(0x3880_0000));
            let w_norm: Vec<f32> = idx
                .iter()
                .map(|&e| refmath::vdsp_div(probs[e], wsum))
                .collect();
            let cur_moe = refmath::rms_norm(hp, Some(&lw.pre_norm_2), eps);
            let cur_q = DgActivation::new(&cur_moe);
            Prep {
                xq,
                cur_q,
                idx,
                w_norm,
            }
        })
        .collect();
    mark!("prep", t);

    // ---- SoA packing helpers ----
    let bpr = hidden / 256;
    fn pack_q8k_soa(acts: &[&DgActivation], bpr: usize) -> Option<(Vec<f32>, Vec<i8>)> {
        let n = acts.len();
        let mut scales = vec![0f32; n * bpr];
        let mut quants = vec![0i8; n * bpr * 256];
        for (pos, a) in acts.iter().enumerate() {
            let blocks = a.q8_k.as_ref()?;
            if blocks.len() != bpr {
                return None;
            }
            for (b, blk) in blocks.iter().enumerate() {
                scales[pos * bpr + b] = blk.d;
                let off = (pos * bpr + b) * 256;
                quants[off..off + 256].copy_from_slice(&blk.qs);
            }
        }
        Some((scales, quants))
    }
    fn whole(w: &DgWire) -> Option<&[u8]> {
        w.mmap.bytes(w.offset, w.rows * w.row_bytes()).ok()
    }

    // ---- dense branch: gate/up (Q4_K), geglu, down (Q8_0), post-norm ----
    let xq_refs: Vec<&DgActivation> = preps.iter().map(|pr| &pr.xq).collect();
    let (xs, xqn) = pack_q8k_soa(&xq_refs, bpr).or_else(|| {
        dbg_bail("pack_xq");
        None
    })?;
    let ident_base = vec![0i64; n];
    let ident_pos: Vec<i32> = (0..n as i32).collect();
    let gate = cuda::fast_gemm_id(
        whole(&lw.ffn_gate)?,
        (lw.ffn_gate.mmap.path(), lw.ffn_gate.offset),
        cuda::DgExpertKind::Q4K,
        &ident_base,
        &ident_pos,
        ffn_dim,
        bpr,
        &xs,
        &xqn,
    )?;
    let up = cuda::fast_gemm_id(
        whole(&lw.ffn_up)?,
        (lw.ffn_up.mmap.path(), lw.ffn_up.offset),
        cuda::DgExpertKind::Q4K,
        &ident_base,
        &ident_pos,
        ffn_dim,
        bpr,
        &xs,
        &xqn,
    )
    .or_else(|| {
        dbg_bail("dense_gemm");
        None
    })?;
    mark!("dense_gemm", t);
    let dense_acts: Vec<DgActivation> = (0..n)
        .into_par_iter()
        .map(|pos| {
            let g = &gate[pos * ffn_dim..(pos + 1) * ffn_dim];
            let u = &up[pos * ffn_dim..(pos + 1) * ffn_dim];
            let act: Vec<f32> = g.iter().zip(u).map(|(gv, uv)| dg_gelu(*gv) * uv).collect();
            DgActivation::new(&act)
        })
        .collect();
    let nb_dense = ffn_dim / 32;
    let mut das = vec![0f32; n * nb_dense];
    let mut daq = vec![0i8; n * nb_dense * 32];
    for (pos, a) in dense_acts.iter().enumerate() {
        if a.q8_0.len() != nb_dense {
            dbg_bail("nb_dense");
            return None;
        }
        for (b, blk) in a.q8_0.iter().enumerate() {
            das[pos * nb_dense + b] = blk.scale;
            let off = (pos * nb_dense + b) * 32;
            daq[off..off + 32].copy_from_slice(&blk.quants);
        }
    }
    let mlp_raw = cuda::fast_gemm_id(
        whole(&lw.ffn_down)?,
        (lw.ffn_down.mmap.path(), lw.ffn_down.offset),
        dense_down_kind,
        &ident_base,
        &ident_pos,
        hidden,
        nb_dense,
        &das,
        &daq,
    )
    .or_else(|| {
        dbg_bail("mlp");
        None
    })?;
    let mlps: Vec<Vec<f32>> = (0..n)
        .into_par_iter()
        .map(|pos| {
            refmath::rms_norm(
                &mlp_raw[pos * hidden..(pos + 1) * hidden],
                Some(&lw.post_norm_1),
                eps,
            )
        })
        .collect();
    mark!("dense_down", t);

    // ---- MoE branch: expert gate_up (Q4_K) + down (Q8_0) over (pos, slot)
    // pairs ----
    let cq_refs: Vec<&DgActivation> = preps.iter().map(|pr| &pr.cur_q).collect();
    let (cs, cq) = pack_q8k_soa(&cq_refs, bpr).or_else(|| {
        dbg_bail("pack_cq");
        None
    })?;
    let n_pairs = n * k;
    let mut gu_base = Vec::with_capacity(n_pairs);
    let mut gu_pos = Vec::with_capacity(n_pairs);
    let mut dn_base = Vec::with_capacity(n_pairs);
    for (pos, prep) in preps.iter().enumerate() {
        for &e in &prep.idx {
            gu_base.push((e * two_nff) as i64);
            gu_pos.push(pos as i32);
            dn_base.push((e * hidden) as i64);
        }
    }
    let gate_up = cuda::fast_gemm_id(
        whole(&lw.gate_up_exps)?,
        (lw.gate_up_exps.mmap.path(), lw.gate_up_exps.offset),
        cuda::DgExpertKind::Q4K,
        &gu_base,
        &gu_pos,
        two_nff,
        bpr,
        &cs,
        &cq,
    )
    .or_else(|| {
        dbg_bail("exp_gate_up");
        None
    })?;
    mark!("exp_gate_up", t);
    let hexp_acts: Vec<DgActivation> = (0..n_pairs)
        .into_par_iter()
        .map(|pair| {
            let gu = &gate_up[pair * two_nff..(pair + 1) * two_nff];
            let hexp: Vec<f32> = (0..n_ff_exp)
                .map(|o| dg_gelu(gu[o]) * gu[o + n_ff_exp])
                .collect();
            DgActivation::new(&hexp)
        })
        .collect();
    let nb_exp = n_ff_exp / 32;
    let mut hs = vec![0f32; n_pairs * nb_exp];
    let mut hq = vec![0i8; n_pairs * nb_exp * 32];
    for (pair, a) in hexp_acts.iter().enumerate() {
        if a.q8_0.len() != nb_exp {
            dbg_bail("nb_exp");
            return None;
        }
        for (b, blk) in a.q8_0.iter().enumerate() {
            hs[pair * nb_exp + b] = blk.scale;
            let off = (pair * nb_exp + b) * 32;
            hq[off..off + 32].copy_from_slice(&blk.quants);
        }
    }
    mark!("hexp", t);
    let pair_ident: Vec<i32> = (0..n_pairs as i32).collect();
    let down = cuda::fast_gemm_id(
        whole(&lw.down_exps)?,
        (lw.down_exps.mmap.path(), lw.down_exps.offset),
        exp_down_kind,
        &dn_base,
        &pair_ident,
        hidden,
        nb_exp,
        &hs,
        &hq,
    )
    .or_else(|| {
        dbg_bail("exp_down");
        None
    })?;
    mark!("exp_down", t);

    // ---- finalize (parallel; the only h mutation, after all GPU calls) ----
    h.par_iter_mut().enumerate().for_each(|(pos, hp)| {
        let prep = &preps[pos];
        let mut moe_acc = vec![0f32; hidden];
        for (slot, &e) in prep.idx.iter().enumerate() {
            let pair = pos * k + slot;
            let y = &down[pair * hidden..(pair + 1) * hidden];
            let s_e = lw.down_exps_scale[e];
            let w = prep.w_norm[slot];
            for (a, yv) in moe_acc.iter_mut().zip(y) {
                *a += yv * s_e * w;
            }
        }
        let moe_out = refmath::rms_norm(&moe_acc, Some(&lw.post_norm_2), eps);
        let mut combined = mlps[pos].clone();
        for (cv, m) in combined.iter_mut().zip(&moe_out) {
            *cv += m;
        }
        let ffn_out = refmath::rms_norm(&combined, Some(&lw.post_ffw_norm), eps);
        for (a, b) in hp.iter_mut().zip(&ffn_out) {
            *a += b;
        }
        let scale = if pos < p {
            lw.enc_out_scale
        } else {
            lw.out_scale
        };
        for v in hp.iter_mut() {
            *v *= scale;
        }
    });
    mark!("finalize", t);
    if prof {
        eprintln!("total={}ms", t0.elapsed().as_millis());
    }
    Some(())
}
#[cfg(not(feature = "cuda"))]
fn ffn_moe_layer_fast(
    _rt: &DgEncoderRuntime,
    _lw: &DgLayer,
    _h: &mut [Vec<f32>],
    _p: usize,
    _eps: f32,
) -> Option<()> {
    None
}

/// FAST-mode whole-layer attention block: qkv projections, per-head norms +
/// RoPE (CPU — cheap and keeps the libm sincos semantics), the bidirectional
/// masked attention (GPU kernel), and the output projection + residual.
/// Mirrors the per-position path's math shape; not bit-exact (f32 GPU
/// attention). Mutates `h` only after every GPU call has succeeded.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn attn_layer_fast(
    lw: &DgLayer,
    h: &mut [Vec<f32>],
    heads: usize,
    head_dim: usize,
    kv_heads: usize,
    group: usize,
    theta: f32,
    rope_factors: Option<&[f32]>,
    sliding: bool,
    win: usize,
    p: usize,
    canvas_prompt_lo: i64,
    eps: f32,
    /* absolute position of h[0] (0 = full pass, p = cached canvas-only) */
    row0: usize,
    /* cached-in prefix K/V to prepend, each [p*kv_dim] (post norm+rope) */
    prefix_kv: Option<(&[f32], &[f32])>,
    /* capture-out: written with the prefix [0..p*kv_dim] slices ONLY as the
    last op before Some(()) — buffer written <=> layer fully succeeded */
    capture: Option<(&mut Vec<f32>, &mut Vec<f32>)>,
) -> Option<()> {
    use rayon::prelude::*;
    if lw.attn_q.format != DgFormat::Q4K || lw.attn_k.format != DgFormat::Q4K {
        return None;
    }
    let v_kind = match lw.attn_v.as_ref().map(|w| w.format) {
        None => None,
        Some(DgFormat::Q4K) => Some(cuda::DgExpertKind::Q4K),
        Some(DgFormat::Q6K) => Some(cuda::DgExpertKind::Q6K),
        Some(_) => return None,
    };
    if lw.attn_output.format != DgFormat::Q4K {
        return None;
    }
    let hidden = h.first()?.len();
    if !hidden.is_multiple_of(256) {
        return None;
    }
    let n = h.len(); // rows materialized (c in cached mode)
    let n_total = row0 + n; // full attention context
    debug_assert!(prefix_kv.is_none() || capture.is_none());
    let q_dim = heads * head_dim;
    let kv_dim = kv_heads * head_dim;
    if !q_dim.is_multiple_of(256) {
        return None;
    }
    if let Some((pk, pv)) = prefix_kv {
        if pk.len() != p * kv_dim || pv.len() != p * kv_dim || row0 != p {
            return None; // malformed cache: caller bails the step
        }
    }

    // Per-position input activation (parallel; no h mutation).
    let acts: Vec<DgActivation> = h
        .par_iter()
        .map(|hp| {
            let xn = refmath::rms_norm(hp, Some(&lw.attn_norm), eps);
            DgActivation::new(&xn)
        })
        .collect();
    let bpr = hidden / 256;
    let mut xs = vec![0f32; n * bpr];
    let mut xqn = vec![0i8; n * bpr * 256];
    for (pos, a) in acts.iter().enumerate() {
        let blocks = a.q8_k.as_ref()?;
        if blocks.len() != bpr {
            return None;
        }
        for (b, blk) in blocks.iter().enumerate() {
            xs[pos * bpr + b] = blk.d;
            let off = (pos * bpr + b) * 256;
            xqn[off..off + 256].copy_from_slice(&blk.qs);
        }
    }
    let ident_base = vec![0i64; n];
    let ident_pos: Vec<i32> = (0..n as i32).collect();
    fn whole(w: &DgWire) -> Option<&[u8]> {
        w.mmap.bytes(w.offset, w.rows * w.row_bytes()).ok()
    }

    // Projections (batched GPU GEMMs).
    let mut q_all = cuda::fast_gemm_id(
        whole(&lw.attn_q)?,
        (lw.attn_q.mmap.path(), lw.attn_q.offset),
        cuda::DgExpertKind::Q4K,
        &ident_base,
        &ident_pos,
        q_dim,
        bpr,
        &xs,
        &xqn,
    )?;
    let mut k_all = cuda::fast_gemm_id(
        whole(&lw.attn_k)?,
        (lw.attn_k.mmap.path(), lw.attn_k.offset),
        cuda::DgExpertKind::Q4K,
        &ident_base,
        &ident_pos,
        kv_dim,
        bpr,
        &xs,
        &xqn,
    )?;
    let mut v_all = match (lw.attn_v.as_ref(), v_kind) {
        (Some(wv), Some(kind)) => cuda::fast_gemm_id(
            whole(wv)?,
            (wv.mmap.path(), wv.offset),
            kind,
            &ident_base,
            &ident_pos,
            kv_dim,
            bpr,
            &xs,
            &xqn,
        )?,
        // V-less layer: V is the RAW K projection (pre-norm, pre-RoPE).
        _ => k_all.clone(),
    };

    // Per-head norms + RoPE (parallel per position; mirrors the CPU path).
    q_all
        .par_chunks_mut(q_dim)
        .zip(
            k_all
                .par_chunks_mut(kv_dim)
                .zip(v_all.par_chunks_mut(kv_dim)),
        )
        .enumerate()
        .for_each(|(pos, (qp, (kp, vp)))| {
            for hh in 0..heads {
                let s = &mut qp[hh * head_dim..(hh + 1) * head_dim];
                s.copy_from_slice(&refmath::rms_norm(s, Some(&lw.q_norm), eps));
            }
            // Absolute positions feed RoPE (row0 = p in cached mode — the
            // single most-likely-missed edit per the blueprint's trap list).
            refmath::rope_neox(qp, heads, head_dim, row0 + pos, theta, rope_factors);
            for hh in 0..kv_heads {
                let s = &mut kp[hh * head_dim..(hh + 1) * head_dim];
                s.copy_from_slice(&refmath::rms_norm(s, Some(&lw.k_norm), eps));
                let sv = &mut vp[hh * head_dim..(hh + 1) * head_dim];
                sv.copy_from_slice(&refmath::rms_norm(sv, None, eps));
            }
            refmath::rope_neox(kp, kv_heads, head_dim, row0 + pos, theta, rope_factors);
        });

    // Cached mode: full-context K/V = [cached prefix | fresh canvas].
    let (k_ctx, v_ctx) = match prefix_kv {
        Some((pk, pv)) => {
            let mut kf = Vec::with_capacity(n_total * kv_dim);
            kf.extend_from_slice(pk);
            kf.extend_from_slice(&k_all);
            let mut vf = Vec::with_capacity(n_total * kv_dim);
            vf.extend_from_slice(pv);
            vf.extend_from_slice(&v_all);
            (kf, vf)
        }
        None => (k_all, v_all),
    };

    // Attention (GPU kernel): queries cover the materialized rows at
    // absolute offset row0; K/V span the full context.
    let attn = cuda::dg_attention_gpu(
        &q_all,
        &k_ctx,
        &v_ctx,
        n_total,
        n,
        row0,
        heads,
        kv_heads,
        head_dim,
        group,
        p,
        win,
        sliding,
        canvas_prompt_lo,
    )?;

    // Output projection: quantize the attention mix, batched GEMM, post-norm
    // + residual (the only h mutation, after all GPU calls).
    let attn_acts: Vec<DgActivation> = attn.par_chunks(q_dim).map(DgActivation::new).collect();
    let abpr = q_dim / 256;
    let mut asx = vec![0f32; n * abpr];
    let mut aqn = vec![0i8; n * abpr * 256];
    for (pos, a) in attn_acts.iter().enumerate() {
        let blocks = a.q8_k.as_ref()?;
        if blocks.len() != abpr {
            return None;
        }
        for (b, blk) in blocks.iter().enumerate() {
            asx[pos * abpr + b] = blk.d;
            let off = (pos * abpr + b) * 256;
            aqn[off..off + 256].copy_from_slice(&blk.qs);
        }
    }
    let o_all = cuda::fast_gemm_id(
        whole(&lw.attn_output)?,
        (lw.attn_output.mmap.path(), lw.attn_output.offset),
        cuda::DgExpertKind::Q4K,
        &ident_base,
        &ident_pos,
        hidden,
        abpr,
        &asx,
        &aqn,
    )?;
    h.par_iter_mut().enumerate().for_each(|(pos, hp)| {
        let on = refmath::rms_norm(
            &o_all[pos * hidden..(pos + 1) * hidden],
            Some(&lw.post_attn_norm),
            eps,
        );
        for (a, b) in hp.iter_mut().zip(&on) {
            *a += b;
        }
    });
    if let Some((ck, cv)) = capture {
        // Prefix rows only, written strictly AFTER the residual mutation —
        // a written buffer certifies total layer success (the cache seal
        // check relies on exactly this).
        ck.clear();
        ck.extend_from_slice(&k_ctx[..p * kv_dim]);
        cv.clear();
        cv.extend_from_slice(&v_ctx[..p * kv_dim]);
    }
    Some(())
}
#[cfg(not(feature = "cuda"))]
#[allow(clippy::too_many_arguments)]
fn attn_layer_fast(
    _lw: &DgLayer,
    _h: &mut [Vec<f32>],
    _heads: usize,
    _head_dim: usize,
    _kv_heads: usize,
    _group: usize,
    _theta: f32,
    _rope_factors: Option<&[f32]>,
    _sliding: bool,
    _win: usize,
    _p: usize,
    _canvas_prompt_lo: i64,
    _eps: f32,
    _row0: usize,
    _prefix_kv: Option<(&[f32], &[f32])>,
    _capture: Option<(&mut Vec<f32>, &mut Vec<f32>)>,
) -> Option<()> {
    None
}

/// FAST-mode fused SC (device softmax over the resident previous-step
/// logits). A no-op when the `cuda` feature is off.
#[cfg(feature = "cuda")]
fn sc_soft_fused(
    temp_inv: f32,
    embed_scale: f32,
    c: usize,
    hidden: usize,
    n_vocab: usize,
) -> Option<Vec<f32>> {
    cuda::sc_soft_fused_gpu(temp_inv, embed_scale, c, hidden, n_vocab)
}
#[cfg(not(feature = "cuda"))]
fn sc_soft_fused(
    _temp_inv: f32,
    _embed_scale: f32,
    _c: usize,
    _hidden: usize,
    _n_vocab: usize,
) -> Option<Vec<f32>> {
    None
}

/// FAST-mode lm_head: flatten the canvas activations and run the tiled f16
/// GEMM against the resident SC embedding transpose. `None` → the existing
/// (parity-grade) lm_head paths.
#[cfg(feature = "cuda")]
fn lm_head_fast_gemm(
    rt: &DgEncoderRuntime,
    canvas_rns: &[Vec<f32>],
    c: usize,
    hidden: usize,
) -> Option<Vec<f32>> {
    let emb_t = rt.sc_emb_t().ok()?;
    let n_vocab = rt.token_embd.rows;
    let mut flat = Vec::with_capacity(c * hidden);
    for rn in canvas_rns {
        flat.extend_from_slice(rn);
    }
    // Softcapping fuses into the GEMM store (0.0 = none) — the caller must
    // NOT re-apply the host softcap to these logits.
    let cap = rt.final_logit_softcapping.unwrap_or(0.0);
    cuda::lm_head_f16_gemm_gpu(emb_t, &flat, c, hidden, n_vocab, cap)
}
#[cfg(not(feature = "cuda"))]
fn lm_head_fast_gemm(
    _rt: &DgEncoderRuntime,
    _canvas_rns: &[Vec<f32>],
    _c: usize,
    _hidden: usize,
) -> Option<Vec<f32>> {
    None
}

use crate::gguf::{read_metadata, GgufTensorDescriptor, GgufTensorType};

// Expert-selection argsort matching the reference's `ggml_argsort_top_k`
// comparator. The reference's CPU path is libc++ `std::sort` over expert indices
// with the STRICT, no-tie-break comparator (`keys[a] > keys[b]` for DESC). That
// sort is *unstable*: for an exact-equal key tie (a true probability tie) the
// relative order is libc++-introsort-internal.
//
// This is a pure-Rust implementation (no C/C++ shim). Every NON-tie comparison
// is identical to the reference because the comparator is the same strict
// `>`/`<`; only exact f32 ties are resolved differently (we break them by lower
// index, which is deterministic). An unspecified libc++ introsort tie-order
// cannot be reproduced portably, and the lane's Apple-specific math bindings
// (`__sincosf_stret`, vDSP) already place non-macOS targets out of bit-parity
// with the pinned reference regardless. On Apple Silicon, exact-tie bit-parity
// against the reference is therefore no longer guaranteed once the C++ shim is
// removed; re-validate the encoder/decode parity gates on a Mac if that matters
// (the shim is recoverable from git history). See DIFFUSIONGEMMA_RECON.md.
//
// Sorting by the bit-exact router LOGITS with `>` is comparison-identical to the
// reference sorting softmax `selection_probs` (softmax is strictly monotonic:
// logit[a] > logit[b] <=> prob[a] > prob[b], and equal logits <=> equal probs).

/// Indices `[0..keys.len())` ordered by DESCENDING `keys` (strict `>`, lower
/// index breaks exact ties). Use the bit-exact router logits as the key.
fn argsort_desc_experts(keys: &[f32]) -> Vec<i32> {
    let mut out: Vec<i32> = (0..keys.len() as i32).collect();
    out.sort_unstable_by(|&a, &b| {
        keys[b as usize]
            .partial_cmp(&keys[a as usize])
            .unwrap()
            .then(a.cmp(&b))
    });
    out
}

/// Indices ordered by ASCENDING `keys` (strict `<`, lower index breaks exact
/// ties) — the EB sampler's MI-bound position ordering (reference sorts
/// positions by entropy with `std::sort`, strict `<`).
fn argsort_asc_libcpp(keys: &[f32]) -> Vec<i32> {
    let mut out: Vec<i32> = (0..keys.len() as i32).collect();
    out.sort_unstable_by(|&a, &b| {
        keys[a as usize]
            .partial_cmp(&keys[b as usize])
            .unwrap()
            .then(a.cmp(&b))
    });
    out
}

use crate::inference::{
    q6_k_wire_block_dequant, quantize_q8_k_blocks, Q8KBlock, Q4_K_WIRE_BYTES_PER_BLOCK,
    Q5_0_WIRE_BYTES_PER_BLOCK, Q6_K_VALUES_PER_BLOCK, Q6_K_WIRE_BYTES_PER_BLOCK,
};
use crate::model::Gemma4Metadata;
use crate::tensor::{Q8_0Block, TensorStore};
use crate::wire_mmap::GgufWireMmap;
use crate::{BackendError, Result};

const Q8_0_WIRE_BYTES_PER_BLOCK: usize = 34;

/// The reference CPU GELU (`GGML_GELU_FP16`, on unconditionally at the pin):
/// clamp at ±10, otherwise evaluate the tanh approximation AT the
/// f16-rounded input and round the result to f16 — the exact semantics of
/// the `ggml_table_gelu_f16` lookup. The polynomial factoring mirrors
/// `ggml_gelu_f32` exactly (`x*(1 + A*x²)`, not `x + A*x³` — the two round
/// differently in f32). Camelid's exact-tanh `gelu_tanh` is a systematic
/// ~1e-3 divergence per FFN activation against this table; the DG lane's
/// checkpoint gate requires the table semantics.
// constants written to full reference precision on purpose (don't let clippy
// round them and shift the output)
#[allow(clippy::excessive_precision)]
fn dg_gelu(x: f32) -> f32 {
    if x <= -10.0 {
        return 0.0;
    }
    if x >= 10.0 {
        return x;
    }
    const SQRT_2_OVER_PI: f32 = 0.797_884_560_802_865_4;
    const GELU_COEF_A: f32 = 0.044_715;
    let v = crate::tensor::f16_round(x);
    // table-init contraction probe: clang fuses `1.0f + A*x*x` into
    // fma((A*x), x, 1.0) in the compiled table builder
    let inner = (GELU_COEF_A * v).mul_add(v, 1.0);
    let g = 0.5 * v * (1.0 + refmath::libm_tanhf(SQRT_2_OVER_PI * v * inner));
    crate::tensor::f16_round(g)
}

/// Quantized wire formats this lane has dequant-parity evidence for
/// (Phase 0.5) and a reference-mirroring row dot for (Phase 2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DgFormat {
    Q8_0,
    Q5_0,
    Q4K,
    Q6K,
}

impl DgFormat {
    fn from_tensor_type(t: GgufTensorType, name: &str) -> Result<Self> {
        match t {
            GgufTensorType::Q8_0 => Ok(Self::Q8_0),
            GgufTensorType::Q5_0 => Ok(Self::Q5_0),
            GgufTensorType::Q4K => Ok(Self::Q4K),
            GgufTensorType::Q6K => Ok(Self::Q6K),
            other => Err(BackendError::UnsupportedTensorType(format!(
                "tensor {name} is {other:?}; the DiffusionGemma encoder supports Q8_0, Q5_0, \
                 Q4_K, and Q6_K weight rows (the formats with committed parity evidence)"
            ))),
        }
    }

    fn values_per_block(self) -> usize {
        match self {
            Self::Q8_0 | Self::Q5_0 => 32,
            Self::Q4K | Self::Q6K => Q6_K_VALUES_PER_BLOCK,
        }
    }

    fn bytes_per_block(self) -> usize {
        match self {
            Self::Q8_0 => Q8_0_WIRE_BYTES_PER_BLOCK,
            Self::Q5_0 => Q5_0_WIRE_BYTES_PER_BLOCK,
            Self::Q4K => Q4_K_WIRE_BYTES_PER_BLOCK,
            Self::Q6K => Q6_K_WIRE_BYTES_PER_BLOCK,
        }
    }
}

/// One input activation, quantized once per projection group in both shapes
/// the row dots consume.
struct DgActivation {
    q8_0: Vec<Q8_0Block>,
    q8_k: Option<Vec<Q8KBlock>>,
}

impl DgActivation {
    /// `len` must be 32-aligned; the Q8_K form is built only when it is also
    /// 256-aligned (rows of other widths never bind K-quant formats — the
    /// loader's block-alignment validation guarantees it).
    fn new(x: &[f32]) -> Self {
        let q8_k = if x.len().is_multiple_of(Q6_K_VALUES_PER_BLOCK) {
            Some(quantize_q8_k_blocks(x))
        } else {
            None
        };
        // ARM reference semantics (refmath::quantize_q8_0_arm): nearest-even
        // rounding (vcvtnq) and the scale stored at f16 precision. DG-lane
        // local: the proven gemma4 rows keep their committed behavior.
        Self {
            q8_0: refmath::quantize_q8_0_arm(x),
            q8_k,
        }
    }
}

/// A quantized weight matrix read lazily from the GGUF mmap; rows are dotted
/// in place against pre-quantized activations with the reference-mirroring
/// kernels.
struct DgWire {
    mmap: std::sync::Arc<GgufWireMmap>,
    offset: u64,
    in_dim: usize,
    rows: usize,
    format: DgFormat,
    /// Eligible for the CUDA expert pool (MoE expert tensors only): row-range
    /// GEMVs may run on a VRAM-resident copy via kernels that mirror the CPU
    /// reduction bit-for-bit. Never changes the math, only where it runs.
    expert_pool: bool,
}

impl DgWire {
    fn bind(
        mmap: &std::sync::Arc<GgufWireMmap>,
        desc: &GgufTensorDescriptor,
        expect_in_dim: usize,
    ) -> Result<Self> {
        let format = DgFormat::from_tensor_type(desc.tensor_type, &desc.name)?;
        let element_count = desc.dimensions.iter().product::<u64>() as usize;
        let in_dim = desc.dimensions.first().copied().unwrap_or(0) as usize;
        if in_dim != expect_in_dim {
            return Err(BackendError::InvalidTensorData(format!(
                "tensor {} in_dim {in_dim} != expected {expect_in_dim}",
                desc.name
            )));
        }
        if !in_dim.is_multiple_of(format.values_per_block()) {
            return Err(BackendError::InvalidTensorData(format!(
                "tensor {} rows of {in_dim} are not aligned to {:?} blocks",
                desc.name, format
            )));
        }
        let rows = element_count / in_dim;
        let byte_len = element_count / format.values_per_block() * format.bytes_per_block();
        if desc.n_bytes as usize != byte_len {
            return Err(BackendError::InvalidTensorData(format!(
                "tensor {} {:?} byte size {} != expected {byte_len}",
                desc.name, format, desc.n_bytes
            )));
        }
        mmap.bytes(desc.absolute_offset, byte_len)?;
        Ok(Self {
            mmap: mmap.clone(),
            offset: desc.absolute_offset,
            in_dim,
            rows,
            format,
            expert_pool: false,
        })
    }

    fn row_bytes(&self) -> usize {
        self.in_dim / self.format.values_per_block() * self.format.bytes_per_block()
    }

    /// One weight row (raw wire bytes) dotted with one quantized activation,
    /// dispatched by format. This IS the reference reduction order (the `_arm`
    /// kernels, which dispatch to bit-identical AVX2 on x86_64).
    #[inline]
    fn dot_row(&self, row: &[u8], x: &DgActivation) -> f32 {
        match self.format {
            DgFormat::Q8_0 => refmath::q8_0_dot_arm(row, &x.q8_0),
            DgFormat::Q5_0 => refmath::q5_0_dot_arm(row, &x.q8_0),
            DgFormat::Q4K => refmath::q4_k_dot_arm(
                row,
                x.q8_k
                    .as_ref()
                    .expect("K-quant rows imply 256-aligned input"),
            ),
            DgFormat::Q6K => refmath::q6_k_dot_arm(
                row,
                x.q8_k
                    .as_ref()
                    .expect("K-quant rows imply 256-aligned input"),
            ),
        }
    }

    /// y[r] = dequant(W[first_row + r]) · x for `n_rows` rows.
    fn matvec_rows(&self, first_row: usize, n_rows: usize, x: &DgActivation) -> Result<Vec<f32>> {
        if first_row + n_rows > self.rows {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "row range [{first_row}, {}) exceeds {} rows",
                first_row + n_rows,
                self.rows
            )));
        }
        // Expert-pool tensors may compute on a VRAM-resident copy; the GPU
        // kernels mirror the CPU reduction bit-for-bit, so this branch never
        // changes any value — only where (and how fast) it is produced.
        if self.expert_pool {
            if let Some(y) = expert_rows_gpu(self, first_row, n_rows, x) {
                return Ok(y);
            }
        }
        let rb = self.row_bytes();
        let bytes = self
            .mmap
            .bytes(self.offset + (first_row * rb) as u64, n_rows * rb)?;
        let mut out = vec![0f32; n_rows];
        // Row-parallel matvec: every output element is one self-contained row
        // dot over the same input, so distributing ROWS across threads cannot
        // change any value (no shared reduction; each y[r] is written once).
        // The result is bit-identical to the serial path regardless of thread
        // count — the reference's own mul_mat threads partition the same way.
        // Uses the global rayon pool (no per-call OS-thread spawn); small
        // matvecs stay serial because the fork/join would dominate.
        if n_rows * self.in_dim < (1 << 18) {
            for (r, y) in out.iter_mut().enumerate() {
                *y = self.dot_row(&bytes[r * rb..(r + 1) * rb], x);
            }
        } else {
            use rayon::prelude::*;
            out.par_iter_mut().enumerate().for_each(|(r, y)| {
                *y = self.dot_row(&bytes[r * rb..(r + 1) * rb], x);
            });
        }
        Ok(out)
    }

    /// Dense `mul_mat` semantics: Q8_0 weights route through the
    /// tinyBLAS_Q0_ARM element order (llamafile engages for dense Q8_0 GEMMs
    /// at the pin); every other format uses the vec_dot path. The MoE expert
    /// path (`mul_mat_id`) never uses tinyBLAS — experts call `matvec_rows`.
    fn matvec_dense(&self, x: &DgActivation) -> Result<Vec<f32>> {
        // empirically dense Q8_0 matmuls also match the vec_dot path
        // (llamafile does not engage for this graph's shapes)
        self.matvec_rows(0, self.rows, x)
    }
}

struct DgLayer {
    attn_norm: Vec<f32>,
    attn_q: DgWire,
    attn_k: DgWire,
    /// `None` on V-less layers (V = raw K projection).
    attn_v: Option<DgWire>,
    attn_output: DgWire,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    post_attn_norm: Vec<f32>,
    ffn_norm: Vec<f32>,
    ffn_gate: DgWire,
    ffn_up: DgWire,
    ffn_down: DgWire,
    post_ffw_norm: Vec<f32>,
    // MoE branch (every DiffusionGemma layer carries one)
    post_norm_1: Vec<f32>,
    pre_norm_2: Vec<f32>,
    post_norm_2: Vec<f32>,
    gate_inp: Vec<f32>,
    gate_inp_scale: Vec<f32>,
    gate_up_exps: DgWire,
    down_exps: DgWire,
    down_exps_scale: Vec<f32>,
    /// Decoder (canvas) per-layer output scalar — unused by the encoder but
    /// validated at load so Phase 3 starts from a bound weight.
    #[allow(dead_code)]
    out_scale: f32,
    /// Encoder (prompt) per-layer output scalar.
    enc_out_scale: f32,
}

/// Per-layer encoder checkpoints, shapes matching the reference dump rows
/// (positions are the slow axis; vectors are row-major `[P * width]`).
pub struct DgLayerTrace {
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub attn_out: Vec<f32>,
    pub moe_logits: Vec<f32>,
    pub moe_topk: Vec<i32>,
    pub out_scaled: Vec<f32>,
    /// dense (shared-expert) branch output after its post-norm — the
    /// reference's surviving "ffn_mlp" label.
    pub ffn_mlp: Vec<f32>,
    /// MoE branch output after its post-norm — "ffn_moe".
    pub ffn_moe: Vec<f32>,
    /// RAW selected-expert probabilities in slot order (pre-normalization) —
    /// "ffn_moe_weights".
    pub moe_weights: Vec<f32>,
    /// Expert-chain bisection traces (diagnostic), `[pos][slot][dim]`
    /// flattened to match the reference's ne2-major layout.
    pub moe_gate_up: Vec<f32>,
    pub moe_geglu: Vec<f32>,
    pub moe_down: Vec<f32>,
    pub moe_down_scaled: Vec<f32>,
    /// Normalized per-slot weights — "ffn_moe_weights_norm".
    pub moe_weights_norm: Vec<f32>,
    /// PRE-norm weighted slot sum — "ffn_moe_out".
    pub moe_pre_norm: Vec<f32>,
    /// Pre-scalar FFN block output (attn_resid + post_ffw_norm(mlp+moe)) —
    /// reference "ffn_block_out" (cb'd before the region scalar).
    pub ffn_block_out: Vec<f32>,
    /// Phase 5 attention-internal diagnostic: the pre-`wo` KQV in the
    /// reference's "kqv" layout `[head_dim, n_q, n_head]` (index
    /// `d + q*head_dim + h*head_dim*n_q`). Captured for layer 0 only.
    pub kqv: Vec<f32>,
    /// Phase 5: the post-softmax attention weights in the reference's
    /// "kq_soft_max" layout `[n_kv, n_q, n_head]` (index
    /// `k + q*n_kv + h*n_kv*n_q`). Captured for layer 0 only.
    pub kq_soft_max: Vec<f32>,
}

pub struct DgEncoderTrace {
    pub n_pos: usize,
    pub inp_scaled: Vec<f32>,
    pub layers: Vec<DgLayerTrace>,
    /// `output_norm` of every position (reference dumps have carried both
    /// one-row and all-rows shapes; the comparator slices to match).
    pub result_norm_all: Vec<f32>,
    /// `output_norm` of the LAST position only.
    pub result_norm_last: Vec<f32>,
}

pub struct DgEncoderRuntime {
    g: Gemma4Metadata,
    n_layer: usize,
    n_embd: usize,
    n_head: usize,
    n_expert: usize,
    n_expert_used: usize,
    n_ff_exp: usize,
    eps: f32,
    layers: Vec<DgLayer>,
    token_embd: DgWire,
    output_norm: Vec<f32>,
    rope_factors: Option<Vec<f32>>,
    /// `diffusion.canvas_length` — the unified forward's region split.
    canvas_length: usize,
    /// `final_logit_softcapping` (30.0 on the tracked GGUF); `None` skips the
    /// capping exactly as the reference's falsy check does.
    final_logit_softcapping: Option<f32>,
    // self-conditioning gated MLP (Phase 4)
    sc_pre_norm: Vec<f32>,
    sc_gate: DgWire,
    sc_up: DgWire,
    sc_down: DgWire,
    /// The SC soft-embedding weight: `token_embd` dequantized and transposed
    /// to f16 `[n_embd rows][n_vocab]` exactly as the reference's
    /// `dg_ensure_sc_embT` (built lazily once; ~1.5 GB).
    sc_emb_t: std::sync::OnceLock<Vec<u16>>,
}

impl DgEncoderRuntime {
    pub fn load(path: &Path) -> Result<Self> {
        let gguf = read_metadata(path)?;
        let arch = gguf.architecture().unwrap_or_default().to_string();
        if arch != "diffusion-gemma" {
            return Err(BackendError::UnsupportedModelArchitecture(format!(
                "DgEncoderRuntime requires general.architecture diffusion-gemma, got {arch:?}"
            )));
        }
        let g = Gemma4Metadata::from_gguf(&gguf, &arch).ok_or_else(|| {
            BackendError::InvalidModelMetadata("diffusion-gemma metadata parse failed".into())
        })?;
        let meta_u32 = |suffix: &str| -> Result<u32> {
            gguf.metadata_u32(&format!("{arch}.{suffix}"))
                .ok_or_else(|| {
                    BackendError::InvalidModelMetadata(format!("missing {arch}.{suffix}"))
                })
        };
        let n_layer = meta_u32("block_count")? as usize;
        let n_embd = meta_u32("embedding_length")? as usize;
        let n_head = meta_u32("attention.head_count")? as usize;
        let n_expert = meta_u32("expert_count")? as usize;
        let n_expert_used = meta_u32("expert_used_count")? as usize;
        let n_ff_exp = meta_u32("expert_feed_forward_length")? as usize;
        let eps = gguf
            .metadata_f32(&format!("{arch}.attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-6);
        let canvas_length = gguf
            .metadata_u32("diffusion.canvas_length")
            .or_else(|| {
                gguf.metadata_string("diffusion.canvas_length")
                    .and_then(|s| s.parse().ok())
            })
            .ok_or_else(|| {
                BackendError::InvalidModelMetadata(
                    "missing diffusion.canvas_length (not a DiffusionGemma file?)".into(),
                )
            })? as usize;
        if canvas_length == 0 {
            return Err(BackendError::InvalidModelMetadata(
                "diffusion.canvas_length must be positive".into(),
            ));
        }
        let final_logit_softcapping = gguf
            .metadata_f32(&format!("{arch}.final_logit_softcapping"))
            .filter(|&c| c != 0.0);

        let store = TensorStore::open(path, &gguf);
        let mmap = GgufWireMmap::map(path)?;
        let desc = |name: &str| -> Result<&GgufTensorDescriptor> {
            gguf.tensors
                .iter()
                .find(|t| t.name == name)
                .ok_or_else(|| BackendError::InvalidModelMetadata(format!("missing tensor {name}")))
        };
        let f32t = |name: &str| -> Result<Vec<f32>> { Ok(store.load_cpu_f32(name)?.data) };
        let scalar = |name: &str| -> Result<f32> {
            let v = f32t(name)?;
            v.first()
                .copied()
                .ok_or_else(|| BackendError::InvalidTensorData(format!("{name} is empty")))
        };

        let mut layers = Vec::with_capacity(n_layer);
        for l in 0..n_layer {
            let head_dim = g.head_dim_at(l) as usize;
            let q_dim = n_head * head_dim;
            let ffn_dim = g.ffn_length_at(l) as usize;
            let t = |suffix: &str| format!("blk.{l}.{suffix}");
            let wire = |name: &str, in_dim: usize| -> Result<DgWire> {
                DgWire::bind(&mmap, desc(name)?, in_dim)
            };
            let attn_v = if gguf.tensors.iter().any(|d| d.name == t("attn_v.weight")) {
                Some(wire(&t("attn_v.weight"), n_embd)?)
            } else {
                None
            };
            let mut layer = DgLayer {
                attn_norm: f32t(&t("attn_norm.weight"))?,
                attn_q: wire(&t("attn_q.weight"), n_embd)?,
                attn_k: wire(&t("attn_k.weight"), n_embd)?,
                attn_v,
                attn_output: wire(&t("attn_output.weight"), q_dim)?,
                q_norm: f32t(&t("attn_q_norm.weight"))?,
                k_norm: f32t(&t("attn_k_norm.weight"))?,
                post_attn_norm: f32t(&t("post_attention_norm.weight"))?,
                ffn_norm: f32t(&t("ffn_norm.weight"))?,
                ffn_gate: wire(&t("ffn_gate.weight"), n_embd)?,
                ffn_up: wire(&t("ffn_up.weight"), n_embd)?,
                ffn_down: wire(&t("ffn_down.weight"), ffn_dim)?,
                post_ffw_norm: f32t(&t("post_ffw_norm.weight"))?,
                post_norm_1: f32t(&t("post_ffw_norm_1.weight"))?,
                pre_norm_2: f32t(&t("pre_ffw_norm_2.weight"))?,
                post_norm_2: f32t(&t("post_ffw_norm_2.weight"))?,
                gate_inp: f32t(&t("ffn_gate_inp.weight"))?,
                gate_inp_scale: f32t(&t("ffn_gate_inp.scale"))?,
                gate_up_exps: wire(&t("ffn_gate_up_exps.weight"), n_embd)?,
                down_exps: wire(&t("ffn_down_exps.weight"), n_ff_exp)?,
                down_exps_scale: f32t(&t("ffn_down_exps.scale"))?,
                out_scale: scalar(&t("layer_output_scale.weight"))?,
                enc_out_scale: scalar(&t("enc_layer_output_scale.weight"))?,
            };
            // MoE expert tensors are eligible for the VRAM-resident expert
            // pool (cuda feature; budget-gated at first use, CPU otherwise).
            layer.gate_up_exps.expert_pool = true;
            layer.down_exps.expert_pool = true;
            if layer.gate_up_exps.rows != 2 * n_ff_exp * n_expert {
                return Err(BackendError::InvalidTensorData(format!(
                    "layer {l} gate_up_exps rows {} != 2*n_ff_exp*n_expert {}",
                    layer.gate_up_exps.rows,
                    2 * n_ff_exp * n_expert
                )));
            }
            if layer.down_exps.rows != n_embd * n_expert {
                return Err(BackendError::InvalidTensorData(format!(
                    "layer {l} down_exps rows {} != n_embd*n_expert {}",
                    layer.down_exps.rows,
                    n_embd * n_expert
                )));
            }
            layers.push(layer);
        }

        let token_embd = DgWire::bind(&mmap, desc("token_embd.weight")?, n_embd)?;
        if token_embd.format != DgFormat::Q6K {
            return Err(BackendError::UnsupportedTensorType(format!(
                "token_embd is {:?}; the tracked row's embedding is Q6_K and only that \
                 row gather has parity evidence",
                token_embd.format
            )));
        }
        let rope_factors = if gguf.tensors.iter().any(|d| d.name == "rope_freqs.weight") {
            Some(f32t("rope_freqs.weight")?)
        } else {
            None
        };

        Ok(Self {
            g,
            n_layer,
            n_embd,
            n_head,
            n_expert,
            n_expert_used,
            n_ff_exp,
            eps,
            sc_pre_norm: f32t("self_cond_pre_norm.weight")?,
            sc_gate: DgWire::bind(&mmap, desc("self_cond_gate.weight")?, n_embd)?,
            sc_up: DgWire::bind(&mmap, desc("self_cond_up.weight")?, n_embd)?,
            sc_down: {
                let ffn_dim = gguf
                    .tensors
                    .iter()
                    .find(|t| t.name == "self_cond_down.weight")
                    .and_then(|t| t.dimensions.first().copied())
                    .unwrap_or(0) as usize;
                DgWire::bind(&mmap, desc("self_cond_down.weight")?, ffn_dim)?
            },
            sc_emb_t: std::sync::OnceLock::new(),
            layers,
            token_embd,
            output_norm: f32t("output_norm.weight")?,
            rope_factors,
            canvas_length,
            final_logit_softcapping,
        })
    }

    /// Dequantize one embedding row (Q6_K) — mirrors the reference get_rows.
    fn embed_row(&self, token: u32) -> Result<Vec<f32>> {
        let blocks_per_row = self.n_embd / Q6_K_VALUES_PER_BLOCK;
        let rb = blocks_per_row * Q6_K_WIRE_BYTES_PER_BLOCK;
        let bytes = self
            .token_embd
            .mmap
            .bytes(self.token_embd.offset + (token as usize * rb) as u64, rb)?;
        let mut row = Vec::with_capacity(self.n_embd);
        for b in 0..blocks_per_row {
            row.extend_from_slice(&q6_k_wire_block_dequant(
                &bytes[b * Q6_K_WIRE_BYTES_PER_BLOCK..(b + 1) * Q6_K_WIRE_BYTES_PER_BLOCK],
            ));
        }
        Ok(row)
    }

    /// ENCODER prefill: one causal forward over the prompt with encoder-mode
    /// per-layer scalars, layer-major (each weight region is streamed once),
    /// emitting full per-layer checkpoints for the Phase 2 parity gate.
    pub fn encoder_prefill(&self, prompt: &[u32]) -> Result<DgEncoderTrace> {
        self.encoder_prefill_impl(prompt, None)
    }

    /// Diagnostic variant: pin the MoE routing to externally supplied expert
    /// indices (`routing[layer][pos*k + slot]`, the reference's selection)
    /// instead of camelid's own top-k. Used by the Phase 2 divergence report
    /// to isolate knife-edge router ties from every continuous checkpoint:
    /// camelid still computes and reports its own router logits/top-k; only
    /// the experts actually EXECUTED follow the pinned set (with camelid's
    /// own probabilities renormalized over that set, exactly as the
    /// reference renormalizes over its own set).
    pub fn encoder_prefill_with_pinned_routing(
        &self,
        prompt: &[u32],
        routing: &[Vec<i32>],
    ) -> Result<DgEncoderTrace> {
        if routing.len() != self.n_layer {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "pinned routing has {} layers, model has {}",
                routing.len(),
                self.n_layer
            )));
        }
        self.encoder_prefill_impl(prompt, Some(routing))
    }

    fn encoder_prefill_impl(
        &self,
        prompt: &[u32],
        pinned_routing: Option<&[Vec<i32>]>,
    ) -> Result<DgEncoderTrace> {
        let n = prompt.len();
        if n == 0 {
            return Err(BackendError::RuntimeShapeMismatch("empty prompt".into()));
        }
        let eps = self.eps;
        let hidden = self.n_embd;
        let heads = self.n_head;
        let win = self.g.sliding_window as usize;
        let embed_scale = (hidden as f32).sqrt();

        // embeddings (scaled)
        let mut h: Vec<Vec<f32>> = Vec::with_capacity(n);
        let mut inp_scaled = Vec::with_capacity(n * hidden);
        for &tok in prompt {
            let mut e = self.embed_row(tok)?;
            for v in e.iter_mut() {
                *v *= embed_scale;
            }
            inp_scaled.extend_from_slice(&e);
            h.push(e);
        }

        let mut traces = Vec::with_capacity(self.n_layer);
        for (l, lw) in self.layers.iter().enumerate() {
            let sliding = self.g.is_sliding_layer(l);
            let head_dim = self.g.head_dim_at(l) as usize;
            let kv_heads = self.g.kv_heads_at(l) as usize;
            let theta = self.g.rope_freq_base_at(l);
            let q_dim = heads * head_dim;
            let kv_dim = kv_heads * head_dim;
            let group = heads / kv_heads;
            let rope_factors = if sliding {
                None
            } else {
                self.rope_factors.as_deref()
            };

            // ---- projections for every position (layer-major) ----
            let mut qs: Vec<Vec<f32>> = Vec::with_capacity(n);
            let mut ks: Vec<Vec<f32>> = Vec::with_capacity(n);
            let mut vs: Vec<Vec<f32>> = Vec::with_capacity(n);
            for (pos, hp) in h.iter().enumerate() {
                let xn = refmath::rms_norm(hp, Some(&lw.attn_norm), eps);
                let xq = DgActivation::new(&xn);
                let mut q = lw.attn_q.matvec_dense(&xq)?;
                for hh in 0..heads {
                    let s = &mut q[hh * head_dim..(hh + 1) * head_dim];
                    s.copy_from_slice(&refmath::rms_norm(s, Some(&lw.q_norm), eps));
                }
                refmath::rope_neox(&mut q, heads, head_dim, pos, theta, rope_factors);

                let mut k = lw.attn_k.matvec_dense(&xq)?;
                // V-less layers reuse the RAW K projection as V; V then takes
                // the weightless norm and never RoPE.
                let mut v = match lw.attn_v.as_ref() {
                    Some(wv) => wv.matvec_dense(&xq)?,
                    None => k.clone(),
                };
                for hh in 0..kv_heads {
                    let s = &mut k[hh * head_dim..(hh + 1) * head_dim];
                    s.copy_from_slice(&refmath::rms_norm(s, Some(&lw.k_norm), eps));
                    let sv = &mut v[hh * head_dim..(hh + 1) * head_dim];
                    sv.copy_from_slice(&refmath::rms_norm(sv, None, eps));
                }
                refmath::rope_neox(&mut k, kv_heads, head_dim, pos, theta, rope_factors);
                qs.push(q);
                ks.push(k);
                vs.push(v);
            }

            // ---- causal attention (SWA-clipped on sliding layers) ----
            // V columns, contiguous over positions — the reference makes V
            // contiguous (cont∘transpose) so each KQV output element is one
            // vec_dot_f32 over n_kv; replicate that memory shape.
            let mut v_cols: Vec<Vec<f32>> = vec![vec![0f32; n]; kv_heads * head_dim];
            for (p, vp) in vs.iter().enumerate() {
                for (di, &val) in vp.iter().enumerate() {
                    v_cols[di][p] = val;
                }
            }

            let mut attn_out = Vec::with_capacity(n * hidden);
            for pos in 0..n {
                let lo = if sliding {
                    (pos + 1).saturating_sub(win)
                } else {
                    0
                };
                let mut attn = vec![0f32; q_dim];
                for hh in 0..heads {
                    let kvh = hh / group;
                    let qh = &qs[pos][hh * head_dim..(hh + 1) * head_dim];
                    // reference shape: KQ over the FULL row, additive -inf
                    // mask, then one softmax over the whole row (the masked
                    // slots' exp is exactly 0 in the reference's v_expf)
                    let mut row: Vec<f32> = (0..n)
                        .map(|p| {
                            if p < lo || p > pos {
                                f32::NEG_INFINITY
                            } else {
                                let kp = &ks[p][kvh * head_dim..(kvh + 1) * head_dim];
                                refmath::vec_dot_f32(qh, kp)
                            }
                        })
                        .collect();
                    refmath::softmax_row(&mut row);
                    let out = &mut attn[hh * head_dim..(hh + 1) * head_dim];
                    for (d, o) in out.iter_mut().enumerate() {
                        *o = refmath::vec_dot_f32(&v_cols[kvh * head_dim + d], &row);
                    }
                }
                let aq = DgActivation::new(&attn);
                let o = lw.attn_output.matvec_dense(&aq)?;
                let on = refmath::rms_norm(&o, Some(&lw.post_attn_norm), eps);
                for (a, b) in h[pos].iter_mut().zip(&on) {
                    *a += b;
                }
                attn_out.extend_from_slice(&h[pos]);
            }

            // ---- dense shared-expert MLP + 128-expert MoE ----
            let mut moe_logits_all = Vec::with_capacity(n * self.n_expert);
            let mut moe_topk_all = Vec::with_capacity(n * self.n_expert_used);
            let mut out_scaled = Vec::with_capacity(n * hidden);
            let mut ffn_mlp_all = Vec::with_capacity(n * hidden);
            let mut ffn_moe_all = Vec::with_capacity(n * hidden);
            let mut moe_weights_all = Vec::with_capacity(n * self.n_expert_used);
            let mut moe_gate_up_all = Vec::new();
            let mut moe_geglu_all = Vec::new();
            let mut moe_down_all = Vec::new();
            let mut moe_down_scaled_all = Vec::new();
            let mut moe_weights_norm_all = Vec::new();
            let mut moe_pre_norm_all = Vec::new();
            for (pos, hp) in h.iter_mut().enumerate() {
                let attn_resid = hp.clone();
                let xn = refmath::rms_norm(&attn_resid, Some(&lw.ffn_norm), eps);
                let xq = DgActivation::new(&xn);
                let gate = lw.ffn_gate.matvec_dense(&xq)?;
                let up = lw.ffn_up.matvec_dense(&xq)?;
                let act: Vec<f32> = gate.iter().zip(&up).map(|(g, u)| dg_gelu(*g) * u).collect();
                let actq = DgActivation::new(&act);
                let mlp = lw.ffn_down.matvec_dense(&actq)?;
                let mlp = refmath::rms_norm(&mlp, Some(&lw.post_norm_1), eps);
                ffn_mlp_all.extend_from_slice(&mlp);

                // Router: weightless RMS of the post-attention residual,
                // scaled by 1/sqrt(n_embd), then the elementwise input scale.
                let mut r = refmath::rms_norm(&attn_resid, None, eps);
                let inv = 1.0f32 / (hidden as f32).sqrt();
                for (rv, sv) in r.iter_mut().zip(&lw.gate_inp_scale) {
                    *rv = *rv * inv * sv;
                }
                // empirically the router GEMM matches the vec_dot_f32 tree, not
                // the tinyBLAS per-element order (llamafile did not engage)
                let logits: Vec<f32> = (0..self.n_expert)
                    .map(|e| refmath::vec_dot_f32(&lw.gate_inp[e * hidden..(e + 1) * hidden], &r))
                    .collect();
                moe_logits_all.extend_from_slice(&logits);

                // softmax over all experts (the reference's ggml_soft_max —
                // same kernel semantics as the attention softmax); weights come
                // from these probs.
                let mut probs: Vec<f32> = logits.clone();
                refmath::softmax_row(&mut probs);
                // expert ORDER must match the reference's ggml_argsort_top_k =
                // libc++ std::sort over DESC router logits (strict `>`; see
                // argsort_desc_experts). Sorting by the bit-exact logits is
                // comparison-identical to sorting softmax probs.
                let order = argsort_desc_experts(&logits);
                let mut idx: Vec<usize> = order[..self.n_expert_used]
                    .iter()
                    .map(|&e| e as usize)
                    .collect();
                for &e in &idx {
                    moe_topk_all.push(e as i32);
                    moe_weights_all.push(probs[e]);
                }
                // Diagnostic pinned routing: execute the supplied expert set
                // instead of our own (probabilities renormalize over the
                // executed set either way). The reported moe_topk above stays
                // camelid's own selection.
                if let Some(routing) = pinned_routing {
                    let k = self.n_expert_used;
                    idx = routing[l][pos * k..(pos + 1) * k]
                        .iter()
                        .map(|&e| e as usize)
                        .collect();
                }
                // ggml_sum_rows accumulates in double; the clamp constant is
                // exactly 2^-14 (6.103515625e-5)
                let selected: Vec<f32> = idx.iter().map(|&e| probs[e]).collect();
                let mut wsum = refmath::vec_sum_f32(&selected);
                wsum = wsum.max(f32::from_bits(0x3880_0000));

                let cur_moe = refmath::rms_norm(&attn_resid, Some(&lw.pre_norm_2), eps);
                let cur_moe_q = DgActivation::new(&cur_moe);
                let two_nff = 2 * self.n_ff_exp;
                let mut moe_acc = vec![0f32; hidden];
                for &e in &idx {
                    // the reference's weight normalization divides via
                    // Apple's vDSP_vdiv (not IEEE division) — bind it
                    let w = refmath::vdsp_div(probs[e], wsum);
                    moe_weights_norm_all.push(w);
                    let gate_up = lw
                        .gate_up_exps
                        .matvec_rows(e * two_nff, two_nff, &cur_moe_q)?;
                    let hexp: Vec<f32> = (0..self.n_ff_exp)
                        .map(|o| dg_gelu(gate_up[o]) * gate_up[o + self.n_ff_exp])
                        .collect();
                    let hexp_q = DgActivation::new(&hexp);
                    let y = lw.down_exps.matvec_rows(e * hidden, hidden, &hexp_q)?;
                    // reference order: down → ×per-expert scale → ×weight,
                    // separate multiplies, slots summed in selection order
                    let s_e = lw.down_exps_scale[e];
                    moe_gate_up_all.extend_from_slice(&gate_up);
                    moe_geglu_all.extend_from_slice(&hexp);
                    moe_down_all.extend_from_slice(&y);
                    moe_down_scaled_all.extend(y.iter().map(|yv| yv * s_e));
                    for (a, yv) in moe_acc.iter_mut().zip(&y) {
                        *a += yv * s_e * w;
                    }
                }
                moe_pre_norm_all.extend_from_slice(&moe_acc);
                let moe_out = refmath::rms_norm(&moe_acc, Some(&lw.post_norm_2), eps);
                ffn_moe_all.extend_from_slice(&moe_out);

                let mut combined = mlp;
                for (c, m) in combined.iter_mut().zip(&moe_out) {
                    *c += m;
                }
                let ffn_out = refmath::rms_norm(&combined, Some(&lw.post_ffw_norm), eps);
                for (a, b) in hp.iter_mut().zip(&ffn_out) {
                    *a += b;
                }
                // ENCODER mode: prompt rows scale by the encoder scalar.
                for v in hp.iter_mut() {
                    *v *= lw.enc_out_scale;
                }
                out_scaled.extend_from_slice(hp);
            }

            let mut k_flat = Vec::with_capacity(n * kv_dim);
            let mut v_flat = Vec::with_capacity(n * kv_dim);
            for pos in 0..n {
                k_flat.extend_from_slice(&ks[pos]);
                v_flat.extend_from_slice(&vs[pos]);
            }
            traces.push(DgLayerTrace {
                k: k_flat,
                v: v_flat,
                attn_out,
                moe_logits: moe_logits_all,
                moe_topk: moe_topk_all,
                out_scaled,
                ffn_mlp: ffn_mlp_all,
                ffn_moe: ffn_moe_all,
                moe_weights: moe_weights_all,
                moe_gate_up: moe_gate_up_all,
                moe_geglu: moe_geglu_all,
                moe_down: moe_down_all,
                moe_down_scaled: moe_down_scaled_all,
                moe_weights_norm: moe_weights_norm_all,
                moe_pre_norm: moe_pre_norm_all,
                ffn_block_out: Vec::new(),
                kqv: Vec::new(),
                kq_soft_max: Vec::new(),
            });
        }

        let mut result_norm_all = Vec::with_capacity(n * hidden);
        for hp in &h {
            result_norm_all.extend_from_slice(&refmath::rms_norm(hp, Some(&self.output_norm), eps));
        }
        let result_norm_last = result_norm_all[(n - 1) * hidden..].to_vec();
        let _ = &result_norm_last;

        Ok(DgEncoderTrace {
            n_pos: n,
            inp_scaled,
            layers: traces,
            result_norm_all,
            result_norm_last,
        })
    }
}

/// Output of one unified (zero-SC) `[prompt | canvas]` forward.
pub struct DgUnifiedOut {
    pub n_prompt: usize,
    pub n_canvas: usize,
    pub n_vocab: usize,
    /// Canvas-row logits, `[C * n_vocab]` row-major — the Phase 3 gate
    /// surface (`llama-diffusion-gemma-eval` writes exactly these rows).
    pub logits: Vec<f32>,
    /// Per-layer checkpoint trace over ALL `P + C` positions (ladder
    /// debugging only; ~1 GB at full canvas — request it explicitly).
    pub trace: Option<DgEncoderTrace>,
}

/// One Entropy-Bound denoiser step's host-math outputs (reference:
/// `diffusion_generate_entropy_bound`, examples/diffusion/diffusion.cpp at
/// the pin — per-position worker, MI-bound acceptance, renoise).
pub struct DgEbStep {
    pub t: f32,
    pub temp_inv: f32,
    pub argmax: Vec<i32>,
    pub entropy: Vec<f32>,
    pub denoiser: Vec<i32>,
    pub accepted: Vec<bool>,
    pub next_canvas: Vec<i32>,
    /// Sequential f32 sum of entropies (the reference's adaptive-stop input).
    pub entropy_sum: f32,
}

/// Raw-pointer Send wrapper for the disjoint-index embT scatter (each
/// worker writes a disjoint set of `[e][v]` slots).
struct ScSendPtr(*mut u16);
unsafe impl Sync for ScSendPtr {}

/// Self-conditioning input for one unified forward: the previous step's RAW
/// canvas logits, the PREVIOUS step's `1/t`, and the {0,1} gate (0 on the EB
/// loop's first step — the SC chain still runs, exactly like the reference
/// graph, so the gated `±0.0` add semantics are preserved).
pub struct DgScInput<'a> {
    pub logits: &'a [f32],
    pub temp_inv: f32,
    pub use_sc: f32,
}

/// Wave 6: step-invariant prompt-side K/V for one EB block (fast mode only).
/// Mask invariance: prompt queries attend causally over the prompt only
/// (`k_pos <= q_pos < p` in the region mask), so prompt hiddens — and hence
/// every layer's prompt K/V — are identical across all denoise steps of a
/// block. Captured at the first step whose fast attention succeeded on EVERY
/// layer; canvas-only steps prepend these to each layer's fresh canvas K/V.
/// Layout per layer: `[p * kv_dim(l)]` f32 — K post k_norm + RoPE, V post
/// v-norm (NO RoPE). kv_dim differs per layer (sliding vs global kv_heads).
pub struct DgPrefixCache {
    /// prompt length these entries were captured at
    p: usize,
    /// per layer `[p * kv_dim(l)]`
    k: Vec<Vec<f32>>,
    /// per layer `[p * kv_dim(l)]`
    v: Vec<Vec<f32>>,
}

/// Internal mode selector for `unified_forward_impl`.
enum DgFwdMode<'a> {
    /// today's full `[prompt | canvas]` pass, bit-identical behavior
    Full,
    /// full pass, additionally capturing prefix K/V per layer into the
    /// candidate cache (buffers written ONLY by a fully-successful
    /// `attn_layer_fast`; empty buffer == that layer fell to CPU)
    FullCapture(&'a mut DgPrefixCache),
    /// canvas-only pass against a sealed cache; returns `Ok(None)` (bail)
    /// if any layer's fast attention fails — caller re-runs the full pass
    Cached(&'a DgPrefixCache),
}

/// Reference EB sampler parameters (`diffusion_eb_params` defaults at pin).
pub struct DgEbParams {
    pub seed: u32,
    pub max_steps: i32,
    pub t_min: f32,
    pub t_max: f32,
    pub entropy_bound: f32,
    pub stability_threshold: i32,
    pub confidence_threshold: f32,
}

impl Default for DgEbParams {
    fn default() -> Self {
        Self {
            seed: 0,
            max_steps: 48,
            t_min: 0.4,
            t_max: 0.8,
            entropy_bound: 0.1,
            stability_threshold: 1,
            confidence_threshold: 0.005,
        }
    }
}

/// One executed EB denoise step's full record (the Phase 4 gate surface).
pub struct DgEbStepRecord {
    pub step_idx: i32,
    pub canvas_in: Vec<i32>,
    pub step: DgEbStep,
    pub finished: bool,
}

impl DgEncoderRuntime {
    /// The SC soft-embedding weight, built once: every `token_embd` row
    /// dequantized (Q6_K, bit-exact) and rounded to f16, stored TRANSPOSED
    /// as `[n_embd rows][n_vocab]` — `dg_ensure_sc_embT` semantics.
    fn sc_emb_t(&self) -> Result<&[u16]> {
        if let Some(t) = self.sc_emb_t.get() {
            return Ok(t);
        }
        let n_vocab = self.token_embd.rows;
        let hidden = self.n_embd;
        let mut t = vec![0u16; n_vocab * hidden];
        let nth = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .min(8);
        let chunk = n_vocab.div_ceil(nth);
        let t_ptr = ScSendPtr(t.as_mut_ptr());
        std::thread::scope(|s| -> Result<()> {
            let mut handles = Vec::new();
            for ci in 0..nth {
                let v0 = ci * chunk;
                let v1 = (v0 + chunk).min(n_vocab);
                if v0 >= v1 {
                    continue;
                }
                let t_ptr = &t_ptr;
                handles.push(s.spawn(move || -> Result<()> {
                    for v in v0..v1 {
                        let row = self.embed_row(v as u32)?;
                        for (e, &val) in row.iter().enumerate() {
                            // transposed scatter: embT[e][v]
                            unsafe {
                                *t_ptr.0.add(e * n_vocab + v) = crate::tensor::f32_to_f16_bits(val);
                            }
                        }
                    }
                    Ok(())
                }));
            }
            for h in handles {
                h.join().expect("embT worker panicked")?;
            }
            Ok(())
        })?;
        Ok(self.sc_emb_t.get_or_init(|| t))
    }

    /// The self-conditioning signal for every canvas position (PRE
    /// `sc_use`-gate): `softmax(prev_logits * temp_inv)` per position, the
    /// f16 soft-embedding matmul scaled by `sqrt(n_embd)`, then the gated
    /// MLP `sc_down(gelu(sc_gate(normed)) * sc_up(normed))` — mirroring
    /// `dg_canvas_embed`'s SC subgraph op for op.
    fn sc_signal(&self, sc: &DgScInput, c: usize) -> Result<Vec<Vec<f32>>> {
        let n_vocab = self.token_embd.rows;
        if sc.logits.len() != c * n_vocab {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "sc logits length {} != C*n_vocab {}",
                sc.logits.len(),
                c * n_vocab
            )));
        }
        let hidden = self.n_embd;
        let eps = self.eps;
        let emb_t = self.sc_emb_t()?;
        let embed_scale = (hidden as f32).sqrt();

        let mut sigs = Vec::with_capacity(c);
        // Phase 5 SC-stage forensics dump (env-gated): concatenated [c][dim]
        // for each stage, compared against the reference's cb'd sc_* tensors.
        let sc_dump = std::env::var("CAMELID_DG_SC_DUMP_DIR").ok();
        let mut d_soft: Vec<f32> = Vec::new();
        let mut d_normed: Vec<f32> = Vec::new();
        let mut d_g: Vec<f32> = Vec::new();
        let mut d_sig: Vec<f32> = Vec::new();

        // FAST mode: fused device path — softmax+f16 straight from the
        // device-resident previous-step logits into the soft matmul (no host
        // softmax, no 134 MB probs upload). Falls through to the plain paths.
        let fused = if dg_fast_enabled() {
            sc_soft_fused(sc.temp_inv, embed_scale, c, hidden, n_vocab)
        } else {
            None
        };
        // probs = softmax(scale(sc_logits, temp_inv)) per position, then the
        // F16 src1 conversion (ggml quantizes src1 rows to the f16 vec_dot
        // type). Gathered for all positions so the soft-embedding matmul can
        // run as one batched GPU kernel. Skipped entirely on the fused path.
        let mut probs_f16_all = Vec::new();
        let soft_all = match fused {
            Some(v) => Some(v),
            None => {
                probs_f16_all = vec![0u16; c * n_vocab];
                for pos in 0..c {
                    let mut probs: Vec<f32> = sc.logits[pos * n_vocab..(pos + 1) * n_vocab]
                        .iter()
                        .map(|&x| x * sc.temp_inv)
                        .collect();
                    refmath::softmax_row(&mut probs);
                    for (v, &p) in probs.iter().enumerate() {
                        probs_f16_all[pos * n_vocab + v] = crate::tensor::f32_to_f16_bits(p);
                    }
                }
                // soft[pos] = (embT @ probs[pos]) * sqrt(n_embd). GPU when
                // available (f32 reduction — NOT bit-identical to the CPU f16
                // emulation); else the reference per-row f16 dot below.
                sc_soft_gpu(emb_t, &probs_f16_all, c, hidden, n_vocab, embed_scale)
            }
        };

        for pos in 0..c {
            let soft: Vec<f32> = match &soft_all {
                Some(all) => all[pos * hidden..(pos + 1) * hidden].to_vec(),
                None => {
                    let probs_f16 = &probs_f16_all[pos * n_vocab..(pos + 1) * n_vocab];
                    let mut soft = vec![0f32; hidden];
                    let nth = std::thread::available_parallelism()
                        .map(|n| n.get())
                        .unwrap_or(1)
                        .min(8);
                    let chunk = hidden.div_ceil(nth);
                    std::thread::scope(|s| {
                        for (ci, ys) in soft.chunks_mut(chunk).enumerate() {
                            s.spawn(move || {
                                for (i, y) in ys.iter_mut().enumerate() {
                                    let e = ci * chunk + i;
                                    *y = reff16::vec_dot_f16(
                                        &emb_t[e * n_vocab..(e + 1) * n_vocab],
                                        probs_f16,
                                    ) * embed_scale;
                                }
                            });
                        }
                    });
                    soft
                }
            };

            // SC gated MLP: pre_norm -> down( gelu(gate) * up )
            let normed = refmath::rms_norm(&soft, Some(&self.sc_pre_norm), eps);
            let nq = DgActivation::new(&normed);
            let g = self.sc_gate.matvec_dense(&nq)?;
            let u = self.sc_up.matvec_dense(&nq)?;
            let h: Vec<f32> = g.iter().zip(&u).map(|(gv, uv)| dg_gelu(*gv) * uv).collect();
            let hq = DgActivation::new(&h);
            let sig = self.sc_down.matvec_dense(&hq)?;
            if sc_dump.is_some() {
                // g here is the PRE-gelu gate matmul; the reference cb's sc_g
                // AFTER gelu — store gelu(g) to match.
                d_soft.extend_from_slice(&soft);
                d_normed.extend_from_slice(&normed);
                d_g.extend(g.iter().map(|&gv| dg_gelu(gv)));
                d_sig.extend_from_slice(&sig);
            }
            sigs.push(sig);
        }
        if let Some(dir) = sc_dump {
            let w = |name: &str, v: &[f32]| {
                let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
                let _ = std::fs::write(format!("{dir}/{name}.bin"), bytes);
            };
            w("cam_sc_soft", &d_soft);
            w("cam_sc_normed", &d_normed);
            w("cam_sc_g", &d_g);
            w("cam_sc_sig", &d_sig);
            eprintln!("[sc-dump] wrote cam_sc_{{soft,normed,g,sig}} to {dir}");
        }
        Ok(sigs)
    }

    /// UNIFIED decode-surface forward (Phase 3): one no-cache bidirectional
    /// pass over `[prompt | canvas]` with zero self-conditioning, mirroring
    /// the reference graph (src/models/diffusion-gemma.cpp at the pin):
    /// canvas embeddings take a weightless rms_norm after the embed scale,
    /// the region mask keeps prompt queries causal (SWA-clipped) over the
    /// prompt only while canvas queries are bidirectional (sliding layers
    /// reach the last `n_swa-1` prompt positions), prompt rows scale by the
    /// encoder per-layer scalar and canvas rows by the decoder scalar, and
    /// the canvas rows finish through the tied lm_head with final-logit
    /// softcapping.
    pub fn unified_forward(
        &self,
        prompt: &[u32],
        canvas: &[u32],
        want_trace: bool,
    ) -> Result<DgUnifiedOut> {
        self.unified_forward_sc(prompt, canvas, None, want_trace)
    }

    /// Unified forward with optional self-conditioning (the EB loop's
    /// per-step decode). With `sc`, the SC subgraph runs exactly as the
    /// reference graph does — including on the gated-off first step, where
    /// `sig * 0.0` adds a signed zero per element.
    pub fn unified_forward_sc(
        &self,
        prompt: &[u32],
        canvas: &[u32],
        sc: Option<&DgScInput>,
        want_trace: bool,
    ) -> Result<DgUnifiedOut> {
        Ok(self
            .unified_forward_impl(prompt, canvas, sc, want_trace, DgFwdMode::Full)?
            .expect("full-mode forward never bails"))
    }

    /// Wave-6 EB-step forward: [`unified_forward_sc`] plus a per-block
    /// prefix-KV cache. FAST MODE ONLY — with `dg_fast_enabled()` false this
    /// delegates verbatim (parity lane untouched). First step (or any step
    /// with no sealed cache) runs the FULL forward, capturing every layer's
    /// prompt K/V; the cache seals only if ALL layers captured (a layer that
    /// fell to the CPU attention loop leaves its buffer empty). Sealed steps
    /// run canvas-only. A cached-mode mid-step fast-attention failure drops
    /// the cache and re-runs the step as a full forward (with re-capture).
    pub fn unified_forward_step(
        &self,
        prompt: &[u32],
        canvas: &[u32],
        sc: Option<&DgScInput>,
        cache: &mut Option<DgPrefixCache>,
    ) -> Result<DgUnifiedOut> {
        if !dg_fast_enabled() {
            return self.unified_forward_sc(prompt, canvas, sc, false);
        }
        let p = prompt.len();
        if let Some(cc) = cache.as_ref().filter(|cc| cc.p == p) {
            if let Some(out) =
                self.unified_forward_impl(prompt, canvas, sc, false, DgFwdMode::Cached(cc))?
            {
                return Ok(out);
            }
            // fast attention failed mid-step: fall through, drop, run full
        }
        *cache = None;
        let mut cand = DgPrefixCache {
            p,
            k: vec![Vec::new(); self.n_layer],
            v: vec![Vec::new(); self.n_layer],
        };
        let out = self
            .unified_forward_impl(prompt, canvas, sc, false, DgFwdMode::FullCapture(&mut cand))?
            .expect("full-mode forward never bails");
        let sealed = (0..self.n_layer).all(|l| {
            let kv = p * self.kv_dim_at(l);
            cand.k[l].len() == kv && cand.v[l].len() == kv
        });
        if sealed {
            *cache = Some(cand);
        }
        Ok(out)
    }

    /// Parameterized body of [`unified_forward_sc`]. `Ok(None)` is the
    /// cached-mode mid-step bail (fast attention failed; the canvas-only `h`
    /// cannot feed the CPU fallback loops) — only `DgFwdMode::Cached` can
    /// return it.
    fn unified_forward_impl(
        &self,
        prompt: &[u32],
        canvas: &[u32],
        sc: Option<&DgScInput>,
        want_trace: bool,
        mode: DgFwdMode<'_>,
    ) -> Result<Option<DgUnifiedOut>> {
        let p = prompt.len();
        let c = canvas.len();
        let n = p + c;
        if p == 0 {
            return Err(BackendError::RuntimeShapeMismatch("empty prompt".into()));
        }
        if c != self.canvas_length {
            // the reference graph splits on the GGUF's canvas_length, so any
            // other canvas size would silently exercise a different region
            // split — fail closed instead
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "canvas length {c} != diffusion.canvas_length {}",
                self.canvas_length
            )));
        }
        let (row0, cache_in, mut capture_out) = match mode {
            DgFwdMode::Full => (0usize, None, None),
            DgFwdMode::FullCapture(cap) => (0usize, None, Some(cap)),
            DgFwdMode::Cached(cc) => (p, Some(cc), None),
        };
        // trace never coexists with the fast-mode variants
        debug_assert!(!(want_trace && (cache_in.is_some() || capture_out.is_some())));
        // number of leading PROMPT rows materialized in h (p full, 0 cached)
        let p_rows = p - row0;
        if let Some(cc) = cache_in {
            if cc.p != p || cc.k.len() != self.n_layer || cc.v.len() != self.n_layer {
                return Ok(None); // stale/malformed cache: caller re-runs full
            }
        }
        let eps = self.eps;
        let hidden = self.n_embd;
        let heads = self.n_head;
        let win = self.g.sliding_window as usize;
        let embed_scale = (hidden as f32).sqrt();

        // self-conditioning signal per canvas position (graph order: the SC
        // subgraph feeds the canvas embedding). When use_sc == 0 (step 0) the
        // signal is added as `sv * 0` below — i.e. discarded — so skip the
        // ~1.9e11-MAC soft-embedding matmul entirely. Bit-identical: every
        // embedding row is left unchanged either way (the reference's step-0
        // graph likewise contributes nothing).
        let _t_sc = std::time::Instant::now();
        let sigs = match sc {
            Some(sc_in) if sc_in.use_sc != 0.0 => Some(self.sc_signal(sc_in, c)?),
            _ => None,
        };
        if sigs.is_some() && std::env::var("CAMELID_DG_STAGE_TIMINGS").is_ok() {
            eprintln!("[dg-prof] sc={}ms", _t_sc.elapsed().as_millis());
        }

        // embeddings: every row scaled by sqrt(n_embd); canvas rows add the
        // gated SC signal (when enabled) and then take the weightless
        // rms_norm. Cached mode materializes ONLY canvas rows: `.skip(row0)`
        // applies after `.enumerate()`, so `pos` stays ABSOLUTE and the body
        // (region test, `sigs[pos - p]`) is byte-identical in both modes.
        let mut h: Vec<Vec<f32>> = Vec::with_capacity(n - row0);
        let mut inp_scaled = Vec::with_capacity(if want_trace { n * hidden } else { 0 });
        for (pos, &tok) in prompt.iter().chain(canvas.iter()).enumerate().skip(row0) {
            let mut e = self.embed_row(tok)?;
            for v in e.iter_mut() {
                *v *= embed_scale;
            }
            if pos >= p {
                if let (Some(sigs), Some(sc_in)) = (&sigs, sc) {
                    // canvas = add(canvas, scale(sc_sig, sc_use)) — the
                    // scale-by-{0,1} multiply runs per element (±0.0 at
                    // step 0), then the add
                    for (ev, sv) in e.iter_mut().zip(&sigs[pos - p]) {
                        *ev += sv * sc_in.use_sc;
                    }
                }
                e = refmath::rms_norm(&e, None, eps);
            }
            if want_trace {
                inp_scaled.extend_from_slice(&e);
            }
            h.push(e);
        }

        // canvas->prompt sliding bound: last (n_swa - 1) prompt positions
        let canvas_prompt_lo = p as i64 - win as i64 + 1;

        let mut traces = Vec::with_capacity(if want_trace { self.n_layer } else { 0 });
        let dg_prof = std::env::var("CAMELID_DG_STAGE_TIMINGS").is_ok();
        let (mut t_qkv, mut t_attn, mut t_ffn) = (0u128, 0u128, 0u128);
        for (l, lw) in self.layers.iter().enumerate() {
            let sliding = self.g.is_sliding_layer(l);
            let head_dim = self.g.head_dim_at(l) as usize;
            let kv_heads = self.g.kv_heads_at(l) as usize;
            let theta = self.g.rope_freq_base_at(l);
            let q_dim = heads * head_dim;
            let kv_dim = kv_heads * head_dim;
            let group = heads / kv_heads;
            let rope_factors = if sliding {
                None
            } else {
                self.rope_factors.as_deref()
            };

            // ---- projections for every position (identical to the encoder
            // path; absolute positions feed RoPE) ----
            let mut qs: Vec<Vec<f32>> = Vec::with_capacity(n);
            let mut ks: Vec<Vec<f32>> = Vec::with_capacity(n);
            let mut vs: Vec<Vec<f32>> = Vec::with_capacity(n);
            let _t_qkv = std::time::Instant::now();
            // FAST mode: whole attention block (projections + masked
            // bidirectional attention + output projection) as batched GPU
            // work; the per-position loops below are skipped on success.
            let prefix_l: Option<(&[f32], &[f32])> =
                cache_in.map(|cc| (cc.k[l].as_slice(), cc.v[l].as_slice()));
            let cap_l: Option<(&mut Vec<f32>, &mut Vec<f32>)> = match &mut capture_out {
                Some(cap) => Some((&mut cap.k[l], &mut cap.v[l])),
                None => None,
            };
            let fast_attn = !want_trace
                && dg_fast_enabled()
                && attn_layer_fast(
                    lw,
                    &mut h,
                    heads,
                    head_dim,
                    kv_heads,
                    group,
                    theta,
                    rope_factors,
                    sliding,
                    win,
                    p,
                    canvas_prompt_lo,
                    eps,
                    row0,
                    prefix_l,
                    cap_l,
                )
                .is_some();
            if !fast_attn && cache_in.is_some() {
                // Cached mode cannot fall back: the CPU attention loops index
                // h as the full [prompt|canvas] context, which a canvas-only
                // h cannot feed. No h mutation has happened for this layer
                // (attn_layer_fast mutates h only after every GPU call
                // succeeded), so aborting the step here is clean.
                return Ok(None);
            }
            let qkv_range = if fast_attn { 0..0 } else { 0..h.len() };
            for pos in qkv_range {
                let hp = &h[pos];
                let xn = refmath::rms_norm(hp, Some(&lw.attn_norm), eps);
                let xq = DgActivation::new(&xn);
                let mut q = lw.attn_q.matvec_dense(&xq)?;
                for hh in 0..heads {
                    let s = &mut q[hh * head_dim..(hh + 1) * head_dim];
                    s.copy_from_slice(&refmath::rms_norm(s, Some(&lw.q_norm), eps));
                }
                refmath::rope_neox(&mut q, heads, head_dim, pos, theta, rope_factors);

                let mut k = lw.attn_k.matvec_dense(&xq)?;
                let mut v = match lw.attn_v.as_ref() {
                    Some(wv) => wv.matvec_dense(&xq)?,
                    None => k.clone(),
                };
                for hh in 0..kv_heads {
                    let s = &mut k[hh * head_dim..(hh + 1) * head_dim];
                    s.copy_from_slice(&refmath::rms_norm(s, Some(&lw.k_norm), eps));
                    let sv = &mut v[hh * head_dim..(hh + 1) * head_dim];
                    sv.copy_from_slice(&refmath::rms_norm(sv, None, eps));
                }
                refmath::rope_neox(&mut k, kv_heads, head_dim, pos, theta, rope_factors);
                qs.push(q);
                ks.push(k);
                vs.push(v);
            }

            t_qkv += _t_qkv.elapsed().as_nanos();
            let _t_attn = std::time::Instant::now();
            // region-aware mask (llm_graph_input_attn_diffusion::set_input):
            // prompt queries causal over the prompt only (SWA-clipped on
            // sliding layers); canvas queries bidirectional — global layers
            // see everything, sliding layers see all canvas plus the last
            // (n_swa - 1) prompt positions
            let allow = |q_pos: usize, k_pos: usize| -> bool {
                if q_pos >= p {
                    if sliding {
                        k_pos >= p || k_pos as i64 >= canvas_prompt_lo
                    } else {
                        true
                    }
                } else {
                    k_pos <= q_pos && (!sliding || k_pos + win > q_pos)
                }
            };

            let mut v_cols: Vec<Vec<f32>> = if fast_attn {
                Vec::new()
            } else {
                vec![vec![0f32; n]; kv_heads * head_dim]
            };
            for (pp, vp) in vs.iter().enumerate() {
                for (di, &val) in vp.iter().enumerate() {
                    v_cols[di][pp] = val;
                }
            }

            let mut attn_out = Vec::with_capacity(if want_trace { n * hidden } else { 0 });
            // Phase 5 KQV capture (layer 0 only): reference "kqv" layout
            // [head_dim, n_q, n_head], index d + q*head_dim + h*head_dim*n_q
            let capture_kqv = want_trace && l == 0;
            let mut kqv_cap = if capture_kqv {
                vec![0f32; head_dim * n * heads]
            } else {
                Vec::new()
            };
            let mut softmax_cap = if capture_kqv {
                vec![0f32; n * n * heads]
            } else {
                Vec::new()
            };
            let attn_range = if fast_attn { 0..0 } else { 0..n };
            for pos in attn_range {
                let mut attn = vec![0f32; q_dim];
                for hh in 0..heads {
                    let kvh = hh / group;
                    let qh = &qs[pos][hh * head_dim..(hh + 1) * head_dim];
                    let mut row: Vec<f32> = (0..n)
                        .map(|kp| {
                            if allow(pos, kp) {
                                let kk = &ks[kp][kvh * head_dim..(kvh + 1) * head_dim];
                                refmath::vec_dot_f32(qh, kk)
                            } else {
                                f32::NEG_INFINITY
                            }
                        })
                        .collect();
                    refmath::softmax_row(&mut row);
                    if capture_kqv {
                        for (kp, &pr) in row.iter().enumerate() {
                            softmax_cap[kp + pos * n + hh * n * n] = pr;
                        }
                    }
                    let out = &mut attn[hh * head_dim..(hh + 1) * head_dim];
                    for (d, o) in out.iter_mut().enumerate() {
                        *o = refmath::vec_dot_f32(&v_cols[kvh * head_dim + d], &row);
                        if capture_kqv {
                            kqv_cap[d + pos * head_dim + hh * head_dim * n] = *o;
                        }
                    }
                }
                let aq = DgActivation::new(&attn);
                let o = lw.attn_output.matvec_dense(&aq)?;
                let on = refmath::rms_norm(&o, Some(&lw.post_attn_norm), eps);
                for (a, b) in h[pos].iter_mut().zip(&on) {
                    *a += b;
                }
                if want_trace {
                    attn_out.extend_from_slice(&h[pos]);
                }
            }

            t_attn += _t_attn.elapsed().as_nanos();
            let _t_ffn = std::time::Instant::now();
            // ---- dense shared-expert MLP + 128-expert MoE (identical math
            // to the encoder path; only the per-layer output scalar is
            // region-aware) ----
            let mut moe_logits_all = Vec::new();
            let mut moe_topk_all = Vec::new();
            let mut out_scaled = Vec::new();
            // FFN sub-chain trace buffers (mirror the encoder forward so the
            // diag actually compares them — they were empty before).
            let mut ffn_mlp_all = Vec::new();
            let mut ffn_moe_all = Vec::new();
            let mut moe_pre_norm_all = Vec::new();
            let mut moe_weights_norm_all = Vec::new();
            let mut moe_gate_up_all = Vec::new();
            let mut moe_geglu_all = Vec::new();
            let mut moe_down_all = Vec::new();
            let mut moe_down_scaled_all = Vec::new();
            let mut ffn_block_out_all = Vec::new();
            // FAST mode: run the whole layer's FFN+MoE as batched GPU GEMMs
            // (weights read once per step, not once per position). Falls back
            // to the per-position loop below on any failure; never taken when
            // tracing (the trace buffers need the per-position intermediates).
            let fast_done = !want_trace
                && dg_fast_enabled()
                && ffn_moe_layer_fast(self, lw, &mut h, p_rows, eps).is_some();
            let pos_range = if fast_done { 0..0 } else { 0..h.len() };
            for pos in pos_range {
                let hp = &mut h[pos];
                let attn_resid = hp.clone();
                let xn = refmath::rms_norm(&attn_resid, Some(&lw.ffn_norm), eps);
                let xq = DgActivation::new(&xn);
                let gate = lw.ffn_gate.matvec_dense(&xq)?;
                let up = lw.ffn_up.matvec_dense(&xq)?;
                let act: Vec<f32> = gate.iter().zip(&up).map(|(g, u)| dg_gelu(*g) * u).collect();
                let actq = DgActivation::new(&act);
                let mlp = lw.ffn_down.matvec_dense(&actq)?;
                let mlp = refmath::rms_norm(&mlp, Some(&lw.post_norm_1), eps);
                if want_trace {
                    ffn_mlp_all.extend_from_slice(&mlp);
                }

                let mut r = refmath::rms_norm(&attn_resid, None, eps);
                let inv = 1.0f32 / (hidden as f32).sqrt();
                for (rv, sv) in r.iter_mut().zip(&lw.gate_inp_scale) {
                    *rv = *rv * inv * sv;
                }
                let logits: Vec<f32> = (0..self.n_expert)
                    .map(|e| refmath::vec_dot_f32(&lw.gate_inp[e * hidden..(e + 1) * hidden], &r))
                    .collect();

                let mut probs: Vec<f32> = logits.clone();
                refmath::softmax_row(&mut probs);
                // expert ORDER must match the reference's ggml_argsort_top_k =
                // libc++ std::sort over DESC router logits (strict `>`; see
                // argsort_desc_experts). Exact ties break by lower index (the
                // reference's introsort tie-order is not portably reproducible).
                let order = argsort_desc_experts(&logits);
                let idx: Vec<usize> = order[..self.n_expert_used]
                    .iter()
                    .map(|&e| e as usize)
                    .collect();
                if want_trace {
                    moe_logits_all.extend_from_slice(&logits);
                    for &e in &idx {
                        moe_topk_all.push(e as i32);
                    }
                }
                let selected: Vec<f32> = idx.iter().map(|&e| probs[e]).collect();
                let mut wsum = refmath::vec_sum_f32(&selected);
                wsum = wsum.max(f32::from_bits(0x3880_0000));

                let cur_moe = refmath::rms_norm(&attn_resid, Some(&lw.pre_norm_2), eps);
                let cur_moe_q = DgActivation::new(&cur_moe);
                let two_nff = 2 * self.n_ff_exp;
                let mut moe_acc = vec![0f32; hidden];
                for &e in &idx {
                    let w = refmath::vdsp_div(probs[e], wsum);
                    let gate_up = lw
                        .gate_up_exps
                        .matvec_rows(e * two_nff, two_nff, &cur_moe_q)?;
                    let hexp: Vec<f32> = (0..self.n_ff_exp)
                        .map(|o| dg_gelu(gate_up[o]) * gate_up[o + self.n_ff_exp])
                        .collect();
                    let hexp_q = DgActivation::new(&hexp);
                    let y = lw.down_exps.matvec_rows(e * hidden, hidden, &hexp_q)?;
                    let s_e = lw.down_exps_scale[e];
                    if want_trace {
                        moe_weights_norm_all.push(w);
                        moe_gate_up_all.extend_from_slice(&gate_up);
                        moe_geglu_all.extend_from_slice(&hexp);
                        moe_down_all.extend_from_slice(&y);
                        moe_down_scaled_all.extend(y.iter().map(|yv| yv * s_e));
                    }
                    for (a, yv) in moe_acc.iter_mut().zip(&y) {
                        *a += yv * s_e * w;
                    }
                }
                if want_trace {
                    moe_pre_norm_all.extend_from_slice(&moe_acc);
                }
                let moe_out = refmath::rms_norm(&moe_acc, Some(&lw.post_norm_2), eps);
                if want_trace {
                    ffn_moe_all.extend_from_slice(&moe_out);
                }

                let mut combined = mlp;
                for (cv, m) in combined.iter_mut().zip(&moe_out) {
                    *cv += m;
                }
                let ffn_out = refmath::rms_norm(&combined, Some(&lw.post_ffw_norm), eps);
                for (a, b) in hp.iter_mut().zip(&ffn_out) {
                    *a += b;
                }
                if want_trace {
                    // pre-scalar FFN block output (== reference's `cur` after
                    // gemma4_build_ffn_moe, before the region scalar)
                    ffn_block_out_all.extend_from_slice(hp);
                }
                // region-aware per-layer scalar: prompt rows take the encoder
                // scalar, canvas rows the decoder scalar. `pos` indexes h,
                // whose leading prompt rows number p_rows (p full, 0 cached).
                let scale = if pos < p_rows {
                    lw.enc_out_scale
                } else {
                    lw.out_scale
                };
                for v in hp.iter_mut() {
                    *v *= scale;
                }
                if want_trace {
                    out_scaled.extend_from_slice(hp);
                }
            }

            t_ffn += _t_ffn.elapsed().as_nanos();
            if want_trace {
                let mut k_flat = Vec::with_capacity(n * kv_dim);
                let mut v_flat = Vec::with_capacity(n * kv_dim);
                for pos in 0..n {
                    k_flat.extend_from_slice(&ks[pos]);
                    v_flat.extend_from_slice(&vs[pos]);
                }
                traces.push(DgLayerTrace {
                    k: k_flat,
                    v: v_flat,
                    attn_out,
                    moe_logits: moe_logits_all,
                    moe_topk: moe_topk_all,
                    out_scaled,
                    ffn_mlp: ffn_mlp_all,
                    ffn_moe: ffn_moe_all,
                    moe_weights: Vec::new(),
                    moe_gate_up: moe_gate_up_all,
                    moe_geglu: moe_geglu_all,
                    moe_down: moe_down_all,
                    moe_down_scaled: moe_down_scaled_all,
                    moe_weights_norm: moe_weights_norm_all,
                    moe_pre_norm: moe_pre_norm_all,
                    ffn_block_out: ffn_block_out_all,
                    kqv: kqv_cap,
                    kq_soft_max: softmax_cap,
                });
            }
        }

        // final norm on every row; lm_head (tied Q6_K token embedding) +
        // final-logit softcapping on the CANVAS rows only (the gate surface)
        let _t_lm = std::time::Instant::now();
        let n_vocab = self.token_embd.rows;
        let mut result_norm_all = Vec::with_capacity(if want_trace { n * hidden } else { 0 });
        let rns: Vec<Vec<f32>> = h
            .iter()
            .map(|hp| refmath::rms_norm(hp, Some(&self.output_norm), eps))
            .collect();
        if want_trace {
            for rn in &rns {
                result_norm_all.extend_from_slice(rn);
            }
        }
        // Canvas rows finish through the tied Q6_K lm_head. FAST mode runs a
        // tiled f16 GEMM against the resident SC embedding transpose (shared
        // VRAM slot, ~100x fewer weight reads); otherwise quantize each canvas
        // activation to Q8_K and run one batched GPU GEMV (bit-close to the
        // CPU q6_k_dot) when available, else the per-position CPU matvec.
        let fast_logits = if !want_trace && dg_fast_enabled() {
            lm_head_fast_gemm(self, &rns[p_rows..], c, hidden)
        } else {
            None
        };
        // The fast GEMM fuses the softcap into its store; do not re-apply.
        let fast_capped = fast_logits.is_some();
        let canvas_acts: Vec<DgActivation> = if fast_capped {
            Vec::new()
        } else {
            rns[p_rows..]
                .iter()
                .map(|rn| DgActivation::new(rn))
                .collect()
        };
        let gpu_logits = match fast_logits {
            Some(v) => Some(v),
            None => lm_head_gpu(&self.token_embd, &canvas_acts, c, hidden),
        };
        let mut logits = Vec::with_capacity(c * n_vocab);
        let softcap = |row: &mut [f32]| {
            if let Some(cap) = self.final_logit_softcapping {
                // reference: scale(1/cap) -> tanh -> scale(cap); the reciprocal
                // is computed in f32 at graph build
                let inv_cap = 1.0f32 / cap;
                for v in row.iter_mut() {
                    *v = refmath::libm_tanhf(*v * inv_cap) * cap;
                }
            }
        };
        match gpu_logits {
            Some(all) => {
                if fast_capped {
                    logits = all;
                } else {
                    for pos in 0..c {
                        let mut row = all[pos * n_vocab..(pos + 1) * n_vocab].to_vec();
                        softcap(&mut row);
                        logits.extend_from_slice(&row);
                    }
                }
            }
            None => {
                for rq in &canvas_acts {
                    let mut row = self.token_embd.matvec_dense(rq)?;
                    softcap(&mut row);
                    logits.extend_from_slice(&row);
                }
            }
        }
        if dg_prof {
            eprintln!(
                "[dg-prof] qkv={}ms attn={}ms ffn+moe={}ms lm_head={}ms (n={n} c={c})",
                t_qkv / 1_000_000,
                t_attn / 1_000_000,
                t_ffn / 1_000_000,
                _t_lm.elapsed().as_millis(),
            );
        }

        let trace = want_trace.then(|| DgEncoderTrace {
            n_pos: n,
            inp_scaled,
            layers: traces,
            result_norm_last: result_norm_all[(n - 1) * hidden..].to_vec(),
            result_norm_all,
        });

        Ok(Some(DgUnifiedOut {
            n_prompt: p,
            n_canvas: c,
            n_vocab,
            logits,
            trace,
        }))
    }

    /// One Entropy-Bound denoiser step's host math, transcribed from the
    /// reference sampler (diffusion.cpp at the pin): per position the argmax
    /// of `logits/t`, the entropy of `softmax(logits/t)`, and a multinomial
    /// draw via the pre-drawn `u`; then the MI-bound acceptance (lowest
    /// entropies whose STRICTLY-EARLIER cumulative sum stays within the
    /// bound, double accumulator) and the renoise rule (accepted -> sampled
    /// token, rest -> the pre-drawn fresh random token). `step_idx`/`s` set
    /// the linear temperature schedule (`cur_step = s - step_idx`).
    ///
    /// Tie caveat: the reference orders positions with `std::sort` on
    /// entropy; equal entropies across positions land in unspecified order,
    /// and an acceptance boundary INSIDE such a tie group is the one case
    /// where this port (sort_unstable_by) could legally differ.
    #[allow(clippy::too_many_arguments)]
    pub fn eb_step(
        logits: &[f32],
        n_vocab: usize,
        step_idx: i32,
        s: i32,
        t_min: f32,
        t_max: f32,
        entropy_bound: f32,
        u: &[f32],
        renoise: &[i32],
    ) -> DgEbStep {
        let c = u.len();
        debug_assert_eq!(logits.len(), c * n_vocab);
        debug_assert_eq!(renoise.len(), c);
        let cur_step = s - step_idx;
        // the reference's `t_min + (t_max - t_min) * ratio` is one expression
        // and contracts to fma under clang's default -ffp-contract=on
        let t = (t_max - t_min).mul_add(cur_step as f32 / s as f32, t_min);
        let temp_inv = 1.0f32 / t;

        let mut argmax = vec![0i32; c];
        let mut entropy = vec![0f32; c];
        let mut denoiser = vec![0i32; c];
        // Row-parallel: every position's argmax/entropy/CDF walk is a fully
        // independent serial chain over its own 262K logits, so distributing
        // POSITIONS across threads is bit-identical to the serial loop (same
        // per-row float order; each output written once). Single-threaded
        // this loop was ~1.1s/step.
        {
            use rayon::prelude::*;
            argmax
                .par_iter_mut()
                .zip(entropy.par_iter_mut().zip(denoiser.par_iter_mut()))
                .enumerate()
                .for_each(|(pos, (amax_out, (hent_out, samp_out)))| {
                    let row = &logits[pos * n_vocab..(pos + 1) * n_vocab];
                    let mut m = f32::NEG_INFINITY;
                    let mut amax = 0i32;
                    for (v, &x) in row.iter().enumerate() {
                        let z = x * temp_inv;
                        if z > m {
                            m = z;
                            amax = v as i32;
                        }
                    }
                    // the reference's `expf(row[v] * temp_inv - m)` argument
                    // and its `H -= p * logf(p)` update both CONTRACT under
                    // clang's default -ffp-contract=on (fmadd / fmsub in the
                    // oracle's disassembly) — mirror the single-rounding forms
                    let neg_m = -m;
                    let mut zsum = 0.0f32;
                    for &x in row {
                        zsum += refmath::libm_expf(x.mul_add(temp_inv, neg_m));
                    }
                    let target = u[pos] * zsum;
                    let mut cum = 0.0f32;
                    let mut hent = 0.0f32;
                    let mut sampled = (n_vocab - 1) as i32;
                    let mut picked = false;
                    for (v, &x) in row.iter().enumerate() {
                        let e = refmath::libm_expf(x.mul_add(temp_inv, neg_m));
                        let pr = e / zsum;
                        if pr > 0.0 {
                            hent = (-pr).mul_add(refmath::libm_logf(pr), hent);
                        }
                        cum += e;
                        if !picked && cum >= target {
                            sampled = v as i32;
                            picked = true;
                        }
                    }
                    *amax_out = amax;
                    *hent_out = hent;
                    *samp_out = sampled;
                });
        }

        // MI-bound position order: match the reference's `std::sort(order,
        // entropy[a] < entropy[b])` (libc++ unstable tie order) so an exact
        // entropy tie accepts the same positions — same fix class as the
        // expert argsort (Rust sort_unstable would break ties differently).
        let order: Vec<usize> = argsort_asc_libcpp(&entropy)
            .into_iter()
            .map(|i| i as usize)
            .collect();
        let mut accepted = vec![false; c];
        let mut cum_e = 0f64;
        for &pos in &order {
            cum_e += entropy[pos] as f64;
            if cum_e - entropy[pos] as f64 <= entropy_bound as f64 {
                accepted[pos] = true;
            }
        }

        let mut next_canvas = vec![0i32; c];
        let mut entropy_sum = 0.0f32;
        for pos in 0..c {
            next_canvas[pos] = if accepted[pos] {
                denoiser[pos]
            } else {
                renoise[pos]
            };
            entropy_sum += entropy[pos];
        }

        DgEbStep {
            t,
            temp_inv,
            argmax,
            entropy,
            denoiser,
            accepted,
            next_canvas,
            entropy_sum,
        }
    }

    /// The full Entropy-Bound denoise loop
    /// (`diffusion_generate_entropy_bound` at the pin, default unified
    /// no-KV-cache path): canvas random-init from the seed, per step one
    /// unified forward with self-conditioning (gated off on step 0, then
    /// `softmax(prev raw logits * prev 1/t)`), the per-position worker, the
    /// MI-bound acceptance + renoise, and the adaptive stop (argmax stable
    /// for `stability_threshold` steps AND mean entropy below
    /// `confidence_threshold`). `on_step` observes each executed step's
    /// record and the step's raw canvas logits.
    pub fn eb_generate(
        &self,
        prompt: &[u32],
        params: &DgEbParams,
        mut on_step: impl FnMut(&DgEbStepRecord, &[f32]),
    ) -> Result<Vec<DgEbStepRecord>> {
        let n_vocab = self.token_embd.rows;
        let c = self.canvas_length;
        let s = params.max_steps.max(1);
        // Generation boundary: never let a previous generation's device
        // logits feed this one's self-conditioning (audit F5).
        #[cfg(feature = "cuda")]
        cuda::dg_generation_reset();
        let draws = refrng::eb_draws(params.seed, n_vocab as i32, c, s as usize);

        let mut current_canvas: Vec<i32> = draws.canvas_init.clone();
        let mut sc_buffer = vec![0f32; c * n_vocab];
        let mut prev_temp_inv = 1.0f32;
        let mut prev_argmax: Vec<i32> = vec![-1; c];
        let mut held = 0i32;
        let mut records = Vec::new();
        // Wave 6 prefix-KV step cache: prompt K/V are step-invariant within
        // one EB block, so the first fully-fast step captures them and later
        // steps run canvas-only forwards. Local to this call => rebuilt per
        // block (mc_generate grows the prefix between eb_generate calls).
        let mut prefix_cache: Option<DgPrefixCache> = None;

        // Diagnostic-only executed-step cap (does NOT alter the temperature
        // schedule, which is driven by `s`): stop after this many steps so the
        // block-1 logit ladder runs a few steps instead of the full loop.
        let exec_cap: Option<usize> = std::env::var("CAMELID_DG_EB_CAP")
            .ok()
            .and_then(|v| v.parse().ok());

        for cur_step in (1..=s).rev() {
            let step_idx = s - cur_step;
            let canvas_u32: Vec<u32> = current_canvas.iter().map(|&v| v as u32).collect();
            let sc_in = DgScInput {
                logits: &sc_buffer,
                temp_inv: prev_temp_inv,
                use_sc: if step_idx == 0 { 0.0 } else { 1.0 },
            };
            let out =
                self.unified_forward_step(prompt, &canvas_u32, Some(&sc_in), &mut prefix_cache)?;
            sc_buffer.copy_from_slice(&out.logits);

            let _t_eb = std::time::Instant::now();
            let step = Self::eb_step(
                &out.logits,
                n_vocab,
                step_idx,
                s,
                params.t_min,
                params.t_max,
                params.entropy_bound,
                &draws.u[step_idx as usize],
                &draws.renoise[step_idx as usize],
            );
            if std::env::var("CAMELID_DG_STAGE_TIMINGS").is_ok() {
                eprintln!("[dg-prof] eb_step={}ms", _t_eb.elapsed().as_millis());
            }

            current_canvas = step.next_canvas.clone();
            held = if prev_argmax == step.argmax {
                held + 1
            } else {
                0
            };
            let confident = step.entropy_sum / (c as f32) < params.confidence_threshold;
            let finished = held >= params.stability_threshold && confident;
            prev_argmax = step.argmax.clone();
            prev_temp_inv = step.temp_inv;

            let record = DgEbStepRecord {
                step_idx,
                canvas_in: canvas_u32.iter().map(|&v| v as i32).collect(),
                step,
                finished,
            };
            on_step(&record, &out.logits);
            records.push(record);
            if finished {
                break;
            }
            if let Some(cap) = exec_cap {
                if records.len() >= cap {
                    break;
                }
            }
        }
        Ok(records)
    }

    /// `trim_canvas` (diffusion-cli.cpp at the pin): cut at the first
    /// end-of-generation token, or at the onset of a repetition loop (a
    /// token recurring at stride 1-2 for >= 6 steps). NOTE the repetition
    /// scan runs over the FULL canvas length even when an EOG cut already
    /// shortened `cut` — reference behavior, kept verbatim.
    pub fn trim_canvas(canvas: &[i32], eog: &std::collections::HashSet<i32>) -> usize {
        let n = canvas.len();
        let mut cut = n;
        for (i, t) in canvas.iter().enumerate() {
            if eog.contains(t) {
                cut = i;
                break;
            }
        }
        let mut i = 0;
        while i + 1 < cut {
            let mut found_loop = false;
            for stride in 1..=2usize {
                if found_loop {
                    break;
                }
                let mut reps = 0;
                let mut j = i;
                while j + stride < n && canvas[j] == canvas[j + stride] {
                    reps += 1;
                    j += stride;
                }
                found_loop = reps >= 6;
            }
            if found_loop {
                cut = i;
                break;
            }
            i += 1;
        }
        cut
    }

    /// The multi-canvas (block-autoregressive) loop — diffusion-cli.cpp's
    /// `run_turn` canvas path at the pin: per block one full EB denoise of
    /// `[prefix | canvas]` (the rng RE-SEEDS with the same seed each block,
    /// as in the reference where it is local to the EB function), then
    /// trim; a partial cut (end token / repetition loop) completes the
    /// answer, a full canvas commits to the prefix; the ubatch budget guard
    /// stops when `[prefix | canvas]` no longer fits. Returns the per-block
    /// (final canvas, cut) pairs, the trimmed response tokens, and the stop
    /// reason.
    #[allow(clippy::type_complexity)]
    pub fn mc_generate(
        &self,
        prompt: &[u32],
        params: &DgEbParams,
        n_blocks: i32,
        max_ubatch: i32,
        eog: &std::collections::HashSet<i32>,
        on_block: impl FnMut(usize, &[u32], &[DgEbStepRecord], &[i32], usize),
    ) -> Result<(Vec<(Vec<i32>, usize)>, Vec<i32>, String)> {
        self.mc_generate_with_steps(
            prompt,
            params,
            n_blocks,
            max_ubatch,
            eog,
            |_, _, _| {},
            on_block,
        )
    }

    /// [`mc_generate`] with a PER-STEP observer: `on_step(block, step_idx,
    /// argmax_canvas)` fires after every denoise step — the live-preview
    /// source (a diffusion model's full draft canvas exists from the first
    /// step and refines in place). Identical generation by construction.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::type_complexity)]
    pub fn mc_generate_with_steps(
        &self,
        prompt: &[u32],
        params: &DgEbParams,
        n_blocks: i32,
        max_ubatch: i32,
        eog: &std::collections::HashSet<i32>,
        mut on_step: impl FnMut(usize, i32, &[i32]),
        mut on_block: impl FnMut(usize, &[u32], &[DgEbStepRecord], &[i32], usize),
    ) -> Result<(Vec<(Vec<i32>, usize)>, Vec<i32>, String)> {
        let c = self.canvas_length;
        let mut prefix: Vec<u32> = prompt.to_vec();
        let mut response: Vec<i32> = Vec::new();
        let mut blocks: Vec<(Vec<i32>, usize)> = Vec::new();
        let mut stop_reason = "blocks";

        for b in 0..n_blocks.max(1) {
            let prefix_len = prefix.len();
            let max_length = prefix_len + c;
            if max_length > max_ubatch as usize {
                if b == 0 {
                    return Err(BackendError::RuntimeShapeMismatch(format!(
                        "[prompt | canvas] needs one ubatch: {prefix_len} + {c} > {max_ubatch}"
                    )));
                }
                stop_reason = "ubatch";
                break;
            }

            let records = self.eb_generate(&prefix, params, |rec, _| {
                on_step(b as usize, rec.step_idx, &rec.step.argmax)
            })?;
            let final_canvas: Vec<i32> =
                records
                    .last()
                    .map(|r| r.step.argmax.clone())
                    .ok_or_else(|| {
                        BackendError::RuntimeShapeMismatch("EB loop produced no steps".into())
                    })?;
            let cut = Self::trim_canvas(&final_canvas, eog);
            on_block(b as usize, &prefix, &records, &final_canvas, cut);

            response.extend_from_slice(&final_canvas[..cut]);
            let full = cut == c;
            blocks.push((final_canvas.clone(), cut));
            if !full {
                stop_reason = "trim";
                break;
            }
            prefix.extend(final_canvas[..cut].iter().map(|&t| t as u32));
        }
        Ok((blocks, response, stop_reason.to_string()))
    }
}

/// Re-export the metadata so the parity test can sanity-check shapes.
impl DgEncoderRuntime {
    pub fn n_layer(&self) -> usize {
        self.n_layer
    }
    pub fn kv_dim_at(&self, l: usize) -> usize {
        self.g.kv_heads_at(l) as usize * self.g.head_dim_at(l) as usize
    }
    pub fn n_embd(&self) -> usize {
        self.n_embd
    }
    pub fn n_expert(&self) -> usize {
        self.n_expert
    }
    pub fn n_vocab(&self) -> usize {
        self.token_embd.rows
    }
    pub fn n_expert_used(&self) -> usize {
        self.n_expert_used
    }
}

/// Loading any non-diffusion file through this runtime must fail closed.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::excessive_precision)]
    fn dg_gelu_mirrors_reference_table_semantics() {
        // clamps
        assert_eq!(dg_gelu(-10.0), 0.0);
        assert_eq!(dg_gelu(-12.5), 0.0);
        assert_eq!(dg_gelu(10.0), 10.0);
        assert_eq!(dg_gelu(42.5), 42.5);
        // interior: result is the tanh approximation evaluated at the
        // f16-rounded input, rounded to f16 (both roundings observable)
        let x = 0.123_456_7f32;
        let v = crate::tensor::f16_round(x);
        assert_ne!(v, x, "test value must not be f16-representable");
        let expect = crate::tensor::f16_round(
            0.5 * v * (1.0 + (0.797_884_56f32 * v * (1.0 + 0.044_715 * v * v)).tanh()),
        );
        assert_eq!(dg_gelu(x).to_bits(), expect.to_bits());
        // f16 quantization is observable: two nearby inputs that round to
        // the same f16 produce the identical output
        let y = x + 1e-6;
        assert_eq!(dg_gelu(x).to_bits(), dg_gelu(y).to_bits());
    }

    #[test]
    fn dg_argsort_orders_and_breaks_ties_by_index() {
        // DESC: strictly decreasing key order; the comparator matches the
        // reference's ggml_argsort_top_k (`keys[a] > keys[b]`).
        let keys = [0.1f32, 0.9, 0.5, 0.3];
        assert_eq!(argsort_desc_experts(&keys), vec![1, 2, 3, 0]);
        // ASC: strictly increasing key order (EB MI-bound position ordering).
        assert_eq!(argsort_asc_libcpp(&keys), vec![0, 3, 2, 1]);
        // Exact ties resolve by lower index, deterministically, in both orders.
        let tied = [0.5f32, 0.5, 0.2, 0.5];
        assert_eq!(argsort_desc_experts(&tied), vec![0, 1, 3, 2]);
        assert_eq!(argsort_asc_libcpp(&tied), vec![2, 0, 1, 3]);
        // Pure-Rust path is deterministic: same input -> same order every call.
        assert_eq!(argsort_desc_experts(&tied), argsort_desc_experts(&tied));
    }

    #[test]
    fn dg_format_fails_closed_on_unproven_types() {
        let err = DgFormat::from_tensor_type(GgufTensorType::Q4_0, "blk.0.test").unwrap_err();
        assert!(matches!(err, BackendError::UnsupportedTensorType(_)));
        let err = DgFormat::from_tensor_type(GgufTensorType::F16, "blk.0.test").unwrap_err();
        assert!(matches!(err, BackendError::UnsupportedTensorType(_)));
    }

    /// Phase 5 diagnostic: compare camelid's scaled prompt embeddings
    /// against an oracle `inp_region` dump (env CAMELID_DG_EMB_GGUF /
    /// CAMELID_DG_EMB_IDS / CAMELID_DG_EMB_REF). Localizes whether the story
    /// block-0 divergence is already in the token embeddings (high token
    /// ids never used as direct prompt embeddings before).
    #[test]
    fn dg_prompt_embedding_diag() {
        let (Ok(g), Ok(i), Ok(r)) = (
            std::env::var("CAMELID_DG_EMB_GGUF"),
            std::env::var("CAMELID_DG_EMB_IDS"),
            std::env::var("CAMELID_DG_EMB_REF"),
        ) else {
            eprintln!("skipping: CAMELID_DG_EMB_* not set");
            return;
        };
        let prompt: Vec<u32> = std::fs::read(&i)
            .unwrap()
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as u32)
            .collect();
        let inp_region: Vec<f32> = std::fs::read(&r)
            .unwrap()
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let rt = DgEncoderRuntime::load(Path::new(&g)).expect("load");
        let h = rt.n_embd;
        let scale = (h as f32).sqrt();
        // prompt rows are the first P rows of inp_region (canvas rows are
        // rms-normed; prompt rows are just scaled embeddings)
        for (pos, &tok) in prompt.iter().enumerate() {
            let mut e = rt.embed_row(tok).unwrap();
            for v in e.iter_mut() {
                *v *= scale;
            }
            let refrow = &inp_region[pos * h..(pos + 1) * h];
            let bad = e
                .iter()
                .zip(refrow)
                .filter(|(a, b)| a.to_bits() != b.to_bits())
                .count();
            let maxabs = e
                .iter()
                .zip(refrow)
                .map(|(a, b)| (a - b).abs())
                .fold(0f32, f32::max);
            if bad > 0 {
                eprintln!(
                    "EMB DIAG pos {pos} tok {tok}: {bad}/{h} not bit-exact, maxabs={maxabs:.3e}"
                );
            }
        }
        eprintln!("EMB DIAG done ({} prompt tokens)", prompt.len());
    }
}
