//! Reference-order fp16 arithmetic for the self-conditioning soft-embedding
//! matmul (`ggml_vec_dot_f16`, vec.cpp — the
//! `__ARM_FEATURE_FP16_VECTOR_ARITHMETIC` NEON branch the pinned reference
//! executes): 4 accumulators of 8 fp16 lanes, `vfmaq_f16` per lane (FUSED
//! fp16 multiply-add, ONE rounding to f16), an f16 `vaddq` reduce tree, f32
//! lane conversion with the `vaddvq` pairwise horizontal add, and a double
//! total. Kernel structure (c) the llama.cpp / ggml authors.
//!
//! The portable implementation emulates fp16 ops exactly: both operands of
//! every op are converted to f64 (exact — fp16 products need 22 mantissa
//! bits, fp16 sums align within 40), computed in f64, and rounded ONCE to
//! f16 nearest-even. Double rounding f64→f16 is innocuous because
//! 53 >= 2*11 + 2. On aarch64 a `fmla v.8h` inline-asm fast path runs the
//! literal instructions; a unit test pins both against hardware ground
//! truth from `scripts/dg-f16-dump.cpp`.

// On aarch64 the emulation chain is exercised by the hardware-parity tests
// only (the asm fast path serves the runtime); elsewhere it IS the runtime.
/// Exact f16 → f64 (all f16 values are exactly representable).
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
#[inline]
pub(crate) fn f16_to_f64(bits: u16) -> f64 {
    crate::tensor::f16_bits_to_f32(bits) as f64
}

/// Round an f64 to f16 with IEEE round-to-nearest-even (subnormals,
/// overflow→inf). The single rounding step of the emulated fp16 ops.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
pub(crate) fn f64_to_f16_rne(x: f64) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 48) & 0x8000) as u16;
    let abs = bits & 0x7fff_ffff_ffff_ffff;
    if abs == 0 {
        return sign;
    }
    let biased = (abs >> 52) as i64;
    let mant = abs & 0xf_ffff_ffff_ffff;
    if biased == 0x7ff {
        return if mant != 0 {
            sign | 0x7e00
        } else {
            sign | 0x7c00
        };
    }
    // f64 subnormals (and anything below half the smallest f16 subnormal)
    // round to zero; the tie/near-tie band is handled by the general path
    let exp = biased - 1023;
    if exp >= 16 {
        return sign | 0x7c00; // certain overflow
    }
    if exp >= -14 {
        // normal f16: round the 52-bit mantissa to 10 bits
        let shift = 42u32;
        let keep = (mant >> shift) as u16;
        let rem = mant & ((1u64 << shift) - 1);
        let half = 1u64 << (shift - 1);
        let mut h = ((((exp + 15) as u16) << 10) | keep) as u32;
        if rem > half || (rem == half && (keep & 1) == 1) {
            h += 1; // a mantissa carry rolls into the exponent (up to inf)
        }
        return sign | h as u16;
    }
    if exp <= -26 || biased == 0 {
        return sign; // below half of the smallest subnormal
    }
    // subnormal f16: significand 1.mant scaled by 2^(exp+14) into 10 bits
    let shift = (42 + (-14 - exp)) as u32; // 43..=53
    let full = mant | (1u64 << 52);
    let keep = (full >> shift) as u16;
    let rem = full & ((1u64 << shift) - 1);
    let half = 1u64 << (shift - 1);
    let mut h = keep as u32;
    if rem > half || (rem == half && (keep & 1) == 1) {
        h += 1;
    }
    sign | h as u16
}

/// `vfmaq_f16` lane semantics: round(a*b + acc) with ONE rounding to f16.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
#[inline]
pub(crate) fn fma_f16(acc: u16, a: u16, b: u16) -> u16 {
    f64_to_f16_rne(f16_to_f64(a) * f16_to_f64(b) + f16_to_f64(acc))
}

/// `vaddq_f16` lane semantics: round(a + b) to f16.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
#[inline]
pub(crate) fn add_f16(a: u16, b: u16) -> u16 {
    f64_to_f16_rne(f16_to_f64(a) + f16_to_f64(b))
}

/// `ggml_vec_dot_f16` — portable emulation of the NEON branch.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
pub(crate) fn vec_dot_f16_emu(x: &[u16], y: &[u16]) -> f32 {
    debug_assert_eq!(x.len(), y.len());
    let n = x.len();
    let np = n & !31;
    let mut sum = [[0u16; 8]; 4];
    let mut i = 0;
    while i < np {
        for (j, accj) in sum.iter_mut().enumerate() {
            let o = i + j * 8;
            for (l, acc) in accj.iter_mut().enumerate() {
                *acc = fma_f16(*acc, x[o + l], y[o + l]);
            }
        }
        i += 32;
    }
    // GGML_F16x8_REDUCE: f16 lane adds, then f32 conversion + vaddvq
    #[allow(clippy::needless_range_loop)]
    for l in 0..8 {
        sum[0][l] = add_f16(sum[0][l], sum[2][l]);
        sum[1][l] = add_f16(sum[1][l], sum[3][l]);
        sum[0][l] = add_f16(sum[0][l], sum[1][l]);
    }
    let s: Vec<f32> = sum[0]
        .iter()
        .map(|&b| crate::tensor::f16_bits_to_f32(b))
        .collect();
    // vaddq_f32(t0, t1) then vaddvq_f32: (l0+l1) + (l2+l3) of the lane sums
    let t = [s[0] + s[4], s[1] + s[5], s[2] + s[6], s[3] + s[7]];
    let mut sumf = ((t[0] + t[1]) + (t[2] + t[3])) as f64;
    for k in np..n {
        sumf +=
            (crate::tensor::f16_bits_to_f32(x[k]) * crate::tensor::f16_bits_to_f32(y[k])) as f64;
    }
    sumf as f32
}

/// aarch64 fast path: the literal `fmla v.8h` kernel (identical instructions
/// to the reference). Falls back to the emulation elsewhere.
#[cfg(target_arch = "aarch64")]
pub(crate) fn vec_dot_f16(x: &[u16], y: &[u16]) -> f32 {
    debug_assert_eq!(x.len(), y.len());
    let n = x.len();
    let np = n & !31;
    let mut lanes = [0u16; 8];
    if np > 0 {
        unsafe {
            core::arch::asm!(
                // v0..v3 = f16 accumulators
                "movi v0.8h, #0",
                "movi v1.8h, #0",
                "movi v2.8h, #0",
                "movi v3.8h, #0",
                "2:",
                "ldp q4, q5, [{x}], #32",
                "ldp q6, q7, [{y}], #32",
                "fmla v0.8h, v4.8h, v6.8h",
                "fmla v1.8h, v5.8h, v7.8h",
                "ldp q4, q5, [{x}], #32",
                "ldp q6, q7, [{y}], #32",
                "fmla v2.8h, v4.8h, v6.8h",
                "fmla v3.8h, v5.8h, v7.8h",
                "subs {cnt}, {cnt}, #32",
                "b.ne 2b",
                // reduce: sum0+=sum2, sum1+=sum3, sum0+=sum1 (f16 adds)
                "fadd v0.8h, v0.8h, v2.8h",
                "fadd v1.8h, v1.8h, v3.8h",
                "fadd v0.8h, v0.8h, v1.8h",
                "str q0, [{out}]",
                x = inout(reg) x.as_ptr() => _,
                y = inout(reg) y.as_ptr() => _,
                cnt = inout(reg) np => _,
                out = in(reg) lanes.as_mut_ptr(),
                out("v0") _, out("v1") _, out("v2") _, out("v3") _,
                out("v4") _, out("v5") _, out("v6") _, out("v7") _,
            );
        }
    }
    let s: Vec<f32> = lanes
        .iter()
        .map(|&b| crate::tensor::f16_bits_to_f32(b))
        .collect();
    let t = [s[0] + s[4], s[1] + s[5], s[2] + s[6], s[3] + s[7]];
    let mut sumf = ((t[0] + t[1]) + (t[2] + t[3])) as f64;
    for k in np..n {
        sumf +=
            (crate::tensor::f16_bits_to_f32(x[k]) * crate::tensor::f16_bits_to_f32(y[k])) as f64;
    }
    sumf as f32
}

#[cfg(not(target_arch = "aarch64"))]
pub(crate) fn vec_dot_f16(x: &[u16], y: &[u16]) -> f32 {
    vec_dot_f16_emu(x, y)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn f16_dir() -> Option<PathBuf> {
        std::env::var("CAMELID_DG_F16_REF").ok().map(PathBuf::from)
    }

    /// Hardware ground truth (scripts/dg-f16-dump.cpp): every emulated FMA
    /// must equal the vfmaq_f16 lane result bit-for-bit.
    #[test]
    fn fma_f16_matches_hardware() {
        let Some(dir) = f16_dir() else {
            eprintln!("skipping: CAMELID_DG_F16_REF not set");
            return;
        };
        let raw = std::fs::read(dir.join("f16-fma.bin")).expect("f16-fma.bin");
        let mut n = 0usize;
        for rec in raw.chunks_exact(8) {
            let g = |i: usize| u16::from_le_bytes([rec[i * 2], rec[i * 2 + 1]]);
            let (a, b, c, want) = (g(0), g(1), g(2), g(3));
            let got = fma_f16(c, a, b);
            assert_eq!(
                got, want,
                "fma_f16 mismatch: a={a:#06x} b={b:#06x} c={c:#06x} got={got:#06x} want={want:#06x}"
            );
            n += 1;
        }
        assert!(n > 0);
    }

    #[test]
    fn add_f16_matches_hardware() {
        let Some(dir) = f16_dir() else {
            eprintln!("skipping: CAMELID_DG_F16_REF not set");
            return;
        };
        let raw = std::fs::read(dir.join("f16-add.bin")).expect("f16-add.bin");
        for rec in raw.chunks_exact(6) {
            let g = |i: usize| u16::from_le_bytes([rec[i * 2], rec[i * 2 + 1]]);
            let (a, b, want) = (g(0), g(1), g(2));
            assert_eq!(
                add_f16(a, b),
                want,
                "add_f16 mismatch: a={a:#06x} b={b:#06x}"
            );
        }
    }

    /// Full-dot parity: emulation AND the aarch64 asm path against the
    /// hardware kernel transcription, across lengths incl. the SC matmul's
    /// 262144 and a leftover-bearing 100.
    #[test]
    fn vec_dot_f16_matches_hardware() {
        let Some(dir) = f16_dir() else {
            eprintln!("skipping: CAMELID_DG_F16_REF not set");
            return;
        };
        for n in [32usize, 64, 100, 4096, 262144] {
            let raw = std::fs::read(dir.join(format!("f16-dot-{n}.bin"))).expect("dot file");
            assert_eq!(raw.len(), n * 4 + 4);
            let x: Vec<u16> = raw[..n * 2]
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect();
            let y: Vec<u16> = raw[n * 2..n * 4]
                .chunks_exact(2)
                .map(|c| u16::from_le_bytes([c[0], c[1]]))
                .collect();
            let want =
                f32::from_le_bytes([raw[n * 4], raw[n * 4 + 1], raw[n * 4 + 2], raw[n * 4 + 3]]);
            let emu = vec_dot_f16_emu(&x, &y);
            let fast = vec_dot_f16(&x, &y);
            assert_eq!(emu.to_bits(), want.to_bits(), "emu dot n={n}");
            assert_eq!(fast.to_bits(), want.to_bits(), "fast dot n={n}");
        }
    }

    /// The asm fast path must equal the emulation on random data even
    /// without the hardware dump (cheap self-consistency, runs everywhere).
    #[test]
    fn vec_dot_f16_fast_path_equals_emulation() {
        let mut state = 0x1234_5678u32;
        let mut rng = move || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            state
        };
        for n in [32usize, 96, 100, 4096] {
            let gen = |rng: &mut dyn FnMut() -> u32, n: usize| -> Vec<u16> {
                (0..n)
                    .map(|_| loop {
                        let b = (rng() >> 13) as u16;
                        if (b & 0x7c00) != 0x7c00 && ((b >> 10) & 0x1f) < 15 {
                            break b;
                        }
                    })
                    .collect()
            };
            let x = gen(&mut rng, n);
            let y = gen(&mut rng, n);
            assert_eq!(
                vec_dot_f16(&x, &y).to_bits(),
                vec_dot_f16_emu(&x, &y).to_bits(),
                "n={n}"
            );
        }
    }
}
