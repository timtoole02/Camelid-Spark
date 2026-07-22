//! Canonical blocked f32 reduction kernels for decode attention.
//!
//! Camelid's determinism guarantee for the attention QK dot product and the
//! V-weighted accumulation is a *fixed, documented reduction order* — not the
//! legacy scalar left-to-right order. This module defines that canonical
//! blocked order and implements it twice: a portable scalar form (the
//! reference on every platform) and an AVX2/FMA form that realizes the same
//! order with intrinsics and is therefore bitwise identical to the reference
//! by construction. Bitwise equality is enforced by property tests below; any
//! divergence is a bug in the kernel, never an accepted tolerance.
//!
//! Normative reduction order for `dot(x, y)` with `len = x.len()`:
//!
//! - 32 partial accumulators, conceptually four 8-lane f32 vectors
//!   `acc0..acc3` (`p[k*8 + l]` = lane `l` of `acc_k`), all starting at 0.0.
//! - Main loop over `i` in steps of 32 while `i + 32 <= len`:
//!   `acc_k.lane[l] = fma(x[i + k*8 + l], y[i + k*8 + l], acc_k.lane[l])`.
//!   FMA is canonical: a single rounding per multiply-add.
//! - Combine, lane-wise, in this exact order: `s01 = acc0 + acc1`,
//!   `s23 = acc2 + acc3`, `s = s01 + s23`.
//! - Horizontal sum of `s`, fixed tree: `t0 = s[0]+s[4]`, `t1 = s[1]+s[5]`,
//!   `t2 = s[2]+s[6]`, `t3 = s[3]+s[7]`, then `u0 = t0+t2`, `u1 = t1+t3`,
//!   result `h = u0 + u1`.
//! - Tail (`len % 32` elements): scalar, ascending index, each element folded
//!   as `h = fma(x[i], y[i], h)`.
//!
//! Normative form for the V accumulation (`axpy`): per element, ascending,
//! `out[d] = fma(prob, v[d], out[d])`. This is elementwise (no cross-lane
//! reduction), so the scalar and SIMD forms agree bitwise as long as both use
//! a single-rounding fused multiply-add.
//!
//! `f32::mul_add` lowers to a hardware FMA on every target this lane can be
//! enabled on (the workspace pins `target-cpu=x86-64-v3`, which includes FMA)
//! and to a correctly-rounded libm `fmaf` elsewhere, so the scalar reference
//! is single-rounding everywhere.

#[cfg(target_arch = "x86_64")]
use std::sync::OnceLock;

const BLOCK: usize = 32;
const LANES: usize = 8;

/// Canonical blocked dot product — portable scalar realization of the
/// normative reduction order documented at module level. This is the
/// reference implementation on all platforms.
pub(super) fn dot_blocked_scalar(x: &[f32], y: &[f32]) -> f32 {
    debug_assert_eq!(x.len(), y.len());
    let len = x.len();
    let mut acc = [[0.0f32; LANES]; 4];
    let mut i = 0;
    while i + BLOCK <= len {
        for (k, acc_k) in acc.iter_mut().enumerate() {
            let base = i + k * LANES;
            for (l, lane) in acc_k.iter_mut().enumerate() {
                *lane = x[base + l].mul_add(y[base + l], *lane);
            }
        }
        i += BLOCK;
    }
    let mut s = [0.0f32; LANES];
    for (l, s_lane) in s.iter_mut().enumerate() {
        let s01 = acc[0][l] + acc[1][l];
        let s23 = acc[2][l] + acc[3][l];
        *s_lane = s01 + s23;
    }
    let t0 = s[0] + s[4];
    let t1 = s[1] + s[5];
    let t2 = s[2] + s[6];
    let t3 = s[3] + s[7];
    let u0 = t0 + t2;
    let u1 = t1 + t3;
    let mut h = u0 + u1;
    while i < len {
        h = x[i].mul_add(y[i], h);
        i += 1;
    }
    h
}

/// Canonical V accumulation — `out[d] = fma(prob, v[d], out[d])`, ascending.
pub(super) fn axpy_blocked_scalar(out: &mut [f32], prob: f32, v: &[f32]) {
    debug_assert_eq!(out.len(), v.len());
    for (out_value, value) in out.iter_mut().zip(v) {
        *out_value = prob.mul_add(*value, *out_value);
    }
}

/// AVX2/FMA realization of the canonical blocked dot product. Bitwise
/// identical to [`dot_blocked_scalar`] by construction: four independent
/// `__m256` accumulators updated with `_mm256_fmadd_ps` (one rounding per
/// lane, same as `mul_add`), the two specified lane-wise combine adds, then
/// the fixed horizontal tree performed on extracted lanes (no `hadd`).
///
/// # Safety
/// Caller must ensure AVX2 and FMA are available on the running CPU.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn dot_blocked_avx2(x: &[f32], y: &[f32]) -> f32 {
    use std::arch::x86_64::{
        _mm256_add_ps, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_setzero_ps, _mm256_storeu_ps,
    };
    debug_assert_eq!(x.len(), y.len());
    let len = x.len();
    let xp = x.as_ptr();
    let yp = y.as_ptr();
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();
    let mut i = 0;
    while i + BLOCK <= len {
        // SAFETY: the loop guard ensures `i + 32 <= len`, so every 8-lane
        // load below stays inside both slices.
        unsafe {
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(xp.add(i)), _mm256_loadu_ps(yp.add(i)), acc0);
            acc1 = _mm256_fmadd_ps(
                _mm256_loadu_ps(xp.add(i + 8)),
                _mm256_loadu_ps(yp.add(i + 8)),
                acc1,
            );
            acc2 = _mm256_fmadd_ps(
                _mm256_loadu_ps(xp.add(i + 16)),
                _mm256_loadu_ps(yp.add(i + 16)),
                acc2,
            );
            acc3 = _mm256_fmadd_ps(
                _mm256_loadu_ps(xp.add(i + 24)),
                _mm256_loadu_ps(yp.add(i + 24)),
                acc3,
            );
        }
        i += BLOCK;
    }
    let s01 = _mm256_add_ps(acc0, acc1);
    let s23 = _mm256_add_ps(acc2, acc3);
    let s = _mm256_add_ps(s01, s23);
    let mut lanes = [0.0f32; LANES];
    // SAFETY: `lanes` is exactly 8 f32s, the width of one __m256 store.
    unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), s) };
    let t0 = lanes[0] + lanes[4];
    let t1 = lanes[1] + lanes[5];
    let t2 = lanes[2] + lanes[6];
    let t3 = lanes[3] + lanes[7];
    let u0 = t0 + t2;
    let u1 = t1 + t3;
    let mut h = u0 + u1;
    while i < len {
        h = x[i].mul_add(y[i], h);
        i += 1;
    }
    h
}

/// AVX2/FMA realization of the canonical V accumulation. Elementwise fused
/// multiply-add, so it agrees bitwise with [`axpy_blocked_scalar`] on every
/// element by construction.
///
/// # Safety
/// Caller must ensure AVX2 and FMA are available on the running CPU.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub(super) unsafe fn axpy_blocked_avx2(out: &mut [f32], prob: f32, v: &[f32]) {
    use std::arch::x86_64::{_mm256_fmadd_ps, _mm256_loadu_ps, _mm256_set1_ps, _mm256_storeu_ps};
    debug_assert_eq!(out.len(), v.len());
    let len = out.len();
    let prob_v = _mm256_set1_ps(prob);
    let mut i = 0;
    while i + LANES <= len {
        // SAFETY: the loop guard ensures `i + 8 <= len` for both slices.
        unsafe {
            let value = _mm256_loadu_ps(v.as_ptr().add(i));
            let current = _mm256_loadu_ps(out.as_ptr().add(i));
            _mm256_storeu_ps(
                out.as_mut_ptr().add(i),
                _mm256_fmadd_ps(prob_v, value, current),
            );
        }
        i += LANES;
    }
    while i < len {
        out[i] = prob.mul_add(v[i], out[i]);
        i += 1;
    }
}

/// Runtime dispatch guard for the AVX2/FMA realizations, resolved once.
#[cfg(target_arch = "x86_64")]
fn attn_f32_avx2_fma_available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma")
    })
}

/// Canonical blocked dot product with safe dispatch: AVX2/FMA when the CPU
/// has it, otherwise the scalar reference. Both paths produce identical bits.
pub(super) fn dot_blocked(x: &[f32], y: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if attn_f32_avx2_fma_available() {
        // SAFETY: guarded by the runtime AVX2+FMA feature check above.
        return unsafe { dot_blocked_avx2(x, y) };
    }
    dot_blocked_scalar(x, y)
}

/// Canonical V accumulation with safe dispatch, mirroring [`dot_blocked`].
pub(super) fn axpy_blocked(out: &mut [f32], prob: f32, v: &[f32]) {
    #[cfg(target_arch = "x86_64")]
    if attn_f32_avx2_fma_available() {
        // SAFETY: guarded by the runtime AVX2+FMA feature check above.
        unsafe { axpy_blocked_avx2(out, prob, v) };
        return;
    }
    axpy_blocked_scalar(out, prob, v);
}

// ---------------------------------------------------------------------------
// f16-operand variants (KV f16 storage lane, `CAMELID_KV_F16`).
//
// Canonical order: convert each f16 element to f32 (exact expansion, pinned
// to the existing-helper semantics — see `kv_f16`), THEN the identical
// blocked order above. The conversion is fused into the loads (`vcvtph2ps`
// on the fast path) so no f32 staging buffer re-adds the traffic the lane
// removes. Bitwise-locked to their scalar references by the same >=10k-case
// to_bits() property tests as the f32 kernels.
// ---------------------------------------------------------------------------

use super::kv_f16::f16_to_f32_kv;

/// Canonical blocked dot with an f16 K operand — scalar reference.
pub(super) fn dot_blocked_f16_scalar(x: &[f32], y16: &[u16]) -> f32 {
    debug_assert_eq!(x.len(), y16.len());
    let len = x.len();
    let mut acc = [[0.0f32; LANES]; 4];
    let mut i = 0;
    while i + BLOCK <= len {
        for (k, acc_k) in acc.iter_mut().enumerate() {
            let base = i + k * LANES;
            for (l, lane) in acc_k.iter_mut().enumerate() {
                *lane = x[base + l].mul_add(f16_to_f32_kv(y16[base + l]), *lane);
            }
        }
        i += BLOCK;
    }
    let mut s = [0.0f32; LANES];
    for (l, s_lane) in s.iter_mut().enumerate() {
        let s01 = acc[0][l] + acc[1][l];
        let s23 = acc[2][l] + acc[3][l];
        *s_lane = s01 + s23;
    }
    let t0 = s[0] + s[4];
    let t1 = s[1] + s[5];
    let t2 = s[2] + s[6];
    let t3 = s[3] + s[7];
    let u0 = t0 + t2;
    let u1 = t1 + t3;
    let mut h = u0 + u1;
    while i < len {
        h = x[i].mul_add(f16_to_f32_kv(y16[i]), h);
        i += 1;
    }
    h
}

/// Canonical V accumulation with an f16 V operand — scalar reference.
pub(super) fn axpy_blocked_f16_scalar(out: &mut [f32], prob: f32, v16: &[u16]) {
    debug_assert_eq!(out.len(), v16.len());
    for (out_value, value) in out.iter_mut().zip(v16) {
        *out_value = prob.mul_add(f16_to_f32_kv(*value), *out_value);
    }
}

/// AVX2/FMA/F16C realization of [`dot_blocked_f16_scalar`]: 8 halves per
/// load expanded with `vcvtph2ps` (exact, matches the scalar expansion
/// bitwise on all 2^16 inputs), then the identical accumulator/combine/tail
/// structure as [`dot_blocked_avx2`].
///
/// # Safety
/// Caller must ensure AVX2, FMA, and F16C are available on the running CPU.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
pub(super) unsafe fn dot_blocked_f16_avx2(x: &[f32], y16: &[u16]) -> f32 {
    use std::arch::x86_64::{
        __m128i, _mm256_add_ps, _mm256_cvtph_ps, _mm256_fmadd_ps, _mm256_loadu_ps,
        _mm256_setzero_ps, _mm256_storeu_ps, _mm_loadu_si128,
    };
    debug_assert_eq!(x.len(), y16.len());
    let len = x.len();
    let xp = x.as_ptr();
    let yp = y16.as_ptr();
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();
    let mut i = 0;
    while i + BLOCK <= len {
        // SAFETY: the loop guard ensures `i + 32 <= len`, so every 8-lane
        // load below stays inside both slices.
        unsafe {
            let y0 = _mm256_cvtph_ps(_mm_loadu_si128(yp.add(i) as *const __m128i));
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(xp.add(i)), y0, acc0);
            let y1 = _mm256_cvtph_ps(_mm_loadu_si128(yp.add(i + 8) as *const __m128i));
            acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(xp.add(i + 8)), y1, acc1);
            let y2 = _mm256_cvtph_ps(_mm_loadu_si128(yp.add(i + 16) as *const __m128i));
            acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(xp.add(i + 16)), y2, acc2);
            let y3 = _mm256_cvtph_ps(_mm_loadu_si128(yp.add(i + 24) as *const __m128i));
            acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(xp.add(i + 24)), y3, acc3);
        }
        i += BLOCK;
    }
    let s01 = _mm256_add_ps(acc0, acc1);
    let s23 = _mm256_add_ps(acc2, acc3);
    let s = _mm256_add_ps(s01, s23);
    let mut lanes = [0.0f32; LANES];
    // SAFETY: `lanes` is exactly 8 f32s, the width of one __m256 store.
    unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), s) };
    let t0 = lanes[0] + lanes[4];
    let t1 = lanes[1] + lanes[5];
    let t2 = lanes[2] + lanes[6];
    let t3 = lanes[3] + lanes[7];
    let u0 = t0 + t2;
    let u1 = t1 + t3;
    let mut h = u0 + u1;
    while i < len {
        h = x[i].mul_add(f16_to_f32_kv(y16[i]), h);
        i += 1;
    }
    h
}

/// AVX2/FMA/F16C realization of [`axpy_blocked_f16_scalar`].
///
/// # Safety
/// Caller must ensure AVX2, FMA, and F16C are available on the running CPU.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
pub(super) unsafe fn axpy_blocked_f16_avx2(out: &mut [f32], prob: f32, v16: &[u16]) {
    use std::arch::x86_64::{
        __m128i, _mm256_cvtph_ps, _mm256_fmadd_ps, _mm256_loadu_ps, _mm256_set1_ps,
        _mm256_storeu_ps, _mm_loadu_si128,
    };
    debug_assert_eq!(out.len(), v16.len());
    let len = out.len();
    let prob_v = _mm256_set1_ps(prob);
    let mut i = 0;
    while i + LANES <= len {
        // SAFETY: the loop guard ensures `i + 8 <= len` for both slices.
        unsafe {
            let value = _mm256_cvtph_ps(_mm_loadu_si128(v16.as_ptr().add(i) as *const __m128i));
            let current = _mm256_loadu_ps(out.as_ptr().add(i));
            _mm256_storeu_ps(
                out.as_mut_ptr().add(i),
                _mm256_fmadd_ps(prob_v, value, current),
            );
        }
        i += LANES;
    }
    while i < len {
        out[i] = prob.mul_add(f16_to_f32_kv(v16[i]), out[i]);
        i += 1;
    }
}

/// Runtime dispatch guard for the f16 kernel fast paths, resolved once.
#[cfg(target_arch = "x86_64")]
fn attn_f16_kernels_available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
            && std::arch::is_x86_feature_detected!("f16c")
    })
}

/// f16-K blocked dot with safe dispatch. Both paths produce identical bits.
pub(super) fn dot_blocked_f16(x: &[f32], y16: &[u16]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if attn_f16_kernels_available() {
        // SAFETY: guarded by the runtime AVX2+FMA+F16C feature check above.
        return unsafe { dot_blocked_f16_avx2(x, y16) };
    }
    dot_blocked_f16_scalar(x, y16)
}

/// f16-V blocked axpy with safe dispatch, mirroring [`dot_blocked_f16`].
pub(super) fn axpy_blocked_f16(out: &mut [f32], prob: f32, v16: &[u16]) {
    #[cfg(target_arch = "x86_64")]
    if attn_f16_kernels_available() {
        // SAFETY: guarded by the runtime AVX2+FMA+F16C feature check above.
        unsafe { axpy_blocked_f16_avx2(out, prob, v16) };
        return;
    }
    axpy_blocked_f16_scalar(out, prob, v16);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic xorshift64* generator so the property tests are
    /// reproducible without pulling in a rand dependency.
    struct XorShift64Star(u64);

    impl XorShift64Star {
        fn next_u64(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }

        fn next_usize(&mut self, bound: usize) -> usize {
            (self.next_u64() % bound as u64) as usize
        }

        /// Finite f32 spanning subnormals through large magnitudes, both
        /// signs. Magnitudes cap at 1e15 so pairwise products (≤ ~2.5e29) and
        /// any partial sum stay finite in f32 — the canonical-order contract
        /// is about reduction order on finite reals, not overflow behavior.
        fn next_f32(&mut self) -> f32 {
            let magnitude_class = self.next_u64() % 8;
            let unit = ((self.next_u64() >> 40) as f32) / (1u64 << 24) as f32 - 0.5;
            match magnitude_class {
                0 => unit * f32::MIN_POSITIVE * 0.5, // subnormal territory
                1 => unit * 1e-20,
                2 => unit * 1e-6,
                3 | 4 => unit * 2.0,
                5 => unit * 1e4,
                6 => unit * 1e12,
                _ => unit * 1e15,
            }
        }
    }

    fn fill(rng: &mut XorShift64Star, len: usize) -> Vec<f32> {
        (0..len).map(|_| rng.next_f32()).collect()
    }

    /// Random u16 across the FULL binary16 domain — finite values,
    /// subnormals, ±inf, and NaN payloads all occur; the f16 kernels must be
    /// bitwise-locked on all of them.
    fn fill_u16(rng: &mut XorShift64Star, len: usize) -> Vec<u16> {
        (0..len).map(|_| rng.next_u64() as u16).collect()
    }

    #[cfg(target_arch = "x86_64")]
    fn f16_kernels_testable() -> bool {
        std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
            && std::arch::is_x86_feature_detected!("f16c")
    }

    /// Fusion oracle: the f16 scalar kernel must equal "expand every element
    /// through the pinned conversion, then the existing f32 blocked order".
    #[test]
    fn dot_blocked_f16_scalar_equals_expand_then_blocked() {
        let mut rng = XorShift64Star(0xf16d_0001);
        for case in 0..2_000 {
            let len = match case % 4 {
                0 => 64,
                1 => 128,
                _ => 1 + rng.next_usize(512),
            };
            let x = fill(&mut rng, len);
            let y16 = fill_u16(&mut rng, len);
            let expanded: Vec<f32> = y16.iter().map(|&b| super::f16_to_f32_kv(b)).collect();
            let fused = dot_blocked_f16_scalar(&x, &y16);
            let staged = dot_blocked_scalar(&x, &expanded);
            assert_eq!(
                fused.to_bits(),
                staged.to_bits(),
                "case {case}: len {len}, fused {fused}, staged {staged}"
            );
            let prob = rng.next_f32();
            let mut out_fused = x.clone();
            let mut out_staged = x.clone();
            axpy_blocked_f16_scalar(&mut out_fused, prob, &y16);
            axpy_blocked_scalar(&mut out_staged, prob, &expanded);
            for (a, b) in out_fused.iter().zip(&out_staged) {
                assert_eq!(a.to_bits(), b.to_bits(), "axpy case {case}");
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn dot_blocked_f16_avx2_matches_scalar_bitwise() {
        if !f16_kernels_testable() {
            eprintln!("skipping: AVX2+FMA+F16C not available on this host");
            return;
        }
        let mut rng = XorShift64Star(0xf16b_0001);
        for case in 0..10_000 {
            let len = match case % 4 {
                0 => 64,
                1 => 128,
                _ => 1 + rng.next_usize(512),
            };
            let x = fill(&mut rng, len);
            let y16 = fill_u16(&mut rng, len);
            let scalar = dot_blocked_f16_scalar(&x, &y16);
            // SAFETY: guarded by the runtime feature check above.
            let avx2 = unsafe { dot_blocked_f16_avx2(&x, &y16) };
            assert_eq!(
                scalar.to_bits(),
                avx2.to_bits(),
                "case {case}: len {len}, scalar {scalar}, avx2 {avx2}"
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn axpy_blocked_f16_avx2_matches_scalar_bitwise() {
        if !f16_kernels_testable() {
            eprintln!("skipping: AVX2+FMA+F16C not available on this host");
            return;
        }
        let mut rng = XorShift64Star(0xf16a_0001);
        for case in 0..10_000 {
            let len = match case % 4 {
                0 => 64,
                1 => 128,
                _ => 1 + rng.next_usize(512),
            };
            let prob = rng.next_f32();
            let base = fill(&mut rng, len);
            let v16 = fill_u16(&mut rng, len);
            let mut out_scalar = base.clone();
            let mut out_avx2 = base;
            axpy_blocked_f16_scalar(&mut out_scalar, prob, &v16);
            // SAFETY: guarded by the runtime feature check above.
            unsafe { axpy_blocked_f16_avx2(&mut out_avx2, prob, &v16) };
            for (d, (s, a)) in out_scalar.iter().zip(&out_avx2).enumerate() {
                assert_eq!(
                    s.to_bits(),
                    a.to_bits(),
                    "case {case}: len {len}, element {d}, scalar {s}, avx2 {a}"
                );
            }
        }
    }

    #[test]
    fn dot_blocked_scalar_known_answers() {
        // All-ones: exact integer sums, main loop only (64, 128) and with a
        // tail (103 = 3*32 + 7).
        for len in [64usize, 128, 103] {
            let x = vec![1.0f32; len];
            let y = vec![1.0f32; len];
            assert_eq!(dot_blocked_scalar(&x, &y), len as f32, "len {len}");
        }
        // Alternating signs cancel exactly regardless of reduction order.
        let x: Vec<f32> = (0..128)
            .map(|i| if i % 2 == 0 { 3.0 } else { -3.0 })
            .collect();
        let y = vec![2.5f32; 128];
        assert_eq!(dot_blocked_scalar(&x, &y), 0.0);
        // Exactly representable dyadic values, len not a multiple of 32:
        // dot = sum of i * 0.25 for i in 0..103 = 5253 * 0.25 = 1313.25.
        let x: Vec<f32> = (0..103).map(|i| i as f32).collect();
        let y = vec![0.25f32; 103];
        assert_eq!(dot_blocked_scalar(&x, &y), 1313.25);
        // Subnormals survive: 32 * (MIN_POSITIVE/2) * 1.0, folded through the
        // blocked tree, is still 16 * MIN_POSITIVE.
        let x = vec![f32::MIN_POSITIVE * 0.5; 32];
        let y = vec![1.0f32; 32];
        assert_eq!(dot_blocked_scalar(&x, &y), f32::MIN_POSITIVE * 16.0);
        // Magnitude spread with negatives, checked against an f64 oracle with
        // the standard condition-aware forward-error bound: for any reduction
        // order, |dot_f32 - dot_f64| <= len * eps_f32 * sum(|x*y|). A naive
        // relative tolerance would be flaky under catastrophic cancellation.
        let mut rng = XorShift64Star(0x5EED_0001);
        let len = 96 + 7;
        let x = fill(&mut rng, len);
        let y = fill(&mut rng, len);
        let oracle: f64 = x
            .iter()
            .zip(&y)
            .map(|(a, b)| f64::from(*a) * f64::from(*b))
            .sum();
        let magnitude: f64 = x
            .iter()
            .zip(&y)
            .map(|(a, b)| (f64::from(*a) * f64::from(*b)).abs())
            .sum();
        let got = f64::from(dot_blocked_scalar(&x, &y));
        let bound = len as f64 * f64::from(f32::EPSILON) * magnitude + f64::from(f32::MIN_POSITIVE);
        assert!(
            (got - oracle).abs() <= bound,
            "blocked dot strayed from f64 oracle: got {got}, oracle {oracle}, bound {bound}"
        );
    }

    #[test]
    fn axpy_blocked_scalar_known_answers() {
        let mut out = vec![1.0f32; 103];
        let v: Vec<f32> = (0..103).map(|i| i as f32).collect();
        axpy_blocked_scalar(&mut out, 0.5, &v);
        for (i, value) in out.iter().enumerate() {
            assert_eq!(*value, 1.0 + 0.5 * i as f32, "index {i}");
        }
    }

    /// The canonical blocked order intentionally differs from the legacy
    /// scalar chain in `crate::tensor::dot_product`. This test locks that in
    /// so nobody "fixes" the blocked kernels back to the legacy order: the
    /// two are DIFFERENT reductions and must be allowed to disagree in the
    /// last bits. (Constructed input, not a fluke: mixed magnitudes make the
    /// serial chain and the blocked FMA tree round differently.)
    #[test]
    fn dot_blocked_scalar_differs_from_legacy_dot_product() {
        let mut rng = XorShift64Star(0xD1FF_0001);
        let mut found_difference = false;
        for _ in 0..64 {
            let x = fill(&mut rng, 128);
            let y = fill(&mut rng, 128);
            let blocked = dot_blocked_scalar(&x, &y);
            let legacy = crate::tensor::dot_product(&x, &y);
            if blocked.to_bits() != legacy.to_bits() {
                found_difference = true;
                break;
            }
        }
        assert!(
            found_difference,
            "blocked order unexpectedly matched the legacy scalar chain on \
             64 mixed-magnitude inputs — the canonical order should differ"
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn dot_blocked_avx2_matches_scalar_bitwise() {
        if !attn_f32_avx2_fma_available() {
            eprintln!("skipping: AVX2+FMA not available on this host");
            return;
        }
        let mut rng = XorShift64Star(0xB10C_0001);
        for case in 0..10_000 {
            // Bias lengths toward the real head dims (64/128); otherwise
            // exercise the whole [1, 512] range including tails.
            let len = match case % 4 {
                0 => 64,
                1 => 128,
                _ => 1 + rng.next_usize(512),
            };
            let x = fill(&mut rng, len);
            let y = fill(&mut rng, len);
            let scalar = dot_blocked_scalar(&x, &y);
            // SAFETY: guarded by the runtime AVX2+FMA feature check above.
            let avx2 = unsafe { dot_blocked_avx2(&x, &y) };
            assert_eq!(
                scalar.to_bits(),
                avx2.to_bits(),
                "case {case}: len {len}, scalar {scalar}, avx2 {avx2}"
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn axpy_blocked_avx2_matches_scalar_bitwise() {
        if !attn_f32_avx2_fma_available() {
            eprintln!("skipping: AVX2+FMA not available on this host");
            return;
        }
        let mut rng = XorShift64Star(0xA4B9_0001);
        for case in 0..10_000 {
            let len = match case % 4 {
                0 => 64,
                1 => 128,
                _ => 1 + rng.next_usize(512),
            };
            let prob = rng.next_f32();
            let base = fill(&mut rng, len);
            let v = fill(&mut rng, len);
            let mut out_scalar = base.clone();
            let mut out_avx2 = base;
            axpy_blocked_scalar(&mut out_scalar, prob, &v);
            // SAFETY: guarded by the runtime AVX2+FMA feature check above.
            unsafe { axpy_blocked_avx2(&mut out_avx2, prob, &v) };
            for (d, (s, a)) in out_scalar.iter().zip(&out_avx2).enumerate() {
                assert_eq!(
                    s.to_bits(),
                    a.to_bits(),
                    "case {case}: len {len}, element {d}, scalar {s}, avx2 {a}"
                );
            }
        }
    }

    #[test]
    fn dispatch_matches_scalar_bitwise() {
        let mut rng = XorShift64Star(0xD15B_0001);
        for _ in 0..256 {
            let len = 1 + rng.next_usize(512);
            let prob = rng.next_f32();
            let x = fill(&mut rng, len);
            let y = fill(&mut rng, len);
            assert_eq!(
                dot_blocked(&x, &y).to_bits(),
                dot_blocked_scalar(&x, &y).to_bits()
            );
            let mut out_dispatch = x.clone();
            let mut out_scalar = x.clone();
            axpy_blocked(&mut out_dispatch, prob, &y);
            axpy_blocked_scalar(&mut out_scalar, prob, &y);
            for (a, b) in out_dispatch.iter().zip(&out_scalar) {
                assert_eq!(a.to_bits(), b.to_bits());
            }
        }
    }
}
