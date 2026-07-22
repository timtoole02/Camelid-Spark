//! Canonical f32<->f16 (IEEE binary16) conversions for the KV f16 storage
//! lane (`CAMELID_KV_F16`).
//!
//! Load-bearing discovery (2026-07-01): the CPU KV write path ALREADY rounds
//! every stored key/value through f16 unconditionally
//! (`copy_to_f16_kv_cache_storage` = `f16_bits_to_f32(f32_to_f16_bits(x))`
//! in both `write_kv_cache` and `write_kv_cache_batch`), so main's f32
//! buffers hold exactly-f16-representable values at 2x the bytes. The f16
//! storage lane therefore introduces ZERO new rounding: storing
//! `f32_to_f16_bits(x)` as u16 and expanding with `f16_bits_to_f32` on read
//! reproduces today's buffer values bit-for-bit, upgrading this lane to the
//! bitwise-identity contract.
//!
//! Canonical STORE semantics are consequently pinned to the EXISTING helper
//! (round-to-nearest-even; overflow to ±inf incl. the 65520 tie; RNE
//! subnormals; every NaN canonicalized to sign|0x7E00), NOT to raw F16C: the
//! F16C write fast path must match that bitwise on every input, which
//! requires a NaN fixup after `vcvtps2ph` (hardware preserves truncated NaN
//! payloads; the canonical form does not). The READ direction is exact and
//! pinned to `vcvtph2ps` on the full 2^16 domain; it differs from the repo's
//! `f16_bits_to_f32` only on signalling-NaN inputs (hardware quiets them),
//! which the canonical store makes unreachable. All enforced by the
//! exhaustive + randomized property tests below.
//!
//! All arithmetic in the attention kernels stays f32; these conversions are
//! the only place the dtype changes, applied at KV store and fused into the
//! blocked kernels at read (see `attn_f32_dot`).

/// Reference f32 -> f16: RNE, NaN -> sign|0x7E00. Semantically identical to
/// `crate::inference::f32_to_f16_bits` (locked by test) — duplicated here so
/// the KV lane's canonical conversion is self-contained and cannot drift if
/// the Metal/GPU helper ever changes.
pub(super) fn f32_to_f16_kv(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x007f_ffff;

    if exp == 0xff {
        return sign | if mant == 0 { 0x7c00 } else { 0x7e00 };
    }
    let half_exp = exp - 127 + 15;
    if half_exp >= 0x1f {
        return sign | 0x7c00;
    }
    if half_exp <= 0 {
        if half_exp < -10 {
            return sign;
        }
        let mantissa = mant | 0x0080_0000;
        let shift = 14 - half_exp;
        let mut half_mant = (mantissa >> shift) as u16;
        let round_bit = 1_u32 << (shift - 1);
        if (mantissa & round_bit) != 0
            && ((mantissa & (round_bit - 1)) != 0 || (half_mant & 1) != 0)
        {
            half_mant = half_mant.wrapping_add(1);
        }
        return sign | half_mant;
    }

    let mut half = sign | ((half_exp as u16) << 10) | ((mant >> 13) as u16);
    if (mant & 0x0000_1000) != 0 && ((mant & 0x0000_0fff) != 0 || (half & 1) != 0) {
        half = half.wrapping_add(1);
    }
    half
}

/// Reference f16 -> f32 (exact), bitwise-equal to `vcvtph2ps` on the FULL
/// 2^16 domain — including quieting signalling NaNs (hardware sets the f32
/// quiet bit; payload otherwise preserved). This is the one place the read
/// direction differs from the repo's `f16_bits_to_f32`, which passes sNaN
/// through un-quieted: the delta is UNREACHABLE from the KV lane, because
/// the pinned store conversion canonicalizes every NaN to sign|0x7E00 (a
/// quiet NaN) before anything lands in the cache — locked by the tests
/// below.
pub(super) fn f16_to_f32_kv(bits: u16) -> f32 {
    let sign = (u32::from(bits & 0x8000)) << 16;
    let exp = (bits & 0x7c00) >> 10;
    let frac = u32::from(bits & 0x03ff);
    let out = match exp {
        0 => {
            if frac == 0 {
                sign
            } else {
                let mut mant = frac;
                let mut e = -14_i32;
                while (mant & 0x0400) == 0 {
                    mant <<= 1;
                    e -= 1;
                }
                mant &= 0x03ff;
                let exp32 = (e + 127) as u32;
                sign | (exp32 << 23) | (mant << 13)
            }
        }
        0x1f => {
            if frac == 0 {
                sign | 0x7f80_0000
            } else {
                // NaN: quiet bit forced, payload preserved — vcvtph2ps.
                sign | 0x7f80_0000 | 0x0040_0000 | (frac << 13)
            }
        }
        _ => {
            let exp32 = u32::from(exp) + (127 - 15);
            sign | (exp32 << 23) | (frac << 13)
        }
    };
    f32::from_bits(out)
}

/// Convert a full f32 slice into f16 storage, F16C-accelerated where the CPU
/// has it. Both paths produce identical bits (property-tested).
pub(super) fn convert_f32_slice_to_f16(source: &[f32], dest: &mut [u16]) {
    debug_assert_eq!(source.len(), dest.len());
    #[cfg(target_arch = "x86_64")]
    if f16c_available() {
        // SAFETY: guarded by the runtime F16C/AVX feature check.
        unsafe { convert_f32_slice_to_f16_f16c(source, dest) };
        return;
    }
    for (d, s) in dest.iter_mut().zip(source) {
        *d = f32_to_f16_kv(*s);
    }
}

/// Runtime gate for the F16C fast paths, resolved once.
#[cfg(target_arch = "x86_64")]
pub(super) fn f16c_available() -> bool {
    static AVAILABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        std::arch::is_x86_feature_detected!("f16c") && std::arch::is_x86_feature_detected!("avx")
    })
}

/// F16C bulk f32 -> f16 (RNE). Bitwise-equal to the scalar reference,
/// including the canonical-NaN semantics: hardware `vcvtps2ph` preserves
/// truncated NaN payloads, so NaN lanes (detected via unordered compare,
/// branch taken only when present) are re-converted through the reference.
///
/// # Safety
/// Caller must ensure F16C and AVX are available on the running CPU.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,f16c")]
unsafe fn convert_f32_slice_to_f16_f16c(source: &[f32], dest: &mut [u16]) {
    use std::arch::x86_64::{
        __m128i, _mm256_cmp_ps, _mm256_cvtps_ph, _mm256_loadu_ps, _mm256_movemask_ps,
        _mm_storeu_si128, _CMP_UNORD_Q, _MM_FROUND_TO_NEAREST_INT,
    };
    let len = source.len();
    let mut i = 0;
    while i + 8 <= len {
        // SAFETY: the loop guard keeps every 8-lane load/store in bounds.
        unsafe {
            let wide = _mm256_loadu_ps(source.as_ptr().add(i));
            let half = _mm256_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(wide);
            _mm_storeu_si128(dest.as_mut_ptr().add(i) as *mut __m128i, half);
            let nan_lanes = _mm256_movemask_ps(_mm256_cmp_ps::<_CMP_UNORD_Q>(wide, wide));
            if nan_lanes != 0 {
                for lane in 0..8 {
                    if nan_lanes & (1 << lane) != 0 {
                        dest[i + lane] = f32_to_f16_kv(source[i + lane]);
                    }
                }
            }
        }
        i += 8;
    }
    while i < len {
        dest[i] = f32_to_f16_kv(source[i]);
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(target_arch = "x86_64")]
    fn hardware_f16_to_f32(bits: u16) -> f32 {
        use std::arch::x86_64::{_mm256_cvtph_ps, _mm256_storeu_ps, _mm_set1_epi16};
        assert!(f16c_available(), "test requires F16C");
        let mut out = [0.0f32; 8];
        // SAFETY: guarded by the f16c_available assertion above.
        unsafe {
            let half = _mm_set1_epi16(bits as i16);
            let wide = _mm256_cvtph_ps(half);
            _mm256_storeu_ps(out.as_mut_ptr(), wide);
        }
        out[0]
    }

    #[cfg(target_arch = "x86_64")]
    fn hardware_f32_to_f16(value: f32) -> u16 {
        use std::arch::x86_64::{
            __m128i, _mm256_cvtps_ph, _mm256_set1_ps, _mm_storeu_si128, _MM_FROUND_TO_NEAREST_INT,
        };
        assert!(f16c_available(), "test requires F16C");
        let mut out = [0u16; 8];
        // SAFETY: guarded by the f16c_available assertion above.
        unsafe {
            let wide = _mm256_set1_ps(value);
            let half = _mm256_cvtps_ph::<_MM_FROUND_TO_NEAREST_INT>(wide);
            _mm_storeu_si128(out.as_mut_ptr() as *mut __m128i, half);
        }
        out[0]
    }

    /// Exhaustive: every one of the 2^16 f16 bit patterns converts to f32
    /// bitwise-equal to hardware AND to the repo's existing helper, and
    /// round-trips back to itself (non-NaN) or to the canonical NaN.
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn f16_to_f32_matches_hardware_exhaustively_and_round_trips() {
        if !f16c_available() {
            eprintln!("skipping: F16C not available on this host");
            return;
        }
        for bits in 0..=u16::MAX {
            let reference = f16_to_f32_kv(bits);
            let hardware = hardware_f16_to_f32(bits);
            assert_eq!(
                reference.to_bits(),
                hardware.to_bits(),
                "f16->f32 mismatch at bits {bits:#06x}: ref {reference}, hw {hardware}"
            );
            let is_snan = (bits & 0x7c00) == 0x7c00 && (bits & 0x03ff) != 0 && (bits & 0x0200) == 0;
            if is_snan {
                // Documented delta: the existing helper passes sNaN through
                // un-quieted; hardware (and this reference) quiets it. sNaN
                // cannot reach the cache — the store conversion below
                // canonicalizes every NaN to a quiet 0x7E00.
                assert_eq!(
                    reference.to_bits(),
                    crate::inference::f16_bits_to_f32(bits).to_bits() | 0x0040_0000,
                    "sNaN quieting delta changed shape at bits {bits:#06x}"
                );
            } else {
                assert_eq!(
                    reference.to_bits(),
                    crate::inference::f16_bits_to_f32(bits).to_bits(),
                    "f16->f32 drifted from the existing helper at bits {bits:#06x}"
                );
            }
            let round_trip = f32_to_f16_kv(reference);
            let is_nan = (bits & 0x7c00) == 0x7c00 && (bits & 0x03ff) != 0;
            if is_nan {
                assert_eq!(
                    round_trip,
                    (bits & 0x8000) | 0x7e00,
                    "NaN canonicalization failed at bits {bits:#06x}"
                );
            } else {
                assert_eq!(
                    round_trip, bits,
                    "round-trip failed at bits {bits:#06x} (got {round_trip:#06x})"
                );
            }
        }
    }

    /// At least 1.2M randomized f32 inputs across all magnitude classes,
    /// plus targeted edge sets (subnormal f32/f16 territory, overflow
    /// boundary, rounding ties, ±inf, NaN payloads): f32->f16 bitwise-equal
    /// to hardware RNE (non-NaN) and to the canonical-NaN form (NaN).
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn f32_to_f16_matches_hardware_randomized_and_edges() {
        if !f16c_available() {
            eprintln!("skipping: F16C not available on this host");
            return;
        }
        let check = |value: f32| {
            let reference = f32_to_f16_kv(value);
            // Semantic lock to main: the KV canonical conversion must equal
            // the existing helper (whose output today's f32 buffers hold) on
            // EVERY input, NaN included.
            assert_eq!(
                reference,
                crate::inference::f32_to_f16_bits(value),
                "f32->f16 drifted from the existing helper at {:#010x}",
                value.to_bits()
            );
            if value.is_nan() {
                // Canonical NaN by definition; hardware preserves payloads
                // instead, so the fast path carries a NaN fixup (covered by
                // bulk_conversion_matches_reference).
                assert_eq!(
                    reference,
                    ((value.to_bits() >> 16) as u16 & 0x8000) | 0x7e00
                );
                return;
            }
            let hardware = hardware_f32_to_f16(value);
            assert_eq!(
                reference,
                hardware,
                "f32->f16 mismatch at {value} ({:#010x}): ref {reference:#06x}, hw {hardware:#06x}",
                value.to_bits()
            );
        };

        // Targeted edges.
        for value in [
            0.0f32,
            -0.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            f32::from_bits(0x7f80_0001), // signalling NaN, minimal payload
            f32::from_bits(0xffc0_1234), // negative quiet NaN with payload
            f32::from_bits(0x7fbf_ffff), // max-payload signalling NaN
            65504.0,                     // f16 max finite
            65519.999,
            65520.0, // tie: rounds to inf
            65536.0,
            6.1035156e-5, // f16 min normal
            6.097555e-5,  // largest f16 subnormal
            5.9604645e-8, // f16 min subnormal
            2.9802322e-8, // half the min subnormal (tie to zero)
            2.9802326e-8, // just above the tie
            f32::MIN_POSITIVE,
            f32::from_bits(1), // min f32 subnormal
        ] {
            check(value);
        }
        // Every exponent x a few mantissa patterns, both signs.
        for exp in 0..=0xffu32 {
            for mant in [0u32, 1, 0x1000, 0x0fff, 0x1fff, 0x2000, 0x7f_ffff] {
                for sign in [0u32, 0x8000_0000] {
                    check(f32::from_bits(sign | (exp << 23) | mant));
                }
            }
        }
        // Randomized sweep.
        let mut state = 0x5eed_f16c_0001u64;
        let mut next = move || {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            state.wrapping_mul(0x2545_F491_4F6C_DD1D)
        };
        for _ in 0..1_200_000 {
            check(f32::from_bits(next() as u32));
        }
    }

    /// Bulk conversion (F16C path incl. the scalar tail) matches the scalar
    /// reference element-wise.
    #[test]
    fn bulk_conversion_matches_reference() {
        let mut state = 0xb01d_f16c_0002u64;
        let mut next = move || {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            state.wrapping_mul(0x2545_F491_4F6C_DD1D)
        };
        for len in [1usize, 7, 8, 9, 64, 127, 128, 1000] {
            let source: Vec<f32> = (0..len).map(|_| f32::from_bits(next() as u32)).collect();
            let mut dest = vec![0u16; len];
            convert_f32_slice_to_f16(&source, &mut dest);
            for (i, (s, d)) in source.iter().zip(&dest).enumerate() {
                assert_eq!(
                    *d,
                    f32_to_f16_kv(*s),
                    "bulk mismatch at len {len} index {i}"
                );
            }
        }
    }
}
