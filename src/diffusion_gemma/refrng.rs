//! Reference-order host RNG for the DiffusionGemma Entropy-Bound denoiser.
//!
//! The reference sampler (`diffusion_generate_entropy_bound`,
//! `examples/diffusion/diffusion.cpp` at the pinned llama.cpp checkout)
//! draws every random value on the host from `std::mt19937 rng(seed)`
//! through `std::uniform_int_distribution<int32_t>` (canvas init + renoise)
//! and `std::uniform_real_distribution<float>` (multinomial draws). The
//! Mersenne Twister engine is fully standard-specified, but the
//! distributions are implementation-defined — these are ports of LIBC++'S
//! algorithms specifically (`__random/uniform_int_distribution.h`,
//! `__random/generate_canonical.h`; Apple clang is the pinned reference's
//! toolchain), gated bit-for-bit against `scripts/dg-rng-dump.cpp` compiled
//! with that same toolchain. All algorithm semantics credit the LLVM libc++
//! authors (Apache-2.0 WITH LLVM-exception) and the llama.cpp authors (MIT).

/// `std::mt19937`: the standard 32-bit Mersenne Twister (MT19937).
pub struct Mt19937 {
    state: [u32; 624],
    index: usize,
}

impl Mt19937 {
    /// `std::mt19937 rng(seed)` — single-value constructor semantics.
    pub fn new(seed: u32) -> Self {
        let mut state = [0u32; 624];
        state[0] = seed;
        for i in 1..624 {
            let prev = state[i - 1];
            state[i] = 1_812_433_253u32
                .wrapping_mul(prev ^ (prev >> 30))
                .wrapping_add(i as u32);
        }
        Self { state, index: 624 }
    }

    fn twist(&mut self) {
        const M: usize = 397;
        const MATRIX_A: u32 = 0x9908_b0df;
        const UPPER_MASK: u32 = 0x8000_0000;
        const LOWER_MASK: u32 = 0x7fff_ffff;
        for i in 0..624 {
            let y = (self.state[i] & UPPER_MASK) | (self.state[(i + 1) % 624] & LOWER_MASK);
            let mut next = self.state[(i + M) % 624] ^ (y >> 1);
            if y & 1 != 0 {
                next ^= MATRIX_A;
            }
            self.state[i] = next;
        }
        self.index = 0;
    }

    /// One raw engine draw (tempered).
    pub fn next_u32(&mut self) -> u32 {
        if self.index >= 624 {
            self.twist();
        }
        let mut y = self.state[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }
}

/// `std::uniform_int_distribution<int32_t>{a, b}(mt19937)` — libc++
/// algorithm. For a full-range engine (mt19937's working range wraps to 0),
/// `__independent_bits_engine::__eval` collapses to ONE masked raw draw of
/// the distribution's bit width `w`, rejected until `< b - a + 1`.
pub fn uniform_int_i32(rng: &mut Mt19937, a: i32, b: i32) -> i32 {
    debug_assert!(a <= b);
    let rp64 = (b as i64 - a as i64 + 1) as u64;
    if rp64 == 1 {
        return a;
    }
    if rp64 > u32::MAX as u64 {
        // full 32-bit range: a single raw draw (mask covers the whole word)
        return (rng.next_u32() as i64 + a as i64) as i32;
    }
    let rp = rp64 as u32;
    // bit width of the smallest mask covering [0, rp)
    let mut w = 32 - rp.leading_zeros() - 1;
    if rp & (u32::MAX >> (32 - w)) != 0 {
        w += 1;
    }
    let mask = if w >= 32 { u32::MAX } else { (1u32 << w) - 1 };
    loop {
        let u = rng.next_u32() & mask;
        if u < rp {
            return (u as i64 + a as i64) as i32;
        }
    }
}

/// `std::uniform_real_distribution<float>{0,1}(mt19937)` — libc++
/// `generate_canonical<float, 24>` over a 32-bit engine takes exactly ONE
/// raw draw: `(float)x / 2^32` (the cast rounds to nearest-even; the divide
/// is an exact exponent shift). NOTE the well-known libc++ quirk: the result
/// CAN be exactly 1.0f when the draw rounds up to 2^32 — preserved, since
/// the reference sampler inherits it.
pub fn uniform_real_f32_01(rng: &mut Mt19937) -> f32 {
    rng.next_u32() as f32 / 4_294_967_296.0f32
}

/// The EB denoiser's pre-drawn randomness, in the reference's exact stream
/// order: C canvas-init token draws first, then per STEP and per position one
/// `u` (multinomial) draw and one `renoise` token draw, interleaved.
pub struct EbDraws {
    pub canvas_init: Vec<i32>,
    /// `u[step][pos]`
    pub u: Vec<Vec<f32>>,
    /// `renoise[step][pos]`
    pub renoise: Vec<Vec<i32>>,
}

pub fn eb_draws(seed: u32, n_vocab: i32, c: usize, n_steps: usize) -> EbDraws {
    let mut rng = Mt19937::new(seed);
    let canvas_init: Vec<i32> = (0..c)
        .map(|_| uniform_int_i32(&mut rng, 0, n_vocab - 1))
        .collect();
    let mut u = Vec::with_capacity(n_steps);
    let mut renoise = Vec::with_capacity(n_steps);
    for _ in 0..n_steps {
        let mut us = Vec::with_capacity(c);
        let mut rs = Vec::with_capacity(c);
        for _ in 0..c {
            us.push(uniform_real_f32_01(&mut rng));
            rs.push(uniform_int_i32(&mut rng, 0, n_vocab - 1));
        }
        u.push(us);
        renoise.push(rs);
    }
    EbDraws {
        canvas_init,
        u,
        renoise,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The C++ standard's normative check: the 10000th consecutive
    /// invocation of a default-constructed (seed 5489) mt19937 is
    /// 4123659995.
    #[test]
    fn mt19937_matches_standard_checkpoint() {
        let mut rng = Mt19937::new(5489);
        let mut last = 0u32;
        for _ in 0..10_000 {
            last = rng.next_u32();
        }
        assert_eq!(last, 4_123_659_995);
    }

    #[test]
    fn uniform_int_power_of_two_range_is_single_masked_draw() {
        // rp = 2^18 exactly -> w = 18, mask = 0x3FFFF, no rejection possible
        let mut a = Mt19937::new(7);
        let mut b = Mt19937::new(7);
        for _ in 0..1000 {
            let want = (b.next_u32() & 0x3_FFFF) as i32;
            assert_eq!(uniform_int_i32(&mut a, 0, (1 << 18) - 1), want);
        }
    }

    /// Pinned ground truth from `scripts/dg-rng-dump.cpp` compiled with the
    /// reference toolchain (Apple clang / libc++) and run as
    /// `dg-rng-dump 0 262144 256 1`: the first canvas-init draws and the
    /// first interleaved step-0 u/renoise draws of the EB stream.
    #[test]
    fn eb_stream_matches_libcxx_ground_truth() {
        let d = eb_draws(0, 262_144, 256, 1);
        assert_eq!(&d.canvas_init[..4], &[199_340, 43_567, 173_685, 117_952]);
        assert_eq!(d.u[0][0].to_bits(), 0x3F29_0122);
        assert_eq!(d.u[0][1].to_bits(), 0x3E94_850D);
        assert_eq!(d.renoise[0][0], 116_551);
        assert_eq!(d.renoise[0][1], 169_112);
    }

    #[test]
    fn uniform_real_is_one_draw_div_2_32() {
        let mut a = Mt19937::new(11);
        let mut b = Mt19937::new(11);
        for _ in 0..1000 {
            let want = b.next_u32() as f32 / 4_294_967_296.0f32;
            assert_eq!(uniform_real_f32_01(&mut a).to_bits(), want.to_bits());
        }
    }
}
