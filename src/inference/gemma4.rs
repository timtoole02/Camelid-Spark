//! Gemma 4 (`general.architecture = "gemma4"`) forward-pass primitives.
//!
//! These are the scalar/vector building blocks that differ from the Llama-family
//! path. They are deliberately small, pure, and unit-tested against the reference
//! `transformers` implementation (`modeling_gemma4.py`) so the values are locked
//! down before they are wired into the optimized decode path.
//!
//! Differences from the Llama path captured here:
//! - **Embedding scale**: token embeddings are multiplied by `sqrt(hidden_size)`.
//! - **RMSNorm**: Gemma 4 uses a *standard* `x_normed * weight` RMSNorm (the
//!   weight is initialized to ones), unlike Gemma 2/3 which used `(1 + weight)`.
//!   So the engine's existing `rms_norm` applies directly — no weight folding.
//! - **FFN**: GeGLU with `gelu_pytorch_tanh`, i.e. `down(gelu_tanh(gate) * up)`,
//!   where Llama uses SwiGLU (`silu`).
//! - **Final logits**: soft-capped as `cap * tanh(logits / cap)` (cap = 30).
//!
//! QK-norm, the per-layer-type head dims, sliding-window masking and the dual
//! RoPE bases are described by [`crate::model::Gemma4Metadata`]; their wiring into
//! the layer loop is layered on top of these primitives.
//!
// These primitives are validated by their unit tests but not yet called from the
// decode path; allow dead_code until the Gemma 4 layer loop is wired in.
#![allow(dead_code)]

/// The embedding scale applied to token embeddings: `sqrt(hidden_size)`.
///
/// Gemma multiplies the looked-up embeddings by this normalizer before the first
/// decoder layer. Computed in f32 to match the reference, which scales in float32
/// regardless of the activation dtype.
pub(crate) fn embedding_scale(hidden_size: u32) -> f32 {
    (hidden_size as f32).sqrt()
}

/// `gelu_pytorch_tanh` — the tanh approximation of GELU used by Gemma's GeGLU MLP.
///
/// `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))`
// The constants are written to full precision to match the reference exactly; the
// extra digits are intentional (don't let clippy round them and shift the output).
#[allow(clippy::excessive_precision)]
pub(crate) fn gelu_tanh(x: f32) -> f32 {
    const SQRT_2_OVER_PI: f32 = 0.797_884_56; // sqrt(2/pi)
    const COEFF: f32 = 0.044_715;
    let inner = SQRT_2_OVER_PI * (x + COEFF * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

/// GeGLU activation for the Gemma MLP intermediate: `gelu_tanh(gate) * up`,
/// written element-wise into `out`. `gate`, `up`, and `out` must be the same
/// length (the intermediate size). This is the input to `down_proj`.
pub(crate) fn geglu_into(gate: &[f32], up: &[f32], out: &mut [f32]) {
    debug_assert_eq!(gate.len(), up.len());
    debug_assert_eq!(gate.len(), out.len());
    for ((o, &g), &u) in out.iter_mut().zip(gate.iter()).zip(up.iter()) {
        *o = gelu_tanh(g) * u;
    }
}

/// Soft-cap a logit vector in place: `x <- cap * tanh(x / cap)`.
///
/// Applied to the final logits before sampling when
/// `Gemma4Metadata::final_logit_softcapping` is present (cap = 30 for Gemma 4).
/// A non-finite or non-positive cap is treated as "disabled" and leaves the
/// logits untouched.
pub(crate) fn soft_cap_in_place(logits: &mut [f32], cap: f32) {
    if !cap.is_finite() || cap <= 0.0 {
        return;
    }
    for l in logits.iter_mut() {
        *l = cap * (*l / cap).tanh();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn embedding_scale_matches_sqrt_hidden() {
        // E4B hidden_size = 2560, 12B = 3840.
        assert!(approx(embedding_scale(2560), 50.596_443, 1e-3));
        assert!(approx(embedding_scale(3840), 61.967_734, 1e-3));
        assert_eq!(embedding_scale(256), 16.0);
    }

    #[test]
    fn gelu_tanh_matches_reference_values() {
        // Reference gelu_pytorch_tanh values.
        assert!(approx(gelu_tanh(0.0), 0.0, 1e-6));
        assert!(approx(gelu_tanh(1.0), 0.841_192, 1e-4));
        assert!(approx(gelu_tanh(-1.0), -0.158_808, 1e-4));
        assert!(approx(gelu_tanh(2.0), 1.954_598, 1e-4));
        // Saturates toward identity for large positive, toward 0 for large negative.
        assert!(approx(gelu_tanh(8.0), 8.0, 1e-3));
        assert!(gelu_tanh(-8.0).abs() < 1e-3);
    }

    #[test]
    fn geglu_combines_gate_and_up() {
        let gate = [0.0_f32, 1.0, 2.0];
        let up = [3.0_f32, 4.0, 5.0];
        let mut out = [0.0_f32; 3];
        geglu_into(&gate, &up, &mut out);
        assert!(approx(out[0], 0.0, 1e-6)); // gelu(0)=0
        assert!(approx(out[1], gelu_tanh(1.0) * 4.0, 1e-6));
        assert!(approx(out[2], gelu_tanh(2.0) * 5.0, 1e-6));
    }

    #[test]
    fn soft_cap_saturates_and_passes_small_values() {
        let cap = 30.0_f32;
        let mut logits = [0.0_f32, 30.0, 300.0, -300.0];
        soft_cap_in_place(&mut logits, cap);
        assert!(approx(logits[0], 0.0, 1e-6)); // 0 -> 0
        assert!(approx(logits[1], 30.0 * 1.0_f32.tanh(), 1e-4)); // ~22.84
        assert!(approx(logits[2], cap, 1e-3)); // saturates to +cap
        assert!(approx(logits[3], -cap, 1e-3)); // saturates to -cap
    }

    #[test]
    fn soft_cap_disabled_for_nonpositive_cap() {
        let mut logits = [1.0_f32, 2.0, 3.0];
        let before = logits;
        soft_cap_in_place(&mut logits, 0.0);
        assert_eq!(logits, before);
        soft_cap_in_place(&mut logits, f32::NAN);
        assert_eq!(logits, before);
    }
}
