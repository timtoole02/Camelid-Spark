//! CPU-vs-CUDA greedy parity gate (Task 5).
//!
//! Closes the "parity comparison was not performed" note for the GPU lane by
//! making the CPU↔CUDA comparison a first-class, gated artifact — mirroring the
//! repo's existing parity-diag pattern (`qa/parity_*_diag.json`), which pairs a
//! token stream with match booleans.
//!
//! ## Tolerance gate (and why)
//!
//! The CUDA Q8_0 decode kernel is written to mirror the CPU reference
//! op-for-op and is compiled with `--fmad=false`, so the f32 logits are
//! **bit-identical** and the greedy argmax — and therefore the token IDs —
//! are identical (see `src/cuda.rs` header). For that path the tolerance is
//! exact: **zero token divergences allowed**.
//!
//! Floating-point matmul *ordering* is not generally associative, so paths that
//! do not preserve the CPU reduction order (e.g. a future cuBLAS-backed matmul,
//! or the offload streaming path if it ever reorders reductions) are **not**
//! expected to be bit-exact. For those, parity is defined on a tolerance:
//! greedy token IDs should still match (argmax is robust to tiny logit noise),
//! and a max absolute logit delta (`logit_abs_tol`) bounds the numeric drift.
//! The gate records which regime applied so divergence is a documented,
//! explained outcome — never a bare pass/fail.

use serde_json::{json, Value};

/// The parity tolerance policy applied to a CPU↔CUDA comparison.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ToleranceGate {
    /// Maximum number of differing greedy token IDs tolerated. `0` for the
    /// bit-exact encoded-linear kernel.
    pub max_token_divergences: usize,
    /// Maximum absolute per-logit delta tolerated for non-bit-exact paths.
    /// `0.0` asserts bit-exactness; a small positive value documents the
    /// accepted numeric drift for reduction-order-divergent paths.
    pub logit_abs_tol: f32,
    /// Human label of the regime (e.g. "bit-exact encoded-linear kernel").
    pub regime: &'static str,
}

impl ToleranceGate {
    /// The default: bit-exact greedy parity, matching the shipped Q8_0 kernel.
    pub fn bit_exact() -> Self {
        ToleranceGate {
            max_token_divergences: 0,
            logit_abs_tol: 0.0,
            regime: "bit-exact encoded-linear kernel (--fmad=false, CPU-mirrored order)",
        }
    }

    /// A tolerance regime for paths that do not preserve the CPU reduction order.
    /// Greedy token IDs must still match; logits may drift up to `logit_abs_tol`.
    pub fn argmax_stable(logit_abs_tol: f32) -> Self {
        ToleranceGate {
            max_token_divergences: 0,
            logit_abs_tol,
            regime: "argmax-stable (reduction order may differ; token IDs must match)",
        }
    }
}

/// Result of comparing two greedy token streams.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenComparison {
    /// Index of the first differing token, if any (token-by-token).
    pub first_divergence: Option<usize>,
    /// Number of positions compared (min of the two lengths).
    pub compared: usize,
    /// Total differing positions within the compared prefix.
    pub divergences: usize,
    /// True when the two streams differ in length (a structural divergence).
    pub length_mismatch: bool,
}

/// Compare two greedy token streams position-by-position, reporting the first
/// divergence index and the divergence count. Pure.
pub fn compare_tokens(cpu: &[u32], cuda: &[u32]) -> TokenComparison {
    let compared = cpu.len().min(cuda.len());
    let mut first_divergence = None;
    let mut divergences = 0;
    for i in 0..compared {
        if cpu[i] != cuda[i] {
            if first_divergence.is_none() {
                first_divergence = Some(i);
            }
            divergences += 1;
        }
    }
    let length_mismatch = cpu.len() != cuda.len();
    if first_divergence.is_none() && length_mismatch {
        // Streams agree on the prefix but one is longer — the divergence is at
        // the point one ran out.
        first_divergence = Some(compared);
    }
    TokenComparison {
        first_divergence,
        compared,
        divergences,
        length_mismatch,
    }
}

/// A complete CPU↔CUDA parity artifact for one (model, fixture) pair.
#[derive(Debug, Clone)]
pub struct ParityArtifact {
    pub model: String,
    pub fixture: String,
    pub gate: ToleranceGate,
    pub comparison: TokenComparison,
    pub cpu_tokens: Vec<u32>,
    pub cuda_tokens: Vec<u32>,
}

impl ParityArtifact {
    /// Gate a comparison: `pass` iff divergences and length-mismatch are within
    /// the tolerance policy.
    pub fn evaluate(
        model: impl Into<String>,
        fixture: impl Into<String>,
        gate: ToleranceGate,
        cpu_tokens: Vec<u32>,
        cuda_tokens: Vec<u32>,
    ) -> Self {
        let comparison = compare_tokens(&cpu_tokens, &cuda_tokens);
        ParityArtifact {
            model: model.into(),
            fixture: fixture.into(),
            gate,
            comparison,
            cpu_tokens,
            cuda_tokens,
        }
    }

    /// Pass iff token divergences ≤ the gate's allowance and lengths match.
    pub fn passed(&self) -> bool {
        !self.comparison.length_mismatch
            && self.comparison.divergences <= self.gate.max_token_divergences
    }

    pub fn verdict(&self) -> &'static str {
        if self.passed() {
            "PASS"
        } else {
            "FAIL"
        }
    }

    /// The parity artifact JSON (schema `camelid.cpu_cuda_parity/v1`).
    pub fn to_json(&self) -> Value {
        json!({
            "schema": "camelid.cpu_cuda_parity/v1",
            "model": self.model,
            "fixture": self.fixture,
            "verdict": self.verdict(),
            "tolerance": {
                "regime": self.gate.regime,
                "max_token_divergences": self.gate.max_token_divergences,
                "logit_abs_tol": self.gate.logit_abs_tol,
            },
            "first_divergence_index": self.comparison.first_divergence,
            "tokens_compared": self.comparison.compared,
            "token_divergences": self.comparison.divergences,
            "length_mismatch": self.comparison.length_mismatch,
            "cpu_token_count": self.cpu_tokens.len(),
            "cuda_token_count": self.cuda_tokens.len(),
            "cpu_tokens": self.cpu_tokens,
            "cuda_tokens": self.cuda_tokens,
        })
    }
}

/// Parse a `generated_tokens` / `backend_generated_tokens` array from a diag-style
/// JSON value (mirrors the existing `qa/parity_*_diag.json` shape). Returns the
/// first array found among the known keys.
pub fn tokens_from_diag(v: &Value) -> Option<Vec<u32>> {
    for key in [
        "generated_tokens",
        "backend_generated_tokens",
        "tokens",
        "cpu_tokens",
        "cuda_tokens",
    ] {
        if let Some(arr) = v.get(key).and_then(Value::as_array) {
            return Some(
                arr.iter()
                    .filter_map(|x| x.as_u64().map(|n| n as u32))
                    .collect(),
            );
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_streams_pass_bit_exact() {
        let a = ParityArtifact::evaluate(
            "m",
            "f",
            ToleranceGate::bit_exact(),
            vec![1, 2, 3, 4],
            vec![1, 2, 3, 4],
        );
        assert!(a.passed());
        assert_eq!(a.verdict(), "PASS");
        assert_eq!(a.comparison.first_divergence, None);
        assert_eq!(a.comparison.divergences, 0);
    }

    #[test]
    fn first_divergence_is_reported() {
        let a = ParityArtifact::evaluate(
            "m",
            "f",
            ToleranceGate::bit_exact(),
            vec![1, 2, 9, 4],
            vec![1, 2, 3, 4],
        );
        assert!(!a.passed());
        assert_eq!(a.verdict(), "FAIL");
        assert_eq!(a.comparison.first_divergence, Some(2));
        assert_eq!(a.comparison.divergences, 1);
    }

    #[test]
    fn length_mismatch_is_a_divergence() {
        let a = ParityArtifact::evaluate(
            "m",
            "f",
            ToleranceGate::bit_exact(),
            vec![1, 2, 3],
            vec![1, 2, 3, 4],
        );
        assert!(!a.passed());
        assert!(a.comparison.length_mismatch);
        assert_eq!(a.comparison.first_divergence, Some(3));
    }

    #[test]
    fn bit_exact_allows_zero_divergences() {
        // Even one differing token fails the bit-exact gate.
        let g = ToleranceGate::bit_exact();
        assert_eq!(g.max_token_divergences, 0);
        let a = ParityArtifact::evaluate("m", "f", g, vec![5], vec![6]);
        assert!(!a.passed());
    }

    #[test]
    fn json_schema_shape() {
        let a = ParityArtifact::evaluate(
            "tinyllama",
            "hello",
            ToleranceGate::bit_exact(),
            vec![1, 2],
            vec![1, 2],
        );
        let j = a.to_json();
        assert_eq!(j["schema"], "camelid.cpu_cuda_parity/v1");
        assert_eq!(j["verdict"], "PASS");
        assert_eq!(j["first_divergence_index"], Value::Null);
        assert_eq!(j["tolerance"]["max_token_divergences"], 0);
    }

    #[test]
    fn tokens_from_diag_reads_known_keys() {
        let v = json!({"backend_generated_tokens": [10, 20, 30]});
        assert_eq!(tokens_from_diag(&v), Some(vec![10, 20, 30]));
        assert_eq!(tokens_from_diag(&json!({"nope": 1})), None);
    }
}
