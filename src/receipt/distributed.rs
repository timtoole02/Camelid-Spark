//! Distributed parity receipts.
//!
//! A [`DistributedParityReceipt`] records that ONE deterministic request, run across a
//! pipeline-parallel layer-sharded topology, produced output **token-identical** to a
//! single-node reference (single-node Camelid, or a llama.cpp oracle when the model fits
//! nowhere whole). It is the gate for the distributed lane: a distributed config is not
//! "working" because activations crossed the wire and fluent text came out — it is working
//! only when this receipt's `generated_token_ids_match` is true with
//! `first_divergent_generated_token_index == -1` (see DISTRIBUTED_RECON.md / DECISIONS.md).
//!
//! The same two rules from [`super`] apply: a receipt proves one request matched a
//! reference, it is NOT a support promotion; and it is only meaningful for deterministic
//! (greedy, temperature 0) runs. A distributed config never inherits the single-node
//! support row by resemblance — it earns its own receipt.
//!
//! This type deliberately reuses the parent module's primitives — [`LaneIdentity`],
//! [`canonical_json`](super::canonical_json), [`sha256_hex`](super::sha256_hex), and the
//! `NO_DIVERGENCE` sentinel — so the digest/sealing discipline is identical to the
//! single-node `camelid.parity-receipt/v1` and there is one source of truth for canonical
//! JSON.

use serde::{Deserialize, Serialize};
use serde_json::Value;

use super::{canonical_json, sha256_hex, LaneIdentity, ReceiptError, NO_DIVERGENCE};

/// Schema identifier stamped into every v1 distributed receipt.
pub const DISTRIBUTED_RECEIPT_SCHEMA_V1: &str = "camelid.distributed-parity-receipt/v1";

/// One node in the pipeline topology the receipt describes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TopologyNode {
    /// Stable node label, e.g. `coordinator`, `shard-a`.
    pub node: String,
    /// Where it ran, e.g. `127.0.0.1` or `127.0.0.1:9311`.
    pub host: String,
    /// `coordinator` (embeddings in + final norm/output/sampling out) or `shard`.
    pub role: String,
    /// `[start, end)` decoder layers this node owns. `None` for a pure coordinator that
    /// owns no decoder layers; `Some` for shards (and for a coordinator that also runs a
    /// head block).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layer_range: Option<[u32; 2]>,
}

impl TopologyNode {
    pub fn coordinator(node: &str, host: &str, layer_range: Option<[u32; 2]>) -> Self {
        Self {
            node: node.to_string(),
            host: host.to_string(),
            role: "coordinator".to_string(),
            layer_range,
        }
    }

    pub fn shard(node: &str, host: &str, layer_range: [u32; 2]) -> Self {
        Self {
            node: node.to_string(),
            host: host.to_string(),
            role: "shard".to_string(),
            layer_range: Some(layer_range),
        }
    }
}

/// The comparison of a distributed run against its single-node reference.
///
/// Built by [`ParityVerdict::compare`] so the match fields and the first-divergence index
/// are derived from the token id streams, never hand-set — a receipt cannot claim a match
/// it did not compute.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParityVerdict {
    pub prompt_tokens_match: bool,
    pub generated_token_ids_match: bool,
    pub generated_text_match: bool,
    /// `-1` ([`NO_DIVERGENCE`]) when the generated token streams are identical; otherwise
    /// the index of the first generated token that differs.
    pub first_divergent_generated_token_index: i64,
}

impl ParityVerdict {
    /// Compare a distributed run against a single-node reference. The first-divergence
    /// index is computed over the generated token ids; a length mismatch with no earlier
    /// divergence reports the first index past the shorter stream (so truncation is a
    /// divergence, not a silent pass).
    pub fn compare(
        reference_prompt_ids: &[u32],
        reference_generated_ids: &[u32],
        reference_text: &str,
        distributed_prompt_ids: &[u32],
        distributed_generated_ids: &[u32],
        distributed_text: &str,
    ) -> Self {
        let first_divergent = first_divergence(reference_generated_ids, distributed_generated_ids);
        Self {
            prompt_tokens_match: reference_prompt_ids == distributed_prompt_ids,
            generated_token_ids_match: reference_generated_ids == distributed_generated_ids,
            generated_text_match: reference_text == distributed_text,
            first_divergent_generated_token_index: first_divergent,
        }
    }

    /// True only when every parity dimension matched and no token diverged. This is the
    /// distributed lane's gate condition.
    pub fn is_token_identical(&self) -> bool {
        self.prompt_tokens_match
            && self.generated_token_ids_match
            && self.generated_text_match
            && self.first_divergent_generated_token_index == NO_DIVERGENCE
    }
}

/// Index of the first differing element, or [`NO_DIVERGENCE`] when one stream is a prefix
/// of the other AND they are the same length. A length difference with no earlier
/// divergence reports the first index past the shorter stream.
fn first_divergence(reference: &[u32], distributed: &[u32]) -> i64 {
    for (i, (a, b)) in reference.iter().zip(distributed.iter()).enumerate() {
        if a != b {
            return i as i64;
        }
    }
    if reference.len() == distributed.len() {
        NO_DIVERGENCE
    } else {
        reference.len().min(distributed.len()) as i64
    }
}

/// A sealed, fingerprintable record of one distributed run vs its single-node reference.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DistributedParityReceipt {
    /// Always [`DISTRIBUTED_RECEIPT_SCHEMA_V1`].
    pub schema: String,
    /// SHA-256 of the canonical receipt body (every field except this one).
    pub receipt_id: String,
    /// RFC 3339 UTC timestamp of receipt creation.
    pub created_utc: String,
    /// Human label for the exact distributed configuration, e.g.
    /// `loopback-2shard-tinyllama-q8`.
    pub config_id: String,
    /// Identity of the exact model lane (reused from the single-node receipt framework).
    pub lane: LaneIdentity,
    /// What the distributed output was compared against, e.g. `single-node-camelid` or
    /// `llama.cpp`.
    pub reference: String,
    pub prompt: String,
    pub seed: Option<u64>,
    pub temperature: f64,
    pub max_tokens: u32,
    pub topology: Vec<TopologyNode>,
    pub prompt_token_ids: Vec<u32>,
    pub generated_token_ids: Vec<u32>,
    pub generated_text: String,
    pub completion_tokens: u32,
    /// `false` for any run with nondeterministic sampling; such a receipt is never
    /// presented as verifiable (rule 2 in [`super`]).
    pub reproducible: bool,
    pub prompt_tokens_match: bool,
    pub generated_token_ids_match: bool,
    pub generated_text_match: bool,
    /// `-1` ([`NO_DIVERGENCE`]) when token-identical.
    pub first_divergent_generated_token_index: i64,
}

/// Everything needed to build a receipt except the derived/sealed fields.
pub struct DistributedRunRecord {
    pub config_id: String,
    pub lane: LaneIdentity,
    pub reference: String,
    pub prompt: String,
    pub seed: Option<u64>,
    pub temperature: f64,
    pub max_tokens: u32,
    pub topology: Vec<TopologyNode>,
    pub prompt_token_ids: Vec<u32>,
    pub generated_token_ids: Vec<u32>,
    pub generated_text: String,
}

impl DistributedParityReceipt {
    /// Build and seal a receipt from a distributed run and its parity verdict. The verdict
    /// must have been computed (not hand-set) via [`ParityVerdict::compare`]; the match
    /// fields are copied from it verbatim. `created_utc` is supplied by the caller so the
    /// receipt is reproducible and testable.
    pub fn build(
        record: DistributedRunRecord,
        verdict: &ParityVerdict,
        created_utc: String,
    ) -> Result<Self, ReceiptError> {
        let completion_tokens = record.generated_token_ids.len() as u32;
        // Greedy (temperature 0, no sampling) is the only reproducible mode.
        let reproducible = record.temperature == 0.0 && record.seed.is_none();
        let mut receipt = Self {
            schema: DISTRIBUTED_RECEIPT_SCHEMA_V1.to_string(),
            receipt_id: String::new(),
            created_utc,
            config_id: record.config_id,
            lane: record.lane,
            reference: record.reference,
            prompt: record.prompt,
            seed: record.seed,
            temperature: record.temperature,
            max_tokens: record.max_tokens,
            topology: record.topology,
            prompt_token_ids: record.prompt_token_ids,
            generated_token_ids: record.generated_token_ids,
            generated_text: record.generated_text,
            completion_tokens,
            reproducible,
            prompt_tokens_match: verdict.prompt_tokens_match,
            generated_token_ids_match: verdict.generated_token_ids_match,
            generated_text_match: verdict.generated_text_match,
            first_divergent_generated_token_index: verdict.first_divergent_generated_token_index,
        };
        receipt.seal()?;
        Ok(receipt)
    }

    /// Canonical serialization of the body: every field except `receipt_id`, recursively
    /// key-sorted, compact. The byte string the digest is computed over.
    pub fn canonical_body(&self) -> Result<String, ReceiptError> {
        let mut value = serde_json::to_value(self)?;
        if let Value::Object(map) = &mut value {
            map.remove("receipt_id");
        }
        Ok(canonical_json(&value))
    }

    /// Recompute the digest the `receipt_id` field should hold.
    pub fn compute_receipt_id(&self) -> Result<String, ReceiptError> {
        Ok(sha256_hex(self.canonical_body()?.as_bytes()))
    }

    /// Populate `receipt_id` from the canonical body. Called by [`Self::build`].
    pub fn seal(&mut self) -> Result<(), ReceiptError> {
        self.receipt_id = self.compute_receipt_id()?;
        Ok(())
    }

    /// Cheap tamper check: recompute the digest and confirm it matches the stored id.
    pub fn verify_self_digest(&self) -> Result<(), ReceiptError> {
        let computed = self.compute_receipt_id()?;
        if computed == self.receipt_id {
            Ok(())
        } else {
            Err(ReceiptError::DigestMismatch {
                stored: self.receipt_id.clone(),
                computed,
            })
        }
    }

    /// The lane gate: token-identical to the reference AND a reproducible (greedy) run.
    pub fn is_validated(&self) -> bool {
        self.reproducible
            && self.prompt_tokens_match
            && self.generated_token_ids_match
            && self.generated_text_match
            && self.first_divergent_generated_token_index == NO_DIVERGENCE
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lane() -> LaneIdentity {
        LaneIdentity {
            model_id: "tinyllama".into(),
            gguf_sha256: "abc".into(),
            gguf_filename: "tinyllama.Q8_0.gguf".into(),
            quantization: "Q8_0".into(),
            architecture: "llama".into(),
            tokenizer_kind: "llama".into(),
            tokenizer_sha256: None,
            camelid_version: "test".into(),
            camelid_commit: "test".into(),
        }
    }

    fn record(generated: Vec<u32>) -> DistributedRunRecord {
        DistributedRunRecord {
            config_id: "loopback-2shard-tinyllama-q8".into(),
            lane: lane(),
            reference: "single-node-camelid".into(),
            prompt: "hello".into(),
            seed: None,
            temperature: 0.0,
            max_tokens: 50,
            topology: vec![
                TopologyNode::coordinator("coordinator", "127.0.0.1", Some([0, 11])),
                TopologyNode::shard("shard-b", "127.0.0.1:9312", [11, 22]),
            ],
            prompt_token_ids: vec![1, 22172],
            generated_token_ids: generated.clone(),
            generated_text: format!("{generated:?}"),
        }
    }

    #[test]
    fn identical_streams_have_no_divergence() {
        let v = ParityVerdict::compare(&[1, 2], &[5, 6, 7], "x", &[1, 2], &[5, 6, 7], "x");
        assert!(v.is_token_identical());
        assert_eq!(v.first_divergent_generated_token_index, NO_DIVERGENCE);
    }

    #[test]
    fn first_divergence_is_reported_not_smoothed() {
        let v = ParityVerdict::compare(&[1], &[5, 6, 9], "a", &[1], &[5, 6, 7], "b");
        assert!(!v.generated_token_ids_match);
        assert!(!v.generated_text_match);
        assert_eq!(v.first_divergent_generated_token_index, 2);
        assert!(!v.is_token_identical());
    }

    #[test]
    fn truncation_counts_as_divergence() {
        let v = ParityVerdict::compare(&[1], &[5, 6, 7], "x", &[1], &[5, 6], "x");
        assert!(!v.generated_token_ids_match);
        assert_eq!(v.first_divergent_generated_token_index, 2);
    }

    #[test]
    fn build_seals_and_self_verifies() {
        let gen = vec![29892, 322, 769];
        let verdict = ParityVerdict::compare(&[1, 22172], &gen, "out", &[1, 22172], &gen, "out");
        let receipt =
            DistributedParityReceipt::build(record(gen), &verdict, "1970-01-01T00:00:00Z".into())
                .unwrap();
        assert_eq!(receipt.schema, DISTRIBUTED_RECEIPT_SCHEMA_V1);
        assert!(!receipt.receipt_id.is_empty());
        assert!(receipt.verify_self_digest().is_ok());
        assert!(receipt.is_validated());
        assert_eq!(receipt.completion_tokens, 3);
    }

    #[test]
    fn tamper_is_detected() {
        let gen = vec![1, 2, 3];
        let verdict = ParityVerdict::compare(&[1], &gen, "o", &[1], &gen, "o");
        let mut receipt =
            DistributedParityReceipt::build(record(gen), &verdict, "1970-01-01T00:00:00Z".into())
                .unwrap();
        receipt.generated_text = "tampered".into();
        assert!(receipt.verify_self_digest().is_err());
    }

    #[test]
    fn round_trips_through_json() {
        let gen = vec![29892, 322];
        let verdict = ParityVerdict::compare(&[1, 22172], &gen, "t", &[1, 22172], &gen, "t");
        let receipt =
            DistributedParityReceipt::build(record(gen), &verdict, "1970-01-01T00:00:00Z".into())
                .unwrap();
        let json = serde_json::to_string(&receipt).unwrap();
        let back: DistributedParityReceipt = serde_json::from_str(&json).unwrap();
        assert_eq!(receipt, back);
        assert!(back.verify_self_digest().is_ok());
    }
}
