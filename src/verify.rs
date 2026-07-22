//! User-facing exact-row verification built on Camelid's receipt replay path.
//!
//! A verified report means one pinned deterministic request reproduced the
//! reference-anchored output for one exact GGUF. It is not a broad model,
//! backend, performance, or support claim.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::receipt::{
    canonical_json, rfc3339_utc_now, sha256_file_hex, ReceiptRequest, ReceiptResult, NO_DIVERGENCE,
};

pub const VERIFY_PROFILE_SCHEMA_V1: &str = "camelid.verify-profile/v1";
pub const VERIFY_REPORT_SCHEMA_V1: &str = "camelid.verify-report/v1";

const LLAMA32_1B_Q8_PROFILE_JSON: &str = include_str!("verify/llama-3.2-1b-instruct-q8_0.json");

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerificationProfile {
    pub schema: String,
    pub profile_id: String,
    pub model: VerificationModel,
    pub request: ReceiptRequest,
    pub expected: ReceiptResult,
    pub evidence: VerificationEvidence,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerificationModel {
    pub model_id: String,
    pub gguf_filename: String,
    pub gguf_sha256: String,
    pub quantization: String,
    pub architecture: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerificationEvidence {
    pub oracle: String,
    pub capability_receipt: String,
    pub expected_result_source: String,
    pub claim: String,
}

impl VerificationProfile {
    fn computed_profile_id(&self) -> Result<String, serde_json::Error> {
        let mut value = serde_json::to_value(self)?;
        if let serde_json::Value::Object(map) = &mut value {
            map.remove("profile_id");
        }
        Ok(sha256_hex(canonical_json(&value).as_bytes()))
    }

    fn validate(&self) -> Result<(), String> {
        if self.schema != VERIFY_PROFILE_SCHEMA_V1 {
            return Err(format!(
                "unknown verification profile schema {:?}",
                self.schema
            ));
        }
        let computed = self
            .computed_profile_id()
            .map_err(|err| format!("could not compute verification profile id: {err}"))?;
        if computed != self.profile_id {
            return Err(format!(
                "verification profile digest mismatch: stored {}, computed {computed}",
                self.profile_id
            ));
        }
        if self.request.temperature != 0.0
            || self.request.top_p.is_some()
            || self.request.top_k.is_some()
        {
            return Err("verification profiles must use deterministic greedy decoding".to_string());
        }
        if self.expected.completion_tokens as usize != self.expected.generated_token_ids.len() {
            return Err(
                "verification profile completion_tokens must match generated_token_ids".to_string(),
            );
        }
        Ok(())
    }
}

pub fn built_in_profiles() -> Result<Vec<VerificationProfile>, String> {
    let profile: VerificationProfile = serde_json::from_str(LLAMA32_1B_Q8_PROFILE_JSON)
        .map_err(|err| format!("built-in verification profile does not parse: {err}"))?;
    profile.validate()?;
    Ok(vec![profile])
}

pub fn profile_for_sha256(sha256: &str) -> Result<Option<VerificationProfile>, String> {
    Ok(built_in_profiles()?
        .into_iter()
        .find(|profile| profile.model.gguf_sha256.eq_ignore_ascii_case(sha256)))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VerificationOutcome {
    Verified,
    NotVerified,
    NoProfile,
}

impl VerificationOutcome {
    pub fn exit_code(self) -> i32 {
        match self {
            Self::Verified => 0,
            Self::NotVerified => 1,
            Self::NoProfile => 2,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerificationComparison {
    pub prompt_tokens_match: bool,
    pub generated_tokens_match: bool,
    pub generated_text_match: bool,
    pub completion_tokens_match: bool,
    pub finish_reason_match: bool,
    pub first_divergent_prompt_token_index: i64,
    pub first_divergent_generated_token_index: i64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VerificationReport {
    pub schema: String,
    pub report_id: String,
    pub created_utc: String,
    pub outcome: VerificationOutcome,
    pub model: VerificationModel,
    pub profile_id: Option<String>,
    pub comparison: Option<VerificationComparison>,
    pub detail: String,
}

impl VerificationReport {
    fn new(
        outcome: VerificationOutcome,
        model: VerificationModel,
        profile_id: Option<String>,
        comparison: Option<VerificationComparison>,
        detail: String,
    ) -> Result<Self, serde_json::Error> {
        let mut report = Self {
            schema: VERIFY_REPORT_SCHEMA_V1.to_string(),
            report_id: String::new(),
            created_utc: rfc3339_utc_now(),
            outcome,
            model,
            profile_id,
            comparison,
            detail,
        };
        report.seal()?;
        Ok(report)
    }

    pub fn seal(&mut self) -> Result<(), serde_json::Error> {
        let mut value = serde_json::to_value(&*self)?;
        if let serde_json::Value::Object(map) = &mut value {
            map.remove("report_id");
        }
        self.report_id = sha256_hex(canonical_json(&value).as_bytes());
        Ok(())
    }

    pub fn verify_self_digest(&self) -> Result<(), String> {
        let mut value = serde_json::to_value(self)
            .map_err(|err| format!("could not serialize verification report: {err}"))?;
        if let serde_json::Value::Object(map) = &mut value {
            map.remove("report_id");
        }
        let computed = sha256_hex(canonical_json(&value).as_bytes());
        if computed == self.report_id {
            Ok(())
        } else {
            Err(format!(
                "verification report digest mismatch: stored {}, computed {computed}",
                self.report_id
            ))
        }
    }
}

pub fn compare(profile: &VerificationProfile, actual: &ReceiptResult) -> VerificationComparison {
    let prompt_divergence = crate::receipt::verify::first_divergent_index(
        &profile.expected.prompt_token_ids,
        &actual.prompt_token_ids,
    );
    let generated_divergence = crate::receipt::verify::first_divergent_index(
        &profile.expected.generated_token_ids,
        &actual.generated_token_ids,
    );
    VerificationComparison {
        prompt_tokens_match: prompt_divergence == NO_DIVERGENCE,
        generated_tokens_match: generated_divergence == NO_DIVERGENCE,
        generated_text_match: profile.expected.generated_text == actual.generated_text,
        completion_tokens_match: profile.expected.completion_tokens == actual.completion_tokens,
        finish_reason_match: profile.expected.finish_reason == actual.finish_reason,
        first_divergent_prompt_token_index: prompt_divergence,
        first_divergent_generated_token_index: generated_divergence,
    }
}

pub fn evaluate(
    profile: &VerificationProfile,
    actual: &ReceiptResult,
) -> Result<VerificationReport, serde_json::Error> {
    let comparison = compare(profile, actual);
    let verified = comparison.prompt_tokens_match
        && comparison.generated_tokens_match
        && comparison.generated_text_match
        && comparison.completion_tokens_match
        && comparison.finish_reason_match;
    VerificationReport::new(
        if verified {
            VerificationOutcome::Verified
        } else {
            VerificationOutcome::NotVerified
        },
        profile.model.clone(),
        Some(profile.profile_id.clone()),
        Some(comparison),
        if verified {
            "Pinned deterministic request matched the reference-anchored verification profile. This proves one request for one exact GGUF, not broad model support."
        } else {
            "Pinned deterministic request did not match the reference-anchored verification profile."
        }
        .to_string(),
    )
}

pub async fn run(gguf: &Path, threads: Option<usize>) -> Result<VerificationReport, String> {
    let sha256 = sha256_file_hex(gguf).map_err(|err| err.to_string())?;
    let Some(profile) = profile_for_sha256(&sha256)? else {
        let filename = gguf
            .file_name()
            .map(|name| name.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown.gguf".to_string());
        return VerificationReport::new(
            VerificationOutcome::NoProfile,
            VerificationModel {
                model_id: filename.clone(),
                gguf_filename: filename,
                gguf_sha256: sha256,
                quantization: "unknown".to_string(),
                architecture: "unknown".to_string(),
            },
            None,
            None,
            "No built-in verification profile matches this exact GGUF. Camelid abstained."
                .to_string(),
        )
        .map_err(|err| format!("could not build verification report: {err}"));
    };
    let replay = crate::api::replay_receipt_request(gguf, threads, &profile.request).await?;
    if replay.lane.gguf_sha256 != profile.model.gguf_sha256 {
        return Err(
            "replay loaded a different GGUF identity than the selected profile".to_string(),
        );
    }
    evaluate(&profile, &replay.result)
        .map_err(|err| format!("could not build verification report: {err}"))
}

pub fn default_report_path(gguf: &Path) -> PathBuf {
    let stem = gguf
        .file_stem()
        .map(|stem| stem.to_string_lossy().to_string())
        .unwrap_or_else(|| "model".to_string());
    PathBuf::from(format!("{stem}.verify.json"))
}

pub fn write_report(path: &Path, report: &VerificationReport) -> Result<(), String> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent)
        .map_err(|err| format!("could not create {}: {err}", parent.display()))?;
    let temp = path.with_extension(format!("verify.json.{}.tmp", std::process::id()));
    let mut json = serde_json::to_string_pretty(report)
        .map_err(|err| format!("could not serialize verification report: {err}"))?;
    json.push('\n');
    std::fs::write(&temp, json)
        .map_err(|err| format!("could not write {}: {err}", temp.display()))?;
    if path.exists() {
        std::fs::remove_file(path)
            .map_err(|err| format!("could not replace {}: {err}", path.display()))?;
    }
    std::fs::rename(&temp, path).map_err(|err| {
        let _ = std::fs::remove_file(&temp);
        format!("could not publish {}: {err}", path.display())
    })
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile() -> VerificationProfile {
        built_in_profiles().expect("profiles").remove(0)
    }

    #[test]
    fn built_in_profile_is_digest_sealed_and_exact_hash_selected() {
        let profile = profile();
        assert_eq!(profile.schema, VERIFY_PROFILE_SCHEMA_V1);
        assert_eq!(profile.computed_profile_id().unwrap(), profile.profile_id);
        assert_eq!(
            profile_for_sha256(&profile.model.gguf_sha256)
                .unwrap()
                .unwrap(),
            profile
        );
        assert!(profile_for_sha256(&"0".repeat(64)).unwrap().is_none());
    }

    #[test]
    fn exact_result_verifies_and_report_is_digest_sealed() {
        let profile = profile();
        let report = evaluate(&profile, &profile.expected).unwrap();
        assert_eq!(report.outcome, VerificationOutcome::Verified);
        assert_eq!(
            report
                .comparison
                .as_ref()
                .unwrap()
                .first_divergent_generated_token_index,
            NO_DIVERGENCE
        );
        report.verify_self_digest().unwrap();
    }

    #[test]
    fn changed_token_fails_at_the_first_changed_index() {
        let profile = profile();
        let mut actual = profile.expected.clone();
        actual.generated_token_ids[1] += 1;
        let report = evaluate(&profile, &actual).unwrap();
        assert_eq!(report.outcome, VerificationOutcome::NotVerified);
        assert_eq!(
            report
                .comparison
                .as_ref()
                .unwrap()
                .first_divergent_generated_token_index,
            1
        );
    }

    #[test]
    fn changed_completion_count_fails_verification() {
        let profile = profile();
        let mut actual = profile.expected.clone();
        actual.completion_tokens += 1;
        let report = evaluate(&profile, &actual).unwrap();
        assert_eq!(report.outcome, VerificationOutcome::NotVerified);
        assert!(!report.comparison.unwrap().completion_tokens_match);
    }

    #[test]
    fn report_tampering_is_detected() {
        let profile = profile();
        let mut report = evaluate(&profile, &profile.expected).unwrap();
        report.detail.push_str(" changed");
        assert!(report.verify_self_digest().is_err());
    }

    #[test]
    fn write_report_replaces_an_existing_file() {
        let profile = profile();
        let report = evaluate(&profile, &profile.expected).unwrap();
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("model.verify.json");
        std::fs::write(&path, "old").unwrap();
        write_report(&path, &report).unwrap();
        let written: VerificationReport =
            serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        assert_eq!(written, report);
        written.verify_self_digest().unwrap();
    }
}
