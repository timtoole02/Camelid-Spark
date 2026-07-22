//! Runnable lane — generic, f32-only, breadth-first GGUF execution path.
//!
//! The runnable lane is the promotion oracle for the supported lane: any GGUF in
//! the covered-set must either run deterministically or be **refused at admission**
//! with a precise, machine-readable reason. Refusal logic is as load-bearing as
//! execution logic — it is the evidence gate applied at the door
//! (`RUNNABLE_LANE_SPEC.md`, principle #2).
//!
//! Phase 1 delivers the admission gate (`admit`). Execution (dequant → parametric
//! decoder block → logits) lands in later phases.

pub mod admit;
pub mod dequant;
pub mod model;
pub mod smoke;

pub use admit::{admit, AdmissionAxis, AdmissionOk, AdmissionReject, TokenizerFamily};
pub use dequant::dequantize;
pub use model::RunnableModel;
pub use smoke::{headline_quant_of, oracle_qualified, smoke_admit, SmokeReport};
