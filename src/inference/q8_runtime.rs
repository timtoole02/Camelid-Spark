use std::env;

use super::{diagnostic_linear_accumulation_precision, LinearAccumulationPrecision};
use crate::Result;

pub(super) const X86_Q8_PACKED_ROWS4_MATMUL_GROUPS_PER_CHUNK_DEFAULT: usize = 8;

/// Lane 1 — scope of the unified tiled Q8_0 PREFILL GEMM owner. The owner is a bit-exact,
/// role-agnostic drop-in for the per-projection prefill block-dot: it reuses the proven 4x4
/// register microkernel but flips the loop nest so each weight band stays L1/L2-resident
/// while every input row streams against it (the arithmetic-intensity fix for the
/// bandwidth-bound host). Default: `All` on win-x86_64 (D15, b9918 re-validation receipts),
/// `Off` elsewhere — each other host still owes its own receipt per the benchmark treaty.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Q8MatmulOwnerScope {
    Off,
    FfnDown,
    All,
}

impl Q8MatmulOwnerScope {
    fn from_env() -> Self {
        match env::var("CAMELID_X86_Q8_MATMUL_OWNER") {
            Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
                "all" | "on" | "1" | "true" | "enabled" | "yes" => Self::All,
                "ffn_down" | "ffn-down" | "ffndown" => Self::FfnDown,
                _ => Self::Off,
            },
            // DEFAULT ON for win-x86_64 only (D15): re-validated at the b9918
            // pin with the engaged-checked paired sweep — +12.3% (3B) /
            // +11.9% (4B) prefill, CI excludes 1.0, 8/8 rounds, bit-exact.
            // Other targets keep Off pending their own host receipts (the
            // BENCHMARK_TREATY both-host rule is scoped, not waived).
            // Explicit rollback: CAMELID_X86_Q8_MATMUL_OWNER=off.
            Err(_) => {
                #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
                {
                    Self::All
                }
                #[cfg(not(all(target_os = "windows", target_arch = "x86_64")))]
                {
                    Self::Off
                }
            }
        }
    }

    pub(super) fn is_on(self) -> bool {
        !matches!(self, Self::Off)
    }

    /// Does the owner cover this rectangular role under the active scope?
    pub(super) fn covers_role(self, role: &str) -> bool {
        match self {
            Self::Off => false,
            Self::FfnDown => role == "ffn_down",
            Self::All => true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Q8RuntimeFlags {
    pub(super) block_dot: bool,
    pub(super) file_reader_block_dot: bool,
    pub(super) attention_projection_decode_consumer: bool,
    pub(super) attention_output_decode_consumer: bool,
    pub(super) attention_output_packed_rows4_matmul: bool,
    pub(super) attention_qkv_decode_consumer: bool,
    pub(super) attention_qkv_decode_group_chunking: bool,
    pub(super) attention_qkv_packed_rows4_matmul: bool,
    pub(super) output_packed_rows4_matmul: bool,
    pub(super) output_amx_prefill: bool,
    pub(super) output_decode_owner: bool,
    pub(super) ffn_gate_up_decode_consumer: bool,
    pub(super) ffn_gate_up_decode_group_chunking: bool,
    pub(super) ffn_gate_up_decode_fused_activation: bool,
    pub(super) ffn_gate_up_decode_paired_dot: bool,
    pub(super) ffn_decode_chain: bool,
    pub(super) ffn_gate_up_packed_rows4_matmul: bool,
    pub(super) ffn_gate_up_single_owner: bool,
    pub(super) ffn_down_decode_consumer: bool,
    pub(super) ffn_down_decode_group_chunking: bool,
    pub(super) ffn_down_packed_rows4_matmul: bool,
    pub(super) ffn_down_gemm4_prefill: bool,
    pub(super) ffn_down_gemm4_row_group_schedule: bool,
    pub(super) ffn_down_gemm4_avx2: bool,
    pub(super) ffn_down_amx_prefill: bool,
    pub(super) ffn_down_single_owner: bool,
    pub(super) ffn_down_vnni_decode: bool,
    pub(super) ffn_down_vnni_decode_rawptr: bool,
    pub(super) metal: bool,
    pub(super) cuda: bool,
    pub(super) metal_retained: bool,
    pub(super) hybrid_retained: bool,
    pub(super) hybrid_gpu_rows: Option<usize>,
    pub(super) hybrid_gpu_percent: usize,
    /// Lane 1: unified tiled Q8_0 prefill GEMM owner scope (default `Off`).
    pub(super) q8_matmul_owner: Q8MatmulOwnerScope,
    /// Use the AVX2 4x4 microkernel inside the owner. Defaults ON whenever the owner is on
    /// (avoids the GEMM4 split-flag trap where prefill-on-but-avx2-off ran the slow scalar 4x4).
    pub(super) q8_matmul_owner_avx2: bool,
    /// Use the AVX-512 VNNI (dpbusd) microkernel inside the owner when the CPU supports it (v2,
    /// llama's tinyBLAS compute technique). Defaults ON; gated by runtime feature detection at
    /// dispatch. Set `CAMELID_X86_Q8_MATMUL_OWNER_VNNI=0` to force the AVX2 microkernel (v1).
    pub(super) q8_matmul_owner_vnni: bool,
    /// Use the wider 4x8 VNNI tile (v3): two output groups per input load. Defaults ON — the
    /// hardened in-process paired sweep (`camelid bench-owner-sweep`) shows a SIGNIFICANT +3.3% (3B)
    /// / +3.8% (4B) over the 4x4 tile, 7-8/8 rounds, CI excludes 1.0 (the earlier "null" was
    /// cross-invocation thermal noise, not a real tie). Set `..._4X8=0` to force the 4x4 tile (v2).
    pub(super) q8_matmul_owner_4x8: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Q8PackedRows4MatmulSchedule {
    pub(super) groups_per_chunk: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ResolvedRuntimePlan {
    pub(super) linear_accumulation_precision: LinearAccumulationPrecision,
    pub(super) q8: Q8RuntimeFlags,
    pub(super) q8_packed_rows4_matmul_schedule: Q8PackedRows4MatmulSchedule,
}

/// In-process benchmark sweeps (`bench-owner-sweep`) mutate the owner env
/// keys between configs; the process-lifetime plan caches below would
/// silently ignore that — every config after the first would measure the
/// first-resolved plan (a fake null). `CAMELID_BENCH_UNCACHED_RUNTIME_PLAN=1`
/// (itself read once) makes both resolvers re-read the env per call.
/// Sweep-only escape hatch: normal runs pay one cached-bool branch.
/// (In test builds every resolver is uncached already, so this has no
/// callers there — hence the cfg_attr.)
#[cfg_attr(test, allow(dead_code))]
pub(super) fn bench_uncached_runtime_plan() -> bool {
    static BYPASS: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *BYPASS.get_or_init(|| q8_0_env_flag_enabled_default_off("CAMELID_BENCH_UNCACHED_RUNTIME_PLAN"))
}

impl ResolvedRuntimePlan {
    pub(super) fn from_env() -> Result<Self> {
        // Resolve ONCE per process outside tests: the ~20 env flags below are
        // fixed post-startup (every lane flag already assumes this via its
        // own OnceLock), and this constructor used to run on per-op paths —
        // ~20 raw env::var reads per projection call. Tests keep the uncached
        // read so env-manipulating tests observe their changes.
        #[cfg(test)]
        {
            Self::from_env_uncached()
        }
        #[cfg(not(test))]
        {
            if bench_uncached_runtime_plan() {
                return Self::from_env_uncached();
            }
            static RESOLVED: std::sync::OnceLock<ResolvedRuntimePlan> = std::sync::OnceLock::new();
            if let Some(plan) = RESOLVED.get() {
                return Ok(*plan);
            }
            let plan = Self::from_env_uncached()?;
            Ok(*RESOLVED.get_or_init(|| plan))
        }
    }

    fn from_env_uncached() -> Result<Self> {
        let q8 = Q8RuntimeFlags::from_env();
        Ok(Self {
            linear_accumulation_precision: diagnostic_linear_accumulation_precision()?,
            q8,
            q8_packed_rows4_matmul_schedule: Q8PackedRows4MatmulSchedule::from_q8_flags(q8),
        })
    }
}

impl Q8RuntimeFlags {
    pub(super) fn from_env() -> Self {
        // Same caching contract as `ResolvedRuntimePlan::from_env` above: the
        // flags are fixed post-startup, and callers outside the resolved-plan
        // path (e.g. the shared-QKV block-dot predicate) used to re-read ~30
        // env vars per layer per decode token — on Windows every env::var
        // read allocates, even on a miss. Tests keep the uncached read so
        // env-manipulating tests observe their changes.
        #[cfg(test)]
        {
            Self::from_env_uncached()
        }
        #[cfg(not(test))]
        {
            if bench_uncached_runtime_plan() {
                return Self::from_env_uncached();
            }
            static RESOLVED: std::sync::OnceLock<Q8RuntimeFlags> = std::sync::OnceLock::new();
            if let Some(flags) = RESOLVED.get() {
                return *flags;
            }
            let flags = Self::from_env_uncached();
            *RESOLVED.get_or_init(|| flags)
        }
    }

    fn from_env_uncached() -> Self {
        Self {
            block_dot: q8_0_env_flag_enabled_default_on_fail_closed("CAMELID_Q8_0_BLOCK_DOT"),
            file_reader_block_dot: q8_0_env_flag_enabled_default_on_fail_closed(
                "CAMELID_Q8_0_FILE_READER_BLOCK_DOT",
            ),
            attention_projection_decode_consumer: q8_0_env_flag_enabled_default_off(
                "CAMELID_X86_Q8_ATTENTION_PROJECTION_DECODE_CONSUMER",
            ),
            attention_output_decode_consumer: q8_0_env_flag_enabled_default_off(
                "CAMELID_X86_Q8_ATTENTION_OUTPUT_DECODE_CONSUMER",
            ),
            attention_output_packed_rows4_matmul: q8_0_env_flag_enabled_default_off(
                "CAMELID_X86_Q8_ATTENTION_OUTPUT_PACKED_ROWS4_MATMUL",
            ),
            attention_qkv_decode_consumer: q8_0_env_flag_enabled_default_off(
                "CAMELID_X86_Q8_ATTENTION_QKV_DECODE_CONSUMER",
            ),
            attention_qkv_decode_group_chunking: q8_0_env_flag_enabled_default_off(
                "CAMELID_X86_Q8_ATTENTION_QKV_DECODE_GROUP_CHUNKING",
            ),
            attention_qkv_packed_rows4_matmul: q8_0_env_flag_enabled_default_off(
                "CAMELID_X86_Q8_ATTENTION_QKV_PACKED_ROWS4_MATMUL",
            ),
            output_packed_rows4_matmul: q8_0_env_flag_enabled_default_off(
                "CAMELID_X86_Q8_OUTPUT_PACKED_ROWS4_MATMUL",
            ),
            output_amx_prefill: q8_0_env_flag_enabled_default_off(
                "CAMELID_X86_Q8_OUTPUT_AMX_PREFILL",
            ),
            output_decode_owner: q8_0_env_flag_enabled_default_off(
                "CAMELID_X86_Q8_OUTPUT_DECODE_OWNER",
            ),
            ffn_gate_up_decode_consumer: q8_0_env_flag_enabled_default_off(
                "CAMELID_X86_Q8_FFN_GATE_UP_DECODE_CONSUMER",
            ) || q8_0_env_flag_enabled_default_off(
                "CAMELID_MAC_Q8_FFN_GATE_UP_DECODE_CONSUMER",
            ),
            ffn_gate_up_decode_group_chunking: q8_0_env_flag_enabled_default_off(
                "CAMELID_X86_Q8_FFN_GATE_UP_DECODE_GROUP_CHUNKING",
            ),
            ffn_gate_up_decode_fused_activation: q8_0_env_flag_enabled_default_off(
                "CAMELID_X86_Q8_FFN_GATE_UP_DECODE_FUSED_ACTIVATION",
            ) || q8_0_env_flag_enabled_default_off(
                "CAMELID_X86_Q8_FFN_GATE_UP_DECODE_FUSED",
            ),
            ffn_gate_up_decode_paired_dot: q8_0_env_flag_enabled_default_off(
                "CAMELID_X86_Q8_FFN_GATE_UP_DECODE_PAIRED_DOT",
            ),
            ffn_decode_chain: q8_0_env_flag_enabled_default_off("CAMELID_X86_Q8_FFN_DECODE_CHAIN"),
            ffn_gate_up_packed_rows4_matmul: q8_0_env_flag_enabled_default_off(
                "CAMELID_X86_Q8_FFN_GATE_UP_PACKED_ROWS4_MATMUL",
            ),
            ffn_gate_up_single_owner: q8_0_env_flag_enabled_default_off(
                "CAMELID_X86_Q8_FFN_GATE_UP_SINGLE_OWNER",
            ),
            ffn_down_decode_consumer: q8_0_env_flag_enabled_default_off(
                "CAMELID_X86_Q8_FFN_DOWN_DECODE_CONSUMER",
            ) || q8_0_env_flag_enabled_default_off(
                "CAMELID_MAC_Q8_FFN_DOWN_DECODE_CONSUMER",
            ),
            ffn_down_decode_group_chunking: q8_0_env_flag_enabled_default_off(
                "CAMELID_X86_Q8_FFN_DOWN_DECODE_GROUP_CHUNKING",
            ) || q8_0_env_flag_enabled_default_off(
                "CAMELID_MAC_Q8_FFN_DOWN_DECODE_GROUP_CHUNKING",
            ),
            ffn_down_packed_rows4_matmul: x86_q8_ffn_down_packed_rows4_matmul_enabled(),
            ffn_down_gemm4_prefill: q8_0_env_flag_enabled_default_off(
                "CAMELID_X86_Q8_FFN_DOWN_GEMM4_PREFILL",
            ),
            ffn_down_gemm4_row_group_schedule: q8_0_env_flag_enabled_default_off(
                "CAMELID_X86_Q8_FFN_DOWN_GEMM4_ROW_GROUP_SCHED",
            ),
            ffn_down_gemm4_avx2: q8_0_env_flag_enabled_default_off(
                "CAMELID_X86_Q8_FFN_DOWN_GEMM4_AVX2",
            ),
            ffn_down_amx_prefill: q8_0_env_flag_enabled_default_off(
                "CAMELID_X86_Q8_FFN_DOWN_AMX_PREFILL",
            ),
            ffn_down_single_owner: q8_0_env_flag_enabled_default_off(
                "CAMELID_X86_Q8_FFN_DOWN_SINGLE_OWNER",
            ),
            ffn_down_vnni_decode: q8_0_env_flag_enabled_default_off(
                "CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE",
            ),
            ffn_down_vnni_decode_rawptr: q8_0_env_flag_enabled_default_off(
                "CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE_RAWPTR",
            ),
            metal: q8_0_env_flag_enabled_default_off("CAMELID_METAL_Q8"),
            // Opt-in CUDA Q8_0 hybrid linear (decode). Controlled by the runtime
            // switch (seeded from CAMELID_CUDA_Q8, flippable from the UI). The CPU
            // path stays the reference; harmless without the `cuda` feature or a
            // device because `cuda::try_*` falls back to CPU.
            cuda: crate::cuda::runtime_enabled(),
            metal_retained: q8_0_env_flag_enabled_default_off("CAMELID_METAL_Q8_RETAINED"),
            hybrid_retained: q8_0_env_flag_enabled_default_off("CAMELID_HYBRID_Q8_RETAINED"),
            hybrid_gpu_rows: env::var("CAMELID_HYBRID_Q8_GPU_ROWS")
                .ok()
                .and_then(|value| value.trim().parse::<usize>().ok()),
            hybrid_gpu_percent: env::var("CAMELID_HYBRID_Q8_GPU_PERCENT")
                .ok()
                .and_then(|value| value.trim().parse::<usize>().ok())
                .filter(|value| *value > 0)
                .unwrap_or(10)
                .min(90),
            q8_matmul_owner: Q8MatmulOwnerScope::from_env(),
            q8_matmul_owner_avx2: q8_0_env_flag_enabled_default_on_fail_closed(
                "CAMELID_X86_Q8_MATMUL_OWNER_AVX2",
            ),
            q8_matmul_owner_vnni: q8_0_env_flag_enabled_default_on_fail_closed(
                "CAMELID_X86_Q8_MATMUL_OWNER_VNNI",
            ),
            q8_matmul_owner_4x8: q8_0_env_flag_enabled_default_on_fail_closed(
                "CAMELID_X86_Q8_MATMUL_OWNER_4X8",
            ),
        }
    }

    fn any_packed_rows4_matmul_enabled(self) -> bool {
        self.attention_output_packed_rows4_matmul
            || self.attention_qkv_packed_rows4_matmul
            || self.output_packed_rows4_matmul
            || self.ffn_gate_up_packed_rows4_matmul
            || self.ffn_down_packed_rows4_matmul
    }

    pub(super) fn hybrid_gpu_rows_for_output(self, output_rows: usize) -> usize {
        if output_rows < 2 {
            return 0;
        }
        if let Some(rows) = self.hybrid_gpu_rows {
            return rows.min(output_rows.saturating_sub(1));
        }
        ((output_rows * self.hybrid_gpu_percent).div_ceil(100))
            .max(1)
            .min(output_rows.saturating_sub(1))
    }
}

impl Default for Q8PackedRows4MatmulSchedule {
    fn default() -> Self {
        Self {
            groups_per_chunk: X86_Q8_PACKED_ROWS4_MATMUL_GROUPS_PER_CHUNK_DEFAULT,
        }
    }
}

impl Q8PackedRows4MatmulSchedule {
    pub(super) fn from_q8_flags(q8: Q8RuntimeFlags) -> Self {
        if !q8.any_packed_rows4_matmul_enabled() && !q8.q8_matmul_owner.is_on() {
            return Self::default();
        }
        Self {
            groups_per_chunk: env::var("CAMELID_X86_Q8_PACKED_ROWS4_MATMUL_GROUPS_PER_CHUNK")
                .ok()
                .and_then(|value| value.trim().parse::<usize>().ok())
                .filter(|value| *value > 0)
                .unwrap_or(X86_Q8_PACKED_ROWS4_MATMUL_GROUPS_PER_CHUNK_DEFAULT),
        }
    }
}

pub(super) fn q8_0_env_flag_enabled_default_on_fail_closed(key: &str) -> bool {
    match env::var(key) {
        Ok(value) => {
            let value = value.trim();
            value.eq_ignore_ascii_case("1")
                || value.eq_ignore_ascii_case("true")
                || value.eq_ignore_ascii_case("on")
                || value.eq_ignore_ascii_case("enabled")
                || value.eq_ignore_ascii_case("yes")
        }
        Err(_) => true,
    }
}

pub(super) fn q8_0_env_flag_enabled_default_off(key: &str) -> bool {
    env::var(key)
        .map(|value| {
            let value = value.trim();
            value.eq_ignore_ascii_case("1")
                || value.eq_ignore_ascii_case("true")
                || value.eq_ignore_ascii_case("on")
                || value.eq_ignore_ascii_case("enabled")
                || value.eq_ignore_ascii_case("yes")
        })
        .unwrap_or(false)
}

// All callers are arch/OS-gated (aarch64 dotprod dispatch, Apple Accelerate), so this is
// dead code on other targets.
#[cfg_attr(
    not(any(target_arch = "aarch64", target_os = "macos")),
    allow(dead_code)
)]
pub(super) fn q8_0_env_flag_disabled(key: &str) -> bool {
    env::var(key)
        .map(|value| {
            let value = value.trim();
            value.eq_ignore_ascii_case("0")
                || value.eq_ignore_ascii_case("false")
                || value.eq_ignore_ascii_case("off")
                || value.eq_ignore_ascii_case("disabled")
                || value.eq_ignore_ascii_case("dequantized")
                || value.eq_ignore_ascii_case("f32")
        })
        .unwrap_or(false)
}

fn x86_q8_ffn_down_packed_rows4_matmul_enabled() -> bool {
    q8_0_env_flag_enabled_default_off("CAMELID_X86_Q8_FFN_DOWN_PACKED_ROWS4_MATMUL")
        || q8_0_env_flag_enabled_default_off("CAMELID_X86_Q8_PACKED_ROWS4_MATMUL")
}
