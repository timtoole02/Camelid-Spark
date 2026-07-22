use std::{collections::BTreeMap, env, path::Path};

use serde::{Deserialize, Serialize};

use crate::gguf::{GgufFile, GgufTensorType};

const MANAGED_ENV_KEYS: &[&str] = &[
    "CAMELID_PARALLEL_LINEAR",
    "CAMELID_MAC_Q8_REPACK",
    "CAMELID_MAC_Q8_PREFILL_I8MM",
    "CAMELID_MAC_Q8_SCHED",
    "CAMELID_MAC_Q8_FFN_DOWN_DECODE_CONSUMER",
    "CAMELID_MAC_Q8_FFN_GATE_UP_DECODE_CONSUMER",
    "CAMELID_FORWARD_RSS_TIMINGS",
    "CAMELID_X86_Q8_REPACK",
    "CAMELID_X86_Q8_KERNEL",
    "CAMELID_X86_Q8_ATTENTION_PROJECTION_DECODE_CONSUMER",
    "CAMELID_X86_Q8_ATTENTION_OUTPUT_DECODE_CONSUMER",
    "CAMELID_X86_Q8_ATTENTION_OUTPUT_PACKED_ROWS4_MATMUL",
    "CAMELID_X86_Q8_ATTENTION_QKV_DECODE_CONSUMER",
    "CAMELID_X86_Q8_ATTENTION_QKV_DECODE_GROUP_CHUNKING",
    "CAMELID_X86_Q8_ATTENTION_QKV_PACKED_ROWS4_MATMUL",
    "CAMELID_X86_Q8_OUTPUT_PACKED_ROWS4_MATMUL",
    "CAMELID_X86_Q8_OUTPUT_AMX_PREFILL",
    "CAMELID_X86_Q8_PACKED_ROWS4_SERIAL_DECODE",
    "CAMELID_X86_Q8_PARALLEL_INPUT_QUANTIZE",
    "CAMELID_X86_Q8_FFN_GATE_UP_DECODE_CONSUMER",
    "CAMELID_X86_Q8_FFN_GATE_UP_DECODE_GROUP_CHUNKING",
    "CAMELID_X86_Q8_FFN_GATE_UP_DECODE_FUSED_ACTIVATION",
    "CAMELID_X86_Q8_FFN_GATE_UP_DECODE_PAIRED_DOT",
    "CAMELID_X86_Q8_FFN_DECODE_CHAIN",
    "CAMELID_X86_Q8_FFN_GATE_UP_PACKED_ROWS4_MATMUL",
    "CAMELID_X86_Q8_FFN_GATE_UP_SINGLE_OWNER",
    "CAMELID_X86_Q8_FFN_DOWN_DECODE_CONSUMER",
    "CAMELID_X86_Q8_FFN_DOWN_PACKED_ROWS4_MATMUL",
    "CAMELID_X86_Q8_PACKED_ROWS4_MATMUL",
    "CAMELID_X86_Q8_FFN_DOWN_GEMM4_PREFILL",
    "CAMELID_X86_Q8_FFN_DOWN_GEMM4_ROW_GROUP_SCHED",
    "CAMELID_X86_Q8_FFN_DOWN_GEMM4_AVX2",
    "CAMELID_X86_Q8_FFN_DOWN_AMX_PREFILL",
    "CAMELID_X86_Q8_FFN_DOWN_SINGLE_OWNER",
    "CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE",
    "CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE_RAWPTR",
    "CAMELID_X86_Q8_FFN_DOWN_DECODE_OWNER",
    "CAMELID_X86_Q8_OUTPUT_DECODE_OWNER",
    "CAMELID_X86_Q8_ATTENTION_QKV_DECODE_GROUPS_PER_CHUNK",
    "CAMELID_X86_Q8_FFN_GATE_UP_DECODE_GROUPS_PER_CHUNK",
    "CAMELID_X86_Q8_FFN_DOWN_GEMM4_ROW_GROUP_MIN_INPUT_GROUPS",
    "CAMELID_X86_Q8_PACKED_ROWS4_MATMUL_GROUPS_PER_CHUNK",
];

struct ManagedPassthroughEnvKey {
    key: &'static str,
    owner_gate: &'static str,
}

const MANAGED_PASSTHROUGH_ENV_KEYS: &[ManagedPassthroughEnvKey] = &[
    ManagedPassthroughEnvKey {
        key: "CAMELID_X86_Q8_ATTENTION_QKV_DECODE_GROUPS_PER_CHUNK",
        owner_gate: "CAMELID_X86_Q8_ATTENTION_QKV_DECODE_GROUP_CHUNKING",
    },
    ManagedPassthroughEnvKey {
        key: "CAMELID_X86_Q8_FFN_GATE_UP_DECODE_GROUPS_PER_CHUNK",
        owner_gate: "CAMELID_X86_Q8_FFN_GATE_UP_DECODE_GROUP_CHUNKING",
    },
    ManagedPassthroughEnvKey {
        key: "CAMELID_X86_Q8_FFN_DOWN_GEMM4_ROW_GROUP_MIN_INPUT_GROUPS",
        owner_gate: "CAMELID_X86_Q8_FFN_DOWN_GEMM4_ROW_GROUP_SCHED",
    },
    ManagedPassthroughEnvKey {
        key: "CAMELID_X86_Q8_PACKED_ROWS4_MATMUL_GROUPS_PER_CHUNK",
        owner_gate: "CAMELID_X86_Q8_PACKED_ROWS4_MATMUL",
    },
];

pub const MAC_Q8_PREFILL_I8MM_MIN_ROWS: usize = 4;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionProfile {
    Safe,
    Auto,
    Experimental,
    Debug,
}

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct ExecutionPlan {
    pub profile: ExecutionProfile,
    pub operating_system: String,
    pub architecture: String,
    pub platform_label: String,
    pub cpu_model: String,
    pub cpu_features: Vec<String>,
    pub model_family: String,
    /// Human quantization label: the declared `general.file_type` (llama.cpp
    /// ftype naming) when present and credible, else a tensor-scan bucket.
    /// Descriptive only — routing branches on tensor predicates, not this.
    pub quant_type: String,
    /// Row identity as recognized from `general.name` (or the filename when
    /// the name is junk). Name-derived: it does NOT imply the quant on disk
    /// matches the row's evidence — see `support_level`.
    pub exact_model_row: String,
    /// Plan-level support string for the recognized row, quant-gated: only a
    /// Q8_0 file of a recognized row reports that row's level; every other
    /// quant reports `unknown_or_unvalidated`. `/api/capabilities` remains
    /// the support source of truth.
    pub support_level: String,
    pub selected_backend: String,
    pub selected_q8_path: String,
    pub prefill_path: String,
    pub prefill_runtime_policy: String,
    pub decode_path: String,
    pub thread_count: usize,
    pub diagnostics_status: String,
    pub fallback_path: String,
    /// True when the GPU-resident CUDA decode engine drives decode for this process
    /// (surfaced in `/api/capabilities` so a loaded row reports the live GPU path). The
    /// `selected_backend`/`decode_path` above carry the `cuda_resident_q8_*` labels when
    /// this is set; mirrors the Metal lane's `metal_available` capabilities signal.
    pub cuda_resident_active: bool,
    pub reasons: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct ExecutionPlanOutcome {
    pub plan: ExecutionPlan,
    pub env_updates: BTreeMap<&'static str, Option<&'static str>>,
}

#[derive(Clone, Debug, Default)]
pub struct PlannerEnv {
    passthrough_env: BTreeMap<&'static str, Option<String>>,
}

impl PlannerEnv {
    pub fn capture() -> Self {
        let passthrough_env = MANAGED_PASSTHROUGH_ENV_KEYS
            .iter()
            .map(|entry| {
                (
                    entry.key,
                    env::var(entry.key)
                        .ok()
                        .filter(|value| managed_positive_usize_value(value)),
                )
            })
            .collect();
        Self { passthrough_env }
    }

    pub fn apply(&self, updates: &BTreeMap<&'static str, Option<&'static str>>) {
        for key in MANAGED_ENV_KEYS {
            match updates.get(key).copied().flatten() {
                Some(value) => env::set_var(key, value),
                None => env::remove_var(key),
            }
        }
        for entry in MANAGED_PASSTHROUGH_ENV_KEYS {
            if env_updates_enable_gate(updates, entry.owner_gate) {
                match self
                    .passthrough_env
                    .get(entry.key)
                    .and_then(|value| value.as_deref())
                {
                    Some(value) => env::set_var(entry.key, value),
                    None => env::remove_var(entry.key),
                }
            } else {
                env::remove_var(entry.key);
            }
        }
    }
}

fn managed_positive_usize_value(value: &str) -> bool {
    value
        .trim()
        .parse::<usize>()
        .ok()
        .filter(|value| *value > 0)
        .is_some()
}

fn env_updates_enable_gate(
    updates: &BTreeMap<&'static str, Option<&'static str>>,
    key: &'static str,
) -> bool {
    matches!(updates.get(key).copied().flatten(), Some("on"))
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlanPlatform {
    pub operating_system: String,
    pub architecture: String,
    pub platform_label: String,
    pub cpu_model: String,
    pub cpu_features: Vec<String>,
    /// A usable Metal compute device exists on this host (always false off macOS).
    pub metal_available: bool,
    /// The CUDA resident decode engine will drive decode for this process (a usable
    /// CUDA device is present, GPU acceleration is on, and neither deterministic mode
    /// nor `CAMELID_CUDA_RESIDENT_DECODE=0` forces the CPU reference). When true, the
    /// CPU Q8 rows4 repack is skipped: the GPU resident engine consumes plain RAM-
    /// resident Q8_0 blocks, and the repack replaces them (the two are mutually
    /// exclusive on weight storage, exactly as the Metal-resident plan handles).
    pub cuda_resident_active: bool,
}

impl PlanPlatform {
    pub fn current() -> Self {
        let operating_system = env::consts::OS.to_string();
        let architecture = env::consts::ARCH.to_string();
        let cpu_features = cpu_features();
        let cpu_model = cpu_model();
        let platform_label = platform_label(&operating_system, &architecture, &cpu_model);
        let metal_available = crate::metal::detect_metal_device().available;
        let cuda_resident_active = cuda_resident_decode_will_run();
        Self {
            operating_system,
            architecture,
            platform_label,
            cpu_model,
            cpu_features,
            metal_available,
            cuda_resident_active,
        }
    }
}

/// Planning-time mirror of `inference::resident_decode_cuda_enabled`: true when the GPU
/// resident decode engine will run, so the CPU Q8 rows4 repack must be skipped (the GPU
/// needs un-repacked plain Q8_0 blocks). Deterministic mode and
/// `CAMELID_CUDA_RESIDENT_DECODE=0` force it false (CPU reference), matching the runtime
/// gate. On a host without a usable CUDA device (or a build without the `cuda` feature)
/// `cuda::is_available()` is false, so the CPU repack path is unaffected.
fn cuda_resident_decode_will_run() -> bool {
    if env_flag_enabled("CAMELID_DETERMINISTIC") {
        return false;
    }
    if let Ok(value) = env::var("CAMELID_CUDA_RESIDENT_DECODE") {
        if matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "off" | "no"
        ) {
            return false;
        }
    }
    crate::cuda::is_available() && crate::cuda::gpu_accel_enabled()
}

pub fn plan_for_model(
    model_path: &Path,
    gguf: &GgufFile,
    threads: Option<usize>,
) -> ExecutionPlanOutcome {
    plan_for_model_with_platform(model_path, gguf, threads, PlanPlatform::current())
}

pub fn plan_for_model_with_platform(
    model_path: &Path,
    gguf: &GgufFile,
    threads: Option<usize>,
    platform: PlanPlatform,
) -> ExecutionPlanOutcome {
    // GAIT selector (bring-up gate `CAMELID_GAIT`, default off): consult the
    // per-(model × machine) gait store for a cached gait. With the gate off, or
    // on any miss/empty store, this returns None and the existing default path
    // runs unchanged — keeping this byte-identical to today. When a gait is
    // found, apply its scheduling substrate (the coarse profile is applied by
    // the env machinery below).
    let (profile, profile_reason) = match crate::gait::maybe_select_profile(gguf) {
        Some(gait) => {
            crate::gait::apply_selected_gait(&gait);
            (gait.profile, gait.reason)
        }
        None => requested_profile(),
    };
    let row = exact_model_row(model_path, gguf);
    let model_family = model_family(&row, gguf);
    let quant_type = quant_type(gguf);
    let support_level = support_level(&row, &quant_type);
    let thread_count = threads.unwrap_or_else(default_thread_count);
    let diagnostics_status = match profile {
        ExecutionProfile::Debug => {
            "debug diagnostics enabled; performance claims disabled".to_string()
        }
        _ => "standard diagnostics; RSS timings disabled by default".to_string(),
    };

    let mut reasons = vec![profile_reason];
    reasons.push(format!("exact_model_row={row}"));
    reasons.push(format!("support_level={support_level}"));
    reasons.push(format!("quant_type={quant_type}"));

    let mut env_updates: BTreeMap<&'static str, Option<&'static str>> = BTreeMap::new();
    if matches!(profile, ExecutionProfile::Debug) {
        env_updates.insert("CAMELID_FORWARD_RSS_TIMINGS", Some("on"));
    }

    // Routing branches on the tensor scan, not the human `quant_type` label
    // above: the label may be more specific (Q6_K, Q5_K_M, TQ1_0) than the
    // lanes it admits to, and a file with a wrong declared file_type must
    // still route by what its tensors actually are. These predicates preserve
    // the pre-label-fix branch semantics exactly: "Q8_0" meant any Q8_0
    // tensor present, "Q4_K_M" meant K-quant tensors with no Q8_0.
    let has_tensor = |t: GgufTensorType| gguf.tensors.iter().any(|tensor| tensor.tensor_type == t);
    let has_q8_0_tensors = has_tensor(GgufTensorType::Q8_0);
    let has_kquant_tensors = has_tensor(GgufTensorType::Q4K) || has_tensor(GgufTensorType::Q6K);

    let (
        selected_backend,
        selected_q8_path,
        prefill_path,
        prefill_runtime_policy,
        decode_path,
        fallback_path,
    ) = if has_q8_0_tensors && is_supported_exact_q8_row(&row) {
        if platform.operating_system == "macos" && platform.architecture == "aarch64" {
            select_macos_q8_plan(&profile, &platform, &mut env_updates, &mut reasons)
        } else if platform.architecture == "x86_64"
            && (platform.operating_system == "linux" || platform.operating_system == "windows")
        {
            // The x86_64 Q8 runtime-repack + AVX2 packed-rows4 path is platform-agnostic
            // Rust (no OS-specific kernels) and is parity-validated bit-identical to the
            // scalar reference on Windows as well as Linux, so both share this plan.
            select_x86_q8_plan(&profile, &platform, &mut env_updates, &mut reasons)
        } else {
            reasons.push(
                    "no validated platform-specific Q8_0 plan for this OS/arch; failing closed to safe path"
                        .into(),
                );
            safe_q8_plan()
        }
    } else if has_q8_0_tensors
        && platform.cuda_resident_active
        && is_gpu_runnable_arch(gguf)
        && !env_flag_disabled("CAMELID_GPU_RUNNABLE_TIER")
    {
        // On by DEFAULT: an uncurated but architecturally-compatible Q8_0 model should just run
        // on the GPU without the user having to opt in. This is safe because admission is gated
        // at runtime by the GPU-vs-CPU parity self-check — a model that is not bit-exact falls
        // back to the CPU reference path. Opt out with CAMELID_GPU_RUNNABLE_TIER=0 (forces CPU).
        reasons.push(
            "non-curated Q8_0 on a resident-capable dense arch: GPU-runnable tier (NOT a \
             supported row) — resident path admitted subject to the runtime parity self-check"
                .into(),
        );
        cuda_resident_q8_runnable_plan()
    } else if !has_q8_0_tensors && has_kquant_tensors {
        select_kquant_plan(&platform, &mut reasons)
    } else {
        reasons.push("non-validated row or quant; failing closed to safe path".into());
        (
            "cpu_reference",
            "safe_dense_or_q8_cpu",
            "safe_cpu_prefill",
            "always_retained_reference_path",
            "safe_cpu_decode",
            "safe_cpu_reference_path",
        )
    };

    let plan = ExecutionPlan {
        profile,
        operating_system: platform.operating_system,
        architecture: platform.architecture,
        platform_label: platform.platform_label,
        cpu_model: platform.cpu_model,
        cpu_features: platform.cpu_features,
        model_family,
        quant_type,
        exact_model_row: row,
        support_level,
        selected_backend: selected_backend.to_string(),
        selected_q8_path: selected_q8_path.to_string(),
        prefill_path: prefill_path.to_string(),
        prefill_runtime_policy: prefill_runtime_policy.to_string(),
        decode_path: decode_path.to_string(),
        thread_count,
        diagnostics_status,
        fallback_path: fallback_path.to_string(),
        cuda_resident_active: platform.cuda_resident_active,
        reasons,
    };
    ExecutionPlanOutcome { plan, env_updates }
}

fn select_macos_q8_plan(
    profile: &ExecutionProfile,
    platform: &PlanPlatform,
    env_updates: &mut BTreeMap<&'static str, Option<&'static str>>,
    reasons: &mut Vec<String>,
) -> (
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
) {
    if matches!(profile, ExecutionProfile::Safe) {
        reasons.push("safe profile selected; optimized Mac Q8 paths disabled".into());
        return safe_q8_plan();
    }
    if env_flag_disabled("CAMELID_MAC_Q8_REPACK") {
        reasons
            .push("CAMELID_MAC_Q8_REPACK disables Mac repack; failing closed to safe path".into());
        env_updates.insert("CAMELID_MAC_Q8_REPACK", Some("off"));
        return safe_q8_plan();
    }

    // The Metal-resident Q8_0 stack outranks the CPU repack when the host can run it.
    // The GPU path requires plain RAM-resident Q8_0 blocks, which the rows4 repack
    // replaces — the two are mutually exclusive on weight storage — so selecting Metal
    // means loading plain blocks (the CPU plain-block reference path remains the
    // in-process fallback for sessions the resident gates reject). Selection requires
    // the resident-decode gate (on by default in the CLI entry; absent for embedders
    // and test suites, which keep the validated CPU plans) plus an actual Metal
    // device; CAMELID_MAC_Q8_METAL_PLAN=0 opts back into the CPU repack plan.
    if env_flag_enabled("CAMELID_METAL_RESIDENT_DECODE")
        && !env_flag_disabled("CAMELID_MAC_Q8_METAL_PLAN")
        && platform.metal_available
    {
        env_updates.insert("CAMELID_MAC_Q8_REPACK", Some("off"));
        env_updates.insert("CAMELID_PARALLEL_LINEAR", Some("on"));
        reasons.push(
            "Metal resident Q8_0 stack selected (Metal device present, resident decode              enabled); weights stay plain RAM-resident Q8_0 blocks — the rows4 CPU repack              is disabled because the GPU-resident path requires the plain blocks"
                .into(),
        );
        reasons.push("parallel linear enabled by execution plan".into());
        return (
            "metal_resident_q8_runtime",
            "metal_resident_q8_0_wire",
            "q8_0_metal_resident_prefill",
            "resident_single_command_buffer_prefill",
            "q8_0_metal_resident_decode",
            "retained_q8_reference_path",
        );
    }

    let dotprod = has_feature(&platform.cpu_features, "dotprod");
    let i8mm = has_feature(&platform.cpu_features, "i8mm");
    if !dotprod {
        reasons.push("Apple Silicon dotprod not detected; failing closed to safe path".into());
        return safe_q8_plan();
    }

    env_updates.insert("CAMELID_PARALLEL_LINEAR", Some("on"));
    env_updates.insert("CAMELID_MAC_Q8_REPACK", Some("on"));
    reasons.push("validated macOS Apple Silicon Q8_0 runtime repack enabled".into());
    reasons.push("parallel linear enabled by execution plan".into());

    if env_flag_disabled("CAMELID_MAC_Q8_FFN_DOWN_DECODE_CONSUMER") {
        env_updates.insert("CAMELID_MAC_Q8_FFN_DOWN_DECODE_CONSUMER", Some("off"));
        reasons.push("Mac FFN-down decode consumer disabled".into());
    } else {
        env_updates.insert("CAMELID_MAC_Q8_FFN_DOWN_DECODE_CONSUMER", Some("on"));
        reasons.push("Mac FFN-down decode consumer gate enabled by default".into());
    }

    if env_flag_disabled("CAMELID_MAC_Q8_FFN_GATE_UP_DECODE_CONSUMER") {
        env_updates.insert("CAMELID_MAC_Q8_FFN_GATE_UP_DECODE_CONSUMER", Some("off"));
        reasons.push("Mac FFN gate/up decode consumer disabled".into());
    } else {
        env_updates.insert("CAMELID_MAC_Q8_FFN_GATE_UP_DECODE_CONSUMER", Some("on"));
        reasons.push("Mac FFN gate/up decode consumer gate enabled by default".into());
    }

    let prefill_i8mm_requested = !env_flag_disabled("CAMELID_MAC_Q8_PREFILL_I8MM");
    let prefill_path = if i8mm && prefill_i8mm_requested {
        env_updates.insert("CAMELID_MAC_Q8_PREFILL_I8MM", Some("on"));
        reasons.push("direct-pack prefill I8MM gate enabled by default".into());
        reasons.push(format!(
            "direct-pack I8MM dispatch engages only when prefill rows >= {}",
            MAC_Q8_PREFILL_I8MM_MIN_ROWS
        ));
        if matches!(profile, ExecutionProfile::Experimental) {
            env_updates.insert("CAMELID_MAC_Q8_SCHED", Some("packed_prefill"));
            reasons.push(
                "experimental packed prefill scheduling enabled; single-token decode remains GEMV/DOTPROD"
                    .into(),
            );
            "q8_0_experimental_packed_prefill_i8mm_available"
        } else {
            env_updates.insert("CAMELID_MAC_Q8_SCHED", Some("off"));
            reasons.push(
                "packed prefill scheduling remains experimental and is disabled for auto profile"
                    .into(),
            );
            "q8_0_direct_pack_prefill_i8mm_available"
        }
    } else {
        env_updates.insert("CAMELID_MAC_Q8_PREFILL_I8MM", Some("off"));
        env_updates.insert("CAMELID_MAC_Q8_SCHED", Some("off"));
        if env_flag_disabled("CAMELID_MAC_Q8_PREFILL_I8MM") {
            reasons.push("CAMELID_MAC_Q8_PREFILL_I8MM disables I8MM prefill".into());
        } else {
            reasons
                .push("I8MM/MATMUL_INT8 unavailable; using packed Q8 CPU prefill fallback".into());
        }
        "q8_0_cpu_packed_prefill_fallback_available"
    };

    if matches!(profile, ExecutionProfile::Experimental) {
        reasons.push("experimental profile active; support claims remain unchanged".into());
    }
    if matches!(profile, ExecutionProfile::Debug) {
        reasons.push(
            "debug profile active; RSS timings enabled and performance claims disabled".into(),
        );
    }

    (
        "cpu_q8_runtime_repack",
        "mac_validated_q8_0_repack",
        prefill_path,
        "enabled_when_prefill_rows_gte_4",
        "q8_0_decode_gemv_dotprod",
        "retained_q8_reference_path",
    )
}

fn select_x86_q8_plan(
    profile: &ExecutionProfile,
    platform: &PlanPlatform,
    env_updates: &mut BTreeMap<&'static str, Option<&'static str>>,
    reasons: &mut Vec<String>,
) -> (
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
) {
    if matches!(profile, ExecutionProfile::Safe) {
        reasons.push("safe profile selected; optimized x86 Q8 paths disabled".into());
        return safe_q8_plan();
    }
    if platform.cuda_resident_active {
        // A CUDA device is driving decode: keep weights as plain RAM-resident Q8_0
        // blocks (the GPU resident engine cannot consume the CPU rows4 repack — the two
        // are mutually exclusive on weight storage). The CPU repack path only wins when
        // the CPU actually runs decode (no GPU, GPU toggled off, or deterministic mode).
        reasons.push(
            "CUDA resident decode active; GPU-resident Q8_0 engine drives decode (weights stay plain RAM-resident Q8_0 blocks — the CPU rows4 repack is disabled while the GPU drives decode)"
                .into(),
        );
        return cuda_resident_q8_plan();
    }
    if env_flag_disabled("CAMELID_X86_Q8_REPACK") || env_flag_disabled("CAMELID_X86_Q8_KERNEL") {
        reasons.push(
            "x86 Q8 override disables optimized kernel/repack; failing closed to safe path".into(),
        );
        if env_flag_disabled("CAMELID_X86_Q8_REPACK") {
            env_updates.insert("CAMELID_X86_Q8_REPACK", Some("off"));
        }
        if env_flag_disabled("CAMELID_X86_Q8_KERNEL") {
            env_updates.insert("CAMELID_X86_Q8_KERNEL", Some("off"));
        }
        return safe_q8_plan();
    }
    if let Some(invalid) = invalid_x86_kernel_override() {
        reasons.push(format!(
            "invalid CAMELID_X86_Q8_KERNEL={invalid}; failing closed to safe path"
        ));
        env_updates.insert("CAMELID_X86_Q8_KERNEL", Some("off"));
        return safe_q8_plan();
    }
    if !has_feature(&platform.cpu_features, "avx2") {
        reasons.push(
            "AVX2 feature not detected for x86 Q8 kernel; failing closed to safe path".into(),
        );
        return safe_q8_plan();
    }

    env_updates.insert("CAMELID_PARALLEL_LINEAR", Some("on"));
    env_updates.insert("CAMELID_X86_Q8_REPACK", Some("on"));
    env_updates.insert("CAMELID_X86_Q8_KERNEL", Some("avx2"));
    let optional_x86_q8_gate = |name| {
        if env_flag_disabled(name) {
            Some("off")
        } else {
            Some("on")
        }
    };
    env_updates.insert(
        "CAMELID_X86_Q8_ATTENTION_PROJECTION_DECODE_CONSUMER",
        optional_x86_q8_gate("CAMELID_X86_Q8_ATTENTION_PROJECTION_DECODE_CONSUMER"),
    );
    env_updates.insert(
        "CAMELID_X86_Q8_ATTENTION_OUTPUT_DECODE_CONSUMER",
        optional_x86_q8_gate("CAMELID_X86_Q8_ATTENTION_OUTPUT_DECODE_CONSUMER"),
    );
    env_updates.insert(
        "CAMELID_X86_Q8_ATTENTION_OUTPUT_PACKED_ROWS4_MATMUL",
        optional_x86_q8_gate("CAMELID_X86_Q8_ATTENTION_OUTPUT_PACKED_ROWS4_MATMUL"),
    );
    env_updates.insert(
        "CAMELID_X86_Q8_ATTENTION_QKV_DECODE_CONSUMER",
        optional_x86_q8_gate("CAMELID_X86_Q8_ATTENTION_QKV_DECODE_CONSUMER"),
    );
    env_updates.insert(
        "CAMELID_X86_Q8_ATTENTION_QKV_DECODE_GROUP_CHUNKING",
        optional_x86_q8_gate("CAMELID_X86_Q8_ATTENTION_QKV_DECODE_GROUP_CHUNKING"),
    );
    env_updates.insert(
        "CAMELID_X86_Q8_ATTENTION_QKV_PACKED_ROWS4_MATMUL",
        optional_x86_q8_gate("CAMELID_X86_Q8_ATTENTION_QKV_PACKED_ROWS4_MATMUL"),
    );
    env_updates.insert(
        "CAMELID_X86_Q8_OUTPUT_PACKED_ROWS4_MATMUL",
        optional_x86_q8_gate("CAMELID_X86_Q8_OUTPUT_PACKED_ROWS4_MATMUL"),
    );
    env_updates.insert(
        "CAMELID_X86_Q8_OUTPUT_AMX_PREFILL",
        optional_x86_q8_gate("CAMELID_X86_Q8_OUTPUT_AMX_PREFILL"),
    );
    // Serial packed decode is the validated Linux default, but on Windows the parallel
    // packed decode runs ~2x faster (TinyLlama 11 -> 19 tok/s, ffn_down 20 -> 10 ms) and
    // stays bit-identical to the reference (each output row is an independent dot, so
    // parallelizing across rows does not change any reduction order). Windows therefore
    // defaults serial-decode OFF; an explicit env opt-in still forces it on.
    let serial_packed_decode = if platform.operating_system == "windows" {
        if env_flag_enabled("CAMELID_X86_Q8_PACKED_ROWS4_SERIAL_DECODE") {
            Some("on")
        } else {
            Some("off")
        }
    } else {
        optional_x86_q8_gate("CAMELID_X86_Q8_PACKED_ROWS4_SERIAL_DECODE")
    };
    env_updates.insert(
        "CAMELID_X86_Q8_PACKED_ROWS4_SERIAL_DECODE",
        serial_packed_decode,
    );
    env_updates.insert(
        "CAMELID_X86_Q8_PARALLEL_INPUT_QUANTIZE",
        optional_x86_q8_gate("CAMELID_X86_Q8_PARALLEL_INPUT_QUANTIZE"),
    );
    let ffn_decode_chain_enabled = env_flag_enabled("CAMELID_X86_Q8_FFN_DECODE_CHAIN");
    let ffn_gate_up_decode_consumer_enabled =
        ffn_decode_chain_enabled || env_flag_enabled("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_CONSUMER");
    env_updates.insert(
        "CAMELID_X86_Q8_FFN_GATE_UP_DECODE_CONSUMER",
        if ffn_gate_up_decode_consumer_enabled {
            Some("on")
        } else {
            Some("off")
        },
    );
    env_updates.insert(
        "CAMELID_X86_Q8_FFN_GATE_UP_DECODE_GROUP_CHUNKING",
        optional_x86_q8_gate("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_GROUP_CHUNKING"),
    );
    env_updates.insert(
        "CAMELID_X86_Q8_FFN_GATE_UP_DECODE_FUSED_ACTIVATION",
        optional_x86_q8_gate("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_FUSED_ACTIVATION"),
    );
    env_updates.insert(
        "CAMELID_X86_Q8_FFN_GATE_UP_DECODE_PAIRED_DOT",
        optional_x86_q8_gate("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_PAIRED_DOT"),
    );
    env_updates.insert(
        "CAMELID_X86_Q8_FFN_DECODE_CHAIN",
        if ffn_decode_chain_enabled {
            Some("on")
        } else {
            Some("off")
        },
    );
    env_updates.insert(
        "CAMELID_X86_Q8_FFN_GATE_UP_PACKED_ROWS4_MATMUL",
        optional_x86_q8_gate("CAMELID_X86_Q8_FFN_GATE_UP_PACKED_ROWS4_MATMUL"),
    );
    let ffn_down_decode_consumer_enabled =
        ffn_decode_chain_enabled || env_flag_enabled("CAMELID_X86_Q8_FFN_DOWN_DECODE_CONSUMER");
    env_updates.insert(
        "CAMELID_X86_Q8_FFN_DOWN_DECODE_CONSUMER",
        if ffn_down_decode_consumer_enabled {
            Some("on")
        } else {
            Some("off")
        },
    );
    env_updates.insert(
        "CAMELID_X86_Q8_FFN_DOWN_PACKED_ROWS4_MATMUL",
        optional_x86_q8_gate("CAMELID_X86_Q8_FFN_DOWN_PACKED_ROWS4_MATMUL"),
    );
    env_updates.insert(
        "CAMELID_X86_Q8_PACKED_ROWS4_MATMUL",
        optional_x86_q8_gate("CAMELID_X86_Q8_PACKED_ROWS4_MATMUL"),
    );
    env_updates.insert(
        "CAMELID_X86_Q8_FFN_DOWN_GEMM4_PREFILL",
        optional_x86_q8_gate("CAMELID_X86_Q8_FFN_DOWN_GEMM4_PREFILL"),
    );
    env_updates.insert(
        "CAMELID_X86_Q8_FFN_DOWN_GEMM4_ROW_GROUP_SCHED",
        optional_x86_q8_gate("CAMELID_X86_Q8_FFN_DOWN_GEMM4_ROW_GROUP_SCHED"),
    );
    env_updates.insert(
        "CAMELID_X86_Q8_FFN_DOWN_GEMM4_AVX2",
        optional_x86_q8_gate("CAMELID_X86_Q8_FFN_DOWN_GEMM4_AVX2"),
    );
    env_updates.insert(
        "CAMELID_X86_Q8_FFN_DOWN_AMX_PREFILL",
        optional_x86_q8_gate("CAMELID_X86_Q8_FFN_DOWN_AMX_PREFILL"),
    );
    env_updates.insert(
        "CAMELID_X86_Q8_FFN_DOWN_SINGLE_OWNER",
        optional_x86_q8_gate("CAMELID_X86_Q8_FFN_DOWN_SINGLE_OWNER"),
    );
    env_updates.insert(
        "CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE",
        optional_x86_q8_gate("CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE"),
    );
    env_updates.insert(
        "CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE_RAWPTR",
        optional_x86_q8_gate("CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE_RAWPTR"),
    );
    env_updates.insert(
        "CAMELID_X86_Q8_FFN_GATE_UP_SINGLE_OWNER",
        optional_x86_q8_gate("CAMELID_X86_Q8_FFN_GATE_UP_SINGLE_OWNER"),
    );
    env_updates.insert("CAMELID_X86_Q8_FFN_DOWN_DECODE_OWNER", Some("off"));
    env_updates.insert(
        "CAMELID_X86_Q8_OUTPUT_DECODE_OWNER",
        optional_x86_q8_gate("CAMELID_X86_Q8_OUTPUT_DECODE_OWNER"),
    );

    if ffn_decode_chain_enabled && !env_flag_enabled("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_CONSUMER") {
        reasons.push(
            "FFN decode-chain opt-in also enables the required FFN gate/up decode consumer gate"
                .into(),
        );
    }
    if ffn_decode_chain_enabled && !env_flag_enabled("CAMELID_X86_Q8_FFN_DOWN_DECODE_CONSUMER") {
        reasons.push(
            "FFN decode-chain opt-in also enables the required FFN-down decode consumer gate"
                .into(),
        );
    }
    reasons.push("validated x86_64 (Linux/Windows) Rust Q8 runtime repack enabled".into());
    reasons.push("validated Rust AVX2 Q8 packed rows4 kernel selected".into());
    reasons.push("attention, FFN, and output experiments enabled by default".into());
    if matches!(profile, ExecutionProfile::Experimental) {
        reasons.push("experimental profile active; support claims remain unchanged".into());
    }

    (
        "cpu_q8_runtime_repack",
        "x86_experimental_q8_0_avx2_rust",
        "q8_0_runtime_packed_rows4_prefill_avx2_available",
        "enabled_when_q8_runtime_storage_active",
        "q8_0_decode_packed_rows4_avx2",
        "retained_q8_reference_path",
    )
}

fn safe_q8_plan() -> (
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
) {
    (
        "cpu_reference",
        "safe_q8_0_block_dot",
        "safe_cpu_prefill",
        "always_retained_reference_path",
        "safe_cpu_decode",
        "retained_q8_reference_path",
    )
}

/// Plan labels when the GPU-resident CUDA decode engine drives this process (the NVIDIA
/// analog of `metal_resident_q8_runtime`). Weights stay plain RAM-resident Q8_0 blocks —
/// the engine uploads them to VRAM once and decodes on-device; the CPU rows4 repack is
/// disabled because the GPU consumes the plain blocks. The `retained_q8_reference_path`
/// CPU plan remains the in-process fallback for any token/config the resident gates
/// reject. Validated token-AND-text-identical to the CPU reference (transitively
/// llama.cpp) on the dense Qwen3 Q8_0 ChatML rows; see the COMPATIBILITY.md Windows CUDA
/// section and the `qwen3-*-windows-cuda-resident-parity-*` evidence bundles.
fn cuda_resident_q8_plan() -> (
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
) {
    (
        "cuda_resident_q8_runtime",
        "cuda_resident_q8_0_wire",
        "q8_0_cuda_resident_prefill",
        "resident_single_shot_prefill",
        "q8_0_cuda_resident_decode",
        "retained_q8_reference_path",
    )
}

/// Plan labels for the GPU-runnable tier: a Q8_0 model on a resident-capable dense
/// architecture that is NOT a curated supported exact-row. Byte-for-byte the same GPU
/// route as [`cuda_resident_q8_plan`], but every label carries a `_runnable_unvalidated`
/// suffix so telemetry, receipts, and the UI can never mistake it for a supported row.
/// The support_level stays `unknown_or_unvalidated`; admission to this tier is gated at
/// runtime by a GPU-vs-CPU parity self-check (see `inference.rs`), which falls the model
/// back to the CPU reference path if the resident output is not token-identical.
fn cuda_resident_q8_runnable_plan() -> (
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
) {
    (
        "cuda_resident_q8_runtime_runnable_unvalidated",
        "cuda_resident_q8_0_wire",
        "q8_0_cuda_resident_prefill_runnable_unvalidated",
        "resident_single_shot_prefill",
        "q8_0_cuda_resident_decode_runnable_unvalidated",
        "retained_q8_reference_path",
    )
}

/// Architectures the CUDA resident engine implements token-identically to the CPU
/// reference, and for which the GPU-runnable tier is eligible when a model is Q8_0 but
/// not a curated support row. Deliberately narrow — dense llama/qwen/mistral only; MoE
/// (expert routing) and not-yet-resident archs (gemma/phi/ssm/qwen35) are excluded so we
/// never route a model the resident dense kernel cannot run under a GPU label. The
/// runtime `resident_decode_eligible` check + the parity self-check are the backstops.
fn is_gpu_runnable_arch(gguf: &GgufFile) -> bool {
    let arch = gguf.architecture().unwrap_or("");
    if !matches!(arch, "llama" | "qwen2" | "qwen3" | "mistral") {
        return false;
    }
    // Exclude MoE: the resident dense kernel does not implement expert routing. A missing
    // key means dense (the common case); a present non-zero expert_count means MoE.
    gguf.metadata_u32(&format!("{arch}.expert_count"))
        .map(|experts| experts == 0)
        .unwrap_or(true)
}

/// Plan labels for a mixed K-quant (Q4_K_M = Q4_K + Q6_K) model. K-quant 2-D linears
/// load WIRE-ONLY and are decoded either by the GPU-resident engine (`q4k_gemv`/
/// `q6k_gemv`) when CUDA resident decode is driving this process, or by the CPU
/// block-dot (`q4_k_dot_avx2` + `q6_k_wire_row_dot`) otherwise — neither materializes
/// f32. Descriptive only (no env_updates): the actual route is chosen at runtime by
/// `resident_decode_cuda_active()` + `q4_k_cpu_block_dot_enabled()`. This replaces the
/// old `cpu_reference`/`dense_or_other` mislabel that reported a CPU fallback for a lane
/// that actually runs GPU-resident (K-quant conductor disclosure fix). Greedy parity vs
/// llama.cpp is recorded in the `*-q4_k_m-*-parity-*` evidence bundles.
fn select_kquant_plan(
    platform: &PlanPlatform,
    reasons: &mut Vec<String>,
) -> (
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    &'static str,
) {
    if platform.cuda_resident_active {
        reasons.push(
            "CUDA resident decode active; GPU-resident K-quant engine (q4k_gemv/q6k_gemv) drives decode from wire-only Q4_K/Q6_K blocks"
                .into(),
        );
        (
            "cuda_resident_kquant_runtime",
            "cuda_resident_kquant_wire",
            "kquant_cuda_resident_prefill",
            "resident_single_shot_prefill",
            "kquant_cuda_resident_decode",
            "kquant_cpu_block_dot_reference_path",
        )
    } else if crate::inference::q4_k_cpu_block_dot_enabled() {
        reasons.push(
            "CPU K-quant block-dot decode (Q4_K AVX2 + Q6_K 8-lane scalar) reads wire-only blocks; no f32 materialization"
                .into(),
        );
        (
            "cpu_kquant_block_dot",
            "kquant_wire_block_dot",
            "cpu_kquant_block_dot_prefill",
            "always_retained_reference_path",
            "kquant_cpu_block_dot_decode",
            "kquant_cpu_block_dot_reference_path",
        )
    } else {
        reasons.push(
            "K-quant CPU block-dot disabled (CAMELID_X86_Q4K_DECODE=0) and no resident GPU; K-quant linears have no CPU consumer"
                .into(),
        );
        (
            "cpu_reference",
            "safe_dense_or_q8_cpu",
            "safe_cpu_prefill",
            "always_retained_reference_path",
            "safe_cpu_decode",
            "safe_cpu_reference_path",
        )
    }
}

fn requested_profile() -> (ExecutionProfile, String) {
    match env::var("CAMELID_PROFILE").ok() {
        None => (ExecutionProfile::Auto, "profile=auto default".into()),
        Some(value) if value.eq_ignore_ascii_case("safe") => {
            (ExecutionProfile::Safe, "profile=safe requested".into())
        }
        Some(value) if value.eq_ignore_ascii_case("auto") => {
            (ExecutionProfile::Auto, "profile=auto requested".into())
        }
        Some(value) if value.eq_ignore_ascii_case("experimental") => (
            ExecutionProfile::Experimental,
            "profile=experimental requested; warnings enabled".into(),
        ),
        Some(value) if value.eq_ignore_ascii_case("debug") => (
            ExecutionProfile::Debug,
            "profile=debug requested; diagnostics enabled".into(),
        ),
        Some(value) => (
            ExecutionProfile::Safe,
            format!("invalid CAMELID_PROFILE={value}; failing closed to safe"),
        ),
    }
}

fn exact_model_row(model_path: &Path, gguf: &GgufFile) -> String {
    let from_name = gguf.model_name().map(|value| value.to_string());
    let from_file = model_path
        .file_name()
        .map(|v| v.to_string_lossy().to_string());
    // Prefer the GGUF `general.name`, but if it does NOT map to a recognized support row
    // while the FILENAME does, use the filename. Some GGUF conversions ship a junk
    // `general.name` (e.g. "hub") that would otherwise shadow a perfectly recognizable
    // filename and drop a known, validated model onto the slow cpu_reference path
    // instead of its GPU lane. This only ever UPGRADES an unrecognized name to a
    // recognized row — it never overrides a name that already matches a row.
    if let (Some(name), Some(file)) = (&from_name, &from_file) {
        if recognized_row_level(name) == "unknown_or_unvalidated"
            && recognized_row_level(file) != "unknown_or_unvalidated"
        {
            return file.clone();
        }
    }
    from_name.or(from_file).unwrap_or_else(|| "unknown".into())
}

/// The plan's support-level string for a recognized row, gated on the quant
/// the row's evidence actually covers. Every row in the recognized table is
/// Q8_0 evidence, so any other quant of the same model name reports
/// `unknown_or_unvalidated` here — a Q6_K TinyLlama must not echo the Q8_0
/// gate row's level. `/api/capabilities` (filename-anchored, quant-aware)
/// remains the support source of truth; this string only describes the row
/// the load-time plan recognized.
fn support_level(row: &str, quant_type: &str) -> String {
    if quant_type != "Q8_0" {
        return "unknown_or_unvalidated".into();
    }
    recognized_row_level(row).into()
}

/// Quant-blind name→level table. Besides `support_level` above, this powers
/// row *recognition* — the junk general.name → filename upgrade in
/// `exact_model_row` and the supported-row planner gate — which must keep
/// working for non-Q8_0 files: recognition and support claims are different
/// questions.
fn recognized_row_level(row: &str) -> &'static str {
    let normalized = normalize_row(row);
    if normalized.contains("tinyllama") {
        "supported_current_gate"
    } else if normalized.contains("llama_3_2_1b_instruct") {
        "supported_exact_row_smoke_512_1024_2048_4096_8192"
    } else if normalized.contains("llama_3_2_3b_instruct") {
        // Full 512-8192 ladder re-validated on the anchored canonical GGUF
        // (raw-decode greedy parity vs llama.cpp acd79d603; see the
        // llama32-3b-context-512-8192-anchored evidence bundle). Split from the
        // 8B arm below, whose checked packs remain 512/1024/2048.
        "supported_exact_row_smoke_512_1024_2048_4096_8192"
    } else if normalized.contains("llama_3_8b_instruct")
        || normalized.contains("meta_llama_3_8b_instruct")
    {
        "supported_exact_row_smoke_512_1024_2048"
    } else if normalized.contains("mistral_7b_instruct_v0_3") {
        "supported_exact_row_smoke_512_1024_2048_4096_8192"
    } else if normalized.contains("qwen3_0_6b_instruct")
        || normalized.contains("qwen3_1_7b_instruct")
        || normalized.contains("qwen3_4b_instruct")
        || normalized.contains("qwen3_8b_instruct")
    {
        // Dense Qwen3 Q8_0 ChatML rows (thinking disabled), validated token+text
        // identical to llama.cpp at 1/5/50 on the cpu_reference path and on the
        // x86_64 runtime-repack/AVX2 Q8 path (parity re-validated on Windows).
        // Scoped to the short-chat smoke envelope; MoE (A3B), base variants, other
        // sizes/quants, longer context, and thinking-mode are NOT covered.
        // (Replaces the broader `contains("qwen3")` branch from PR #283, whose
        // label claimed 512/1024/2048 context packs and matched MoE/base/other
        // sizes — neither validated for qwen3.)
        "supported_exact_row_smoke_chatml"
    } else if normalized.contains("mixtral_8x7b_instruct_v0_1") {
        "bounded_runtime_only_unsupported"
    } else {
        "unknown_or_unvalidated"
    }
}

fn is_supported_exact_q8_row(row: &str) -> bool {
    matches!(
        recognized_row_level(row),
        "supported_current_gate"
            | "supported_exact_row_smoke_512_1024_2048_4096_8192"
            | "supported_exact_row_smoke_512_1024_2048"
            | "supported_exact_row_smoke_chatml"
    )
}

fn model_family(row: &str, gguf: &GgufFile) -> String {
    let normalized = normalize_row(row);
    if normalized.contains("tinyllama") {
        "tinyllama".into()
    } else if normalized.contains("llama") {
        "llama".into()
    } else if normalized.contains("mistral") {
        "mistral".into()
    } else if normalized.contains("mixtral") {
        "mixtral".into()
    } else {
        gguf.architecture().unwrap_or("unknown").to_string()
    }
}

/// Human label for the file's quantization, reported in the plan (and from
/// there `/v1/health` and the System page). The declared `general.file_type`
/// (llama.cpp ftype naming, shared with receipts) is the most specific
/// truthful source — it keeps a pure-Q6_K or TQ1_0 file from being collapsed
/// into the coarse "Q4_K_M"/"Q8_0" buckets of the tensor-scan fallback. A
/// declared "Q8_0" is accepted only when the scan actually finds Q8_0 tensors,
/// because the Q8_0 label is what gates `support_level` onto a promoted row.
/// Routing never reads this label — the planner branches on the tensor
/// predicates directly (see `plan_for_model_with_platform`).
fn quant_type(gguf: &GgufFile) -> String {
    let has = |t: GgufTensorType| gguf.tensors.iter().any(|tensor| tensor.tensor_type == t);
    let has_q8_0 = has(GgufTensorType::Q8_0);
    if let Some(declared) = crate::receipt::declared_file_type_label(gguf) {
        if declared != "Q8_0" || has_q8_0 {
            return declared.into();
        }
    }
    if has_q8_0 {
        "Q8_0".into()
    } else if has(GgufTensorType::Q4K) {
        // Undeclared K-quant mix (Q4_K_M = Q4_K + Q6_K). Decoded by the GPU-resident
        // engine (q4k_gemv/q6k_gemv) or, on CPU, the K-quant block-dot — both consume
        // the wire-only blocks. Recognized here so the plan stops mislabeling it as
        // the `dense_or_other` cpu_reference fallback (K-quant conductor disclosure fix).
        "Q4_K_M".into()
    } else if has(GgufTensorType::Q6K) {
        "Q6_K".into()
    } else {
        "dense_or_other".into()
    }
}

fn normalize_row(row: &str) -> String {
    row.to_ascii_lowercase()
        .chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
        .collect::<String>()
}

fn cpu_features() -> Vec<String> {
    let mut out = Vec::new();
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            out.push("dotprod".into());
        }
        if std::arch::is_aarch64_feature_detected!("i8mm") {
            out.push("i8mm".into());
        }
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            out.push("avx2".into());
        }
        if std::arch::is_x86_feature_detected!("avx512f") {
            out.push("avx512f".into());
        }
        let cpuinfo_flags = cpuinfo_flags();
        if cpuinfo_has_flag(&cpuinfo_flags, "avx_vnni") {
            out.push("avx_vnni".into());
        }
        if cpuinfo_has_flag(&cpuinfo_flags, "avx512_vnni") {
            out.push("avx512_vnni".into());
        }
        if cpuinfo_has_flag(&cpuinfo_flags, "amx_tile") {
            out.push("amx_tile".into());
        }
        if cpuinfo_has_flag(&cpuinfo_flags, "amx_int8") {
            out.push("amx_int8".into());
        }
    }
    out
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn cpuinfo_flags() -> String {
    #[cfg(target_os = "linux")]
    {
        std::fs::read_to_string("/proc/cpuinfo")
            .ok()
            .and_then(|content| {
                content.lines().find_map(|line| {
                    let (key, value) = line.split_once(':')?;
                    key.trim()
                        .eq_ignore_ascii_case("flags")
                        .then(|| value.trim().to_string())
                })
            })
            .unwrap_or_default()
    }
    #[cfg(not(target_os = "linux"))]
    {
        String::new()
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn cpuinfo_has_flag(flags: &str, wanted: &str) -> bool {
    flags.split_whitespace().any(|flag| flag == wanted)
}

fn cpu_model() -> String {
    #[cfg(target_os = "macos")]
    {
        command_output("sysctl", &["-n", "machdep.cpu.brand_string"])
            .unwrap_or_else(|| "unknown".into())
    }
    #[cfg(target_os = "linux")]
    {
        std::fs::read_to_string("/proc/cpuinfo")
            .ok()
            .and_then(|contents| {
                contents.lines().find_map(|line| {
                    line.strip_prefix("model name").and_then(|rest| {
                        rest.split_once(':')
                            .map(|(_, value)| value.trim().to_string())
                    })
                })
            })
            .unwrap_or_else(|| "unknown".into())
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        "unknown".into()
    }
}

#[cfg(target_os = "macos")]
fn command_output(program: &str, args: &[&str]) -> Option<String> {
    std::process::Command::new(program)
        .args(args)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
        .filter(|value| !value.is_empty())
}

fn platform_label(os: &str, arch: &str, cpu_model: &str) -> String {
    if os == "macos" && arch == "aarch64" {
        if cpu_model.to_ascii_lowercase().contains("apple") {
            "macOS arm64 Apple Silicon".into()
        } else {
            "macOS arm64".into()
        }
    } else if os == "linux" && arch == "x86_64" {
        "Ubuntu/Linux x86_64".into()
    } else {
        format!("{os} {arch}")
    }
}

fn has_feature(features: &[String], wanted: &str) -> bool {
    features.iter().any(|feature| feature == wanted)
}

fn default_thread_count() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
}

fn env_flag_disabled(key: &str) -> bool {
    env::var(key)
        .map(|value| {
            let value = value.trim();
            value.eq_ignore_ascii_case("0")
                || value.eq_ignore_ascii_case("off")
                || value.eq_ignore_ascii_case("false")
                || value.eq_ignore_ascii_case("disabled")
                || value.eq_ignore_ascii_case("cpu")
        })
        .unwrap_or(false)
}

#[allow(dead_code)]
fn env_flag_enabled(key: &str) -> bool {
    env::var(key)
        .map(|value| {
            let value = value.trim();
            value.eq_ignore_ascii_case("1")
                || value.eq_ignore_ascii_case("on")
                || value.eq_ignore_ascii_case("true")
                || value.eq_ignore_ascii_case("enabled")
        })
        .unwrap_or(false)
}

#[allow(dead_code)]
fn x86_kernel_avx2_explicitly_requested() -> bool {
    env::var("CAMELID_X86_Q8_KERNEL")
        .map(|value| value.trim().eq_ignore_ascii_case("avx2"))
        .unwrap_or(false)
}

fn invalid_x86_kernel_override() -> Option<String> {
    let value = env::var("CAMELID_X86_Q8_KERNEL").ok()?;
    let trimmed = value.trim();
    if trimmed.is_empty()
        || trimmed.eq_ignore_ascii_case("0")
        || trimmed.eq_ignore_ascii_case("off")
        || trimmed.eq_ignore_ascii_case("false")
        || trimmed.eq_ignore_ascii_case("disabled")
        || trimmed.eq_ignore_ascii_case("avx2")
        || trimmed.eq_ignore_ascii_case("on")
        || trimmed == "1"
        || trimmed.eq_ignore_ascii_case("true")
    {
        None
    } else {
        Some(value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        gguf::{GgufFile, GgufMetadataValue, GgufTensorDescriptor},
        test_support::env_lock,
    };
    use std::{collections::BTreeMap, path::PathBuf};

    fn platform(os: &str, arch: &str, features: &[&str]) -> PlanPlatform {
        PlanPlatform {
            operating_system: os.into(),
            architecture: arch.into(),
            platform_label: platform_label(os, arch, "Apple M4"),
            cpu_model: "fixture cpu".into(),
            cpu_features: features.iter().map(|feature| (*feature).into()).collect(),
            metal_available: false,
            cuda_resident_active: false,
        }
    }

    fn metal_platform(os: &str, arch: &str, features: &[&str]) -> PlanPlatform {
        PlanPlatform {
            metal_available: true,
            ..platform(os, arch, features)
        }
    }

    fn cuda_platform(os: &str, arch: &str, features: &[&str]) -> PlanPlatform {
        PlanPlatform {
            cuda_resident_active: true,
            ..platform(os, arch, features)
        }
    }

    fn fixture(name: &str) -> GgufFile {
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "general.name".into(),
            GgufMetadataValue::String(name.into()),
        );
        metadata.insert(
            "general.architecture".into(),
            GgufMetadataValue::String("llama".into()),
        );
        GgufFile {
            path: PathBuf::from("/tmp/model.gguf"),
            version: 3,
            tensor_count: 1,
            metadata_count: metadata.len() as i64,
            alignment: 32,
            data_start_offset: 0,
            metadata,
            tensors: vec![GgufTensorDescriptor {
                name: "blk.0.attn_q.weight".into(),
                dimensions: vec![32, 32],
                tensor_type: GgufTensorType::Q8_0,
                relative_offset: 0,
                absolute_offset: 0,
                n_bytes: 34,
            }],
        }
    }

    /// `fixture` with explicit tensor types and an optional declared
    /// `general.file_type`, for the quant-label truth tests.
    fn quant_fixture(
        name: &str,
        file_type: Option<u32>,
        tensor_types: &[GgufTensorType],
    ) -> GgufFile {
        let mut gguf = fixture(name);
        if let Some(ft) = file_type {
            gguf.metadata
                .insert("general.file_type".into(), GgufMetadataValue::U32(ft));
        }
        gguf.tensor_count = tensor_types.len() as i64;
        gguf.tensors = tensor_types
            .iter()
            .enumerate()
            .map(|(i, t)| GgufTensorDescriptor {
                name: format!("blk.{i}.attn_q.weight"),
                dimensions: vec![32, 32],
                tensor_type: *t,
                relative_offset: 0,
                absolute_offset: 0,
                n_bytes: 34,
            })
            .collect();
        gguf
    }

    fn clear_profile_env() {
        for key in [
            "CAMELID_PROFILE",
            "CAMELID_MAC_Q8_REPACK",
            "CAMELID_MAC_Q8_PREFILL_I8MM",
            "CAMELID_MAC_Q8_SCHED",
            "CAMELID_MAC_Q8_FFN_DOWN_DECODE_CONSUMER",
            "CAMELID_MAC_Q8_FFN_GATE_UP_DECODE_CONSUMER",
            "CAMELID_X86_Q8_REPACK",
            "CAMELID_X86_Q8_KERNEL",
            "CAMELID_X86_Q8_ATTENTION_PROJECTION_DECODE_CONSUMER",
            "CAMELID_X86_Q8_ATTENTION_OUTPUT_DECODE_CONSUMER",
            "CAMELID_X86_Q8_ATTENTION_OUTPUT_PACKED_ROWS4_MATMUL",
            "CAMELID_X86_Q8_ATTENTION_QKV_DECODE_CONSUMER",
            "CAMELID_X86_Q8_ATTENTION_QKV_DECODE_GROUP_CHUNKING",
            "CAMELID_X86_Q8_ATTENTION_QKV_DECODE_GROUPS_PER_CHUNK",
            "CAMELID_X86_Q8_ATTENTION_QKV_PACKED_ROWS4_MATMUL",
            "CAMELID_X86_Q8_OUTPUT_PACKED_ROWS4_MATMUL",
            "CAMELID_X86_Q8_OUTPUT_AMX_PREFILL",
            "CAMELID_X86_Q8_PACKED_ROWS4_SERIAL_DECODE",
            "CAMELID_X86_Q8_PARALLEL_INPUT_QUANTIZE",
            "CAMELID_X86_Q8_FFN_GATE_UP_DECODE_CONSUMER",
            "CAMELID_X86_Q8_FFN_GATE_UP_DECODE_GROUP_CHUNKING",
            "CAMELID_X86_Q8_FFN_GATE_UP_DECODE_GROUPS_PER_CHUNK",
            "CAMELID_X86_Q8_FFN_GATE_UP_DECODE_FUSED_ACTIVATION",
            "CAMELID_X86_Q8_FFN_GATE_UP_DECODE_PAIRED_DOT",
            "CAMELID_X86_Q8_FFN_DECODE_CHAIN",
            "CAMELID_X86_Q8_FFN_GATE_UP_PACKED_ROWS4_MATMUL",
            "CAMELID_X86_Q8_FFN_GATE_UP_SINGLE_OWNER",
            "CAMELID_X86_Q8_FFN_DOWN_DECODE_CONSUMER",
            "CAMELID_X86_Q8_FFN_DOWN_PACKED_ROWS4_MATMUL",
            "CAMELID_X86_Q8_PACKED_ROWS4_MATMUL",
            "CAMELID_X86_Q8_FFN_DOWN_GEMM4_PREFILL",
            "CAMELID_X86_Q8_FFN_DOWN_GEMM4_ROW_GROUP_SCHED",
            "CAMELID_X86_Q8_FFN_DOWN_GEMM4_ROW_GROUP_MIN_INPUT_GROUPS",
            "CAMELID_X86_Q8_FFN_DOWN_GEMM4_AVX2",
            "CAMELID_X86_Q8_FFN_DOWN_AMX_PREFILL",
            "CAMELID_X86_Q8_FFN_DOWN_SINGLE_OWNER",
            "CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE",
            "CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE_RAWPTR",
            "CAMELID_X86_Q8_FFN_DOWN_DECODE_OWNER",
            "CAMELID_X86_Q8_OUTPUT_DECODE_OWNER",
            "CAMELID_X86_Q8_PACKED_ROWS4_MATMUL_GROUPS_PER_CHUNK",
        ] {
            env::remove_var(key);
        }
    }

    #[test]
    fn junk_general_name_falls_back_to_recognizable_filename() {
        // A junk general.name ("hub", from some conversions) maps to no support row; it
        // must defer to a recognizable filename so the model reaches its validated row
        // instead of failing closed — and must never override a name that already matches.
        let row = exact_model_row(
            &PathBuf::from("/models/Meta-Llama-3-8B-Instruct.Q8_0.gguf"),
            &fixture("hub"),
        );
        assert!(
            is_supported_exact_q8_row(&row),
            "junk general.name must fall back to the recognized filename; got {row:?}"
        );
        // Junk name AND unrecognizable filename stays unrecognized.
        assert_eq!(
            recognized_row_level(&exact_model_row(
                &PathBuf::from("/models/mystery.gguf"),
                &fixture("hub")
            )),
            "unknown_or_unvalidated"
        );
        // A recognized general.name is never overridden by an unrelated filename.
        assert_eq!(
            exact_model_row(
                &PathBuf::from("/models/whatever.gguf"),
                &fixture("Llama 3.2 1B Instruct")
            ),
            "Llama 3.2 1B Instruct"
        );
    }

    #[test]
    fn junk_named_recognizable_8b_takes_gpu_lane_not_cpu() {
        let _guard = env_lock();
        clear_profile_env();
        let outcome = plan_for_model_with_platform(
            &PathBuf::from("/models/Meta-Llama-3-8B-Instruct.Q8_0.gguf"),
            &fixture("hub"),
            Some(8),
            cuda_platform("windows", "x86_64", &["avx2"]),
        );
        clear_profile_env();
        assert_eq!(
            outcome.plan.exact_model_row,
            "Meta-Llama-3-8B-Instruct.Q8_0.gguf"
        );
        assert_ne!(
            outcome.plan.selected_backend, "cpu_reference",
            "a recognizable 8B with a junk general.name must not fail closed to CPU"
        );
    }

    #[test]
    fn gpu_runnable_tier_admits_uncurated_q8_llama_by_default_optout_forces_cpu() {
        let _guard = env_lock();
        clear_profile_env();
        // Uncurated: neither general.name ("hub") nor the filename maps to a support row.
        let uncurated = fixture("hub");
        let path = PathBuf::from("/models/my-custom-llama-Q8_0.gguf");
        // DEFAULT (unset): admitted to the GPU-runnable tier with an honest, distinct label; the
        // support_level stays unknown_or_unvalidated (never claims a supported row). Admission is
        // gated at runtime by the parity self-check, so default-on is safe.
        env::remove_var("CAMELID_GPU_RUNNABLE_TIER");
        let on = plan_for_model_with_platform(
            &path,
            &uncurated,
            Some(8),
            cuda_platform("windows", "x86_64", &["avx2"]),
        );
        assert_eq!(
            on.plan.selected_backend,
            "cuda_resident_q8_runtime_runnable_unvalidated"
        );
        assert_eq!(
            on.plan.support_level, "unknown_or_unvalidated",
            "the runnable tier must never claim a supported row"
        );
        // Explicit opt-out (=0): forced back to the safe CPU reference path.
        env::set_var("CAMELID_GPU_RUNNABLE_TIER", "0");
        let off = plan_for_model_with_platform(
            &path,
            &uncurated,
            Some(8),
            cuda_platform("windows", "x86_64", &["avx2"]),
        );
        env::remove_var("CAMELID_GPU_RUNNABLE_TIER");
        clear_profile_env();
        assert_eq!(off.plan.selected_backend, "cpu_reference");
    }

    #[test]
    fn gpu_runnable_tier_never_changes_a_curated_row() {
        let _guard = env_lock();
        clear_profile_env();
        let curated = fixture("Llama 3.2 1B Instruct");
        let path = PathBuf::from("/models/Llama-3.2-1B-Instruct-Q8_0.gguf");
        // Tier ON (default, unset).
        env::remove_var("CAMELID_GPU_RUNNABLE_TIER");
        let on = plan_for_model_with_platform(
            &path,
            &curated,
            Some(8),
            cuda_platform("windows", "x86_64", &["avx2"]),
        );
        // Tier opted out (=0).
        env::set_var("CAMELID_GPU_RUNNABLE_TIER", "0");
        let off = plan_for_model_with_platform(
            &path,
            &curated,
            Some(8),
            cuda_platform("windows", "x86_64", &["avx2"]),
        );
        env::remove_var("CAMELID_GPU_RUNNABLE_TIER");
        clear_profile_env();
        // A curated row takes the supported plan and is byte-for-byte identical whether the tier
        // is on (default) or opted out — the tier is a pure additive else-branch after the
        // curated arm, so it can never alter a supported row.
        assert_eq!(on.plan.selected_backend, "cuda_resident_q8_runtime");
        assert_eq!(on.plan.selected_backend, off.plan.selected_backend);
        assert_eq!(on.plan.support_level, off.plan.support_level);
        assert_eq!(on.plan.decode_path, off.plan.decode_path);
    }

    #[test]
    fn safe_profile_fails_closed() {
        let _guard = env_lock();
        clear_profile_env();
        env::set_var("CAMELID_PROFILE", "safe");
        let outcome = plan_for_model_with_platform(
            &PathBuf::from("/tmp/Llama-3.2-3B-Instruct-Q8_0.gguf"),
            &fixture("Llama 3.2 3B Instruct"),
            Some(8),
            platform("macos", "aarch64", &["dotprod", "i8mm"]),
        );
        assert_eq!(outcome.plan.profile, ExecutionProfile::Safe);
        assert_eq!(outcome.plan.selected_backend, "cpu_reference");
        assert_eq!(
            outcome.plan.prefill_runtime_policy,
            "always_retained_reference_path"
        );
        assert!(!outcome.env_updates.contains_key("CAMELID_MAC_Q8_REPACK"));
        clear_profile_env();
    }

    #[test]
    fn kquant_plan_labels_resident_and_cpu_block_dot_not_cpu_reference() {
        // Disclosure fix: a Q4_K_M model must NOT be labeled the dense_or_other /
        // cpu_reference fallback. quant_type is Q4_K_M, and the backend reflects the
        // real lane: GPU-resident when CUDA drives decode, CPU block-dot otherwise,
        // and only cpu_reference when the block-dot is explicitly disabled with no GPU.
        let _guard = env_lock();
        clear_profile_env();
        env::remove_var("CAMELID_X86_Q4K_DECODE");
        let mut gguf = fixture("Qwen3 4B Instruct Q4_K_M");
        gguf.tensors[0].tensor_type = GgufTensorType::Q4K;
        let path = PathBuf::from("/tmp/Qwen3-4B-Q4_K_M.gguf");

        let cpu = plan_for_model_with_platform(
            &path,
            &gguf,
            Some(8),
            platform("windows", "x86_64", &["avx2"]),
        );
        assert_eq!(cpu.plan.quant_type, "Q4_K_M");
        assert_eq!(cpu.plan.selected_backend, "cpu_kquant_block_dot");
        assert_eq!(cpu.plan.decode_path, "kquant_cpu_block_dot_decode");

        let gpu = plan_for_model_with_platform(
            &path,
            &gguf,
            Some(8),
            cuda_platform("windows", "x86_64", &["avx2"]),
        );
        assert_eq!(gpu.plan.selected_backend, "cuda_resident_kquant_runtime");
        assert_eq!(gpu.plan.decode_path, "kquant_cuda_resident_decode");
        assert!(gpu.plan.cuda_resident_active);

        env::set_var("CAMELID_X86_Q4K_DECODE", "0");
        let off = plan_for_model_with_platform(
            &path,
            &gguf,
            Some(8),
            platform("windows", "x86_64", &["avx2"]),
        );
        assert_eq!(off.plan.selected_backend, "cpu_reference");
        env::remove_var("CAMELID_X86_Q4K_DECODE");
        clear_profile_env();
    }

    #[test]
    fn pure_q6k_reports_q6k_label_unknown_level_and_keeps_kquant_routing() {
        // Truth-in-labeling: a pure-Q6_K file (declared ftype 18) must not be
        // collapsed into the "Q4_K_M" bucket, and a Q6_K file of the TinyLlama
        // gate row's NAME must not echo the Q8_0 row's support level. Routing
        // is unchanged: it branches on the tensor scan, never on the label.
        let _guard = env_lock();
        clear_profile_env();
        env::remove_var("CAMELID_X86_Q4K_DECODE");
        let q6k = quant_fixture("TinyLlama 1.1B Chat v1.0", Some(18), &[GgufTensorType::Q6K]);
        let path = PathBuf::from("/tmp/tinyllama-1.1b-chat-v1.0.Q6_K.gguf");

        let cpu = plan_for_model_with_platform(
            &path,
            &q6k,
            Some(8),
            platform("windows", "x86_64", &["avx2"]),
        );
        assert_eq!(cpu.plan.quant_type, "Q6_K");
        assert_eq!(
            cpu.plan.support_level, "unknown_or_unvalidated",
            "a Q6_K TinyLlama must not inherit the Q8_0 gate row's support level"
        );
        assert_eq!(
            cpu.plan.selected_backend, "cpu_kquant_block_dot",
            "the label fix must not change K-quant routing"
        );

        let gpu = plan_for_model_with_platform(
            &path,
            &q6k,
            Some(8),
            cuda_platform("windows", "x86_64", &["avx2"]),
        );
        assert_eq!(gpu.plan.selected_backend, "cuda_resident_kquant_runtime");
        clear_profile_env();
    }

    #[test]
    fn declared_file_type_names_tq_and_kquant_variants() {
        // A TQ1_0 file that carries a few K-quant tensors reports TQ1_0 (not a
        // K-quant mix guess), and a declared Q5_K_M names itself precisely.
        let tq = quant_fixture(
            "Ternary Bonsai 4B",
            Some(36),
            &[GgufTensorType::Tq1_0, GgufTensorType::Q4K],
        );
        assert_eq!(quant_type(&tq), "TQ1_0");
        let q5km = quant_fixture(
            "whatever",
            Some(17),
            &[GgufTensorType::Q5K, GgufTensorType::Q6K],
        );
        assert_eq!(quant_type(&q5km), "Q5_K_M");
        // Undeclared file_type falls back to the tensor-scan buckets.
        let undeclared_q6k = quant_fixture("whatever", None, &[GgufTensorType::Q6K]);
        assert_eq!(quant_type(&undeclared_q6k), "Q6_K");
        let undeclared_mix = quant_fixture(
            "whatever",
            None,
            &[GgufTensorType::Q4K, GgufTensorType::Q6K],
        );
        assert_eq!(quant_type(&undeclared_mix), "Q4_K_M");
    }

    #[test]
    fn declared_q8_0_without_q8_0_tensors_is_refused() {
        // A wrong general.file_type=7 with zero Q8_0 tensors must not produce
        // the "Q8_0" label — that label is what gates support_level onto a
        // promoted row (and the supported-Q8 planner branch keys on tensors,
        // so the file routes as what it actually is either way).
        let lying = quant_fixture("TinyLlama 1.1B Chat v1.0", Some(7), &[GgufTensorType::Q6K]);
        assert_eq!(quant_type(&lying), "Q6_K");
        assert_eq!(
            support_level("TinyLlama 1.1B Chat v1.0", &quant_type(&lying)),
            "unknown_or_unvalidated"
        );
    }

    #[test]
    fn supported_q8_row_keeps_level_and_kquant_name_variant_reports_unknown() {
        // Anchor: quant-gating must not move any promoted Q8_0 row…
        let _guard = env_lock();
        clear_profile_env();
        let q8 = quant_fixture("TinyLlama 1.1B Chat v1.0", Some(7), &[GgufTensorType::Q8_0]);
        let outcome = plan_for_model_with_platform(
            &PathBuf::from("/tmp/tinyllama-1.1b-chat-v1.0.Q8_0.gguf"),
            &q8,
            Some(8),
            platform("windows", "x86_64", &["avx2"]),
        );
        assert_eq!(outcome.plan.quant_type, "Q8_0");
        assert_eq!(outcome.plan.support_level, "supported_current_gate");
        assert_eq!(outcome.plan.selected_backend, "cpu_q8_runtime_repack");
        clear_profile_env();

        // …while a K-quant file sharing a supported row's NAME stays unknown in
        // the plan string (its own K-quant claim lives in /api/capabilities),
        // and row recognition stays quant-blind for the planner gate.
        assert_eq!(
            support_level("Llama 3.2 3B Instruct", "Q4_K_M"),
            "unknown_or_unvalidated"
        );
        assert!(is_supported_exact_q8_row("Llama 3.2 3B Instruct"));
    }

    #[test]
    fn mac_metal_resident_plan_selected_when_device_and_gate_present() {
        let _guard = env_lock();
        clear_profile_env();
        env::set_var("CAMELID_METAL_RESIDENT_DECODE", "1");
        let outcome = plan_for_model_with_platform(
            &PathBuf::from("/tmp/Llama-3.2-3B-Instruct-Q8_0.gguf"),
            &fixture("Llama 3.2 3B Instruct"),
            Some(10),
            metal_platform("macos", "aarch64", &["dotprod", "i8mm"]),
        );
        assert_eq!(outcome.plan.selected_backend, "metal_resident_q8_runtime");
        assert_eq!(outcome.plan.decode_path, "q8_0_metal_resident_decode");
        // The rows4 repack must stay OFF: the GPU path needs plain Q8_0 blocks.
        assert_eq!(
            outcome.env_updates.get("CAMELID_MAC_Q8_REPACK"),
            Some(&Some("off"))
        );
        env::remove_var("CAMELID_METAL_RESIDENT_DECODE");
        clear_profile_env();
    }

    #[test]
    fn windows_cuda_resident_plan_selected_when_engine_active() {
        let _guard = env_lock();
        clear_profile_env();
        // A supported Qwen3 Q8_0 row on a Windows x86_64 host where the CUDA resident
        // decode engine is active: the plan surfaces the GPU-resident backend/decode
        // labels and reports cuda_resident_active, while keeping the row's
        // supported_exact_row_smoke_chatml support level (engine-agnostic).
        let outcome = plan_for_model_with_platform(
            &PathBuf::from("/tmp/Qwen3-0.6B-Q8_0.gguf"),
            &fixture("Qwen3 0.6B Instruct"),
            Some(8),
            cuda_platform("windows", "x86_64", &["avx2"]),
        );
        assert_eq!(outcome.plan.selected_backend, "cuda_resident_q8_runtime");
        assert_eq!(outcome.plan.decode_path, "q8_0_cuda_resident_decode");
        assert_eq!(outcome.plan.prefill_path, "q8_0_cuda_resident_prefill");
        assert!(outcome.plan.cuda_resident_active);
        assert_eq!(
            outcome.plan.support_level, "supported_exact_row_smoke_chatml",
            "GPU lane reuses the row-keyed support level (Phase 1 design)"
        );
        // The GPU consumes plain Q8_0 blocks: the x86 rows4 repack must NOT be enabled.
        assert_ne!(
            outcome.env_updates.get("CAMELID_X86_Q8_REPACK"),
            Some(&Some("on"))
        );
        clear_profile_env();
    }

    #[test]
    fn windows_cuda_resident_inactive_keeps_cpu_repack_plan() {
        let _guard = env_lock();
        clear_profile_env();
        // Same row/host but no active CUDA engine: the validated x86_64 CPU repack plan.
        let outcome = plan_for_model_with_platform(
            &PathBuf::from("/tmp/Qwen3-0.6B-Q8_0.gguf"),
            &fixture("Qwen3 0.6B Instruct"),
            Some(8),
            platform("windows", "x86_64", &["avx2"]),
        );
        assert_eq!(outcome.plan.selected_backend, "cpu_q8_runtime_repack");
        assert!(!outcome.plan.cuda_resident_active);
        clear_profile_env();
    }

    #[test]
    fn mac_metal_plan_requires_device_and_gate() {
        let _guard = env_lock();
        clear_profile_env();
        // Gate present but no Metal device: validated CPU repack plan.
        env::set_var("CAMELID_METAL_RESIDENT_DECODE", "1");
        let outcome = plan_for_model_with_platform(
            &PathBuf::from("/tmp/Llama-3.2-3B-Instruct-Q8_0.gguf"),
            &fixture("Llama 3.2 3B Instruct"),
            Some(10),
            platform("macos", "aarch64", &["dotprod", "i8mm"]),
        );
        assert_eq!(outcome.plan.selected_q8_path, "mac_validated_q8_0_repack");
        env::remove_var("CAMELID_METAL_RESIDENT_DECODE");
        // Device present but gate absent (embedder/test default): CPU repack plan.
        let outcome = plan_for_model_with_platform(
            &PathBuf::from("/tmp/Llama-3.2-3B-Instruct-Q8_0.gguf"),
            &fixture("Llama 3.2 3B Instruct"),
            Some(10),
            metal_platform("macos", "aarch64", &["dotprod", "i8mm"]),
        );
        assert_eq!(outcome.plan.selected_q8_path, "mac_validated_q8_0_repack");
        // Explicit opt-out returns the CPU repack plan even with device + gate.
        env::set_var("CAMELID_METAL_RESIDENT_DECODE", "1");
        env::set_var("CAMELID_MAC_Q8_METAL_PLAN", "0");
        let outcome = plan_for_model_with_platform(
            &PathBuf::from("/tmp/Llama-3.2-3B-Instruct-Q8_0.gguf"),
            &fixture("Llama 3.2 3B Instruct"),
            Some(10),
            metal_platform("macos", "aarch64", &["dotprod", "i8mm"]),
        );
        assert_eq!(outcome.plan.selected_q8_path, "mac_validated_q8_0_repack");
        env::remove_var("CAMELID_METAL_RESIDENT_DECODE");
        env::remove_var("CAMELID_MAC_Q8_METAL_PLAN");
        clear_profile_env();
    }

    #[test]
    fn mac_auto_selects_validated_mac_path() {
        let _guard = env_lock();
        clear_profile_env();
        let outcome = plan_for_model_with_platform(
            &PathBuf::from("/tmp/Llama-3.2-3B-Instruct-Q8_0.gguf"),
            &fixture("Llama 3.2 3B Instruct"),
            Some(10),
            platform("macos", "aarch64", &["dotprod", "i8mm"]),
        );
        assert_eq!(outcome.plan.profile, ExecutionProfile::Auto);
        assert_eq!(outcome.plan.selected_q8_path, "mac_validated_q8_0_repack");
        assert_eq!(
            outcome.plan.prefill_path,
            "q8_0_direct_pack_prefill_i8mm_available"
        );
        assert_eq!(
            outcome.plan.prefill_runtime_policy,
            "enabled_when_prefill_rows_gte_4"
        );
        assert_eq!(
            outcome.env_updates.get("CAMELID_PARALLEL_LINEAR"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome.env_updates.get("CAMELID_MAC_Q8_REPACK"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome.env_updates.get("CAMELID_MAC_Q8_PREFILL_I8MM"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome.env_updates.get("CAMELID_MAC_Q8_SCHED"),
            Some(&Some("off"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_MAC_Q8_FFN_DOWN_DECODE_CONSUMER"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_MAC_Q8_FFN_GATE_UP_DECODE_CONSUMER"),
            Some(&Some("on"))
        );
        assert!(!outcome.env_updates.contains_key("CAMELID_X86_Q8_KERNEL"));
        clear_profile_env();
    }

    #[test]
    fn mac_experimental_allows_packed_prefill_scheduler() {
        let _guard = env_lock();
        clear_profile_env();
        env::set_var("CAMELID_PROFILE", "experimental");
        env::set_var("CAMELID_MAC_Q8_PREFILL_I8MM", "on");
        let outcome = plan_for_model_with_platform(
            &PathBuf::from("/tmp/Llama-3.2-3B-Instruct-Q8_0.gguf"),
            &fixture("Llama 3.2 3B Instruct"),
            Some(10),
            platform("macos", "aarch64", &["dotprod", "i8mm"]),
        );
        assert_eq!(outcome.plan.profile, ExecutionProfile::Experimental);
        assert_eq!(
            outcome.plan.prefill_path,
            "q8_0_experimental_packed_prefill_i8mm_available"
        );
        assert_eq!(
            outcome.plan.prefill_runtime_policy,
            "enabled_when_prefill_rows_gte_4"
        );
        assert_eq!(
            outcome.env_updates.get("CAMELID_MAC_Q8_SCHED"),
            Some(&Some("packed_prefill"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_MAC_Q8_FFN_DOWN_DECODE_CONSUMER"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_MAC_Q8_FFN_GATE_UP_DECODE_CONSUMER"),
            Some(&Some("on"))
        );
        clear_profile_env();
    }

    #[test]
    fn mac_ffn_decode_consumer_plan_gates_are_default_on_and_opt_out() {
        let _guard = env_lock();
        clear_profile_env();
        let default_outcome = plan_for_model_with_platform(
            &PathBuf::from("/tmp/Llama-3.2-3B-Instruct-Q8_0.gguf"),
            &fixture("Llama 3.2 3B Instruct"),
            Some(10),
            platform("macos", "aarch64", &["dotprod", "i8mm"]),
        );
        assert_eq!(
            default_outcome
                .env_updates
                .get("CAMELID_MAC_Q8_FFN_DOWN_DECODE_CONSUMER"),
            Some(&Some("on"))
        );
        assert_eq!(
            default_outcome
                .env_updates
                .get("CAMELID_MAC_Q8_FFN_GATE_UP_DECODE_CONSUMER"),
            Some(&Some("on"))
        );

        clear_profile_env();
        env::set_var("CAMELID_MAC_Q8_FFN_DOWN_DECODE_CONSUMER", "off");
        env::set_var("CAMELID_MAC_Q8_FFN_GATE_UP_DECODE_CONSUMER", "off");
        let opt_out_outcome = plan_for_model_with_platform(
            &PathBuf::from("/tmp/Llama-3.2-3B-Instruct-Q8_0.gguf"),
            &fixture("Llama 3.2 3B Instruct"),
            Some(10),
            platform("macos", "aarch64", &["dotprod", "i8mm"]),
        );
        assert_eq!(
            opt_out_outcome
                .env_updates
                .get("CAMELID_MAC_Q8_FFN_DOWN_DECODE_CONSUMER"),
            Some(&Some("off"))
        );
        assert_eq!(
            opt_out_outcome
                .env_updates
                .get("CAMELID_MAC_Q8_FFN_GATE_UP_DECODE_CONSUMER"),
            Some(&Some("off"))
        );
        clear_profile_env();
    }

    #[test]
    fn mac_auto_explicit_matches_auto_default_plan() {
        let _guard = env_lock();
        clear_profile_env();
        let default_outcome = plan_for_model_with_platform(
            &PathBuf::from("/tmp/Llama-3.2-3B-Instruct-Q8_0.gguf"),
            &fixture("Llama 3.2 3B Instruct"),
            Some(10),
            platform("macos", "aarch64", &["dotprod", "i8mm"]),
        );
        clear_profile_env();
        env::set_var("CAMELID_PROFILE", "auto");
        let explicit_outcome = plan_for_model_with_platform(
            &PathBuf::from("/tmp/Llama-3.2-3B-Instruct-Q8_0.gguf"),
            &fixture("Llama 3.2 3B Instruct"),
            Some(10),
            platform("macos", "aarch64", &["dotprod", "i8mm"]),
        );
        assert_eq!(default_outcome.plan.profile, explicit_outcome.plan.profile);
        assert_eq!(
            default_outcome.plan.selected_backend,
            explicit_outcome.plan.selected_backend
        );
        assert_eq!(
            default_outcome.plan.selected_q8_path,
            explicit_outcome.plan.selected_q8_path
        );
        assert_eq!(
            default_outcome.plan.prefill_path,
            explicit_outcome.plan.prefill_path
        );
        assert_eq!(
            default_outcome.plan.prefill_runtime_policy,
            explicit_outcome.plan.prefill_runtime_policy
        );
        assert_eq!(
            default_outcome.plan.decode_path,
            explicit_outcome.plan.decode_path
        );
        assert_eq!(
            default_outcome.plan.fallback_path,
            explicit_outcome.plan.fallback_path
        );
        assert_eq!(default_outcome.env_updates, explicit_outcome.env_updates);
        clear_profile_env();
    }

    #[test]
    fn ubuntu_auto_enables_x86_optimizations_by_default() {
        let _guard = env_lock();
        clear_profile_env();
        let outcome = plan_for_model_with_platform(
            &PathBuf::from("/tmp/Llama-3.2-3B-Instruct-Q8_0.gguf"),
            &fixture("Llama 3.2 3B Instruct"),
            Some(16),
            platform(
                "linux",
                "x86_64",
                &["avx2", "avx512f", "avx512_vnni", "amx_int8"],
            ),
        );
        assert_eq!(outcome.plan.profile, ExecutionProfile::Auto);
        assert_eq!(outcome.plan.selected_backend, "cpu_q8_runtime_repack");
        assert_eq!(
            outcome.plan.selected_q8_path,
            "x86_experimental_q8_0_avx2_rust"
        );
        assert_eq!(
            outcome.env_updates.get("CAMELID_X86_Q8_KERNEL"),
            Some(&Some("avx2"))
        );
        assert_eq!(
            outcome.env_updates.get("CAMELID_X86_Q8_REPACK"),
            Some(&Some("on"))
        );
        assert!(!outcome.env_updates.contains_key("CAMELID_MAC_Q8_REPACK"));
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_FFN_DOWN_DECODE_CONSUMER"),
            Some(&Some("off"))
        );
        clear_profile_env();
    }

    #[test]
    fn ubuntu_experimental_validated_gates_select_rust_avx2_q8_path() {
        let _guard = env_lock();
        clear_profile_env();
        env::set_var("CAMELID_PROFILE", "experimental");
        env::set_var("CAMELID_X86_Q8_REPACK", "on");
        env::set_var("CAMELID_X86_Q8_KERNEL", "avx2");
        let outcome = plan_for_model_with_platform(
            &PathBuf::from("/tmp/Llama-3.2-3B-Instruct-Q8_0.gguf"),
            &fixture("Llama 3.2 3B Instruct"),
            Some(16),
            platform("linux", "x86_64", &["avx2", "avx512f"]),
        );
        assert_eq!(outcome.plan.profile, ExecutionProfile::Experimental);
        assert_eq!(outcome.plan.selected_backend, "cpu_q8_runtime_repack");
        assert_eq!(
            outcome.plan.selected_q8_path,
            "x86_experimental_q8_0_avx2_rust"
        );
        assert_eq!(
            outcome.plan.prefill_runtime_policy,
            "enabled_when_q8_runtime_storage_active"
        );
        assert_eq!(
            outcome.env_updates.get("CAMELID_PARALLEL_LINEAR"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome.env_updates.get("CAMELID_X86_Q8_REPACK"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome.env_updates.get("CAMELID_X86_Q8_KERNEL"),
            Some(&Some("avx2"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_ATTENTION_PROJECTION_DECODE_CONSUMER"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_ATTENTION_OUTPUT_DECODE_CONSUMER"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_ATTENTION_OUTPUT_PACKED_ROWS4_MATMUL"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_ATTENTION_QKV_DECODE_CONSUMER"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_ATTENTION_QKV_DECODE_GROUP_CHUNKING"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_ATTENTION_QKV_PACKED_ROWS4_MATMUL"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_OUTPUT_PACKED_ROWS4_MATMUL"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome.env_updates.get("CAMELID_X86_Q8_OUTPUT_AMX_PREFILL"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_PACKED_ROWS4_SERIAL_DECODE"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_PARALLEL_INPUT_QUANTIZE"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_CONSUMER"),
            Some(&Some("off"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_GROUP_CHUNKING"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_FUSED_ACTIVATION"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_PAIRED_DOT"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome.env_updates.get("CAMELID_X86_Q8_FFN_DECODE_CHAIN"),
            Some(&Some("off"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_FFN_GATE_UP_PACKED_ROWS4_MATMUL"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_FFN_GATE_UP_SINGLE_OWNER"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_FFN_DOWN_DECODE_CONSUMER"),
            Some(&Some("off"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_PACKED_ROWS4_MATMUL"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_FFN_DOWN_PACKED_ROWS4_MATMUL"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_FFN_DOWN_GEMM4_PREFILL"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_FFN_DOWN_GEMM4_ROW_GROUP_SCHED"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_FFN_DOWN_GEMM4_AVX2"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_FFN_DOWN_AMX_PREFILL"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_FFN_DOWN_SINGLE_OWNER"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE_RAWPTR"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_FFN_DOWN_DECODE_OWNER"),
            Some(&Some("off"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_OUTPUT_DECODE_OWNER"),
            Some(&Some("on"))
        );
        clear_profile_env();
    }

    #[test]
    fn ubuntu_experimental_preserves_explicit_x86_q8_gate_opt_ins() {
        let _guard = env_lock();
        clear_profile_env();
        env::set_var("CAMELID_PROFILE", "experimental");
        env::set_var("CAMELID_X86_Q8_REPACK", "on");
        env::set_var("CAMELID_X86_Q8_KERNEL", "avx2");
        env::set_var("CAMELID_X86_Q8_FFN_DOWN_DECODE_CONSUMER", "on");
        env::set_var("CAMELID_X86_Q8_FFN_DOWN_PACKED_ROWS4_MATMUL", "on");
        env::set_var("CAMELID_X86_Q8_FFN_DOWN_GEMM4_PREFILL", "on");
        env::set_var("CAMELID_X86_Q8_FFN_DOWN_GEMM4_ROW_GROUP_SCHED", "on");
        env::set_var("CAMELID_X86_Q8_FFN_DOWN_GEMM4_AVX2", "on");
        env::set_var("CAMELID_X86_Q8_FFN_DOWN_AMX_PREFILL", "on");
        env::set_var("CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE", "on");
        env::set_var("CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE_RAWPTR", "on");
        env::set_var("CAMELID_X86_Q8_OUTPUT_DECODE_OWNER", "on");
        env::set_var("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_PAIRED_DOT", "on");
        env::set_var("CAMELID_X86_Q8_FFN_DECODE_CHAIN", "on");
        env::set_var("CAMELID_X86_Q8_PACKED_ROWS4_SERIAL_DECODE", "on");
        env::set_var("CAMELID_X86_Q8_PARALLEL_INPUT_QUANTIZE", "on");
        env::set_var("CAMELID_X86_Q8_ATTENTION_QKV_PACKED_ROWS4_MATMUL", "off");
        let outcome = plan_for_model_with_platform(
            &PathBuf::from("/tmp/Llama-3.2-3B-Instruct-Q8_0.gguf"),
            &fixture("Llama 3.2 3B Instruct"),
            Some(16),
            platform("linux", "x86_64", &["avx2", "avx512f"]),
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_FFN_DOWN_DECODE_CONSUMER"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_FFN_DOWN_PACKED_ROWS4_MATMUL"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_FFN_DOWN_GEMM4_PREFILL"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_FFN_DOWN_GEMM4_ROW_GROUP_SCHED"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_FFN_DOWN_GEMM4_AVX2"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_FFN_DOWN_AMX_PREFILL"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE_RAWPTR"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_OUTPUT_DECODE_OWNER"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_PAIRED_DOT"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome.env_updates.get("CAMELID_X86_Q8_FFN_DECODE_CHAIN"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_PACKED_ROWS4_SERIAL_DECODE"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_PARALLEL_INPUT_QUANTIZE"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_ATTENTION_QKV_PACKED_ROWS4_MATMUL"),
            Some(&Some("off"))
        );
        clear_profile_env();
    }

    #[test]
    fn ubuntu_experimental_ffn_decode_chain_enables_required_gate_up_and_down_legs() {
        let _guard = env_lock();
        clear_profile_env();
        env::set_var("CAMELID_PROFILE", "experimental");
        env::set_var("CAMELID_X86_Q8_REPACK", "on");
        env::set_var("CAMELID_X86_Q8_KERNEL", "avx2");
        env::set_var("CAMELID_X86_Q8_FFN_DECODE_CHAIN", "on");
        let outcome = plan_for_model_with_platform(
            &PathBuf::from("/tmp/Llama-3.2-3B-Instruct-Q8_0.gguf"),
            &fixture("Llama 3.2 3B Instruct"),
            Some(16),
            platform("linux", "x86_64", &["avx2", "avx512f"]),
        );
        assert_eq!(
            outcome.env_updates.get("CAMELID_X86_Q8_FFN_DECODE_CHAIN"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_CONSUMER"),
            Some(&Some("on"))
        );
        assert_eq!(
            outcome
                .env_updates
                .get("CAMELID_X86_Q8_FFN_DOWN_DECODE_CONSUMER"),
            Some(&Some("on"))
        );
        assert!(outcome.plan.reasons.iter().any(|reason| reason.contains(
            "FFN decode-chain opt-in also enables the required FFN gate/up decode consumer gate"
        )));
        assert!(outcome.plan.reasons.iter().any(|reason| reason.contains(
            "FFN decode-chain opt-in also enables the required FFN-down decode consumer gate"
        )));
        clear_profile_env();
    }

    #[test]
    fn planner_env_apply_clears_stale_x86_q8_decode_consumer_flags() {
        let _guard = env_lock();
        clear_profile_env();
        env::set_var("CAMELID_X86_Q8_ATTENTION_PROJECTION_DECODE_CONSUMER", "on");
        env::set_var("CAMELID_X86_Q8_ATTENTION_OUTPUT_DECODE_CONSUMER", "on");
        env::set_var("CAMELID_X86_Q8_ATTENTION_OUTPUT_PACKED_ROWS4_MATMUL", "on");
        env::set_var("CAMELID_X86_Q8_ATTENTION_QKV_DECODE_CONSUMER", "on");
        env::set_var("CAMELID_X86_Q8_ATTENTION_QKV_DECODE_GROUP_CHUNKING", "on");
        env::set_var("CAMELID_X86_Q8_ATTENTION_QKV_DECODE_GROUPS_PER_CHUNK", "7");
        env::set_var("CAMELID_X86_Q8_ATTENTION_QKV_PACKED_ROWS4_MATMUL", "on");
        env::set_var("CAMELID_X86_Q8_OUTPUT_PACKED_ROWS4_MATMUL", "on");
        env::set_var("CAMELID_X86_Q8_OUTPUT_AMX_PREFILL", "on");
        env::set_var("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_CONSUMER", "on");
        env::set_var("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_GROUP_CHUNKING", "on");
        env::set_var("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_GROUPS_PER_CHUNK", "5");
        env::set_var("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_FUSED_ACTIVATION", "on");
        env::set_var("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_PAIRED_DOT", "on");
        env::set_var("CAMELID_X86_Q8_FFN_DECODE_CHAIN", "on");
        env::set_var("CAMELID_X86_Q8_FFN_GATE_UP_PACKED_ROWS4_MATMUL", "on");
        env::set_var("CAMELID_X86_Q8_FFN_GATE_UP_SINGLE_OWNER", "on");
        env::set_var("CAMELID_X86_Q8_FFN_DOWN_DECODE_CONSUMER", "on");
        env::set_var("CAMELID_X86_Q8_FFN_DOWN_PACKED_ROWS4_MATMUL", "on");
        env::set_var("CAMELID_X86_Q8_PACKED_ROWS4_MATMUL", "on");
        env::set_var("CAMELID_X86_Q8_FFN_DOWN_GEMM4_PREFILL", "on");
        env::set_var("CAMELID_X86_Q8_FFN_DOWN_GEMM4_ROW_GROUP_SCHED", "on");
        env::set_var(
            "CAMELID_X86_Q8_FFN_DOWN_GEMM4_ROW_GROUP_MIN_INPUT_GROUPS",
            "3",
        );
        env::set_var("CAMELID_X86_Q8_PACKED_ROWS4_MATMUL_GROUPS_PER_CHUNK", "9");
        env::set_var("CAMELID_X86_Q8_FFN_DOWN_GEMM4_AVX2", "on");
        env::set_var("CAMELID_X86_Q8_FFN_DOWN_AMX_PREFILL", "on");
        env::set_var("CAMELID_X86_Q8_FFN_DOWN_SINGLE_OWNER", "on");
        env::set_var("CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE", "on");
        env::set_var("CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE_RAWPTR", "on");
        env::set_var("CAMELID_X86_Q8_OUTPUT_DECODE_OWNER", "on");
        env::set_var("CAMELID_X86_Q8_PACKED_ROWS4_SERIAL_DECODE", "on");
        env::set_var("CAMELID_X86_Q8_PARALLEL_INPUT_QUANTIZE", "on");

        PlannerEnv::capture().apply(&BTreeMap::new());

        assert!(env::var("CAMELID_X86_Q8_ATTENTION_PROJECTION_DECODE_CONSUMER").is_err());
        assert!(env::var("CAMELID_X86_Q8_ATTENTION_OUTPUT_DECODE_CONSUMER").is_err());
        assert!(env::var("CAMELID_X86_Q8_ATTENTION_OUTPUT_PACKED_ROWS4_MATMUL").is_err());
        assert!(env::var("CAMELID_X86_Q8_ATTENTION_QKV_DECODE_CONSUMER").is_err());
        assert!(env::var("CAMELID_X86_Q8_ATTENTION_QKV_DECODE_GROUP_CHUNKING").is_err());
        assert!(env::var("CAMELID_X86_Q8_ATTENTION_QKV_DECODE_GROUPS_PER_CHUNK").is_err());
        assert!(env::var("CAMELID_X86_Q8_ATTENTION_QKV_PACKED_ROWS4_MATMUL").is_err());
        assert!(env::var("CAMELID_X86_Q8_OUTPUT_PACKED_ROWS4_MATMUL").is_err());
        assert!(env::var("CAMELID_X86_Q8_OUTPUT_AMX_PREFILL").is_err());
        assert!(env::var("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_CONSUMER").is_err());
        assert!(env::var("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_GROUP_CHUNKING").is_err());
        assert!(env::var("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_GROUPS_PER_CHUNK").is_err());
        assert!(env::var("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_FUSED_ACTIVATION").is_err());
        assert!(env::var("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_PAIRED_DOT").is_err());
        assert!(env::var("CAMELID_X86_Q8_FFN_DECODE_CHAIN").is_err());
        assert!(env::var("CAMELID_X86_Q8_FFN_GATE_UP_PACKED_ROWS4_MATMUL").is_err());
        assert!(env::var("CAMELID_X86_Q8_FFN_GATE_UP_SINGLE_OWNER").is_err());
        assert!(env::var("CAMELID_X86_Q8_FFN_DOWN_DECODE_CONSUMER").is_err());
        assert!(env::var("CAMELID_X86_Q8_FFN_DOWN_PACKED_ROWS4_MATMUL").is_err());
        assert!(env::var("CAMELID_X86_Q8_PACKED_ROWS4_MATMUL").is_err());
        assert!(env::var("CAMELID_X86_Q8_FFN_DOWN_GEMM4_PREFILL").is_err());
        assert!(env::var("CAMELID_X86_Q8_FFN_DOWN_GEMM4_ROW_GROUP_SCHED").is_err());
        assert!(env::var("CAMELID_X86_Q8_FFN_DOWN_GEMM4_ROW_GROUP_MIN_INPUT_GROUPS").is_err());
        assert!(env::var("CAMELID_X86_Q8_PACKED_ROWS4_MATMUL_GROUPS_PER_CHUNK").is_err());
        assert!(env::var("CAMELID_X86_Q8_FFN_DOWN_GEMM4_AVX2").is_err());
        assert!(env::var("CAMELID_X86_Q8_FFN_DOWN_AMX_PREFILL").is_err());
        assert!(env::var("CAMELID_X86_Q8_FFN_DOWN_SINGLE_OWNER").is_err());
        assert!(env::var("CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE").is_err());
        assert!(env::var("CAMELID_X86_Q8_FFN_DOWN_VNNI_DECODE_RAWPTR").is_err());
        assert!(env::var("CAMELID_X86_Q8_OUTPUT_DECODE_OWNER").is_err());
        assert!(env::var("CAMELID_X86_Q8_PACKED_ROWS4_SERIAL_DECODE").is_err());
        assert!(env::var("CAMELID_X86_Q8_PARALLEL_INPUT_QUANTIZE").is_err());
        clear_profile_env();
    }

    #[test]
    fn planner_env_apply_restores_owned_x86_q8_passthrough_knobs() {
        let _guard = env_lock();
        clear_profile_env();
        env::set_var("CAMELID_X86_Q8_ATTENTION_QKV_DECODE_GROUPS_PER_CHUNK", "7");
        env::set_var("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_GROUPS_PER_CHUNK", "5");
        env::set_var(
            "CAMELID_X86_Q8_FFN_DOWN_GEMM4_ROW_GROUP_MIN_INPUT_GROUPS",
            "3",
        );
        env::set_var("CAMELID_X86_Q8_PACKED_ROWS4_MATMUL_GROUPS_PER_CHUNK", "9");
        let planner_env = PlannerEnv::capture();

        env::set_var("CAMELID_X86_Q8_ATTENTION_QKV_DECODE_GROUPS_PER_CHUNK", "99");
        env::set_var("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_GROUPS_PER_CHUNK", "99");
        env::set_var(
            "CAMELID_X86_Q8_FFN_DOWN_GEMM4_ROW_GROUP_MIN_INPUT_GROUPS",
            "99",
        );
        env::set_var("CAMELID_X86_Q8_PACKED_ROWS4_MATMUL_GROUPS_PER_CHUNK", "99");

        let updates = BTreeMap::from([
            (
                "CAMELID_X86_Q8_ATTENTION_QKV_DECODE_GROUP_CHUNKING",
                Some("on"),
            ),
            (
                "CAMELID_X86_Q8_FFN_GATE_UP_DECODE_GROUP_CHUNKING",
                Some("on"),
            ),
            ("CAMELID_X86_Q8_FFN_DOWN_GEMM4_ROW_GROUP_SCHED", Some("on")),
            ("CAMELID_X86_Q8_PACKED_ROWS4_MATMUL", Some("on")),
        ]);
        planner_env.apply(&updates);

        assert_eq!(
            env::var("CAMELID_X86_Q8_ATTENTION_QKV_DECODE_GROUPS_PER_CHUNK").ok(),
            Some("7".into())
        );
        assert_eq!(
            env::var("CAMELID_X86_Q8_FFN_GATE_UP_DECODE_GROUPS_PER_CHUNK").ok(),
            Some("5".into())
        );
        assert_eq!(
            env::var("CAMELID_X86_Q8_FFN_DOWN_GEMM4_ROW_GROUP_MIN_INPUT_GROUPS").ok(),
            Some("3".into())
        );
        assert_eq!(
            env::var("CAMELID_X86_Q8_PACKED_ROWS4_MATMUL_GROUPS_PER_CHUNK").ok(),
            Some("9".into())
        );
        clear_profile_env();
    }

    #[test]
    fn planner_env_apply_does_not_restore_packed_rows4_matmul_chunk_groups_without_owner_gate() {
        let _guard = env_lock();
        clear_profile_env();
        env::set_var("CAMELID_X86_Q8_PACKED_ROWS4_MATMUL_GROUPS_PER_CHUNK", "9");
        let planner_env = PlannerEnv::capture();

        env::set_var("CAMELID_X86_Q8_PACKED_ROWS4_MATMUL_GROUPS_PER_CHUNK", "99");

        planner_env.apply(&BTreeMap::new());

        assert!(env::var("CAMELID_X86_Q8_PACKED_ROWS4_MATMUL_GROUPS_PER_CHUNK").is_err());
        clear_profile_env();
    }

    #[test]
    fn ubuntu_experimental_disabled_repack_fails_closed() {
        let _guard = env_lock();
        clear_profile_env();
        env::set_var("CAMELID_PROFILE", "experimental");
        env::set_var("CAMELID_X86_Q8_REPACK", "off");
        env::set_var("CAMELID_X86_Q8_KERNEL", "avx2");
        let outcome = plan_for_model_with_platform(
            &PathBuf::from("/tmp/Llama-3.2-3B-Instruct-Q8_0.gguf"),
            &fixture("Llama 3.2 3B Instruct"),
            Some(16),
            platform("linux", "x86_64", &["avx2"]),
        );
        assert_eq!(outcome.plan.selected_backend, "cpu_reference");
        assert_eq!(outcome.plan.selected_q8_path, "safe_q8_0_block_dot");
        assert_eq!(
            outcome.env_updates.get("CAMELID_X86_Q8_REPACK"),
            Some(&Some("off"))
        );
        clear_profile_env();
    }

    #[test]
    fn ubuntu_experimental_without_avx2_feature_fails_closed() {
        let _guard = env_lock();
        clear_profile_env();
        env::set_var("CAMELID_PROFILE", "experimental");
        env::set_var("CAMELID_X86_Q8_REPACK", "on");
        env::set_var("CAMELID_X86_Q8_KERNEL", "avx2");
        let outcome = plan_for_model_with_platform(
            &PathBuf::from("/tmp/Llama-3.2-3B-Instruct-Q8_0.gguf"),
            &fixture("Llama 3.2 3B Instruct"),
            Some(16),
            platform("linux", "x86_64", &["sse4_2"]),
        );
        assert_eq!(outcome.plan.selected_backend, "cpu_reference");
        assert_eq!(outcome.plan.selected_q8_path, "safe_q8_0_block_dot");
        clear_profile_env();
    }

    #[test]
    fn debug_profile_enables_diagnostics_without_changing_claims() {
        let _guard = env_lock();
        clear_profile_env();
        env::set_var("CAMELID_PROFILE", "debug");
        let outcome = plan_for_model_with_platform(
            &PathBuf::from("/tmp/Llama-3.2-3B-Instruct-Q8_0.gguf"),
            &fixture("Llama 3.2 3B Instruct"),
            Some(4),
            platform("macos", "aarch64", &["dotprod", "i8mm"]),
        );
        assert!(outcome
            .plan
            .diagnostics_status
            .contains("debug diagnostics"));
        assert_eq!(
            outcome.env_updates.get("CAMELID_FORWARD_RSS_TIMINGS"),
            Some(&Some("on"))
        );
        clear_profile_env();
    }

    #[test]
    fn explicit_disable_override_falls_back_to_safe() {
        let _guard = env_lock();
        clear_profile_env();
        env::set_var("CAMELID_MAC_Q8_REPACK", "off");
        let outcome = plan_for_model_with_platform(
            &PathBuf::from("/tmp/Llama-3.2-3B-Instruct-Q8_0.gguf"),
            &fixture("Llama 3.2 3B Instruct"),
            None,
            platform("macos", "aarch64", &["dotprod", "i8mm"]),
        );
        assert_eq!(outcome.plan.selected_backend, "cpu_reference");
        assert_eq!(
            outcome.env_updates.get("CAMELID_MAC_Q8_REPACK"),
            Some(&Some("off"))
        );
        clear_profile_env();
    }

    #[test]
    fn invalid_x86_kernel_override_fails_closed() {
        let _guard = env_lock();
        clear_profile_env();
        env::set_var("CAMELID_PROFILE", "experimental");
        env::set_var("CAMELID_X86_Q8_KERNEL", "amx_now_please");
        let outcome = plan_for_model_with_platform(
            &PathBuf::from("/tmp/Llama-3.2-3B-Instruct-Q8_0.gguf"),
            &fixture("Llama 3.2 3B Instruct"),
            None,
            platform("linux", "x86_64", &["avx2", "amx_int8"]),
        );
        assert_eq!(outcome.plan.selected_backend, "cpu_reference");
        assert_eq!(
            outcome.env_updates.get("CAMELID_X86_Q8_KERNEL"),
            Some(&Some("off"))
        );
        clear_profile_env();
    }

    #[test]
    fn unsupported_row_stays_safe() {
        let _guard = env_lock();
        clear_profile_env();
        let outcome = plan_for_model_with_platform(
            &PathBuf::from("/tmp/Qwen2.5-7B-Instruct-Q8_0.gguf"),
            &fixture("Qwen2.5-7B-Instruct-Q8_0.gguf"),
            None,
            platform("linux", "x86_64", &["avx2"]),
        );
        assert_eq!(outcome.plan.support_level, "unknown_or_unvalidated");
        assert_eq!(outcome.plan.selected_backend, "cpu_reference");
        clear_profile_env();
    }

    #[test]
    fn mistral_row_selects_validated_q8_plan() {
        let _guard = env_lock();
        clear_profile_env();
        let outcome = plan_for_model_with_platform(
            &PathBuf::from("/tmp/Mistral-7B-Instruct-v0.3.Q8_0.gguf"),
            &fixture("Mistral-7B-Instruct-v0.3.Q8_0.gguf"),
            None,
            platform("macos", "aarch64", &["dotprod", "i8mm"]),
        );
        assert_eq!(
            outcome.plan.support_level,
            "supported_exact_row_smoke_512_1024_2048_4096_8192"
        );
        assert_eq!(outcome.plan.selected_q8_path, "mac_validated_q8_0_repack");
        clear_profile_env();
    }

    #[test]
    fn qwen3_rows_select_validated_x86_q8_plan() {
        let _guard = env_lock();
        for name in [
            "Qwen3-0.6B-Instruct-Q8_0.gguf",
            "Qwen3-1.7B-Instruct-Q8_0.gguf",
            "Qwen3-4B-Instruct-Q8_0.gguf",
            "Qwen3-8B-Instruct-Q8_0.gguf",
        ] {
            clear_profile_env();
            let outcome = plan_for_model_with_platform(
                &PathBuf::from(format!("/tmp/{name}")),
                &fixture(name),
                None,
                platform("windows", "x86_64", &["avx2"]),
            );
            assert_eq!(
                outcome.plan.support_level, "supported_exact_row_smoke_chatml",
                "row {name} support_level"
            );
            // Supported Qwen3 Q8 rows engage the validated x86_64 runtime-repack/AVX2
            // plan (not the scalar safe path), matching the other supported Q8 rows.
            assert_eq!(
                outcome.plan.selected_backend, "cpu_q8_runtime_repack",
                "row {name} backend"
            );
            assert_eq!(
                outcome.plan.selected_q8_path, "x86_experimental_q8_0_avx2_rust",
                "row {name} q8 path"
            );
            clear_profile_env();
        }
    }
}
