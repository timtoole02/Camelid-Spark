//! DiffusionGemma lane Phase 0.5 gate: lazy wire dequantization parity against
//! llama.cpp's reference dequant (`scripts/dg-dequant-dump.cpp`, built against
//! the pinned checkout) on the SAME blocks of the SAME tracked GGUF.
//!
//! Env-gated: skips unless `CAMELID_DG_QUANT_PARITY_DIR` (a directory holding
//! `manifest.json` + the reference `.bin` dumps, produced by
//! `scripts/dg-quant-parity.sh`) and `CAMELID_DG_GGUF` (the tracked model
//! file) are both set. Run via the script, not by hand, so the manifest and
//! dumps always come from the pinned reference.
//!
//! Tolerance: ZERO (bit-exact). Dequantization is a pure deterministic
//! function of the wire bytes — both sides compute f16→f32 scales and integer
//! unpacking in f32 with a single canonical formula per format — so any
//! nonzero difference is a real defect, not noise. The gate artifact still
//! records max-abs / mean-abs per format so a failure quantifies itself.

use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;

use camelid::gguf::read_metadata;
use camelid::tensor::wire_dequant::LazyWireTensor;
use camelid::wire_mmap::GgufWireMmap;

#[derive(Debug)]
struct ManifestEntry {
    tensor: String,
    type_name: String,
    first_block: usize,
    n_blocks: usize,
    values: usize,
    dump: String,
}

/// Minimal field extraction from the harness's flat one-line JSON objects
/// (string values without escapes, integer numbers) — avoids a dev-dependency
/// for what is a fixed, machine-generated format.
fn json_str(line: &str, key: &str) -> Option<String> {
    let pat = format!("\"{key}\":\"");
    let start = line.find(&pat)? + pat.len();
    let end = line[start..].find('"')? + start;
    Some(line[start..end].to_string())
}

fn json_u64(line: &str, key: &str) -> Option<u64> {
    let pat = format!("\"{key}\":");
    let start = line.find(&pat)? + pat.len();
    let digits: String = line[start..]
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    digits.parse().ok()
}

fn parse_manifest(path: &Path) -> Vec<ManifestEntry> {
    let text = std::fs::read_to_string(path).expect("read manifest.json");
    text.lines()
        .filter(|l| l.trim_start().starts_with('{'))
        .map(|line| ManifestEntry {
            tensor: json_str(line, "tensor").expect("manifest tensor"),
            type_name: json_str(line, "type").expect("manifest type"),
            first_block: json_u64(line, "first_block").expect("manifest first_block") as usize,
            n_blocks: json_u64(line, "n_blocks").expect("manifest n_blocks") as usize,
            values: json_u64(line, "values").expect("manifest values") as usize,
            dump: json_str(line, "dump").expect("manifest dump"),
        })
        .collect()
}

fn read_f32_le(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    assert!(bytes.len().is_multiple_of(4), "dump not f32-aligned");
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[derive(Default)]
struct FormatStats {
    ranges: usize,
    values: usize,
    bit_mismatches: usize,
    max_abs_err: f64,
    sum_abs_err: f64,
}

#[test]
fn dg_lazy_wire_dequant_matches_pinned_llamacpp() {
    let (Ok(dump_dir), Ok(gguf_path)) = (
        std::env::var("CAMELID_DG_QUANT_PARITY_DIR"),
        std::env::var("CAMELID_DG_GGUF"),
    ) else {
        eprintln!(
            "skipping: CAMELID_DG_QUANT_PARITY_DIR / CAMELID_DG_GGUF not set \
             (run scripts/dg-quant-parity.sh)"
        );
        return;
    };
    let dump_dir = Path::new(&dump_dir);
    let entries = parse_manifest(&dump_dir.join("manifest.json"));
    assert!(!entries.is_empty(), "manifest has no entries");

    let gguf = read_metadata(Path::new(&gguf_path)).expect("read tracked GGUF metadata");
    let mmap = GgufWireMmap::map(Path::new(&gguf_path)).expect("map tracked GGUF");

    let mut by_format: BTreeMap<String, FormatStats> = BTreeMap::new();
    let mut failures: Vec<String> = Vec::new();

    for entry in &entries {
        let desc = gguf
            .tensors
            .iter()
            .find(|t| t.name == entry.tensor)
            .unwrap_or_else(|| panic!("tensor {} not in GGUF", entry.tensor));
        let lazy = LazyWireTensor::from_descriptor(&mmap, desc)
            .unwrap_or_else(|e| panic!("bind {}: {e}", entry.tensor));
        let got = lazy
            .dequantize_blocks(entry.first_block, entry.n_blocks)
            .unwrap_or_else(|e| panic!("dequant {}: {e}", entry.tensor));
        let reference = read_f32_le(&dump_dir.join(&entry.dump));
        assert_eq!(
            got.len(),
            entry.values,
            "{}: value count vs manifest",
            entry.tensor
        );
        assert_eq!(
            reference.len(),
            entry.values,
            "{}: reference dump length vs manifest",
            entry.tensor
        );

        let stats = by_format.entry(entry.type_name.clone()).or_default();
        stats.ranges += 1;
        stats.values += got.len();
        let mut first_bad: Option<usize> = None;
        for (i, (&a, &b)) in got.iter().zip(reference.iter()).enumerate() {
            if a.to_bits() != b.to_bits() {
                stats.bit_mismatches += 1;
                if first_bad.is_none() {
                    first_bad = Some(i);
                }
            }
            let err = (a as f64 - b as f64).abs();
            stats.sum_abs_err += err;
            if err > stats.max_abs_err {
                stats.max_abs_err = err;
            }
        }
        if let Some(i) = first_bad {
            failures.push(format!(
                "{} ({}) blocks [{}, {}): first bit mismatch at value {} (camelid {:?} vs llama.cpp {:?})",
                entry.tensor,
                entry.type_name,
                entry.first_block,
                entry.first_block + entry.n_blocks,
                i,
                got[i],
                reference[i],
            ));
        }
    }

    // Gate artifact body. Written before asserting so a failing run still
    // leaves a quantified record.
    let mut report = String::new();
    report.push_str("{\n  \"comparison\": \"camelid LazyWireTensor::dequantize_blocks vs llama.cpp ggml to_float (scripts/dg-dequant-dump.cpp at the pinned commit)\",\n");
    report.push_str(&format!(
        "  \"llamacpp_pinned_commit\": \"{}\",\n",
        std::env::var("CAMELID_DG_PIN_SHA").unwrap_or_else(|_| "UNRECORDED".to_string())
    ));
    report.push_str(&format!(
        "  \"gguf\": \"{}\",\n",
        Path::new(&gguf_path)
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default()
    ));
    report.push_str("  \"tolerance\": \"bit-exact (0.0); dequantization is a deterministic pure function of the wire bytes\",\n");
    report.push_str("  \"per_format\": {\n");
    let mut first = true;
    for (format, stats) in &by_format {
        if !first {
            report.push_str(",\n");
        }
        first = false;
        report.push_str(&format!(
            "    \"{format}\": {{\"ranges\": {}, \"values\": {}, \"bit_mismatches\": {}, \"max_abs_err\": {:e}, \"mean_abs_err\": {:e}}}",
            stats.ranges,
            stats.values,
            stats.bit_mismatches,
            stats.max_abs_err,
            stats.sum_abs_err / (stats.values.max(1) as f64),
        ));
    }
    report.push_str("\n  },\n");
    report.push_str(&format!(
        "  \"pass\": {}\n}}\n",
        if failures.is_empty() { "true" } else { "false" }
    ));

    if let Ok(out_path) = std::env::var("CAMELID_DG_QUANT_PARITY_OUT") {
        let mut f = std::fs::File::create(&out_path).expect("create gate report");
        f.write_all(report.as_bytes()).expect("write gate report");
        eprintln!("gate report written to {out_path}");
    }
    eprintln!("{report}");

    assert!(
        failures.is_empty(),
        "dequant parity failures:\n{}",
        failures.join("\n")
    );
}
