//! BASALT Amendment 3 §2.4 — invariant-lane matrix binding + meta-test.
//!
//! Mechanism (recorded in DECISIONS.md D17 micro-decisions §2.4):
//!   (a) COMPILE-TIME file binding — every file named by an enforced cell of
//!       `qa/invariant_lanes.json` (and the matrix + schema themselves) is
//!       `include_str!`-bound below, so a rename/move breaks the build;
//!   (b) TEST-TIME name binding — the meta-test parses the matrix, validates it
//!       against `qa/invariant_lanes.schema.json` with a small hand validator
//!       (serde_json only, no new deps), and asserts every named test fn
//!       appears in its bound file's text (`fn <name>(`), so a fn rename fails
//!       the meta-test even though it cannot fail the build;
//!   (c) TRIPPING execution — the §2.6 fixtures under `tests/fixtures/gguf/`
//!       are byte-pinned and tripped here (parse-level unknown-type and K%64
//!       refusals; the D-B2 sidecar reject through runnable admission; the §9
//!       platform-gate cfg twin), while cells whose refusal is already tripped
//!       by an existing per-lane test reference that test instead of
//!       duplicating execution (S1: `tests/nvfp4_wire_lane_refusals.rs`).
//!
//! HONEST CELLS ONLY: the meta-test also holds the matrix to its own rules —
//! full population (no empty cells), nonempty na reasons, open cells only for
//! phases that have not landed (with in-source proxies that FAIL this suite
//! the moment P4/P2b land without closing their cells — ratchet rule R3/R4),
//! and live structural anchors for the I-cache-quant na verdicts.

use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// (a) Compile-time file binding. Paths are relative to tests/.
// ---------------------------------------------------------------------------

const MATRIX_JSON: &str = include_str!("../qa/invariant_lanes.json");
const SCHEMA_JSON: &str = include_str!("../qa/invariant_lanes.schema.json");

/// Every repo file the matrix's enforced cells (or na-cell structural anchors)
/// name. A rename or move of any of these breaks the build right here.
const BOUND: &[(&str, &str)] = &[
    ("src/api/mod.rs", include_str!("../src/api/mod.rs")),
    (
        "src/cuda_resident.rs",
        include_str!("../src/cuda_resident.rs"),
    ),
    (
        "src/cuda_resident/tests.rs",
        include_str!("../src/cuda_resident/tests.rs"),
    ),
    (
        "src/gemma4_runtime.rs",
        include_str!("../src/gemma4_runtime.rs"),
    ),
    ("src/gguf/reader.rs", include_str!("../src/gguf/reader.rs")),
    ("src/inference.rs", include_str!("../src/inference.rs")),
    (
        "src/runnable/admit.rs",
        include_str!("../src/runnable/admit.rs"),
    ),
    (
        "src/runnable/dequant.rs",
        include_str!("../src/runnable/dequant.rs"),
    ),
    (
        "src/runnable/smoke.rs",
        include_str!("../src/runnable/smoke.rs"),
    ),
    ("src/tensor/mod.rs", include_str!("../src/tensor/mod.rs")),
    (
        "tests/invariant_matrix_binding.rs",
        include_str!("invariant_matrix_binding.rs"),
    ),
    ("tests/nvfp4_format.rs", include_str!("nvfp4_format.rs")),
    (
        "tests/nvfp4_wire_lane_refusals.rs",
        include_str!("nvfp4_wire_lane_refusals.rs"),
    ),
];

fn bound(path: &str) -> &'static str {
    BOUND
        .iter()
        .find(|(p, _)| *p == path)
        .map(|(_, text)| *text)
        .unwrap_or_else(|| {
            panic!(
                "{path} is named by the invariant matrix but not include_str!-bound in \
                 tests/invariant_matrix_binding.rs — add it to BOUND (Amendment 3 §2.4a)"
            )
        })
}

fn matrix() -> Value {
    serde_json::from_str(MATRIX_JSON).expect("qa/invariant_lanes.json must be valid JSON")
}

fn schema() -> Value {
    serde_json::from_str(SCHEMA_JSON).expect("qa/invariant_lanes.schema.json must be valid JSON")
}

fn str_field<'a>(v: &'a Value, key: &str, ctx: &str) -> &'a str {
    v.get(key)
        .and_then(Value::as_str)
        .unwrap_or_else(|| panic!("{ctx}: missing/non-string field {key:?}"))
}

fn nonempty_str_field<'a>(v: &'a Value, key: &str, ctx: &str) -> &'a str {
    let s = str_field(v, key, ctx);
    assert!(
        !s.trim().is_empty(),
        "{ctx}: field {key:?} must be nonempty"
    );
    s
}

fn schema_enum(schema: &Value, definition: &str) -> BTreeSet<String> {
    schema["definitions"][definition]["enum"]
        .as_array()
        .unwrap_or_else(|| panic!("schema definitions.{definition}.enum missing"))
        .iter()
        .map(|v| v.as_str().expect("enum entries are strings").to_string())
        .collect()
}

/// Fn presence is asserted textually as `fn <name>(` — cfg-twinned tests keep
/// their names in source text on every target, so this check is
/// platform-independent even when the fn itself only compiles on one leg.
fn assert_fn_in_bound_file(file: &str, fn_name: &str, ctx: &str) {
    let needle = format!("fn {fn_name}(");
    assert!(
        bound(file).contains(&needle),
        "{ctx}: `{needle}` not found in bound {file} — the named test was renamed or \
         removed; fix the matrix cell or restore the test (Amendment 3 §2.4b)"
    );
}

// ---------------------------------------------------------------------------
// (b) Meta-test: schema validation + population + binding assertions.
// ---------------------------------------------------------------------------

const CELL_STATUSES: &[&str] = &["enforced", "na", "open"];

fn validate_cell(lane: &str, col: &str, cell: &Value) {
    let ctx = format!("cells[{lane}][{col}]");
    let obj = cell
        .as_object()
        .unwrap_or_else(|| panic!("{ctx}: cell must be an object"));
    let status = nonempty_str_field(cell, "status", &ctx);
    assert!(
        CELL_STATUSES.contains(&status),
        "{ctx}: status {status:?} not one of {CELL_STATUSES:?}"
    );
    let allowed: &[&str] = match status {
        "enforced" => &[
            "status",
            "test_file",
            "test_fn",
            "source",
            "companion_tests",
            "fixture",
            "fixture_trip",
            "note",
        ],
        "na" => &["status", "reason", "note"],
        _ => &["status", "phase", "note"],
    };
    for key in obj.keys() {
        assert!(
            allowed.contains(&key.as_str()),
            "{ctx}: key {key:?} not allowed for a {status} cell"
        );
    }
    match status {
        "enforced" => {
            nonempty_str_field(cell, "test_file", &ctx);
            nonempty_str_field(cell, "test_fn", &ctx);
            nonempty_str_field(cell, "source", &ctx);
            if let Some(companions) = obj.get("companion_tests") {
                let list = companions
                    .as_array()
                    .unwrap_or_else(|| panic!("{ctx}: companion_tests must be an array"));
                assert!(!list.is_empty(), "{ctx}: companion_tests must not be empty");
                for (i, c) in list.iter().enumerate() {
                    nonempty_str_field(c, "file", &format!("{ctx}.companion_tests[{i}]"));
                    nonempty_str_field(c, "fn", &format!("{ctx}.companion_tests[{i}]"));
                }
            }
            if let Some(fixture) = obj.get("fixture") {
                let fixture = fixture
                    .as_str()
                    .unwrap_or_else(|| panic!("{ctx}: fixture must be a string"));
                if let Some(reason) = fixture.strip_prefix("seam:") {
                    assert!(
                        !reason.trim().is_empty(),
                        "{ctx}: seam fixture entry must carry a nonempty reason"
                    );
                    assert!(
                        obj.get("fixture_trip").is_none(),
                        "{ctx}: seam entries carry no fixture_trip (nothing file-trips)"
                    );
                } else {
                    let trip = nonempty_str_field(cell, "fixture_trip", &ctx);
                    assert!(
                        trip.split_once("::").is_some_and(|(file, fn_name)| {
                            file.ends_with(".rs") && !fn_name.trim().is_empty()
                        }),
                        "{ctx}: fixture_trip {trip:?} must be `<test file>::<fn>`"
                    );
                }
            }
        }
        "na" => {
            nonempty_str_field(cell, "reason", &ctx);
        }
        _ => {
            let phase = nonempty_str_field(cell, "phase", &ctx);
            assert!(
                ["P4", "P2b", "P3-FINDING"].contains(&phase),
                "{ctx}: open phase {phase:?} is not an unlanded phase or the flagged \
                 P3-FINDING — open cells may not park on landed work"
            );
        }
    }
}

#[test]
fn matrix_validates_against_schema_and_is_fully_populated() {
    let matrix = matrix();
    let schema = schema();

    // Version tags: data, schema $id, and schema const must all agree.
    let version = str_field(&matrix, "matrix_version", "matrix");
    assert_eq!(version, "camelid.invariant-lanes/v1");
    assert_eq!(schema["$id"].as_str(), Some(version));
    assert_eq!(
        schema["properties"]["matrix_version"]["const"].as_str(),
        Some(version),
        "schema const and data matrix_version drifted"
    );

    // Top-level shape.
    let top = matrix.as_object().expect("matrix is an object");
    let expected_keys: BTreeSet<&str> = [
        "matrix_version",
        "provenance",
        "columns",
        "lanes",
        "cells",
        "ratchet",
    ]
    .into_iter()
    .collect();
    let got_keys: BTreeSet<&str> = top.keys().map(String::as_str).collect();
    assert_eq!(got_keys, expected_keys, "matrix top-level keys");

    // Provenance block (ledger-convention mirror).
    let prov = &matrix["provenance"];
    assert_eq!(str_field(prov, "campaign", "provenance"), "BASALT");
    assert_eq!(
        str_field(prov, "amendment", "provenance"),
        "Amendment 3 §2 (S2)"
    );
    nonempty_str_field(prov, "source_head", "provenance");
    nonempty_str_field(prov, "note", "provenance");

    // Columns and lanes must match the schema's enums EXACTLY (no drift in
    // either direction: an id added to only one file fails here).
    let schema_lanes = schema_enum(&schema, "laneId");
    let schema_columns = schema_enum(&schema, "columnId");
    let columns: Vec<&Value> = matrix["columns"]
        .as_array()
        .expect("columns array")
        .iter()
        .collect();
    let lanes: Vec<&Value> = matrix["lanes"]
        .as_array()
        .expect("lanes array")
        .iter()
        .collect();
    let column_ids: BTreeSet<String> = columns
        .iter()
        .map(|c| nonempty_str_field(c, "id", "column").to_string())
        .collect();
    let lane_ids: BTreeSet<String> = lanes
        .iter()
        .map(|l| nonempty_str_field(l, "id", "lane").to_string())
        .collect();
    assert_eq!(column_ids.len(), columns.len(), "duplicate column ids");
    assert_eq!(lane_ids.len(), lanes.len(), "duplicate lane ids");
    assert_eq!(column_ids, schema_columns, "columns vs schema enum drift");
    assert_eq!(lane_ids, schema_lanes, "lanes vs schema enum drift");

    // Columns are EXTRACTED, never invented: every column cites >=1 source.
    for c in &columns {
        let id = str_field(c, "id", "column");
        nonempty_str_field(c, "invariant", &format!("column {id}"));
        let sources = c["sources"]
            .as_array()
            .unwrap_or_else(|| panic!("column {id}: sources array"));
        assert!(
            !sources.is_empty(),
            "column {id}: must cite at least one source"
        );
        for s in sources {
            assert!(
                !s.as_str().unwrap_or("").trim().is_empty(),
                "column {id}: empty source entry"
            );
        }
    }

    // Lanes: code homes bound; only the never-scheduled quantizer lane may
    // have no code home.
    for l in &lanes {
        let id = str_field(l, "id", "lane");
        nonempty_str_field(l, "name", &format!("lane {id}"));
        let sources = l["sources"]
            .as_array()
            .unwrap_or_else(|| panic!("lane {id}: sources array"));
        if sources.is_empty() {
            assert_eq!(
                id, "L5-native-quantizer",
                "lane {id}: only the never-scheduled L5 may have no code home"
            );
        }
        for s in sources {
            let s = s.as_str().expect("lane source is a string");
            let path = s.split_whitespace().next().unwrap_or("");
            if path.ends_with(".rs") {
                bound(path); // panics with the §2.4a message if unbound
            }
        }
    }

    // FULL population: every (lane x column) cell present exactly once, every
    // cell exactly one of enforced/na/open with its required fields — no
    // empty cells, anywhere.
    let cells = matrix["cells"].as_object().expect("cells object");
    let cell_lane_ids: BTreeSet<String> = cells.keys().cloned().collect();
    assert_eq!(cell_lane_ids, lane_ids, "cells rows vs lanes drift");
    for (lane, row) in cells {
        let row = row
            .as_object()
            .unwrap_or_else(|| panic!("cells[{lane}] object"));
        let row_cols: BTreeSet<String> = row.keys().cloned().collect();
        assert_eq!(
            row_cols, column_ids,
            "cells[{lane}] columns vs matrix columns drift"
        );
        for (col, cell) in row {
            validate_cell(lane, col, cell);
        }
    }

    // Ratchet rules present as data (§2.5).
    let rules = matrix["ratchet"]["rules"]
        .as_array()
        .expect("ratchet.rules array");
    assert!(rules.len() >= 3, "ratchet must carry at least R1-R3");
    for r in rules {
        let id = nonempty_str_field(r, "id", "ratchet rule");
        assert!(
            id.strip_prefix('R')
                .is_some_and(|n| n.chars().all(|c| c.is_ascii_digit()) && !n.is_empty()),
            "ratchet rule id {id:?} must match R<digits>"
        );
        nonempty_str_field(r, "rule", &format!("ratchet {id}"));
    }
}

#[test]
fn enforced_cells_bind_test_fns_to_source_text() {
    let matrix = matrix();
    let cells = matrix["cells"].as_object().expect("cells object");
    let mut enforced = 0usize;
    for (lane, row) in cells {
        for (col, cell) in row.as_object().expect("row object") {
            if cell["status"].as_str() != Some("enforced") {
                continue;
            }
            enforced += 1;
            let ctx = format!("cells[{lane}][{col}]");
            let test_file = str_field(cell, "test_file", &ctx);
            let test_fn = str_field(cell, "test_fn", &ctx);
            let source = str_field(cell, "source", &ctx);
            // Source and test files must be compile-time bound; the primary
            // test fn's NAME must appear in its bound file text.
            bound(source);
            assert_fn_in_bound_file(test_file, test_fn, &ctx);
            if let Some(companions) = cell.get("companion_tests").and_then(Value::as_array) {
                for c in companions {
                    assert_fn_in_bound_file(
                        str_field(c, "file", &ctx),
                        str_field(c, "fn", &ctx),
                        &format!("{ctx} companion"),
                    );
                }
            }
            match cell.get("fixture").and_then(Value::as_str) {
                Some(fixture) if !fixture.starts_with("seam:") => {
                    // The fixture must be committed, and its named trip test
                    // must exist in a bound file (referencing an existing
                    // per-lane trip is fine — no duplicate execution).
                    assert!(
                        fixture_path(fixture).is_file(),
                        "{ctx}: fixture {fixture} missing from tests/fixtures/gguf/"
                    );
                    let trip = str_field(cell, "fixture_trip", &ctx);
                    let (trip_file, trip_fn) = trip.split_once("::").expect("validated shape");
                    assert_fn_in_bound_file(trip_file, trip_fn, &format!("{ctx} fixture_trip"));
                }
                _ => {}
            }
        }
    }
    assert!(enforced > 0, "an all-open matrix would be a bug in itself");
}

#[test]
fn open_cells_reference_only_unlanded_phases() {
    // Ratchet R3/R4 teeth: the phase proxies below are texts that MUST vanish
    // when the phase lands (P4 removes the CUDA-lane refusal; P2b turns the
    // test-anchoring encoder into a product surface). When one vanishes while
    // an open cell still cites the phase, this test fails the merge.
    let matrix = matrix();
    for (lane, row) in matrix["cells"].as_object().expect("cells object") {
        for (col, cell) in row.as_object().expect("row object") {
            if cell["status"].as_str() != Some("open") {
                continue;
            }
            let ctx = format!("cells[{lane}][{col}]");
            match cell["phase"].as_str().expect("validated") {
                "P4" => assert!(
                    bound("src/gemma4_runtime.rs")
                        .contains("NVFP4 CUDA-resident lane is Phase 4 (BASALT)"),
                    "{ctx}: cites open:P4 but the CUDA-lane Phase-4 refusal is gone — \
                     P4 landed; close this cell (ratchet R3)"
                ),
                "P2b" => assert!(
                    bound("src/tensor/mod.rs").contains("TEST-ANCHORING ONLY"),
                    "{ctx}: cites open:P2b but the encoder is no longer test-anchoring-only — \
                     P2b landed; close this cell (ratchet R4)"
                ),
                "P3-FINDING" => {} // orchestrator-surfaced; closed by Tim's verdict
                other => panic!("{ctx}: unexpected phase {other:?}"),
            }
        }
    }
}

#[test]
fn na_cells_carry_live_structural_anchors() {
    // The I-cache-quant na verdicts rest on structural facts; if any anchor
    // disappears from the bound sources, the na cells must be re-investigated
    // rather than silently going stale.
    let api = bound("src/api/mod.rs");
    for anchor in [
        "fn is_runnable_serve_arch",
        "fn resolve_gemma4_runtime_for_model",
        "fn lookup_prompt_prefix_cache",
        "fn store_prompt_prefix_cache",
        "fn clear_prompt_prefix_cache",
        "fn prompt_prefix_cache_reuses_exact_prompt_and_invalidates_key_changes",
        "\"model_not_ready\"",
        "cached.model_path == prepared.model_path",
    ] {
        assert!(
            api.contains(anchor),
            "I-cache-quant na anchor {anchor:?} vanished from src/api/mod.rs — \
             re-investigate the cache-quant cells before trusting the na verdict"
        );
    }
    // The claim "zero prompt-prefix-cache call sites in the gemma4 wire lane"
    // is asserted literally.
    assert!(
        !bound("src/gemma4_runtime.rs").contains("prompt_prefix_cache"),
        "gemma4_runtime.rs now touches the prompt-prefix cache — the L2 \
         I-cache-quant na verdict is void; re-investigate and re-sign the cell"
    );
    // Every L5 na reason must keep citing the D-B5 decision it rests on.
    let matrix = matrix();
    for (col, cell) in matrix["cells"]["L5-native-quantizer"]
        .as_object()
        .expect("L5 row")
    {
        assert_eq!(
            cell["status"].as_str(),
            Some("na"),
            "L5 {col} must stay na until P2b"
        );
        assert!(
            cell["reason"].as_str().unwrap_or("").contains("D-B5"),
            "L5 {col}: na reason must cite D-B5"
        );
    }
}

// ---------------------------------------------------------------------------
// (c) Tripping execution on the committed §2.6 fixtures.
// ---------------------------------------------------------------------------

const UNKNOWN_TYPE_FIXTURE: &str = "nvfp4_unknown_type_trip.gguf";
const UNKNOWN_TYPE_SHA256: &str =
    "69ecd545663630dca0ec856f649e6ee7853232d50e34e305235f83d063d782a2";
const K_DIV_FIXTURE: &str = "nvfp4_k_div_trip.gguf";
const K_DIV_SHA256: &str = "f05c7fbcc3440051c3e727ba514eee963bacc99dcaa7b0a4de213fd143ff8152";
const SIDECAR_ADMIT_FIXTURE: &str = "nvfp4_sidecar_admit_trip.gguf";
const SIDECAR_ADMIT_SHA256: &str =
    "0a4f1593b03cef42da21f3fb2be530f569102fcefd429eff128a38cb92c01ee8";
const PILOT_ADMIT_FIXTURE: &str = "nvfp4_pilot_admit.gguf";
const PILOT_ADMIT_SHA256: &str = "a935284b0ffd1d1f7c31a4b73d40a3f5eaedebd950a1861f387f2ababa760e28";

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("gguf")
        .join(name)
}

fn assert_fixture_pinned(name: &str, want_sha: &str) {
    let bytes = std::fs::read(fixture_path(name)).expect("committed fixture must exist");
    let got = format!("{:x}", Sha256::digest(&bytes));
    assert_eq!(
        got, want_sha,
        "{name} drifted from the generator's pinned bytes; re-run \
         scripts/basalt-nvfp4-golden/gen_sidecar_fixture.mjs and re-pin deliberately"
    );
}

#[test]
fn s2_fixtures_are_byte_pinned_and_listed_in_sha256sums() {
    let sums = std::fs::read_to_string(fixture_path("SHA256SUMS"))
        .expect("tests/fixtures/gguf/SHA256SUMS must exist");
    // All six fixtures (S1 pair + S2 quartet) appear with their pinned hashes;
    // the S1 pair's pins are asserted byte-for-byte by nvfp4_wire_lane_refusals.
    for (name, sha) in [
        (
            "nvfp4_sidecar_trip.gguf",
            "4220f812bfdc4cb7825241963604bf568963fad41e0bcba6a1c6e2b7e92b7d2d",
        ),
        (
            "nvfp4_nan_sentinel_trip.gguf",
            "29dda31ac380982ccfa354a0f63963673cb7defa5221f17338df1361552ab2cc",
        ),
        (UNKNOWN_TYPE_FIXTURE, UNKNOWN_TYPE_SHA256),
        (K_DIV_FIXTURE, K_DIV_SHA256),
        (SIDECAR_ADMIT_FIXTURE, SIDECAR_ADMIT_SHA256),
        (PILOT_ADMIT_FIXTURE, PILOT_ADMIT_SHA256),
    ] {
        assert_fixture_pinned(name, sha);
        let line = format!("{sha}  {name}");
        assert!(
            sums.lines().any(|l| l == line),
            "SHA256SUMS is missing the receipt line {line:?}"
        );
    }
}

/// I-unknown-type, file boundary (shared by every lane whose loads begin at
/// `read_metadata`): a GGML type id that does not exist at the pin (41) must
/// refuse AT PARSE with the named unknown-type message — never a silent
/// fall-through into admission or binding.
#[test]
fn unknown_type_fixture_trips_parse_refusal() {
    assert_fixture_pinned(UNKNOWN_TYPE_FIXTURE, UNKNOWN_TYPE_SHA256);
    let msg = match camelid::gguf::read_metadata(fixture_path(UNKNOWN_TYPE_FIXTURE)) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("type id 41 must refuse at parse (fail closed)"),
    };
    assert!(
        msg.contains("unknown or removed GGML type"),
        "must be the named parse refusal: {msg}"
    );
    assert!(
        msg.contains("Unknown(41)"),
        "must name the offending id: {msg}"
    );
    assert!(
        msg.contains("blk.0.mystery.weight"),
        "must name the offending tensor: {msg}"
    );
}

/// I-k-div, file boundary: an NVFP4 tensor whose first dimension (48) is not
/// divisible by the 64-element superblock must refuse at parse — never a
/// silent pad.
#[test]
fn k_div_fixture_trips_parse_refusal() {
    assert_fixture_pinned(K_DIV_FIXTURE, K_DIV_SHA256);
    let msg = match camelid::gguf::read_metadata(fixture_path(K_DIV_FIXTURE)) {
        Err(e) => e.to_string(),
        Ok(_) => panic!("K%64 != 0 must refuse at parse (never pad silently)"),
    };
    assert!(
        msg.contains("first dimension 48 is not divisible by block size 64"),
        "must be the named divisibility refusal: {msg}"
    );
    assert!(
        msg.contains("blk.0.ffn_down.weight"),
        "must name the offending tensor: {msg}"
    );
}

/// I-sidecar, L1 file boundary: with the tokenizer axis satisfied, runnable
/// admission reaches the quant axis and the D-B2 sidecar reject fires
/// end-to-end from committed file bytes. Platform-independent by construction:
/// the sidecar check precedes the §9 platform gate inside `check_quants`.
#[test]
fn sidecar_admit_fixture_trips_d_b2_at_admission() {
    use camelid::runnable::{admit, AdmissionAxis};
    assert_fixture_pinned(SIDECAR_ADMIT_FIXTURE, SIDECAR_ADMIT_SHA256);
    let gguf = camelid::gguf::read_metadata(fixture_path(SIDECAR_ADMIT_FIXTURE))
        .expect("fixture must PARSE; the refusal is post-parse");
    assert_eq!(gguf.architecture(), Some("gemma4"));
    let reject = admit(&gguf).expect_err("sidecar-bearing NVFP4 must refuse (D-B2)");
    assert_eq!(reject.axis, AdmissionAxis::Quant);
    assert_eq!(reject.offending_value, "NVFP4");
    assert_eq!(
        reject.tensor.as_deref(),
        Some("blk.0.ffn_down.weight.scale")
    );
    assert!(
        reject.message.contains("sidecar") && reject.message.contains("D-B2"),
        "must name the D-B2 sidecar refusal: {}",
        reject.message
    );
}

/// I-plat, L1 file boundary — the §9 cfg twin on one fixture: the BF16-free
/// pilot shape ADMITS on the Windows and macOS legs (positive control: the
/// platform gate does not misfire where NVFP4 is supported, and the D-B3
/// carve-out admits at the file boundary), and refuses with the named TK2
/// message on the remaining (ubuntu/linux) legs. macOS joined the admit set at
/// GABBRO M2 once its CPU decode was proven bit-exact (Gate G-M1).
#[test]
fn pilot_admit_fixture_is_the_platform_gate_twin() {
    assert_fixture_pinned(PILOT_ADMIT_FIXTURE, PILOT_ADMIT_SHA256);
    let gguf = camelid::gguf::read_metadata(fixture_path(PILOT_ADMIT_FIXTURE))
        .expect("fixture must parse on every platform");
    assert_eq!(gguf.architecture(), Some("gemma4"));

    #[cfg(any(target_os = "windows", target_os = "macos"))]
    {
        use camelid::gguf::GgufTensorType;
        use camelid::runnable::{admit, TokenizerFamily};
        let ok = admit(&gguf).expect("gemma4+NVFP4 pilot must admit on Windows/macOS (D-B3)");
        assert_eq!(ok.architecture, "gemma4");
        assert_eq!(ok.tokenizer, TokenizerFamily::Spm);
        assert!(ok.quants.contains(&GgufTensorType::NVFP4));
    }

    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    {
        use camelid::runnable::{admit, AdmissionAxis};
        let reject =
            admit(&gguf).expect_err("NVFP4 must refuse on unvalidated platforms (Amendment 3 §9)");
        assert_eq!(reject.axis, AdmissionAxis::Quant);
        assert_eq!(reject.offending_value, "NVFP4");
        assert_eq!(reject.tensor.as_deref(), Some("blk.0.ffn_down.weight"));
        assert_eq!(
            reject.message,
            "NVFP4 is Windows/macOS-only in this release; see SUPPORT_MATRIX"
        );
    }
}
