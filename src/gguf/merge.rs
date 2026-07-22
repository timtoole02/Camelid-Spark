//! Merge llama.cpp-style split GGUF shards into one single GGUF file (FLINT).
//!
//! Big models on Hugging Face ship as N shards named `<stem>-00001-of-0000N.gguf`.
//! Each shard is a COMPLETE GGUF v3 file: its own header, KV section, tensor
//! table, and aligned data section. Shard 1 carries the full model metadata plus
//! three split keys (`split.no` u16, `split.count` u16, `split.tensors.count`
//! i32); later shards carry (at least) the split keys and their tensor slice.
//!
//! Camelid's runtime is single-file end to end (`TensorStore`, the mmap fast
//! path, and the GPU-resident lanes all read `(absolute_offset, n_bytes)` from
//! ONE path), so rather than teach every lane about shards, this module merges
//! shards into the single file the whole engine already understands:
//!
//! * the merged KV section is shard 1's KV bytes copied VERBATIM (a byte-level
//!   span walk — the parsed `BTreeMap` loses key order and empty-array element
//!   types, so re-serializing from it could not be faithful), minus the three
//!   `split.*` keys;
//! * the merged tensor table is every shard's descriptors in shard order with
//!   offsets recomputed densely (the reader at `reader.rs` enforces exact
//!   `align(prev_end)` contiguity, which this recurrence reproduces);
//! * tensor data is stream-copied per shard (each shard's data section is
//!   already densely packed at the same alignment, so a whole-shard copy at an
//!   aligned base preserves every intra-shard offset).
//!
//! The inverse (`split_gguf`) exists as a dev/test harness so the round trip
//! `split(original)` → `merge(shards)` can be proven byte-identical on a real
//! model without any external tooling.

use std::{
    collections::BTreeSet,
    fs::File,
    io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};

use crate::{BackendError, Result};

use super::reader::{read_metadata, GgufFile, GgufMetadataValue};

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
/// The llama.cpp gguf-split metadata keys stripped from (or added to) shards.
pub const SPLIT_KEYS: [&str; 3] = ["split.no", "split.count", "split.tensors.count"];
/// Hard cap on a shard's header+KV region during the byte-level span walk.
/// Real files are a few MB (the 70B shard 1 KV section is ~7.8 MB, dominated by
/// the vocab); 256 MiB is far past anything legitimate and bounds memory.
const MAX_HEADER_BYTES: usize = 256 * 1024 * 1024;

fn invalid(msg: String) -> BackendError {
    BackendError::InvalidGguf(msg)
}

/// One key/value record's byte span inside a shard's raw KV section.
struct KvSpan {
    key: String,
    start: usize,
    end: usize,
}

/// The byte-level view of one shard's header the merger needs: the raw
/// header+KV bytes, the KV spans within them, and the parsed counts.
struct RawHeader {
    /// Every byte from offset 0 through the end of the last KV record.
    bytes: Vec<u8>,
    kv_spans: Vec<KvSpan>,
    kv_count: u64,
}

/// A `Read`er that remembers every byte it has produced, so records can be
/// sliced back out as verbatim spans after a forward-only walk.
struct TeeReader<R: Read> {
    inner: R,
    seen: Vec<u8>,
}

impl<R: Read> TeeReader<R> {
    fn new(inner: R) -> Self {
        Self {
            inner,
            seen: Vec::new(),
        }
    }

    fn pos(&self) -> usize {
        self.seen.len()
    }

    fn take(&mut self, n: usize) -> Result<&[u8]> {
        if self.seen.len() + n > MAX_HEADER_BYTES {
            return Err(invalid(format!(
                "GGUF header/KV region exceeds {MAX_HEADER_BYTES} bytes; refusing"
            )));
        }
        let start = self.seen.len();
        self.seen.resize(start + n, 0);
        self.inner
            .read_exact(&mut self.seen[start..])
            .map_err(|e| invalid(format!("unexpected EOF in GGUF header: {e}")))?;
        Ok(&self.seen[start..])
    }

    fn read_u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn read_u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    fn read_i32(&mut self) -> Result<i32> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn read_string(&mut self) -> Result<String> {
        let len = self.read_u64()?;
        let bytes = self.take(len as usize)?;
        String::from_utf8(bytes.to_vec())
            .map_err(|_| invalid("non-UTF8 string in GGUF header".to_string()))
    }

    /// Skip one KV value of wire type `ty`, consuming its exact byte span.
    fn skip_value(&mut self, ty: i32) -> Result<()> {
        match ty {
            0 | 1 | 7 => {
                self.take(1)?;
            } // u8 / i8 / bool
            2 | 3 => {
                self.take(2)?;
            } // u16 / i16
            4 | 5 | 6 => {
                self.take(4)?;
            } // u32 / i32 / f32
            10 | 11 | 12 => {
                self.take(8)?;
            } // u64 / i64 / f64
            8 => {
                let len = self.read_u64()?;
                self.take(len as usize)?;
            } // string
            9 => {
                let elem_ty = self.read_i32()?;
                if elem_ty == 9 {
                    return Err(invalid("nested metadata arrays".to_string()));
                }
                let count = self.read_u64()?;
                match elem_ty {
                    0 | 1 | 7 => {
                        self.take(count as usize)?;
                    }
                    2 | 3 => {
                        self.take((count as usize) * 2)?;
                    }
                    4 | 5 | 6 => {
                        self.take((count as usize) * 4)?;
                    }
                    10 | 11 | 12 => {
                        self.take((count as usize) * 8)?;
                    }
                    8 => {
                        for _ in 0..count {
                            let len = self.read_u64()?;
                            self.take(len as usize)?;
                        }
                    }
                    other => return Err(invalid(format!("unknown KV array element type {other}"))),
                }
            }
            other => return Err(invalid(format!("unknown KV value type {other}"))),
        }
        Ok(())
    }
}

/// Walk a shard's header at the byte level, recording each KV record's span.
/// This intentionally re-parses independently of `read_metadata` (which keeps
/// only typed values): the merger copies KV VALUE BYTES verbatim, so the walk
/// only needs record boundaries, never value semantics.
fn walk_raw_header(path: &Path) -> Result<RawHeader> {
    let file = File::open(path).map_err(|e| BackendError::Io {
        path: path.to_path_buf(),
        source: e,
    })?;
    let mut r = TeeReader::new(BufReader::new(file));

    let magic = r.take(4)?;
    if magic != GGUF_MAGIC {
        return Err(invalid(format!("{}: not a GGUF file", path.display())));
    }
    let version = r.read_u32()?;
    if version != 3 {
        return Err(BackendError::UnsupportedGguf(format!(
            "{}: split merge supports GGUF v3 only (found v{version})",
            path.display()
        )));
    }
    let _tensor_count = r.read_u64()?;
    let kv_count = r.read_u64()?;

    let mut kv_spans = Vec::with_capacity(kv_count as usize);
    for _ in 0..kv_count {
        let start = r.pos();
        let key = r.read_string()?;
        let ty = r.read_i32()?;
        r.skip_value(ty)?;
        kv_spans.push(KvSpan {
            key,
            start,
            end: r.pos(),
        });
    }

    Ok(RawHeader {
        bytes: r.seen,
        kv_spans,
        kv_count,
    })
}

fn align_up(value: u64, alignment: u64) -> u64 {
    value.div_ceil(alignment) * alignment
}

fn meta_u64(file: &GgufFile, key: &str) -> Option<u64> {
    match file.metadata.get(key) {
        Some(GgufMetadataValue::U8(v)) => Some(u64::from(*v)),
        Some(GgufMetadataValue::U16(v)) => Some(u64::from(*v)),
        Some(GgufMetadataValue::U32(v)) => Some(u64::from(*v)),
        Some(GgufMetadataValue::U64(v)) => Some(*v),
        Some(GgufMetadataValue::I32(v)) if *v >= 0 => Some(*v as u64),
        _ => None,
    }
}

/// An ordered, existence-checked set of shard paths for one split model.
#[derive(Debug, Clone)]
pub struct ShardSet {
    pub paths: Vec<PathBuf>,
    /// `<stem>.gguf` — the natural merged filename next to the shards.
    pub merged_name: String,
}

/// Parse `<stem>-NNNNN-of-MMMMM.gguf` out of a filename.
pub fn parse_shard_name(filename: &str) -> Option<(String, usize, usize)> {
    let rest = filename.strip_suffix(".gguf")?;
    // <stem>-NNNNN-of-MMMMM
    let (head, total) = rest.rsplit_once("-of-")?;
    let (stem, no) = head.rsplit_once('-')?;
    if no.len() != 5 || total.len() != 5 {
        return None;
    }
    let no: usize = no.parse().ok()?;
    let total: usize = total.parse().ok()?;
    if no == 0 || total == 0 || no > total {
        return None;
    }
    Some((stem.to_string(), no, total))
}

/// Given ANY shard of a split set, resolve the full ordered sibling list and
/// verify every part exists on disk.
pub fn discover_shard_set(any_shard: &Path) -> Result<ShardSet> {
    let filename = any_shard
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| invalid(format!("{}: not a file path", any_shard.display())))?;
    let (stem, _no, total) = parse_shard_name(filename).ok_or_else(|| {
        invalid(format!(
            "{filename}: expected a split shard named <stem>-NNNNN-of-MMMMM.gguf"
        ))
    })?;
    let dir = any_shard.parent().unwrap_or_else(|| Path::new("."));
    let mut paths = Vec::with_capacity(total);
    for i in 1..=total {
        let p = dir.join(format!("{stem}-{i:05}-of-{total:05}.gguf"));
        if !p.is_file() {
            return Err(invalid(format!(
                "missing shard {i} of {total}: {}",
                p.display()
            )));
        }
        paths.push(p);
    }
    Ok(ShardSet {
        paths,
        merged_name: format!("{stem}.gguf"),
    })
}

/// What `merge_shards` did, for logs and tests.
#[derive(Debug)]
pub struct MergeReport {
    pub shards: usize,
    pub tensors: u64,
    pub kv_kept: u64,
    pub kv_dropped: u64,
    pub bytes_written: u64,
    pub alignment: u64,
}

/// Merge ordered `shards` into a single GGUF at `out_path` (overwritten).
/// Callers wanting atomicity should pass a temp path and rename on success.
pub fn merge_shards(shards: &[PathBuf], out_path: &Path) -> Result<MergeReport> {
    if shards.len() < 2 {
        return Err(invalid(format!(
            "need at least 2 shards to merge, got {}",
            shards.len()
        )));
    }

    // Parse every shard with the production reader: this validates headers,
    // per-shard offset contiguity, data extents, and computes n_bytes.
    let mut parsed: Vec<GgufFile> = Vec::with_capacity(shards.len());
    for p in shards {
        parsed.push(read_metadata(p)?);
    }

    let alignment = parsed[0].alignment;
    for (i, f) in parsed.iter().enumerate() {
        if f.version != 3 {
            return Err(BackendError::UnsupportedGguf(format!(
                "shard {}: split merge supports GGUF v3 only (found v{})",
                i + 1,
                f.version
            )));
        }
        if f.alignment != alignment {
            return Err(invalid(format!(
                "shard {} alignment {} differs from shard 1 alignment {alignment}",
                i + 1,
                f.alignment
            )));
        }
        // llama.cpp gguf-split stamps every shard; require coherent stamps.
        match meta_u64(f, "split.no") {
            Some(no) if no == i as u64 => {}
            Some(no) => {
                return Err(invalid(format!(
                    "shard {} declares split.no={no}; expected {i}",
                    i + 1
                )))
            }
            None => {
                return Err(invalid(format!(
                    "shard {} carries no split.no key — not a gguf-split shard",
                    i + 1
                )))
            }
        }
        match meta_u64(f, "split.count") {
            Some(c) if c == shards.len() as u64 => {}
            Some(c) => {
                return Err(invalid(format!(
                    "shard {} declares split.count={c}; {} shard files were provided",
                    i + 1,
                    shards.len()
                )))
            }
            None => {
                return Err(invalid(format!(
                    "shard {} carries no split.count key — not a gguf-split shard",
                    i + 1
                )))
            }
        }
    }

    let total_tensors: u64 = parsed.iter().map(|f| f.tensors.len() as u64).sum();
    if let Some(declared) = meta_u64(&parsed[0], "split.tensors.count") {
        if declared != total_tensors {
            return Err(invalid(format!(
                "split.tensors.count={declared} but shards carry {total_tensors} tensors"
            )));
        }
    }

    // Cross-shard duplicate tensor names would silently shadow at load time.
    let mut seen = BTreeSet::new();
    for f in &parsed {
        for t in &f.tensors {
            if !seen.insert(t.name.clone()) {
                return Err(invalid(format!(
                    "tensor {} appears in more than one shard",
                    t.name
                )));
            }
        }
    }

    // Byte-level walk of shard 1 for verbatim KV spans (minus split.* keys).
    let raw = walk_raw_header(&shards[0])?;
    let kept: Vec<&KvSpan> = raw
        .kv_spans
        .iter()
        .filter(|s| !SPLIT_KEYS.contains(&s.key.as_str()))
        .collect();
    let kv_dropped = raw.kv_count - kept.len() as u64;

    // ---- Write the merged file ------------------------------------------
    let out_file = File::create(out_path).map_err(|e| BackendError::Io {
        path: out_path.to_path_buf(),
        source: e,
    })?;
    let mut out = BufWriter::with_capacity(8 * 1024 * 1024, out_file);
    let io_err = |e: std::io::Error| BackendError::Io {
        path: out_path.to_path_buf(),
        source: e,
    };

    // Header.
    out.write_all(GGUF_MAGIC).map_err(io_err)?;
    out.write_all(&3u32.to_le_bytes()).map_err(io_err)?;
    out.write_all(&total_tensors.to_le_bytes())
        .map_err(io_err)?;
    out.write_all(&(kept.len() as u64).to_le_bytes())
        .map_err(io_err)?;
    let mut written: u64 = 4 + 4 + 8 + 8;

    // KV section: shard 1's records, verbatim bytes, split.* dropped.
    for span in &kept {
        out.write_all(&raw.bytes[span.start..span.end])
            .map_err(io_err)?;
        written += (span.end - span.start) as u64;
    }

    // Tensor table: every shard's descriptors in shard order, offsets
    // recomputed with the same align(prev_end) recurrence the reader enforces.
    let mut new_offsets: Vec<Vec<u64>> = Vec::with_capacity(parsed.len());
    let mut running: u64 = 0;
    for f in &parsed {
        let mut offs = Vec::with_capacity(f.tensors.len());
        for t in &f.tensors {
            offs.push(running);
            running = align_up(running + t.n_bytes, alignment);
        }
        new_offsets.push(offs);
    }
    for (f, offs) in parsed.iter().zip(&new_offsets) {
        for (t, &off) in f.tensors.iter().zip(offs) {
            out.write_all(&(t.name.len() as u64).to_le_bytes())
                .map_err(io_err)?;
            out.write_all(t.name.as_bytes()).map_err(io_err)?;
            out.write_all(&(t.dimensions.len() as u32).to_le_bytes())
                .map_err(io_err)?;
            for d in &t.dimensions {
                out.write_all(&(*d as i64).to_le_bytes()).map_err(io_err)?;
            }
            out.write_all(&t.tensor_type.wire_id().to_le_bytes())
                .map_err(io_err)?;
            out.write_all(&off.to_le_bytes()).map_err(io_err)?;
            written += 8 + t.name.len() as u64 + 4 + 8 * t.dimensions.len() as u64 + 4 + 8;
        }
    }

    // Pad to the aligned data start, then stream each shard's data section
    // wholesale. Each shard's section is internally dense at `alignment`, and
    // every shard base lands aligned, so intra-shard offsets are preserved —
    // matching the recomputed table exactly.
    let data_start = align_up(written, alignment);
    write_zeros(&mut out, data_start - written).map_err(io_err)?;
    written = data_start;

    let mut data_written: u64 = 0;
    for (i, f) in parsed.iter().enumerate() {
        debug_assert_eq!(data_written % alignment, 0);
        debug_assert_eq!(new_offsets[i].first().copied().unwrap_or(0), data_written);
        let (Some(first), Some(last)) = (f.tensors.first(), f.tensors.last()) else {
            continue; // tensor-less shard contributes no data
        };
        let start = first.absolute_offset;
        let len = last.absolute_offset + last.n_bytes - start;
        let src = File::open(&f.path).map_err(|e| BackendError::Io {
            path: f.path.clone(),
            source: e,
        })?;
        let mut src = BufReader::with_capacity(8 * 1024 * 1024, src);
        src.seek(SeekFrom::Start(start))
            .map_err(|e| BackendError::Io {
                path: f.path.clone(),
                source: e,
            })?;
        let copied = std::io::copy(&mut (&mut src).take(len), &mut out).map_err(io_err)?;
        if copied != len {
            return Err(invalid(format!(
                "shard {}: expected {len} data bytes, copied {copied}",
                i + 1
            )));
        }
        data_written += len;
        let padded = align_up(data_written, alignment);
        write_zeros(&mut out, padded - data_written).map_err(io_err)?;
        data_written = padded;
    }
    written += data_written;

    out.flush().map_err(io_err)?;
    out.into_inner()
        .map_err(|e| io_err(e.into_error()))?
        .sync_all()
        .map_err(io_err)?;

    // Prove the artifact before reporting success: the production reader must
    // accept the merged file (offset contiguity, extents, counts — all teeth).
    let merged = read_metadata(out_path)?;
    if merged.tensors.len() as u64 != total_tensors {
        return Err(invalid(format!(
            "merged file has {} tensors; expected {total_tensors}",
            merged.tensors.len()
        )));
    }

    Ok(MergeReport {
        shards: shards.len(),
        tensors: total_tensors,
        kv_kept: kept.len() as u64,
        kv_dropped,
        bytes_written: written,
        alignment,
    })
}

fn write_zeros<W: Write>(out: &mut W, n: u64) -> std::io::Result<()> {
    const ZEROS: [u8; 4096] = [0; 4096];
    let mut left = n;
    while left > 0 {
        let chunk = left.min(ZEROS.len() as u64) as usize;
        out.write_all(&ZEROS[..chunk])?;
        left -= chunk as u64;
    }
    Ok(())
}

/// Dev/test harness: split a single GGUF v3 into `n_parts` shards using the
/// llama.cpp gguf-split conventions (`split.no`/`split.count`/
/// `split.tensors.count`, `-NNNNN-of-NNNNN.gguf` names). Exists so the
/// round trip `merge(split(x)) == x` can be proven on a real model locally.
pub fn split_gguf(src: &Path, n_parts: usize, out_dir: &Path) -> Result<Vec<PathBuf>> {
    if n_parts < 2 {
        return Err(invalid(format!("need at least 2 parts, got {n_parts}")));
    }
    let parsed = read_metadata(src)?;
    if parsed.version != 3 {
        return Err(BackendError::UnsupportedGguf(
            "split supports GGUF v3 only".to_string(),
        ));
    }
    // Nesting refusal first: pointing split at an existing shard is a more
    // fundamental mistake than a part-count problem, so it wins the error.
    let raw = walk_raw_header(src)?;
    if raw
        .kv_spans
        .iter()
        .any(|s| SPLIT_KEYS.contains(&s.key.as_str()))
    {
        return Err(invalid(
            "source already carries split.* keys; refusing to nest splits".to_string(),
        ));
    }
    if parsed.tensors.len() < n_parts {
        return Err(invalid(format!(
            "{} tensors cannot fill {n_parts} shards",
            parsed.tensors.len()
        )));
    }

    // Partition tensors into contiguous groups balanced by data bytes.
    let total_bytes: u64 = parsed.tensors.iter().map(|t| t.n_bytes).sum();
    let target = total_bytes.div_ceil(n_parts as u64).max(1);
    let mut groups: Vec<Vec<usize>> = vec![Vec::new()];
    let mut acc = 0u64;
    for (idx, t) in parsed.tensors.iter().enumerate() {
        let remaining_tensors = parsed.tensors.len() - idx;
        let remaining_groups = n_parts - (groups.len() - 1);
        let must_break =
            remaining_tensors == remaining_groups && !groups.last().expect("non-empty").is_empty();
        if groups.len() < n_parts && (acc >= target || must_break) {
            groups.push(Vec::new());
            acc = 0;
        }
        groups.last_mut().expect("non-empty").push(idx);
        acc += t.n_bytes;
    }
    if groups.len() != n_parts || groups.iter().any(|g| g.is_empty()) {
        return Err(invalid(format!(
            "could not partition {} tensors into {n_parts} non-empty shards",
            parsed.tensors.len()
        )));
    }

    let stem = src
        .file_name()
        .and_then(|n| n.to_str())
        .and_then(|n| n.strip_suffix(".gguf"))
        .ok_or_else(|| invalid(format!("{}: expected a .gguf filename", src.display())))?;

    let src_file = File::open(src).map_err(|e| BackendError::Io {
        path: src.to_path_buf(),
        source: e,
    })?;

    let mut out_paths = Vec::with_capacity(n_parts);
    for (i, group) in groups.iter().enumerate() {
        let out_path = out_dir.join(format!("{stem}-{:05}-of-{:05}.gguf", i + 1, n_parts));
        let io_err = |e: std::io::Error| BackendError::Io {
            path: out_path.clone(),
            source: e,
        };
        let file = File::create(&out_path).map_err(io_err)?;
        let mut out = BufWriter::with_capacity(8 * 1024 * 1024, file);

        // Split KVs appended (shard 1) or standalone (later shards).
        let mut split_kvs: Vec<u8> = Vec::new();
        push_kv_u16(&mut split_kvs, "split.no", i as u16);
        push_kv_u16(&mut split_kvs, "split.count", n_parts as u16);
        push_kv_i32(
            &mut split_kvs,
            "split.tensors.count",
            parsed.tensors.len() as i32,
        );

        let kv_count: u64 = if i == 0 { raw.kv_count + 3 } else { 3 };

        out.write_all(GGUF_MAGIC).map_err(io_err)?;
        out.write_all(&3u32.to_le_bytes()).map_err(io_err)?;
        out.write_all(&(group.len() as u64).to_le_bytes())
            .map_err(io_err)?;
        out.write_all(&kv_count.to_le_bytes()).map_err(io_err)?;
        if i == 0 {
            for span in &raw.kv_spans {
                out.write_all(&raw.bytes[span.start..span.end])
                    .map_err(io_err)?;
            }
        }
        out.write_all(&split_kvs).map_err(io_err)?;

        // Table with offsets recomputed from 0 for this shard.
        let mut running = 0u64;
        for &idx in group {
            let t = &parsed.tensors[idx];
            out.write_all(&(t.name.len() as u64).to_le_bytes())
                .map_err(io_err)?;
            out.write_all(t.name.as_bytes()).map_err(io_err)?;
            out.write_all(&(t.dimensions.len() as u32).to_le_bytes())
                .map_err(io_err)?;
            for d in &t.dimensions {
                out.write_all(&(*d as i64).to_le_bytes()).map_err(io_err)?;
            }
            out.write_all(&t.tensor_type.wire_id().to_le_bytes())
                .map_err(io_err)?;
            out.write_all(&running.to_le_bytes()).map_err(io_err)?;
            running = align_up(running + t.n_bytes, parsed.alignment);
        }

        // Data: this group is contiguous in the source, so one streamed copy.
        let table_end = current_len(&out)?;
        let data_start = align_up(table_end, parsed.alignment);
        write_zeros(&mut out, data_start - table_end).map_err(io_err)?;
        let first = &parsed.tensors[group[0]];
        let last = &parsed.tensors[*group.last().expect("non-empty group")];
        let start = first.absolute_offset;
        let len = last.absolute_offset + last.n_bytes - start;
        let mut src_r = BufReader::with_capacity(
            8 * 1024 * 1024,
            src_file.try_clone().map_err(|e| BackendError::Io {
                path: src.to_path_buf(),
                source: e,
            })?,
        );
        src_r
            .seek(SeekFrom::Start(start))
            .map_err(|e| BackendError::Io {
                path: src.to_path_buf(),
                source: e,
            })?;
        let copied = std::io::copy(&mut (&mut src_r).take(len), &mut out).map_err(io_err)?;
        if copied != len {
            return Err(invalid(format!(
                "shard {}: expected {len} data bytes, copied {copied}",
                i + 1
            )));
        }
        out.flush().map_err(io_err)?;
        out.into_inner()
            .map_err(|e| io_err(e.into_error()))?
            .sync_all()
            .map_err(io_err)?;

        // Every emitted shard must itself parse.
        read_metadata(&out_path)?;
        out_paths.push(out_path);
    }
    Ok(out_paths)
}

/// Current logical length of a `BufWriter<File>` being written sequentially.
fn current_len(out: &BufWriter<File>) -> Result<u64> {
    // Sequential writes only: position == bytes written == buffered + flushed.
    let flushed = out
        .get_ref()
        .metadata()
        .map_err(|e| invalid(format!("stat failed: {e}")))?
        .len();
    Ok(flushed + out.buffer().len() as u64)
}

fn push_string(buf: &mut Vec<u8>, s: &str) {
    buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
    buf.extend_from_slice(s.as_bytes());
}

fn push_kv_u16(buf: &mut Vec<u8>, key: &str, value: u16) {
    push_string(buf, key);
    buf.extend_from_slice(&2i32.to_le_bytes()); // wire type 2 = u16
    buf.extend_from_slice(&value.to_le_bytes());
}

fn push_kv_i32(buf: &mut Vec<u8>, key: &str, value: i32) {
    push_string(buf, key);
    buf.extend_from_slice(&5i32.to_le_bytes()); // wire type 5 = i32
    buf.extend_from_slice(&value.to_le_bytes());
}
