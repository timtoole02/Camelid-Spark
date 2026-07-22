//! Ghost (layer-streaming) mode: the `.cghost` container format and its reader/writer.
//!
//! Standard GGUF files scatter a transformer block's tensors across the file, which turns a
//! layer-by-layer streaming pass into random reads. A `.cghost` file is a pure re-layout of
//! a GGUF at **source quantization**: every tensor a block needs is contiguous on disk, so
//! streaming one layer is ONE sequential read. v1 deliberately does not requantize â€”
//! identical bytes mean the ghost path can be parity-gated against the resident path.
//!
//! Layout:
//! ```text
//! [magic "CGHOST1\0"][u64 index_offset][pad to 16 KiB]
//! [group 0 payload][pad][group 1 payload][pad]...
//! [index JSON]                                    <- at index_offset, runs to EOF
//! ```
//! Group payload offsets are 16 KiB aligned (Apple Silicon page size) so future no-copy /
//! madvise work operates on whole pages; tensors inside a group are packed back-to-back.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use crate::platform_fs::read_exact_at;

use serde::{Deserialize, Serialize};

use crate::error::{BackendError, Result};
use crate::gguf::GgufTensorType;
use crate::inference::{DecodeLinearBindings, LlamaLayerWeights};
use crate::model::{LlamaFfnTensors, LlamaTensorBinding};
use crate::tensor::{cpu_tensor_from_gguf_bytes, CpuTensor, TensorStore};

pub const CGHOST_MAGIC: &[u8; 8] = b"CGHOST1\0";
pub const CGHOST_ALIGN: u64 = 16 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CghostTensor {
    pub name: String,
    /// Stable role inside the group ("attn_norm", "attn_q", ..., "token_embedding").
    pub role: String,
    pub dtype: GgufTensorType,
    pub dims: Vec<u64>,
    /// Absolute file offset of this tensor's raw GGUF bytes.
    pub offset: u64,
    pub len: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CghostGroup {
    /// "pre" (embedding + rope), "blk.N" (one transformer block), "post" (output norm/proj).
    pub id: String,
    pub tensors: Vec<CghostTensor>,
}

impl CghostGroup {
    /// Contiguous (start, len) span of this group's payload in the file.
    pub fn span(&self) -> (u64, u64) {
        let start = self.tensors.first().map(|t| t.offset).unwrap_or(0);
        let end = self
            .tensors
            .last()
            .map(|t| t.offset + t.len)
            .unwrap_or(start);
        (start, end - start)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CghostIndex {
    pub version: u32,
    pub source_model: String,
    pub block_count: usize,
    pub tied_output: bool,
    pub groups: Vec<CghostGroup>,
}

/// The fixed tensor order inside a "blk.N" group. Decoding is positional-by-role, so the
/// writer and reader must agree on this list.
const LAYER_ROLES: [&str; 9] = [
    "attn_norm",
    "attn_q",
    "attn_k",
    "attn_v",
    "attn_output",
    "ffn_norm",
    "ffn_gate",
    "ffn_up",
    "ffn_down",
];

fn invalid(msg: String) -> BackendError {
    BackendError::InvalidModelMetadata(msg)
}

fn io_err(path: &Path, source: std::io::Error) -> BackendError {
    BackendError::Io {
        path: path.to_path_buf(),
        source,
    }
}

/// Write a `.cghost` re-layout of `store`'s GGUF. Dense models only â€” ghost v1 refuses MoE
/// (the streaming window assumes one fixed-size group per block).
///
/// `layer_range` selects a pipeline shard: only those "blk.N" groups are written, the "pre"
/// group (embedding) only when the range starts at layer 0, and the "post" group
/// (output norm/projection) only when it ends at the last layer â€” so each mesh node hosts
/// just its own half of the payload on local disk. `None` writes the whole model.
pub fn write_cghost(
    store: &TensorStore,
    binding: &LlamaTensorBinding,
    source_model: &str,
    out_path: &Path,
    layer_range: Option<std::ops::Range<usize>>,
) -> Result<CghostIndex> {
    let total_layers = binding.layers.len();
    let range = layer_range.unwrap_or(0..total_layers);
    if range.start >= range.end || range.end > total_layers {
        return Err(invalid(format!(
            "layer range {range:?} is invalid for a {total_layers}-layer model"
        )));
    }
    // Plan the group contents (names + roles) first.
    let mut planned: Vec<(String, Vec<(String, String)>)> = Vec::new();
    if range.start == 0 {
        let mut pre = vec![(
            "token_embedding".to_string(),
            binding.token_embedding.name.clone(),
        )];
        if let Some(rope) = &binding.rope_freqs {
            pre.push(("rope_freqs".to_string(), rope.name.clone()));
        }
        planned.push(("pre".to_string(), pre));
    }
    for (layer_idx, layer) in binding.layers.iter().enumerate() {
        if !range.contains(&layer_idx) {
            continue;
        }
        let (gate, up, down) = match &layer.ffn {
            LlamaFfnTensors::Dense { gate, up, down } => (gate, up, down),
            LlamaFfnTensors::MoE { .. } => {
                return Err(invalid(format!(
                    "layer {layer_idx} is MoE; ghost v1 supports dense models only"
                )))
            }
        };
        let tensors = vec![
            ("attn_norm".to_string(), layer.attention_norm.name.clone()),
            ("attn_q".to_string(), layer.attention_q.name.clone()),
            ("attn_k".to_string(), layer.attention_k.name.clone()),
            ("attn_v".to_string(), layer.attention_v.name.clone()),
            (
                "attn_output".to_string(),
                layer.attention_output.name.clone(),
            ),
            ("ffn_norm".to_string(), layer.ffn_norm.name.clone()),
            ("ffn_gate".to_string(), gate.name.clone()),
            ("ffn_up".to_string(), up.name.clone()),
            ("ffn_down".to_string(), down.name.clone()),
        ];
        planned.push((format!("blk.{layer_idx}"), tensors));
    }
    if range.end == total_layers {
        let mut post = vec![("output_norm".to_string(), binding.output_norm.name.clone())];
        if !binding.output_is_tied_embedding {
            post.push(("output".to_string(), binding.output.name.clone()));
        }
        planned.push(("post".to_string(), post));
    }

    // Stream the payload out group by group, page-aligning each group start.
    let mut file = File::create(out_path).map_err(|e| io_err(out_path, e))?;
    file.write_all(CGHOST_MAGIC)
        .map_err(|e| io_err(out_path, e))?;
    file.write_all(&0u64.to_le_bytes())
        .map_err(|e| io_err(out_path, e))?;
    let mut cursor: u64 = (CGHOST_MAGIC.len() + 8) as u64;
    let mut groups = Vec::with_capacity(planned.len());
    for (id, tensors) in planned {
        let aligned = cursor.next_multiple_of(CGHOST_ALIGN);
        if aligned > cursor {
            file.write_all(&vec![0u8; (aligned - cursor) as usize])
                .map_err(|e| io_err(out_path, e))?;
            cursor = aligned;
        }
        let mut group = CghostGroup {
            id,
            tensors: Vec::with_capacity(tensors.len()),
        };
        for (role, name) in tensors {
            let desc = store.descriptor(&name)?.clone();
            let bytes = store.tensor_bytes(&name)?;
            file.write_all(&bytes).map_err(|e| io_err(out_path, e))?;
            group.tensors.push(CghostTensor {
                name,
                role,
                dtype: desc.tensor_type,
                dims: desc.dimensions.clone(),
                offset: cursor,
                len: bytes.len() as u64,
            });
            cursor += bytes.len() as u64;
        }
        groups.push(group);
    }

    let index = CghostIndex {
        version: 1,
        source_model: source_model.to_string(),
        block_count: binding.layers.len(),
        tied_output: binding.output_is_tied_embedding,
        groups,
    };
    let index_json = serde_json::to_vec(&index)
        .map_err(|e| invalid(format!("failed to serialize .cghost index: {e}")))?;
    let index_offset = cursor;
    file.write_all(&index_json)
        .map_err(|e| io_err(out_path, e))?;
    file.seek(SeekFrom::Start(CGHOST_MAGIC.len() as u64))
        .map_err(|e| io_err(out_path, e))?;
    file.write_all(&index_offset.to_le_bytes())
        .map_err(|e| io_err(out_path, e))?;
    file.sync_all().map_err(|e| io_err(out_path, e))?;
    Ok(index)
}

/// Open `.cghost` reader: parses the index once, then serves page-aligned group reads.
pub struct GhostFile {
    pub index: CghostIndex,
    file: File,
    /// Windows strict-ceiling mode: a second handle to the same file opened with
    /// `FILE_FLAG_NO_BUFFERING`, so streamed group reads bypass the OS page cache and hit
    /// the device — the equivalent of macOS `F_NOCACHE`, which Windows cannot toggle on an
    /// already-open handle. `None` unless `--evict-page-cache` requested it and the open
    /// succeeded; the read path falls back to the buffered `file` handle otherwise. Behind a
    /// `Mutex` because `GhostFile` is shared across the prefetch worker via `Arc` (needs
    /// `Sync`) and the aligned scratch buffer is interior-mutable; reads are single-threaded
    /// per file so the lock is uncontended.
    #[cfg(windows)]
    uncached: Option<std::sync::Mutex<UncachedReader>>,
}

impl GhostFile {
    /// Open with optional strict-ceiling mode: `evict_page_cache` sets `F_NOCACHE` on the
    /// handle (macOS) so streamed reads bypass the page cache entirely. For models that fit
    /// in RAM the cache is a free win (leave this off); for the over-RAM models ghost mode
    /// targets, the cache can only thrash and the OS must not accumulate the file's pages.
    /// (`posix_madvise(DONTNEED)` does not apply here â€” that is for mmap'd ranges, and the
    /// streamer uses positioned reads.)
    /// (Windows: `evict_page_cache` opens a second `FILE_FLAG_NO_BUFFERING` handle so streamed
    /// group reads bypass the page cache and hit the device -- Windows cannot toggle
    /// no-buffering on an already-open handle, and the macOS `F_NOCACHE` fcntl is a no-op here.
    /// This makes cold-disk streaming cost MEASURABLE on a box where the model would otherwise
    /// cache in RAM. Falls back to buffered reads if the unbuffered open fails.)
    #[cfg_attr(not(windows), allow(unused_mut))]
    pub fn open_with_options(path: &Path, evict_page_cache: bool) -> Result<Self> {
        let mut this = Self::open(path)?;
        if evict_page_cache {
            crate::tensor::disable_file_cache_best_effort(&this.file);
            #[cfg(windows)]
            match UncachedReader::open(path, this.max_layer_span()) {
                Ok(reader) => this.uncached = Some(std::sync::Mutex::new(reader)),
                Err(e) => eprintln!(
                    "[ghost] --evict-page-cache: unbuffered (FILE_FLAG_NO_BUFFERING) open failed \
                     ({e}); falling back to buffered reads (page cache active)"
                ),
            }
        }
        Ok(this)
    }

    pub fn open(path: &Path) -> Result<Self> {
        let mut file = File::open(path).map_err(|e| io_err(path, e))?;
        let mut magic = [0u8; 8];
        file.read_exact(&mut magic).map_err(|e| io_err(path, e))?;
        if &magic != CGHOST_MAGIC {
            return Err(invalid(format!(
                "{} is not a .cghost file (bad magic)",
                path.display()
            )));
        }
        let mut off = [0u8; 8];
        file.read_exact(&mut off).map_err(|e| io_err(path, e))?;
        let index_offset = u64::from_le_bytes(off);
        file.seek(SeekFrom::Start(index_offset))
            .map_err(|e| io_err(path, e))?;
        let mut index_json = Vec::new();
        file.read_to_end(&mut index_json)
            .map_err(|e| io_err(path, e))?;
        let index: CghostIndex = serde_json::from_slice(&index_json)
            .map_err(|e| invalid(format!("failed to parse .cghost index: {e}")))?;
        if index.version != 1 {
            return Err(invalid(format!(
                "unsupported .cghost version {}",
                index.version
            )));
        }
        Ok(Self {
            index,
            file,
            #[cfg(windows)]
            uncached: None,
        })
    }

    fn group(&self, id: &str) -> Result<&CghostGroup> {
        self.index
            .groups
            .iter()
            .find(|g| g.id == id)
            .ok_or_else(|| invalid(format!(".cghost file has no \"{id}\" group")))
    }

    /// Read a whole group's payload with ONE sequential read into `buf` (reused across
    /// calls). Returns the group and the span start so tensor slices can be located.
    fn read_group_payload<'a>(
        &'a self,
        id: &str,
        buf: &mut Vec<u8>,
    ) -> Result<(&'a CghostGroup, u64)> {
        let group = self.group(id)?;
        let (start, len) = group.span();
        buf.resize(len as usize, 0);
        // Strict-ceiling mode (Windows): serve the read from the unbuffered handle so it
        // bypasses the page cache. read_into returns false when it declines (a group that is
        // not sector-aligned or whose aligned span would pass EOF -- never for a v1 .cghost's
        // 16 KiB-aligned blk groups), in which case we fall through to the buffered read.
        #[cfg(windows)]
        if let Some(uncached) = &self.uncached {
            let served = uncached
                .lock()
                .unwrap_or_else(|p| p.into_inner())
                .read_into(start, len, &mut buf[..])
                .map_err(|e| io_err(Path::new("<cghost>"), e))?;
            if served {
                return Ok((group, start));
            }
        }
        read_exact_at(&self.file, buf, start).map_err(|e| io_err(Path::new("<cghost>"), e))?;
        Ok((group, start))
    }

    fn decode_group_tensor(
        tensor: &CghostTensor,
        span_start: u64,
        buf: &[u8],
    ) -> Result<CpuTensor> {
        let lo = (tensor.offset - span_start) as usize;
        let hi = lo + tensor.len as usize;
        cpu_tensor_from_gguf_bytes(&tensor.name, tensor.dtype, &tensor.dims, &buf[lo..hi])
    }

    /// Stream one transformer block's weights: one sequential read + decode to the same
    /// in-RAM storage the resident loader produces. Returns `(weights, bytes, read_us,
    /// decode_us)` — the read (positioned I/O, served from disk or page cache) and decode
    /// (dequant to the resident CpuTensor layout) times are split so a WRAITH Phase-0
    /// receipt can tell a disk-bound stall from a decode-bound one. Measurement-only: the
    /// two `Instant`s add no work to the streaming path.
    pub fn read_layer(
        &self,
        layer_idx: usize,
        buf: &mut Vec<u8>,
    ) -> Result<(LlamaLayerWeights, u64, u128, u128)> {
        let (span_len, read_us) = self.read_layer_bytes(layer_idx, buf)?;
        let (weights, _span, decode_us) = self.decode_layer(layer_idx, &buf[..])?;
        Ok((weights, span_len, read_us, decode_us))
    }

    /// The READ stage of `read_layer`, split out for the stage-split pipeline: one positioned
    /// read of the blk group's payload into `buf`. Returns `(span_bytes, read_us)`. Pair with
    /// `decode_layer` on the SAME `layer_idx` and the resulting `buf` to reconstruct exactly
    /// what `read_layer` produces (the two are byte-identical by construction).
    pub fn read_layer_bytes(&self, layer_idx: usize, buf: &mut Vec<u8>) -> Result<(u64, u128)> {
        let id = format!("blk.{layer_idx}");
        let read_started = std::time::Instant::now();
        let (group, _start) = self.read_group_payload(&id, buf)?;
        let read_us = read_started.elapsed().as_micros();
        let (_, span_len) = group.span();
        Ok((span_len, read_us))
    }

    /// The DECODE stage of `read_layer`: dequant the blk group payload already read into `buf`
    /// (by `read_layer_bytes` for the same `layer_idx`) into the resident `LlamaLayerWeights`
    /// layout. Returns `(weights, span_bytes, decode_us)`. Pure over `buf` — no file I/O — so a
    /// reader thread can be filling layer N+1's buffer while this runs on layer N.
    pub fn decode_layer(
        &self,
        layer_idx: usize,
        buf: &[u8],
    ) -> Result<(LlamaLayerWeights, u64, u128)> {
        let id = format!("blk.{layer_idx}");
        let group = self.group(&id)?;
        let (start, span_len) = group.span();
        let decode_started = std::time::Instant::now();
        let mut by_role: Vec<Option<CpuTensor>> = vec![None; LAYER_ROLES.len()];
        for tensor in &group.tensors {
            if let Some(slot) = LAYER_ROLES.iter().position(|r| *r == tensor.role) {
                by_role[slot] = Some(Self::decode_group_tensor(tensor, start, buf)?);
            }
        }
        let decode_us = decode_started.elapsed().as_micros();
        let mut take = |role: &str| -> Result<CpuTensor> {
            let slot = LAYER_ROLES.iter().position(|r| *r == role).unwrap();
            by_role[slot]
                .take()
                .ok_or_else(|| invalid(format!("group {id} is missing role \"{role}\"")))
        };
        Ok((
            LlamaLayerWeights {
                attention_norm: take("attn_norm")?,
                attention_q: take("attn_q")?,
                attention_k: take("attn_k")?,
                attention_v: take("attn_v")?,
                attention_output: take("attn_output")?,
                // The ghost (.cghost layer-streaming) format predates QK-norm and
                // carries no attn_q_norm/attn_k_norm roles, so ghost mode does not
                // support Qwen3-style models. Left None here; a Qwen3 ghost run is
                // not a supported configuration.
                attention_q_norm: None,
                attention_k_norm: None,
                ffn_norm: take("ffn_norm")?,
                ffn_gate: take("ffn_gate")?,
                ffn_up: take("ffn_up")?,
                ffn_down: take("ffn_down")?,
                moe_router: None,
                decode_bindings: DecodeLinearBindings::default(),
            },
            span_len,
            decode_us,
        ))
    }

    /// Largest "blk.N" group payload in bytes â€” the size of the streaming read buffer.
    pub fn max_layer_span(&self) -> u64 {
        self.index
            .groups
            .iter()
            .filter(|g| g.id.starts_with("blk."))
            .map(|g| g.span().1)
            .max()
            .unwrap_or(0)
    }
}

/// Windows unbuffered (`FILE_FLAG_NO_BUFFERING`) reader for the strict-ceiling / cold-disk
/// measurement mode. Windows sets no-buffering at `CreateFile` time (not toggleable on an
/// open handle like macOS `F_NOCACHE`), so this owns a second handle to the same `.cghost`.
/// No-buffering imposes sector alignment on every read: the file offset, the transfer length,
/// and the destination buffer address must all be sector-aligned. `.cghost` blk-group offsets
/// are 16 KiB-aligned by the writer (a superset of the 4 KiB sector requirement), so only the
/// length (rounded up to a sector multiple, over-reading harmlessly into the following group)
/// and the scratch buffer (an aligned sub-slice) need handling here.
#[cfg(windows)]
struct UncachedReader {
    file: File,
    /// Over-allocated so a `SECTOR`-aligned sub-slice of the largest (rounded-up) group span
    /// always fits; sized once at open and never grown (growing would move the base pointer).
    scratch: Vec<u8>,
    file_len: u64,
}

#[cfg(windows)]
impl UncachedReader {
    /// 4 KiB covers both 512e and 4Kn NVMe logical sectors; a 4 KiB-aligned offset/length is
    /// also 512-aligned, so this is a safe universal alignment for `FILE_FLAG_NO_BUFFERING`.
    const SECTOR: usize = 4096;
    const FILE_FLAG_NO_BUFFERING: u32 = 0x2000_0000;

    fn open(path: &Path, max_layer_span: u64) -> std::io::Result<Self> {
        use std::os::windows::fs::OpenOptionsExt;
        let file = std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(Self::FILE_FLAG_NO_BUFFERING)
            .open(path)?;
        let file_len = file.metadata()?.len();
        let cap = (max_layer_span as usize).next_multiple_of(Self::SECTOR) + Self::SECTOR;
        Ok(Self {
            file,
            scratch: vec![0u8; cap],
            file_len,
        })
    }

    /// Fill `out` (already sized to `len`) with `file[start..start+len]` via cache-bypassing
    /// reads. Returns `Ok(true)` when served unbuffered, `Ok(false)` when it declines (offset
    /// not sector-aligned, or the aligned span would pass EOF, or the scratch can't hold it)
    /// so the caller uses the buffered handle instead. `Err` is a real I/O failure.
    fn read_into(&mut self, start: u64, len: u64, out: &mut [u8]) -> std::io::Result<bool> {
        use std::os::windows::fs::FileExt;
        let len = len as usize;
        let aligned_len = len.next_multiple_of(Self::SECTOR);
        // Locate a sector-aligned window inside the over-allocated scratch.
        let base = self.scratch.as_ptr() as usize;
        let pad = (Self::SECTOR - (base % Self::SECTOR)) % Self::SECTOR;
        if !start.is_multiple_of(Self::SECTOR as u64)
            || start + aligned_len as u64 > self.file_len
            || pad + aligned_len > self.scratch.len()
        {
            return Ok(false);
        }
        let region = &mut self.scratch[pad..pad + aligned_len];
        let mut filled = 0usize;
        while filled < aligned_len {
            // NO_BUFFERING transfers whole sectors, so `filled` stays sector-aligned and each
            // subsequent positioned read keeps its aligned-offset contract.
            let n = self
                .file
                .seek_read(&mut region[filled..], start + filled as u64)?;
            if n == 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "unbuffered .cghost read hit EOF before filling the sector-aligned span",
                ));
            }
            filled += n;
        }
        out.copy_from_slice(&self.scratch[pad..pad + len]);
        Ok(true)
    }
}

/// One layer's weights, read + decoded off the critical path by [`GhostPrefetcher`].
pub struct PrefetchedLayer {
    pub layer_idx: usize,
    pub weights: LlamaLayerWeights,
    /// Bytes read from disk for this layer's group.
    pub bytes: u64,
    /// Wall time the worker spent reading + decoding (overlapped with the main thread's
    /// forward, so it only shows up in token latency when it exceeds the compute time).
    pub worker_us: u128,
    /// Split of `worker_us`: positioned-read I/O time (disk or page cache) ...
    pub read_us: u128,
    /// ... and dequant-to-CpuTensor decode time. WRAITH Phase-0 uses the split to attribute
    /// the streaming stall to disk vs decode.
    pub decode_us: u128,
}

/// Double-buffered streaming: a background worker reads + decodes layer N+1 from the
/// `.cghost` file while the main thread runs layer N's forward. The result channel is a
/// rendezvous (capacity 0), so at most TWO layer windows exist at any instant â€” one in the
/// session being computed, one finished in the worker's hand awaiting handoff. The worker
/// fulfills requests strictly in order; the main thread queues each chunk's layer indices
/// ahead of consuming them (and primes the next chunk before the current one finishes, so
/// the disk is already rewinding to layer 0 of token N+1 during the last forwards of
/// token N).
pub struct GhostPrefetcher {
    request_tx: Option<std::sync::mpsc::Sender<usize>>,
    result_rx: Option<std::sync::mpsc::Receiver<Result<PrefetchedLayer>>>,
    worker: Option<std::thread::JoinHandle<()>>,
}

impl GhostPrefetcher {
    pub fn spawn(ghost: std::sync::Arc<GhostFile>) -> Self {
        let (request_tx, request_rx) = std::sync::mpsc::channel::<usize>();
        let (result_tx, result_rx) = std::sync::mpsc::sync_channel::<Result<PrefetchedLayer>>(0);
        let worker = std::thread::Builder::new()
            .name("ghost-prefetch".to_string())
            .spawn(move || {
                let mut buf: Vec<u8> = Vec::with_capacity(ghost.max_layer_span() as usize);
                while let Ok(layer_idx) = request_rx.recv() {
                    let started = std::time::Instant::now();
                    let result = ghost.read_layer(layer_idx, &mut buf).map(
                        |(weights, bytes, read_us, decode_us)| PrefetchedLayer {
                            layer_idx,
                            weights,
                            bytes,
                            worker_us: started.elapsed().as_micros(),
                            read_us,
                            decode_us,
                        },
                    );
                    // The receiver dropping mid-stream (generation ended) is a normal exit.
                    if result_tx.send(result).is_err() {
                        break;
                    }
                }
            })
            .expect("failed to spawn ghost-prefetch thread");
        Self {
            request_tx: Some(request_tx),
            result_rx: Some(result_rx),
            worker: Some(worker),
        }
    }

    /// Queue a layer to be read + decoded. Requests are fulfilled strictly in order.
    pub fn request(&self, layer_idx: usize) -> Result<()> {
        self.request_tx
            .as_ref()
            .expect("prefetcher closed")
            .send(layer_idx)
            .map_err(|_| invalid("ghost-prefetch worker exited unexpectedly".to_string()))
    }

    /// Block until the next requested layer is ready (instant when the worker finished
    /// while the previous layer was computing â€” the double-buffered steady state).
    pub fn next(&self) -> Result<PrefetchedLayer> {
        self.result_rx
            .as_ref()
            .expect("prefetcher closed")
            .recv()
            .map_err(|_| invalid("ghost-prefetch worker exited unexpectedly".to_string()))?
    }
}

impl Drop for GhostPrefetcher {
    fn drop(&mut self) {
        // Close BOTH channel ends before joining: dropping the request sender ends the
        // worker's request loop, and dropping the result receiver releases a worker that
        // is blocked mid-handoff on the rendezvous send (otherwise the join deadlocks).
        drop(self.request_tx.take());
        drop(self.result_rx.take());
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

/// Two-stage streaming prefetcher (WRAITH Phase-2 read‖decode stage-split): a READER thread
/// fills layer N+1's raw buffer while a DECODER thread dequants layer N, so a layer's disk
/// read overlaps the previous layer's CPU dequant. The single-worker [`GhostPrefetcher`] does
/// read-then-decode serially, so its per-layer throughput is `read + decode`; this pipeline's
/// is `max(read, decode)` — the win is largest when read and decode are comparable (the
/// cold-NVMe regime). Same strict in-order handoff to the main thread and the same rendezvous
/// bound on decoded windows; the extra footprint is a small pool of raw byte buffers
/// (`read_ahead + 1` layer spans) that the ceiling accounting must include. Parity-identical
/// to the single-worker path by construction — same bytes read, same decode. Opt-in via
/// ghost-run `--stage-split`.
pub struct GhostPipelinePrefetcher {
    request_tx: Option<std::sync::mpsc::Sender<usize>>,
    result_rx: Option<std::sync::mpsc::Receiver<Result<PrefetchedLayer>>>,
    reader: Option<std::thread::JoinHandle<()>>,
    decoder: Option<std::thread::JoinHandle<()>>,
}

impl GhostPipelinePrefetcher {
    pub fn spawn(ghost: std::sync::Arc<GhostFile>, read_ahead: usize) -> Self {
        use std::sync::mpsc::{channel, sync_channel};
        let read_ahead = read_ahead.max(1);
        let (request_tx, request_rx) = channel::<usize>();
        // reader -> decoder: (layer_idx, filled buffer, Ok(read_us) | Err). Bounded so the
        // reader runs at most `read_ahead` layers ahead of the decoder (backpressure).
        let (stage_tx, stage_rx) = sync_channel::<(usize, Vec<u8>, Result<u128>)>(read_ahead);
        // decoder -> reader: drained buffers returned for reuse (bounds raw-buffer memory).
        let (free_tx, free_rx) = channel::<Vec<u8>>();
        // decoder -> main: rendezvous (capacity 0) — at most two decoded windows exist at once.
        let (result_tx, result_rx) = sync_channel::<Result<PrefetchedLayer>>(0);

        // Prime the raw-buffer pool (read_ahead + 1 spans) before moving free_tx to the decoder.
        let span_cap = ghost.max_layer_span() as usize;
        for _ in 0..(read_ahead + 1) {
            let _ = free_tx.send(Vec::with_capacity(span_cap));
        }

        let reader_ghost = std::sync::Arc::clone(&ghost);
        let reader = std::thread::Builder::new()
            .name("ghost-read".to_string())
            .spawn(move || {
                while let Ok(layer_idx) = request_rx.recv() {
                    // Reuse a pooled buffer; recv only blocks when the decoder is `read_ahead`
                    // layers behind (backpressure), or errors when the decoder has gone away.
                    let mut buf = match free_rx.recv() {
                        Ok(b) => b,
                        Err(_) => break,
                    };
                    let outcome = reader_ghost
                        .read_layer_bytes(layer_idx, &mut buf)
                        .map(|(_span, read_us)| read_us);
                    let read_failed = outcome.is_err();
                    if stage_tx.send((layer_idx, buf, outcome)).is_err() {
                        break;
                    }
                    if read_failed {
                        break; // terminal read error already forwarded to the decoder
                    }
                }
            })
            .expect("failed to spawn ghost-read thread");

        let decoder_ghost = ghost;
        let decoder = std::thread::Builder::new()
            .name("ghost-decode".to_string())
            .spawn(move || {
                while let Ok((layer_idx, buf, read_outcome)) = stage_rx.recv() {
                    let result = match read_outcome {
                        Ok(read_us) => decoder_ghost.decode_layer(layer_idx, &buf).map(
                            |(weights, span, decode_us)| PrefetchedLayer {
                                layer_idx,
                                weights,
                                bytes: span,
                                worker_us: read_us + decode_us,
                                read_us,
                                decode_us,
                            },
                        ),
                        Err(e) => Err(e),
                    };
                    let failed = result.is_err();
                    // Return the buffer to the pool BEFORE the (possibly blocking) rendezvous
                    // handoff, so the reader keeps working ahead while main consumes this layer.
                    let _ = free_tx.send(buf);
                    if result_tx.send(result).is_err() {
                        break;
                    }
                    if failed {
                        break;
                    }
                }
            })
            .expect("failed to spawn ghost-decode thread");

        Self {
            request_tx: Some(request_tx),
            result_rx: Some(result_rx),
            reader: Some(reader),
            decoder: Some(decoder),
        }
    }

    /// Queue a layer for the reader; fulfilled strictly in order (matches [`GhostPrefetcher`]).
    pub fn request(&self, layer_idx: usize) -> Result<()> {
        self.request_tx
            .as_ref()
            .expect("pipeline prefetcher closed")
            .send(layer_idx)
            .map_err(|_| invalid("ghost stage-split reader exited unexpectedly".to_string()))
    }

    /// Block until the next layer is decoded and handed off.
    pub fn next(&self) -> Result<PrefetchedLayer> {
        self.result_rx
            .as_ref()
            .expect("pipeline prefetcher closed")
            .recv()
            .map_err(|_| invalid("ghost stage-split decoder exited unexpectedly".to_string()))?
    }
}

impl Drop for GhostPipelinePrefetcher {
    fn drop(&mut self) {
        // Cascade shutdown: dropping request_tx ends an idle reader; dropping result_rx frees a
        // decoder blocked on the rendezvous send. A decoder exit drops its stage_rx + free_tx,
        // releasing a reader blocked on stage_tx.send (full) or free_rx.recv (empty); a reader
        // exit drops stage_tx, releasing a decoder blocked on stage_rx.recv — so both unwind.
        drop(self.request_tx.take());
        drop(self.result_rx.take());
        if let Some(decoder) = self.decoder.take() {
            let _ = decoder.join();
        }
        if let Some(reader) = self.reader.take() {
            let _ = reader.join();
        }
    }
}
