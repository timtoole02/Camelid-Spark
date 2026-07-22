//! Lazy, file-backed block dequantization for wire-format GGUF tensors.
//!
//! Reads ONLY the requested block range out of a [`GgufWireMmap`] and decodes
//! it through the same block decoders the eager CPU loader uses, so there is a
//! single source of truth for every dequant formula. Nothing here materializes
//! a whole tensor to f32: callers ask for bounded block ranges (a row, a probe
//! window), which is the same discipline as the Q8_0 file-backed path and what
//! `CAMELID_MAX_CPU_WEIGHT_MATERIALIZATION_BYTES` exists to enforce.
//!
//! Format coverage is intentionally exactly the set present in the tracked
//! DiffusionGemma GGUF (see `docs/recon/DIFFUSIONGEMMA_RECON.md`): F32, Q8_0,
//! Q5_0, Q4_K, Q6_K. Anything else fails closed with a typed error — adding a
//! format here requires its own llama.cpp dequant-parity evidence first.

use std::sync::Arc;

use super::{
    f16_bits_to_f32, Q4KBlock, Q5_0Block, Q6KBlock, Q4_K_BLOCK_BYTES, Q5_0_BLOCK_BYTES,
    Q6_K_BLOCK_BYTES, QK_K_BLOCK_SIZE,
};
use crate::gguf::{GgufTensorDescriptor, GgufTensorType};
use crate::wire_mmap::GgufWireMmap;
use crate::{BackendError, Result};

const Q8_0_BLOCK_BYTES: usize = 34;
const Q8_0_BLOCK_VALUES: usize = 32;
const Q5_0_BLOCK_VALUES: usize = 32;

/// Wire formats with a proven lazy dequant path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LazyWireFormat {
    F32,
    Q8_0,
    Q5_0,
    Q4K,
    Q6K,
}

impl LazyWireFormat {
    pub fn from_tensor_type(tensor_type: GgufTensorType, name: &str) -> Result<Self> {
        match tensor_type {
            GgufTensorType::F32 => Ok(Self::F32),
            GgufTensorType::Q8_0 => Ok(Self::Q8_0),
            GgufTensorType::Q5_0 => Ok(Self::Q5_0),
            GgufTensorType::Q4K => Ok(Self::Q4K),
            GgufTensorType::Q6K => Ok(Self::Q6K),
            other => Err(BackendError::UnsupportedTensorType(format!(
                "tensor {name} is {other:?}; lazy wire dequantization supports F32, Q8_0, Q5_0, \
                 Q4_K, and Q6_K (the formats with committed dequant-parity evidence)"
            ))),
        }
    }

    /// Decoded f32 values per wire block (1 for F32, mirroring ggml's
    /// `blck_size`).
    pub fn values_per_block(self) -> usize {
        match self {
            Self::F32 => 1,
            Self::Q8_0 => Q8_0_BLOCK_VALUES,
            Self::Q5_0 => Q5_0_BLOCK_VALUES,
            Self::Q4K | Self::Q6K => QK_K_BLOCK_SIZE,
        }
    }

    /// Wire bytes per block.
    pub fn bytes_per_block(self) -> usize {
        match self {
            Self::F32 => std::mem::size_of::<f32>(),
            Self::Q8_0 => Q8_0_BLOCK_BYTES,
            Self::Q5_0 => Q5_0_BLOCK_BYTES,
            Self::Q4K => Q4_K_BLOCK_BYTES,
            Self::Q6K => Q6_K_BLOCK_BYTES,
        }
    }
}

/// A quantized tensor read lazily from the memory-mapped GGUF. Holds only the
/// mapping handle and the tensor's validated extent; bytes fault in when a
/// block range is actually dequantized.
pub struct LazyWireTensor {
    mmap: Arc<GgufWireMmap>,
    byte_offset: u64,
    element_count: usize,
    format: LazyWireFormat,
}

impl LazyWireTensor {
    /// Bind a tensor descriptor to its mapped wire bytes. Validates the format,
    /// block alignment, byte length, and that the whole extent lies inside the
    /// mapping, so block reads can be range-checked arithmetic only.
    pub fn from_descriptor(mmap: &Arc<GgufWireMmap>, desc: &GgufTensorDescriptor) -> Result<Self> {
        let format = LazyWireFormat::from_tensor_type(desc.tensor_type, &desc.name)?;
        let element_count = desc.dimensions.iter().product::<u64>() as usize;
        if element_count == 0 {
            return Err(BackendError::InvalidTensorData(format!(
                "tensor {} has zero elements",
                desc.name
            )));
        }
        if !element_count.is_multiple_of(format.values_per_block()) {
            return Err(BackendError::InvalidTensorData(format!(
                "tensor {} element count {element_count} is not aligned to {:?} blocks of {}",
                desc.name,
                format,
                format.values_per_block()
            )));
        }
        let expected_bytes = element_count / format.values_per_block() * format.bytes_per_block();
        if desc.n_bytes as usize != expected_bytes {
            return Err(BackendError::InvalidTensorData(format!(
                "tensor {} {:?} byte size {} != expected {expected_bytes}",
                desc.name, format, desc.n_bytes
            )));
        }
        mmap.bytes(desc.absolute_offset, expected_bytes)?;
        Ok(Self {
            mmap: mmap.clone(),
            byte_offset: desc.absolute_offset,
            element_count,
            format,
        })
    }

    pub fn format(&self) -> LazyWireFormat {
        self.format
    }

    pub fn element_count(&self) -> usize {
        self.element_count
    }

    pub fn num_blocks(&self) -> usize {
        self.element_count / self.format.values_per_block()
    }

    /// Dequantize `n_blocks` wire blocks starting at `first_block`, reading
    /// only that byte range from the mapping. Returns
    /// `n_blocks * values_per_block()` f32 values.
    pub fn dequantize_blocks(&self, first_block: usize, n_blocks: usize) -> Result<Vec<f32>> {
        let total = self.num_blocks();
        let end = first_block.checked_add(n_blocks).ok_or_else(|| {
            BackendError::RuntimeShapeMismatch("wire dequant block range overflow".to_string())
        })?;
        if end > total {
            return Err(BackendError::RuntimeShapeMismatch(format!(
                "wire dequant range [{first_block}, {end}) exceeds {total} blocks"
            )));
        }
        let bpb = self.format.bytes_per_block();
        let vpb = self.format.values_per_block();
        let bytes = self.mmap.bytes(
            self.byte_offset + (first_block * bpb) as u64,
            n_blocks * bpb,
        )?;
        let mut out = vec![0f32; n_blocks * vpb];

        match self.format {
            LazyWireFormat::F32 => {
                for (value, chunk) in out.iter_mut().zip(bytes.chunks_exact(4)) {
                    *value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                }
            }
            LazyWireFormat::Q8_0 => {
                // value = f16(scale) * q, identical to the eager decode_q8_0_blocks path.
                for (block_bytes, out_block) in bytes
                    .chunks_exact(Q8_0_BLOCK_BYTES)
                    .zip(out.chunks_exact_mut(Q8_0_BLOCK_VALUES))
                {
                    let scale =
                        f16_bits_to_f32(u16::from_le_bytes([block_bytes[0], block_bytes[1]]));
                    for (value, &q) in out_block.iter_mut().zip(&block_bytes[2..]) {
                        *value = scale * (q as i8) as f32;
                    }
                }
            }
            LazyWireFormat::Q5_0 => {
                for (block_bytes, out_block) in bytes
                    .chunks_exact(Q5_0_BLOCK_BYTES)
                    .zip(out.chunks_exact_mut(Q5_0_BLOCK_VALUES))
                {
                    let mut raw = [0u8; Q5_0_BLOCK_BYTES];
                    raw.copy_from_slice(block_bytes);
                    let block = Q5_0Block::from_bytes(&raw);
                    let scale = block.scale_f32();
                    for (value, &q) in out_block.iter_mut().zip(block.unpack_values().iter()) {
                        *value = scale * q as f32;
                    }
                }
            }
            LazyWireFormat::Q4K => {
                let mut decoded = [0f32; QK_K_BLOCK_SIZE];
                for (block_bytes, out_block) in bytes
                    .chunks_exact(Q4_K_BLOCK_BYTES)
                    .zip(out.chunks_exact_mut(QK_K_BLOCK_SIZE))
                {
                    let mut raw = [0u8; Q4_K_BLOCK_BYTES];
                    raw.copy_from_slice(block_bytes);
                    Q4KBlock::from_bytes(&raw).dequantize(&mut decoded);
                    out_block.copy_from_slice(&decoded);
                }
            }
            LazyWireFormat::Q6K => {
                let mut decoded = [0f32; QK_K_BLOCK_SIZE];
                for (block_bytes, out_block) in bytes
                    .chunks_exact(Q6_K_BLOCK_BYTES)
                    .zip(out.chunks_exact_mut(QK_K_BLOCK_SIZE))
                {
                    let mut raw = [0u8; Q6_K_BLOCK_BYTES];
                    raw.copy_from_slice(block_bytes);
                    Q6KBlock::from_bytes(&raw).dequantize(&mut decoded);
                    out_block.copy_from_slice(&decoded);
                }
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{decode_q4_k_blocks, decode_q5_0_blocks, decode_q6_k_blocks};
    use std::io::Write;

    /// Deterministic byte pattern (xorshift; no RNG seeding concerns in CI).
    fn pattern_bytes(len: usize, mut state: u32) -> Vec<u8> {
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            out.push((state >> 8) as u8);
        }
        out
    }

    /// Write wire bytes to a temp file and map them, returning a descriptor
    /// rooted at offset 0.
    fn mapped_fixture(
        name: &str,
        tensor_type: GgufTensorType,
        dims: Vec<u64>,
        wire: &[u8],
    ) -> (Arc<GgufWireMmap>, GgufTensorDescriptor) {
        let dir = std::env::temp_dir().join("camelid-wire-dequant-tests");
        std::fs::create_dir_all(&dir).expect("create temp dir");
        let path = dir.join(format!("{name}.bin"));
        let mut file = std::fs::File::create(&path).expect("create fixture");
        file.write_all(wire).expect("write fixture");
        file.sync_all().expect("sync fixture");
        drop(file);
        let mmap = GgufWireMmap::map(&path).expect("map fixture");
        let desc = GgufTensorDescriptor {
            name: name.to_string(),
            dimensions: dims,
            tensor_type,
            relative_offset: 0,
            absolute_offset: 0,
            n_bytes: wire.len() as u64,
        };
        (mmap, desc)
    }

    /// For a patched f16 scale field, force a sane exponent so scales are
    /// finite and non-degenerate.
    fn sanitize_f16_scale(bytes: &mut [u8], block_bytes: usize, scale_offset: usize) {
        for block in bytes.chunks_exact_mut(block_bytes) {
            // exponent bits 10..15 -> clamp to 0x3C00..0x43FF range (1.0..~8);
            // the low byte (mantissa) keeps its pattern value
            block[scale_offset + 1] = 0x3c | (block[scale_offset + 1] & 0x03);
        }
    }

    #[test]
    fn q8_0_lazy_matches_block_decode() {
        let blocks = 7usize;
        let mut wire = pattern_bytes(blocks * 34, 0x1234_5678);
        sanitize_f16_scale(&mut wire, 34, 0);
        let (mmap, desc) = mapped_fixture(
            "q8_0_fixture",
            GgufTensorType::Q8_0,
            vec![32, blocks as u64],
            &wire,
        );
        let lazy = LazyWireTensor::from_descriptor(&mmap, &desc).expect("bind");
        let got = lazy.dequantize_blocks(0, blocks).expect("dequant");
        for (b, chunk) in wire.chunks_exact(34).enumerate() {
            let scale = f16_bits_to_f32(u16::from_le_bytes([chunk[0], chunk[1]]));
            for (i, &q) in chunk[2..].iter().enumerate() {
                let expect = scale * (q as i8) as f32;
                assert_eq!(got[b * 32 + i].to_bits(), expect.to_bits());
            }
        }
        // sub-range read matches the same offsets of the full read
        let sub = lazy.dequantize_blocks(3, 2).expect("sub-range");
        assert_eq!(&sub[..], &got[3 * 32..5 * 32]);
    }

    #[test]
    fn q5_0_lazy_matches_block_decode() {
        let blocks = 9usize;
        let mut wire = pattern_bytes(blocks * Q5_0_BLOCK_BYTES, 0x9e37_79b9);
        sanitize_f16_scale(&mut wire, Q5_0_BLOCK_BYTES, 0);
        let (mmap, desc) = mapped_fixture(
            "q5_0_fixture",
            GgufTensorType::Q5_0,
            vec![32, blocks as u64],
            &wire,
        );
        let lazy = LazyWireTensor::from_descriptor(&mmap, &desc).expect("bind");
        let got = lazy.dequantize_blocks(0, blocks).expect("dequant");
        let reference = decode_q5_0_blocks(&wire).expect("eager blocks");
        for (b, block) in reference.iter().enumerate() {
            let scale = block.scale_f32();
            for (i, &q) in block.unpack_values().iter().enumerate() {
                let expect = scale * q as f32;
                assert_eq!(got[b * 32 + i].to_bits(), expect.to_bits());
            }
        }
    }

    #[test]
    fn q4_k_lazy_matches_block_decode() {
        let blocks = 5usize;
        let mut wire = pattern_bytes(blocks * Q4_K_BLOCK_BYTES, 0x0bad_f00d);
        // q4_K super-block header: d (f16) at 0, dmin (f16) at 2
        sanitize_f16_scale(&mut wire, Q4_K_BLOCK_BYTES, 0);
        sanitize_f16_scale(&mut wire, Q4_K_BLOCK_BYTES, 2);
        let (mmap, desc) = mapped_fixture(
            "q4_k_fixture",
            GgufTensorType::Q4K,
            vec![256, blocks as u64],
            &wire,
        );
        let lazy = LazyWireTensor::from_descriptor(&mmap, &desc).expect("bind");
        let got = lazy.dequantize_blocks(0, blocks).expect("dequant");
        let reference = decode_q4_k_blocks(&wire).expect("eager blocks");
        let mut expect = [0f32; QK_K_BLOCK_SIZE];
        for (b, block) in reference.iter().enumerate() {
            block.dequantize(&mut expect);
            for i in 0..QK_K_BLOCK_SIZE {
                assert_eq!(got[b * QK_K_BLOCK_SIZE + i].to_bits(), expect[i].to_bits());
            }
        }
        let sub = lazy.dequantize_blocks(2, 2).expect("sub-range");
        assert_eq!(&sub[..], &got[2 * QK_K_BLOCK_SIZE..4 * QK_K_BLOCK_SIZE]);
    }

    #[test]
    fn q6_k_lazy_matches_block_decode() {
        let blocks = 4usize;
        let mut wire = pattern_bytes(blocks * Q6_K_BLOCK_BYTES, 0xfeed_beef);
        // q6_K super-block scale d (f16) sits in the last two bytes
        sanitize_f16_scale(&mut wire, Q6_K_BLOCK_BYTES, Q6_K_BLOCK_BYTES - 2);
        let (mmap, desc) = mapped_fixture(
            "q6_k_fixture",
            GgufTensorType::Q6K,
            vec![256, blocks as u64],
            &wire,
        );
        let lazy = LazyWireTensor::from_descriptor(&mmap, &desc).expect("bind");
        let got = lazy.dequantize_blocks(0, blocks).expect("dequant");
        let reference = decode_q6_k_blocks(&wire).expect("eager blocks");
        let mut expect = [0f32; QK_K_BLOCK_SIZE];
        for (b, block) in reference.iter().enumerate() {
            block.dequantize(&mut expect);
            for i in 0..QK_K_BLOCK_SIZE {
                assert_eq!(got[b * QK_K_BLOCK_SIZE + i].to_bits(), expect[i].to_bits());
            }
        }
    }

    #[test]
    fn f32_lazy_roundtrips() {
        let values: Vec<f32> = (0..96).map(|i| (i as f32) * 0.5 - 17.25).collect();
        let wire: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let (mmap, desc) = mapped_fixture("f32_fixture", GgufTensorType::F32, vec![96], &wire);
        let lazy = LazyWireTensor::from_descriptor(&mmap, &desc).expect("bind");
        let got = lazy.dequantize_blocks(16, 32).expect("dequant");
        assert_eq!(&got[..], &values[16..48]);
    }

    #[test]
    fn unsupported_format_fails_closed() {
        let wire = pattern_bytes(2 * 18, 1);
        let (mmap, desc) = mapped_fixture("q4_0_fixture", GgufTensorType::Q4_0, vec![32, 2], &wire);
        let err = LazyWireTensor::from_descriptor(&mmap, &desc)
            .err()
            .expect("Q4_0 must fail closed here (no committed lazy parity evidence)");
        assert!(matches!(err, BackendError::UnsupportedTensorType(_)));
    }

    #[test]
    fn out_of_range_block_read_fails() {
        let wire = pattern_bytes(3 * 34, 42);
        let (mmap, desc) = mapped_fixture(
            "q8_0_range_fixture",
            GgufTensorType::Q8_0,
            vec![32, 3],
            &wire,
        );
        let lazy = LazyWireTensor::from_descriptor(&mmap, &desc).expect("bind");
        assert!(lazy.dequantize_blocks(2, 2).is_err());
        assert!(lazy.dequantize_blocks(0, 3).is_ok());
    }
}
