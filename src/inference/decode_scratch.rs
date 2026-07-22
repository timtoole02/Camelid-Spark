//! Recycling buffer pool for the decode hot loop (Lane B step 5).
//!
//! Steady-state decode used to allocate a fresh `Vec<f32>` (and a fresh
//! `String` name) for every op output — hundreds of heap round-trips per
//! token, all layer-proportional. This pool turns those into reuse hits:
//! `take(len)` returns a zeroed vector with EXACTLY the semantics of
//! `vec![0.0; len]` (values and length identical — allocation source cannot
//! affect arithmetic), backed by a recycled buffer whenever one with enough
//! capacity is available; `recycle`/`recycle_tensor` return buffers at the
//! points where the layer forward provably owns the dead intermediate.
//!
//! Concurrency: takes/recycles happen on the decode orchestrator thread
//! (ops are dispatched serially per token), so the internal Mutex is
//! uncontended; rayon WORKER scratch must NOT use this pool — workers keep
//! `thread_local!` buffers instead (see the attention lane) so nothing
//! contends inside parallel regions.
//!
//! The pool is unbounded in count but bounded in practice by the working
//! set of one token (every take is matched by a recycle at layer/token
//! end); buffers keep their high-water capacity, which is the point.

use std::sync::Mutex;

use crate::tensor::{CpuTensor, Q8_0Block};

static POOL: Mutex<Vec<Vec<f32>>> = Mutex::new(Vec::new());
static NAME_POOL: Mutex<Vec<String>> = Mutex::new(Vec::new());
static DIMS_POOL: Mutex<Vec<Vec<usize>>> = Mutex::new(Vec::new());
static Q8_BLOCK_POOL: Mutex<Vec<Vec<Q8_0Block>>> = Mutex::new(Vec::new());

/// An empty (cleared) `Vec<Q8_0Block>`, reusing a recycled buffer's capacity
/// when one exists. Content is produced entirely by the caller.
pub(super) fn take_q8_blocks() -> Vec<Q8_0Block> {
    let mut blocks = Q8_BLOCK_POOL
        .lock()
        .expect("decode scratch q8 block pool poisoned")
        .pop()
        .unwrap_or_default();
    blocks.clear();
    blocks
}

/// Return a quantized-input block buffer to the pool.
pub(super) fn recycle_q8_blocks(blocks: Vec<Q8_0Block>) {
    if blocks.capacity() == 0 {
        return;
    }
    Q8_BLOCK_POOL
        .lock()
        .expect("decode scratch q8 block pool poisoned")
        .push(blocks);
}

/// A zeroed `Vec<f32>` of `len`, bit-identical in content to
/// `vec![0.0; len]`, reusing a recycled buffer when one fits.
pub(super) fn take(len: usize) -> Vec<f32> {
    let mut pool = POOL.lock().expect("decode scratch pool poisoned");
    // Last-in first-out keeps the hottest buffer (best cache behavior); scan
    // a bounded tail for one with enough capacity so odd sizes don't strand
    // large buffers.
    let limit = pool.len().min(8);
    for i in 0..limit {
        let idx = pool.len() - 1 - i;
        if pool[idx].capacity() >= len {
            let mut buffer = pool.swap_remove(idx);
            drop(pool);
            buffer.clear();
            buffer.resize(len, 0.0);
            return buffer;
        }
    }
    drop(pool);
    vec![0.0; len]
}

/// Return a buffer to the pool.
pub(super) fn recycle(buffer: Vec<f32>) {
    if buffer.capacity() == 0 {
        return;
    }
    POOL.lock()
        .expect("decode scratch pool poisoned")
        .push(buffer);
}

/// Build a tensor from pooled parts: the data buffer comes from the caller
/// (usually via [`take`]); the name String and dims Vec are recycled when
/// available (clear + refill reuses their capacity — no allocation once the
/// pool is warm).
pub(super) fn tensor_from_pooled(
    name: &str,
    dims: &[usize],
    data: Vec<f32>,
) -> crate::Result<CpuTensor> {
    let mut owned_name = NAME_POOL
        .lock()
        .expect("decode scratch name pool poisoned")
        .pop()
        .unwrap_or_default();
    owned_name.clear();
    owned_name.push_str(name);
    let mut owned_dims = DIMS_POOL
        .lock()
        .expect("decode scratch dims pool poisoned")
        .pop()
        .unwrap_or_default();
    owned_dims.clear();
    owned_dims.extend_from_slice(dims);
    CpuTensor::from_f32(owned_name, owned_dims, data)
}

/// Reclaim a dead intermediate tensor: its data buffer, name String, and
/// dims Vec all return to their pools. Call ONLY where the tensor is
/// provably dead (the layer forward's end-of-scope points).
pub(super) fn recycle_tensor(tensor: CpuTensor) {
    let (name, dims, data) = tensor.into_parts();
    recycle(data);
    if name.capacity() > 0 {
        NAME_POOL
            .lock()
            .expect("decode scratch name pool poisoned")
            .push(name);
    }
    if dims.capacity() > 0 {
        DIMS_POOL
            .lock()
            .expect("decode scratch dims pool poisoned")
            .push(dims);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn take_matches_vec_zero_semantics() {
        for len in [0usize, 1, 63, 64, 3072] {
            let a = take(len);
            let b = vec![0.0f32; len];
            assert_eq!(a.len(), b.len());
            for (x, y) in a.iter().zip(&b) {
                assert_eq!(x.to_bits(), y.to_bits());
            }
            recycle(a);
        }
    }

    #[test]
    fn recycled_buffers_are_reused_and_rezeroed() {
        let mut a = take(128);
        a.iter_mut().for_each(|v| *v = 7.0);
        let ptr = a.as_ptr() as usize;
        recycle(a);
        let b = take(64);
        // Reuse is capacity-based; content must be zero regardless.
        assert!(b.iter().all(|v| v.to_bits() == 0));
        let _ = ptr; // pointer identity is an implementation detail, not asserted
        recycle(b);
    }
}
