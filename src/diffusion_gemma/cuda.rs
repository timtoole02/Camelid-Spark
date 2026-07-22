//! Experimental CUDA acceleration for the DiffusionGemma forward.
//!
//! Two stages run on the GPU behind `--features cuda` (opt-out
//! `CAMELID_DG_CUDA=0`), each falling back to the CPU path on any failure:
//!
//! * `sc_soft_embedding_gpu` — the self-conditioning soft-embedding matmul
//!   (`emb_t @ probs`), the multi-step bottleneck. `emb_t` (~1.47 GB f16) is
//!   held resident; f32 accumulation (non-bit-exact vs the CPU f16 emulation).
//! * `lm_head_q6k_gpu` — the tied Q6_K lm_head GEMV over the canvas rows. The
//!   integer dot is exact (i64) and the per-block f32 term is fused in block
//!   order, mirroring the CPU `q6_k_dot` reduction, so this stage is
//!   bit-close / bit-identical to the CPU oracle.
//! * `expert_rows_gemv_gpu` — MoE expert row-range GEMVs on a VRAM-resident
//!   expert pool (budget-gated; `CAMELID_DG_EXPERT_VRAM_MB`). The Q4_K/Q8_0
//!   kernels mirror `q4_k_dot_scalar` / `q0_pair_dot` exactly → BIT-IDENTICAL
//!   to the CPU path (unit-gated). Besides speed this is a *capacity* lever:
//!   resident expert bytes are never faulted by the CPU-side mmap again.
//!
//! `CAMELID_DG_CUDA_SC=0` keeps just the (non-bit-exact) SC stage on CPU so
//! bit-exact-stage runs can be compared byte-for-byte against the CPU oracle.
//!
//! A single CUDA context/module is shared; each large weight tensor is uploaded
//! once and cached resident (keyed by host pointer/len).
#![cfg(feature = "cuda")]

use std::sync::{Arc, Mutex, OnceLock};

use cudarc::driver::{
    CudaContext, CudaEvent, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

// NVRTC has no cuda_fp16.h here, so f16->f32 is hand-rolled (mirrors the
// resident engine's `f16_bits_to_f32`).
const KERNEL: &str = r#"
// NVRTC has no math.h: spell -inf via the bit pattern.
#define DG_NEG_INF (__int_as_float(0xff800000))

__device__ __forceinline__ float f16_bits_to_f32(unsigned short bits) {
    unsigned int sign = ((unsigned int)(bits & 0x8000u)) << 16;
    unsigned int exp = (bits & 0x7c00u) >> 10;
    unsigned int frac = (unsigned int)(bits & 0x03ffu);
    unsigned int out;
    if (exp == 0u) {
        if (frac == 0u) {
            out = sign;
        } else {
            unsigned int mant = frac; int e = -14;
            while ((mant & 0x0400u) == 0u) { mant <<= 1; e -= 1; }
            mant &= 0x03ffu;
            unsigned int exp32 = (unsigned int)(e + 127);
            out = sign | (exp32 << 23) | (mant << 13);
        }
    } else if (exp == 0x1fu) {
        out = sign | 0x7f800000u | (frac << 13);
    } else {
        unsigned int exp32 = exp + (127u - 15u);
        out = sign | (exp32 << 23) | (frac << 13);
    }
    return __uint_as_float(out);
}

// f32 -> f16 bits, round-to-nearest-even (no cuda_fp16.h under NVRTC).
__device__ __forceinline__ unsigned short f32_to_f16_bits_rne(float f) {
    unsigned int x = __float_as_uint(f);
    unsigned int sign = (x >> 16) & 0x8000u;
    unsigned int e8 = (x >> 23) & 0xffu;
    unsigned int mant = x & 0x7fffffu;
    if (e8 == 0xffu) return (unsigned short)(sign | 0x7c00u | (mant ? 0x200u : 0u));
    int exp = (int)e8 - 127 + 15;
    if (exp >= 0x1f) return (unsigned short)(sign | 0x7c00u);
    if (exp <= 0) {
        if (exp < -10) return (unsigned short)sign;
        mant |= 0x800000u;
        unsigned int shift = (unsigned int)(14 - exp);
        unsigned int half = mant >> shift;
        unsigned int rem = mant & ((1u << shift) - 1u);
        unsigned int halfway = 1u << (shift - 1u);
        if (rem > halfway || (rem == halfway && (half & 1u))) half++;
        return (unsigned short)(sign | half);
    }
    unsigned int half = ((unsigned int)exp << 10) | (mant >> 13);
    unsigned int rem = mant & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (half & 1u))) half++;
    return (unsigned short)(sign | half);
}

// FAST-mode fused SC probabilities: probs[pos] = f16(softmax(logits[pos] *
// temp_inv)) computed straight from the DEVICE-RESIDENT previous-step logits
// (the lm_head GEMM output) — no host softmax, no 134 MB re-upload. One block
// per canvas position.
extern "C" __global__ void sc_probs_f16(
    const float* __restrict__ logits,
    unsigned short* __restrict__ probs,
    int n_vocab, float temp_inv)
{
    int pos = blockIdx.x;
    const float* row = logits + (long long)pos * n_vocab;
    unsigned short* out = probs + (long long)pos * n_vocab;
    __shared__ float red[256];
    float m = DG_NEG_INF;
    for (int v = threadIdx.x; v < n_vocab; v += blockDim.x)
        m = fmaxf(m, row[v] * temp_inv);
    red[threadIdx.x] = m;
    __syncthreads();
    for (int s2 = blockDim.x >> 1; s2 > 0; s2 >>= 1) {
        if (threadIdx.x < s2)
            red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s2]);
        __syncthreads();
    }
    m = red[0];
    __syncthreads();
    float sum = 0.0f;
    for (int v = threadIdx.x; v < n_vocab; v += blockDim.x)
        sum += expf(row[v] * temp_inv - m);
    red[threadIdx.x] = sum;
    __syncthreads();
    for (int s2 = blockDim.x >> 1; s2 > 0; s2 >>= 1) {
        if (threadIdx.x < s2) red[threadIdx.x] += red[threadIdx.x + s2];
        __syncthreads();
    }
    float inv = 1.0f / red[0];
    __syncthreads();
    for (int v = threadIdx.x; v < n_vocab; v += blockDim.x)
        out[v] = f32_to_f16_bits_rne(expf(row[v] * temp_inv - m) * inv);
}

// FAST-mode SC soft-embedding as a tiled f16xf16 GEMM:
// soft[pos][e] = scale * sum_v emb_t[e][v] * probs[pos][v].
// A = emb_t [m=hidden][k=vocab] (row-major, coalesced k-tiles); B = probs
// [n=c][k=vocab] (row-major, loaded coalesced along k and transposed in
// shared). Reads A once per 16-wide n-stripe (~23 GB total) instead of the
// naive per-(pos,e) shape's ~377 GB.
extern "C" __global__ void sc_f16_gemm(
    const unsigned short* __restrict__ a,
    const unsigned short* __restrict__ bt,
    float* __restrict__ cmat,
    int m, int k, int n, float scale)
{
    __shared__ float As[16][17];
    __shared__ float Bs[16][17];
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = blockIdx.y * 16 + ty;
    int col = blockIdx.x * 16 + tx;
    float acc = 0.0f;
    for (int k0 = 0; k0 < k; k0 += 16) {
        As[ty][tx] = (row < m && (k0 + tx) < k)
            ? f16_bits_to_f32(a[(long long)row * k + k0 + tx])
            : 0.0f;
        int bcol = blockIdx.x * 16 + ty;
        Bs[tx][ty] = (bcol < n && (k0 + tx) < k)
            ? f16_bits_to_f32(bt[(long long)bcol * k + k0 + tx])
            : 0.0f;
        __syncthreads();
        for (int kk = 0; kk < 16; kk++) acc += As[ty][kk] * Bs[kk][tx];
        __syncthreads();
    }
    if (row < m && col < n) cmat[(long long)col * m + row] = acc * scale;
}

// Self-conditioning soft-embedding: one block per output (pos, e); the block
// strides over the vocab, accumulates in f32, reduces in shared.
extern "C" __global__ void sc_soft_embedding(
    const unsigned short* __restrict__ emb_t,  // [hidden * n_vocab]
    const unsigned short* __restrict__ probs,  // [c * n_vocab]
    float* __restrict__ soft,                  // [c * hidden]
    int hidden, int n_vocab, int c, float embed_scale)
{
    long out_idx = (long)blockIdx.x;
    int pos = (int)(out_idx / hidden);
    int e = (int)(out_idx % hidden);
    if (pos >= c) return;
    const unsigned short* erow = emb_t + (long)e * (long)n_vocab;
    const unsigned short* prow = probs + (long)pos * (long)n_vocab;
    float acc = 0.0f;
    for (int v = threadIdx.x; v < n_vocab; v += blockDim.x) {
        acc += f16_bits_to_f32(erow[v]) * f16_bits_to_f32(prow[v]);
    }
    __shared__ float sh[256];
    sh[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) soft[(long)pos * (long)hidden + e] = sh[0] * embed_scale;
}

// Q6_K x Q8_K GEMV (lm_head). One thread per output (pos, row): decode the
// row's Q6_K blocks and dot with the position's Q8_K activation, mirroring the
// CPU q6_k_dot reduction (exact i64 integer math; fused per-block f32 term in
// block order). wire = [rows * bpr * 210] u8; act_scales = [c * bpr] f32;
// act_quants = [c * bpr * 256] i8; out = [c * rows] f32 (row-major per pos).
extern "C" __global__ void q6k_gemv_q8k(
    const unsigned char* __restrict__ wire,
    const float* __restrict__ act_scales,
    const signed char* __restrict__ act_quants,
    int rows, int bpr, int c,
    float* __restrict__ out)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long total = (long)c * rows;
    if (idx >= total) return;
    int pos = (int)(idx / rows);
    int row = (int)(idx % rows);
    const unsigned char* rowp = wire + (long)row * bpr * 210;
    const signed char* act_q = act_quants + (long)pos * bpr * 256;
    const float* act_s = act_scales + (long)pos * bpr;
    float sumf = 0.0f;
    for (int b = 0; b < bpr; b++) {
        const unsigned char* block = rowp + (long)b * 210;
        const unsigned char* ql = block;
        const unsigned char* qh = block + 128;
        const signed char* scales = (const signed char*)(block + 192);
        unsigned short d_bits = (unsigned short)block[208]
            | ((unsigned short)block[209] << 8);
        float d_all = f16_bits_to_f32(d_bits);
        const signed char* q8 = act_q + (long)b * 256;
        float y_d = act_s[b];
        long isum = 0;
        for (int half = 0; half < 2; half++) {
            const unsigned char* qlh = ql + half * 64;
            const unsigned char* qhh = qh + half * 32;
            const signed char* q8h = q8 + half * 128;
            const signed char* sc = scales + half * 8;
            long gs0 = 0, gs1 = 0, gs2 = 0, gs3 = 0, gs4 = 0, gs5 = 0, gs6 = 0, gs7 = 0;
            for (int l = 0; l < 32; l++) {
                int v0 = (qlh[l] & 0xF) | ((qhh[l] & 3) << 4);
                int v1 = (qlh[32 + l] & 0xF) | (((qhh[l] >> 2) & 3) << 4);
                int v2 = (qlh[l] >> 4) | (((qhh[l] >> 4) & 3) << 4);
                int v3 = (qlh[32 + l] >> 4) | (((qhh[l] >> 6) & 3) << 4);
                if (l < 16) {
                    gs0 += (long)v0 * q8h[l];
                    gs2 += (long)v1 * q8h[32 + l];
                    gs4 += (long)v2 * q8h[64 + l];
                    gs6 += (long)v3 * q8h[96 + l];
                } else {
                    gs1 += (long)v0 * q8h[l];
                    gs3 += (long)v1 * q8h[32 + l];
                    gs5 += (long)v2 * q8h[64 + l];
                    gs7 += (long)v3 * q8h[96 + l];
                }
            }
            isum += gs0 * (long)sc[0] + gs1 * (long)sc[1] + gs2 * (long)sc[2]
                + gs3 * (long)sc[3] + gs4 * (long)sc[4] + gs5 * (long)sc[5]
                + gs6 * (long)sc[6] + gs7 * (long)sc[7];
        }
        long isum_mins = 0;
        for (int t = 0; t < 16; t++) {
            long bs = 0;
            for (int l = 0; l < 16; l++) bs += q8[t * 16 + l];
            isum_mins += bs * (long)scales[t];
        }
        float dd = d_all * y_d;
        sumf = fmaf(dd, (float)(isum - 32 * isum_mins), sumf);
    }
    out[(long)pos * rows + row] = sumf;
}

// Q4_K scale/min helper: word pair unpacked with the kmask scheme; g in 0..8.
__device__ __forceinline__ long long kq4_byte(unsigned int w0, unsigned int w1, int g) {
    unsigned int w = (g < 4) ? w0 : w1;
    return (long long)((w >> (8 * (g & 3))) & 0xffu);
}

// Q4_K x Q8_K row-range GEMV (MoE expert gate_up rows). One thread per output
// row; mirrors refmath::q4_k_dot_scalar EXACTLY: per superblock, exact 64-bit
// integer mins/main dots, then the two fused f32 terms IN ORDER
// (sumf = fmaf(-dmin, prod, sumf); sumf = fmaf(d, sumi1+sumi2, sumf)).
// wire = the WHOLE resident tensor; rows [first_row, first_row+n_rows) are
// dotted with ONE activation (act_scales [bpr] f32, act_quants [bpr*256] i8).
extern "C" __global__ void q4k_rows_gemv(
    const unsigned char* __restrict__ wire,
    const float* __restrict__ act_scales,
    const signed char* __restrict__ act_quants,
    long long first_row, int n_rows, int bpr,
    float* __restrict__ out)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_rows) return;
    const unsigned char* rowp = wire + (first_row + (long long)r) * (long long)bpr * 144;
    float sumf = 0.0f;
    for (int i = 0; i < bpr; i++) {
        const unsigned char* block = rowp + (long long)i * 144;
        float yd = act_scales[i];
        const signed char* q8 = act_quants + (long long)i * 256;
        float d = yd * f16_bits_to_f32((unsigned short)block[0]
            | ((unsigned short)block[1] << 8));
        float dmin = yd * f16_bits_to_f32((unsigned short)block[2]
            | ((unsigned short)block[3] << 8));
        const unsigned char* sc = block + 4;
        const unsigned char* qs = block + 16;
        unsigned int utmp0 = (unsigned int)sc[0] | ((unsigned int)sc[1] << 8)
            | ((unsigned int)sc[2] << 16) | ((unsigned int)sc[3] << 24);
        unsigned int utmp1 = (unsigned int)sc[4] | ((unsigned int)sc[5] << 8)
            | ((unsigned int)sc[6] << 16) | ((unsigned int)sc[7] << 24);
        unsigned int utmp2 = (unsigned int)sc[8] | ((unsigned int)sc[9] << 8)
            | ((unsigned int)sc[10] << 16) | ((unsigned int)sc[11] << 24);
        unsigned int mins0 = utmp1 & 0x3f3f3f3fu;
        unsigned int mins1 = ((utmp2 >> 4) & 0x0f0f0f0fu)
            | (((utmp1 >> 6) & 0x03030303u) << 4);
        unsigned int scw0 = utmp0 & 0x3f3f3f3fu;
        unsigned int scw1 = (utmp2 & 0x0f0f0f0fu)
            | (((utmp0 >> 6) & 0x03030303u) << 4);
        // mins side: per-32 activation sums x mins (bsum pairs), exact ints.
        long long prod = 0;
        for (int g = 0; g < 8; g++) {
            int bs = 0;
            for (int t = 0; t < 32; t++) bs += q8[g * 32 + t];
            prod += (long long)bs * kq4_byte(mins0, mins1, g);
        }
        sumf = fmaf(-dmin, (float)prod, sumf);
        // main side: 4 chunks; low nibbles dot q8[64j..+32] with scale(2j),
        // high nibbles dot q8[64j+32..+32] with scale(2j+1).
        long long sumi1 = 0, sumi2 = 0;
        for (int j = 0; j < 4; j++) {
            const unsigned char* q4 = qs + j * 32;
            const signed char* q8j = q8 + j * 64;
            long long lo = 0, hi = 0;
            for (int t = 0; t < 32; t++) {
                lo += (long long)(q4[t] & 0xf) * q8j[t];
                hi += (long long)(q4[t] >> 4) * q8j[32 + t];
            }
            sumi1 += lo * kq4_byte(scw0, scw1, 2 * j);
            sumi2 += hi * kq4_byte(scw0, scw1, 2 * j + 1);
        }
        sumf = fmaf(d, (float)(sumi1 + sumi2), sumf);
    }
    out[r] = sumf;
}

// Q8_0 x Q8_0 row-range GEMV (MoE expert down rows). One thread per output
// row; mirrors refmath::q0_pair_dot EXACTLY: 8 accumulators acc[parity][lane],
// per block acc[i%2][l] = fmaf(lane_dot, d_w*y_scale, acc), and the fixed
// pairwise reduction tree at the end. wire = whole resident tensor; the
// activation is act_scales [nb] f32 + act_quants [nb*32] i8.
extern "C" __global__ void q8_0_rows_gemv(
    const unsigned char* __restrict__ wire,
    const float* __restrict__ act_scales,
    const signed char* __restrict__ act_quants,
    long long first_row, int n_rows, int nb,
    float* __restrict__ out)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_rows) return;
    const unsigned char* rowp = wire + (first_row + (long long)r) * (long long)nb * 34;
    float acc0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float acc1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = 0; i < nb; i++) {
        const unsigned char* block = rowp + (long long)i * 34;
        float dw = f16_bits_to_f32((unsigned short)block[0]
            | ((unsigned short)block[1] << 8));
        float s = dw * act_scales[i];
        const signed char* wq = (const signed char*)(block + 2);
        const signed char* yq = act_quants + (long long)i * 32;
        float* acc = (i & 1) ? acc1 : acc0;
        for (int l = 0; l < 4; l++) {
            int lane = 0;
            for (int t = 0; t < 4; t++) {
                lane += (int)wq[4 * l + t] * (int)yq[4 * l + t];
                lane += (int)wq[16 + 4 * l + t] * (int)yq[16 + 4 * l + t];
            }
            acc[l] = fmaf((float)lane, s, acc[l]);
        }
    }
    out[r] = ((acc0[0] + acc0[1]) + (acc0[2] + acc0[3]))
        + ((acc1[0] + acc1[1]) + (acc1[2] + acc1[3]));
}

// ---- FAST-mode GROUPED-TILE GEMM kernels (CAMELID_DG_FAST) -----------------
// Grid: (x = 32-row tiles within a row window, y = pair GROUPS). Each group
// is up to DG_TILE_P pairs sharing ONE pair_base (host-sorted by expert);
// empty slots carry pos = -1. Block = (32 rows, DG_TILE_P pairs). The block
// cooperatively stages a contiguous 32-row x chunk-of-blocks wire tile into
// shared memory with linear coalesced loads (the ONLY coalescing/reuse
// opportunity for uncached device-mapped host memory), then every (row, pair)
// thread runs the EXACT scalar per-block reduction from shared — the per-row
// float order is unchanged, so outputs stay bit-identical to the scalar
// reference per row. Shared row strides are padded to an ODD word count
// (bank-conflict-free). All threads reach every barrier (no early returns).
//
// Byte economics: one shared copy serves up to DG_TILE_P pairs, dividing
// wire traffic by the group occupancy — the difference between ~32x PCIe
// amplification and line rate on the zero-copy tier.
#define DG_TILE_P 8
#define DG_TILE_ROWS 32

// Q4_K: 144B superblocks; stage 5 superblocks/row (720B -> padded 724B).
extern "C" __global__ void q4k_gemm_tile(
    const unsigned char* __restrict__ wire,
    const float* __restrict__ act_scales,       // [n_acts * bpr]
    const signed char* __restrict__ act_quants, // [n_acts * bpr * 256]
    const long long* __restrict__ g_base,       // [n_groups]
    const int* __restrict__ g_pos,              // [n_groups * DG_TILE_P]
    const int* __restrict__ g_orig,             // [n_groups * DG_TILE_P]
    int rows_per, int bpr,
    float* __restrict__ out)
{
    const int SB_STAGE = 5;
    const int ROW_STRIDE = 724; // 5*144 padded to odd word count (181 words)
    extern __shared__ unsigned char tile[]; // [32 * ROW_STRIDE]
    int tx = threadIdx.x; // row lane 0..31
    int ty = threadIdx.y; // pair slot
    int g = blockIdx.y;
    int row = blockIdx.x * DG_TILE_ROWS + tx;
    bool row_ok = row < rows_per;
    int pos = g_pos[g * DG_TILE_P + ty];
    bool pair_ok = pos >= 0;
    long long base = g_base[g];
    const int lin = ty * DG_TILE_ROWS + tx; // 0..255 linear tid
    const int nthreads = DG_TILE_ROWS * DG_TILE_P;
    float sumf = 0.0f;
    for (int sb0 = 0; sb0 < bpr; sb0 += SB_STAGE) {
        int nsb = min(SB_STAGE, bpr - sb0);
        // Cooperative coalesced copy: for each of the 32 rows, copy nsb*144
        // bytes as u32 words (row bytes are 16B-aligned for Q4_K).
        int words_per_row = (nsb * 144) / 4;
        int total_words = DG_TILE_ROWS * words_per_row;
        for (int w = lin; w < total_words; w += nthreads) {
            int r = w / words_per_row;
            int wo = w % words_per_row;
            int gr = blockIdx.x * DG_TILE_ROWS + r;
            unsigned int val = 0u;
            if (gr < rows_per) {
                const unsigned char* src = wire
                    + (base + gr) * (long long)bpr * 144
                    + (long long)sb0 * 144;
                val = *(const unsigned int*)(src + wo * 4);
            }
            *(unsigned int*)(tile + r * ROW_STRIDE + wo * 4) = val;
        }
        __syncthreads();
        if (row_ok && pair_ok) {
            const float* a_s = act_scales + (long long)pos * bpr;
            const signed char* a_q = act_quants + (long long)pos * bpr * 256;
            for (int i = 0; i < nsb; i++) {
                const unsigned char* block = tile + tx * ROW_STRIDE + i * 144;
                int sb = sb0 + i;
                float yd = a_s[sb];
                const signed char* q8 = a_q + (long long)sb * 256;
                float d = yd * f16_bits_to_f32((unsigned short)block[0]
                    | ((unsigned short)block[1] << 8));
                float dmin = yd * f16_bits_to_f32((unsigned short)block[2]
                    | ((unsigned short)block[3] << 8));
                const unsigned char* sc = block + 4;
                const unsigned char* qs = block + 16;
                unsigned int utmp0 = (unsigned int)sc[0] | ((unsigned int)sc[1] << 8)
                    | ((unsigned int)sc[2] << 16) | ((unsigned int)sc[3] << 24);
                unsigned int utmp1 = (unsigned int)sc[4] | ((unsigned int)sc[5] << 8)
                    | ((unsigned int)sc[6] << 16) | ((unsigned int)sc[7] << 24);
                unsigned int utmp2 = (unsigned int)sc[8] | ((unsigned int)sc[9] << 8)
                    | ((unsigned int)sc[10] << 16) | ((unsigned int)sc[11] << 24);
                unsigned int mins0 = utmp1 & 0x3f3f3f3fu;
                unsigned int mins1 = ((utmp2 >> 4) & 0x0f0f0f0fu)
                    | (((utmp1 >> 6) & 0x03030303u) << 4);
                unsigned int scw0 = utmp0 & 0x3f3f3f3fu;
                unsigned int scw1 = (utmp2 & 0x0f0f0f0fu)
                    | (((utmp0 >> 6) & 0x03030303u) << 4);
                long long prod = 0;
                for (int gg = 0; gg < 8; gg++) {
                    int bs = 0;
                    for (int t = 0; t < 32; t++) bs += q8[gg * 32 + t];
                    prod += (long long)bs * kq4_byte(mins0, mins1, gg);
                }
                sumf = fmaf(-dmin, (float)prod, sumf);
                long long sumi1 = 0, sumi2 = 0;
                for (int j = 0; j < 4; j++) {
                    const unsigned char* q4 = qs + j * 32;
                    const signed char* q8j = q8 + j * 64;
                    long long lo = 0, hi = 0;
                    for (int t = 0; t < 32; t++) {
                        lo += (long long)(q4[t] & 0xf) * q8j[t];
                        hi += (long long)(q4[t] >> 4) * q8j[32 + t];
                    }
                    sumi1 += lo * kq4_byte(scw0, scw1, 2 * j);
                    sumi2 += hi * kq4_byte(scw0, scw1, 2 * j + 1);
                }
                sumf = fmaf(d, (float)(sumi1 + sumi2), sumf);
            }
        }
        __syncthreads();
    }
    if (row_ok && pair_ok) {
        int orig = g_orig[g * DG_TILE_P + ty];
        out[(long long)orig * rows_per + row] = sumf;
    }
}

// Q8_0: 34B blocks; stage 20 blocks/row (680B -> padded 684B, 171 words odd).
extern "C" __global__ void q8_0_gemm_tile(
    const unsigned char* __restrict__ wire,
    const float* __restrict__ act_scales,       // [n_acts * nb]
    const signed char* __restrict__ act_quants, // [n_acts * nb * 32]
    const long long* __restrict__ g_base,
    const int* __restrict__ g_pos,
    const int* __restrict__ g_orig,
    int rows_per, int nb,
    float* __restrict__ out)
{
    const int B_STAGE = 20;
    const int ROW_STRIDE = 684;
    extern __shared__ unsigned char tile[];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int g = blockIdx.y;
    int row = blockIdx.x * DG_TILE_ROWS + tx;
    bool row_ok = row < rows_per;
    int pos = g_pos[g * DG_TILE_P + ty];
    bool pair_ok = pos >= 0;
    long long base = g_base[g];
    const int lin = ty * DG_TILE_ROWS + tx;
    const int nthreads = DG_TILE_ROWS * DG_TILE_P;
    float acc0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float acc1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int b0 = 0; b0 < nb; b0 += B_STAGE) {
        int nbs = min(B_STAGE, nb - b0);
        // Row bytes are 4B-aligned only at even block indices (34B blocks);
        // stage starts land on b0 multiples of 20 (even) so u16 is the safe
        // universal unit here: copy as u16 (rows are 2B-aligned always).
        int hw_per_row = (nbs * 34) / 2;
        int total_hw = DG_TILE_ROWS * hw_per_row;
        for (int w = lin; w < total_hw; w += nthreads) {
            int r = w / hw_per_row;
            int wo = w % hw_per_row;
            int gr = blockIdx.x * DG_TILE_ROWS + r;
            unsigned short val = 0u;
            if (gr < rows_per) {
                const unsigned char* src = wire
                    + (base + gr) * (long long)nb * 34
                    + (long long)b0 * 34;
                val = *(const unsigned short*)(src + wo * 2);
            }
            *(unsigned short*)(tile + r * ROW_STRIDE + wo * 2) = val;
        }
        __syncthreads();
        if (row_ok && pair_ok) {
            const float* a_s = act_scales + (long long)pos * nb;
            const signed char* a_q = act_quants + (long long)pos * nb * 32;
            for (int i = 0; i < nbs; i++) {
                const unsigned char* block = tile + tx * ROW_STRIDE + i * 34;
                int b = b0 + i;
                float dw = f16_bits_to_f32((unsigned short)block[0]
                    | ((unsigned short)block[1] << 8));
                float s = dw * a_s[b];
                const signed char* wq = (const signed char*)(block + 2);
                const signed char* yq = a_q + (long long)b * 32;
                float* acc = (b & 1) ? acc1 : acc0;
                for (int l = 0; l < 4; l++) {
                    int lane = 0;
                    for (int t = 0; t < 4; t++) {
                        lane += (int)wq[4 * l + t] * (int)yq[4 * l + t];
                        lane += (int)wq[16 + 4 * l + t] * (int)yq[16 + 4 * l + t];
                    }
                    acc[l] = fmaf((float)lane, s, acc[l]);
                }
            }
        }
        __syncthreads();
    }
    if (row_ok && pair_ok) {
        int orig = g_orig[g * DG_TILE_P + ty];
        out[(long long)orig * rows_per + row] =
            ((acc0[0] + acc0[1]) + (acc0[2] + acc0[3]))
            + ((acc1[0] + acc1[1]) + (acc1[2] + acc1[3]));
    }
}

// Q5_0: 22B blocks; stage 30 blocks/row (660B, 165 words odd — no pad).
extern "C" __global__ void q5_0_gemm_tile(
    const unsigned char* __restrict__ wire,
    const float* __restrict__ act_scales,
    const signed char* __restrict__ act_quants,
    const long long* __restrict__ g_base,
    const int* __restrict__ g_pos,
    const int* __restrict__ g_orig,
    int rows_per, int nb,
    float* __restrict__ out)
{
    const int B_STAGE = 30;
    const int ROW_STRIDE = 660;
    extern __shared__ unsigned char tile[];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int g = blockIdx.y;
    int row = blockIdx.x * DG_TILE_ROWS + tx;
    bool row_ok = row < rows_per;
    int pos = g_pos[g * DG_TILE_P + ty];
    bool pair_ok = pos >= 0;
    long long base = g_base[g];
    const int lin = ty * DG_TILE_ROWS + tx;
    const int nthreads = DG_TILE_ROWS * DG_TILE_P;
    float acc0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float acc1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int b0 = 0; b0 < nb; b0 += B_STAGE) {
        int nbs = min(B_STAGE, nb - b0);
        int hw_per_row = (nbs * 22) / 2;
        int total_hw = DG_TILE_ROWS * hw_per_row;
        for (int w = lin; w < total_hw; w += nthreads) {
            int r = w / hw_per_row;
            int wo = w % hw_per_row;
            int gr = blockIdx.x * DG_TILE_ROWS + r;
            unsigned short val = 0u;
            if (gr < rows_per) {
                const unsigned char* src = wire
                    + (base + gr) * (long long)nb * 22
                    + (long long)b0 * 22;
                val = *(const unsigned short*)(src + wo * 2);
            }
            *(unsigned short*)(tile + r * ROW_STRIDE + wo * 2) = val;
        }
        __syncthreads();
        if (row_ok && pair_ok) {
            const float* a_s = act_scales + (long long)pos * nb;
            const signed char* a_q = act_quants + (long long)pos * nb * 32;
            for (int i = 0; i < nbs; i++) {
                const unsigned char* block = tile + tx * ROW_STRIDE + i * 22;
                int b = b0 + i;
                float dw = f16_bits_to_f32((unsigned short)block[0]
                    | ((unsigned short)block[1] << 8));
                unsigned int qh = (unsigned int)block[2] | ((unsigned int)block[3] << 8)
                    | ((unsigned int)block[4] << 16) | ((unsigned int)block[5] << 24);
                const unsigned char* qs = block + 6;
                float s = dw * a_s[b];
                const signed char* yq = a_q + (long long)b * 32;
                float* acc = (b & 1) ? acc1 : acc0;
                for (int l = 0; l < 4; l++) {
                    int lane = 0;
                    for (int t = 0; t < 4; t++) {
                        int i0 = 4 * l + t;
                        int w0 = (int)((qs[i0] & 0x0f) | (((qh >> i0) & 1) << 4)) - 16;
                        lane += w0 * (int)yq[i0];
                        int i1 = 16 + 4 * l + t;
                        int w1 = (int)((qs[i1 - 16] >> 4) | (((qh >> i1) & 1) << 4)) - 16;
                        lane += w1 * (int)yq[i1];
                    }
                    acc[l] = fmaf((float)lane, s, acc[l]);
                }
            }
        }
        __syncthreads();
    }
    if (row_ok && pair_ok) {
        int orig = g_orig[g * DG_TILE_P + ty];
        out[(long long)orig * rows_per + row] =
            ((acc0[0] + acc0[1]) + (acc0[2] + acc0[3]))
            + ((acc1[0] + acc1[1]) + (acc1[2] + acc1[3]));
    }
}

// Q6_K: 210B superblocks (2B-aligned only); stage 3/row (630B -> padded 636).
extern "C" __global__ void q6k_gemm_tile(
    const unsigned char* __restrict__ wire,
    const float* __restrict__ act_scales,
    const signed char* __restrict__ act_quants,
    const long long* __restrict__ g_base,
    const int* __restrict__ g_pos,
    const int* __restrict__ g_orig,
    int rows_per, int bpr,
    float* __restrict__ out)
{
    const int SB_STAGE = 3;
    const int ROW_STRIDE = 636; // 3*210 padded to odd word count (159 words)
    extern __shared__ unsigned char tile[];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int g = blockIdx.y;
    int row = blockIdx.x * DG_TILE_ROWS + tx;
    bool row_ok = row < rows_per;
    int pos = g_pos[g * DG_TILE_P + ty];
    bool pair_ok = pos >= 0;
    long long base = g_base[g];
    const int lin = ty * DG_TILE_ROWS + tx;
    const int nthreads = DG_TILE_ROWS * DG_TILE_P;
    float sumf = 0.0f;
    for (int sb0 = 0; sb0 < bpr; sb0 += SB_STAGE) {
        int nsb = min(SB_STAGE, bpr - sb0);
        int hw_per_row = (nsb * 210) / 2;
        int total_hw = DG_TILE_ROWS * hw_per_row;
        for (int w = lin; w < total_hw; w += nthreads) {
            int r = w / hw_per_row;
            int wo = w % hw_per_row;
            int gr = blockIdx.x * DG_TILE_ROWS + r;
            unsigned short val = 0u;
            if (gr < rows_per) {
                const unsigned char* src = wire
                    + (base + gr) * (long long)bpr * 210
                    + (long long)sb0 * 210;
                val = *(const unsigned short*)(src + wo * 2);
            }
            *(unsigned short*)(tile + r * ROW_STRIDE + wo * 2) = val;
        }
        __syncthreads();
        if (row_ok && pair_ok) {
            const float* a_s = act_scales + (long long)pos * bpr;
            const signed char* a_q = act_quants + (long long)pos * bpr * 256;
            for (int i = 0; i < nsb; i++) {
                const unsigned char* block = tile + tx * ROW_STRIDE + i * 210;
                int sb = sb0 + i;
                const unsigned char* ql = block;
                const unsigned char* qh = block + 128;
                const signed char* scales = (const signed char*)(block + 192);
                float d_all = f16_bits_to_f32((unsigned short)block[208]
                    | ((unsigned short)block[209] << 8));
                const signed char* q8 = a_q + (long long)sb * 256;
                float y_d = a_s[sb];
                long long isum = 0;
                for (int half = 0; half < 2; half++) {
                    const unsigned char* qlh = ql + half * 64;
                    const unsigned char* qhh = qh + half * 32;
                    const signed char* q8h = q8 + half * 128;
                    const signed char* sc = scales + half * 8;
                    long long gs0 = 0, gs1 = 0, gs2 = 0, gs3 = 0,
                        gs4 = 0, gs5 = 0, gs6 = 0, gs7 = 0;
                    for (int l = 0; l < 32; l++) {
                        int v0 = (qlh[l] & 0xF) | ((qhh[l] & 3) << 4);
                        int v1 = (qlh[32 + l] & 0xF) | (((qhh[l] >> 2) & 3) << 4);
                        int v2 = (qlh[l] >> 4) | (((qhh[l] >> 4) & 3) << 4);
                        int v3 = (qlh[32 + l] >> 4) | (((qhh[l] >> 6) & 3) << 4);
                        if (l < 16) {
                            gs0 += (long long)v0 * q8h[l];
                            gs2 += (long long)v1 * q8h[32 + l];
                            gs4 += (long long)v2 * q8h[64 + l];
                            gs6 += (long long)v3 * q8h[96 + l];
                        } else {
                            gs1 += (long long)v0 * q8h[l];
                            gs3 += (long long)v1 * q8h[32 + l];
                            gs5 += (long long)v2 * q8h[64 + l];
                            gs7 += (long long)v3 * q8h[96 + l];
                        }
                    }
                    isum += gs0 * (long long)sc[0] + gs1 * (long long)sc[1]
                        + gs2 * (long long)sc[2] + gs3 * (long long)sc[3]
                        + gs4 * (long long)sc[4] + gs5 * (long long)sc[5]
                        + gs6 * (long long)sc[6] + gs7 * (long long)sc[7];
                }
                long long isum_mins = 0;
                for (int t = 0; t < 16; t++) {
                    long long bs = 0;
                    for (int l = 0; l < 16; l++) bs += q8[t * 16 + l];
                    isum_mins += bs * (long long)scales[t];
                }
                sumf = fmaf(d_all * y_d, (float)(isum - 32 * isum_mins), sumf);
            }
        }
        __syncthreads();
    }
    if (row_ok && pair_ok) {
        int orig = g_orig[g * DG_TILE_P + ty];
        out[(long long)orig * rows_per + row] = sumf;
    }
}

// ---- legacy one-thread-per-output ID kernels (kept for reference; the
// grouped-tile kernels above replaced them in dispatch) --------------------

extern "C" __global__ void q4k_gemm_id(
    const unsigned char* __restrict__ wire,
    const float* __restrict__ act_scales,      // [P * bpr]
    const signed char* __restrict__ act_quants, // [P * bpr * 256]
    const long long* __restrict__ pair_base,   // [n_pairs] first row
    const int* __restrict__ pair_pos,          // [n_pairs] activation index
    int n_pairs, int rows_per, int bpr,
    float* __restrict__ out)                   // [n_pairs * rows_per]
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)n_pairs * rows_per;
    if (idx >= total) return;
    int pair = (int)(idx / rows_per);
    int r = (int)(idx % rows_per);
    const unsigned char* rowp = wire + (pair_base[pair] + r) * (long long)bpr * 144;
    const float* a_s = act_scales + (long long)pair_pos[pair] * bpr;
    const signed char* a_q = act_quants + (long long)pair_pos[pair] * bpr * 256;
    float sumf = 0.0f;
    for (int i = 0; i < bpr; i++) {
        const unsigned char* block = rowp + (long long)i * 144;
        float yd = a_s[i];
        const signed char* q8 = a_q + (long long)i * 256;
        float d = yd * f16_bits_to_f32((unsigned short)block[0]
            | ((unsigned short)block[1] << 8));
        float dmin = yd * f16_bits_to_f32((unsigned short)block[2]
            | ((unsigned short)block[3] << 8));
        const unsigned char* sc = block + 4;
        const unsigned char* qs = block + 16;
        unsigned int utmp0 = (unsigned int)sc[0] | ((unsigned int)sc[1] << 8)
            | ((unsigned int)sc[2] << 16) | ((unsigned int)sc[3] << 24);
        unsigned int utmp1 = (unsigned int)sc[4] | ((unsigned int)sc[5] << 8)
            | ((unsigned int)sc[6] << 16) | ((unsigned int)sc[7] << 24);
        unsigned int utmp2 = (unsigned int)sc[8] | ((unsigned int)sc[9] << 8)
            | ((unsigned int)sc[10] << 16) | ((unsigned int)sc[11] << 24);
        unsigned int mins0 = utmp1 & 0x3f3f3f3fu;
        unsigned int mins1 = ((utmp2 >> 4) & 0x0f0f0f0fu)
            | (((utmp1 >> 6) & 0x03030303u) << 4);
        unsigned int scw0 = utmp0 & 0x3f3f3f3fu;
        unsigned int scw1 = (utmp2 & 0x0f0f0f0fu)
            | (((utmp0 >> 6) & 0x03030303u) << 4);
        long long prod = 0;
        for (int g = 0; g < 8; g++) {
            int bs = 0;
            for (int t = 0; t < 32; t++) bs += q8[g * 32 + t];
            prod += (long long)bs * kq4_byte(mins0, mins1, g);
        }
        sumf = fmaf(-dmin, (float)prod, sumf);
        long long sumi1 = 0, sumi2 = 0;
        for (int j = 0; j < 4; j++) {
            const unsigned char* q4 = qs + j * 32;
            const signed char* q8j = q8 + j * 64;
            long long lo = 0, hi = 0;
            for (int t = 0; t < 32; t++) {
                lo += (long long)(q4[t] & 0xf) * q8j[t];
                hi += (long long)(q4[t] >> 4) * q8j[32 + t];
            }
            sumi1 += lo * kq4_byte(scw0, scw1, 2 * j);
            sumi2 += hi * kq4_byte(scw0, scw1, 2 * j + 1);
        }
        sumf = fmaf(d, (float)(sumi1 + sumi2), sumf);
    }
    out[idx] = sumf;
}

// Q5_0 x Q8_0 batched-ID GEMM: 22-byte blocks (f16 d, u32 qh, 16 nibble
// bytes); weight w[idx] = ((nibble | (qh_bit << 4)) - 16). Same accumulator
// structure as the Q8_0 kernel (q0_pair_dot shape).
extern "C" __global__ void q5_0_gemm_id(
    const unsigned char* __restrict__ wire,
    const float* __restrict__ act_scales,
    const signed char* __restrict__ act_quants,
    const long long* __restrict__ pair_base,
    const int* __restrict__ pair_pos,
    int n_pairs, int rows_per, int nb,
    float* __restrict__ out)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)n_pairs * rows_per;
    if (idx >= total) return;
    int pair = (int)(idx / rows_per);
    int r = (int)(idx % rows_per);
    const unsigned char* rowp = wire + (pair_base[pair] + r) * (long long)nb * 22;
    const float* a_s = act_scales + (long long)pair_pos[pair] * nb;
    const signed char* a_q = act_quants + (long long)pair_pos[pair] * nb * 32;
    float acc0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float acc1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = 0; i < nb; i++) {
        const unsigned char* block = rowp + (long long)i * 22;
        float dw = f16_bits_to_f32((unsigned short)block[0]
            | ((unsigned short)block[1] << 8));
        unsigned int qh = (unsigned int)block[2] | ((unsigned int)block[3] << 8)
            | ((unsigned int)block[4] << 16) | ((unsigned int)block[5] << 24);
        const unsigned char* qs = block + 6;
        float s = dw * a_s[i];
        const signed char* yq = a_q + (long long)i * 32;
        float* acc = (i & 1) ? acc1 : acc0;
        for (int l = 0; l < 4; l++) {
            int lane = 0;
            for (int t = 0; t < 4; t++) {
                int i0 = 4 * l + t;
                int w0 = (int)((qs[i0] & 0x0f) | (((qh >> i0) & 1) << 4)) - 16;
                lane += w0 * (int)yq[i0];
                int i1 = 16 + 4 * l + t;
                int w1 = (int)((qs[i1 - 16] >> 4) | (((qh >> i1) & 1) << 4)) - 16;
                lane += w1 * (int)yq[i1];
            }
            acc[l] = fmaf((float)lane, s, acc[l]);
        }
    }
    out[idx] = ((acc0[0] + acc0[1]) + (acc0[2] + acc0[3]))
        + ((acc1[0] + acc1[1]) + (acc1[2] + acc1[3]));
}

// Q6_K x Q8_K batched-ID GEMM (210-byte superblocks; same decode as
// q6k_gemv_q8k above, with the (pair_base, pair_pos) indexing).
extern "C" __global__ void q6k_gemm_id(
    const unsigned char* __restrict__ wire,
    const float* __restrict__ act_scales,
    const signed char* __restrict__ act_quants,
    const long long* __restrict__ pair_base,
    const int* __restrict__ pair_pos,
    int n_pairs, int rows_per, int bpr,
    float* __restrict__ out)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)n_pairs * rows_per;
    if (idx >= total) return;
    int pair = (int)(idx / rows_per);
    int r = (int)(idx % rows_per);
    const unsigned char* rowp = wire + (pair_base[pair] + r) * (long long)bpr * 210;
    const float* a_s = act_scales + (long long)pair_pos[pair] * bpr;
    const signed char* a_q = act_quants + (long long)pair_pos[pair] * bpr * 256;
    float sumf = 0.0f;
    for (int b = 0; b < bpr; b++) {
        const unsigned char* block = rowp + (long long)b * 210;
        const unsigned char* ql = block;
        const unsigned char* qh = block + 128;
        const signed char* scales = (const signed char*)(block + 192);
        float d_all = f16_bits_to_f32((unsigned short)block[208]
            | ((unsigned short)block[209] << 8));
        const signed char* q8 = a_q + (long long)b * 256;
        float y_d = a_s[b];
        long long isum = 0;
        for (int half = 0; half < 2; half++) {
            const unsigned char* qlh = ql + half * 64;
            const unsigned char* qhh = qh + half * 32;
            const signed char* q8h = q8 + half * 128;
            const signed char* sc = scales + half * 8;
            long long gs0 = 0, gs1 = 0, gs2 = 0, gs3 = 0,
                gs4 = 0, gs5 = 0, gs6 = 0, gs7 = 0;
            for (int l = 0; l < 32; l++) {
                int v0 = (qlh[l] & 0xF) | ((qhh[l] & 3) << 4);
                int v1 = (qlh[32 + l] & 0xF) | (((qhh[l] >> 2) & 3) << 4);
                int v2 = (qlh[l] >> 4) | (((qhh[l] >> 4) & 3) << 4);
                int v3 = (qlh[32 + l] >> 4) | (((qhh[l] >> 6) & 3) << 4);
                if (l < 16) {
                    gs0 += (long long)v0 * q8h[l];
                    gs2 += (long long)v1 * q8h[32 + l];
                    gs4 += (long long)v2 * q8h[64 + l];
                    gs6 += (long long)v3 * q8h[96 + l];
                } else {
                    gs1 += (long long)v0 * q8h[l];
                    gs3 += (long long)v1 * q8h[32 + l];
                    gs5 += (long long)v2 * q8h[64 + l];
                    gs7 += (long long)v3 * q8h[96 + l];
                }
            }
            isum += gs0 * (long long)sc[0] + gs1 * (long long)sc[1]
                + gs2 * (long long)sc[2] + gs3 * (long long)sc[3]
                + gs4 * (long long)sc[4] + gs5 * (long long)sc[5]
                + gs6 * (long long)sc[6] + gs7 * (long long)sc[7];
        }
        long long isum_mins = 0;
        for (int t = 0; t < 16; t++) {
            long long bs = 0;
            for (int l = 0; l < 16; l++) bs += q8[t * 16 + l];
            isum_mins += bs * (long long)scales[t];
        }
        sumf = fmaf(d_all * y_d, (float)(isum - 32 * isum_mins), sumf);
    }
    out[idx] = sumf;
}

// FAST-mode bidirectional diffusion attention: one block per (pos, head).
// Mirrors the CPU path's math shape (raw QK dots -> masked softmax -> V mix)
// with the region-aware mask: prompt queries are causal over the prompt
// (SWA-clipped on sliding layers); canvas queries see everything on global
// layers, and all canvas plus the last (n_swa - 1) prompt positions on
// sliding layers. f32 throughout (fast mode: not bit-exact).
// q_off: queries cover positions [q_off, q_off + nq) of the full n-position
// context (prefix-KV cached mode passes q_off = p with canvas-only queries;
// the full pass uses q_off = 0, nq = n). K/V always span all n positions.
extern "C" __global__ void dg_attn(
    const float* __restrict__ q,   // [nq * heads * hd]
    const float* __restrict__ k,   // [n * kv_heads * hd]
    const float* __restrict__ v,   // [n * kv_heads * hd]
    float* __restrict__ out,       // [nq * heads * hd]
    int n, int nq, int q_off, int heads, int kv_heads, int hd, int group,
    int p, int win, int sliding, long long lo)
{
    int lpos = blockIdx.x / heads;
    int hh = blockIdx.x % heads;
    if (lpos >= nq) return;
    int pos = q_off + lpos;
    int kvh = hh / group;
    extern __shared__ float row[]; // [n] scores
    __shared__ float red[256];
    const float* qh = q + ((long long)lpos * heads + hh) * hd;
    for (int kp = threadIdx.x; kp < n; kp += blockDim.x) {
        bool ok;
        if (pos >= p) {
            ok = sliding ? (kp >= p || (long long)kp >= lo) : true;
        } else {
            ok = (kp <= pos) && (!sliding || kp + win > pos);
        }
        float s = DG_NEG_INF;
        if (ok) {
            const float* kk = k + ((long long)kp * kv_heads + kvh) * hd;
            float acc = 0.0f;
            for (int d = 0; d < hd; d++) acc += qh[d] * kk[d];
            s = acc;
        }
        row[kp] = s;
    }
    __syncthreads();
    float m = DG_NEG_INF;
    for (int i = threadIdx.x; i < n; i += blockDim.x) m = fmaxf(m, row[i]);
    red[threadIdx.x] = m;
    __syncthreads();
    for (int s2 = blockDim.x >> 1; s2 > 0; s2 >>= 1) {
        if (threadIdx.x < s2)
            red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s2]);
        __syncthreads();
    }
    m = red[0];
    __syncthreads();
    float sum = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float e = expf(row[i] - m);
        row[i] = e;
        sum += e;
    }
    red[threadIdx.x] = sum;
    __syncthreads();
    for (int s2 = blockDim.x >> 1; s2 > 0; s2 >>= 1) {
        if (threadIdx.x < s2) red[threadIdx.x] += red[threadIdx.x + s2];
        __syncthreads();
    }
    float inv = 1.0f / red[0];
    __syncthreads();
    float* op = out + ((long long)lpos * heads + hh) * hd;
    for (int d = threadIdx.x; d < hd; d += blockDim.x) {
        float acc = 0.0f;
        for (int kp = 0; kp < n; kp++) {
            acc += row[kp] * v[((long long)kp * kv_heads + kvh) * hd + d];
        }
        op[d] = acc * inv;
    }
}

// FAST-mode lm_head: tiled f32xf16 GEMM against the RESIDENT transposed
// embedding (the SC stage's emb_t, [hidden][vocab] row-major — already the
// B-matrix layout a GEMM wants, and already in VRAM). C[m*n] = A[m*k] x B[k*n].
// cap > 0 fuses the final-logit softcapping (tanh(x/cap)*cap) into the store.
extern "C" __global__ void lm_head_f16_gemm(
    const float* __restrict__ a,
    const unsigned short* __restrict__ bt,
    float* __restrict__ cmat,
    int m, int k, int n, float cap)
{
    __shared__ float As[16][16];
    __shared__ float Bs[16][17];
    int tx = threadIdx.x, ty = threadIdx.y;
    int col = blockIdx.x * 16 + tx;
    int rowc = blockIdx.y * 16 + ty;
    float acc = 0.0f;
    for (int k0 = 0; k0 < k; k0 += 16) {
        As[ty][tx] = (rowc < m && (k0 + tx) < k) ? a[(long long)rowc * k + k0 + tx] : 0.0f;
        int brow = k0 + ty;
        Bs[ty][tx] = (brow < k && col < n)
            ? f16_bits_to_f32(bt[(long long)brow * n + col])
            : 0.0f;
        __syncthreads();
        for (int kk = 0; kk < 16; kk++) acc += As[ty][kk] * Bs[kk][tx];
        __syncthreads();
    }
    if (rowc < m && col < n) {
        if (cap > 0.0f) acc = tanhf(acc * (1.0f / cap)) * cap;
        cmat[(long long)rowc * n + col] = acc;
    }
}

extern "C" __global__ void q8_0_gemm_id(
    const unsigned char* __restrict__ wire,
    const float* __restrict__ act_scales,      // [P * nb]
    const signed char* __restrict__ act_quants, // [P * nb * 32]
    const long long* __restrict__ pair_base,
    const int* __restrict__ pair_pos,
    int n_pairs, int rows_per, int nb,
    float* __restrict__ out)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)n_pairs * rows_per;
    if (idx >= total) return;
    int pair = (int)(idx / rows_per);
    int r = (int)(idx % rows_per);
    const unsigned char* rowp = wire + (pair_base[pair] + r) * (long long)nb * 34;
    const float* a_s = act_scales + (long long)pair_pos[pair] * nb;
    const signed char* a_q = act_quants + (long long)pair_pos[pair] * nb * 32;
    float acc0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float acc1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int i = 0; i < nb; i++) {
        const unsigned char* block = rowp + (long long)i * 34;
        float dw = f16_bits_to_f32((unsigned short)block[0]
            | ((unsigned short)block[1] << 8));
        float s = dw * a_s[i];
        const signed char* wq = (const signed char*)(block + 2);
        const signed char* yq = a_q + (long long)i * 32;
        float* acc = (i & 1) ? acc1 : acc0;
        for (int l = 0; l < 4; l++) {
            int lane = 0;
            for (int t = 0; t < 4; t++) {
                lane += (int)wq[4 * l + t] * (int)yq[4 * l + t];
                lane += (int)wq[16 + 4 * l + t] * (int)yq[16 + 4 * l + t];
            }
            acc[l] = fmaf((float)lane, s, acc[l]);
        }
    }
    out[idx] = ((acc0[0] + acc0[1]) + (acc0[2] + acc0[3]))
        + ((acc1[0] + acc1[1]) + (acc1[2] + acc1[3]));
}
"#;

struct Engine {
    stream: Arc<CudaStream>,
    ctx: Arc<CudaContext>,
    sc_func: CudaFunction,
    lm_func: CudaFunction,
    q4k_rows_func: CudaFunction,
    q80_rows_func: CudaFunction,
    q4k_tile_func: CudaFunction,
    q80_tile_func: CudaFunction,
    q50_tile_func: CudaFunction,
    q6k_tile_func: CudaFunction,
    attn_func: CudaFunction,
    lm_gemm_func: CudaFunction,
    sc_probs_func: CudaFunction,
    sc_gemm_func: CudaFunction,
    /// Previous step's lm_head logits, left device-resident by the fast lm
    /// GEMM for the next step's fused SC probs. Consumed one-shot (`take`)
    /// so a stale buffer can never silently feed the SC stage.
    last_logits: Option<(CudaSlice<f32>, usize, usize)>,
    /// FAST-mode streaming scratch for tensors that miss the resident pool
    /// (grown to the largest streamed tensor; one at a time).
    scratch: Option<CudaSlice<u8>>,
    /// Second device scratch: the copy stream DMAs the NEXT streamed tensor
    /// here while the current one computes (device-side double buffering).
    scratch2: Option<CudaSlice<u8>>,
    /// Dedicated copy stream for device-prefetch DMAs.
    copy_stream: Option<Arc<CudaStream>>,
    /// In-flight device prefetch: (key, into-scratch2?, DMA event, pinned buf
    /// index held until the event completes).
    dev_pf: Option<((usize, usize), bool, cudarc::driver::CudaEvent, usize)>,
    /// Pinned (page-locked) host staging ring for streamed uploads. The
    /// read-ahead worker fills upcoming buffers while the current one feeds
    /// the htod; a deeper ring keeps the disk continuously busy.
    pin_bufs: Vec<cudarc::driver::PinnedHostSlice<u8>>,
    /// Read-ahead pipeline (see `ReadAhead`): the per-step streamed-tensor
    /// sequence is deterministic, so step 0 learns it and steps 1+ overlap
    /// each tensor's FILE READ (the actual wall, ~10x the PCIe copy) with
    /// the previous tensor's GPU work.
    ra: Option<ReadAhead>,
    /// Resident transposed embedding (f16) for the SC matmul.
    sc_emb: Option<(CudaSlice<u16>, (usize, usize))>,
    /// Resident Q6_K lm_head weight (wire bytes).
    lm_wire: Option<(CudaSlice<u8>, (usize, usize))>,
    /// MoE expert pool: whole expert tensors resident in VRAM, keyed by
    /// (host base ptr, len). Uploaded greedily until the byte budget runs out;
    /// tensors that miss the budget are remembered so they fail fast to CPU.
    expert_pool: std::collections::HashMap<(usize, usize), CudaSlice<u8>>,
    expert_rejected: std::collections::HashSet<(usize, usize)>,
    /// Remaining expert-pool byte budget; `None` until first use (computed
    /// from free VRAM minus a reserve, or `CAMELID_DG_EXPERT_VRAM_MB`).
    expert_budget: Option<u64>,
    /// Carve-out for SMALL tensors (attention/dense projections, ~40 MiB per
    /// layer total). Without it the greedy expert uploads exhaust the budget
    /// first and the small HOT tensors re-stream every layer, every step.
    small_budget: Option<u64>,
    /// ZERO-COPY tier: expert tensors pinned in DEVICE-MAPPED host RAM. GPU
    /// kernels read them in place over PCIe (no staging, no per-step copies);
    /// the disk is read once at first touch, then never again. The wrapped
    /// `CudaSlice` values are deliberately never dropped (the engine static
    /// lives for the process) — dropping would `cuMemFree` a HOST pointer.
    host_pool: std::collections::HashMap<(usize, usize), CudaSlice<u8>>,
    host_rejected: std::collections::HashSet<(usize, usize)>,
    /// Remaining zero-copy byte budget (`CAMELID_DG_HOST_POOL_MB`, default
    /// 8 GiB; 0 disables the tier).
    host_budget: Option<u64>,
}

// SAFETY: the engine is only touched while holding ENGINE's mutex (the same
// single-owner discipline the resident decode cache uses for cudarc handles).
unsafe impl Send for Engine {}

static ENGINE: OnceLock<Mutex<Option<Engine>>> = OnceLock::new();

/// Largest streamable tensor (pinned staging buffers are fixed at this cap;
/// the biggest streamed DG tensor is ~285 MiB).
const DG_PIN_CAP: usize = 300 * 1024 * 1024;

/// Pinned staging ring depth (buffers): 1 feeds the current htod while the
/// worker reads up to DG_RA_BUFS-1 upcoming tensors, keeping the disk busy.
const DG_RA_BUFS: usize = 4;

/// Residency split: tensors under this size (attention/dense projections)
/// draw from the small carve-out; expert tensors draw from the remainder.
const DG_SMALL_TENSOR: usize = 32 * 1024 * 1024;
/// Carve-out for the small hot tensors (all 30 layers' attn+dense ≈ 1.2 GiB).
const DG_SMALL_BUDGET: u64 = 1300 * 1024 * 1024;

/// Initialize the split pool budgets on first use.
fn ensure_budgets(eng: &mut Engine) {
    if eng.expert_budget.is_some() {
        return;
    }
    let total = match std::env::var("CAMELID_DG_EXPERT_VRAM_MB")
        .ok()
        .and_then(|v| v.parse::<u64>().ok())
    {
        Some(mb) => mb * 1024 * 1024,
        None => {
            // Must cover: SC emb 1.47G + scratch 0.3G + scratch2 0.3G +
            // persistent logits 0.27G + probs 0.13G + transients.
            const RESERVE: u64 = 2600 * 1024 * 1024;
            let free = cudarc::driver::result::mem_get_info()
                .map(|(f, _)| f as u64)
                .unwrap_or(0);
            free.saturating_sub(RESERVE)
        }
    };
    let small = total.min(DG_SMALL_BUDGET);
    eng.small_budget = Some(small);
    eng.expert_budget = Some(total - small);
    eprintln!(
        "[dg-cuda] pool budgets: {:.2} GiB experts + {:.2} GiB small tensors",
        (total - small) as f64 / (1u64 << 30) as f64,
        small as f64 / (1u64 << 30) as f64
    );
}

/// Set once when the NVRTC engine build fails: every later GPU call must
/// fail FAST to the CPU path. Without this, each of the thousands of
/// per-step calls would retry the full kernel compile (~0.5 s each) and a
/// broken kernel would run 20-30x SLOWER than plain CPU.
static ENGINE_FAILED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn gate_off() -> bool {
    ENGINE_FAILED.load(std::sync::atomic::Ordering::Relaxed)
        || std::env::var("CAMELID_DG_CUDA").as_deref() == Ok("0")
        || !crate::cuda::is_available()
}

fn build_engine() -> Result<Engine, String> {
    let ordinal = crate::cuda::selected_device_ordinal();
    let ctx = CudaContext::new(ordinal).map_err(|e| format!("CudaContext::new({ordinal}): {e}"))?;
    let stream = ctx.default_stream();
    // Retain the CUDA mempool across syncs: the default release threshold (0)
    // hands every freed block back to the OS at each synchronization, and
    // this engine allocates ~1,700 transient buffers per denoise step — each
    // would re-page through WDDM kernel mode. Best-effort (harmless if the
    // driver refuses).
    unsafe {
        use cudarc::driver::sys;
        let mut pool: sys::CUmemoryPool = std::ptr::null_mut();
        if sys::cuDeviceGetDefaultMemPool(&mut pool, ordinal as i32) == sys::CUresult::CUDA_SUCCESS
            && !pool.is_null()
        {
            let mut threshold: u64 = u64::MAX;
            let _ = sys::cuMemPoolSetAttribute(
                pool,
                sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                &mut threshold as *mut u64 as *mut core::ffi::c_void,
            );
        }
    }
    let opts = CompileOptions {
        fmad: Some(false),
        arch: Some("compute_61"),
        ..Default::default()
    };
    let ptx = compile_ptx_with_opts(KERNEL, opts).map_err(|e| format!("nvrtc: {e}"))?;
    let m = ctx
        .load_module(ptx)
        .map_err(|e| format!("load_module: {e}"))?;
    let sc_func = m
        .load_function("sc_soft_embedding")
        .map_err(|e| format!("load sc_soft_embedding: {e}"))?;
    let lm_func = m
        .load_function("q6k_gemv_q8k")
        .map_err(|e| format!("load q6k_gemv_q8k: {e}"))?;
    let q4k_rows_func = m
        .load_function("q4k_rows_gemv")
        .map_err(|e| format!("load q4k_rows_gemv: {e}"))?;
    let q80_rows_func = m
        .load_function("q8_0_rows_gemv")
        .map_err(|e| format!("load q8_0_rows_gemv: {e}"))?;
    let q4k_tile_func = m
        .load_function("q4k_gemm_tile")
        .map_err(|e| format!("load q4k_gemm_tile: {e}"))?;
    let q80_tile_func = m
        .load_function("q8_0_gemm_tile")
        .map_err(|e| format!("load q8_0_gemm_tile: {e}"))?;
    let q50_tile_func = m
        .load_function("q5_0_gemm_tile")
        .map_err(|e| format!("load q5_0_gemm_tile: {e}"))?;
    let q6k_tile_func = m
        .load_function("q6k_gemm_tile")
        .map_err(|e| format!("load q6k_gemm_tile: {e}"))?;
    let attn_func = m
        .load_function("dg_attn")
        .map_err(|e| format!("load dg_attn: {e}"))?;
    let lm_gemm_func = m
        .load_function("lm_head_f16_gemm")
        .map_err(|e| format!("load lm_head_f16_gemm: {e}"))?;
    let sc_probs_func = m
        .load_function("sc_probs_f16")
        .map_err(|e| format!("load sc_probs_f16: {e}"))?;
    let sc_gemm_func = m
        .load_function("sc_f16_gemm")
        .map_err(|e| format!("load sc_f16_gemm: {e}"))?;
    Ok(Engine {
        stream,
        ctx,
        sc_func,
        lm_func,
        q4k_rows_func,
        q80_rows_func,
        q4k_tile_func,
        q80_tile_func,
        q50_tile_func,
        q6k_tile_func,
        attn_func,
        lm_gemm_func,
        sc_probs_func,
        sc_gemm_func,
        last_logits: None,
        scratch: None,
        scratch2: None,
        copy_stream: None,
        dev_pf: None,
        pin_bufs: Vec::new(),
        ra: None,
        sc_emb: None,
        lm_wire: None,
        expert_pool: std::collections::HashMap::new(),
        expert_rejected: std::collections::HashSet::new(),
        expert_budget: None,
        small_budget: None,
        host_pool: std::collections::HashMap::new(),
        host_rejected: std::collections::HashSet::new(),
        host_budget: None,
    })
}

/// `soft[pos*hidden + e] = (Σ_v emb_t[e][v] * probs[pos][v]) * embed_scale` on
/// the GPU (f32 accumulate). `emb_t` is cached resident across calls. Returns
/// `None` (→ CPU fallback) on any failure.
pub(crate) fn sc_soft_embedding_gpu(
    emb_t: &[u16],
    probs_f16: &[u16],
    c: usize,
    hidden: usize,
    n_vocab: usize,
    embed_scale: f32,
) -> Option<Vec<f32>> {
    if gate_off() {
        return None;
    }
    // Per-stage opt-out: the SC matmul is the one non-bit-exact GPU stage
    // (f32 accumulation). CAMELID_DG_CUDA_SC=0 keeps it on CPU so a run with
    // the bit-exact stages (expert pool, lm_head) can be compared byte-for-
    // byte against the CPU-pure oracle.
    if std::env::var("CAMELID_DG_CUDA_SC").as_deref() == Ok("0") {
        return None;
    }
    let cell = ENGINE.get_or_init(|| Mutex::new(None));
    let mut guard = cell.lock().ok()?;
    if guard.is_none() {
        match build_engine() {
            Ok(e) => *guard = Some(e),
            Err(err) => {
                ENGINE_FAILED.store(true, std::sync::atomic::Ordering::Relaxed);
                eprintln!("[dg-cuda] engine build failed ({err}); CPU fallback");
                return None;
            }
        }
    }
    let eng = guard.as_mut()?;
    let mut run = || -> Result<Vec<f32>, String> {
        let s = eng.stream.clone();
        let key = (emb_t.as_ptr() as usize, emb_t.len());
        if eng.sc_emb.as_ref().map(|(_, k)| *k != key).unwrap_or(true) {
            let mut dev = s
                .alloc_zeros::<u16>(emb_t.len())
                .map_err(|e| format!("alloc emb_t: {e}"))?;
            s.memcpy_htod(emb_t, &mut dev)
                .map_err(|e| format!("upload emb_t: {e}"))?;
            eng.sc_emb = Some((dev, key));
        }
        let emb_dev = &eng.sc_emb.as_ref().unwrap().0;
        let mut probs_dev = s
            .alloc_zeros::<u16>(probs_f16.len())
            .map_err(|e| format!("alloc probs: {e}"))?;
        s.memcpy_htod(probs_f16, &mut probs_dev)
            .map_err(|e| format!("upload probs: {e}"))?;
        let mut soft_dev = s
            .alloc_zeros::<f32>(c * hidden)
            .map_err(|e| format!("alloc soft: {e}"))?;
        let cfg = LaunchConfig {
            grid_dim: ((c * hidden) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let (h, nv, cc) = (hidden as i32, n_vocab as i32, c as i32);
        let mut b = s.launch_builder(&eng.sc_func);
        b.arg(emb_dev)
            .arg(&probs_dev)
            .arg(&mut soft_dev)
            .arg(&h)
            .arg(&nv)
            .arg(&cc)
            .arg(&embed_scale);
        unsafe { b.launch(cfg) }.map_err(|e| format!("launch: {e}"))?;
        let mut out = vec![0f32; c * hidden];
        s.memcpy_dtoh(&soft_dev, &mut out)
            .map_err(|e| format!("download: {e}"))?;
        eng.ctx.synchronize().map_err(|e| format!("sync: {e}"))?;
        Ok(out)
    };
    match run() {
        Ok(v) => Some(v),
        Err(err) => {
            eprintln!("[dg-cuda] sc matmul failed ({err}); CPU fallback");
            None
        }
    }
}

/// Q6_K lm_head GEMV over the `c` canvas positions: `out[pos*rows + row]` =
/// dot(Q6_K row, Q8_K activation[pos]). `wire` (the Q6_K weight bytes) is cached
/// resident. `act_scales`/`act_quants` are the per-position Q8_K blocks packed
/// SoA. Returns the `[c*rows]` logits, or `None` on any failure.
pub(crate) fn lm_head_q6k_gpu(
    wire: &[u8],
    rows: usize,
    bpr: usize,
    act_scales: &[f32],
    act_quants: &[i8],
    c: usize,
) -> Option<Vec<f32>> {
    if gate_off() {
        return None;
    }
    debug_assert_eq!(wire.len(), rows * bpr * 210);
    debug_assert_eq!(act_scales.len(), c * bpr);
    debug_assert_eq!(act_quants.len(), c * bpr * 256);
    let cell = ENGINE.get_or_init(|| Mutex::new(None));
    let mut guard = cell.lock().ok()?;
    if guard.is_none() {
        match build_engine() {
            Ok(e) => *guard = Some(e),
            Err(err) => {
                ENGINE_FAILED.store(true, std::sync::atomic::Ordering::Relaxed);
                eprintln!("[dg-cuda] engine build failed ({err}); CPU fallback");
                return None;
            }
        }
    }
    let eng = guard.as_mut()?;
    let mut run = || -> Result<Vec<f32>, String> {
        let s = eng.stream.clone();
        let key = (wire.as_ptr() as usize, wire.len());
        if eng.lm_wire.as_ref().map(|(_, k)| *k != key).unwrap_or(true) {
            let mut dev = s
                .alloc_zeros::<u8>(wire.len())
                .map_err(|e| format!("alloc lm wire: {e}"))?;
            s.memcpy_htod(wire, &mut dev)
                .map_err(|e| format!("upload lm wire: {e}"))?;
            eng.lm_wire = Some((dev, key));
        }
        let wire_dev = &eng.lm_wire.as_ref().unwrap().0;
        let mut sc_dev = s
            .alloc_zeros::<f32>(act_scales.len())
            .map_err(|e| format!("alloc act scales: {e}"))?;
        s.memcpy_htod(act_scales, &mut sc_dev)
            .map_err(|e| format!("upload act scales: {e}"))?;
        let mut q_dev = s
            .alloc_zeros::<i8>(act_quants.len())
            .map_err(|e| format!("alloc act quants: {e}"))?;
        s.memcpy_htod(act_quants, &mut q_dev)
            .map_err(|e| format!("upload act quants: {e}"))?;
        let mut out_dev = s
            .alloc_zeros::<f32>(c * rows)
            .map_err(|e| format!("alloc logits: {e}"))?;
        let total = (c * rows) as u32;
        let block = 256u32;
        let cfg = LaunchConfig {
            grid_dim: (total.div_ceil(block), 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let (r, bp, cc) = (rows as i32, bpr as i32, c as i32);
        let mut b = s.launch_builder(&eng.lm_func);
        b.arg(wire_dev)
            .arg(&sc_dev)
            .arg(&q_dev)
            .arg(&r)
            .arg(&bp)
            .arg(&cc)
            .arg(&mut out_dev);
        unsafe { b.launch(cfg) }.map_err(|e| format!("launch: {e}"))?;
        let mut out = vec![0f32; c * rows];
        s.memcpy_dtoh(&out_dev, &mut out)
            .map_err(|e| format!("download logits: {e}"))?;
        eng.ctx.synchronize().map_err(|e| format!("sync: {e}"))?;
        Ok(out)
    };
    match run() {
        Ok(v) => Some(v),
        Err(err) => {
            eprintln!("[dg-cuda] lm_head failed ({err}); CPU fallback");
            None
        }
    }
}

/// Which expert-tensor GEMV kernel to run (matches the DG expert formats).
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum DgExpertKind {
    /// 144-byte superblocks; activation is Q8_K (scales [bpr], quants [bpr*256]).
    Q4K,
    /// 34-byte blocks; activation is Q8_0 (scales [nb], quants [nb*32]).
    Q80,
    /// 22-byte blocks; activation is Q8_0 (same layout as `Q80`). FAST mode only.
    Q50,
    /// 210-byte superblocks; activation is Q8_K (same layout as `Q4K`).
    /// FAST mode only (attn_v / lm_head class tensors).
    Q6K,
}

/// MoE expert row-range GEMV on the VRAM-resident expert pool.
///
/// `tensor` is the WHOLE tensor's wire bytes (an mmap slice — creating it does
/// not fault pages; only an upload reads it). On first sight of a tensor it is
/// uploaded resident if the pool budget allows, otherwise remembered as
/// rejected (fast CPU fallback thereafter). The kernels mirror the CPU
/// `q4_k_dot_scalar` / `q0_pair_dot` reductions exactly, so a resident expert
/// computes bit-identically to the CPU path — where a weight lives never
/// changes the math. Budget: `CAMELID_DG_EXPERT_VRAM_MB` or free VRAM minus a
/// reserve for the SC embedding + lm_head + scratch.
pub(crate) fn expert_rows_gemv_gpu(
    tensor: &[u8],
    kind: DgExpertKind,
    first_row: usize,
    n_rows: usize,
    blocks_per_row: usize,
    act_scales: &[f32],
    act_quants: &[i8],
) -> Option<Vec<f32>> {
    if gate_off() {
        return None;
    }
    let cell = ENGINE.get_or_init(|| Mutex::new(None));
    let mut guard = cell.lock().ok()?;
    if guard.is_none() {
        match build_engine() {
            Ok(e) => *guard = Some(e),
            Err(err) => {
                ENGINE_FAILED.store(true, std::sync::atomic::Ordering::Relaxed);
                eprintln!("[dg-cuda] engine build failed ({err}); CPU fallback");
                return None;
            }
        }
    }
    let eng = guard.as_mut()?;
    let key = (tensor.as_ptr() as usize, tensor.len());
    if eng.expert_rejected.contains(&key) {
        return None;
    }
    if !eng.expert_pool.contains_key(&key) {
        // Initialize the pool budgets on first use (expert tensors draw from
        // the big-tensor share; see `ensure_budgets`).
        ensure_budgets(eng);
        let budget = eng.expert_budget.unwrap();
        if (tensor.len() as u64) > budget {
            eng.expert_rejected.insert(key);
            eprintln!(
                "[dg-cuda] expert tensor {:.0} MiB over remaining budget {:.0} MiB; CPU keeps it",
                tensor.len() as f64 / (1 << 20) as f64,
                budget as f64 / (1 << 20) as f64
            );
            return None;
        }
        let s = eng.stream.clone();
        let mut dev = match s.alloc_zeros::<u8>(tensor.len()) {
            Ok(d) => d,
            Err(e) => {
                eng.expert_rejected.insert(key);
                eprintln!("[dg-cuda] expert alloc failed ({e}); CPU keeps it");
                return None;
            }
        };
        if let Err(e) = s.memcpy_htod(tensor, &mut dev) {
            eng.expert_rejected.insert(key);
            eprintln!("[dg-cuda] expert upload failed ({e}); CPU keeps it");
            return None;
        }
        eng.expert_budget = Some(budget - tensor.len() as u64);
        eng.expert_pool.insert(key, dev);
        eprintln!(
            "[dg-cuda] expert tensor resident: {:.0} MiB ({} tensors, {:.2} GiB budget left)",
            tensor.len() as f64 / (1 << 20) as f64,
            eng.expert_pool.len(),
            eng.expert_budget.unwrap() as f64 / (1u64 << 30) as f64
        );
    }
    let func = match kind {
        DgExpertKind::Q4K => &eng.q4k_rows_func,
        DgExpertKind::Q80 => &eng.q80_rows_func,
        // No single-activation Q5_0/Q6_K kernels: the parity path never
        // routes them (fast-mode-only kinds).
        DgExpertKind::Q50 | DgExpertKind::Q6K => return None,
    };
    let run = || -> Result<Vec<f32>, String> {
        let s = eng.stream.clone();
        let wire_dev = eng.expert_pool.get(&key).unwrap();
        let mut sc_dev = s
            .alloc_zeros::<f32>(act_scales.len())
            .map_err(|e| format!("alloc act scales: {e}"))?;
        s.memcpy_htod(act_scales, &mut sc_dev)
            .map_err(|e| format!("upload act scales: {e}"))?;
        let mut q_dev = s
            .alloc_zeros::<i8>(act_quants.len())
            .map_err(|e| format!("alloc act quants: {e}"))?;
        s.memcpy_htod(act_quants, &mut q_dev)
            .map_err(|e| format!("upload act quants: {e}"))?;
        let mut out_dev = s
            .alloc_zeros::<f32>(n_rows)
            .map_err(|e| format!("alloc out: {e}"))?;
        let block = 256u32;
        let cfg = LaunchConfig {
            grid_dim: ((n_rows as u32).div_ceil(block), 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        };
        let (fr, nr, bp) = (first_row as i64, n_rows as i32, blocks_per_row as i32);
        let mut b = s.launch_builder(func);
        b.arg(wire_dev)
            .arg(&sc_dev)
            .arg(&q_dev)
            .arg(&fr)
            .arg(&nr)
            .arg(&bp)
            .arg(&mut out_dev);
        unsafe { b.launch(cfg) }.map_err(|e| format!("launch: {e}"))?;
        let mut out = vec![0f32; n_rows];
        s.memcpy_dtoh(&out_dev, &mut out)
            .map_err(|e| format!("download: {e}"))?;
        eng.ctx.synchronize().map_err(|e| format!("sync: {e}"))?;
        Ok(out)
    };
    match run() {
        Ok(v) => Some(v),
        Err(err) => {
            eprintln!("[dg-cuda] expert gemv failed ({err}); CPU fallback");
            None
        }
    }
}

/// FAST-mode batched-ID GEMM (`CAMELID_DG_FAST`): every (pair, row) output in
/// one launch. `pair_base[n]` selects the row window (expert offset, 0 for a
/// dense tensor), `pair_pos[n]` selects the activation. The tensor computes
/// from the resident pool when it fits the budget, else it streams through a
/// reusable scratch upload (~tens of ms for a 272 MiB expert tensor — amortized
/// over ALL pairs, which is the whole point vs per-position reads). Returns
/// `[n_pairs * rows_per]` outputs or `None` (→ CPU fallback).
/// A prefetch request to the read-ahead worker: read `len` bytes at `off`
/// of `path` into the raw pinned buffer at `ptr`.
struct RaReq {
    key: (usize, usize),
    /// Pinned ring buffer index this read fills (echoed in the done message
    /// so completions may arrive out of order — QD2 protocol).
    buf: usize,
    ptr: usize,
    len: usize,
    path: std::path::PathBuf,
    off: u64,
}

/// Read-ahead pipeline for streamed tensors. A ring of fixed pinned buffers:
/// the worker thread fills upcoming buffers with the NEXT streamed tensors'
/// bytes (the streamed order repeats exactly every denoise step, learned on
/// step 0) while the current one feeds the htod — a depth > 1 keeps the disk
/// continuously busy instead of one-read-per-gap. The worker does file I/O
/// only — no CUDA — so there is no cross-thread context juggling; buffer
/// handoff is strictly dispatch → recv done (FIFO) → consume.
struct ReadAhead {
    tx: std::sync::mpsc::Sender<RaReq>,
    done_rx: std::sync::mpsc::Receiver<((usize, usize), usize, bool)>,
    /// Reads dispatched and not yet completed, keyed by tensor; the value is
    /// the pinned-buffer index. TWO workers complete OUT OF ORDER — the done
    /// message carries (key, buf, ok), so no FIFO assumption exists anywhere.
    in_flight: std::collections::HashMap<(usize, usize), usize>,
    /// Reads that completed (drained non-blocking) and await consumption —
    /// by a host htod or by the device-prefetch DMA.
    ready: std::collections::HashMap<(usize, usize), usize>,
    /// Pinned-buffer indices free for dispatch or synchronous use.
    free: Vec<usize>,
    /// Learned per-step order of streamed tensors: (key, path, offset, len).
    order: Vec<((usize, usize), std::path::PathBuf, u64, usize)>,
    learned: bool,
    /// Index into `order` (mod len) of the next tensor to dispatch.
    cursor: usize,
}

impl ReadAhead {
    fn new(n_bufs: usize) -> Self {
        let (tx, rx) = std::sync::mpsc::channel::<RaReq>();
        let rx = std::sync::Arc::new(std::sync::Mutex::new(rx));
        let (done_tx, done_rx) = std::sync::mpsc::channel::<((usize, usize), usize, bool)>();
        // TWO readers: laptop NVMe gains ~15-35% at QD2 for these 200-285 MB
        // sequential reads, and the Windows cache-manager copy parallelizes.
        for w in 0..2 {
            let rx = rx.clone();
            let done_tx = done_tx.clone();
            std::thread::Builder::new()
                .name(format!("dg-readahead-{w}"))
                .spawn(move || {
                    loop {
                        // Take one request; the lock drops BEFORE the
                        // blocking file I/O so the other worker can pull.
                        let req = {
                            let guard = match rx.lock() {
                                Ok(g) => g,
                                Err(_) => return,
                            };
                            match guard.recv() {
                                Ok(r) => r,
                                Err(_) => return,
                            }
                        };
                        // SAFETY: ptr/len name an exclusively-handed-off
                        // pinned buffer region; the dispatcher does not touch
                        // it until the done message for this key arrives.
                        let dst =
                            unsafe { std::slice::from_raw_parts_mut(req.ptr as *mut u8, req.len) };
                        let ok = std::fs::File::open(&req.path)
                            .and_then(|f| read_at_into(&f, req.off, dst))
                            .is_ok();
                        let _ = done_tx.send((req.key, req.buf, ok));
                    }
                })
                .expect("spawn dg-readahead");
        }
        Self {
            tx,
            done_rx,
            in_flight: std::collections::HashMap::new(),
            ready: std::collections::HashMap::new(),
            free: (0..n_bufs).collect(),
            order: Vec::new(),
            learned: false,
            cursor: 0,
        }
    }

    /// Route one done message: the buffer goes to `ready` on success or back
    /// to `free` on a failed read (the consumer falls back to a sync read).
    fn route_done(&mut self, key: (usize, usize), buf: usize, ok: bool) {
        self.in_flight.remove(&key);
        if ok {
            self.ready.insert(key, buf);
        } else {
            self.free.push(buf);
        }
    }

    /// Drain every in-flight read (blocking) and reclaim the buffers,
    /// including any completed-but-unconsumed `ready` entries.
    fn drain(&mut self) {
        while !self.in_flight.is_empty() {
            match self.done_rx.recv() {
                Ok((k, b, ok)) => self.route_done(k, b, ok),
                Err(_) => break,
            }
        }
        for (_, bi) in self.ready.drain() {
            self.free.push(bi);
        }
    }

    /// Non-blocking: move completed reads into `ready`.
    fn drain_ready(&mut self) {
        while let Ok((k, b, ok)) = self.done_rx.try_recv() {
            self.route_done(k, b, ok);
        }
    }

    /// Blocking: wait until `key`'s read completes (it must be in `ready` or
    /// `in_flight`). Returns the pinned buffer index, or None if the read
    /// failed / was never dispatched.
    fn wait_ready(&mut self, key: (usize, usize)) -> Option<usize> {
        self.drain_ready();
        if let Some(bi) = self.ready.remove(&key) {
            return Some(bi);
        }
        while self.in_flight.contains_key(&key) {
            match self.done_rx.recv() {
                Ok((k, b, ok)) => self.route_done(k, b, ok),
                Err(_) => return None,
            }
        }
        self.ready.remove(&key)
    }
}

/// ZERO-COPY tier: allocate DEVICE-MAPPED pinned host memory, fill it with
/// the tensor's bytes via one direct file read (falling back to the mmap
/// slice), and wrap the device-visible pointer for kernel arguments. GPU
/// kernels then read the tensor in place over PCIe — no staging buffer, no
/// per-step copy, and the disk is never touched again for this tensor.
///
/// The returned `CudaSlice` must NEVER be dropped (drop would `cuMemFree` a
/// HOST allocation): entries live in the engine static for the process.
fn host_map_tensor(
    ctx: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    src: (&std::path::Path, u64),
    tensor: &[u8],
) -> Result<CudaSlice<u8>, String> {
    use cudarc::driver::{result, sys};
    let need = tensor.len();
    ctx.bind_to_thread().map_err(|e| format!("bind ctx: {e}"))?;
    let hp = unsafe { result::malloc_host(need, sys::CU_MEMHOSTALLOC_DEVICEMAP) }
        .map_err(|e| format!("malloc_host(DEVICEMAP): {e}"))?;
    // SAFETY: hp names a fresh `need`-byte allocation owned here.
    let dst = unsafe { std::slice::from_raw_parts_mut(hp as *mut u8, need) };
    let read_ok = std::fs::File::open(src.0)
        .and_then(|f| read_at_into(&f, src.1, dst))
        .is_ok();
    if !read_ok {
        // One-time slow path: fault the bytes in from the mmap slice.
        dst.copy_from_slice(tensor);
    }
    let mut dptr: sys::CUdeviceptr = 0;
    unsafe { sys::cuMemHostGetDevicePointer_v2(&mut dptr, hp, 0) }
        .result()
        .map_err(|e| format!("host devptr: {e}"))?;
    // SAFETY: dptr is a valid device-visible mapping of `need` bytes.
    Ok(unsafe { stream.upgrade_device_ptr::<u8>(dptr, need) })
}

/// Positioned read into `dst` (Windows `seek_read` / Unix `read_exact_at`).
fn read_at_into(file: &std::fs::File, offset: u64, dst: &mut [u8]) -> std::io::Result<()> {
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        let mut done = 0usize;
        while done < dst.len() {
            let k = file.seek_read(&mut dst[done..], offset + done as u64)?;
            if k == 0 {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "tensor read hit EOF",
                ));
            }
            done += k;
        }
        Ok(())
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.read_exact_at(dst, offset)
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn fast_gemm_id(
    tensor: &[u8],
    src: (&std::path::Path, u64),
    kind: DgExpertKind,
    pair_base: &[i64],
    pair_pos: &[i32],
    rows_per: usize,
    blocks_per_row: usize,
    act_scales: &[f32],
    act_quants: &[i8],
) -> Option<Vec<f32>> {
    if gate_off() {
        return None;
    }
    let n_pairs = pair_base.len();
    if n_pairs == 0 || n_pairs != pair_pos.len() {
        return None;
    }
    let cell = ENGINE.get_or_init(|| Mutex::new(None));
    let mut guard = cell.lock().ok()?;
    if guard.is_none() {
        match build_engine() {
            Ok(e) => *guard = Some(e),
            Err(err) => {
                ENGINE_FAILED.store(true, std::sync::atomic::Ordering::Relaxed);
                eprintln!("[dg-cuda] engine build failed ({err}); CPU fallback");
                return None;
            }
        }
    }
    let eng = guard.as_mut()?;
    let key = (tensor.as_ptr() as usize, tensor.len());
    // Residency: pool if the budget allows (same accounting as the parity
    // expert pool), else stream through the scratch buffer.
    let mut streamed = false;
    let mut streamed_upload_done = false;
    // (consumed key, Some(pinned-buffer index) when a host buffer fed this
    // call's htod; None on a device-prefetch hit). Drives the post-run
    // pipeline refill and buffer recycling.
    let mut stream_consumed: Option<((usize, usize), Option<usize>)> = None;
    // Device-prefetch state for this call: the wire bytes may already sit in
    // scratch2 (or scratch) from the copy stream.
    let mut wire_in_scratch2 = false;
    let mut dev_hit = false;
    if !eng.expert_pool.contains_key(&key) && !eng.expert_rejected.contains(&key) {
        ensure_budgets(eng);
        let is_small = tensor.len() < DG_SMALL_TENSOR;
        let budget = if is_small {
            eng.small_budget.unwrap()
        } else {
            eng.expert_budget.unwrap()
        };
        if (tensor.len() as u64) <= budget {
            let s = eng.stream.clone();
            match s.alloc_zeros::<u8>(tensor.len()) {
                Ok(mut dev) => {
                    if s.memcpy_htod(tensor, &mut dev).is_ok() {
                        let left = budget - tensor.len() as u64;
                        if is_small {
                            eng.small_budget = Some(left);
                        } else {
                            eng.expert_budget = Some(left);
                        }
                        eng.expert_pool.insert(key, dev);
                    } else {
                        eng.expert_rejected.insert(key);
                    }
                }
                Err(_) => {
                    eng.expert_rejected.insert(key);
                }
            }
        } else {
            eng.expert_rejected.insert(key);
        }
    }
    // ZERO-COPY tier: tensors that miss the VRAM pool live pinned in
    // device-mapped host RAM — kernels read them in place over PCIe, and the
    // disk drops out of the steady-state loop entirely.
    if !eng.expert_pool.contains_key(&key)
        && !eng.host_pool.contains_key(&key)
        && !eng.host_rejected.contains(&key)
    {
        if eng.host_budget.is_none() {
            // Modest default: pinned pages compete with the model mmap and the
            // OS; over-pinning stalls cuMemHostAlloc for minutes before OOM.
            // The grouped-tile kernels make the in-place PCIe reads coalesced
            // and group-amortized, so a small pool still pays.
            let mb = std::env::var("CAMELID_DG_HOST_POOL_MB")
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(2560);
            eng.host_budget = Some(mb * 1024 * 1024);
            eprintln!(
                "[dg-cuda] zero-copy host pool budget {:.2} GiB",
                (mb * 1024 * 1024) as f64 / (1u64 << 30) as f64
            );
        }
        let budget = eng.host_budget.unwrap();
        if (tensor.len() as u64) <= budget {
            match host_map_tensor(&eng.ctx, &eng.stream, src, tensor) {
                Ok(dev) => {
                    eng.host_budget = Some(budget - tensor.len() as u64);
                    eng.host_pool.insert(key, dev);
                    if eng.host_pool.len() % 10 == 0 || budget - (tensor.len() as u64) < (1 << 28) {
                        eprintln!(
                            "[dg-cuda] zero-copy resident: {} tensors, {:.2} GiB host budget left",
                            eng.host_pool.len(),
                            eng.host_budget.unwrap() as f64 / (1u64 << 30) as f64
                        );
                    }
                }
                Err(e) => {
                    // A pin failure means the OS is already under lock pressure;
                    // further attempts each stall for seconds. Kill the tier.
                    eng.host_rejected.insert(key);
                    eng.host_budget = Some(0);
                    eprintln!("[dg-cuda] zero-copy map failed ({e}); tier disabled for this run");
                }
            }
        } else {
            eng.host_rejected.insert(key);
        }
    }
    if !eng.expert_pool.contains_key(&key) && !eng.host_pool.contains_key(&key) {
        streamed = true;
        let s = eng.stream.clone();
        let need = tensor.len();
        // Audit F6: the cap bail must precede any scratch (re)allocation — a
        // regrow drops a scratch a pending prefetch DMA may still write.
        if need > DG_PIN_CAP {
            eprintln!("[dg-cuda] streamed tensor exceeds pin cap; CPU fallback");
            return None;
        }
        // Fixed-cap scratch (the prefetch pipeline alternates scratch and
        // scratch2, so both must fit any streamed tensor). Allocated once;
        // never regrown (see the cap bail above).
        if eng.scratch.is_none() {
            match s.alloc_zeros::<u8>(DG_PIN_CAP) {
                Ok(b) => eng.scratch = Some(b),
                Err(e) => {
                    eprintln!("[dg-cuda] fast scratch alloc failed ({e}); CPU fallback");
                    return None;
                }
            }
        }
        // Stage through pinned memory, filled by a DIRECT positioned file
        // read (mmap demand paging under RAM pressure runs at random-fault
        // speed, ~0.6 GB/s observed; a sequential read + pinned DMA is far
        // faster). The read-ahead ring overlaps upcoming tensors' reads (the
        // dominant cost) with GPU work; the per-step sequence is learned on
        // step 0 and the pipeline refills after every call.
        while eng.pin_bufs.len() < DG_RA_BUFS {
            match unsafe { eng.ctx.alloc_pinned::<u8>(DG_PIN_CAP) } {
                Ok(b) => eng.pin_bufs.push(b),
                Err(e) => {
                    eprintln!("[dg-cuda] pinned ring alloc failed ({e}); CPU fallback");
                    return None;
                }
            }
        }
        if eng.ra.is_none() {
            eng.ra = Some(ReadAhead::new(DG_RA_BUFS));
        }
        if eng.copy_stream.is_none() {
            eng.copy_stream = eng.ctx.new_stream().ok();
        }
        // The device-prefetch target alternates against the scratch the
        // current call reads; both sized to the pin cap once.
        if eng.scratch2.is_none() {
            eng.scratch2 = eng.stream.alloc_zeros::<u8>(DG_PIN_CAP).ok();
        }
        // 0. Device-prefetch hit: the copy stream already DMA'd this tensor
        // into a device scratch during the PREVIOUS call's kernel — wait the
        // event (usually already done), release the pinned buffer, and skip
        // all host work for this tensor.
        if let Some((k, s2, ev, pinbuf)) = eng.dev_pf.take() {
            let _ = ev.synchronize();
            if let Some(ra) = eng.ra.as_mut() {
                ra.free.push(pinbuf);
            }
            if k == key {
                wire_in_scratch2 = s2;
                dev_hit = true;
                stream_consumed = Some((key, None));
            }
        }
        // 1. Resolve the host bytes: a completed read-ahead (non-blocking
        // drain, then targeted wait), else a synchronous read.
        let mut ready_buf: Option<usize> = None;
        {
            let ra = eng.ra.as_mut().unwrap();
            if !dev_hit {
                ra.drain_ready();
                if let Some(bi) = ra.ready.remove(&key) {
                    ready_buf = Some(bi);
                } else if ra.in_flight.iter().any(|(k, _)| *k == key) {
                    ready_buf = ra.wait_ready(key);
                } else if !ra.in_flight.is_empty() || !ra.ready.is_empty() {
                    // Mispredicted pipeline (block boundary / order change):
                    // reset it; the refill below restarts from this key.
                    ra.drain();
                }
            }
            // Learn the per-step sequence on the first pass.
            if !ra.learned {
                if ra.order.first().map(|(k, ..)| *k == key).unwrap_or(false) {
                    ra.learned = true;
                    ra.cursor = 1; // this call consumes order[0]
                    eprintln!(
                        "[dg-cuda] read-ahead learned {} streamed tensors/step",
                        ra.order.len()
                    );
                } else {
                    ra.order.push((key, src.0.to_path_buf(), src.1, need));
                }
            }
        }
        if !dev_hit {
            let sync_buf = if ready_buf.is_none() {
                let ra = eng.ra.as_mut().unwrap();
                let Some(bi) = ra.free.pop() else {
                    eprintln!("[dg-cuda] pinned ring exhausted; CPU fallback");
                    return None;
                };
                Some(bi)
            } else {
                None
            };
            let use_buf = ready_buf.or(sync_buf).unwrap();
            if ready_buf.is_none() {
                let read_ok = {
                    let pin = &mut eng.pin_bufs[use_buf];
                    match pin.as_mut_ptr() {
                        Ok(dst) => {
                            // SAFETY: DG_PIN_CAP >= need (checked above).
                            let dst_slice = unsafe { std::slice::from_raw_parts_mut(dst, need) };
                            std::fs::File::open(src.0)
                                .and_then(|f| read_at_into(&f, src.1, dst_slice))
                                .is_ok()
                        }
                        Err(_) => false,
                    }
                };
                if !read_ok {
                    // Last-resort: pageable upload straight from the mmap slice.
                    eng.ra.as_mut().unwrap().free.push(use_buf);
                    let buf = eng.scratch.as_mut().unwrap();
                    let mut view = buf.slice_mut(0..need);
                    if let Err(e) = s.memcpy_htod(tensor, &mut view) {
                        eprintln!("[dg-cuda] fast stream upload failed ({e}); CPU fallback");
                        return None;
                    }
                    streamed_upload_done = true;
                }
            }
            if !streamed_upload_done {
                let host = eng.pin_bufs[use_buf].as_slice().ok().map(|sl| &sl[..need]);
                let Some(host) = host else {
                    eprintln!("[dg-cuda] pinned staging unavailable; CPU fallback");
                    return None;
                };
                let buf = eng.scratch.as_mut().unwrap();
                let mut view = buf.slice_mut(0..need);
                if let Err(e) = s.memcpy_htod(host, &mut view) {
                    eprintln!("[dg-cuda] fast stream upload failed ({e}); CPU fallback");
                    return None;
                }
                stream_consumed = Some((key, Some(use_buf)));
            }
        }
    }
    let func = match kind {
        DgExpertKind::Q4K => &eng.q4k_tile_func,
        DgExpertKind::Q80 => &eng.q80_tile_func,
        DgExpertKind::Q50 => &eng.q50_tile_func,
        DgExpertKind::Q6K => &eng.q6k_tile_func,
    };
    // Shared-tile bytes per block: 32 rows x padded row stride (see kernels).
    let tile_shared: u32 = match kind {
        DgExpertKind::Q4K => 32 * 724,
        DgExpertKind::Q80 => 32 * 684,
        DgExpertKind::Q50 => 32 * 660,
        DgExpertKind::Q6K => 32 * 636,
    };
    // Group pairs by identical row window (host-stable sort by base): one
    // shared wire tile then serves up to DG_TILE_P pairs — the byte-traffic
    // divisor that makes the zero-copy tier viable and cuts VRAM traffic on
    // the MoE pair GEMMs.
    const TILE_P: usize = 8;
    let mut order: Vec<usize> = (0..n_pairs).collect();
    order.sort_by_key(|&i| pair_base[i]);
    let mut g_base: Vec<i64> = Vec::new();
    let mut g_pos: Vec<i32> = Vec::new();
    let mut g_orig: Vec<i32> = Vec::new();
    let mut i = 0usize;
    while i < n_pairs {
        let b = pair_base[order[i]];
        g_base.push(b);
        let mut slot = 0usize;
        while slot < TILE_P && i < n_pairs && pair_base[order[i]] == b {
            g_pos.push(pair_pos[order[i]]);
            g_orig.push(order[i] as i32);
            i += 1;
            slot += 1;
        }
        while slot < TILE_P {
            g_pos.push(-1);
            g_orig.push(-1);
            slot += 1;
        }
    }
    let n_groups = g_base.len();
    // Device-prefetch plan: if the NEXT streamed tensor's host bytes are
    // already read (non-blocking check), its DMA is enqueued on the copy
    // stream right after this call's kernel launch — the transfer hides
    // under the kernel + dtoh instead of serializing before the next one.
    let mut pf_plan: Option<((usize, usize), bool, usize, usize)> = None;
    if streamed && eng.copy_stream.is_some() && eng.scratch2.is_some() {
        let next: Option<((usize, usize), usize)> = {
            let ra = eng.ra.as_mut().unwrap();
            ra.drain_ready();
            if ra.learned && !ra.order.is_empty() {
                ra.order.iter().position(|(k, ..)| *k == key).and_then(|i| {
                    let len_o = ra.order.len();
                    (1..len_o).find_map(|step| {
                        let (nk, _, _, nlen) = &ra.order[(i + step) % len_o];
                        if ra.ready.contains_key(nk) && *nlen <= DG_PIN_CAP {
                            Some((*nk, *nlen))
                        } else {
                            None
                        }
                    })
                })
            } else {
                None
            }
        };
        if let Some((nk, nlen)) = next {
            if !eng.expert_pool.contains_key(&nk) && !eng.host_pool.contains_key(&nk) {
                if let Some(pb) = eng.ra.as_mut().unwrap().ready.remove(&nk) {
                    // Target the scratch the CURRENT call is NOT reading.
                    pf_plan = Some((nk, !wire_in_scratch2, pb, nlen));
                }
            }
        }
    }
    // Scratches leave the engine for the call so the wire read (immutable)
    // and the prefetch DMA target (mutable) borrow disjoint locals.
    let mut scratch_a = eng.scratch.take();
    let mut scratch_b = eng.scratch2.take();
    let (wire_scratch, pf_dst) = if wire_in_scratch2 {
        (scratch_b.as_ref(), scratch_a.as_mut())
    } else {
        (scratch_a.as_ref(), scratch_b.as_mut())
    };
    // Prefetch outcome slots: written inside run(), consumed after — on BOTH
    // success and error paths (audit F3).
    let mut pf_out_slot: Option<((usize, usize), bool, CudaEvent, usize)> = None;
    let mut pf_fail_slot: Option<((usize, usize), usize)> = None;
    #[allow(unused_mut)]
    let mut run = || -> Result<Vec<f32>, String> {
        let s = eng.stream.clone();
        let mut sc_dev = s
            .alloc_zeros::<f32>(act_scales.len())
            .map_err(|e| format!("alloc act scales: {e}"))?;
        s.memcpy_htod(act_scales, &mut sc_dev)
            .map_err(|e| format!("upload act scales: {e}"))?;
        let mut q_dev = s
            .alloc_zeros::<i8>(act_quants.len())
            .map_err(|e| format!("alloc act quants: {e}"))?;
        s.memcpy_htod(act_quants, &mut q_dev)
            .map_err(|e| format!("upload act quants: {e}"))?;
        let mut base_dev = s
            .alloc_zeros::<i64>(n_groups)
            .map_err(|e| format!("alloc group base: {e}"))?;
        s.memcpy_htod(&g_base, &mut base_dev)
            .map_err(|e| format!("upload group base: {e}"))?;
        let mut pos_dev = s
            .alloc_zeros::<i32>(g_pos.len())
            .map_err(|e| format!("alloc group pos: {e}"))?;
        s.memcpy_htod(&g_pos, &mut pos_dev)
            .map_err(|e| format!("upload group pos: {e}"))?;
        let mut orig_dev = s
            .alloc_zeros::<i32>(g_orig.len())
            .map_err(|e| format!("alloc group orig: {e}"))?;
        s.memcpy_htod(&g_orig, &mut orig_dev)
            .map_err(|e| format!("upload group orig: {e}"))?;
        let mut out_dev = s
            .alloc_zeros::<f32>(n_pairs * rows_per)
            .map_err(|e| format!("alloc out: {e}"))?;
        let cfg = LaunchConfig {
            grid_dim: ((rows_per as u32).div_ceil(32), n_groups as u32, 1),
            block_dim: (32, TILE_P as u32, 1),
            shared_mem_bytes: tile_shared,
        };
        let (rp, bp) = (rows_per as i32, blocks_per_row as i32);
        let mut b = s.launch_builder(func);
        if streamed {
            b.arg(wire_scratch.ok_or_else(|| "scratch missing".to_string())?);
        } else if let Some(dev) = eng.expert_pool.get(&key) {
            b.arg(dev);
        } else {
            // Zero-copy tier: the kernel reads device-mapped host RAM in place.
            b.arg(eng.host_pool.get(&key).unwrap());
        }
        b.arg(&sc_dev)
            .arg(&q_dev)
            .arg(&base_dev)
            .arg(&pos_dev)
            .arg(&orig_dev)
            .arg(&rp)
            .arg(&bp)
            .arg(&mut out_dev);
        unsafe { b.launch(cfg) }.map_err(|e| format!("launch: {e}"))?;
        // Enqueue the planned device prefetch on the copy stream — it runs
        // concurrently with this call's kernel and dtoh. The target scratch
        // is the one the current kernel is NOT reading, and its previous
        // reader fully synced, so the async write cannot race. Audit F3/F4:
        // the outcome is written to CAPTURED SLOTS (not the return value) so
        // an error later in this closure still installs the tracking event —
        // an enqueued DMA must never be left orphaned; and an event failure
        // after a successful enqueue synchronizes the copy stream before the
        // buffer is handed back.
        if let Some((nk, into_s2, pb, nlen)) = pf_plan.take() {
            let cs = eng.copy_stream.as_ref().unwrap().clone();
            let host = eng.pin_bufs[pb].as_slice().ok();
            let enq = match (host, pf_dst) {
                (Some(host), Some(dst)) => {
                    let mut view = dst.slice_mut(0..nlen);
                    cs.memcpy_htod(&host[..nlen], &mut view).is_ok()
                }
                _ => false,
            };
            if enq {
                if let Ok(ev) = eng.ctx.new_event(None) {
                    if ev.record(&cs).is_ok() {
                        pf_out_slot = Some((nk, into_s2, ev, pb));
                    }
                }
            }
            if pf_out_slot.is_none() {
                if enq {
                    // The DMA is in flight with no event: drain it before the
                    // buffer can be reused (F4).
                    let _ = cs.synchronize();
                }
                pf_fail_slot = Some((nk, pb));
            }
        }
        let mut out = vec![0f32; n_pairs * rows_per];
        s.memcpy_dtoh(&out_dev, &mut out)
            .map_err(|e| format!("download: {e}"))?;
        // Main-stream sync only: the copy stream's prefetch keeps running
        // past this call and is waited via its event at consumption.
        s.synchronize().map_err(|e| format!("sync: {e}"))?;
        Ok(out)
    };
    let result = run();
    // Restore the scratches taken for the call, then unpack the prefetch
    // outcome. All bookkeeping below runs on BOTH success and failure paths
    // (audit F1-F4): buffers must always return to a tracked set.
    eng.scratch = scratch_a;
    eng.scratch2 = scratch_b;
    if let Some((nk, pb)) = pf_fail_slot.take() {
        if let Some(ra) = eng.ra.as_mut() {
            ra.ready.insert(nk, pb);
        }
    }
    if pf_out_slot.is_some() {
        eng.dev_pf = pf_out_slot.take();
    }
    // Audit F2: a plan consumed from `ready` but never enqueued (run() erred
    // before the enqueue point) must be reinstated.
    if let Some((nk, _, pb, _)) = pf_plan.take() {
        if let Some(ra) = eng.ra.as_mut() {
            ra.ready.insert(nk, pb);
        }
    }
    // Audit F1: the consumed host buffer returns to the free ring even when
    // run() failed — otherwise each transient failure permanently eats one
    // of the 4 ring buffers and the stream silently degrades to CPU.
    if let Some((_, Some(bi))) = stream_consumed {
        if let Some(ra) = eng.ra.as_mut() {
            ra.free.push(bi);
        }
    }
    // Post-sync prefetch dispatch: refill the read-ahead pipeline for
    // upcoming streamed tensors in the learned per-step order. The wrap at
    // the sequence end prefetches the next STEP's first tensor across the
    // attention/lm_head/sampler gap.
    if result.is_ok() {
        if let Some((ck, _)) = stream_consumed {
            // Refill the pipeline: dispatch reads for upcoming streamed
            // tensors (skipping pool-resident entries) while free buffers
            // remain. The cursor wraps, so the sequence end prefetches the
            // next STEP's first tensors across the attention/lm_head gap.
            let learned = eng.ra.as_ref().map(|ra| ra.learned).unwrap_or(false);
            if learned {
                // Keep the cursor ahead of the just-consumed entry.
                {
                    let ra = eng.ra.as_mut().unwrap();
                    if let Some(i) = ra.order.iter().position(|(k, ..)| *k == ck) {
                        let len = ra.order.len();
                        let ahead = ra.in_flight.len();
                        // cursor must sit `ahead` entries past i+1's stream
                        // position at minimum; a simple monotonic bump keeps
                        // it consistent because consumption follows order.
                        if ahead == 0 {
                            ra.cursor = (i + 1) % len;
                        }
                    }
                }
                let order_len = eng.ra.as_ref().unwrap().order.len();
                let mut scanned = 0;
                while scanned < order_len {
                    let ra = eng.ra.as_ref().unwrap();
                    if ra.free.is_empty() {
                        break;
                    }
                    let (nk, npath, noff, nlen) = ra.order[ra.cursor % order_len].clone();
                    scanned += 1;
                    let already_staged = {
                        let ra = eng.ra.as_ref().unwrap();
                        ra.in_flight.iter().any(|(k, _)| *k == nk)
                            || ra.ready.contains_key(&nk)
                            || eng.dev_pf.as_ref().map(|(k, ..)| *k == nk).unwrap_or(false)
                    };
                    if eng.expert_pool.contains_key(&nk)
                        || eng.host_pool.contains_key(&nk)
                        || nlen == 0
                        || nlen > DG_PIN_CAP
                        || already_staged
                    {
                        eng.ra.as_mut().unwrap().cursor += 1;
                        continue;
                    }
                    let ra = eng.ra.as_mut().unwrap();
                    let bi = ra.free.pop().unwrap();
                    let ptr = match eng.pin_bufs[bi].as_mut_ptr() {
                        Ok(p) => p as usize,
                        Err(_) => {
                            eng.ra.as_mut().unwrap().free.push(bi);
                            break;
                        }
                    };
                    let ra = eng.ra.as_mut().unwrap();
                    if ra
                        .tx
                        .send(RaReq {
                            key: nk,
                            buf: bi,
                            ptr,
                            len: nlen,
                            path: npath,
                            off: noff,
                        })
                        .is_ok()
                    {
                        ra.in_flight.insert(nk, bi);
                        ra.cursor += 1;
                    } else {
                        ra.free.push(bi);
                        break;
                    }
                }
            }
        }
    }
    match result {
        Ok(v) => Some(v),
        Err(err) => {
            eprintln!("[dg-cuda] fast gemm failed ({err}); CPU fallback");
            None
        }
    }
}

/// FAST-mode bidirectional diffusion attention on the GPU (see the `dg_attn`
/// kernel). `q` is `[n*heads*hd]`, `k`/`v` are `[n*kv_heads*hd]` (post
/// norm+RoPE, f32). Returns the pre-`attn_output` mix `[n*heads*hd]`, or
/// `None` on any failure (→ CPU fallback).
#[allow(clippy::too_many_arguments)]
pub(crate) fn dg_attention_gpu(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n: usize,
    nq: usize,
    q_off: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    group: usize,
    p: usize,
    win: usize,
    sliding: bool,
    canvas_prompt_lo: i64,
) -> Option<Vec<f32>> {
    if gate_off() {
        return None;
    }
    let cell = ENGINE.get_or_init(|| Mutex::new(None));
    let mut guard = cell.lock().ok()?;
    if guard.is_none() {
        match build_engine() {
            Ok(e) => *guard = Some(e),
            Err(err) => {
                ENGINE_FAILED.store(true, std::sync::atomic::Ordering::Relaxed);
                eprintln!("[dg-cuda] engine build failed ({err}); CPU fallback");
                return None;
            }
        }
    }
    let eng = guard.as_mut()?;
    let run = || -> Result<Vec<f32>, String> {
        let s = eng.stream.clone();
        let mut q_dev = s
            .alloc_zeros::<f32>(q.len())
            .map_err(|e| format!("alloc q: {e}"))?;
        s.memcpy_htod(q, &mut q_dev)
            .map_err(|e| format!("upload q: {e}"))?;
        let mut k_dev = s
            .alloc_zeros::<f32>(k.len())
            .map_err(|e| format!("alloc k: {e}"))?;
        s.memcpy_htod(k, &mut k_dev)
            .map_err(|e| format!("upload k: {e}"))?;
        let mut v_dev = s
            .alloc_zeros::<f32>(v.len())
            .map_err(|e| format!("alloc v: {e}"))?;
        s.memcpy_htod(v, &mut v_dev)
            .map_err(|e| format!("upload v: {e}"))?;
        let mut out_dev = s
            .alloc_zeros::<f32>(nq * heads * head_dim)
            .map_err(|e| format!("alloc attn out: {e}"))?;
        let cfg = LaunchConfig {
            grid_dim: ((nq * heads) as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: (n * 4) as u32,
        };
        let (ni, nqi, qo, hi, kvi, hdi, gi, pi, wi, sl) = (
            n as i32,
            nq as i32,
            q_off as i32,
            heads as i32,
            kv_heads as i32,
            head_dim as i32,
            group as i32,
            p as i32,
            win as i32,
            sliding as i32,
        );
        let mut b = s.launch_builder(&eng.attn_func);
        b.arg(&q_dev)
            .arg(&k_dev)
            .arg(&v_dev)
            .arg(&mut out_dev)
            .arg(&ni)
            .arg(&nqi)
            .arg(&qo)
            .arg(&hi)
            .arg(&kvi)
            .arg(&hdi)
            .arg(&gi)
            .arg(&pi)
            .arg(&wi)
            .arg(&sl)
            .arg(&canvas_prompt_lo);
        unsafe { b.launch(cfg) }.map_err(|e| format!("launch attn: {e}"))?;
        let mut out = vec![0f32; nq * heads * head_dim];
        s.memcpy_dtoh(&out_dev, &mut out)
            .map_err(|e| format!("download attn: {e}"))?;
        eng.ctx.synchronize().map_err(|e| format!("sync: {e}"))?;
        Ok(out)
    };
    match run() {
        Ok(out) => Some(out),
        Err(err) => {
            eprintln!("[dg-cuda] attention failed ({err}); CPU fallback");
            None
        }
    }
}

/// FAST-mode lm_head over the canvas rows: `logits[pos][v] = Σ_e h[pos][e] ×
/// emb_t[e][v]` as a tiled f32×f16 GEMM against the SC stage's RESIDENT
/// transposed embedding (uploaded once, shared with `sc_soft_embedding_gpu`
/// via the same cache slot). Not bit-exact vs the Q6_K integer dot (f16
/// weights, f32 accumulation) — FAST mode only.
pub(crate) fn lm_head_f16_gemm_gpu(
    emb_t: &[u16],
    h_canvas: &[f32],
    c: usize,
    hidden: usize,
    n_vocab: usize,
    softcap: f32,
) -> Option<Vec<f32>> {
    if gate_off() {
        return None;
    }
    debug_assert_eq!(emb_t.len(), hidden * n_vocab);
    debug_assert_eq!(h_canvas.len(), c * hidden);
    let cell = ENGINE.get_or_init(|| Mutex::new(None));
    let mut guard = cell.lock().ok()?;
    if guard.is_none() {
        match build_engine() {
            Ok(e) => *guard = Some(e),
            Err(err) => {
                ENGINE_FAILED.store(true, std::sync::atomic::Ordering::Relaxed);
                eprintln!("[dg-cuda] engine build failed ({err}); CPU fallback");
                return None;
            }
        }
    }
    let eng = guard.as_mut()?;
    let mut run = || -> Result<(Vec<f32>, CudaSlice<f32>), String> {
        let s = eng.stream.clone();
        let key = (emb_t.as_ptr() as usize, emb_t.len());
        if eng.sc_emb.as_ref().map(|(_, k)| *k != key).unwrap_or(true) {
            let mut dev = s
                .alloc_zeros::<u16>(emb_t.len())
                .map_err(|e| format!("alloc emb_t: {e}"))?;
            s.memcpy_htod(emb_t, &mut dev)
                .map_err(|e| format!("upload emb_t: {e}"))?;
            eng.sc_emb = Some((dev, key));
        }
        let emb_dev = &eng.sc_emb.as_ref().unwrap().0;
        let mut a_dev = s
            .alloc_zeros::<f32>(h_canvas.len())
            .map_err(|e| format!("alloc h: {e}"))?;
        s.memcpy_htod(h_canvas, &mut a_dev)
            .map_err(|e| format!("upload h: {e}"))?;
        let mut c_dev = s
            .alloc_zeros::<f32>(c * n_vocab)
            .map_err(|e| format!("alloc logits: {e}"))?;
        let cfg = LaunchConfig {
            grid_dim: ((n_vocab as u32).div_ceil(16), (c as u32).div_ceil(16), 1),
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0,
        };
        let (mi, ki, ni) = (c as i32, hidden as i32, n_vocab as i32);
        let mut b = s.launch_builder(&eng.lm_gemm_func);
        b.arg(&a_dev)
            .arg(emb_dev)
            .arg(&mut c_dev)
            .arg(&mi)
            .arg(&ki)
            .arg(&ni)
            .arg(&softcap);
        unsafe { b.launch(cfg) }.map_err(|e| format!("launch lm gemm: {e}"))?;
        let mut out = vec![0f32; c * n_vocab];
        s.memcpy_dtoh(&c_dev, &mut out)
            .map_err(|e| format!("download logits: {e}"))?;
        eng.ctx.synchronize().map_err(|e| format!("sync: {e}"))?;
        Ok((out, c_dev))
    };
    match run() {
        Ok((v, c_dev)) => {
            // Leave the logits device-resident for the next step's fused SC.
            eng.last_logits = Some((c_dev, c, n_vocab));
            Some(v)
        }
        Err(err) => {
            eprintln!("[dg-cuda] lm gemm failed ({err}); CPU fallback");
            None
        }
    }
}

/// Generation-boundary reset (audit F5): a stale `last_logits` buffer from a
/// PREVIOUS generation must never feed a new one's self-conditioning — step 0
/// skips SC, so a transient lm_head fallback at step 0 would otherwise leave
/// the old buffer alive for step 1 to consume (silent cross-request
/// corruption in serve). Call at every `eb_generate` entry.
pub(crate) fn dg_generation_reset() {
    if let Some(cell) = ENGINE.get() {
        if let Ok(mut guard) = cell.lock() {
            if let Some(eng) = guard.as_mut() {
                eng.last_logits = None;
            }
        }
    }
}

/// FAST-mode fused SC soft-embedding: consume the device-resident previous
/// lm_head logits (one-shot), compute f16 softmax probs on the GPU, and run
/// the resident-embedding soft matmul — no host softmax, no probs upload.
/// `None` → the plain `sc_soft_embedding_gpu` / CPU paths.
pub(crate) fn sc_soft_fused_gpu(
    temp_inv: f32,
    embed_scale: f32,
    c: usize,
    hidden: usize,
    n_vocab: usize,
) -> Option<Vec<f32>> {
    if gate_off() || std::env::var("CAMELID_DG_CUDA_SC").as_deref() == Ok("0") {
        return None;
    }
    let cell = ENGINE.get_or_init(|| Mutex::new(None));
    let mut guard = cell.lock().ok()?;
    let eng = guard.as_mut()?;
    let (logits_dev, lc, lv) = eng.last_logits.take()?;
    if lc != c || lv != n_vocab {
        return None;
    }
    // The resident embedding must already be uploaded (the lm_head GEMM and
    // SC share the slot; it is by the time SC runs).
    let (emb_dev_len, _) = eng.sc_emb.as_ref().map(|(d, k)| (d.len(), k))?;
    if emb_dev_len != hidden * n_vocab {
        return None;
    }
    let run = || -> Result<Vec<f32>, String> {
        let s = eng.stream.clone();
        let mut probs_dev = s
            .alloc_zeros::<u16>(c * n_vocab)
            .map_err(|e| format!("alloc probs: {e}"))?;
        let cfg = LaunchConfig {
            grid_dim: (c as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let nv = n_vocab as i32;
        let mut b = s.launch_builder(&eng.sc_probs_func);
        b.arg(&logits_dev)
            .arg(&mut probs_dev)
            .arg(&nv)
            .arg(&temp_inv);
        unsafe { b.launch(cfg) }.map_err(|e| format!("launch sc probs: {e}"))?;
        let emb_dev = &eng.sc_emb.as_ref().unwrap().0;
        let mut soft_dev = s
            .alloc_zeros::<f32>(c * hidden)
            .map_err(|e| format!("alloc soft: {e}"))?;
        let cfg2 = LaunchConfig {
            grid_dim: ((c as u32).div_ceil(16), (hidden as u32).div_ceil(16), 1),
            block_dim: (16, 16, 1),
            shared_mem_bytes: 0,
        };
        let (h, cc) = (hidden as i32, c as i32);
        let mut b = s.launch_builder(&eng.sc_gemm_func);
        b.arg(emb_dev)
            .arg(&probs_dev)
            .arg(&mut soft_dev)
            .arg(&h)
            .arg(&nv)
            .arg(&cc)
            .arg(&embed_scale);
        unsafe { b.launch(cfg2) }.map_err(|e| format!("launch sc gemm: {e}"))?;
        let mut out = vec![0f32; c * hidden];
        s.memcpy_dtoh(&soft_dev, &mut out)
            .map_err(|e| format!("download soft: {e}"))?;
        eng.ctx.synchronize().map_err(|e| format!("sync: {e}"))?;
        Ok(out)
    };
    match run() {
        Ok(v) => Some(v),
        Err(err) => {
            eprintln!("[dg-cuda] fused sc failed ({err}); CPU fallback");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn xorshift(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    /// GPU expert kernels vs the CPU reference dots on random data — the GPU
    /// output must be BIT-IDENTICAL per row (same integer dots, same fused f32
    /// order). Skips when no CUDA device is usable.
    #[test]
    fn expert_gemv_gpu_bit_identical_to_cpu() {
        if gate_off() {
            eprintln!("skipping: CUDA unavailable or gated off");
            return;
        }
        let mut state: u64 = 0x0dd0_57a7_e5ee_d001;

        // ---- Q4_K x Q8_K ----
        let (rows, bpr) = (13usize, 3usize);
        let mut wire = vec![0u8; rows * bpr * 144];
        for b in wire.iter_mut() {
            *b = (xorshift(&mut state) & 0xff) as u8;
        }
        // Keep the f16 d/dmin scales finite: real quantized weights never
        // carry Inf/NaN scales, and NaN payloads canonicalize differently on
        // GPU vs x86 (payload bits, not math, would fail the comparison).
        for sb in 0..rows * bpr {
            wire[sb * 144 + 1] &= 0x3f;
            wire[sb * 144 + 3] &= 0x3f;
        }
        let mut blocks = Vec::with_capacity(bpr);
        let mut scales = vec![0f32; bpr];
        let mut quants = vec![0i8; bpr * 256];
        for i in 0..bpr {
            let mut qs = [0i8; 256];
            for q in qs.iter_mut() {
                *q = (xorshift(&mut state) & 0xff) as u8 as i8;
            }
            let d = (xorshift(&mut state) % 1000) as f32 / 333.0 + 0.001;
            scales[i] = d;
            quants[i * 256..(i + 1) * 256].copy_from_slice(&qs);
            blocks.push(crate::inference::Q8KBlock { d, qs });
        }
        // full range and a sub-range (first_row exercised)
        for (first, n) in [(0usize, rows), (5usize, 4usize)] {
            let gpu =
                expert_rows_gemv_gpu(&wire, DgExpertKind::Q4K, first, n, bpr, &scales, &quants)
                    .expect("gpu q4k gemv");
            for r in 0..n {
                let row = &wire[(first + r) * bpr * 144..(first + r + 1) * bpr * 144];
                let cpu = super::super::refmath::q4_k_dot_arm(row, &blocks);
                assert_eq!(
                    cpu.to_bits(),
                    gpu[r].to_bits(),
                    "q4k row {} (first={first}): cpu {cpu} gpu {}",
                    first + r,
                    gpu[r]
                );
            }
        }

        // ---- Q8_0 x Q8_0 ----
        let (rows, nb) = (11usize, 4usize);
        let mut wire = vec![0u8; rows * nb * 34];
        for b in wire.iter_mut() {
            *b = (xorshift(&mut state) & 0xff) as u8;
        }
        // Finite f16 d scales, as above.
        for blk in 0..rows * nb {
            wire[blk * 34 + 1] &= 0x3f;
        }
        let mut q80 = Vec::with_capacity(nb);
        let mut scales = vec![0f32; nb];
        let mut quants = vec![0i8; nb * 32];
        for i in 0..nb {
            let mut qv = [0i8; 32];
            for q in qv.iter_mut() {
                *q = (xorshift(&mut state) & 0xff) as u8 as i8;
            }
            let s = (xorshift(&mut state) % 1000) as f32 / 777.0 + 0.002;
            scales[i] = s;
            quants[i * 32..(i + 1) * 32].copy_from_slice(&qv);
            q80.push(crate::tensor::Q8_0Block {
                scale: s,
                quants: qv,
            });
        }
        for (first, n) in [(0usize, rows), (3usize, 6usize)] {
            let gpu =
                expert_rows_gemv_gpu(&wire, DgExpertKind::Q80, first, n, nb, &scales, &quants)
                    .expect("gpu q8_0 gemv");
            for r in 0..n {
                let row = &wire[(first + r) * nb * 34..(first + r + 1) * nb * 34];
                let cpu = super::super::refmath::q8_0_dot_arm(row, &q80);
                assert_eq!(
                    cpu.to_bits(),
                    gpu[r].to_bits(),
                    "q8_0 row {} (first={first}): cpu {cpu} gpu {}",
                    first + r,
                    gpu[r]
                );
            }
        }
    }
}

#[cfg(test)]
mod fast_tests {
    use super::*;

    fn xorshift(state: &mut u64) -> u64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        *state
    }

    /// FAST batched-ID kernels vs the CPU reference dots on random data with
    /// random (base, pos) pairs — per-row bit-identical (the kernels mirror
    /// the scalar reductions; fast mode's non-bit-exact caveats are about
    /// GEMM batching effects elsewhere, not these kernels). Skips without CUDA.
    #[test]
    fn fast_gemm_id_matches_cpu_dots() {
        if gate_off() {
            eprintln!("skipping: CUDA unavailable or gated off");
            return;
        }
        let mut state: u64 = 0xfa57_f00d_5eed_0001;
        let dummy = std::path::Path::new("unused-resident-path");

        // 3 "experts" x 5 rows each, 2 activations, 4 random pairs.
        let (n_exp, rows_per, n_acts) = (3usize, 5usize, 2usize);
        let total_rows = n_exp * rows_per;
        let pairs: Vec<(i64, i32)> = vec![(0, 0), (5, 1), (10, 0), (5, 0)];
        let base: Vec<i64> = pairs.iter().map(|p| p.0).collect();
        let pos: Vec<i32> = pairs.iter().map(|p| p.1).collect();

        // ---- Q4_K (bpr superblocks of 256) ----
        let bpr = 2usize;
        let mut wire = vec![0u8; total_rows * bpr * 144];
        for b in wire.iter_mut() {
            *b = (xorshift(&mut state) & 0xff) as u8;
        }
        for sb in 0..total_rows * bpr {
            wire[sb * 144 + 1] &= 0x3f;
            wire[sb * 144 + 3] &= 0x3f;
        }
        let mut blocks: Vec<Vec<crate::inference::Q8KBlock>> = Vec::new();
        let mut scales = vec![0f32; n_acts * bpr];
        let mut quants = vec![0i8; n_acts * bpr * 256];
        for a in 0..n_acts {
            let mut act = Vec::new();
            for b in 0..bpr {
                let mut qs = [0i8; 256];
                for q in qs.iter_mut() {
                    *q = (xorshift(&mut state) & 0xff) as u8 as i8;
                }
                let d = (xorshift(&mut state) % 1000) as f32 / 333.0 + 0.001;
                scales[a * bpr + b] = d;
                quants[(a * bpr + b) * 256..(a * bpr + b + 1) * 256].copy_from_slice(&qs);
                act.push(crate::inference::Q8KBlock { d, qs });
            }
            blocks.push(act);
        }
        let out = fast_gemm_id(
            &wire,
            (dummy, 0),
            DgExpertKind::Q4K,
            &base,
            &pos,
            rows_per,
            bpr,
            &scales,
            &quants,
        )
        .expect("q4k id gemm");
        for (pi, &(b, a)) in pairs.iter().enumerate() {
            for r in 0..rows_per {
                let row_i = b as usize + r;
                let row = &wire[row_i * bpr * 144..(row_i + 1) * bpr * 144];
                let cpu = super::super::refmath::q4_k_dot_arm(row, &blocks[a as usize]);
                assert_eq!(
                    cpu.to_bits(),
                    out[pi * rows_per + r].to_bits(),
                    "q4k pair {pi} row {r}"
                );
            }
        }

        // ---- Q8_0 and Q5_0 (nb 32-value blocks; shared activation form) ----
        let nb = 4usize;
        let mut q80_acts: Vec<Vec<crate::tensor::Q8_0Block>> = Vec::new();
        let mut scales = vec![0f32; n_acts * nb];
        let mut quants = vec![0i8; n_acts * nb * 32];
        for a in 0..n_acts {
            let mut act = Vec::new();
            for b in 0..nb {
                let mut qv = [0i8; 32];
                for q in qv.iter_mut() {
                    *q = (xorshift(&mut state) & 0xff) as u8 as i8;
                }
                let s = (xorshift(&mut state) % 1000) as f32 / 777.0 + 0.002;
                scales[a * nb + b] = s;
                quants[(a * nb + b) * 32..(a * nb + b + 1) * 32].copy_from_slice(&qv);
                act.push(crate::tensor::Q8_0Block {
                    scale: s,
                    quants: qv,
                });
            }
            q80_acts.push(act);
        }
        for (kind, wire_bytes, name) in [
            (DgExpertKind::Q80, 34usize, "q8_0"),
            (DgExpertKind::Q50, 22usize, "q5_0"),
        ] {
            let mut wire = vec![0u8; total_rows * nb * wire_bytes];
            for b in wire.iter_mut() {
                *b = (xorshift(&mut state) & 0xff) as u8;
            }
            for blk in 0..total_rows * nb {
                wire[blk * wire_bytes + 1] &= 0x3f;
            }
            let out = fast_gemm_id(
                &wire,
                (dummy, 0),
                kind,
                &base,
                &pos,
                rows_per,
                nb,
                &scales,
                &quants,
            )
            .unwrap_or_else(|| panic!("{name} id gemm"));
            for (pi, &(b, a)) in pairs.iter().enumerate() {
                for r in 0..rows_per {
                    let row_i = b as usize + r;
                    let row = &wire[row_i * nb * wire_bytes..(row_i + 1) * nb * wire_bytes];
                    let cpu = match kind {
                        DgExpertKind::Q80 => {
                            super::super::refmath::q8_0_dot_arm(row, &q80_acts[a as usize])
                        }
                        DgExpertKind::Q50 => {
                            super::super::refmath::q5_0_dot_arm(row, &q80_acts[a as usize])
                        }
                        DgExpertKind::Q4K | DgExpertKind::Q6K => unreachable!(),
                    };
                    assert_eq!(
                        cpu.to_bits(),
                        out[pi * rows_per + r].to_bits(),
                        "{name} pair {pi} row {r}"
                    );
                }
            }
        }
    }
}
