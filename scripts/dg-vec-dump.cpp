// dg-vec-dump — exact ggml NEON kernel ground truth for the DiffusionGemma
// lane's Phase 5 attention-reduction debugging. Reproduces, verbatim from
// the pinned vec.cpp / simd-mappings.h (ARM NEON aarch64 branch),
// ggml_vec_dot_f32 and ggml_vec_soft_max_f32 — the two reductions the
// attention KQV and softmax run over the n_kv dimension — and dumps results
// at several lengths (incl. the hello n_kv=273 and the story n_kv=297) for
// deterministic inputs. Kernel code (c) the llama.cpp / ggml authors.
//
// Build: c++ -std=c++17 -O2 -mcpu=native scripts/dg-vec-dump.cpp -o dg-vec-dump
// Run:   dg-vec-dump <out_dir>
// Emits: vecdot-<n>.bin (x[n] f32, y[n] f32, result f32)
//        softmax-<n>.bin (x[n] f32, max f32, out[n] f32, sum f64)

#include <arm_neon.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// deterministic LCG in [-1, 1)
static uint32_t st = 0x12345u;
static float rnd() {
    st = st * 1664525u + 1013904223u;
    return ((float) (st >> 8) / (float) (1u << 24)) * 2.0f - 1.0f;
}

// ---- ggml_v_expf, NEON (vec.h) ----
static inline float32x4_t ggml_v_expf(float32x4_t x) {
    const float32x4_t r = vdupq_n_f32(0x1.8p23f);
    const float32x4_t z = vfmaq_f32(r, x, vdupq_n_f32(0x1.715476p+0f));
    const float32x4_t n = vsubq_f32(z, r);
    const float32x4_t b =
        vfmsq_f32(vfmsq_f32(x, n, vdupq_n_f32(0x1.62e4p-1f)), n, vdupq_n_f32(0x1.7f7d1cp-20f));
    const uint32x4_t e = vshlq_n_u32(vreinterpretq_u32_f32(z), 23);
    const float32x4_t k = vreinterpretq_f32_u32(vaddq_u32(e, vreinterpretq_u32_f32(vdupq_n_f32(1))));
    const uint32x4_t c = vcagtq_f32(n, vdupq_n_f32(126));
    const float32x4_t u = vmulq_f32(b, b);
    const float32x4_t j = vfmaq_f32(
        vmulq_f32(vdupq_n_f32(0x1.ffffecp-1f), b),
        vfmaq_f32(vfmaq_f32(vdupq_n_f32(0x1.fffdb6p-2f), vdupq_n_f32(0x1.555e66p-3f), b),
                  vfmaq_f32(vdupq_n_f32(0x1.573e2ep-5f), vdupq_n_f32(0x1.0e4020p-7f), b), u),
        u);
    if (!vpaddd_u64(vreinterpretq_u64_u32(c))) {
        return vfmaq_f32(k, j, k);
    }
    const uint32x4_t d = vandq_u32(vclezq_f32(n), vdupq_n_u32(0x82000000));
    const float32x4_t s1 = vreinterpretq_f32_u32(vaddq_u32(d, vdupq_n_u32(0x7f000000)));
    const float32x4_t s2 = vreinterpretq_f32_u32(vsubq_u32(e, d));
    return vbslq_f32(vcagtq_f32(n, vdupq_n_f32(192)), vmulq_f32(s1, s1),
                     vbslq_f32(c, vmulq_f32(vfmaq_f32(s2, j, s2), s1), vfmaq_f32(k, j, k)));
}

// ---- ggml_vec_dot_f32 NEON (GGML_F32_STEP=16, EPR=4, ARR=4) ----
static float vecdot_ref(int n, const float * x, const float * y) {
    const int np = n & ~15;
    float32x4_t sum[4] = { vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0) };
    for (int i = 0; i < np; i += 16) {
        for (int j = 0; j < 4; j++) {
            sum[j] = vfmaq_f32(sum[j], vld1q_f32(x + i + j * 4), vld1q_f32(y + i + j * 4));
        }
    }
    // GGML_F32x4_REDUCE (simd-mappings.h, ARR=4): x0+=x2; x1+=x3; x0+=x1; vaddvq
    sum[0] = vaddq_f32(sum[0], sum[2]);
    sum[1] = vaddq_f32(sum[1], sum[3]);
    sum[0] = vaddq_f32(sum[0], sum[1]);
    float sumf = vaddvq_f32(sum[0]);
    // ggml's leftover `sumf += x[i]*y[i]`, as clang -O2 auto-vectorizes it:
    // the 4-aligned bulk is non-fused (fmul+fadd), the final <4 scalar tail
    // is fused (fmadd). Written explicitly so the dumped ground truth is
    // independent of this harness's own optimization level.
    const int L = n - np;
    const int lead = np + (L & ~3);
    for (int i = np; i < lead; ++i) {
        volatile float p = x[i] * y[i];
        sumf = sumf + p;
    }
    for (int i = lead; i < n; ++i) {
        sumf = fmaf(x[i], y[i], sumf);
    }
    return sumf;
}

// ---- ggml_vec_soft_max_f32 NEON + the soft_max wrapper (max, exp+sum, scale) ----
static double softmax_ref(int n, const float * x, float * out) {
    float max = -INFINITY;
    for (int i = 0; i < n; i++) {
        if (x[i] > max) {
            max = x[i];
        }
    }
    double sum = 0.0;
    int i = 0;
    for (; i + 3 < n; i += 4) {
        float32x4_t val = ggml_v_expf(vsubq_f32(vld1q_f32(x + i), vdupq_n_f32(max)));
        vst1q_f32(out + i, val);
        sum += (double) vaddvq_f32(val);
    }
    for (; i < n; ++i) {
        float val = expf(x[i] - max);
        sum += (double) val;
        out[i] = val;
    }
    double inv = 1.0 / sum;
    for (int k = 0; k < n; k++) {
        out[k] = out[k] * (float) inv;
    }
    return sum;
}

static void wf(const std::string & p, const void * d, size_t nb) {
    FILE * f = fopen(p.c_str(), "wb");
    fwrite(d, 1, nb, f);
    fclose(f);
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <out_dir>\n", argv[0]);
        return 1;
    }
    const std::string out = argv[1];
    // lengths cover the hello/story block-0 reductions (<=304) AND the story
    // block-1 canvas reduction n_kv=553 (bidirectional over the full N=P+C,
    // under the 1024 sliding window so unclipped) plus neighbors spanning the
    // vec_dot leftover residue classes (L = n - (n&~15); fused tail = L&3).
    for (int n : { 256, 273, 288, 296, 297, 304, 305, 512, 528, 540, 544, 549, 552, 553, 560 }) {
        std::vector<float> x(n), y(n);
        for (int i = 0; i < n; i++) {
            x[i] = rnd();
            y[i] = rnd();
        }
        float r = vecdot_ref(n, x.data(), y.data());
        std::vector<uint8_t> blob(n * 8 + 4);
        memcpy(blob.data(), x.data(), n * 4);
        memcpy(blob.data() + n * 4, y.data(), n * 4);
        memcpy(blob.data() + n * 8, &r, 4);
        wf(out + "/vecdot-" + std::to_string(n) + ".bin", blob.data(), blob.size());

        // softmax: feed REALISTIC KQ-like scores — wide magnitude with
        // outliers, so the post-max-subtract exp args span the v_expf
        // underflow region (|n|>126, |n|>192) the attention actually hits
        std::vector<float> sx(n), so(n);
        for (int i = 0; i < n; i++) {
            float r = rnd();
            sx[i] = r * 60.0f;            // base spread ~[-60, 60]
            if (i % 7 == 0) {
                sx[i] += 40.0f;           // a few dominant peaks -> deep underflow tail
            }
            if (i % 13 == 0) {
                sx[i] -= 120.0f;          // far-negative outliers
            }
        }
        double sum = softmax_ref(n, sx.data(), so.data());
        std::vector<uint8_t> sblob(n * 8 + 8);
        memcpy(sblob.data(), sx.data(), n * 4);
        memcpy(sblob.data() + n * 4, so.data(), n * 4);
        memcpy(sblob.data() + n * 8, &sum, 8);
        wf(out + "/softmax-" + std::to_string(n) + ".bin", sblob.data(), sblob.size());
    }
    printf("{\"lengths\":[256,273,288,296,297,304,305,512,528,540,544,549,552,553,560]}\n");
    return 0;
}
