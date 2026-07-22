// dg-f16-dump — hardware fp16 ground truth for the DiffusionGemma lane's
// Phase 4 f16 kernel ports.
//
// The reference self-conditioning soft-embedding matmul runs
// ggml_vec_dot_f16 (vec.cpp, the __ARM_FEATURE_FP16_VECTOR_ARITHMETIC NEON
// branch): 4 accumulators of float16x8, vfmaq_f16 per lane (FUSED fp16
// multiply-add, single rounding to f16), an f16 vaddq reduce tree, f32 lane
// conversion, and a double total. Camelid ports those semantics with
// software-emulated fp16 (exact f64 arithmetic + one round-to-nearest-even
// to f16) plus an aarch64 fast path; THIS dump is the hardware oracle both
// are gated against. Kernel structure (c) the llama.cpp / ggml authors.
//
// Build: c++ -std=c++17 -O2 -mcpu=native scripts/dg-f16-dump.cpp -o dg-f16-dump
// Run:   dg-f16-dump <out_dir>
// Emits: f16-fma.bin   (records of u16 a,b,c,r: r = lane0 of vfmaq_f16(c,a,b))
//        f16-add.bin   (records of u16 a,b,r:   r = lane0 of vaddq_f16(a,b))
//        f16-dot-<n>.bin (x[n] u16, y[n] u16, result f32 — full
//                         ggml_vec_dot_f16 NEON transcription)

#include <arm_neon.h>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// deterministic 32-bit LCG (Numerical Recipes constants)
static uint32_t lcg_state = 0x6d2026u;
static uint32_t lcg() {
    lcg_state = lcg_state * 1664525u + 1013904223u;
    return lcg_state;
}

// random FINITE f16 bit pattern (resample NaN/Inf: exponent 0x1f)
static uint16_t rand_f16_bits() {
    for (;;) {
        uint16_t b = (uint16_t) (lcg() >> 13);
        if ((b & 0x7c00) != 0x7c00) {
            return b;
        }
    }
}

// random small-magnitude f16 (|x| < 2): keeps long dot accumulations finite
static uint16_t rand_f16_small() {
    for (;;) {
        uint16_t b = rand_f16_bits();
        uint16_t exp = (b >> 10) & 0x1f;
        if (exp < 15) { // |x| < 2.0
            return b;
        }
    }
}

static float16x8_t load_dup(uint16_t bits) {
    __fp16 h;
    memcpy(&h, &bits, 2);
    return vdupq_n_f16(h);
}

static uint16_t lane0_bits(float16x8_t v) {
    __fp16 h = vgetq_lane_f16(v, 0);
    uint16_t b;
    memcpy(&b, &h, 2);
    return b;
}

// ggml_vec_dot_f16, NEON branch transcription: GGML_F16_STEP=32, EPR=8,
// ARR=4; fused per-lane f16 FMA; f16 reduce tree; f32 lane convert;
// double total; double-accumulated scalar leftovers.
static float vec_dot_f16_ref(int n, const uint16_t * xb, const uint16_t * yb) {
    const __fp16 * x = (const __fp16 *) xb;
    const __fp16 * y = (const __fp16 *) yb;
    double sumf = 0.0;
    const int np = n & ~31;
    float16x8_t sum[4] = { vdupq_n_f16(0.0f), vdupq_n_f16(0.0f), vdupq_n_f16(0.0f),
                           vdupq_n_f16(0.0f) };
    for (int i = 0; i < np; i += 32) {
        for (int j = 0; j < 4; j++) {
            const float16x8_t ax = vld1q_f16(x + i + j * 8);
            const float16x8_t ay = vld1q_f16(y + i + j * 8);
            sum[j] = vfmaq_f16(sum[j], ax, ay);
        }
    }
    // GGML_F16x8_REDUCE
    sum[0] = vaddq_f16(sum[0], sum[2]);
    sum[1] = vaddq_f16(sum[1], sum[3]);
    sum[0] = vaddq_f16(sum[0], sum[1]);
    const float32x4_t t0 = vcvt_f32_f16(vget_low_f16(sum[0]));
    const float32x4_t t1 = vcvt_f32_f16(vget_high_f16(sum[0]));
    sumf = (double) vaddvq_f32(vaddq_f32(t0, t1));
    for (int i = np; i < n; ++i) {
        sumf += (double) ((float) x[i] * (float) y[i]);
    }
    return (float) sumf;
}

static void write_file(const std::string & path, const void * data, size_t nbytes) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(1); }
    fwrite(data, 1, nbytes, f);
    fclose(f);
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <out_dir>\n", argv[0]);
        return 1;
    }
    const std::string out = argv[1];
    const int N = 65536;

    {
        std::vector<uint16_t> rec(N * 4);
        for (int i = 0; i < N; i++) {
            const uint16_t a = rand_f16_bits();
            const uint16_t b = rand_f16_bits();
            const uint16_t c = rand_f16_bits();
            const float16x8_t r = vfmaq_f16(load_dup(c), load_dup(a), load_dup(b));
            rec[i * 4 + 0] = a;
            rec[i * 4 + 1] = b;
            rec[i * 4 + 2] = c;
            rec[i * 4 + 3] = lane0_bits(r);
        }
        write_file(out + "/f16-fma.bin", rec.data(), rec.size() * 2);
    }
    {
        std::vector<uint16_t> rec(N * 3);
        for (int i = 0; i < N; i++) {
            const uint16_t a = rand_f16_bits();
            const uint16_t b = rand_f16_bits();
            const float16x8_t r = vaddq_f16(load_dup(a), load_dup(b));
            rec[i * 3 + 0] = a;
            rec[i * 3 + 1] = b;
            rec[i * 3 + 2] = lane0_bits(r);
        }
        write_file(out + "/f16-add.bin", rec.data(), rec.size() * 2);
    }
    for (int n : { 32, 64, 100, 4096, 262144 }) {
        std::vector<uint16_t> x(n), y(n);
        for (int i = 0; i < n; i++) {
            x[i] = rand_f16_small();
            y[i] = rand_f16_small();
        }
        const float r = vec_dot_f16_ref(n, x.data(), y.data());
        std::vector<uint8_t> blob(n * 4 + 4);
        memcpy(blob.data(), x.data(), n * 2);
        memcpy(blob.data() + n * 2, y.data(), n * 2);
        memcpy(blob.data() + n * 4, &r, 4);
        write_file(out + "/f16-dot-" + std::to_string(n) + ".bin", blob.data(), blob.size());
    }
    printf("{\"records\":%d,\"dots\":[32,64,100,4096,262144]}\n", N);
    return 0;
}
