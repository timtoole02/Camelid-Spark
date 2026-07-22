// dg-kqv-elem — run the exact ggml NEON ggml_vec_dot_f32 on the isolated
// KQV element (v_col, softmax) extracted by the Phase 5 diag, and compare
// to camelid's result and the reference's stored value. Settles whether
// the reference KQV is a plain ggml_vec_dot_f32 over n_kv or a different
// reduction. Kernel (c) the llama.cpp / ggml authors.
//
// Build: c++ -std=c++17 -O2 -mcpu=native scripts/dg-kqv-elem.cpp -o dg-kqv-elem
// Run:   dg-kqv-elem <kqv-elem.bin>
// Input format: u32 np, np f32 v_col, np f32 softmax, f32 ours, f32 ref

#include <arm_neon.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// ggml_vec_dot_f32, NEON (GGML_F32_STEP=16, EPR=4, ARR=4), with the REAL
// GGML_F32x4_REDUCE order: x0+=x2; x1+=x3; x0+=x1; vaddvq.
static float vecdot_ggml(int n, const float * x, const float * y) {
    const int np = n & ~15;
    float32x4_t sum[4] = { vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0) };
    for (int i = 0; i < np; i += 16) {
        for (int j = 0; j < 4; j++) {
            sum[j] = vfmaq_f32(sum[j], vld1q_f32(x + i + j * 4), vld1q_f32(y + i + j * 4));
        }
    }
    sum[0] = vaddq_f32(sum[0], sum[2]);
    sum[1] = vaddq_f32(sum[1], sum[3]);
    sum[0] = vaddq_f32(sum[0], sum[1]);
    float sumf = vaddvq_f32(sum[0]);
    for (int i = np; i < n; ++i) {
        sumf += x[i] * y[i];
    }
    return sumf;
}

// leftover computed as explicit NON-fused (two roundings), no compiler fma
static float vecdot_ggml_nonfma(int n, const float * x, const float * y) {
    const int np = n & ~15;
    float32x4_t sum[4] = { vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0) };
    for (int i = 0; i < np; i += 16) {
        for (int j = 0; j < 4; j++) {
            sum[j] = vfmaq_f32(sum[j], vld1q_f32(x + i + j * 4), vld1q_f32(y + i + j * 4));
        }
    }
    sum[0] = vaddq_f32(sum[0], sum[2]);
    sum[1] = vaddq_f32(sum[1], sum[3]);
    sum[0] = vaddq_f32(sum[0], sum[1]);
    float sumf = vaddvq_f32(sum[0]);
    for (int i = np; i < n; ++i) {
        volatile float p = x[i] * y[i]; // block fma contraction
        sumf = sumf + p;
    }
    return sumf;
}

// leftover computed as explicit FUSED fmaf (one rounding)
static float vecdot_ggml_fma(int n, const float * x, const float * y) {
    const int np = n & ~15;
    float32x4_t sum[4] = { vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0) };
    for (int i = 0; i < np; i += 16) {
        for (int j = 0; j < 4; j++) {
            sum[j] = vfmaq_f32(sum[j], vld1q_f32(x + i + j * 4), vld1q_f32(y + i + j * 4));
        }
    }
    sum[0] = vaddq_f32(sum[0], sum[2]);
    sum[1] = vaddq_f32(sum[1], sum[3]);
    sum[0] = vaddq_f32(sum[0], sum[1]);
    float sumf = vaddvq_f32(sum[0]);
    for (int i = np; i < n; ++i) {
        sumf = fmaf(x[i], y[i], sumf);
    }
    return sumf;
}

// The mul_mat path also runs the SAME vec_dot but accumulates into the dst
// via tmp[]; for nrc=1 it's identical. Provide a "swapped operand" variant
// too (kq as x, v as y) in case operand order matters for fma association.
static float vecdot_ggml_swapped(int n, const float * x, const float * y) {
    return vecdot_ggml(n, y, x);
}

int main(int argc, char ** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <kqv-elem.bin>\n", argv[0]);
        return 1;
    }
    FILE * f = fopen(argv[1], "rb");
    if (!f) { perror("open"); return 1; }
    uint32_t np = 0;
    fread(&np, 4, 1, f);
    std::vector<float> v(np), s(np);
    fread(v.data(), 4, np, f);
    fread(s.data(), 4, np, f);
    float ours = 0, ref = 0;
    fread(&ours, 4, 1, f);
    fread(&ref, 4, 1, f);
    fclose(f);

    float g  = vecdot_ggml((int) np, v.data(), s.data());      // x=v_col, y=softmax
    float gs = vecdot_ggml_swapped((int) np, v.data(), s.data()); // x=softmax, y=v_col

    auto bits = [](float x) { uint32_t u; memcpy(&u, &x, 4); return u; };
    float nf = vecdot_ggml_nonfma((int) np, v.data(), s.data());
    float ff = vecdot_ggml_fma((int) np, v.data(), s.data());
    // RULE: leftover L -> (L & ~3) non-fused sequential, then (L & 3) fused
    float rule;
    {
        int n = (int) np;
        const float * x = v.data();
        const float * y = s.data();
        const int npp = n & ~15;
        float32x4_t sum[4] = { vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0), vdupq_n_f32(0) };
        for (int i = 0; i < npp; i += 16) {
            for (int j = 0; j < 4; j++) {
                sum[j] = vfmaq_f32(sum[j], vld1q_f32(x + i + j * 4), vld1q_f32(y + i + j * 4));
            }
        }
        sum[0] = vaddq_f32(sum[0], sum[2]);
        sum[1] = vaddq_f32(sum[1], sum[3]);
        sum[0] = vaddq_f32(sum[0], sum[1]);
        float sumf = vaddvq_f32(sum[0]);
        const int L = n - npp;
        const int lead = npp + (L & ~3);
        for (int k = npp; k < lead; ++k) {
            volatile float p = x[k] * y[k];
            sumf = sumf + p; // non-fused
        }
        for (int k = lead; k < n; ++k) {
            sumf = fmaf(x[k], y[k], sumf); // fused tail
        }
        rule = sumf;
    }
    printf("np=%u\n", np);
    printf("ggml_vec_dot(default)   bits=0x%08x\n", bits(g));
    printf("ggml leftover NON-fma   bits=0x%08x\n", bits(nf));
    printf("ggml leftover FMA       bits=0x%08x\n", bits(ff));
    printf("camelid (ours)          bits=0x%08x\n", bits(ours));
    printf("reference (ref)         bits=0x%08x\n", bits(ref));
    printf("RULE(L&~3 nonfma,L&3 fma) bits=0x%08x\n", bits(rule));
    printf("nonfma==ref: %d  fma==ref: %d  default==ref: %d  ours==ref: %d  RULE==ref: %d\n",
           bits(nf) == bits(ref), bits(ff) == bits(ref), bits(g) == bits(ref), bits(ours) == bits(ref),
           bits(rule) == bits(ref));
    return 0;
}
