/*
 * BASALT Phase 1 — NVFP4 golden-vector fixture generator.
 *
 * Provenance: every quantize/dequantize result in the emitted fixtures is
 * produced by the PINNED llama.cpp C reference implementation:
 *
 *   pin:        <llama.cpp> @ git acd79d603cb2e1c84c0886137b80f1ad649b6857
 *   route:      linked-libs — this harness links the pin's own prebuilt
 *               ggml-base.lib / ggml-base.dll (from the pin's build tree at
 *               <llama.cpp>/build/ggml/src, bin at build/bin) and
 *               calls the exported symbols:
 *                   quantize_row_nvfp4_ref   (ggml/src/ggml-quants.c:346)
 *                   dequantize_row_nvfp4     (ggml/src/ggml-quants.c:531)
 *               The UE4M3 helpers ggml_ue4m3_to_fp32 / ggml_fp32_to_ue4m3 are
 *               static-inline in the pin's ggml/src/ggml-impl.h; they are
 *               compiled into this harness verbatim from that header (same
 *               code the DLL was built from).  The emitted ue4m3_table.json is
 *               additionally DERIVED THROUGH THE DLL: for each scale byte b a
 *               block with element code 1 (kvalue=1) is dequantized by
 *               dequantize_row_nvfp4, whose output is exactly 1.0f * d; the
 *               harness asserts the inline and DLL-derived values are
 *               bit-identical and stores the DLL-derived bits.
 *
 * Build (x64 Native Tools / vcvars64):
 *   cl /nologo /O2 /std:c11 /MD /DGGML_SHARED ^
 *      /I <llama.cpp>\ggml\include ^
 *      /I <llama.cpp>\ggml\src ^
 *      nvfp4_fixture_gen.c ^
 *      /link /LIBPATH:<llama.cpp>\build\ggml\src ggml-base.lib
 *   (run with ggml-base.dll from <llama.cpp>\build\bin on PATH
 *    or copied next to the exe)
 *
 * Usage:
 *   nvfp4_fixture_gen gen <outdir>
 *       writes ue4m3_table.json, decode_table.json, random_blocks.json,
 *       encode_vectors.json into <outdir>
 *   nvfp4_fixture_gen dequant <blocks.bin> <nblocks> <out.txt>
 *       reads nblocks * 36 bytes of raw block_nvfp4 wire data, dequantizes
 *       each block via the pin DLL, writes one line per block: 64 * 8
 *       lowercase hex chars (IEEE-754 u32 bits of each f32, element order
 *       0..63, "%08x" each, concatenated).
 *
 * Number formats used in all fixtures:
 *   - f32 values:  lowercase hex of the IEEE-754 binary32 bit pattern ("%08x")
 *   - bulk bytes:  standard base64 (RFC 4648, with padding)
 *   - f32 arrays packed in base64: little-endian u32 per element, order 0..63
 *
 * PRNG (random_blocks.json): PCG32, Melissa O'Neill's pcg32 minimal variant:
 *   state' = state * 6364136223846793005 + inc   (64-bit wrap)
 *   out    = rotr32( ((state >> 18) ^ state) >> 27, state >> 59 )   [pre-advance state]
 *   seeding: state=0; inc=(initseq<<1)|1; step; state+=initstate; step
 *   initstate = 20260716, initseq = 1
 * Block i (0-based, PRNG blocks only) uses distribution kind = i % 4:
 *   kind 0 "prng-uniform": each elem = (next()>>8) * 0x1p-24f * 2.0f - 1.0f
 *   kind 1 "prng-scaled" : e = (int)(next() % 81) - 40 (per block), each elem =
 *                          ldexpf((next()>>8)*0x1p-24f*2.0f-1.0f, e)
 *   kind 2 "prng-bits"   : each elem = bit pattern next(); if !isfinite -> +0.0f
 *   kind 3 "prng-spike"  : each elem u=(next()>>8)*0x1p-24f*2.0f-1.0f scaled
 *                          by 0.01f, then if (next() & 15) == 0 elem *= 4096.0f
 * The 10,000 PRNG blocks are followed by hand-crafted edge blocks (tagged).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <float.h>

#include "ggml-quants.h"   /* pin header: quantize_row_nvfp4_ref / dequantize_row_nvfp4, block_nvfp4 */
#include "ggml-impl.h"     /* pin header: ggml_ue4m3_to_fp32 / ggml_fp32_to_ue4m3 (static inline)   */

#define PIN_SHA   "acd79d603"
#define GENERATOR "nvfp4_fixture_gen.c"
#define ROUTE     "linked-libs"
#define PRNG_DESC "pcg32(initstate=20260716,initseq=1)"
#define PRNG_SEED 20260716
#define N_PRNG_BLOCKS 10000

/* ---------- small utils ---------- */

static uint32_t f32_bits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
static float bits_f32(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }

static void hex_f32(char *dst, float f) { sprintf(dst, "%08x", f32_bits(f)); }

/* 64 floats -> 512 lowercase hex chars + NUL */
static void hex_row(char *dst, const float *y) {
    for (int i = 0; i < 64; i++) sprintf(dst + 8*i, "%08x", f32_bits(y[i]));
}

static const char B64C[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
static void b64(char *dst, const uint8_t *src, size_t n) {
    size_t i = 0, o = 0;
    for (; i + 3 <= n; i += 3) {
        uint32_t v = (src[i] << 16) | (src[i+1] << 8) | src[i+2];
        dst[o++] = B64C[(v >> 18) & 63]; dst[o++] = B64C[(v >> 12) & 63];
        dst[o++] = B64C[(v >>  6) & 63]; dst[o++] = B64C[v & 63];
    }
    if (i + 1 == n) {
        uint32_t v = src[i] << 16;
        dst[o++] = B64C[(v >> 18) & 63]; dst[o++] = B64C[(v >> 12) & 63];
        dst[o++] = '='; dst[o++] = '=';
    } else if (i + 2 == n) {
        uint32_t v = (src[i] << 16) | (src[i+1] << 8);
        dst[o++] = B64C[(v >> 18) & 63]; dst[o++] = B64C[(v >> 12) & 63];
        dst[o++] = B64C[(v >> 6) & 63]; dst[o++] = '=';
    }
    dst[o] = 0;
}

static void die(const char *msg) { fprintf(stderr, "FATAL: %s\n", msg); exit(1); }

static FILE *open_out(const char *dir, const char *name) {
    char path[1024];
    snprintf(path, sizeof path, "%s/%s", dir, name);
    FILE *f = fopen(path, "wb");   /* binary: no CRLF translation, byte-stable output */
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    return f;
}

static void provenance(FILE *f) {
    char date[16];
    time_t t = time(NULL);
    struct tm g;
#ifdef _MSC_VER
    gmtime_s(&g, &t);
#else
    g = *gmtime(&t);
#endif
    strftime(date, sizeof date, "%Y-%m-%d", &g);
    fprintf(f,
        "  \"provenance\": {\n"
        "    \"pin_sha\": \"%s\",\n"
        "    \"generator\": \"%s\",\n"
        "    \"route\": \"%s\",\n"
        "    \"route_detail\": \"quantize_row_nvfp4_ref/dequantize_row_nvfp4 called in the pin-built ggml-base.dll (import lib <llama.cpp>/build/ggml/src/ggml-base.lib); ggml_ue4m3_to_fp32/ggml_fp32_to_ue4m3 are the pin's static-inline ggml/src/ggml-impl.h code compiled into this harness; ue4m3_table additionally cross-verified against DLL dequant output (code 1 -> 1.0f*d)\",\n"
        "    \"compiler\": \"MSVC cl _MSC_FULL_VER=%d _MSC_BUILD=%d (x64)\",\n"
        "    \"date\": \"%s\",\n"
        "    \"prng\": \"%s\",\n"
        "    \"seed\": %d\n"
        "  }",
        PIN_SHA, GENERATOR, ROUTE,
#ifdef _MSC_FULL_VER
        _MSC_FULL_VER, _MSC_BUILD,
#else
        0, 0,
#endif
        date, PRNG_DESC, PRNG_SEED);
}

/* ---------- PCG32 ---------- */

typedef struct { uint64_t state, inc; } pcg32_t;

static uint32_t pcg32_next(pcg32_t *r) {
    uint64_t old = r->state;
    r->state = old * 6364136223846793005ULL + r->inc;
    uint32_t xorshifted = (uint32_t)(((old >> 18u) ^ old) >> 27u);
    uint32_t rot = (uint32_t)(old >> 59u);
    return (xorshifted >> rot) | (xorshifted << ((32 - rot) & 31));
}

static void pcg32_seed(pcg32_t *r, uint64_t initstate, uint64_t initseq) {
    r->state = 0u;
    r->inc = (initseq << 1u) | 1u;
    pcg32_next(r);
    r->state += initstate;
    pcg32_next(r);
}

/* uniform in [-1, 1): 24-bit mantissa grid */
static float pcg_unit(pcg32_t *r) {
    return (float)(pcg32_next(r) >> 8) * 0x1p-24f * 2.0f - 1.0f;
}

/* ---------- pin call helpers ---------- */

static void pin_quant(const float *x, block_nvfp4 *b)  { quantize_row_nvfp4_ref(x, b, 64); }
static void pin_dequant(const block_nvfp4 *b, float *y) { dequantize_row_nvfp4(b, y, 64); }

/* ---------- fixture 1+2: ue4m3 table and decode table ---------- */

static float g_ue4m3_dll[256];  /* d value per scale byte, derived through the DLL */

static void gen_ue4m3_and_decode(const char *outdir) {
    block_nvfp4 blk;
    float y[64];

    /* derive d for every scale byte THROUGH the DLL: code 1 => kvalue 1 => out = 1.0f*d == d */
    for (int sb = 0; sb < 256; sb++) {
        memset(&blk, 0, sizeof blk);
        for (int s = 0; s < 4; s++) blk.d[s] = (uint8_t)sb;
        memset(blk.qs, 0x11, 32);                 /* code 1 in both nibbles everywhere */
        pin_dequant(&blk, y);
        for (int j = 1; j < 64; j++)
            if (f32_bits(y[j]) != f32_bits(y[0])) die("ue4m3 derive: non-uniform output");
        g_ue4m3_dll[sb] = y[0];
        /* cross-verify against the pin's inline ggml_ue4m3_to_fp32 */
        float inline_d = ggml_ue4m3_to_fp32((uint8_t)sb);
        if (f32_bits(inline_d) != f32_bits(y[0])) {
            fprintf(stderr, "ue4m3 mismatch inline=%08x dll=%08x at byte %d\n",
                    f32_bits(inline_d), f32_bits(y[0]), sb);
            exit(1);
        }
    }

    FILE *f = open_out(outdir, "ue4m3_table.json");
    fprintf(f, "{\n");
    provenance(f);
    fprintf(f, ",\n  \"desc\": \"index = UE4M3 scale byte 0..255; value = lowercase hex of IEEE-754 u32 bits of the f32 scale d produced by the pin (dequantize_row_nvfp4 with element code 1: out = 1.0f*d). Matches ggml_ue4m3_to_fp32 (ggml-impl.h): returns raw*0.5; bytes 0x00 and 0x7F -> 0.0.\",\n");
    fprintf(f, "  \"table\": [");
    for (int sb = 0; sb < 256; sb++) {
        char h[9]; hex_f32(h, g_ue4m3_dll[sb]);
        fprintf(f, "%s\"%s\"", sb ? "," : "", h);
        if (sb % 8 == 7 && sb != 255) fprintf(f, "\n            ");
    }
    fprintf(f, "]\n}\n");
    fclose(f);

    /* decode table: 256 scales x 16 codes, each via a real dequant call */
    f = open_out(outdir, "decode_table.json");
    fprintf(f, "{\n");
    provenance(f);
    fprintf(f, ",\n  \"desc\": \"entries[scale][code] = lowercase hex f32 bits of the value dequantize_row_nvfp4 produces for element code (0..15) under UE4M3 scale byte (0..255). Each entry driven by a real DLL call on a uniform crafted block (d[0..3]=scale, all qs nibbles=code); harness asserts all 64 outputs identical. kvalues (kvalues_mxfp4, ggml-common.h) included for reference.\",\n");
    fprintf(f, "  \"kvalues\": [0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12],\n");
    fprintf(f, "  \"entries\": [\n");
    for (int sb = 0; sb < 256; sb++) {
        fprintf(f, "    [");
        for (int code = 0; code < 16; code++) {
            memset(&blk, 0, sizeof blk);
            for (int s = 0; s < 4; s++) blk.d[s] = (uint8_t)sb;
            memset(blk.qs, (code) | (code << 4), 32);
            pin_dequant(&blk, y);
            for (int j = 1; j < 64; j++)
                if (f32_bits(y[j]) != f32_bits(y[0])) die("decode table: non-uniform output");
            char h[9]; hex_f32(h, y[0]);
            fprintf(f, "%s\"%s\"", code ? "," : "", h);
        }
        fprintf(f, "]%s\n", sb == 255 ? "" : ",");
    }
    fprintf(f, "  ],\n");

    /* nibble-position probes: distinct sub-block scales + position-dependent codes,
       so the 64-value output uniquely pins the wire packing order. */
    fprintf(f, "  \"nibble_probe_desc\": \"blocks with distinct per-sub-block scales d=[0x08,0x38,0x40,0x48] (d=0.5*2^-6*... see ue4m3_table) and position-dependent qs patterns; expected = 64 f32 outputs (hex, element order 0..63) from dequantize_row_nvfp4. Pin layout: sub-block s owns qs[s*8..s*8+7]; LOW nibble of qs[s*8+j] = element s*16+j, HIGH nibble = element s*16+8+j.\",\n");
    fprintf(f, "  \"nibble_probes\": [\n");
    static const uint8_t probe_d[4] = { 0x08, 0x38, 0x40, 0x48 };  /* d = 2^-7, 0.5, 1, 2 */
    for (int p = 0; p < 4; p++) {
        memset(&blk, 0, sizeof blk);
        memcpy(blk.d, probe_d, 4);
        for (int k = 0; k < 32; k++) {
            switch (p) {
                case 0: blk.qs[k] = (uint8_t)k;            break;  /* 0x00..0x1f */
                case 1: blk.qs[k] = (uint8_t)(255 - k);    break;  /* 0xff..0xe0 */
                case 2: blk.qs[k] = (uint8_t)((k * 17) & 0xFF); break;
                case 3: blk.qs[k] = (uint8_t)(((2*k) & 0xF) | ((((2*k)+1) & 0xF) << 4)); break;
            }
        }
        pin_dequant(&blk, y);
        char wb64[64]; b64(wb64, (const uint8_t*)&blk, 36);
        char eh[513]; hex_row(eh, y);
        fprintf(f, "    {\"name\":\"probe%d\",\"d\":[8,56,64,72],\"wire\":\"%s\",\"expected\":\"%s\"}%s\n",
                p, wb64, eh, p == 3 ? "" : ",");
    }
    fprintf(f, "  ]\n}\n");
    fclose(f);
}

/* ---------- fixture 3: random blocks ---------- */

static void emit_block(FILE *f, const char *tag, const float *x, int *first) {
    block_nvfp4 blk;
    float y[64];
    pin_quant(x, &blk);
    pin_dequant(&blk, y);
    char ib64[352], wb64[64], eh[513];
    b64(ib64, (const uint8_t*)x, 256);
    b64(wb64, (const uint8_t*)&blk, 36);
    hex_row(eh, y);
    fprintf(f, "%s    {\"tag\":\"%s\",\"i\":\"%s\",\"w\":\"%s\",\"e\":\"%s\"}",
            *first ? "" : ",\n", tag, ib64, wb64, eh);
    *first = 0;
}

static void gen_random_blocks(const char *outdir) {
    FILE *f = open_out(outdir, "random_blocks.json");
    fprintf(f, "{\n");
    provenance(f);
    fprintf(f, ",\n  \"desc\": \"each block: i = input row (base64 of 64 f32, little-endian u32 each, element order 0..63) -> quantize_row_nvfp4_ref -> w = 36 wire bytes (base64: d[4] then qs[32]) -> dequantize_row_nvfp4 -> e = 64 expected f32 (concatenated lowercase %%08x hex of IEEE-754 u32 bits, element order 0..63). First %d blocks are PRNG-driven (see provenance.prng and generator source distribution kinds), remainder are tagged edge blocks.\",\n", N_PRNG_BLOCKS);

    int first = 1;
    fprintf(f, "  \"blocks\": [\n");

    pcg32_t rng;
    pcg32_seed(&rng, PRNG_SEED, 1);
    float x[64];
    static const char *kindtag[4] = { "prng-uniform", "prng-scaled", "prng-bits", "prng-spike" };

    for (int i = 0; i < N_PRNG_BLOCKS; i++) {
        int kind = i % 4;
        switch (kind) {
            case 0:
                for (int j = 0; j < 64; j++) x[j] = pcg_unit(&rng);
                break;
            case 1: {
                int e = (int)(pcg32_next(&rng) % 81) - 40;
                for (int j = 0; j < 64; j++) x[j] = ldexpf(pcg_unit(&rng), e);
                break;
            }
            case 2:
                for (int j = 0; j < 64; j++) {
                    float v = bits_f32(pcg32_next(&rng));
                    x[j] = isfinite(v) ? v : 0.0f;
                }
                break;
            case 3:
                for (int j = 0; j < 64; j++) {
                    float v = pcg_unit(&rng) * 0.01f;
                    if ((pcg32_next(&rng) & 15u) == 0) v *= 4096.0f;
                    x[j] = v;
                }
                break;
        }
        emit_block(f, kindtag[kind], x, &first);
    }

    /* ---- edge blocks ---- */
    const float NEG0 = bits_f32(0x80000000u);

    /* all +0.0 */
    for (int j = 0; j < 64; j++) x[j] = 0.0f;
    emit_block(f, "edge-zero", x, &first);
    /* all -0.0 */
    for (int j = 0; j < 64; j++) x[j] = NEG0;
    emit_block(f, "edge-negzero", x, &first);
    /* alternating +/-0.0 */
    for (int j = 0; j < 64; j++) x[j] = (j & 1) ? NEG0 : 0.0f;
    emit_block(f, "edge-mixzero", x, &first);
    /* max representable: 12 * 224 = 2688 (scale saturates exactly: amax/6 = 448) */
    for (int j = 0; j < 64; j++) x[j] = 2688.0f;
    emit_block(f, "edge-max", x, &first);
    for (int j = 0; j < 64; j++) x[j] = -2688.0f;
    emit_block(f, "edge-negmax", x, &first);
    for (int j = 0; j < 64; j++) x[j] = (j & 1) ? -2688.0f : 2688.0f;
    emit_block(f, "edge-altmax", x, &first);
    /* amax just above saturation: amax/6 = 449 > 448 -> clamp path */
    for (int j = 0; j < 64; j++) x[j] = (j & 1) ? -2694.0f : 2694.0f;
    emit_block(f, "edge-sat-449x6", x, &first);
    /* amax just below: amax/6 = 447 */
    for (int j = 0; j < 64; j++) x[j] = 2682.0f;
    emit_block(f, "edge-sat-447x6", x, &first);
    /* deep saturation */
    for (int j = 0; j < 64; j++) x[j] = 1.0e6f;
    emit_block(f, "edge-sat-1e6", x, &first);
    for (int j = 0; j < 64; j++) x[j] = (j & 1) ? -FLT_MAX : FLT_MAX;
    emit_block(f, "edge-sat-fltmax", x, &first);
    /* saturating spike among small values */
    for (int j = 0; j < 64; j++) x[j] = 0.001f * (float)(j - 32);
    x[5] = FLT_MAX; x[37] = -1.0e7f;
    emit_block(f, "edge-sat-mixed", x, &first);
    /* ties: anchor 6.0 per sub-block -> scale byte 0x38, d=0.5, representable {0,.5,1,1.5,2,3,4,6};
       midpoints 0.25,0.75,1.25,1.75,2.5,3.5,5.0 lock the first-wins rule */
    {
        static const float mids[7] = { 0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f };
        for (int s = 0; s < 4; s++) {
            float *xb = x + 16*s;
            xb[0] = 6.0f;
            for (int j = 0; j < 7; j++) xb[1+j] =  mids[j];
            for (int j = 0; j < 7; j++) xb[8+j] = -mids[j];
            xb[15] = (s & 1) ? -6.0f : 0.0f;
        }
        emit_block(f, "edge-tie-mid-d0.5", x, &first);
    }
    /* ties at another scale: anchor 12.0 -> amax/6=2 -> scale 0x40, d=1.0; midpoints 0.5,1.5,2.5,3.5,5,7,10 */
    {
        static const float mids[7] = { 0.5f, 1.5f, 2.5f, 3.5f, 5.0f, 7.0f, 10.0f };
        for (int s = 0; s < 4; s++) {
            float *xb = x + 16*s;
            xb[0] = 12.0f;
            for (int j = 0; j < 7; j++) xb[1+j] =  mids[j];
            for (int j = 0; j < 7; j++) xb[8+j] = -mids[j];
            xb[15] = 0.0f;
        }
        emit_block(f, "edge-tie-mid-d1.0", x, &first);
    }
    /* exactly representable values (round-trip): kvalues * 0.5, anchor 6.0 */
    {
        static const float reps[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };
        for (int s = 0; s < 4; s++)
            for (int j = 0; j < 16; j++)
                x[16*s + j] = (j < 8) ? reps[j] : -reps[j - 8];
        emit_block(f, "edge-representable-d0.5", x, &first);
    }
    /* subnormal-scale territory: amax = 6*m/512 -> UE4M3 subnormal scale bytes 0x01..0x07 */
    for (int m = 1; m <= 7; m++) {
        float amax = 6.0f * (float)m / 512.0f;
        for (int j = 0; j < 64; j++) x[j] = amax * ((j % 5) - 2) / 2.0f;  /* -amax..amax spread */
        x[0] = amax;
        char tag[32]; snprintf(tag, sizeof tag, "edge-subnormal-scale-m%d", m);
        emit_block(f, tag, x, &first);
    }
    /* amax below smallest nonzero scale: scale rounds to 0 -> d = 0 */
    for (int j = 0; j < 64; j++) x[j] = 1.0e-42f * (float)((j % 3) - 1);
    emit_block(f, "edge-below-scale-floor", x, &first);
    /* f32 subnormal elements */
    for (int j = 0; j < 64; j++) x[j] = bits_f32((uint32_t)(j + 1));       /* tiniest positive subnormals */
    emit_block(f, "edge-f32-subnormal-elems", x, &first);
    /* graded sub-blocks: tiny / unit / large / saturating -> per-sub-block scale independence */
    for (int j = 0; j < 16; j++) x[j]      = 1.0e-6f * (float)(j - 8);
    for (int j = 0; j < 16; j++) x[16 + j] = 0.25f * (float)(j - 8);
    for (int j = 0; j < 16; j++) x[32 + j] = 30.0f * (float)(j - 8);
    for (int j = 0; j < 16; j++) x[48 + j] = 1.0e5f * (float)(j - 8);
    emit_block(f, "edge-graded-subblocks", x, &first);
    /* sign patterns */
    for (int j = 0; j < 64; j++) x[j] = ((j & 1) ? -1.0f : 1.0f) * (float)((j / 2) % 8);
    emit_block(f, "edge-sign-alt", x, &first);
    for (int j = 0; j < 64; j++) x[j] = -0.1f * (float)(j + 1);
    emit_block(f, "edge-neg-only", x, &first);
    /* -0.0 mixed with tiny positives (amax from fabsf(-0.0)=0 cases) */
    for (int j = 0; j < 64; j++) x[j] = (j % 4 == 0) ? NEG0 : 1.0e-3f * (float)(j % 4);
    emit_block(f, "edge-negzero-mix", x, &first);
    /* only sub-block 0 nonzero */
    for (int j = 0; j < 64; j++) x[j] = 0.0f;
    for (int j = 0; j < 16; j++) x[j] = (float)(j - 8) * 0.75f;
    emit_block(f, "edge-first-sub-only", x, &first);
    /* only last element nonzero */
    for (int j = 0; j < 64; j++) x[j] = 0.0f;
    x[63] = -5.5f;
    emit_block(f, "edge-last-elem-only", x, &first);
    /* UE4M3 mantissa round-up cascade: amax/6 just under a power of 2 */
    for (int j = 0; j < 64; j++) x[j] = 11.999999f * ((j % 2) ? -1.0f : 1.0f);  /* amax/6 ~ 1.9999998 */
    emit_block(f, "edge-ue-round-cascade", x, &first);
    for (int j = 0; j < 64; j++) x[j] = 5.9999995f;                              /* amax/6 just under 1 */
    emit_block(f, "edge-ue-round-under1", x, &first);

    fprintf(f, "\n  ],\n");
    fprintf(f, "  \"count_marker\": \"see blocks length\"\n}\n");
    fclose(f);
}

/* ---------- fixture 5: encode vectors ---------- */

static void emit_vec(FILE *f, const char *tag, const float *x, int *first) {
    block_nvfp4 blk;
    float y[64];
    pin_quant(x, &blk);
    pin_dequant(&blk, y);
    char wb64[64], eh[513];
    b64(wb64, (const uint8_t*)&blk, 36);
    hex_row(eh, y);
    fprintf(f, "%s    {\"tag\":\"%s\",\"input\":[", *first ? "" : ",\n", tag);
    for (int j = 0; j < 64; j++) {
        char h[9]; hex_f32(h, x[j]);
        fprintf(f, "%s\"%s\"", j ? "," : "", h);
    }
    fprintf(f, "],\"wire\":\"%s\",\"dequant\":\"%s\"}", wb64, eh);
    *first = 0;
}

static void gen_encode_vectors(const char *outdir) {
    FILE *f = open_out(outdir, "encode_vectors.json");
    fprintf(f, "{\n");
    provenance(f);
    fprintf(f, ",\n  \"desc\": \"encode-parity vectors: input = 64 f32 (lowercase hex of IEEE-754 u32 bits each, element order 0..63) -> quantize_row_nvfp4_ref -> wire = 36 bytes base64 (d[4] then qs[32]); dequant = dequantize_row_nvfp4(wire) as concatenated hex (order 0..63). Pathological inputs (NaN/Inf/-0.0) record whatever the pin does — golden truth, not judged.\",\n");
    fprintf(f, "  \"vectors\": [\n");

    int first = 1;
    float x[64];
    const float NEG0 = bits_f32(0x80000000u);
    const float PINF = bits_f32(0x7f800000u);
    const float NINF = bits_f32(0xff800000u);
    const float QNAN = bits_f32(0x7fc00000u);
    const float NQNAN = bits_f32(0xffc00000u);
    const float SNAN = bits_f32(0x7f800001u);

    /* representable round-trips at several stored scales.
       For scale byte B with raw = 2*d (ue4m3 raw value): amax = 6*raw makes
       fp32_to_ue4m3(amax/6) == B (raw exactly representable), and elements
       kvalues[c]*d are exact. */
    static const uint8_t rt_scales[6] = { 0x38, 0x40, 0x2C, 0x08, 0x04, 0x76 };
    for (int t = 0; t < 6; t++) {
        uint8_t B = rt_scales[t];
        float d = ggml_ue4m3_to_fp32(B);   /* pin inline (verified vs DLL in fixture 1) */
        static const int kv[16] = { 0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12 };
        for (int s = 0; s < 4; s++)
            for (int j = 0; j < 16; j++)
                x[16*s + j] = (float)kv[(j + s) % 16] * d;   /* rotate codes per sub-block */
        /* ensure amax anchor present: kvalue 12 * d = 6*raw */
        x[0] = 12.0f * d; x[16] = 12.0f * d; x[32] = 12.0f * d; x[48] = 12.0f * d;
        char tag[32]; snprintf(tag, sizeof tag, "rt-scale-0x%02x", B);
        emit_vec(f, tag, x, &first);
    }

    /* saturation boundaries */
    for (int j = 0; j < 64; j++) x[j] = 2688.0f;              /* amax/6 == 448 exactly */
    emit_vec(f, "sat-exact-448", x, &first);
    for (int j = 0; j < 64; j++) x[j] = nextafterf(2688.0f, 1e30f);
    emit_vec(f, "sat-448-plus-ulp", x, &first);
    for (int j = 0; j < 64; j++) x[j] = nextafterf(2688.0f, 0.0f);
    emit_vec(f, "sat-448-minus-ulp", x, &first);
    for (int j = 0; j < 64; j++) x[j] = 6.0f * 447.0f;
    emit_vec(f, "sat-447", x, &first);
    for (int j = 0; j < 64; j++) x[j] = 6.0f * 448.5f;
    emit_vec(f, "sat-448.5", x, &first);
    for (int j = 0; j < 64; j++) x[j] = 1.0e4f;
    emit_vec(f, "sat-1e4", x, &first);
    for (int j = 0; j < 64; j++) x[j] = FLT_MAX;
    emit_vec(f, "sat-fltmax", x, &first);

    /* pathological */
    for (int j = 0; j < 64; j++) x[j] = QNAN;
    emit_vec(f, "path-all-qnan", x, &first);
    for (int j = 0; j < 64; j++) x[j] = (j % 8 == 3) ? QNAN : 0.5f * (float)(j % 4);
    emit_vec(f, "path-qnan-mixed", x, &first);
    for (int j = 0; j < 64; j++) x[j] = PINF;
    emit_vec(f, "path-all-pinf", x, &first);
    for (int j = 0; j < 64; j++) x[j] = NINF;
    emit_vec(f, "path-all-ninf", x, &first);
    for (int j = 0; j < 64; j++) x[j] = (j & 1) ? NINF : PINF;
    emit_vec(f, "path-inf-alt", x, &first);
    for (int j = 0; j < 64; j++) x[j] = (j % 16 == 5) ? PINF : 1.0f;
    emit_vec(f, "path-inf-spike", x, &first);
    for (int j = 0; j < 64; j++) x[j] = NEG0;
    emit_vec(f, "path-all-negzero", x, &first);
    for (int j = 0; j < 64; j++) x[j] = NQNAN;
    emit_vec(f, "path-all-neg-qnan", x, &first);
    for (int j = 0; j < 64; j++) x[j] = (j == 0) ? SNAN : 2.0f;
    emit_vec(f, "path-snan-first", x, &first);
    for (int j = 0; j < 64; j++) x[j] = (j < 16) ? QNAN : ((j < 32) ? PINF : ((j < 48) ? NINF : 3.0f));
    emit_vec(f, "path-subblock-mix", x, &first);

    /* tie sweep from the encode side (d=0.5 anchored) */
    {
        static const float mids[7] = { 0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f };
        for (int s = 0; s < 4; s++) {
            float *xb = x + 16*s;
            xb[0] = 6.0f;
            for (int j = 0; j < 7; j++) xb[1+j] = mids[j];
            for (int j = 0; j < 7; j++) xb[8+j] = -mids[j];
            xb[15] = 6.0f;
        }
        emit_vec(f, "tie-mid-d0.5", x, &first);
    }
    /* tiny magnitudes / subnormal input */
    for (int j = 0; j < 64; j++) x[j] = FLT_MIN * (float)(j - 32);
    emit_vec(f, "tiny-fltmin-graded", x, &first);
    for (int j = 0; j < 64; j++) x[j] = bits_f32((uint32_t)(1 + j * 7));  /* f32 subnormals */
    emit_vec(f, "tiny-subnormal-bits", x, &first);
    /* -0.0 amax interaction: single -0.0 with zeros */
    for (int j = 0; j < 64; j++) x[j] = 0.0f;
    x[7] = NEG0;
    emit_vec(f, "negzero-single", x, &first);

    fprintf(f, "\n  ]\n}\n");
    fclose(f);
}

/* ---------- mode: dequant file (real blocks) ---------- */

static int run_dequant_file(const char *inpath, long nblocks, const char *outpath) {
    FILE *fi = fopen(inpath, "rb");
    if (!fi) { fprintf(stderr, "cannot open %s\n", inpath); return 1; }
    FILE *fo = fopen(outpath, "wb");
    if (!fo) { fprintf(stderr, "cannot open %s\n", outpath); return 1; }
    block_nvfp4 blk;
    float y[64];
    char eh[513];
    for (long i = 0; i < nblocks; i++) {
        if (fread(&blk, 1, 36, fi) != 36) { fprintf(stderr, "short read at block %ld\n", i); return 1; }
        pin_dequant(&blk, y);
        hex_row(eh, y);
        fprintf(fo, "%s\n", eh);
    }
    fclose(fi); fclose(fo);
    return 0;
}

/* ---------- main ---------- */

int main(int argc, char **argv) {
    if (sizeof(block_nvfp4) != 36) die("block_nvfp4 size != 36");
    if (argc >= 3 && strcmp(argv[1], "gen") == 0) {
        gen_ue4m3_and_decode(argv[2]);
        gen_random_blocks(argv[2]);
        gen_encode_vectors(argv[2]);
        printf("fixtures written to %s\n", argv[2]);
        return 0;
    }
    if (argc >= 5 && strcmp(argv[1], "dequant") == 0) {
        return run_dequant_file(argv[2], atol(argv[3]), argv[4]);
    }
    fprintf(stderr, "usage: %s gen <outdir> | %s dequant <blocks.bin> <nblocks> <out.txt>\n", argv[0], argv[0]);
    return 2;
}
