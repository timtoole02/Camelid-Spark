// dg-eb-step — reference oracle for ONE Entropy-Bound denoiser step (step 0)
// for the DiffusionGemma lane's Phase 3 parity gate.
//
// The per-position worker math, the acceptance rule, and the renoise rule are
// transcribed LINE-FOR-LINE from diffusion_generate_entropy_bound()
// (examples/diffusion/diffusion.cpp in the pinned llama.cpp checkout, MIT,
// (c) the llama.cpp authors). This harness only adds file I/O: it reads the
// canvas logits [C, n_vocab] (raw f32, e.g. llama-diffusion-gemma-eval
// output), re-derives the step's RNG draws from the seed (same stream
// position as the reference: C canvas-init draws first, then u/renoise
// interleaved per position), and dumps every step output for comparison.
//
// Step 0 specifics mirrored: cur_step = S, so t = t_min + (t_max-t_min)*1.0
// = t_max, temp_inv = 1/t; self-conditioning is gated off (not part of the
// logits file's contract — the zero-SC eval forward produces them).
//
// Build: c++ -std=c++17 -O2 scripts/dg-eb-step.cpp -o <out>/dg-eb-step
// Run:   dg-eb-step <logits.bin> <seed> <n_vocab> <C> <S> <t_min> <t_max>
//                   <entropy_bound> <out_dir>
// Emits: eb-argmax.i32, eb-entropy.f32, eb-denoiser.i32, eb-accepted.u8,
//        eb-next-canvas.i32, eb-meta.json

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

static void write_file(const std::string & path, const void * data, size_t nbytes) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(1); }
    fwrite(data, 1, nbytes, f);
    fclose(f);
}

int main(int argc, char ** argv) {
    if (argc != 10) {
        fprintf(stderr,
                "usage: %s <logits.bin> <seed> <n_vocab> <C> <S> <t_min> <t_max> "
                "<entropy_bound> <out_dir>\n",
                argv[0]);
        return 1;
    }
    const char *      logits_path = argv[1];
    const int32_t     seed        = atoi(argv[2]);
    const int32_t     n_vocab     = atoi(argv[3]);
    const int32_t     C           = atoi(argv[4]);
    const int32_t     S           = atoi(argv[5]);
    const float       t_min       = (float) atof(argv[6]);
    const float       t_max       = (float) atof(argv[7]);
    const float       bound       = (float) atof(argv[8]);
    const std::string out_dir     = argv[9];

    // logits [C, n_vocab] raw f32 (the eval tool's canvas rows)
    FILE * f = fopen(logits_path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", logits_path); return 1; }
    std::vector<float> logits((size_t) C * n_vocab);
    if (fread(logits.data(), 4, logits.size(), f) != logits.size()) {
        fprintf(stderr, "logits file is not C*n_vocab floats\n");
        return 1;
    }
    fclose(f);

    // RNG stream exactly as the reference: canvas init first, then the step's
    // pre-drawn u/renoise (single-threaded, seed-reproducible)
    std::mt19937                           rng(seed);
    std::uniform_real_distribution<float>  uni01(0.0f, 1.0f);
    std::uniform_int_distribution<int32_t> vocab_dist(0, n_vocab - 1);
    std::vector<int32_t> canvas_init(C);
    for (int32_t i = 0; i < C; i++) {
        canvas_init[i] = vocab_dist(rng);
    }
    std::vector<float>   u(C);
    std::vector<int32_t> renoise(C);
    for (int32_t pos = 0; pos < C; pos++) {
        u[pos]       = uni01(rng);
        renoise[pos] = vocab_dist(rng);
    }

    // step 0: cur_step = S
    const int32_t cur_step = S;
    const float   t        = t_min + (t_max - t_min) * ((float) cur_step / (float) S);
    const float   temp_inv = 1.0f / t;

    std::vector<float>   entropy(C);
    std::vector<int32_t> argmax_canvas(C);
    std::vector<int32_t> denoiser(C);

    // per position: argmax, entropy of softmax(raw/t), and a multinomial
    // sample — verbatim from the reference worker (single-threaded here; the
    // reference chunks positions across threads, per-position math identical)
    for (int32_t pos = 0; pos < C; pos++) {
        const float * row = logits.data() + (size_t) pos * n_vocab;
        float m = -INFINITY; int32_t amax = 0;
        for (int32_t v = 0; v < n_vocab; v++) {
            const float z = row[v] * temp_inv;
            if (z > m) { m = z; amax = v; }
        }
        float Z = 0.0f;
        for (int32_t v = 0; v < n_vocab; v++) {
            Z += expf(row[v] * temp_inv - m);
        }
        const float target = u[pos] * Z;
        float   cum = 0.0f, H = 0.0f;
        int32_t sampled = n_vocab - 1; bool picked = false;
        for (int32_t v = 0; v < n_vocab; v++) {
            const float e = expf(row[v] * temp_inv - m);
            const float p = e / Z;
            if (p > 0.0f) { H -= p * logf(p); }
            cum += e;
            if (!picked && cum >= target) { sampled = v; picked = true; }
        }
        entropy[pos]       = H;
        argmax_canvas[pos] = amax;
        denoiser[pos]      = sampled;
    }

    // accept the lowest-entropy positions within the MI bound (sum of
    // strictly-earlier entropies <= bound) — verbatim from the reference
    std::vector<int32_t> order(C);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) { return entropy[a] < entropy[b]; });
    std::vector<char> accepted(C, 0);
    double cumE = 0.0;
    for (int32_t k = 0; k < C; k++) {
        const int32_t pos = order[k];
        cumE += entropy[pos];
        if (cumE - entropy[pos] <= bound) { accepted[pos] = 1; }
    }

    // renoise: accepted -> sampled token, rest -> fresh random
    std::vector<int32_t> next_canvas(C);
    float entropy_sum = 0.0f;
    for (int32_t pos = 0; pos < C; pos++) {
        next_canvas[pos] = accepted[pos] ? denoiser[pos] : renoise[pos];
        entropy_sum += entropy[pos];
    }

    write_file(out_dir + "/eb-argmax.i32",      argmax_canvas.data(), (size_t) C * 4);
    write_file(out_dir + "/eb-entropy.f32",     entropy.data(),       (size_t) C * 4);
    write_file(out_dir + "/eb-denoiser.i32",    denoiser.data(),      (size_t) C * 4);
    write_file(out_dir + "/eb-accepted.u8",     accepted.data(),      (size_t) C);
    write_file(out_dir + "/eb-next-canvas.i32", next_canvas.data(),   (size_t) C * 4);

    int32_t n_accepted = 0;
    for (int32_t pos = 0; pos < C; pos++) { n_accepted += accepted[pos]; }
    char meta[512];
    snprintf(meta, sizeof(meta),
             "{\"seed\":%d,\"n_vocab\":%d,\"C\":%d,\"S\":%d,\"t\":%.9g,\"temp_inv\":%.9g,"
             "\"entropy_bound\":%.9g,\"n_accepted\":%d,\"entropy_sum\":%.9g}\n",
             seed, n_vocab, C, S, t, temp_inv, bound, n_accepted, entropy_sum);
    write_file(out_dir + "/eb-meta.json", meta, strlen(meta));
    fputs(meta, stdout);
    return 0;
}
