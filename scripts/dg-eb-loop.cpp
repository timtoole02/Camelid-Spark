// dg-eb-loop — reference oracle for the FULL Entropy-Bound denoise loop
// (DiffusionGemma lane Phase 4).
//
// Transcribes diffusion_generate_entropy_bound() (examples/diffusion/
// diffusion.cpp at the pinned llama.cpp checkout, MIT, (c) the llama.cpp
// authors) in its DEFAULT configuration — unified no-KV-cache re-decode of
// [prompt | canvas] each step, host self-conditioning buffer, host sampling
// — adding only file dumps: every executed step's canvas-in / argmax /
// entropy / denoiser / accepted / next-canvas, plus the step's raw canvas
// logits for step_idx < n_logit_dumps or the final step.
//
// Runs single-threaded worker math (per-position results are independent of
// the reference's position-chunk threading; proven at step 0 in Phase 3).
// CPU-pure kernel contract: link against the build-cpu libllama
// (GGML_BLAS=OFF, GGML_METAL=OFF) and pin mparams.devices empty.
//
// Build: c++ -std=c++17 -O2 -I <pin>/include -I <pin>/ggml/include \
//        scripts/dg-eb-loop.cpp -L <pin>/build-cpu/bin -lllama -lggml -lggml-base \
//        -Wl,-rpath,<pin>/build-cpu/bin -o dg-eb-loop
// Run:   dg-eb-loop <model.gguf> <prompt_ids.i32> <seed> <S> <n_logit_dumps> <out_dir>

#include "llama.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

static std::vector<int32_t> read_i32(const char * path) {
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<int32_t> v(sz / 4);
    if (fread(v.data(), 4, v.size(), f) != v.size()) { fprintf(stderr, "short read\n"); exit(1); }
    fclose(f);
    return v;
}

static void write_file(const std::string & path, const void * data, size_t nbytes) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(1); }
    fwrite(data, 1, nbytes, f);
    fclose(f);
}

int main(int argc, char ** argv) {
    if (argc != 7) {
        fprintf(stderr, "usage: %s <model.gguf> <prompt_ids.i32> <seed> <S> <n_logit_dumps> <out_dir>\n",
                argv[0]);
        return 1;
    }
    const char *      model_path = argv[1];
    const char *      ids_path   = argv[2];
    const int32_t     seed       = atoi(argv[3]);
    const int32_t     S          = atoi(argv[4]);
    const int32_t     n_dumps    = atoi(argv[5]);
    const std::string out        = argv[6];

    const std::vector<int32_t> prompt = read_i32(ids_path);
    const int P = (int) prompt.size();
    if (P <= 0) { fprintf(stderr, "empty prompt\n"); return 1; }

    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    mparams.use_extra_bufts = false; // no repack (kernel contract)
    static ggml_backend_dev_t no_devices[1] = { nullptr };
    mparams.devices = no_devices;    // no GPU devices in the scheduler
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) { fprintf(stderr, "failed to load model\n"); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);

    char canvas_meta[32] = {};
    if (llama_model_meta_val_str(model, "diffusion.canvas_length", canvas_meta, sizeof(canvas_meta)) < 0) {
        fprintf(stderr, "missing diffusion.canvas_length\n");
        return 1;
    }
    const int32_t C = (int32_t) strtol(canvas_meta, nullptr, 10);
    const int32_t N = P + C;

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx     = N;
    cparams.n_batch   = N;
    cparams.n_ubatch  = N;
    cparams.no_perf   = true;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    if (const char * nt = getenv("DG_NTHREADS")) {
        cparams.n_threads       = atoi(nt);
        cparams.n_threads_batch = atoi(nt);
    }
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "failed to create context\n"); return 1; }

    llama_set_causal_attn(ctx, false);

    // ---- diffusion_generate_entropy_bound, default path, verbatim ----
    std::mt19937                           rng(seed);
    std::uniform_real_distribution<float>  uni01(0.0f, 1.0f);
    std::uniform_int_distribution<int32_t> vocab_dist(0, n_vocab - 1);

    std::vector<int32_t> current_canvas(C);
    for (int32_t i = 0; i < C; i++) {
        current_canvas[i] = vocab_dist(rng);
    }
    write_file(out + "/canvas-init.i32", current_canvas.data(), (size_t) C * 4);

    std::vector<float>   sc_buffer((size_t) C * n_vocab, 0.0f);
    std::vector<int32_t> argmax_canvas(C, 0);
    std::vector<int32_t> prev_argmax(C, -1);
    std::vector<float>   entropy(C);
    std::vector<int32_t> denoiser(C);
    std::vector<int32_t> order(C);
    std::vector<float>   u(C);
    std::vector<int32_t> renoise(C);

    llama_batch batch = llama_batch_init(N, 0, 1);

    float prev_temp_inv = 1.0f;
    int   held          = 0;
    bool  finished      = false;
    int   executed      = 0;
    // Diagnostic-only executed-step cap (does NOT change the t-schedule, which
    // is driven by S): stop after this many steps for the block-1 logit ladder.
    const int cap = getenv("DG_EB_CAP") ? atoi(getenv("DG_EB_CAP")) : 0;

    for (int32_t cur_step = S; cur_step >= 1 && !finished; --cur_step) {
        const int32_t step_idx = S - cur_step;
        const float   t        = 0.4f + (0.8f - 0.4f) * ((float) cur_step / (float) S);
        const float   temp_inv = 1.0f / t;

        batch.n_tokens = N;
        for (int32_t i = 0; i < N; i++) {
            batch.token[i]     = (i < P) ? prompt[i] : current_canvas[i - P];
            batch.pos[i]       = i;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]    = 1;
        }
        // DG-FORENSICS: optionally force pure stateless UNIFIED per step (clear
        // KV memory + pin PKV_UNIFIED), to test whether dg-eb-loop's default
        // (no clear, no set_phase) carries cross-step state at block-1's N.
        if (getenv("DG_EB_FRESH")) {
            llama_memory_clear(llama_get_memory(ctx), true);
            llama_diffusion_set_phase(model, /*PKV_UNIFIED=*/0, 0);
        }
        llama_diffusion_set_sc(model, sc_buffer.data(), step_idx == 0 ? 0.0f : 1.0f,
                               prev_temp_inv, true);
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "decode failed at step %d\n", step_idx);
            return 1;
        }
        const float * logits = llama_get_logits(ctx);

        write_file(out + "/canvas-in-step" + std::to_string(step_idx) + ".i32",
                   current_canvas.data(), (size_t) C * 4);
        if (step_idx < n_dumps) {
            write_file(out + "/logits-step" + std::to_string(step_idx) + ".f32",
                       logits + (size_t) P * n_vocab, (size_t) C * n_vocab * 4);
        }

        std::memcpy(sc_buffer.data(), logits + (size_t) P * n_vocab,
                    (size_t) C * n_vocab * sizeof(float));

        for (int32_t pos = 0; pos < C; pos++) {
            u[pos]       = uni01(rng);
            renoise[pos] = vocab_dist(rng);
        }

        for (int32_t pos = 0; pos < C; pos++) {
            const float * row = logits + (size_t) (P + pos) * n_vocab;
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

        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) { return entropy[a] < entropy[b]; });
        std::vector<char> accepted(C, 0);
        double cumE = 0.0;
        for (int32_t k = 0; k < C; k++) {
            const int32_t pos = order[k];
            cumE += entropy[pos];
            if (cumE - entropy[pos] <= 0.1f) { accepted[pos] = 1; }
        }

        float entropy_sum = 0.0f;
        for (int32_t pos = 0; pos < C; pos++) {
            current_canvas[pos] = accepted[pos] ? denoiser[pos] : renoise[pos];
            entropy_sum += entropy[pos];
        }

        held = (prev_argmax == argmax_canvas) ? held + 1 : 0;
        const bool confident = (entropy_sum / (float) C) < 0.005f;
        if (held >= 1 && confident) { finished = true; }
        prev_argmax   = argmax_canvas;
        prev_temp_inv = temp_inv;

        const std::string ss = std::to_string(step_idx);
        write_file(out + "/argmax-step" + ss + ".i32",   argmax_canvas.data(), (size_t) C * 4);
        write_file(out + "/entropy-step" + ss + ".f32",  entropy.data(),       (size_t) C * 4);
        write_file(out + "/denoiser-step" + ss + ".i32", denoiser.data(),      (size_t) C * 4);
        write_file(out + "/accepted-step" + ss + ".u8",  accepted.data(),      (size_t) C);
        write_file(out + "/next-step" + ss + ".i32",     current_canvas.data(), (size_t) C * 4);
        executed = step_idx + 1;
        fprintf(stderr, "step %d done: t=%.6f n_accepted=%d entropy_sum=%.4f finished=%d\n",
                step_idx, t, (int) std::count(accepted.begin(), accepted.end(), 1), entropy_sum,
                (int) finished);
        if (cap > 0 && executed >= cap) { break; }
    }

    char meta[256];
    snprintf(meta, sizeof(meta),
             "{\"seed\":%d,\"S\":%d,\"C\":%d,\"n_vocab\":%d,\"P\":%d,\"executed\":%d,\"finished\":%s}\n",
             seed, S, C, n_vocab, P, executed, finished ? "true" : "false");
    write_file(out + "/loop-meta.json", meta, strlen(meta));
    fputs(meta, stdout);

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
