// dg-mc-loop — reference oracle for the MULTI-CANVAS (block-autoregressive)
// generation loop (DiffusionGemma lane Phase 5).
//
// Transcribes diffusion-cli.cpp's run_turn canvas path at the pinned
// llama.cpp checkout ((c) the llama.cpp authors, MIT): per block one full
// Entropy-Bound denoise of [prefix | canvas] (the EB loop transcription
// from scripts/dg-eb-loop.cpp — the rng RE-SEEDS with the same seed each
// block, as in the reference where it is local to
// diffusion_generate_entropy_bound), then trim_canvas (first
// end-of-generation token, or a stride-1/2 repetition loop of >= 6 reps),
// commit-or-stop, with the ubatch budget guard. Adds per-block/per-step
// dumps + the vocab's EOG id set + the final response tokens and text.
//
// CPU-pure kernel contract: link against build-cpu, devices pinned empty,
// no repack.
//
// Build: c++ -std=c++17 -O2 -I <pin>/include -I <pin>/ggml/include \
//        scripts/dg-mc-loop.cpp -L <pin>/build-cpu/bin -lllama -lggml -lggml-base \
//        -Wl,-rpath,<pin>/build-cpu/bin -o dg-mc-loop
// Run:   dg-mc-loop <model.gguf> <prompt_ids.i32> <seed> <S> <n_blocks> <max_ub> <out_dir>

#include "llama.h"

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

// one EB denoise of [prefix | canvas] (diffusion_generate_entropy_bound
// default path, dumping per-step outputs under <out>/block<b>-...).
// Returns the final argmax canvas.
static std::vector<int32_t> eb_denoise_block(llama_context * ctx, llama_model * model,
                                             const std::vector<int32_t> & prefix, int32_t C,
                                             int32_t n_vocab, int32_t seed, int32_t S,
                                             const std::string & out, int b) {
    const int32_t P = (int32_t) prefix.size();
    const int32_t N = P + C;
    const std::string bp = out + "/block" + std::to_string(b) + "-";

    std::mt19937                           rng(seed);
    std::uniform_real_distribution<float>  uni01(0.0f, 1.0f);
    std::uniform_int_distribution<int32_t> vocab_dist(0, n_vocab - 1);

    std::vector<int32_t> current_canvas(C);
    for (int32_t i = 0; i < C; i++) {
        current_canvas[i] = vocab_dist(rng);
    }

    std::vector<float>   sc_buffer((size_t) C * n_vocab, 0.0f);
    std::vector<int32_t> argmax_canvas(C, 0), prev_argmax(C, -1), denoiser(C), order(C), renoise(C);
    std::vector<float>   entropy(C), u(C);

    llama_batch batch = llama_batch_init(N, 0, 1);
    float prev_temp_inv = 1.0f;
    int   held = 0;
    bool  finished = false;
    int   executed = 0;

    for (int32_t cur_step = S; cur_step >= 1 && !finished; --cur_step) {
        const int32_t step_idx = S - cur_step;
        const float   t        = 0.4f + (0.8f - 0.4f) * ((float) cur_step / (float) S);
        const float   temp_inv = 1.0f / t;

        batch.n_tokens = N;
        for (int32_t i = 0; i < N; i++) {
            batch.token[i]     = (i < P) ? prefix[i] : current_canvas[i - P];
            batch.pos[i]       = i;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]    = 1;
        }
        llama_diffusion_set_sc(model, sc_buffer.data(), step_idx == 0 ? 0.0f : 1.0f,
                               prev_temp_inv, true);
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "decode failed at block %d step %d\n", b, step_idx);
            exit(1);
        }
        const float * logits = llama_get_logits(ctx);
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
        std::sort(order.begin(), order.end(), [&](int32_t a, int32_t c2) { return entropy[a] < entropy[c2]; });
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
        write_file(bp + "argmax-step" + ss + ".i32",   argmax_canvas.data(), (size_t) C * 4);
        write_file(bp + "entropy-step" + ss + ".f32",  entropy.data(),       (size_t) C * 4);
        write_file(bp + "denoiser-step" + ss + ".i32", denoiser.data(),      (size_t) C * 4);
        write_file(bp + "accepted-step" + ss + ".u8",  accepted.data(),      (size_t) C);
        write_file(bp + "next-step" + ss + ".i32",     current_canvas.data(), (size_t) C * 4);
        executed = step_idx + 1;
        fprintf(stderr, "block %d step %d: n_accepted=%d entropy_sum=%.4f finished=%d\n",
                b, step_idx, (int) std::count(accepted.begin(), accepted.end(), 1), entropy_sum,
                (int) finished);
    }
    llama_batch_free(batch);

    char meta[160];
    snprintf(meta, sizeof(meta), "{\"block\":%d,\"P\":%d,\"executed\":%d,\"finished\":%s}\n",
             b, P, executed, finished ? "true" : "false");
    write_file(bp + "meta.json", meta, strlen(meta));
    return argmax_canvas;
}

int main(int argc, char ** argv) {
    if (argc != 8) {
        fprintf(stderr,
                "usage: %s <model.gguf> <prompt_ids.i32> <seed> <S> <n_blocks> <max_ub> <out_dir>\n",
                argv[0]);
        return 1;
    }
    const char *      model_path = argv[1];
    const char *      ids_path   = argv[2];
    const int32_t     seed       = atoi(argv[3]);
    const int32_t     S          = atoi(argv[4]);
    const int32_t     n_blocks   = atoi(argv[5]);
    const int32_t     max_ub     = atoi(argv[6]);
    const std::string out        = argv[7];

    std::vector<int32_t> prefix = read_i32(ids_path);
    if (prefix.empty()) { fprintf(stderr, "empty prompt\n"); return 1; }

    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    mparams.use_extra_bufts = false;
    static ggml_backend_dev_t no_devices[1] = { nullptr };
    mparams.devices = no_devices;
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) { fprintf(stderr, "failed to load model\n"); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);

    char canvas_meta[32] = {};
    llama_model_meta_val_str(model, "diffusion.canvas_length", canvas_meta, sizeof(canvas_meta));
    const int32_t C = (int32_t) strtol(canvas_meta, nullptr, 10);
    if (C <= 0) { fprintf(stderr, "missing canvas_length\n"); return 1; }

    // the vocab's end-of-generation id set (authoritative for the gate)
    {
        std::vector<int32_t> eog;
        for (int32_t t = 0; t < n_vocab; t++) {
            if (llama_vocab_is_eog(vocab, t)) {
                eog.push_back(t);
            }
        }
        write_file(out + "/eog-ids.i32", eog.data(), eog.size() * 4);
        fprintf(stderr, "EOG ids (%zu):", eog.size());
        for (int32_t t : eog) { fprintf(stderr, " %d", t); }
        fprintf(stderr, "\n");
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx     = max_ub;
    cparams.n_batch   = max_ub;
    cparams.n_ubatch  = max_ub;
    cparams.no_perf   = true;
    cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "failed to create context\n"); return 1; }
    llama_set_causal_attn(ctx, false);

    // ---- run_turn canvas path, verbatim ----
    auto trim_canvas = [&](const int32_t * canvas, size_t n) -> size_t {
        size_t cut = n;
        for (size_t i = 0; i < n; i++) {
            if (llama_vocab_is_eog(vocab, canvas[i])) {
                cut = i;
                break;
            }
        }
        for (size_t i = 0; i + 1 < cut; i++) {
            bool loop = false;
            for (size_t stride = 1; stride <= 2 && !loop; stride++) {
                size_t reps = 0;
                for (size_t j = i; j + stride < n && canvas[j] == canvas[j + stride]; j += stride) {
                    reps++;
                }
                loop = reps >= 6;
            }
            if (loop) {
                cut = i;
                break;
            }
        }
        return cut;
    };

    std::vector<int32_t> response;
    int executed_blocks = 0;
    std::string stop_reason = "blocks";

    for (int b = 0; b < n_blocks; b++) {
        const int32_t prefix_len = (int32_t) prefix.size();
        const int32_t max_length = prefix_len + C;
        if (max_length > max_ub) {
            if (b == 0) { fprintf(stderr, "prompt+canvas exceed ubatch\n"); return 1; }
            stop_reason = "ubatch";
            break;
        }
        write_file(out + "/block" + std::to_string(b) + "-prefix.i32", prefix.data(),
                   prefix.size() * 4);

        std::vector<int32_t> canvas =
            eb_denoise_block(ctx, model, prefix, C, n_vocab, seed, S, out, b);
        const size_t cut = trim_canvas(canvas.data(), (size_t) C);
        write_file(out + "/block" + std::to_string(b) + "-final-canvas.i32", canvas.data(),
                   (size_t) C * 4);
        char cutbuf[64];
        snprintf(cutbuf, sizeof(cutbuf), "{\"cut\":%zu}\n", cut);
        write_file(out + "/block" + std::to_string(b) + "-cut.json", cutbuf, strlen(cutbuf));
        executed_blocks = b + 1;

        response.insert(response.end(), canvas.begin(), canvas.begin() + cut);
        if (cut < (size_t) C) {
            stop_reason = "trim";
            break;
        }
        prefix.insert(prefix.end(), canvas.begin(), canvas.begin() + cut);
    }

    write_file(out + "/response.i32", response.data(), response.size() * 4);
    {
        // detokenize the response (reference text surface)
        std::string text;
        std::vector<char> piece(256);
        for (int32_t tok : response) {
            int n = llama_token_to_piece(vocab, tok, piece.data(), (int) piece.size(), 0, false);
            if (n < 0) { piece.resize(-n); n = llama_token_to_piece(vocab, tok, piece.data(), (int) piece.size(), 0, false); }
            text.append(piece.data(), (size_t) n);
        }
        write_file(out + "/response.txt", text.data(), text.size());
    }
    char meta[256];
    snprintf(meta, sizeof(meta),
             "{\"seed\":%d,\"S\":%d,\"C\":%d,\"n_vocab\":%d,\"n_blocks\":%d,\"max_ub\":%d,"
             "\"executed_blocks\":%d,\"response_len\":%zu,\"stop\":\"%s\"}\n",
             seed, S, C, n_vocab, n_blocks, max_ub, executed_blocks, response.size(),
             stop_reason.c_str());
    write_file(out + "/mc-meta.json", meta, strlen(meta));
    fputs(meta, stdout);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
