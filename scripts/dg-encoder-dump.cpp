// dg-encoder-dump — per-layer checkpoint dumper for the DiffusionGemma
// lane's Phase 2 (PREFILL) and Phase 3 (UNIFIED) parity gates.
//
// Without a canvas file: ONE prompt-only PREFILL forward (the encoder path:
// causal prompt attention, encoder per-layer scalars, prompt K/V store).
// With a canvas file: ONE unified no-cache [prompt | canvas] forward (the
// Phase 3 decode surface: region mask, canvas rms_norm embedding, decoder
// scalars on canvas rows, zero self-conditioning) — plus the final logits
// ("result_output").
// Both run on the CPU backend and capture the graph's named checkpoint
// tensors via the eval callback. Each tensor is written raw (f32/i32
// little-endian) with a JSON manifest line on stdout.
//
// All model-graph semantics here are the work of the llama.cpp / ggml authors
// (https://github.com/ggml-org/llama.cpp, MIT). Compile against the PINNED
// checkout recorded in the lane's llamacpp-pin.json.
//
// Build:
//   c++ -std=c++17 -O2 -I <pin>/include -I <pin>/common -I <pin>/ggml/include \
//       scripts/dg-encoder-dump.cpp -L <pin>/build/bin -lllama \
//       -Wl,-rpath,<pin>/build/bin -o <out>/dg-encoder-dump
// Run:
//   dg-encoder-dump <model.gguf> <prompt_ids.i32> <out_dir> [canvas_ids.i32]

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct capture_ctx {
    std::string out_dir;
    FILE *      manifest;
    int         n_layer;
    int         captured;
};

// NOTE: cb() renames a tensor each time it labels it, so only the LAST label
// survives: the embedding row is observable as "inp_region" (not
// "inp_scaled") and the scaled layer output as "l_out-N" (not
// "out_scaled-N"); the values are identical (cvec is a pass-through here).
static bool want_name(const char * name, int n_layer) {
    if (strcmp(name, "inp_region") == 0 || strcmp(name, "result_norm") == 0 ||
        strcmp(name, "result_output") == 0) {
        return true;
    }
    // Phase 5 SC-signal bisection (forensics cb's in diffusion-gemma.cpp):
    if (strcmp(name, "sc_probs") == 0 || strcmp(name, "sc_soft") == 0 ||
        strcmp(name, "sc_normed") == 0 || strcmp(name, "sc_g") == 0 ||
        strcmp(name, "sc_sig") == 0) {
        return true;
    }
    static const char * per_layer[] = {
        "ffn_block_out", // Phase 5: pre-region-scalar FFN block output
        "Kcur_pos",      "Vcur_normed", "attn_out", "ffn_moe_logits",
        "ffn_moe_topk",  "l_out",
        // pre-attention chain bisection (Phase 3: batch-size-dependent
        // reference kernels suspected between inp_region and Kcur_pos)
        "attn_norm",     "Qcur",        "Kcur",     "Vcur",
        "Qcur_normed",   "Kcur_normed",
        // Phase 5: attention-internal bisection (KQV value-mix vs softmax)
        "Qcur_pos",      "kq_soft_max", "kqv",
        // FFN-branch bisection points (labels survive cb renames)
        "ffn_mlp",       "ffn_moe",     "ffn_moe_weights",
        // expert-chain bisection points (merged gate_up path)
        "ffn_moe_gate_up", "ffn_moe_geglu", "ffn_moe_down", "ffn_moe_down_scaled",
        // combine-chain bisection: normalized weights + PRE-norm slot sum
        "ffn_moe_weights_norm", "ffn_moe_out",
        // ground-truth weight sums (pre/post clamp)
        "ffn_moe_weights_sum", "ffn_moe_weights_sum_clamped",
    };
    for (const char * base : per_layer) {
        const size_t blen = strlen(base);
        if (strncmp(name, base, blen) == 0 && name[blen] == '-') {
            // exact "-<digits>" suffix only: ggml view names append
            // " (permuted)"/" (transposed)" which must not match
            const char * suffix = name + blen + 1;
            if (*suffix == '\0') {
                continue;
            }
            bool digits = true;
            for (const char * c = suffix; *c; ++c) {
                if (*c < '0' || *c > '9') {
                    digits = false;
                    break;
                }
            }
            if (digits) {
                const int il = atoi(suffix);
                if (il >= 0 && il < n_layer) {
                    return true;
                }
            }
        }
    }
    return false;
}

static bool eval_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    capture_ctx * cc = (capture_ctx *) user_data;
    if (!want_name(t->name, cc->n_layer)) {
        return true; // not ours; let the scheduler proceed
    }
    if (ask) {
        return true; // yes, we want to observe this tensor
    }

    const size_t nbytes = ggml_nbytes(t);
    std::vector<uint8_t> host(nbytes);
    ggml_backend_tensor_get(t, host.data(), 0, nbytes);

    // file name: checkpoint name with '-' kept (names contain no '/')
    std::string path = cc->out_dir + "/" + t->name + ".bin";
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", path.c_str());
        exit(1);
    }
    fwrite(host.data(), 1, nbytes, f);
    fclose(f);

    fprintf(cc->manifest,
            "{\"name\":\"%s\",\"type\":\"%s\",\"ne\":[%" PRId64 ",%" PRId64 ",%" PRId64
            ",%" PRId64 "],\"nbytes\":%zu,\"file\":\"%s.bin\"}\n",
            t->name, ggml_type_name(t->type), t->ne[0], t->ne[1], t->ne[2], t->ne[3], nbytes,
            t->name);
    cc->captured++;
    return true;
}

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

static std::vector<float> read_f32(const char * path) {
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<float> v(sz / 4);
    if (fread(v.data(), 4, v.size(), f) != v.size()) { fprintf(stderr, "short read\n"); exit(1); }
    fclose(f);
    return v;
}

int main(int argc, char ** argv) {
    if (argc != 4 && argc != 5 && argc != 7) {
        fprintf(stderr, "usage: %s <model.gguf> <prompt_ids.i32> <out_dir> [canvas_ids.i32 "
                        "[sc_logits.f32 temp_inv]]\n",
                argv[0]);
        return 1;
    }
    const char * model_path  = argv[1];
    const char * ids_path    = argv[2];
    const char * out_dir     = argv[3];
    const char * canvas_path = (argc >= 5) ? argv[4] : nullptr;
    // Phase 5 SC-active mode: when sc_logits + temp_inv are given, apply
    // self-conditioning (use_sc=1) so the dump is the SC-active forward.
    const char * sc_path     = (argc == 7) ? argv[5] : nullptr;
    const float  sc_temp_inv = (argc == 7) ? (float) atof(argv[6]) : 1.0f;

    const std::vector<int32_t> prompt = read_i32(ids_path);
    const int P = (int) prompt.size();
    if (P <= 0) { fprintf(stderr, "empty prompt\n"); return 1; }
    std::vector<int32_t> canvas;
    if (canvas_path) {
        canvas = read_i32(canvas_path);
        if (canvas.empty()) { fprintf(stderr, "empty canvas\n"); return 1; }
    }
    const int C = (int) canvas.size();
    const int N = P + C;

    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0; // CPU only: the parity contract names the backend
    // An EMPTY device list keeps GPU backends out of the scheduler entirely.
    // n_gpu_layers=0 alone is NOT enough: with a Metal device registered, the
    // sched's op-offload policy still runs large-batch ops (n_tokens >= 32)
    // on the GPU — fine at the Phase 2 prompt (17 rows), but the Phase 3
    // unified forward (273 rows) would silently leave the CPU kernels the
    // parity contract names.
    static ggml_backend_dev_t no_devices[1] = { nullptr };
    mparams.devices = no_devices;
    // No weight repacking: repacked Q4_K (q4_K_8x8) materializes ~13 GB into
    // RAM (OOM on a 16 GB host) and runs different kernels than the generic
    // vec_dot path this lane's parity contract mirrors.
    mparams.use_extra_bufts = false;
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) { fprintf(stderr, "failed to load model\n"); return 1; }

    char nl_buf[16] = {};
    llama_model_meta_val_str(model, "diffusion-gemma.block_count", nl_buf, sizeof(nl_buf));
    const int n_layer = atoi(nl_buf);
    if (n_layer <= 0) { fprintf(stderr, "missing diffusion-gemma.block_count\n"); return 1; }
    const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    std::string manifest_path = std::string(out_dir) + "/manifest.json";
    FILE * manifest = fopen(manifest_path.c_str(), "wb");
    if (!manifest) { fprintf(stderr, "cannot open %s\n", manifest_path.c_str()); return 1; }

    capture_ctx cc;
    cc.out_dir  = out_dir;
    cc.manifest = manifest;
    cc.n_layer  = n_layer;
    cc.captured = 0;

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx             = N;
    cparams.n_batch           = N;
    cparams.n_ubatch          = N; // non-causal arch: whole sequence in one ubatch
    cparams.no_perf           = true;
    cparams.flash_attn_type   = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cparams.cb_eval           = eval_cb;
    cparams.cb_eval_user_data = &cc;
    // DG_NTHREADS: pin the CPU thread count. Thread count changes f32/quant
    // accumulation order, so dumping at two settings measures the reference's
    // OWN cross-run determinism envelope for this graph — the noise floor any
    // independent implementation should be compared against.
    if (const char * nt = getenv("DG_NTHREADS")) {
        cparams.n_threads       = atoi(nt);
        cparams.n_threads_batch = atoi(nt);
    }
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "failed to create context\n"); return 1; }

    llama_set_causal_attn(ctx, false); // the arch fills its own region mask

    llama_batch batch = llama_batch_init(N, 0, 1);
    if (!canvas_path) {
        // ENCODER phase: PREFILL the prompt, writing the prompt K/V store
        // (mirrors examples/diffusion-gemma-eval/diffusion-gemma-eval.cpp DG_CACHED)
        llama_diffusion_set_phase(model, /*PKV_PREFILL=*/1, P);
        llama_diffusion_set_sc(model, nullptr, /*use_sc=*/0.0f, /*temp_inv=*/1.0f, /*enabled=*/false);
        batch.n_tokens = P;
        for (int i = 0; i < P; ++i) {
            batch.token[i]     = prompt[i];
            batch.pos[i]       = i;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]    = (i == P - 1) ? 1 : 0;
        }
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "PREFILL decode failed\n");
            return 1;
        }
        llama_diffusion_set_phase(model, /*PKV_UNIFIED=*/0, 0);
    } else {
        // UNIFIED phase (Phase 3 surface): one no-cache [prompt | canvas]
        // forward, zero self-conditioning, logits for every row (mirrors
        // examples/diffusion-gemma-eval/diffusion-gemma-eval.cpp uncached)
        batch.n_tokens = N;
        for (int i = 0; i < N; ++i) {
            batch.token[i]     = (i < P) ? prompt[i] : canvas[i - P];
            batch.pos[i]       = i;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]    = 1;
        }
        // Phase 5 SC-active mode: upload the previous step's raw canvas logits
        // and enable self-conditioning (use_sc=1) so this dump is the SC-active
        // forward (the cb'd sc_probs/sc_soft/sc_normed/sc_g/sc_sig get captured).
        std::vector<float> sc_logits;
        if (sc_path) {
            sc_logits = read_f32(sc_path);
            const size_t want = (size_t) C * n_vocab;
            if (sc_logits.size() != want) {
                fprintf(stderr, "sc_logits size %zu != C*n_vocab %zu\n", sc_logits.size(), want);
                return 1;
            }
            llama_diffusion_set_sc(model, sc_logits.data(), /*use_sc=*/1.0f, sc_temp_inv,
                                   /*enabled=*/true);
            fprintf(stderr, "SC-ACTIVE: use_sc=1 temp_inv=%.7f\n", sc_temp_inv);
        }
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "UNIFIED decode failed\n");
            return 1;
        }
    }
    llama_batch_free(batch);
    fclose(manifest);

    fprintf(stderr, "captured %d checkpoint tensors for P=%d C=%d into %s\n",
            cc.captured, P, C, out_dir);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
