// dg-dequant-dump — reference dequantization dumper for the DiffusionGemma
// lane's quant-parity gate (Phase 0.5).
//
// Reads ONE tensor's wire bytes for a stated block range straight out of a
// GGUF file and dequantizes them with ggml's own type traits (`to_float`),
// writing raw little-endian f32 to the output file and a JSON manifest line
// to stdout. No model is loaded; only the requested byte range is read.
//
// All dequantization semantics here are the work of the llama.cpp / ggml
// authors (https://github.com/ggml-org/llama.cpp, MIT). This harness must be
// compiled against the PINNED checkout recorded in the lane's
// llamacpp-pin.json so the reference values are tied to that exact commit.
//
// Build (from the pinned llama.cpp checkout, after its cmake build):
//   c++ -std=c++17 -O2 -I <pin>/ggml/include scripts/dg-dequant-dump.cpp \
//       -L <pin>/build/bin -lggml -lggml-base -o <out>/dg-dequant-dump
// Run:
//   dg-dequant-dump <model.gguf> <tensor> <first_block> <n_blocks> <out.bin>

#include "ggml.h"
#include "gguf.h"

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int main(int argc, char ** argv) {
    if (argc != 6) {
        fprintf(stderr,
                "usage: %s <model.gguf> <tensor> <first_block> <n_blocks> <out.bin>\n",
                argv[0]);
        return 1;
    }
    const char *  model_path  = argv[1];
    const char *  tensor_name = argv[2];
    const int64_t first_block = strtoll(argv[3], nullptr, 10);
    const int64_t n_blocks    = strtoll(argv[4], nullptr, 10);
    const char *  out_path    = argv[5];

    if (first_block < 0 || n_blocks <= 0) {
        fprintf(stderr, "block range must be non-negative / positive\n");
        return 1;
    }

    struct gguf_init_params params = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
    struct gguf_context * ctx = gguf_init_from_file(model_path, params);
    if (!ctx) {
        fprintf(stderr, "failed to parse %s\n", model_path);
        return 1;
    }

    const int64_t tid = gguf_find_tensor(ctx, tensor_name);
    if (tid < 0) {
        fprintf(stderr, "tensor %s not found\n", tensor_name);
        return 1;
    }

    const enum ggml_type type     = gguf_get_tensor_type(ctx, tid);
    const size_t  tensor_bytes    = gguf_get_tensor_size(ctx, tid);
    const size_t  tensor_offset   = gguf_get_tensor_offset(ctx, tid);
    const size_t  data_offset     = gguf_get_data_offset(ctx);
    const int64_t blck            = ggml_blck_size(type);
    const size_t  type_size       = ggml_type_size(type);
    const int64_t total_blocks    = (int64_t) (tensor_bytes / type_size);

    if (first_block + n_blocks > total_blocks) {
        fprintf(stderr, "range [%" PRId64 ", %" PRId64 ") exceeds %" PRId64 " blocks\n",
                first_block, first_block + n_blocks, total_blocks);
        return 1;
    }

    // F32 is ggml's identity type (no to_float trait): bytes ARE the values.
    const struct ggml_type_traits * traits = nullptr;
    if (type != GGML_TYPE_F32) {
        traits = ggml_get_type_traits(type);
        if (!traits || !traits->to_float) {
            fprintf(stderr, "type %s has no to_float trait\n", ggml_type_name(type));
            return 1;
        }
    }

    FILE * f = fopen(model_path, "rb");
    if (!f) {
        fprintf(stderr, "cannot reopen %s\n", model_path);
        return 1;
    }
    const uint64_t read_at = (uint64_t) data_offset + tensor_offset +
                             (uint64_t) first_block * type_size;
    if (fseeko(f, (off_t) read_at, SEEK_SET) != 0) {
        fprintf(stderr, "seek to %" PRIu64 " failed\n", read_at);
        return 1;
    }
    std::vector<uint8_t> wire((size_t) n_blocks * type_size);
    if (fread(wire.data(), 1, wire.size(), f) != wire.size()) {
        fprintf(stderr, "short read of %zu bytes at %" PRIu64 "\n", wire.size(), read_at);
        return 1;
    }
    fclose(f);

    std::vector<float> out((size_t) n_blocks * blck);
    if (type == GGML_TYPE_F32) {
        memcpy(out.data(), wire.data(), wire.size());
    } else {
        traits->to_float(wire.data(), out.data(), (int64_t) out.size());
    }

    FILE * o = fopen(out_path, "wb");
    if (!o) {
        fprintf(stderr, "cannot open %s for write\n", out_path);
        return 1;
    }
    if (fwrite(out.data(), sizeof(float), out.size(), o) != out.size()) {
        fprintf(stderr, "short write to %s\n", out_path);
        return 1;
    }
    fclose(o);

    printf("{\"tensor\":\"%s\",\"type\":\"%s\",\"first_block\":%" PRId64
           ",\"n_blocks\":%" PRId64 ",\"values\":%zu,\"blck_size\":%" PRId64
           ",\"type_size\":%zu,\"file_read_offset\":%" PRIu64
           ",\"total_blocks\":%" PRId64 ",\"dump\":\"%s\"}\n",
           tensor_name, ggml_type_name(type), first_block, n_blocks, out.size(),
           blck, type_size, read_at, total_blocks, out_path);

    gguf_free(ctx);
    return 0;
}
