// dg-rng-dump — ground-truth dump of the reference EB denoiser's host RNG
// stream for the DiffusionGemma lane's Phase 3 parity gate.
//
// Mirrors the EXACT draw order of diffusion_generate_entropy_bound()
// (examples/diffusion/diffusion.cpp in the pinned llama.cpp checkout, MIT,
// (c) the llama.cpp authors):
//   std::mt19937 rng(seed);
//   canvas init : C draws of uniform_int_distribution<int32_t>(0, n_vocab-1)
//   each step   : per position, u = uniform_real_distribution<float>(0,1)
//                 then renoise = vocab_dist(rng), interleaved
// Compiled with the same Apple clang / libc++ as the pinned reference, this
// is the oracle camelid's Rust mt19937 + libc++ distribution ports are
// gated against (distribution algorithms are implementation-defined, so the
// port must match libc++ specifically, not just the standard).
//
// Build: c++ -std=c++17 -O2 scripts/dg-rng-dump.cpp -o <out>/dg-rng-dump
// Run:   dg-rng-dump <seed> <n_vocab> <C> <n_steps> <out_dir>
// Emits: canvas-ids.i32, u-step<k>.f32, renoise-step<k>.i32  (little-endian)

#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
    if (argc != 6) {
        fprintf(stderr, "usage: %s <seed> <n_vocab> <C> <n_steps> <out_dir>\n", argv[0]);
        return 1;
    }
    const int32_t     seed    = atoi(argv[1]);
    const int32_t     n_vocab = atoi(argv[2]);
    const int32_t     C       = atoi(argv[3]);
    const int32_t     n_steps = atoi(argv[4]);
    const std::string out_dir = argv[5];

    std::mt19937                           rng(seed);
    std::uniform_real_distribution<float>  uni01(0.0f, 1.0f);
    std::uniform_int_distribution<int32_t> vocab_dist(0, n_vocab - 1);

    std::vector<int32_t> canvas(C);
    for (int32_t i = 0; i < C; i++) {
        canvas[i] = vocab_dist(rng);
    }
    write_file(out_dir + "/canvas-ids.i32", canvas.data(), (size_t) C * 4);

    for (int32_t s = 0; s < n_steps; s++) {
        std::vector<float>   u(C);
        std::vector<int32_t> renoise(C);
        for (int32_t pos = 0; pos < C; pos++) {
            u[pos]       = uni01(rng);
            renoise[pos] = vocab_dist(rng);
        }
        write_file(out_dir + "/u-step" + std::to_string(s) + ".f32", u.data(), (size_t) C * 4);
        write_file(out_dir + "/renoise-step" + std::to_string(s) + ".i32", renoise.data(), (size_t) C * 4);
    }

    printf("{\"seed\":%d,\"n_vocab\":%d,\"C\":%d,\"n_steps\":%d}\n", seed, n_vocab, C, n_steps);
    return 0;
}
