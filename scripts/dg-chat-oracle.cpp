// dg-chat-oracle — reference oracle for the DiffusionGemma chat wrapper
// (Phase 6). Mirrors diffusion-cli.cpp's chat path exactly:
//   render: apply_template([{user, M}], add_generation_prompt=true) via the
//           model's own chat template + minja, then common_tokenize with
//           add_special=true / parse_special=true → prompt token ids.
//   decode: common_detokenize(ids, special=false) → text (the run_turn detok).
//
// All tokenizer / chat-template semantics are the work of the llama.cpp / ggml
// authors (https://github.com/ggml-org/llama.cpp, MIT). Compile against the
// PINNED checkout recorded in the lane's llamacpp-pin.json.
//
// Build (from the pinned checkout, after `cmake --build build --target llama-common`):
//   c++ -std=c++17 -O2 -I <pin>/include -I <pin>/common -I <pin>/ggml/include \
//       -I <pin>/vendor scripts/dg-chat-oracle.cpp \
//       -L <pin>/build/bin -lllama -lllama-common -Wl,-rpath,<pin>/build/bin -o <out>/dg-chat-oracle
// Run:
//   dg-chat-oracle <model.gguf> render "<user message>" <out_ids.i32>
//   dg-chat-oracle <model.gguf> decode <in_ids.i32>          # prints text to stdout

#include "llama.h"
#include "common.h"
#include "chat.h"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

static std::vector<int32_t> read_i32(const char * path) {
    std::ifstream f(path, std::ios::binary);
    std::vector<int32_t> v;
    int32_t x;
    while (f.read(reinterpret_cast<char *>(&x), 4)) {
        v.push_back(x);
    }
    return v;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <model.gguf> render \"<msg>\" <out_ids.i32>\n"
                        "       %s <model.gguf> decode <in_ids.i32>\n",
                argv[0], argv[0]);
        return 2;
    }
    const char * model_path = argv[1];
    const std::string mode = argv[2];

    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    mparams.vocab_only = true;
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "failed to load vocab from %s\n", model_path);
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    if (mode == "render") {
        const std::string msg = argv[3];
        const char * out_path = argc > 4 ? argv[4] : nullptr;
        common_chat_templates_ptr chat_templates = common_chat_templates_init(model, "");
        common_chat_msg m;
        m.role = "user";
        m.content = msg;
        common_chat_templates_inputs inputs;
        inputs.messages = {m};
        inputs.add_generation_prompt = true;
        const std::string formatted = common_chat_templates_apply(chat_templates.get(), inputs).prompt;
        // add_special=true, parse_special=true — identical to run_turn.
        const std::vector<llama_token> ids = common_tokenize(vocab, formatted, true, true);
        fprintf(stderr, "formatted prompt (%zu chars):\n%s\n---\n%zu tokens\n",
                formatted.size(), formatted.c_str(), ids.size());
        for (size_t i = 0; i < ids.size(); i++) {
            printf("%s%d", i ? " " : "", ids[i]);
        }
        printf("\n");
        if (out_path) {
            std::ofstream o(out_path, std::ios::binary);
            for (llama_token t : ids) {
                int32_t v = t;
                o.write(reinterpret_cast<const char *>(&v), 4);
            }
        }
    } else if (mode == "decode") {
        std::vector<int32_t> ids32 = read_i32(argv[3]);
        std::vector<llama_token> ids(ids32.begin(), ids32.end());
        // special=false — identical to run_turn's response detokenize.
        const std::string text = common_detokenize(vocab, ids, false);
        fwrite(text.data(), 1, text.size(), stdout);
        printf("\n");
    } else {
        fprintf(stderr, "unknown mode %s\n", mode.c_str());
        return 2;
    }

    llama_model_free(model);
    llama_backend_free();
    return 0;
}
