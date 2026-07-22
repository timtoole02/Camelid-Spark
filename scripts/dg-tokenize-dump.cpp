// dg-tokenize-dump — reference tokenization dumper for the DiffusionGemma
// lane's tokenizer-parity gate (Phase 1).
//
// Loads ONLY the vocab (vocab_only) from the tracked GGUF and, for each case
// in the prompt pack, emits llama.cpp's ground truth: rendered chat prompt
// (when mode == "chat", via the model's own chat template + minja), token ids,
// per-token pieces, and the detokenized string.
//
// All tokenizer and chat-template semantics here are the work of the
// llama.cpp / ggml authors (https://github.com/ggml-org/llama.cpp, MIT).
// Compile against the PINNED checkout recorded in the lane's
// llamacpp-pin.json.
//
// Build (from the pinned checkout, after `cmake --build build --target llama-common`):
//   c++ -std=c++17 -O2 -I <pin>/include -I <pin>/common -I <pin>/ggml/include \
//       -I <pin>/vendor scripts/dg-tokenize-dump.cpp \
//       -L <pin>/build/bin -lllama -lllama-common -Wl,-rpath,<pin>/build/bin \
//       -o <out>/dg-tokenize-dump
// Run:
//   dg-tokenize-dump <model.gguf> <pack.json> <out.json>

#include "llama.h"
#include "common.h"
#include "chat.h"

#include <nlohmann/json.hpp>

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using json = nlohmann::ordered_json;

// Token pieces may be PARTIAL UTF-8 (byte-fallback tokens split multi-byte
// characters); JSON strings cannot carry them, so pieces and detok are stored
// base64 to stay byte-exact.
static std::string b64(const std::string & in) {
    static const char * tbl = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve((in.size() + 2) / 3 * 4);
    size_t i = 0;
    while (i + 2 < in.size()) {
        const uint32_t v = ((uint8_t) in[i] << 16) | ((uint8_t) in[i + 1] << 8) | (uint8_t) in[i + 2];
        out += tbl[(v >> 18) & 63]; out += tbl[(v >> 12) & 63];
        out += tbl[(v >>  6) & 63]; out += tbl[v & 63];
        i += 3;
    }
    if (i + 1 == in.size()) {
        const uint32_t v = (uint8_t) in[i] << 16;
        out += tbl[(v >> 18) & 63]; out += tbl[(v >> 12) & 63]; out += "==";
    } else if (i + 2 == in.size()) {
        const uint32_t v = ((uint8_t) in[i] << 16) | ((uint8_t) in[i + 1] << 8);
        out += tbl[(v >> 18) & 63]; out += tbl[(v >> 12) & 63]; out += tbl[(v >> 6) & 63]; out += "=";
    }
    return out;
}

int main(int argc, char ** argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s <model.gguf> <pack.json> <out.json>\n", argv[0]);
        return 1;
    }
    const char * model_path = argv[1];
    const char * pack_path  = argv[2];
    const char * out_path   = argv[3];

    json pack;
    {
        std::ifstream in(pack_path);
        if (!in) {
            fprintf(stderr, "cannot open %s\n", pack_path);
            return 1;
        }
        in >> pack;
    }

    llama_backend_init();
    llama_model_params mparams = llama_model_default_params();
    mparams.vocab_only = true;
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "failed to load vocab from %s\n", model_path);
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    common_chat_templates_ptr chat_templates = common_chat_templates_init(model, "");

    json out;
    out["object"]   = "dg.tokenizer.reference.v1";
    out["model"]    = model_path;
    out["n_vocab"]  = llama_vocab_n_tokens(vocab);
    out["bos"]      = llama_vocab_bos(vocab);
    out["eos"]      = llama_vocab_eos(vocab);
    out["add_bos"]  = llama_vocab_get_add_bos(vocab);
    out["add_eos"]  = llama_vocab_get_add_eos(vocab);
    out["cases"]    = json::array();

    for (const auto & c : pack["cases"]) {
        const std::string id   = c["id"];
        const std::string mode = c["mode"];

        std::string text;
        bool parse_special = false;
        if (mode == "chat") {
            common_chat_templates_inputs inputs;
            for (const auto & m : c["messages"]) {
                common_chat_msg msg;
                msg.role    = m["role"];
                msg.content = m["content"];
                inputs.messages.push_back(msg);
            }
            inputs.add_generation_prompt = true;
            text = common_chat_templates_apply(chat_templates.get(), inputs).prompt;
            // mirrors examples/diffusion/diffusion-cli.cpp run_turn: rendered
            // chat prompts are tokenized with special parsing enabled
            parse_special = true;
        } else {
            text = c["text"];
            parse_special = false; // raw completion text: no special parsing
        }

        const std::vector<llama_token> tokens =
            common_tokenize(vocab, text, /*add_special=*/true, parse_special);

        json pieces = json::array();
        for (llama_token t : tokens) {
            pieces.push_back(b64(common_token_to_piece(vocab, t, /*special=*/true)));
        }

        json entry;
        entry["id"]         = id;
        entry["mode"]       = mode;
        entry["text_b64"]   = b64(text);
        entry["tokens"]     = tokens;
        entry["pieces_b64"] = pieces;
        entry["detok_b64"]  = b64(common_detokenize(vocab, tokens, /*special=*/true));
        out["cases"].push_back(entry);

        fprintf(stderr, "case %-18s mode=%-4s tokens=%zu\n", id.c_str(), mode.c_str(), tokens.size());
    }

    std::ofstream o(out_path);
    o << out.dump(1) << "\n";
    fprintf(stderr, "wrote %s\n", out_path);

    llama_model_free(model);
    llama_backend_free();
    return 0;
}
