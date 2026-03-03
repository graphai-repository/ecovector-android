#include "Tokenizer.h"
#include <tokenizers_cpp.h>

#include <fstream>
#include <sstream>

namespace ecovector {

Tokenizer::~Tokenizer() {
    // tokenizer_ is a unique_ptr, so it will be automatically deleted
}

bool Tokenizer::load(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) return false;

    std::stringstream buffer;
    buffer << ifs.rdbuf();
    std::string json = buffer.str();

    try {
        tokenizer_ = tokenizers::Tokenizer::FromBlobJSON(json);
        return tokenizer_ != nullptr;
    } catch (...) {
        return false;
    }
}

std::vector<int32_t> Tokenizer::encode(const std::string& text) {
    std::vector<int32_t> ids;
    if (!tokenizer_) return ids;

    try {
        // tokenizers-cpp does not add special tokens automatically
        // We need to manually add [CLS] at the beginning and [SEP] at the end
        // for BERT-based models (KoSimCSE-BERT uses these IDs)
        constexpr int32_t CLS_TOKEN_ID = 2;  // [CLS] token
        constexpr int32_t SEP_TOKEN_ID = 3;  // [SEP] token
        constexpr int32_t PAD_TOKEN_ID = 0;  // [PAD] token
        constexpr size_t MAX_SEQ_LENGTH = 512;  // BERT max position embeddings

        std::vector<int32_t> raw_ids = tokenizer_->Encode(text);

        // Remove trailing padding (0s) that tokenizers-cpp may have added
        while (!raw_ids.empty() && raw_ids.back() == PAD_TOKEN_ID) {
            raw_ids.pop_back();
        }

        // Build final token sequence: [CLS] + tokens + [SEP]
        // No padding — ONNX model accepts dynamic sequence length

        // Add [CLS] token
        ids.push_back(CLS_TOKEN_ID);

        // Add actual tokens (truncate if necessary to fit CLS + tokens + SEP within MAX_SEQ_LENGTH)
        size_t max_tokens = MAX_SEQ_LENGTH - 2;  // Reserve space for CLS and SEP
        size_t tokens_to_add = std::min(raw_ids.size(), max_tokens);
        ids.insert(ids.end(), raw_ids.begin(), raw_ids.begin() + tokens_to_add);

        // Add [SEP] token
        ids.push_back(SEP_TOKEN_ID);
    } catch (...) {
        // Return empty vector on error
    }
    return ids;
}

std::string Tokenizer::decode(const std::vector<int32_t>& ids) {
    if (!tokenizer_) return "";

    try {
        return tokenizer_->Decode(ids);
    } catch (...) {
        return "";
    }
}

} // namespace ecovector
