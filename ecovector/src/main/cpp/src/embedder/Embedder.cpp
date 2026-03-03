#include "Embedder.h"
#include "OnnxRuntime.h"
#include "Tokenizer.h"

namespace ecovector {

Embedder::Embedder(OnnxRuntime* onnx, Tokenizer* tokenizer)
    : onnx_(onnx), tokenizer_(tokenizer) {}

Embedder::~Embedder() = default;

std::vector<float> Embedder::embed(const std::string& text) {
    auto result = onnx_->embed(text);
    OnnxRuntime::normalize(result);
    return result;
}

std::vector<std::vector<float>> Embedder::embedBatch(const std::vector<std::string>& texts) {
    auto results = onnx_->embedBatch(texts);
    OnnxRuntime::normalizeBatch(results);
    return results;
}

} // namespace ecovector
