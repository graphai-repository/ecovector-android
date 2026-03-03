#ifndef ECOVECTOR_EMBEDDER_H
#define ECOVECTOR_EMBEDDER_H

#include <string>
#include <vector>

class OnnxRuntime;

namespace ecovector {

class Tokenizer;

class Embedder {
public:
    Embedder(OnnxRuntime* onnx, Tokenizer* tokenizer);
    ~Embedder();

    std::vector<float> embed(const std::string& text);
    std::vector<std::vector<float>> embedBatch(const std::vector<std::string>& texts);

    OnnxRuntime* getOnnxRuntime() const { return onnx_; }
    Tokenizer* getTokenizer() const { return tokenizer_; }

private:
    OnnxRuntime* onnx_;      // non-owning
    Tokenizer* tokenizer_;   // non-owning
};

} // namespace ecovector

#endif // ECOVECTOR_EMBEDDER_H
