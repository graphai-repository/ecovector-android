#pragma once
#include <string>
#include <vector>
#include <memory>

// Forward declarations
namespace ecovector {
    class Tokenizer;
}
namespace Ort {
    class Session;
}

class OnnxRuntime {
public:
    OnnxRuntime();
    ~OnnxRuntime();

    // Load ONNX model from path
    bool load(const std::string& model_path);

    // Load tokenizer from path
    bool loadTokenizer(const std::string& tokenizer_path);

    // Embed single text and return embedding vector
    // optionally return token IDs via outTokenIds parameter
    std::vector<float> embed(const std::string& text, std::vector<int32_t>* outTokenIds = nullptr);

    // Batch embed multiple texts and return embeddings (batch_size x hidden_size)
    // optionally return token IDs via outTokenIds parameter
    std::vector<std::vector<float>> embedBatch(const std::vector<std::string>& texts,
                                                std::vector<std::vector<int32_t>>* outTokenIds = nullptr);

    // Normalize single vector using L2 normalization
    // v = v / ||v|| where ||v|| = sqrt(sum(v[i]^2))
    static void normalize(std::vector<float>& vector);

    // Normalize multiple vectors in batch
    static void normalizeBatch(std::vector<std::vector<float>>& embeddings);

    // Get access to tokenizer for external use
    ecovector::Tokenizer* getTokenizer() { return tokenizer_.get(); }

private:
    std::unique_ptr<ecovector::Tokenizer> tokenizer_;
    std::unique_ptr<Ort::Session> session_;  // ONNX Runtime session
};
