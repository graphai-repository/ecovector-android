#pragma once
#include <string>
#include <vector>
#include <cstdint>

namespace ecovector {

class Tokenizer;

/**
 * Token-aware sentence-preserving chunker.
 *
 * Splits text into segments (by newlines, then sentence-ending punctuation),
 * counts tokens per segment using HuggingFace tokenizer, and greedily
 * accumulates segments up to maxTokens. Overlap is achieved by carrying
 * trailing segments totaling >= overlapTokens into the next chunk.
 *
 * Default: 216 tokens per chunk, 128 token overlap (step = 88 tokens).
 */
class TokenAwareChunker {
public:
    static constexpr int DEFAULT_MAX_TOKENS = 216;
    static constexpr int DEFAULT_OVERLAP_TOKENS = 128;

    TokenAwareChunker(Tokenizer* tokenizer,
                      int maxTokens = DEFAULT_MAX_TOKENS,
                      int overlapTokens = DEFAULT_OVERLAP_TOKENS);
    ~TokenAwareChunker() = default;

    std::vector<std::string> chunk(const std::string& text);

private:
    Tokenizer* tokenizer_;
    int maxTokens_;
    int overlapTokens_;

    /**
     * Split text into atomic segments preserving sentence boundaries.
     * 1) Split by newlines
     * 2) Within each line, split by sentence-ending punctuation (.?!) followed by space
     * Each segment is trimmed; empty segments are discarded.
     */
    std::vector<std::string> splitSegments(const std::string& text);
};

} // namespace ecovector
