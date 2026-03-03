#pragma once
#include <string>
#include <vector>
#include <cstdint>

/**
 * Sentence-boundary-aware chunker.
 *
 * Accumulates sentences until character limit is reached.
 * Overlaps last N sentences into the next chunk for context continuity.
 *
 * Character-based estimation: Korean BERT subword tokenizer
 * 1 Korean char ~ 1.3-1.5 tokens -> 300 chars ~ 390-450 tokens
 */
class SentenceChunker {
public:
    SentenceChunker(int maxChars = DEFAULT_MAX_CHARS,
                    int overlapSentences = DEFAULT_OVERLAP_SENTENCES);
    ~SentenceChunker() = default;

    std::vector<std::string> chunk(const std::string& text);

    static constexpr int DEFAULT_MAX_CHARS = 300;
    static constexpr int DEFAULT_OVERLAP_SENTENCES = 2;

private:
    int maxChars_;
    int overlapSentences_;

    std::vector<std::string> splitSentences(const std::string& text);
    static int charCount(const std::string& text);
};
