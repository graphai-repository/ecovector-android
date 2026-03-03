#include "TokenAwareChunker.h"
#include "Tokenizer.h"
#include <sstream>
#include <algorithm>
#include <numeric>

namespace ecovector {

TokenAwareChunker::TokenAwareChunker(Tokenizer* tokenizer, int maxTokens, int overlapTokens)
    : tokenizer_(tokenizer), maxTokens_(maxTokens), overlapTokens_(overlapTokens) {}

std::vector<std::string> TokenAwareChunker::splitSegments(const std::string& text) {
    std::vector<std::string> segments;

    // 1) Split by newlines
    std::istringstream stream(text);
    std::string line;

    while (std::getline(stream, line)) {
        // Trim
        size_t s = line.find_first_not_of(" \t\r");
        if (s == std::string::npos) continue;
        size_t e = line.find_last_not_of(" \t\r");
        std::string trimmed = line.substr(s, e - s + 1);
        if (trimmed.empty()) continue;

        // 2) Split by sentence-ending punctuation followed by whitespace or end of string
        size_t sentStart = 0;
        for (size_t i = 0; i < trimmed.size(); ++i) {
            char c = trimmed[i];
            bool isSentEnd = (c == '.' || c == '?' || c == '!');
            if (isSentEnd) {
                bool atEnd = (i + 1 >= trimmed.size());
                bool followedBySpace = (!atEnd && (trimmed[i + 1] == ' ' || trimmed[i + 1] == '\t'));
                if (atEnd || followedBySpace) {
                    std::string seg = trimmed.substr(sentStart, i + 1 - sentStart);
                    size_t ss = seg.find_first_not_of(" \t");
                    if (ss != std::string::npos) {
                        size_t se = seg.find_last_not_of(" \t");
                        segments.push_back(seg.substr(ss, se - ss + 1));
                    }
                    sentStart = i + 1;
                    while (sentStart < trimmed.size() &&
                           (trimmed[sentStart] == ' ' || trimmed[sentStart] == '\t')) {
                        ++sentStart;
                    }
                    i = sentStart > 0 ? sentStart - 1 : 0;
                }
            }
        }
        // Remaining text after last sentence break
        if (sentStart < trimmed.size()) {
            std::string remaining = trimmed.substr(sentStart);
            size_t ss = remaining.find_first_not_of(" \t");
            if (ss != std::string::npos) {
                size_t se = remaining.find_last_not_of(" \t");
                segments.push_back(remaining.substr(ss, se - ss + 1));
            }
        }
    }

    return segments;
}

std::vector<std::string> TokenAwareChunker::chunk(const std::string& text) {
    if (!tokenizer_ || text.empty()) return {};

    auto segments = splitSegments(text);
    if (segments.empty()) return {};

    // Pre-count tokens for each segment
    std::vector<int> tokenCounts(segments.size());
    for (size_t i = 0; i < segments.size(); ++i) {
        tokenCounts[i] = tokenizer_->countTokens(segments[i]);
    }

    // If everything fits in one chunk, return as-is
    int totalTokens = std::accumulate(tokenCounts.begin(), tokenCounts.end(), 0);
    if (totalTokens <= maxTokens_) {
        std::string joined;
        for (size_t i = 0; i < segments.size(); ++i) {
            if (i > 0) joined += ' ';
            joined += segments[i];
        }
        return {joined};
    }

    std::vector<std::string> chunks;
    size_t i = 0;

    while (i < segments.size()) {
        // Greedily accumulate segments until exceeding token budget
        std::vector<size_t> selected;
        int currentTokens = 0;

        size_t j = i;
        while (j < segments.size()) {
            int segTokens = tokenCounts[j];
            // Special case: single segment exceeds budget — include it anyway
            if (selected.empty()) {
                selected.push_back(j);
                currentTokens += segTokens;
                ++j;
                if (currentTokens >= maxTokens_) break;
                continue;
            }
            if (currentTokens + segTokens > maxTokens_) break;
            selected.push_back(j);
            currentTokens += segTokens;
            ++j;
        }

        // Build chunk string from selected segments
        std::string chunkStr;
        for (size_t k = 0; k < selected.size(); ++k) {
            if (k > 0) chunkStr += ' ';
            chunkStr += segments[selected[k]];
        }
        chunks.push_back(std::move(chunkStr));

        // Determine next start: find overlap point
        if (j >= segments.size()) break;  // No more segments

        int overlapAccum = 0;
        size_t overlapStart = j;  // Default: no overlap (start right after selected)
        for (int k = static_cast<int>(selected.size()) - 1; k >= 0; --k) {
            overlapAccum += tokenCounts[selected[k]];
            if (overlapAccum >= overlapTokens_) {
                overlapStart = selected[k];
                break;
            }
            overlapStart = selected[k];
        }

        // Advance: next chunk starts at overlapStart
        if (overlapStart <= i) {
            // Safety: always advance at least 1 segment
            i = (selected.size() > 1) ? selected[1] : j;
        } else {
            i = overlapStart;
        }
    }

    return chunks;
}

} // namespace ecovector
