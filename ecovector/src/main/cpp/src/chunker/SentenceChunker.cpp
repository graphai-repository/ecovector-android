#include "SentenceChunker.h"
#include <sstream>
#include <algorithm>

SentenceChunker::SentenceChunker(int maxChars, int overlapSentences)
    : maxChars_(maxChars), overlapSentences_(overlapSentences) {}

int SentenceChunker::charCount(const std::string& text) {
    // Count non-whitespace characters (token count estimation for Korean)
    int count = 0;
    const auto* p = reinterpret_cast<const unsigned char*>(text.data());
    const auto* end = p + text.size();

    while (p < end) {
        // Skip whitespace (ASCII space, tab, newline, etc.)
        if (*p <= 0x20) {
            ++p;
            continue;
        }
        // Count one "character" (advance past full UTF-8 sequence)
        if (*p < 0x80) {
            ++p;
        } else if (*p < 0xE0) {
            p += 2;
        } else if (*p < 0xF0) {
            p += 3;
        } else {
            p += 4;
        }
        ++count;
    }
    return count;
}

std::vector<std::string> SentenceChunker::splitSentences(const std::string& text) {
    std::vector<std::string> sentences;

    std::istringstream stream(text);
    std::string line;

    while (std::getline(stream, line)) {
        size_t start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos) continue;
        size_t end = line.find_last_not_of(" \t\r\n");
        std::string trimmed = line.substr(start, end - start + 1);
        if (trimmed.empty()) continue;

        // Split after [.?!] followed by whitespace (no regex)
        size_t sentStart = 0;
        for (size_t i = 0; i < trimmed.size(); ++i) {
            char c = trimmed[i];
            if ((c == '.' || c == '?' || c == '!') && i + 1 < trimmed.size()
                && (trimmed[i + 1] == ' ' || trimmed[i + 1] == '\t')) {
                std::string sentence = trimmed.substr(sentStart, i + 1 - sentStart);
                size_t ss = sentence.find_first_not_of(" \t");
                if (ss != std::string::npos) {
                    size_t se = sentence.find_last_not_of(" \t");
                    sentences.push_back(sentence.substr(ss, se - ss + 1));
                }
                sentStart = i + 1;
                while (sentStart < trimmed.size()
                       && (trimmed[sentStart] == ' ' || trimmed[sentStart] == '\t')) {
                    ++sentStart;
                }
                i = sentStart > 0 ? sentStart - 1 : 0;
            }
        }
        if (sentStart < trimmed.size()) {
            std::string remaining = trimmed.substr(sentStart);
            size_t ss = remaining.find_first_not_of(" \t");
            if (ss != std::string::npos) {
                size_t se = remaining.find_last_not_of(" \t");
                sentences.push_back(remaining.substr(ss, se - ss + 1));
            }
        }
    }

    return sentences;
}

std::vector<std::string> SentenceChunker::chunk(const std::string& text) {
    auto sentences = splitSentences(text);
    if (sentences.empty()) return {};

    // If everything fits in one chunk
    std::string joined;
    for (size_t i = 0; i < sentences.size(); ++i) {
        if (i > 0) joined += ' ';
        joined += sentences[i];
    }
    if (charCount(joined) <= maxChars_) return {joined};

    std::vector<std::string> chunks;
    size_t i = 0;

    while (i < sentences.size()) {
        std::vector<std::string> selected = {sentences[i]};

        size_t j = i + 1;
        while (j < sentences.size()) {
            std::string candidate;
            for (size_t k = 0; k < selected.size(); ++k) {
                if (k > 0) candidate += ' ';
                candidate += selected[k];
            }
            candidate += ' ';
            candidate += sentences[j];

            if (charCount(candidate) > maxChars_) break;
            selected.push_back(sentences[j]);
            ++j;
        }

        // Build chunk string
        std::string chunkStr;
        for (size_t k = 0; k < selected.size(); ++k) {
            if (k > 0) chunkStr += ' ';
            chunkStr += selected[k];
        }
        chunks.push_back(std::move(chunkStr));

        int consumed = static_cast<int>(selected.size());
        if (consumed <= overlapSentences_) {
            i += std::max(1, consumed);
        } else {
            i += consumed - overlapSentences_;
        }
    }

    return chunks;
}
