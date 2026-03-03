#include "Chunker.h"
#include <sstream>
#include <algorithm>

std::vector<std::string> Chunker::splitByWhitespace(const std::string& text) {
    std::vector<std::string> words;
    std::istringstream iss(text);
    std::string word;

    while (iss >> word) {
        words.push_back(word);
    }

    return words;
}

std::vector<std::string> Chunker::chunk(
    const std::string& text,
    int32_t chunk_size,
    int32_t sliding_step
) {
    std::vector<std::string> chunks;

    if (chunk_size <= 0 || sliding_step <= 0) {
        return chunks;
    }

    // Split text into words
    std::vector<std::string> words = splitByWhitespace(text);
    if (words.empty()) {
        return chunks;
    }

    // Create chunks with sliding window
    int32_t total_words = static_cast<int32_t>(words.size());

    int32_t start = 0;
    while (start < total_words) {
        int32_t end = std::min(start + chunk_size, total_words);

        // Extract chunk and join with spaces
        std::string chunk_text;
        for (int32_t i = start; i < end; ++i) {
            if (i > start) {
                chunk_text += " ";
            }
            chunk_text += words[i];
        }

        chunks.push_back(chunk_text);

        // If we've reached the end, stop
        if (end >= total_words) {
            break;
        }

        start += sliding_step;
    }

    return chunks;
}
