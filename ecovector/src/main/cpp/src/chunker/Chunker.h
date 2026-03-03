#pragma once
#include <string>
#include <vector>

class Chunker {
public:
    Chunker() = default;
    ~Chunker() = default;

    // Split text into chunks based on word count (space-separated)
    std::vector<std::string> chunk(
        const std::string& text,
        int32_t chunk_size,
        int32_t sliding_step
    );

private:
    std::vector<std::string> splitByWhitespace(const std::string& text);
};
