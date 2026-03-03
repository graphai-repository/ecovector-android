#pragma once

#include <cstdint>
#include <vector>

namespace ecovector {

struct IndexSearchResult {
    uint64_t chunkId;
    float score;  // distance (vector) or relevance score (BM25)
};

}  // namespace ecovector
