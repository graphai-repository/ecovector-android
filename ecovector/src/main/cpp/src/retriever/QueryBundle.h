#ifndef ECOVECTOR_RETRIEVER_QUERY_BUNDLE_H
#define ECOVECTOR_RETRIEVER_QUERY_BUNDLE_H

#include <string>
#include <vector>
#include <cstdint>
#include <unordered_set>

namespace ecovector {

struct QueryBundle {
    std::string rawText;
    std::vector<float> embedding;
    std::vector<int32_t> kiwiTokens;

    // Optional pre-resolved filter: only search within these chunk IDs.
    // Non-owning pointer — caller must ensure lifetime.
    const std::unordered_set<uint64_t>* filterChunkIds = nullptr;
};

} // namespace ecovector

#endif // ECOVECTOR_RETRIEVER_QUERY_BUNDLE_H
