#pragma once

#include "object_box/ObxManager.h"
#include <vector>
#include <unordered_set>
#include <algorithm>

namespace ecovector {

// Deduplicate search results by document_id, keeping the highest-scoring
// chunk per document. Input must be pre-sorted by score (descending).
// Returns at most topK unique-document results.
inline std::vector<ChunkSearchResult> deduplicateByDocument(
    std::vector<ChunkSearchResult>&& results, uint32_t topK) {

    std::vector<ChunkSearchResult> deduped;
    deduped.reserve(std::min(static_cast<size_t>(topK), results.size()));

    std::unordered_set<uint64_t> seenDocumentIds;
    for (auto& sr : results) {
        if (seenDocumentIds.count(sr.chunk.documentId) > 0) continue;
        seenDocumentIds.insert(sr.chunk.documentId);
        deduped.push_back(std::move(sr));
        if (deduped.size() >= topK) break;
    }

    return deduped;
}

// Const-ref overload (copies instead of moves)
inline std::vector<ChunkSearchResult> deduplicateByDocument(
    const std::vector<ChunkSearchResult>& results, uint32_t topK) {

    std::vector<ChunkSearchResult> deduped;
    deduped.reserve(std::min(static_cast<size_t>(topK), results.size()));

    std::unordered_set<uint64_t> seenDocumentIds;
    for (const auto& sr : results) {
        if (seenDocumentIds.count(sr.chunk.documentId) > 0) continue;
        seenDocumentIds.insert(sr.chunk.documentId);
        deduped.push_back(sr);
        if (deduped.size() >= topK) break;
    }

    return deduped;
}

} // namespace ecovector
