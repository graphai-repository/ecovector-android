#pragma once

#include "IndexSearchResult.h"
#include "../object_box/ObxManager.h"

#include <vector>
#include <unordered_map>
#include <cstdint>

namespace ecovector {

/**
 * Index 검색 결과(ID+score)를 ObxManager에서 청크 데이터를 조회하여
 * ChunkSearchResult로 조립하는 공통 유틸리티.
 */
inline std::vector<ChunkSearchResult> hydrateSearchResults(
    const std::vector<IndexSearchResult>& idResults,
    ObxManager* obxManager,
    bool excludeVectors = true,
    bool excludeTokenIds = true,
    bool excludeKiwiTokens = true) {

    if (idResults.empty() || !obxManager) return {};

    std::vector<uint64_t> chunkIds;
    chunkIds.reserve(idResults.size());
    for (const auto& ir : idResults) {
        chunkIds.push_back(ir.chunkId);
    }

    auto chunks = obxManager->getChunksByIds(chunkIds, excludeVectors, excludeTokenIds, excludeKiwiTokens);

    // Build lookup map: chunkId → ChunkData
    std::unordered_map<uint64_t, ChunkData> chunkMap;
    chunkMap.reserve(chunks.size());
    for (auto& c : chunks) {
        uint64_t id = c.id;
        chunkMap.emplace(id, std::move(c));
    }

    // Assemble results preserving original order
    std::vector<ChunkSearchResult> results;
    results.reserve(idResults.size());
    for (const auto& ir : idResults) {
        auto it = chunkMap.find(ir.chunkId);
        if (it != chunkMap.end()) {
            ChunkSearchResult csr;
            csr.chunk = std::move(it->second);
            csr.distance = ir.score;
            results.push_back(std::move(csr));
        }
    }
    return results;
}

}  // namespace ecovector
