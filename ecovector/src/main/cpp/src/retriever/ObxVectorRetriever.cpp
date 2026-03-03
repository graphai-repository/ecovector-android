#define LOG_TAG "ObxVectorRetriever"
#include "../common/Logging.h"
#include "ObxVectorRetriever.h"
#include "HydrationUtil.h"
#include "IndexSearchResult.h"
#include "../object_box/ObxManager.h"
#include "../embedder/Embedder.h"
#include <chrono>

namespace ecovector {

ObxVectorRetriever::ObxVectorRetriever(ObxManager* obxManager, Embedder* embedder)
    : obxManager_(obxManager), embedder_(embedder) {
    LOGD("ObxVectorRetriever constructed");
}

ObxVectorRetriever::~ObxVectorRetriever() {
    LOGD("ObxVectorRetriever destructor");
}

bool ObxVectorRetriever::isReady() const {
    return obxManager_ != nullptr;
}

std::string ObxVectorRetriever::getParamsSummary() const {
    std::string s = "top=" + std::to_string(params.topK);
    if (params.maxResultCount != 100) {
        s += ",maxRes=" + std::to_string(params.maxResultCount);
    }
    return s;
}

std::vector<ChunkSearchResult> ObxVectorRetriever::retrieve(
    const QueryBundle& query, const IRetriever::Params& overrideParams) {

    auto startTime = std::chrono::high_resolution_clock::now();
    const auto* op = dynamic_cast<const Params*>(&overrideParams);
    const uint32_t topK = overrideParams.topK;
    const uint32_t maxResVal = op ? op->maxResultCount : params.maxResultCount;

    if (query.embedding.empty()) {
        LOGE("QueryBundle embedding is empty");
        return {};
    }

    if (!obxManager_) {
        LOGE("ObxManager is null");
        return {};
    }

    // Ensure maxResultCount >= topK for quality
    const uint32_t effectiveMaxRes = std::max(maxResVal, topK);

    // 1. ObjectBox HNSW search → ID + distance
    //    maxResultCount doubles as efSearch; fetch all candidates (filter narrows later)
    auto searchResults = obxManager_->vectorSearch(
        query.embedding.data(), query.embedding.size(),
        effectiveMaxRes, effectiveMaxRes);

    if (searchResults.empty()) return {};

    // 2. Apply metadata filter (post-filtering: same pattern as VectorRetriever/BM25Retriever)
    std::vector<IndexSearchResult> idResults;
    idResults.reserve(searchResults.size());
    for (const auto& sr : searchResults) {
        if (query.filterChunkIds &&
            query.filterChunkIds->find(sr.chunkId) == query.filterChunkIds->end()) {
            continue;  // skip: not in allowed set
        }
        idResults.push_back({sr.chunkId, sr.distance});
    }

    if (idResults.size() > topK) {
        idResults.resize(topK);
    }

    if (idResults.empty()) return {};

    // 3. Hydrate
    auto results = hydrateSearchResults(idResults, obxManager_,
                                        true, true, true);

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime);

    LOGD("[BENCHMARK] ObxVectorRetriever: topK=%u, maxRes=%u, results=%zu, filtered=%s, time=%lld us",
         topK, effectiveMaxRes, results.size(),
         query.filterChunkIds ? "yes" : "no", (long long)duration.count());

    return results;
}

} // namespace ecovector
