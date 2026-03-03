#define LOG_TAG "VectorRetriever"
#include "../common/Logging.h"
#include "VectorRetriever.h"
#include "IndexSearchResult.h"
#include "HydrationUtil.h"
#include "../eco_vector/EcoVectorIndex.h"
#include "../object_box/ObxManager.h"
#include "../embedder/Embedder.h"

#include <chrono>
#include <thread>

namespace ecovector {

VectorRetriever::VectorRetriever(EcoVectorIndex* ecoVectorIndex,
                                 ObxManager* obxManager,
                                 Embedder* embedder)
    : ecoVectorIndex_(ecoVectorIndex)
    , obxManager_(obxManager)
    , embedder_(embedder) {
    LOGD("VectorRetriever constructed (embedder=%p)", embedder);
}

VectorRetriever::~VectorRetriever() {
    LOGD("VectorRetriever destructor");
}

bool VectorRetriever::isReady() const {
    return ecoVectorIndex_ && ecoVectorIndex_->isIndexReady();
}

void VectorRetriever::warmup() {
    if (ecoVectorIndex_) {
        ecoVectorIndex_->preloadIndexes();
    }
}

std::vector<ChunkSearchResult> VectorRetriever::retrieve(
    const QueryBundle& query, const IRetriever::Params& overrideParams) {

    auto startTime = std::chrono::high_resolution_clock::now();
    const auto* vp = dynamic_cast<const Params*>(&overrideParams);
    const uint32_t topK = overrideParams.topK;
    const size_t nprobeVal = vp ? vp->nprobe : params.nprobe;
    const size_t efSearchVal = vp ? vp->efSearch : params.efSearch;

    if (query.embedding.empty()) {
        LOGE("QueryBundle embedding is empty");
        return {};
    }

    if (!ecoVectorIndex_ || !obxManager_) {
        LOGE("EcoVector or ObxManager is null");
        return {};
    }

    if (!isReady()) {
        LOGE("EcoVector index not ready");
        return {};
    }

    auto tid = std::hash<std::thread::id>{}(std::this_thread::get_id()) % 10000;

    // 1. Index-level search: ID+distance only
    auto tSearch0 = std::chrono::high_resolution_clock::now();
    auto idResults = ecoVectorIndex_->searchIds(
        query.embedding, topK, query.filterChunkIds, nprobeVal, efSearchVal);
    auto tSearch1 = std::chrono::high_resolution_clock::now();
    auto searchUs = std::chrono::duration_cast<std::chrono::microseconds>(tSearch1 - tSearch0).count();

    if (idResults.empty()) return {};

    // Truncate before hydration — saves ObjectBox lookups
    if (idResults.size() > topK) {
        idResults.resize(topK);
    }

    // 2. Hydrate: ObxManager에서 청크 데이터 조회 + ChunkSearchResult 조립
    auto tHydrate0 = std::chrono::high_resolution_clock::now();
    auto results = hydrateSearchResults(idResults, obxManager_,
                                        /*excludeVectors=*/true, /*excludeTokenIds=*/true,
                                        /*excludeKiwiTokens=*/true);
    auto tHydrate1 = std::chrono::high_resolution_clock::now();
    auto hydrateUs = std::chrono::duration_cast<std::chrono::microseconds>(tHydrate1 - tHydrate0).count();

    // 3. Final truncation to topK
    if (results.size() > topK) results.resize(topK);

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        endTime - startTime);

    LOGD("[TIMING] VectorRetriever: searchIds=%lld us, hydrate=%lld us, total=%lld us, "
         "idResults=%zu, filter=%s, thread=%zu",
         (long long)searchUs, (long long)hydrateUs, (long long)duration.count(),
         idResults.size(),
         query.filterChunkIds ? std::to_string(query.filterChunkIds->size()).c_str() : "none",
         tid);

    LOGD("[BENCHMARK] VectorRetriever: topK=%u, results=%zu, "
         "nprobe=%zu, efSearch=%zu, time=%lld us",
         topK, results.size(),
         nprobeVal, efSearchVal,
         (long long)duration.count());

    return results;
}

std::vector<ChunkSearchResult> VectorRetriever::retrieve(
    const std::string& text, const Params* overrideParams) {

    if (!embedder_) {
        LOGE("Embedder is null, cannot embed text");
        return {};
    }

    auto embedding = embedder_->embed(text);
    if (embedding.empty()) {
        LOGE("Embedder returned empty embedding for text");
        return {};
    }

    const Params& p = overrideParams ? *overrideParams : params;

    QueryBundle query;
    query.rawText = text;
    query.embedding = std::move(embedding);

    return retrieve(query, p);
}

std::vector<ChunkSearchResult> VectorRetriever::retrieve(
    const float* embedding, size_t dim,
    const Params* overrideParams) {

    if (!embedding || dim == 0) {
        LOGE("Invalid embedding pointer or dimension");
        return {};
    }

    const Params& p = overrideParams ? *overrideParams : params;

    QueryBundle query;
    query.embedding.assign(embedding, embedding + dim);

    return retrieve(query, p);
}

} // namespace ecovector
