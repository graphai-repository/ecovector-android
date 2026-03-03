// app/src/main/cpp/src/retriever/BM25Retriever.cpp
#define LOG_TAG "BM25Retriever"
#include "../common/Logging.h"
#include "BM25Retriever.h"
#include "IndexSearchResult.h"
#include "HydrationUtil.h"
#include "../bm25/BM25Index.h"
#include "../object_box/ObxManager.h"
#include "../kiwi/KiwiTokenizer.h"
#include "../kiwi/KiwiHashUtil.h"

#include <chrono>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <thread>

namespace ecovector {

BM25Retriever::BM25Retriever(BM25Index* bm25Index, ObxManager* obxManager,
                             KiwiTokenizer* kiwiTokenizer)
    : bm25Index_(bm25Index)
    , obxManager_(obxManager)
    , kiwiTokenizer_(kiwiTokenizer) {
    LOGD("BM25Retriever constructed (kiwiTokenizer=%p)", kiwiTokenizer);
}

BM25Retriever::~BM25Retriever() {
    LOGD("BM25Retriever destructor");
}

bool BM25Retriever::isReady() const {
    return bm25Index_ && bm25Index_->isIndexReady();
}

BM25Index::Parameters BM25Retriever::toBM25Params() const {
    BM25Index::Parameters p;
    p.k1 = params.k1;
    p.b = params.b;
    p.idfThreshold = params.idfThreshold;
    p.maxSeedTerms = params.maxSeedTerms;
    p.candidateMultiplier = params.candidateMultiplier;
    p.minCandidates = params.minCandidates;
    p.minScore = params.minScore;
    return p;
}

std::vector<ChunkSearchResult> BM25Retriever::retrieve(
    const QueryBundle& query, const IRetriever::Params& overrideParams) {

    auto startTime = std::chrono::high_resolution_clock::now();
    const auto* bp = dynamic_cast<const Params*>(&overrideParams);
    const uint32_t topK = overrideParams.topK;
    // BM25-specific params: use override if available, else defaults
    const float k1Val = bp ? bp->k1 : params.k1;
    const float bVal = bp ? bp->b : params.b;
    const float idfThresholdVal = bp ? bp->idfThreshold : params.idfThreshold;
    const size_t maxSeedTermsVal = bp ? bp->maxSeedTerms : params.maxSeedTerms;
    const size_t candidateMultiplierVal = bp ? bp->candidateMultiplier : params.candidateMultiplier;
    const size_t minCandidatesVal = bp ? bp->minCandidates : params.minCandidates;
    const float minScoreVal = bp ? bp->minScore : params.minScore;
    const bool rm3EnabledVal = bp ? bp->rm3Enabled : params.rm3Enabled;
    const uint32_t rm3FbDocsVal = bp ? bp->rm3FbDocs : params.rm3FbDocs;
    const uint32_t rm3FbTermsVal = bp ? bp->rm3FbTerms : params.rm3FbTerms;
    const float rm3OrigWeightVal = bp ? bp->rm3OrigWeight : params.rm3OrigWeight;
    const uint32_t rm3MinDfVal = bp ? bp->rm3MinDf : params.rm3MinDf;

    if (query.kiwiTokens.empty()) {
        LOGE("QueryBundle kiwiTokens is empty");
        return {};
    }

    if (!isReady()) {
        LOGE("BM25 index not ready");
        return {};
    }

    // Build BM25Index::Parameters from override/default params (thread-safe, no shared mutation)
    BM25Index::Parameters bm25Params;
    bm25Params.k1 = k1Val;
    bm25Params.b = bVal;
    bm25Params.idfThreshold = idfThresholdVal;
    bm25Params.maxSeedTerms = maxSeedTermsVal;
    bm25Params.candidateMultiplier = candidateMultiplierVal;
    bm25Params.minCandidates = minCandidatesVal;
    bm25Params.minScore = minScoreVal;

    // RM3 query expansion: 2-pass search with language model scoring
    bool useWeightedSearch = false;
    std::vector<std::pair<int32_t, float>> rm3WeightedTerms;

    if (rm3EnabledVal && rm3FbDocsVal > 0 && rm3FbTermsVal > 0) {
        // Pass 1: standard BM25 search → top fbDocs
        auto firstPassResults = bm25Index_->searchIds(
            query.kiwiTokens, rm3FbDocsVal, query.filterChunkIds, &bm25Params);

        if (!firstPassResults.empty()) {
            // Fetch kiwiTokens from feedback docs
            std::vector<uint64_t> chunkIds;
            chunkIds.reserve(firstPassResults.size());
            for (const auto& r : firstPassResults) chunkIds.push_back(r.chunkId);
            auto chunks = obxManager_->getChunksByIds(
                chunkIds, /*excludeVectors=*/true, /*excludeTokenIds=*/true,
                /*excludeKiwiTokens=*/false);

            // Score normalization: P(D|Q) ∝ BM25_score
            float scoreSum = 0.0f;
            for (const auto& r : firstPassResults) scoreSum += r.score;

            // P_RM1(w) = Σ_D [ P(w|D) × P(D|Q) ]
            std::unordered_map<int32_t, float> termScores;
            for (size_t di = 0; di < firstPassResults.size() && di < chunks.size(); di++) {
                float docWeight = (scoreSum > 0.0f) ? firstPassResults[di].score / scoreSum : 1.0f / firstPassResults.size();
                const auto& chunk = chunks[di];
                if (chunk.kiwiTokens.empty()) continue;
                float docLen = static_cast<float>(chunk.kiwiTokens.size());

                // Compute TF for this chunk
                std::unordered_map<int32_t, uint32_t> tf;
                for (int32_t token : chunk.kiwiTokens) tf[token]++;

                for (const auto& [token, count] : tf) {
                    float pw_d = static_cast<float>(count) / docLen;  // P(w|D)
                    termScores[token] += pw_d * docWeight;
                }
            }

            // DF filtering: remove terms with DF < rm3MinDf
            for (auto it = termScores.begin(); it != termScores.end(); ) {
                if (bm25Index_->getTokenDF(it->first) < rm3MinDfVal) {
                    it = termScores.erase(it);
                } else {
                    ++it;
                }
            }

            // L1 normalize P_RM1
            float norm = 0.0f;
            for (const auto& [_, score] : termScores) norm += score;
            if (norm > 0.0f) {
                for (auto& [_, score] : termScores) score /= norm;
            }

            // Select top fbTerms by P_RM1 score
            std::vector<std::pair<int32_t, float>> sorted(termScores.begin(), termScores.end());
            size_t takeCount = std::min(static_cast<size_t>(rm3FbTermsVal), sorted.size());
            if (takeCount > 0) {
                std::partial_sort(sorted.begin(), sorted.begin() + takeCount, sorted.end(),
                    [](const auto& a, const auto& b) { return a.second > b.second; });
                sorted.resize(takeCount);
            }

            // Lambda interpolation: P_RM3(w) = λ × P_ML(w|Q) + (1-λ) × P_RM1(w)
            float lambda = rm3OrigWeightVal;

            // P_ML(w|Q) for original query terms
            std::unordered_map<int32_t, float> queryML;
            for (int32_t token : query.kiwiTokens) queryML[token] += 1.0f;
            float qLen = static_cast<float>(query.kiwiTokens.size());
            for (auto& [_, freq] : queryML) freq /= qLen;

            // Combine into RM3 weighted terms
            std::unordered_map<int32_t, float> rm3Terms;
            for (const auto& [token, prob] : queryML) {
                rm3Terms[token] += lambda * prob;
            }
            for (const auto& [token, prob] : sorted) {
                rm3Terms[token] += (1.0f - lambda) * prob;
            }

            // Convert to weighted query vector
            rm3WeightedTerms.reserve(rm3Terms.size());
            for (const auto& [token, weight] : rm3Terms) {
                if (weight > 0.0f) rm3WeightedTerms.emplace_back(token, weight);
            }

            useWeightedSearch = !rm3WeightedTerms.empty();

            LOGD("[RM3] Expanded: %zu query + %zu expansion = %zu total terms",
                 query.kiwiTokens.size(), takeCount, rm3WeightedTerms.size());
        }
    }

    auto tid = std::hash<std::thread::id>{}(std::this_thread::get_id()) % 10000;

    // Main search (Pass 2 if RM3, sole pass otherwise)
    auto tSearch0 = std::chrono::high_resolution_clock::now();
    std::vector<IndexSearchResult> idResults;
    if (useWeightedSearch) {
        idResults = bm25Index_->searchIdsWeighted(
            rm3WeightedTerms, topK, query.filterChunkIds, &bm25Params);
    } else {
        idResults = bm25Index_->searchIds(
            query.kiwiTokens, topK, query.filterChunkIds, &bm25Params);
    }
    auto tSearch1 = std::chrono::high_resolution_clock::now();
    auto searchUs = std::chrono::duration_cast<std::chrono::microseconds>(tSearch1 - tSearch0).count();

    // Hydrate (BM25 doesn't need vectors, HF tokenIds, or kiwi tokens)
    auto tHydrate0 = std::chrono::high_resolution_clock::now();
    auto results = hydrateSearchResults(idResults, obxManager_,
                                        /*excludeVectors=*/true, /*excludeTokenIds=*/true,
                                        /*excludeKiwiTokens=*/true);
    auto tHydrate1 = std::chrono::high_resolution_clock::now();
    auto hydrateUs = std::chrono::duration_cast<std::chrono::microseconds>(tHydrate1 - tHydrate0).count();

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        endTime - startTime);

    LOGD("[TIMING] BM25Retriever: searchIds=%lld us, hydrate=%lld us, total=%lld us, "
         "idResults=%zu, filter=%s, thread=%zu",
         (long long)searchUs, (long long)hydrateUs, (long long)duration.count(),
         idResults.size(),
         query.filterChunkIds ? std::to_string(query.filterChunkIds->size()).c_str() : "none",
         tid);

    LOGD("[BENCHMARK] BM25Retriever%s: topK=%u, results=%zu, time=%lld us",
         rm3EnabledVal ? "(RM3)" : "", topK, results.size(),
         (long long)duration.count());

    return results;
}

std::vector<ChunkSearchResult> BM25Retriever::retrieve(
    const std::string& text, const Params* overrideParams) {

    if (!kiwiTokenizer_) {
        LOGE("KiwiTokenizer is null, cannot tokenize text");
        return {};
    }

    auto morphemes = kiwiTokenizer_->tokenize(text);
    if (morphemes.empty()) {
        LOGE("KiwiTokenizer returned empty tokens for text");
        return {};
    }

    auto kiwiTokenHashes = hashMorphemes(morphemes);

    const Params& p = overrideParams ? *overrideParams : params;

    QueryBundle query;
    query.rawText = text;
    query.kiwiTokens = std::move(kiwiTokenHashes);

    return retrieve(query, p);
}

std::vector<ChunkSearchResult> BM25Retriever::retrieve(
    const std::vector<int32_t>& kiwiTokenHashes) {

    if (!isReady()) {
        LOGE("BM25 index not ready");
        return {};
    }

    auto bm25Params = toBM25Params();
    auto idResults = bm25Index_->searchIds(kiwiTokenHashes, params.topK, nullptr, &bm25Params);
    return hydrateSearchResults(idResults, obxManager_);
}

std::vector<ChunkSearchResult> BM25Retriever::retrieveRaw(
    const std::vector<int32_t>& kiwiTokenHashes) {

    if (!isReady()) {
        LOGE("BM25 index not ready");
        return {};
    }

    auto bm25Params = toBM25Params();
    auto idResults = bm25Index_->searchIds(kiwiTokenHashes, params.topK, nullptr, &bm25Params);
    return hydrateSearchResults(idResults, obxManager_);
}

} // namespace ecovector
