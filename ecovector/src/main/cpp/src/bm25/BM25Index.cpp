#include "BM25Index.h"
#include "../object_box/ObxManager.h"
#include "../kiwi/KiwiTokenizer.h"

#include <android/log.h>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <unordered_set>
#include <unordered_map>
#include <queue>

#define LOG_TAG "BM25Index"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)

namespace ecovector {

// ============================================================================
// PIMPL Implementation
// ============================================================================

struct BM25Index::Impl {
    std::string basePath;
    std::string indexFilePath;

    Parameters params;
    KiwiTokenizer* tokenizer = nullptr;  // 외부 소유

    // 역인덱스: kiwiToken(int32_t hash) -> {chunkId -> termFrequency}
    std::unordered_map<int32_t, std::unordered_map<uint64_t, uint32_t>> invertedIndex;

    std::unordered_map<uint64_t, uint32_t> chunkLengths;
    uint32_t totalChunks = 0;
    float avgChunkLength = 0.0f;

    // IDF 캐시: kiwiToken(int32_t hash) -> IDF value
    std::unordered_map<int32_t, float> idfCache;

    bool indexReady = false;

    explicit Impl(const std::string& path) : basePath(path) {
        indexFilePath = basePath + "/bm25_kiwi_index.bin";
    }

    float calculateIDF(int32_t token) {
        auto it = invertedIndex.find(token);
        if (it == invertedIndex.end()) {
            return 0.0f;
        }

        uint32_t docFreq = static_cast<uint32_t>(it->second.size());
        float n = static_cast<float>(docFreq);
        float N = static_cast<float>(totalChunks);

        return std::log((N - n + 0.5f) / (n + 0.5f) + 1.0f);
    }

    float calculateBM25ScoreDirect(
        const std::vector<int32_t>& queryTokens,
        uint64_t chunkId,
        const Parameters& p) {

        auto lengthIt = chunkLengths.find(chunkId);
        if (lengthIt == chunkLengths.end()) {
            return 0.0f;
        }

        float docLength = static_cast<float>(lengthIt->second);
        float score = 0.0f;

        for (int32_t token : queryTokens) {
            auto idfIt = idfCache.find(token);
            if (idfIt == idfCache.end()) continue;
            float idf = idfIt->second;

            auto postingsIt = invertedIndex.find(token);
            if (postingsIt == invertedIndex.end()) continue;

            auto tfIt = postingsIt->second.find(chunkId);
            if (tfIt == postingsIt->second.end()) continue;
            float tf = static_cast<float>(tfIt->second);

            float numerator = tf * (p.k1 + 1.0f);
            float denominator = tf + p.k1 * (1.0f - p.b + p.b * docLength / avgChunkLength);

            score += idf * (numerator / denominator);
        }

        return score;
    }
};

// ============================================================================
// Construction / Destruction
// ============================================================================

BM25Index::BM25Index(const std::string& basePath)
    : pImpl_(std::make_unique<Impl>(basePath)) {
    LOGI("BM25Index constructed with basePath: %s", basePath.c_str());
}

BM25Index::~BM25Index() {
    LOGD("BM25Index destructor called");
}

void BM25Index::setTokenizer(KiwiTokenizer* tokenizer) {
    pImpl_->tokenizer = tokenizer;
}

// ============================================================================
// Index Building
// ============================================================================

bool BM25Index::buildIndex(ObxManager* obxManager) {
    if (!obxManager) {
        LOGE("ObxManager is null");
        return false;
    }

    LOGI("Building BM25 index (pre-tokenized kiwi_tokens)...");

    uint32_t totalChunks = obxManager->getChunkCount();
    if (totalChunks == 0) {
        LOGE("No chunks found");
        return false;
    }

    pImpl_->invertedIndex.clear();
    pImpl_->chunkLengths.clear();
    pImpl_->idfCache.clear();
    pImpl_->totalChunks = totalChunks;

    uint64_t totalTokens = 0;
    constexpr size_t BATCH_SIZE = 10000;

    obxManager->forEachChunkBatch(BATCH_SIZE, true,
        [&](const std::vector<ChunkData>& chunks) {
            for (const auto& chunk : chunks) {
                if (chunk.kiwiTokens.empty()) continue;

                pImpl_->chunkLengths[chunk.id] = static_cast<uint32_t>(chunk.kiwiTokens.size());
                totalTokens += chunk.kiwiTokens.size();

                std::unordered_map<int32_t, uint32_t> termFreqs;
                for (int32_t token : chunk.kiwiTokens) {
                    termFreqs[token]++;
                }

                for (const auto& [token, freq] : termFreqs) {
                    pImpl_->invertedIndex[token][chunk.id] = freq;
                }
            }
        });

    pImpl_->avgChunkLength = static_cast<float>(totalTokens) / static_cast<float>(pImpl_->totalChunks);

    for (const auto& [token, postings] : pImpl_->invertedIndex) {
        pImpl_->idfCache[token] = pImpl_->calculateIDF(token);
    }

    pImpl_->indexReady = true;

    LOGI("BM25 index built: %u chunks, %zu unique tokens, avgLength=%.2f",
         pImpl_->totalChunks, pImpl_->invertedIndex.size(), pImpl_->avgChunkLength);

    return saveIndex();
}

// ============================================================================
// Index Persistence
// ============================================================================

bool BM25Index::saveIndex() const {
    if (!pImpl_->indexReady) {
        LOGE("Index not ready, cannot save");
        return false;
    }

    try {
        std::ofstream file(pImpl_->indexFilePath, std::ios::binary);
        if (!file.is_open()) {
            LOGE("Cannot open file for writing: %s", pImpl_->indexFilePath.c_str());
            return false;
        }

        const uint32_t MAGIC = 0x424D3235;  // "BM25"
        const uint32_t VERSION = 3;
        file.write(reinterpret_cast<const char*>(&MAGIC), sizeof(MAGIC));
        file.write(reinterpret_cast<const char*>(&VERSION), sizeof(VERSION));

        file.write(reinterpret_cast<const char*>(&pImpl_->params.k1), sizeof(float));
        file.write(reinterpret_cast<const char*>(&pImpl_->params.b), sizeof(float));

        file.write(reinterpret_cast<const char*>(&pImpl_->totalChunks), sizeof(uint32_t));
        file.write(reinterpret_cast<const char*>(&pImpl_->avgChunkLength), sizeof(float));

        // 문서 길이 맵
        uint32_t chunkLengthsSize = static_cast<uint32_t>(pImpl_->chunkLengths.size());
        file.write(reinterpret_cast<const char*>(&chunkLengthsSize), sizeof(uint32_t));
        for (const auto& [chunkId, length] : pImpl_->chunkLengths) {
            file.write(reinterpret_cast<const char*>(&chunkId), sizeof(uint64_t));
            file.write(reinterpret_cast<const char*>(&length), sizeof(uint32_t));
        }

        // 역인덱스 (int32_t 키)
        uint32_t invertedIndexSize = static_cast<uint32_t>(pImpl_->invertedIndex.size());
        file.write(reinterpret_cast<const char*>(&invertedIndexSize), sizeof(uint32_t));
        for (const auto& [token, postingsMap] : pImpl_->invertedIndex) {
            file.write(reinterpret_cast<const char*>(&token), sizeof(int32_t));

            uint32_t postingsSize = static_cast<uint32_t>(postingsMap.size());
            file.write(reinterpret_cast<const char*>(&postingsSize), sizeof(uint32_t));
            for (const auto& [chunkId, freq] : postingsMap) {
                file.write(reinterpret_cast<const char*>(&chunkId), sizeof(uint64_t));
                file.write(reinterpret_cast<const char*>(&freq), sizeof(uint32_t));
            }
        }

        // IDF 캐시 (int32_t 키)
        uint32_t idfCacheSize = static_cast<uint32_t>(pImpl_->idfCache.size());
        file.write(reinterpret_cast<const char*>(&idfCacheSize), sizeof(uint32_t));
        for (const auto& [token, idf] : pImpl_->idfCache) {
            file.write(reinterpret_cast<const char*>(&token), sizeof(int32_t));
            file.write(reinterpret_cast<const char*>(&idf), sizeof(float));
        }

        file.close();
        LOGI("BM25 index saved to: %s", pImpl_->indexFilePath.c_str());
        return true;

    } catch (const std::exception& e) {
        LOGE("Error saving BM25 index: %s", e.what());
        return false;
    }
}

bool BM25Index::loadIndex() {
    try {
        std::ifstream file(pImpl_->indexFilePath, std::ios::binary);
        if (!file.is_open()) {
            LOGD("BM25 index file not found: %s", pImpl_->indexFilePath.c_str());
            return false;
        }

        uint32_t magic, version;
        file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
        file.read(reinterpret_cast<char*>(&version), sizeof(version));

        if (magic != 0x424D3235 || version != 3) {
            LOGW("Incompatible BM25 index version %u, rebuild required", version);
            return false;
        }

        file.read(reinterpret_cast<char*>(&pImpl_->params.k1), sizeof(float));
        file.read(reinterpret_cast<char*>(&pImpl_->params.b), sizeof(float));

        file.read(reinterpret_cast<char*>(&pImpl_->totalChunks), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&pImpl_->avgChunkLength), sizeof(float));

        // 문서 길이 맵
        pImpl_->chunkLengths.clear();
        uint32_t chunkLengthsSize;
        file.read(reinterpret_cast<char*>(&chunkLengthsSize), sizeof(uint32_t));
        for (uint32_t i = 0; i < chunkLengthsSize; ++i) {
            uint64_t chunkId;
            uint32_t length;
            file.read(reinterpret_cast<char*>(&chunkId), sizeof(uint64_t));
            file.read(reinterpret_cast<char*>(&length), sizeof(uint32_t));
            pImpl_->chunkLengths[chunkId] = length;
        }

        // 역인덱스 (int32_t 키)
        pImpl_->invertedIndex.clear();
        uint32_t invertedIndexSize;
        file.read(reinterpret_cast<char*>(&invertedIndexSize), sizeof(uint32_t));
        for (uint32_t i = 0; i < invertedIndexSize; ++i) {
            int32_t token;
            file.read(reinterpret_cast<char*>(&token), sizeof(int32_t));

            uint32_t postingsSize;
            file.read(reinterpret_cast<char*>(&postingsSize), sizeof(uint32_t));

            std::unordered_map<uint64_t, uint32_t> postingsMap;
            postingsMap.reserve(postingsSize);
            for (uint32_t j = 0; j < postingsSize; ++j) {
                uint64_t chunkId;
                uint32_t freq;
                file.read(reinterpret_cast<char*>(&chunkId), sizeof(uint64_t));
                file.read(reinterpret_cast<char*>(&freq), sizeof(uint32_t));
                postingsMap[chunkId] = freq;
            }
            pImpl_->invertedIndex[token] = std::move(postingsMap);
        }

        // IDF 캐시 (int32_t 키)
        pImpl_->idfCache.clear();
        uint32_t idfCacheSize;
        file.read(reinterpret_cast<char*>(&idfCacheSize), sizeof(uint32_t));
        for (uint32_t i = 0; i < idfCacheSize; ++i) {
            int32_t token;
            file.read(reinterpret_cast<char*>(&token), sizeof(int32_t));

            float idf;
            file.read(reinterpret_cast<char*>(&idf), sizeof(float));
            pImpl_->idfCache[token] = idf;
        }

        file.close();
        pImpl_->indexReady = true;

        LOGI("BM25 index loaded: %u chunks, %zu unique tokens",
             pImpl_->totalChunks, pImpl_->invertedIndex.size());
        return true;

    } catch (const std::exception& e) {
        LOGE("Error loading BM25 index: %s", e.what());
        return false;
    }
}

bool BM25Index::isIndexReady() const {
    return pImpl_->indexReady;
}

void BM25Index::removeIndex() {
    pImpl_->invertedIndex.clear();
    pImpl_->chunkLengths.clear();
    pImpl_->idfCache.clear();
    pImpl_->totalChunks = 0;
    pImpl_->avgChunkLength = 0.0f;
    pImpl_->indexReady = false;

    std::remove(pImpl_->indexFilePath.c_str());

    // 구 버전 인덱스 파일도 삭제
    std::string oldIndexPath = pImpl_->basePath + "/bm25_index.bin";
    std::remove(oldIndexPath.c_str());

    LOGI("BM25 index removed");
}

// ============================================================================
// ID-only search (no ObxManager dependency)
// ============================================================================

std::vector<IndexSearchResult> BM25Index::searchIds(
    const std::vector<int32_t>& queryKiwiTokens,
    uint32_t topK,
    const std::unordered_set<uint64_t>* allowedChunkIds,
    const Parameters* params,
    SearchStats* stats) {

    std::vector<IndexSearchResult> idResults;

    if (!pImpl_->indexReady) {
        LOGE("BM25 index not ready");
        return idResults;
    }

    if (queryKiwiTokens.empty()) {
        LOGD("Query kiwi tokens empty");
        return idResults;
    }

    const Parameters& p = params ? *params : pImpl_->params;

    // [단계 1] IDF 기반 후보 수집
    auto t0 = std::chrono::high_resolution_clock::now();

    std::vector<int32_t> sortedTokens = queryKiwiTokens;
    std::sort(sortedTokens.begin(), sortedTokens.end(),
        [&](int32_t a, int32_t b) {
            auto itA = pImpl_->idfCache.find(a);
            auto itB = pImpl_->idfCache.find(b);
            float idfA = (itA != pImpl_->idfCache.end()) ? itA->second : 0.0f;
            float idfB = (itB != pImpl_->idfCache.end()) ? itB->second : 0.0f;
            return idfA > idfB;
        });

    size_t MAX_SEED_TOKENS = 1;
    if (!sortedTokens.empty()) {
        auto topIt = pImpl_->idfCache.find(sortedTokens[0]);
        float topIdf = (topIt != pImpl_->idfCache.end()) ? topIt->second : 0.0f;
        float threshold = topIdf * p.idfThreshold;
        MAX_SEED_TOKENS = 0;
        for (size_t i = 0; i < std::min(sortedTokens.size(), p.maxSeedTerms); ++i) {
            auto it = pImpl_->idfCache.find(sortedTokens[i]);
            float idf = (it != pImpl_->idfCache.end()) ? it->second : 0.0f;
            if (idf >= threshold) {
                MAX_SEED_TOKENS = i + 1;
            } else {
                break;
            }
        }
        if (MAX_SEED_TOKENS == 0) MAX_SEED_TOKENS = 1;
    }

    std::unordered_map<uint64_t, float> seedScoreMap;

    for (size_t si = 0; si < MAX_SEED_TOKENS; ++si) {
        int32_t token = sortedTokens[si];
        auto idfIt = pImpl_->idfCache.find(token);
        if (idfIt == pImpl_->idfCache.end()) continue;
        float idf = idfIt->second;

        auto it = pImpl_->invertedIndex.find(token);
        if (it == pImpl_->invertedIndex.end()) continue;

        for (const auto& [chunkId, freq] : it->second) {
            if (allowedChunkIds && !allowedChunkIds->count(chunkId)) continue;
            float tf = static_cast<float>(freq);
            seedScoreMap[chunkId] += idf * tf;
        }
    }

    const size_t FULL_SCORE_LIMIT = std::max(
        static_cast<size_t>(topK * p.candidateMultiplier), p.minCandidates);
    std::vector<std::pair<uint64_t, float>> seedRanked(seedScoreMap.begin(), seedScoreMap.end());
    if (seedRanked.size() > FULL_SCORE_LIMIT) {
        std::partial_sort(seedRanked.begin(),
                          seedRanked.begin() + FULL_SCORE_LIMIT,
                          seedRanked.end(),
                          [](const auto& a, const auto& b) { return a.second > b.second; });
        seedRanked.resize(FULL_SCORE_LIMIT);
    }

    std::vector<uint64_t> candidateVec;
    candidateVec.reserve(seedRanked.size());
    for (const auto& [chunkId, seedScore] : seedRanked) {
        candidateVec.push_back(chunkId);
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    if (candidateVec.empty()) {
        LOGD("No matching chunks found for query");
        return idResults;
    }

    // [단계 2] BM25 점수 계산
    std::vector<std::pair<uint64_t, float>> scoredChunks;
    scoredChunks.reserve(candidateVec.size());

    for (uint64_t chunkId : candidateVec) {
        float score = pImpl_->calculateBM25ScoreDirect(queryKiwiTokens, chunkId, p);
        if (score > p.minScore) {
            scoredChunks.emplace_back(chunkId, score);
        }
    }

    auto t2 = std::chrono::high_resolution_clock::now();

    // [단계 3] 정렬
    size_t resultCount = std::min(static_cast<size_t>(topK), scoredChunks.size());
    if (resultCount > 0) {
        std::partial_sort(scoredChunks.begin(),
                          scoredChunks.begin() + resultCount,
                          scoredChunks.end(),
                          [](const auto& a, const auto& b) { return a.second > b.second; });
    }

    auto t3 = std::chrono::high_resolution_clock::now();

    // stats 채우기 (fetchMs is 0 since we don't fetch from DB)
    if (stats) {
        stats->candidateCount     = static_cast<uint32_t>(seedScoreMap.size());
        stats->seedTokenCount     = static_cast<uint32_t>(MAX_SEED_TOKENS);
        stats->candidateCollectMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
        stats->scoringMs          = std::chrono::duration<double, std::milli>(t2 - t1).count();
        stats->sortMs             = std::chrono::duration<double, std::milli>(t3 - t2).count();
        stats->fetchMs            = 0.0;
    }

    // Build IndexSearchResult
    idResults.reserve(resultCount);
    for (size_t i = 0; i < resultCount; ++i) {
        idResults.push_back({scoredChunks[i].first, scoredChunks[i].second});
    }

    LOGD("BM25 searchIds returned %zu results", idResults.size());
    return idResults;
}

// ============================================================================
// Search (Raw - searchIds + hydration, without dedup, for RRF fusion)
// ============================================================================

std::vector<ChunkSearchResult> BM25Index::searchRaw(
    ObxManager* obxManager,
    const std::vector<int32_t>& queryKiwiTokens,
    uint32_t topK,
    const std::unordered_set<uint64_t>* allowedChunkIds,
    SearchStats* stats) {

    std::vector<ChunkSearchResult> results;

    if (!obxManager) {
        LOGE("ObxManager is null");
        return results;
    }

    // 1. Index-level search: ID+score only (use default params)
    auto idResults = searchIds(queryKiwiTokens, topK, allowedChunkIds, nullptr, stats);
    if (idResults.empty()) return results;

    // 2. Hydrate: batch fetch from ObxManager
    auto t3 = std::chrono::high_resolution_clock::now();

    std::vector<uint64_t> chunkIds;
    chunkIds.reserve(idResults.size());
    for (const auto& r : idResults) {
        chunkIds.push_back(r.chunkId);
    }

    auto chunks = obxManager->getChunksByIds(chunkIds, /*excludeVectors=*/true);

    auto t4 = std::chrono::high_resolution_clock::now();

    // Update fetchMs in stats
    if (stats) {
        stats->fetchMs = std::chrono::duration<double, std::milli>(t4 - t3).count();
    }

    // 3. Build ChunkSearchResult (preserve score order)
    std::unordered_map<uint64_t, ChunkData*> chunkMap;
    chunkMap.reserve(chunks.size());
    for (auto& chunk : chunks) {
        chunkMap[chunk.id] = &chunk;
    }

    results.reserve(idResults.size());
    for (const auto& ir : idResults) {
        auto it = chunkMap.find(ir.chunkId);
        if (it == chunkMap.end()) continue;

        ChunkSearchResult result;
        result.chunk = std::move(*it->second);
        result.distance = ir.score;
        results.push_back(std::move(result));
    }

    LOGD("BM25 searchRaw returned %zu results", results.size());
    return results;
}

// ============================================================================
// Search (with document dedup) - searchIds + hydration + dedup
// ============================================================================

std::vector<ChunkSearchResult> BM25Index::search(
    ObxManager* obxManager,
    const std::vector<int32_t>& queryKiwiTokens,
    uint32_t topK,
    const std::unordered_set<uint64_t>* allowedChunkIds) {

    std::vector<ChunkSearchResult> results;

    if (!obxManager) {
        LOGE("ObxManager is null");
        return results;
    }

    // Fetch more candidates for dedup (multiple chunks per document)
    uint32_t fetchTopK = topK * 3;
    auto idResults = searchIds(queryKiwiTokens, fetchTopK, allowedChunkIds);
    if (idResults.empty()) return results;

    // Hydrate
    std::vector<uint64_t> chunkIds;
    chunkIds.reserve(idResults.size());
    for (const auto& r : idResults) {
        chunkIds.push_back(r.chunkId);
    }

    auto chunks = obxManager->getChunksByIds(chunkIds, /*excludeVectors=*/true);

    std::unordered_map<uint64_t, ChunkData*> chunkMap;
    chunkMap.reserve(chunks.size());
    for (auto& chunk : chunks) {
        chunkMap[chunk.id] = &chunk;
    }

    // Document dedup: score 순서대로 처리하면서 중복 문서 제거
    std::unordered_set<uint64_t> seenDocumentIds;
    results.reserve(topK);

    for (const auto& ir : idResults) {
        auto it = chunkMap.find(ir.chunkId);
        if (it == chunkMap.end()) continue;

        uint64_t docId = it->second->documentId;
        if (seenDocumentIds.count(docId) > 0) continue;

        seenDocumentIds.insert(docId);

        ChunkSearchResult result;
        result.chunk = std::move(*it->second);
        result.distance = ir.score;
        results.push_back(std::move(result));

        if (results.size() >= topK) break;
    }

    LOGD("BM25 search returned %zu results", results.size());
    return results;
}

// ============================================================================
// RM3 support: statistics access
// ============================================================================

float BM25Index::getTokenIDF(int32_t token) const {
    auto it = pImpl_->idfCache.find(token);
    return (it != pImpl_->idfCache.end()) ? it->second : 0.0f;
}

uint32_t BM25Index::getTokenDF(int32_t token) const {
    auto it = pImpl_->invertedIndex.find(token);
    return (it != pImpl_->invertedIndex.end())
        ? static_cast<uint32_t>(it->second.size()) : 0;
}

uint32_t BM25Index::getTotalChunks() const {
    return pImpl_->totalChunks;
}

uint32_t BM25Index::getChunkLength(uint64_t chunkId) const {
    auto it = pImpl_->chunkLengths.find(chunkId);
    return (it != pImpl_->chunkLengths.end()) ? it->second : 0;
}

// ============================================================================
// Weighted BM25 search (for RM3 expanded query)
// ============================================================================

std::vector<IndexSearchResult> BM25Index::searchIdsWeighted(
    const std::vector<std::pair<int32_t, float>>& weightedTerms,
    uint32_t topK,
    const std::unordered_set<uint64_t>* allowedChunkIds,
    const Parameters* params,
    SearchStats* stats) {

    std::vector<IndexSearchResult> idResults;
    if (!pImpl_->indexReady || weightedTerms.empty()) return idResults;

    const Parameters& p = params ? *params : pImpl_->params;

    auto t0 = std::chrono::high_resolution_clock::now();

    // [Step 1] Seed selection: sort by weight × IDF descending
    struct WeightedToken {
        int32_t token;
        float weight;
        float idf;
        float score;  // weight × IDF
    };
    std::vector<WeightedToken> tokens;
    tokens.reserve(weightedTerms.size());
    for (const auto& [tok, w] : weightedTerms) {
        auto idfIt = pImpl_->idfCache.find(tok);
        float idf = (idfIt != pImpl_->idfCache.end()) ? idfIt->second : 0.0f;
        if (idf > 0.0f && w > 0.0f) {
            tokens.push_back({tok, w, idf, w * idf});
        }
    }
    std::sort(tokens.begin(), tokens.end(),
        [](const auto& a, const auto& b) { return a.score > b.score; });

    // Seed selection: IDF threshold relative to top score
    size_t maxSeeds = std::min(tokens.size(), p.maxSeedTerms);
    float topScore = tokens.empty() ? 0.0f : tokens[0].score;
    float threshold = topScore * p.idfThreshold;
    size_t seedCount = 0;
    for (size_t i = 0; i < maxSeeds; i++) {
        if (tokens[i].score >= threshold) seedCount = i + 1;
        else break;
    }
    if (seedCount == 0 && !tokens.empty()) seedCount = 1;

    // Candidate collection (same strategy as searchIds)
    std::unordered_map<uint64_t, float> seedScoreMap;
    for (size_t si = 0; si < seedCount; si++) {
        const auto& wt = tokens[si];
        auto it = pImpl_->invertedIndex.find(wt.token);
        if (it == pImpl_->invertedIndex.end()) continue;
        for (const auto& [chunkId, freq] : it->second) {
            if (allowedChunkIds && !allowedChunkIds->count(chunkId)) continue;
            seedScoreMap[chunkId] += wt.score * static_cast<float>(freq);
        }
    }

    const size_t FULL_SCORE_LIMIT = std::max(
        static_cast<size_t>(topK * p.candidateMultiplier), p.minCandidates);
    std::vector<std::pair<uint64_t, float>> seedRanked(seedScoreMap.begin(), seedScoreMap.end());
    if (seedRanked.size() > FULL_SCORE_LIMIT) {
        std::partial_sort(seedRanked.begin(),
            seedRanked.begin() + FULL_SCORE_LIMIT, seedRanked.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });
        seedRanked.resize(FULL_SCORE_LIMIT);
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    if (seedRanked.empty()) return idResults;

    // [Step 2] Weighted BM25 scoring
    std::vector<std::pair<uint64_t, float>> scoredChunks;
    scoredChunks.reserve(seedRanked.size());

    for (const auto& [chunkId, _] : seedRanked) {
        auto lengthIt = pImpl_->chunkLengths.find(chunkId);
        if (lengthIt == pImpl_->chunkLengths.end()) continue;
        float docLength = static_cast<float>(lengthIt->second);

        float score = 0.0f;
        for (const auto& wt : tokens) {
            auto postingsIt = pImpl_->invertedIndex.find(wt.token);
            if (postingsIt == pImpl_->invertedIndex.end()) continue;
            auto tfIt = postingsIt->second.find(chunkId);
            if (tfIt == postingsIt->second.end()) continue;
            float tf = static_cast<float>(tfIt->second);
            float num = tf * (p.k1 + 1.0f);
            float den = tf + p.k1 * (1.0f - p.b + p.b * docLength / pImpl_->avgChunkLength);
            score += wt.weight * wt.idf * (num / den);
        }
        if (score > p.minScore) {
            scoredChunks.emplace_back(chunkId, score);
        }
    }

    auto t2 = std::chrono::high_resolution_clock::now();

    // [Step 3] Sort + top-K
    size_t resultCount = std::min(static_cast<size_t>(topK), scoredChunks.size());
    if (resultCount > 0) {
        std::partial_sort(scoredChunks.begin(),
            scoredChunks.begin() + resultCount, scoredChunks.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });
    }

    auto t3 = std::chrono::high_resolution_clock::now();

    if (stats) {
        stats->candidateCount = static_cast<uint32_t>(seedScoreMap.size());
        stats->seedTokenCount = static_cast<uint32_t>(seedCount);
        stats->candidateCollectMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
        stats->scoringMs = std::chrono::duration<double, std::milli>(t2 - t1).count();
        stats->sortMs = std::chrono::duration<double, std::milli>(t3 - t2).count();
        stats->fetchMs = 0.0;
    }

    idResults.reserve(resultCount);
    for (size_t i = 0; i < resultCount; i++) {
        idResults.push_back({scoredChunks[i].first, scoredChunks[i].second});
    }
    return idResults;
}

// ============================================================================
// Parameters
// ============================================================================

const BM25Index::Parameters& BM25Index::getDefaultParameters() const {
    return pImpl_->params;
}

} // namespace ecovector
