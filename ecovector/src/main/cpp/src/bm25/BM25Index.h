#ifndef ECOVECTOR_BM25_INDEX_H
#define ECOVECTOR_BM25_INDEX_H

#include <string>
#include <vector>
#include <unordered_set>
#include <cstdint>
#include <memory>
#include <utility>

#include "../retriever/IndexSearchResult.h"

namespace ecovector {

class ObxManager;
class KiwiTokenizer;
struct ChunkSearchResult;

class BM25Index {
public:
    struct Parameters {
        float k1 = 1.5f;
        float b = 0.75f;
        float idfThreshold = 0.6f;        // 시드 선택 IDF 비율 (topIdf × ratio)
        size_t maxSeedTerms = 5;           // 시드 토큰 최대 수
        size_t candidateMultiplier = 10;   // 풀스코어링 후보 = topK × multiplier
        size_t minCandidates = 50;         // 풀스코어링 후보 최소 수
        float minScore = 0.0f;             // BM25 점수 하한
    };

    // 검색 단계별 타이밍 (선택적 계측용, nullptr 전달 시 무시)
    struct SearchStats {
        uint32_t candidateCount = 0;      // 후보 청크 수
        uint32_t seedTokenCount = 0;      // 실제 사용된 시드 토큰 수
        double candidateCollectMs = 0.0;  // 역인덱스 순회 + 후보 수집 시간
        double scoringMs = 0.0;           // BM25 점수 계산 시간
        double sortMs = 0.0;              // 정렬 시간
        double fetchMs = 0.0;             // DB 배치 조회 시간
    };

    explicit BM25Index(const std::string& basePath);
    ~BM25Index();

    // Kiwi 토크나이저 설정 (외부 소유, non-owning)
    void setTokenizer(KiwiTokenizer* tokenizer);

    bool buildIndex(ObxManager* obxManager);
    bool saveIndex() const;
    bool loadIndex();
    bool isIndexReady() const;
    void removeIndex();

    // === ID-ONLY SEARCH (no ObxManager dependency) ===
    std::vector<IndexSearchResult> searchIds(
        const std::vector<int32_t>& queryKiwiTokens,
        uint32_t topK = 10,
        const std::unordered_set<uint64_t>* allowedChunkIds = nullptr,
        const Parameters* params = nullptr,
        SearchStats* stats = nullptr);

    // 토큰 기반 BM25 검색 (사전 토큰화된 kiwi token hash)
    std::vector<ChunkSearchResult> searchRaw(
        ObxManager* obxManager,
        const std::vector<int32_t>& queryKiwiTokens,
        uint32_t topK = 10,
        const std::unordered_set<uint64_t>* allowedChunkIds = nullptr,
        SearchStats* stats = nullptr);

    std::vector<ChunkSearchResult> search(
        ObxManager* obxManager,
        const std::vector<int32_t>& queryKiwiTokens,
        uint32_t topK = 10,
        const std::unordered_set<uint64_t>* allowedChunkIds = nullptr);

    // === RM3 query expansion support ===
    float getTokenIDF(int32_t token) const;
    uint32_t getTokenDF(int32_t token) const;
    uint32_t getTotalChunks() const;
    uint32_t getChunkLength(uint64_t chunkId) const;

    /** Weighted BM25 search: score(Q,D) = Σ weight(w) × BM25(w,D) */
    std::vector<IndexSearchResult> searchIdsWeighted(
        const std::vector<std::pair<int32_t, float>>& weightedTerms,
        uint32_t topK = 10,
        const std::unordered_set<uint64_t>* allowedChunkIds = nullptr,
        const Parameters* params = nullptr,
        SearchStats* stats = nullptr);

    const Parameters& getDefaultParameters() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};

} // namespace ecovector

#endif // ECOVECTOR_BM25_INDEX_H
