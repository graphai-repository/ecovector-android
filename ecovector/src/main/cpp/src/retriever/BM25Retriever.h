// app/src/main/cpp/src/retriever/BM25Retriever.h
#ifndef ECOVECTOR_RETRIEVER_BM25_RETRIEVER_H
#define ECOVECTOR_RETRIEVER_BM25_RETRIEVER_H

#include "IRetriever.h"
#include "../bm25/BM25Index.h"
#include <string>
#include <vector>

namespace ecovector {

// Forward declarations
class ObxManager;
class KiwiTokenizer;

/**
 * BM25Retriever - 통합 IRetriever 인터페이스 구현 (BM25 텍스트 검색)
 *
 * BM25Index를 사용한 키워드 기반 검색.
 * KiwiTokenizer DI를 통해 텍스트 → kiwi hash tokens 변환을 내부 처리.
 */
class BM25Retriever : public IRetriever {
public:
    struct Params : IRetriever::Params {
        float k1 = 0.9f;
        float b = 0.25f;
        float idfThreshold = 0.6f;
        size_t maxSeedTerms = 5;
        size_t candidateMultiplier = 10;
        size_t minCandidates = 50;
        float minScore = 0.0f;
        // RM3 (Relevance Model 3) query expansion
        bool rm3Enabled = false;
        uint32_t rm3FbDocs = 10;       // feedback documents (1차 검색 상위 문서 수)
        uint32_t rm3FbTerms = 20;      // expansion term 수
        float rm3OrigWeight = 0.6f;    // λ: original query weight (0.0~1.0)
        uint32_t rm3MinDf = 2;         // DF 필터 하한
        Params() { topK = 15; }
    };

    /**
     * @param bm25Index BM25 인덱스 (외부 소유)
     * @param obxManager ObjectBox 매니저 (외부 소유)
     * @param kiwiTokenizer Kiwi 토크나이저 (외부 소유, nullptr 가능 - 텍스트 검색 불가)
     */
    BM25Retriever(BM25Index* bm25Index, ObxManager* obxManager,
                  KiwiTokenizer* kiwiTokenizer = nullptr);
    ~BM25Retriever() override;

    using IRetriever::retrieve;

    // IRetriever interface
    std::vector<ChunkSearchResult> retrieve(
        const QueryBundle& query, const IRetriever::Params& overrideParams) override;
    const char* getName() const override { return "BM25"; }
    bool isReady() const override;
    const IRetriever::Params& getDefaultParams() const override { return params; }
    bool returnsDistance() const override { return false; }
    bool needsKiwiTokens() const override { return true; }
    std::string getParamsSummary() const override {
        char buf[128];
        std::snprintf(buf, sizeof(buf), "k1=%.1f,b=%.2f,top=%u",
                      params.k1, params.b, params.topK);
        std::string s(buf);
        if (params.idfThreshold != 0.6f) {
            char t[32]; std::snprintf(t, sizeof(t), ",idf=%.2f", params.idfThreshold);
            s += t;
        }
        if (params.maxSeedTerms != 5) s += ",seed=" + std::to_string(params.maxSeedTerms);
        if (params.candidateMultiplier != 10) s += ",cand=" + std::to_string(params.candidateMultiplier);
        if (params.minCandidates != 50) s += ",minC=" + std::to_string(params.minCandidates);
        if (params.minScore != 0.0f) {
            char t[32]; std::snprintf(t, sizeof(t), ",minS=%.1f", params.minScore);
            s += t;
        }
        if (params.rm3Enabled) {
            char t[64];
            std::snprintf(t, sizeof(t), ",rm3(%ud,%ut,\xce\xbb=%.1f)",
                params.rm3FbDocs, params.rm3FbTerms, params.rm3OrigWeight);
            s += t;
        }
        return s;
    }

    // 편의 오버로드: 텍스트로 검색 (내부에서 KiwiTokenizer로 토큰화 + 해싱)
    std::vector<ChunkSearchResult> retrieve(
        const std::string& text, const Params* overrideParams = nullptr);

    // 편의 오버로드: 사전 해싱된 kiwi token으로 직접 검색
    std::vector<ChunkSearchResult> retrieve(
        const std::vector<int32_t>& kiwiTokenHashes);

    // Raw 검색 (중복 제거 없이, RRF fusion용)
    std::vector<ChunkSearchResult> retrieveRaw(
        const std::vector<int32_t>& kiwiTokenHashes);

    Params params;

    // Retriever params → BM25Index::Parameters 변환 (thread-safe)
    BM25Index::Parameters toBM25Params() const;

private:

    BM25Index* bm25Index_;       // non-owning
    ObxManager* obxManager_;     // non-owning
    KiwiTokenizer* kiwiTokenizer_;  // non-owning, nullable
};

} // namespace ecovector

#endif // ECOVECTOR_RETRIEVER_BM25_RETRIEVER_H
