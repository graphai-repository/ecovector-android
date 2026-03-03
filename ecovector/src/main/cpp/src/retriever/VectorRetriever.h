#ifndef ECOVECTOR_RETRIEVER_VECTOR_RETRIEVER_H
#define ECOVECTOR_RETRIEVER_VECTOR_RETRIEVER_H

#include "IRetriever.h"
#include <vector>
#include <string>

namespace ecovector {

class EcoVectorIndex;   // FAISS clustering engine
class ObxManager;
class Embedder;

/**
 * VectorRetriever - 통합 IRetriever 인터페이스 구현 (벡터 검색)
 *
 * EcoVectorIndex 클러스터 HNSW 인덱스를 사용한 벡터 검색.
 * Embedder DI를 통해 텍스트 → 임베딩 변환을 내부 처리.
 */
class VectorRetriever : public IRetriever {
public:
    struct Params : IRetriever::Params {
        size_t efSearch = 20;
        size_t nprobe = 4;
        Params() { topK = 11; }
    };

    /**
     * @param ecoVectorIndex EcoVectorIndex 인덱스 (외부 소유)
     * @param obxManager ObjectBox 매니저 (외부 소유)
     * @param embedder 임베더 (외부 소유)
     */
    VectorRetriever(EcoVectorIndex* ecoVectorIndex, ObxManager* obxManager,
                    Embedder* embedder);
    ~VectorRetriever() override;

    using IRetriever::retrieve;

    // IRetriever interface
    std::vector<ChunkSearchResult> retrieve(
        const QueryBundle& query, const IRetriever::Params& overrideParams) override;
    const char* getName() const override { return "Vector"; }
    bool isReady() const override;
    const IRetriever::Params& getDefaultParams() const override { return params; }
    bool returnsDistance() const override { return true; }
    void warmup() override;
    bool needsEmbedding() const override { return true; }
    std::string getParamsSummary() const override {
        std::string s = "top=" + std::to_string(params.topK);
        if (params.nprobe != 0) s += ",nprobe=" + std::to_string(params.nprobe);
        if (params.efSearch != 20) s += ",ef=" + std::to_string(params.efSearch);
        return s;
    }

    // 편의 오버로드: 텍스트로 검색 (내부에서 Embedder로 임베딩 변환)
    std::vector<ChunkSearchResult> retrieve(
        const std::string& text, const Params* overrideParams = nullptr);

    // 편의 오버로드: raw embedding으로 검색
    std::vector<ChunkSearchResult> retrieve(
        const float* embedding, size_t dim,
        const Params* overrideParams = nullptr);

    Params params;

private:
    EcoVectorIndex* ecoVectorIndex_;   // non-owning
    ObxManager* obxManager_;      // non-owning
    Embedder* embedder_;          // non-owning
};

} // namespace ecovector

#endif // ECOVECTOR_RETRIEVER_VECTOR_RETRIEVER_H
