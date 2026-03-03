#ifndef ECOVECTOR_RETRIEVER_OBX_VECTOR_RETRIEVER_H
#define ECOVECTOR_RETRIEVER_OBX_VECTOR_RETRIEVER_H

#include "IRetriever.h"
#include <string>

namespace ecovector {

class ObxManager;
class Embedder;

class ObxVectorRetriever : public IRetriever {
public:
    struct Params : IRetriever::Params {
        uint32_t maxResultCount = 100;  // HNSW ef (oversampling for quality)
        Params() { topK = 11; }
    };

    ObxVectorRetriever(ObxManager* obxManager, Embedder* embedder);
    ~ObxVectorRetriever() override;

    using IRetriever::retrieve;

    // IRetriever interface
    std::vector<ChunkSearchResult> retrieve(
        const QueryBundle& query, const IRetriever::Params& overrideParams) override;
    const char* getName() const override { return "ObxVector"; }
    bool isReady() const override;
    const IRetriever::Params& getDefaultParams() const override { return params; }
    bool returnsDistance() const override { return true; }
    bool needsEmbedding() const override { return true; }
    std::string getParamsSummary() const override;

    Params params;

private:
    ObxManager* obxManager_;    // non-owning
    Embedder* embedder_;        // non-owning
};

} // namespace ecovector

#endif // ECOVECTOR_RETRIEVER_OBX_VECTOR_RETRIEVER_H
