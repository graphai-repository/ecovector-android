#ifndef ECOVECTOR_RETRIEVER_IRETRIEVER_H
#define ECOVECTOR_RETRIEVER_IRETRIEVER_H

#include "QueryBundle.h"
#include <vector>
#include <string>
#include <cstdint>

// Forward declaration
namespace ecovector { struct ChunkSearchResult; }

namespace ecovector {

class IRetriever {
public:
    /** Base retriever params — concrete retrievers extend this. */
    struct Params {
        uint32_t topK = 5;
        virtual ~Params() = default;
    };

    virtual ~IRetriever() = default;

    /**
     * Primary search method — with explicit params override.
     * Concrete retrievers downcast to their own Params subtype
     * for retriever-specific fields; topK is always available from base.
     */
    virtual std::vector<ChunkSearchResult> retrieve(
        const QueryBundle& query, const Params& params) = 0;

    /** Convenience: uses retriever's default params (set at construction). */
    std::vector<ChunkSearchResult> retrieve(const QueryBundle& query);


    virtual const char* getName() const = 0;
    virtual bool isReady() const = 0;

    /** Returns the retriever's default params. */
    virtual const Params& getDefaultParams() const = 0;

    /** Configured topK for this retriever. */
    virtual uint32_t getTopK() const { return getDefaultParams().topK; }

    // true if distance field = L2 distance (lower = more similar)
    // false if distance field = similarity score (higher = more similar)
    virtual bool returnsDistance() const { return false; }

    /** Non-default params summary for display (e.g., "nprobe=8, efSearch=200"). Empty if all defaults. */
    virtual std::string getParamsSummary() const { return ""; }

    /** Preload indices into memory for latency-sensitive scenarios (e.g., benchmarking). */
    virtual void warmup() {}

    /** Whether this retriever needs embedding vectors in QueryBundle. */
    virtual bool needsEmbedding() const { return false; }

    /** Whether this retriever needs kiwi morpheme tokens in QueryBundle. */
    virtual bool needsKiwiTokens() const { return false; }
};

} // namespace ecovector

#endif // ECOVECTOR_RETRIEVER_IRETRIEVER_H
