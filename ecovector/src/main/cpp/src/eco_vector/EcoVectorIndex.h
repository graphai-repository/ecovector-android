#ifndef ECOVECTOR_ECO_VECTOR_INDEX_H
#define ECOVECTOR_ECO_VECTOR_INDEX_H

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <cstdint>
#include <memory>

#include "../retriever/IndexSearchResult.h"

namespace ecovector {

// Forward declarations
class ObxManager;
struct ChunkData;
struct ChunkSearchResult;

// Reranker type enumeration
enum class RerankerType {
    NONE,                 // No reranking
    LCS                   // LCS with Kiwi morpheme tokens
};

// Configuration constants for EcoVector
struct EcoVectorConfig {
    size_t nCluster = 0;            // Number of clusters (0 = auto: max(8, totalChunks/100))
    size_t hnswM = 16;              // HNSW graph M parameter (connections per node)
    size_t hnswEfConstruction = 100; // HNSW construction parameter (build quality)
    size_t maxTrainSamples = 100000; // Max FAISS training samples (auto: min(nCluster*100, this))
    // Note: efSearch is a per-call parameter (passed to search()), not stored here
};

/**
 * EcoVector - Efficient Clustered Vector Search
 *
 * Implements a two-level hierarchical vector search:
 * 1. First level: Search among cluster centroids to find relevant clusters
 * 2. Second level: Search within selected clusters using per-cluster HNSW indices
 *
 * This approach reduces search time by only scanning a subset of vectors
 * based on their cluster membership.
 */
class EcoVectorIndex {
public:
    /**
     * Constructor
     * @param basePath Base directory for storing EcoVector index files
     * @param config EcoVector configuration parameters
     */
    explicit EcoVectorIndex(const std::string& basePath,
                            const EcoVectorConfig& config = EcoVectorConfig{});

    ~EcoVectorIndex();

    // Prevent copy/move
    EcoVectorIndex(const EcoVectorIndex&) = delete;
    EcoVectorIndex& operator=(const EcoVectorIndex&) = delete;
    EcoVectorIndex(EcoVectorIndex&&) = delete;
    EcoVectorIndex& operator=(EcoVectorIndex&&) = delete;

    /**
     * Create full index from all chunks in the database
     *
     * This method:
     * 1. Loads all chunk vectors from ObjectBox
     * 2. Performs K-means clustering to find centroids
     * 3. Builds HNSW index for centroids
     * 4. Builds per-cluster HNSW indices
     * 5. Saves all indices and mappings to files
     *
     * @param obxManager Pointer to ObjectBox manager (for reading chunks)
     * @param centroidCount Number of clusters to create (default: config.nCluster)
     * @return true if successful, false otherwise
     */
    bool createIndexes(ObxManager* obxManager, size_t centroidCount = 0);

    /**
     * Add a new chunk to the index incrementally
     *
     * This method:
     * 1. Finds the nearest centroid for the new chunk
     * 2. Adds the chunk to the corresponding cluster's HNSW index
     * 3. Updates the cluster mapping
     *
     * Note: For best accuracy, periodically rebuild the full index
     *
     * @param chunk The chunk data to add (must have valid id and vector)
     * @return true if successful, false otherwise
     */
    bool addIndex(const ChunkData& chunk);

    /**
     * Search for similar chunks using EcoVector algorithm
     *
     * This method:
     * 1. Searches centroid index to find top-k relevant clusters
     * 2. Searches within each selected cluster's HNSW index
     * 3. Merges and sorts results by distance
     * 4. Applies reranking (if specified)
     *
     * @param obxManager Pointer to ObjectBox manager (for reading chunk content)
     * @param queryVector The query vector to search for
     * @param limit Maximum number of chunks to return
     * @param rerankerType Type of reranking to apply (default: NONE)
     *        - NONE: No reranking, return results sorted by distance
     *        - LCS: Rerank using LCS on Kiwi morpheme tokens (O(k*m*n))
     * @param queryKiwiTokens Kiwi morpheme tokens for the query (required if rerankerType != NONE)
     * @return Vector of search results:
     *         - If rerankerType == NONE: sorted by vector distance (ascending)
     *         - Otherwise: sorted by reranker score (descending), with distance as tiebreaker
     */
    std::vector<ChunkSearchResult> search(
        ObxManager* obxManager,
        const std::vector<float>& queryVector,
        uint32_t limit = 10,
        RerankerType rerankerType = RerankerType::NONE,
        const std::vector<int32_t>& queryKiwiTokens = {},
        const std::unordered_set<uint64_t>* allowedChunkIds = nullptr,
        size_t nprobe = 0,
        size_t efSearch = 20);

    /**
     * Search for similar chunks using string-based reranker type specification
     *
     * Convenience overload that accepts reranker type as a string.
     *
     * @param obxManager Pointer to ObjectBox manager (for reading chunk content)
     * @param queryVector The query vector to search for
     * @param limit Maximum number of chunks to return
     * @param rerankerTypeStr Reranker type as string: "none" or "lcs"
     *                        (case-insensitive, default: "none")
     * @param queryKiwiTokens Kiwi morpheme tokens for the query (required if rerankerType != NONE)
     * @param allowedChunkIds If non-null, only these chunk IDs will be considered
     * @param nprobe Number of clusters to probe (0 = auto: min(max(4, nCluster/8), 16))
     * @return Vector of search results
     *
     * @throws std::invalid_argument if rerankerTypeStr is not recognized
     */
    std::vector<ChunkSearchResult> search(
        ObxManager* obxManager,
        const std::vector<float>& queryVector,
        uint32_t limit,
        const std::string& rerankerTypeStr,
        const std::vector<int32_t>& queryKiwiTokens = {},
        const std::unordered_set<uint64_t>* allowedChunkIds = nullptr,
        size_t nprobe = 0,
        size_t efSearch = 20);

    // === ID-ONLY SEARCH (no ObxManager dependency) ===
    std::vector<IndexSearchResult> searchIds(
        const std::vector<float>& queryVector,
        uint32_t limit = 10,
        const std::unordered_set<uint64_t>* allowedChunkIds = nullptr,
        size_t nprobe = 0,
        size_t efSearch = 20);

    /**
     * Check if EcoVector index files exist and are valid
     * @return true if index is ready to use
     */
    /** Update config for next createIndexes() call. */
    void setConfig(const EcoVectorConfig& newConfig);

    bool isIndexReady() const;

    /**
     * Preload all indexes into memory for faster search
     * Call this before benchmarking to exclude I/O time from measurements
     * @return true if all indexes loaded successfully
     */
    bool preloadIndexes();

    /**
     * Remove all EcoVector index files
     */
    void removeIndexes();

    /**
     * Get the number of clusters
     */
    size_t getClusterCount() const;

    /**
     * Get the vector dimension
     */
    size_t getDimension() const;

    /**
     * Rerank using LCS on Kiwi morpheme tokens
     * Compares queryKiwiTokens with each result's chunk.kiwiTokens directly.
     * kiwiTokens are pre-filtered to content words at write time.
     * Time complexity: O(k * m * n) where k=result count, m=query tokens, n=chunk tokens
     *
     * @param queryKiwiTokens Kiwi morpheme hash tokens for the query (content words only)
     * @param results Vector of search results to rerank
     * @return Reranked results sorted by LCS score (higher = more similar)
     */
    static std::vector<ChunkSearchResult> rerank_lcs(
        const std::vector<int32_t>& queryKiwiTokens,
        std::vector<ChunkSearchResult>& results);

private:
    class Impl;
    std::unique_ptr<Impl> pImpl_;
};

} // namespace ecovector

#endif // ECOVECTOR_ECO_VECTOR_INDEX_H
