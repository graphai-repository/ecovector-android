#include "EcoVectorIndex.h"
#include "ObxManager.h"

#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/index_io.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/utils/distances.h>

#include <android/log.h>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <unordered_set>
#include <chrono>
#include <omp.h>

#define LOG_TAG "EcoVector"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)

namespace fs = std::filesystem;

namespace ecovector {

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * Convert string to RerankerType (case-insensitive)
 * @param str String representation: "none", "lcs", "token_intersection", or "hybrid"
 * @return Corresponding RerankerType
 * @throws std::invalid_argument if string doesn't match any type
 */
static RerankerType stringToRerankerType(const std::string& str) {
    std::string lower = str;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);

    if (lower == "none") {
        return RerankerType::NONE;
    } else if (lower == "lcs") {
        return RerankerType::LCS;
    } else {
        std::string msg = "Invalid reranker type: " + str +
                         " (expected: 'none' or 'lcs')";
        LOGE("%s", msg.c_str());
        throw std::invalid_argument(msg);
    }
}

/**
 * Convert RerankerType to string
 * @param type RerankerType enum value
 * @return String representation
 */
static const char* rerankerTypeToString(RerankerType type) {
    switch (type) {
        case RerankerType::NONE:
            return "None";
        case RerankerType::LCS:
            return "LCS";
        default:
            return "Unknown";
    }
}

// ============================================================================
// LCS (Longest Common Subsequence) Helper Function
// ============================================================================

/**
 * Calculate LCS length between two integer sequences (2-row DP)
 * @param a First integer sequence
 * @param b Second integer sequence
 * @return Length of longest common subsequence
 */
static uint32_t calculateLCSLength(const std::vector<int32_t>& a,
                                    const std::vector<int32_t>& b) {
    if (a.empty() || b.empty()) return 0;

    size_t n = a.size();
    size_t m = b.size();

    std::vector<uint32_t> prev(m + 1, 0), cur(m + 1, 0);

    for (size_t i = 1; i <= n; ++i) {
        for (size_t j = 1; j <= m; ++j) {
            if (a[i - 1] == b[j - 1]) {
                cur[j] = prev[j - 1] + 1;
            } else {
                cur[j] = std::max(prev[j], cur[j - 1]);
            }
        }
        std::swap(prev, cur);
    }

    return prev[m];
}

// ============================================================================
// File Paths
// ============================================================================

static std::string getCentroidIndexPath(const std::string& basePath) {
    return basePath + "/ecovector/centroid_hnsw.index";
}

static std::string getClusterIndexPath(const std::string& basePath, size_t clusterId) {
    return basePath + "/ecovector/cluster_" + std::to_string(clusterId) + ".index";
}

static std::string getClusterMappingPath(const std::string& basePath) {
    return basePath + "/ecovector/cluster_mapping.bin";
}

static std::string getMetadataPath(const std::string& basePath) {
    return basePath + "/ecovector/metadata.bin";
}

// ============================================================================
// PIMPL Implementation
// ============================================================================

class EcoVectorIndex::Impl {
public:
    std::string basePath;
    EcoVectorConfig config;

    // Cached indices (loaded on demand)
    std::unique_ptr<faiss::IndexHNSWFlat> centroidIndex;
    std::unordered_map<int, std::unique_ptr<faiss::IndexHNSWFlat>> clusterIndices;

    // Cluster mapping: cluster_id -> list of chunk IDs
    std::unordered_map<int, std::vector<uint64_t>> clusterMappings;

    // Metadata
    size_t dimension = 0;
    size_t numClusters = 0;

    explicit Impl(const std::string& path, const EcoVectorConfig& cfg)
        : basePath(path), config(cfg) {}

    // ========== File I/O ==========

    bool saveClusterMappings() {
        try {
            std::string path = getClusterMappingPath(basePath);
            std::ofstream file(path, std::ios::binary);
            if (!file) {
                LOGE("Failed to open cluster mapping file for writing: %s", path.c_str());
                return false;
            }

            int32_t numClusters = static_cast<int32_t>(clusterMappings.size());
            file.write(reinterpret_cast<char*>(&numClusters), sizeof(int32_t));

            for (const auto& [clusterId, chunkIds] : clusterMappings) {
                int32_t cid = clusterId;
                int32_t count = static_cast<int32_t>(chunkIds.size());
                file.write(reinterpret_cast<char*>(&cid), sizeof(int32_t));
                file.write(reinterpret_cast<char*>(&count), sizeof(int32_t));

                for (uint64_t chunkId : chunkIds) {
                    file.write(reinterpret_cast<const char*>(&chunkId), sizeof(uint64_t));
                }
            }

            LOGD("Saved cluster mappings: %d clusters", numClusters);
            return true;

        } catch (const std::exception& e) {
            LOGE("Error saving cluster mappings: %s", e.what());
            return false;
        }
    }

    bool loadClusterMappings() {
        try {
            std::string path = getClusterMappingPath(basePath);
            if (!fs::exists(path)) {
                LOGD("Cluster mapping file not found: %s", path.c_str());
                return false;
            }

            std::ifstream file(path, std::ios::binary);
            if (!file) {
                LOGE("Failed to open cluster mapping file: %s", path.c_str());
                return false;
            }

            clusterMappings.clear();

            int32_t numClusters;
            file.read(reinterpret_cast<char*>(&numClusters), sizeof(int32_t));

            for (int32_t i = 0; i < numClusters; i++) {
                int32_t clusterId, count;
                file.read(reinterpret_cast<char*>(&clusterId), sizeof(int32_t));
                file.read(reinterpret_cast<char*>(&count), sizeof(int32_t));

                std::vector<uint64_t> chunkIds(count);
                for (int32_t j = 0; j < count; j++) {
                    file.read(reinterpret_cast<char*>(&chunkIds[j]), sizeof(uint64_t));
                }

                clusterMappings[clusterId] = std::move(chunkIds);
            }

            LOGD("Loaded cluster mappings: %d clusters", numClusters);
            return true;

        } catch (const std::exception& e) {
            LOGE("Error loading cluster mappings: %s", e.what());
            return false;
        }
    }

    bool saveMetadata() {
        try {
            std::string path = getMetadataPath(basePath);
            std::ofstream file(path, std::ios::binary);
            if (!file) return false;

            file.write(reinterpret_cast<const char*>(&dimension), sizeof(size_t));
            file.write(reinterpret_cast<const char*>(&numClusters), sizeof(size_t));
            return true;

        } catch (const std::exception& e) {
            LOGE("Error saving metadata: %s", e.what());
            return false;
        }
    }

    bool loadMetadata() {
        try {
            std::string path = getMetadataPath(basePath);
            if (!fs::exists(path)) return false;

            std::ifstream file(path, std::ios::binary);
            if (!file) return false;

            file.read(reinterpret_cast<char*>(&dimension), sizeof(size_t));
            file.read(reinterpret_cast<char*>(&numClusters), sizeof(size_t));
            return true;

        } catch (const std::exception& e) {
            LOGE("Error loading metadata: %s", e.what());
            return false;
        }
    }

    // ========== Index Loading ==========

    bool ensureCentroidIndexLoaded() {
        if (centroidIndex) return true;

        std::string path = getCentroidIndexPath(basePath);
        if (!fs::exists(path)) {
            LOGE("Centroid index file not found: %s", path.c_str());
            return false;
        }

        try {
            faiss::Index* idx = faiss::read_index(path.c_str());
            centroidIndex.reset(dynamic_cast<faiss::IndexHNSWFlat*>(idx));
            if (!centroidIndex) {
                delete idx;
                LOGE("Failed to cast centroid index");
                return false;
            }
            LOGD("Loaded centroid index");
            return true;

        } catch (const std::exception& e) {
            LOGE("Error loading centroid index: %s", e.what());
            return false;
        }
    }

    faiss::IndexHNSWFlat* getClusterIndex(int clusterId) {
        // Check cache first
        auto it = clusterIndices.find(clusterId);
        if (it != clusterIndices.end()) {
            return it->second.get();
        }

        // Load from file
        std::string path = getClusterIndexPath(basePath, clusterId);
        if (!fs::exists(path)) {
            LOGD("Cluster index file not found: %s", path.c_str());
            return nullptr;
        }

        try {
            faiss::Index* idx = faiss::read_index(path.c_str());
            auto* hnswIdx = dynamic_cast<faiss::IndexHNSWFlat*>(idx);
            if (!hnswIdx) {
                delete idx;
                LOGE("Failed to cast cluster index %d", clusterId);
                return nullptr;
            }

            clusterIndices[clusterId].reset(hnswIdx);
            return hnswIdx;

        } catch (const std::exception& e) {
            LOGE("Error loading cluster index %d: %s", clusterId, e.what());
            return nullptr;
        }
    }
};

// ============================================================================
// EcoVector Public Methods
// ============================================================================

EcoVectorIndex::EcoVectorIndex(const std::string& basePath, const EcoVectorConfig& config)
    : pImpl_(std::make_unique<Impl>(basePath, config)) {
    LOGI("EcoVector created with base path: %s", basePath.c_str());

    // Try to load existing metadata
    pImpl_->loadMetadata();
}

EcoVectorIndex::~EcoVectorIndex() {
    LOGD("EcoVector destructor");
}

void EcoVectorIndex::setConfig(const EcoVectorConfig& newConfig) {
    pImpl_->config = newConfig;
    LOGD("EcoVectorConfig updated: nCluster=%zu, hnswM=%zu, efConstruction=%zu, maxTrain=%zu",
         newConfig.nCluster, newConfig.hnswM, newConfig.hnswEfConstruction, newConfig.maxTrainSamples);
}

bool EcoVectorIndex::createIndexes(ObxManager* obxManager, size_t centroidCount) {
    if (!obxManager) {
        LOGE("ObxManager is null");
        return false;
    }

    LOGI("Creating EcoVector indexes (requested centroidCount=%zu)...", centroidCount);

    try {
        // 1. Get total chunk count (lightweight query, no data loading)
        uint32_t totalChunks = obxManager->getChunkCount();
        if (totalChunks == 0) {
            LOGE("No chunks in database");
            return false;
        }

        LOGI("Total chunks in database: %u", totalChunks);

        // 2. Auto-calculate cluster count
        // For low-latency mobile: smaller nCluster (16-64) optimized
        // Target: 1000-5000 vectors per cluster, <3ms latency
        if (centroidCount == 0) {
            centroidCount = pImpl_->config.nCluster;  // use config if set
        }
        if (centroidCount == 0) {
            // Formula: clamp(N/100, 8, 4096)
            // Targets ~100 vectors/cluster for consistent per-cluster HNSW efficiency.
            // 270 chunks → 8 (min), 10K → 100, 213K → 2130, 1M → 4096 (max)
            centroidCount = std::max(static_cast<size_t>(8),
                                    std::min(static_cast<size_t>(4096),
                                             static_cast<size_t>(totalChunks) / 100));
            LOGI("Auto nCluster=%zu (from %u chunks, ~%u vectors/cluster)",
                 centroidCount, totalChunks, totalChunks / (uint32_t)centroidCount);
        }

        // Adjust cluster count if needed
        if (centroidCount > totalChunks) {
            centroidCount = totalChunks;
            LOGD("Reduced cluster count to %zu", centroidCount);
        }

        // 3. Calculate sample size for training (FAISS recommendation: 30*K to 256*K)
        //    Auto formula: min(nCluster * 100, maxTrainSamples, totalChunks)
        size_t sampleSize = std::min(centroidCount * 100, pImpl_->config.maxTrainSamples);
        sampleSize = std::min(sampleSize, static_cast<size_t>(totalChunks));

        LOGI("Sampling %zu chunks (out of %u) for FAISS training", sampleSize, totalChunks);

        // 4. Load sampled chunks with vectors
        auto sampledChunks = obxManager->getSampledChunks(sampleSize, false);
        if (sampledChunks.empty()) {
            LOGE("Failed to sample chunks");
            return false;
        }

        // Filter chunks with vectors
        std::vector<const ChunkData*> validSamples;
        for (const auto& chunk : sampledChunks) {
            if (!chunk.vector.empty()) {
                validSamples.push_back(&chunk);
            }
        }

        if (validSamples.empty()) {
            LOGE("No valid vector samples found");
            return false;
        }

        size_t dim = validSamples[0]->vector.size();
        LOGI("Sampled %zu valid vectors with dimension %zu", validSamples.size(), dim);

        // 5. Prepare embedding matrix for training (row-major, contiguous)
        std::vector<float> trainEmbeddings(validSamples.size() * dim);
        for (size_t i = 0; i < validSamples.size(); i++) {
            const auto& vec = validSamples[i]->vector;
            std::copy(vec.begin(), vec.end(), trainEmbeddings.begin() + i * dim);
        }

        // 6. Create output directory
        std::string indexDir = pImpl_->basePath + "/ecovector";
        fs::create_directories(indexDir);

        // 7. Create IVFFlat index for clustering (train on samples only)
        faiss::IndexFlatL2 quantizer(dim);
        faiss::IndexIVFFlat ivfIndex(&quantizer, dim, centroidCount, faiss::METRIC_L2);

        LOGD("Training IVF index on %zu samples...", validSamples.size());
        ivfIndex.train(validSamples.size(), trainEmbeddings.data());

        // Clear training data to free memory
        trainEmbeddings.clear();
        trainEmbeddings.shrink_to_fit();
        sampledChunks.clear();
        validSamples.clear();

        // 8. Extract centroids and build HNSW index for centroids
        std::vector<float> centroids(centroidCount * dim);
        for (size_t i = 0; i < centroidCount; i++) {
            quantizer.reconstruct(i, centroids.data() + i * dim);
        }

        // Force single-threaded HNSW construction for deterministic index building.
        // OpenMP parallel add_with_locks causes non-deterministic graph structure
        // due to lock acquisition order varying between runs.
        int prevOmpThreads = omp_get_max_threads();
        omp_set_num_threads(1);

        faiss::IndexHNSWFlat centroidIndex(dim, pImpl_->config.hnswM);
        centroidIndex.hnsw.efConstruction = pImpl_->config.hnswEfConstruction;
        centroidIndex.add(centroidCount, centroids.data());

        // Save centroid index
        std::string centroidPath = getCentroidIndexPath(pImpl_->basePath);
        faiss::write_index(&centroidIndex, centroidPath.c_str());
        LOGD("Saved centroid index to %s", centroidPath.c_str());

        // 9. Assign all chunks to clusters (batch processing to avoid OOM)
        pImpl_->clusterMappings.clear();

        // Load all chunks (metadata only) to get IDs
        auto allChunksMetadata = obxManager->getAllChunks(true); // excludeVectors = true
        size_t totalValidChunks = allChunksMetadata.size();

        LOGI("Processing %zu chunks in batches for cluster assignment...", totalValidChunks);

        // Batch size (balance between memory and performance)
        const size_t batchSize = 1000;
        size_t numBatches = (totalValidChunks + batchSize - 1) / batchSize;

        for (size_t batchIdx = 0; batchIdx < numBatches; batchIdx++) {
            size_t startIdx = batchIdx * batchSize;
            size_t endIdx = std::min(startIdx + batchSize, totalValidChunks);
            size_t currentBatchSize = endIdx - startIdx;

            // Extract chunk IDs for this batch
            std::vector<uint64_t> batchIds;
            batchIds.reserve(currentBatchSize);
            for (size_t i = startIdx; i < endIdx; i++) {
                batchIds.push_back(allChunksMetadata[i].id);
            }

            // Load chunks with vectors for this batch
            auto batchChunks = obxManager->getChunksByIds(batchIds, false, false);

            // Filter valid vectors and prepare embedding matrix
            std::vector<const ChunkData*> validBatchChunks;
            for (const auto& chunk : batchChunks) {
                if (!chunk.vector.empty()) {
                    validBatchChunks.push_back(&chunk);
                }
            }

            if (validBatchChunks.empty()) {
                continue;
            }

            // Prepare embeddings for this batch
            std::vector<float> batchEmbeddings(validBatchChunks.size() * dim);
            for (size_t i = 0; i < validBatchChunks.size(); i++) {
                const auto& vec = validBatchChunks[i]->vector;
                std::copy(vec.begin(), vec.end(), batchEmbeddings.begin() + i * dim);
            }

            // Assign to clusters
            std::vector<faiss::idx_t> batchAssignments(validBatchChunks.size());
            quantizer.assign(validBatchChunks.size(), batchEmbeddings.data(), batchAssignments.data());

            // Add to IVF index (for per-cluster HNSW building later)
            ivfIndex.add(validBatchChunks.size(), batchEmbeddings.data());

            // Build cluster mappings
            for (size_t i = 0; i < validBatchChunks.size(); i++) {
                int clusterId = static_cast<int>(batchAssignments[i]);
                pImpl_->clusterMappings[clusterId].push_back(validBatchChunks[i]->id);
            }

            if ((batchIdx + 1) % 10 == 0 || batchIdx == numBatches - 1) {
                LOGD("Processed batch %zu/%zu (%zu chunks)", batchIdx + 1, numBatches, endIdx);
            }
        }

        // Clear metadata to free memory
        allChunksMetadata.clear();

        // 10. Build per-cluster HNSW indices
        LOGI("Building per-cluster HNSW indices...");
        for (size_t listNo = 0; listNo < centroidCount; listNo++) {
            size_t listSize = ivfIndex.invlists->list_size(listNo);
            if (listSize == 0) continue;

            // Extract vectors for this cluster
            std::vector<float> listVectors(listSize * dim);
            const float* listCodes = reinterpret_cast<const float*>(
                ivfIndex.invlists->get_codes(listNo));
            for (size_t j = 0; j < listSize; j++) {
                std::memcpy(listVectors.data() + j * dim,
                           listCodes + j * dim,
                           dim * sizeof(float));
            }

            // Build HNSW index for this cluster
            faiss::IndexHNSWFlat listIndex(dim, pImpl_->config.hnswM);
            listIndex.hnsw.efConstruction = pImpl_->config.hnswEfConstruction;
            listIndex.add(listSize, listVectors.data());

            // Save cluster index
            std::string clusterPath = getClusterIndexPath(pImpl_->basePath, listNo);
            faiss::write_index(&listIndex, clusterPath.c_str());
            LOGD("Saved cluster %zu index (%zu vectors)", listNo, listSize);
        }

        // Restore original thread count
        omp_set_num_threads(prevOmpThreads);

        // 11. Save cluster mappings
        pImpl_->saveClusterMappings();

        // 12. Save metadata
        pImpl_->dimension = dim;
        pImpl_->numClusters = centroidCount;
        pImpl_->saveMetadata();

        // Clear cached indices (they'll be reloaded on demand)
        pImpl_->centroidIndex.reset();
        pImpl_->clusterIndices.clear();

        LOGI("EcoVector indexes created successfully: %zu clusters, %u vectors",
             centroidCount, totalChunks);
        return true;

    } catch (const std::exception& e) {
        LOGE("Error creating EcoVector indexes: %s", e.what());
        return false;
    }
}

bool EcoVectorIndex::addIndex(const ChunkData& chunk) {
    if (chunk.vector.empty()) {
        LOGE("Chunk has no vector");
        return false;
    }

    if (!pImpl_->ensureCentroidIndexLoaded()) {
        LOGE("Centroid index not available");
        return false;
    }

    try {
        size_t dim = chunk.vector.size();

        // 1. Find nearest centroid
        std::vector<float> distances(1);
        std::vector<faiss::idx_t> labels(1);
        pImpl_->centroidIndex->search(1, chunk.vector.data(), 1,
                                       distances.data(), labels.data());

        int clusterId = static_cast<int>(labels[0]);
        LOGD("Chunk %llu assigned to cluster %d", (unsigned long long)chunk.id, clusterId);

        // 2. Load cluster index
        faiss::IndexHNSWFlat* clusterIndex = pImpl_->getClusterIndex(clusterId);
        if (!clusterIndex) {
            // Create new cluster index if it doesn't exist
            auto newIndex = std::make_unique<faiss::IndexHNSWFlat>(dim, pImpl_->config.hnswM);
            newIndex->hnsw.efConstruction = pImpl_->config.hnswEfConstruction;
            clusterIndex = newIndex.get();
            pImpl_->clusterIndices[clusterId] = std::move(newIndex);
        }

        // 3. Add vector to cluster index
        clusterIndex->add(1, chunk.vector.data());

        // 4. Update cluster mapping
        if (!pImpl_->clusterMappings.count(clusterId)) {
            pImpl_->loadClusterMappings();
        }
        pImpl_->clusterMappings[clusterId].push_back(chunk.id);

        // 5. Save updated index and mapping
        std::string clusterPath = getClusterIndexPath(pImpl_->basePath, clusterId);
        faiss::write_index(clusterIndex, clusterPath.c_str());
        pImpl_->saveClusterMappings();

        LOGD("Added chunk %llu to cluster %d", (unsigned long long)chunk.id, clusterId);
        return true;

    } catch (const std::exception& e) {
        LOGE("Error adding index: %s", e.what());
        return false;
    }
}

std::vector<ChunkSearchResult> EcoVectorIndex::rerank_lcs(
    const std::vector<int32_t>& queryKiwiTokens,
    std::vector<ChunkSearchResult>& results)
{
    if (queryKiwiTokens.empty() || results.empty()) {
        return results;
    }

    try {
        // Calculate LCS score for each result using kiwiTokens directly
        // (already filtered to content words at write time)
        struct RankItem {
            size_t index;
            uint32_t lcsScore;
        };

        std::vector<RankItem> rankedItems;
        rankedItems.reserve(results.size());

        for (size_t i = 0; i < results.size(); ++i) {
            uint32_t lcsLength = calculateLCSLength(queryKiwiTokens, results[i].chunk.kiwiTokens);
            rankedItems.push_back({i, lcsLength});
        }

        // Sort by LCS score (descending) - maintain stable order for equal scores
        std::stable_sort(rankedItems.begin(), rankedItems.end(),
                         [](const RankItem& a, const RankItem& b) {
                             return a.lcsScore > b.lcsScore;
                         });

        // Extract sorted results
        std::vector<ChunkSearchResult> rerankedResults;
        rerankedResults.reserve(results.size());
        for (const auto& item : rankedItems) {
            rerankedResults.push_back(results[item.index]);
        }

        return rerankedResults;

    } catch (const std::exception& e) {
        LOGE("Error in rerank_lcs: %s", e.what());
        return results;
    }
}

// ============================================================================
// ID-only search (no ObxManager dependency)
// ============================================================================

std::vector<IndexSearchResult> EcoVectorIndex::searchIds(
    const std::vector<float>& queryVector,
    uint32_t limit,
    const std::unordered_set<uint64_t>* allowedChunkIds,
    size_t nprobe,
    size_t efSearch) {

    std::vector<IndexSearchResult> idResults;

    if (queryVector.empty()) {
        LOGE("Query vector is empty");
        return idResults;
    }

    if (!pImpl_->ensureCentroidIndexLoaded()) {
        LOGE("Centroid index not available");
        return idResults;
    }

    try {
        auto tTotal0 = std::chrono::high_resolution_clock::now();

        // Load cluster mappings if not loaded
        if (pImpl_->clusterMappings.empty()) {
            pImpl_->loadClusterMappings();
        }

        size_t numClusters = pImpl_->centroidIndex->ntotal;
        size_t dim = queryVector.size();

        // --- LocalResult struct for collecting per-cluster results ---
        struct LocalResult {
            float distance;
            int clusterId;
            size_t localIndex;
        };

        if (allowedChunkIds) {
            if (allowedChunkIds->empty()) {
                LOGD("searchIds: allowedChunkIds is empty, early return");
                return idResults;
            }

            if (nprobe == 0) {
                size_t autoNprobe = std::max(static_cast<size_t>(4), numClusters / 8);
                nprobe = std::min(autoNprobe, static_cast<size_t>(16));
            }
            size_t initialKCluster = std::min(numClusters, nprobe);
            size_t searchKPerCluster = std::max(static_cast<size_t>(limit),
                static_cast<size_t>(std::ceil(static_cast<float>(limit * 6) / initialKCluster)));

            // Centroid search for cluster ranking
            auto tCentroid0 = std::chrono::high_resolution_clock::now();
            faiss::SearchParametersHNSW centroidParams;
            centroidParams.efSearch = static_cast<int>(efSearch);

            std::vector<float> centroidDists(numClusters);
            std::vector<faiss::idx_t> centroidLabels(numClusters);
            pImpl_->centroidIndex->search(1, queryVector.data(), numClusters,
                                           centroidDists.data(), centroidLabels.data(),
                                           &centroidParams);
            auto tCentroid1 = std::chrono::high_resolution_clock::now();
            auto centroidUs = std::chrono::duration_cast<std::chrono::microseconds>(tCentroid1 - tCentroid0).count();

            // Search clusters in centroid order with IDSelectorBatch filter
            auto tCluster0 = std::chrono::high_resolution_clock::now();
            std::vector<LocalResult> allResults;
            allResults.reserve(initialKCluster * searchKPerCluster);
            size_t searched = 0;
            size_t ci = 0;

            for (; ci < numClusters && searched < initialKCluster; ci++) {
                int clusterId = static_cast<int>(centroidLabels[ci]);
                if (clusterId < 0) continue;

                auto* clusterIndex = pImpl_->getClusterIndex(clusterId);
                if (!clusterIndex || clusterIndex->ntotal == 0) continue;

                auto mapIt = pImpl_->clusterMappings.find(clusterId);
                if (mapIt == pImpl_->clusterMappings.end()) continue;
                const auto& mapping = mapIt->second;

                // Build filtered FAISS local IDs for this cluster
                std::vector<faiss::idx_t> allowedFaissIds;
                for (size_t j = 0; j < mapping.size(); j++) {
                    if (allowedChunkIds->count(mapping[j])) {
                        allowedFaissIds.push_back(static_cast<faiss::idx_t>(j));
                    }
                }
                if (allowedFaissIds.empty()) continue;  // skip cluster with no allowed chunks

                faiss::IDSelectorBatch selector(allowedFaissIds.size(), allowedFaissIds.data());
                faiss::SearchParametersHNSW searchParams;
                searchParams.sel = &selector;
                searchParams.efSearch = static_cast<int>(efSearch);

                size_t k = std::min(searchKPerCluster, static_cast<size_t>(clusterIndex->ntotal));
                std::vector<float> dists(k);
                std::vector<faiss::idx_t> labels(k);
                clusterIndex->search(1, queryVector.data(), k,
                                     dists.data(), labels.data(), &searchParams);

                for (size_t j = 0; j < k; j++) {
                    if (labels[j] >= 0) {
                        allResults.push_back({dists[j], clusterId, static_cast<size_t>(labels[j])});
                    }
                }
                searched++;
            }

            // Expand one cluster at a time if results insufficient
            for (; ci < numClusters && allResults.size() < limit; ci++) {
                int clusterId = static_cast<int>(centroidLabels[ci]);
                if (clusterId < 0) continue;

                auto* clusterIndex = pImpl_->getClusterIndex(clusterId);
                if (!clusterIndex || clusterIndex->ntotal == 0) continue;

                auto mapIt = pImpl_->clusterMappings.find(clusterId);
                if (mapIt == pImpl_->clusterMappings.end()) continue;
                const auto& mapping = mapIt->second;

                std::vector<faiss::idx_t> allowedFaissIds;
                for (size_t j = 0; j < mapping.size(); j++) {
                    if (allowedChunkIds->count(mapping[j])) {
                        allowedFaissIds.push_back(static_cast<faiss::idx_t>(j));
                    }
                }
                if (allowedFaissIds.empty()) continue;

                faiss::IDSelectorBatch selector(allowedFaissIds.size(), allowedFaissIds.data());
                faiss::SearchParametersHNSW searchParams;
                searchParams.sel = &selector;
                searchParams.efSearch = static_cast<int>(efSearch);

                size_t k = std::min(searchKPerCluster, static_cast<size_t>(clusterIndex->ntotal));
                std::vector<float> dists(k);
                std::vector<faiss::idx_t> labels(k);
                clusterIndex->search(1, queryVector.data(), k,
                                     dists.data(), labels.data(), &searchParams);

                for (size_t j = 0; j < k; j++) {
                    if (labels[j] >= 0) {
                        allResults.push_back({dists[j], clusterId, static_cast<size_t>(labels[j])});
                    }
                }
                searched++;
            }
            auto tCluster1 = std::chrono::high_resolution_clock::now();
            auto clusterUs = std::chrono::duration_cast<std::chrono::microseconds>(tCluster1 - tCluster0).count();

            // partial_sort top results
            auto tSort0 = std::chrono::high_resolution_clock::now();
            size_t sortLimit = std::min(allResults.size(), static_cast<size_t>(limit * 3));
            if (sortLimit > 0) {
                std::partial_sort(allResults.begin(),
                                  allResults.begin() + sortLimit,
                                  allResults.end(),
                                  [](const LocalResult& a, const LocalResult& b) {
                                      return a.distance < b.distance;
                                  });
                allResults.resize(sortLimit);
            }

            // Convert LocalResult → IndexSearchResult
            idResults.reserve(allResults.size());
            int lastClusterId = -1;
            const std::vector<uint64_t>* cachedChunkIds = nullptr;

            for (const auto& lr : allResults) {
                if (lr.clusterId != lastClusterId) {
                    auto mapIt = pImpl_->clusterMappings.find(lr.clusterId);
                    cachedChunkIds = (mapIt != pImpl_->clusterMappings.end()) ? &mapIt->second : nullptr;
                    lastClusterId = lr.clusterId;
                }
                if (!cachedChunkIds || lr.localIndex >= cachedChunkIds->size()) continue;
                idResults.push_back({(*cachedChunkIds)[lr.localIndex], lr.distance});
            }
            auto tSort1 = std::chrono::high_resolution_clock::now();
            auto sortUs = std::chrono::duration_cast<std::chrono::microseconds>(tSort1 - tSort0).count();
            auto totalUs = std::chrono::duration_cast<std::chrono::microseconds>(tSort1 - tTotal0).count();

            LOGD("[TIMING] searchIds Filtered: centroid=%lld cluster=%lld sort=%lld total=%lld us, "
                 "allowed=%zu searched=%zu results=%zu",
                 (long long)centroidUs, (long long)clusterUs, (long long)sortUs, (long long)totalUs,
                 allowedChunkIds->size(), searched, allResults.size());

            return idResults;
        }

        // === Unfiltered path: Centroid search → per-cluster HNSW ===

        if (nprobe == 0) {
            size_t autoNprobe = std::max(static_cast<size_t>(4), numClusters / 8);
            nprobe = std::min(autoNprobe, static_cast<size_t>(16));
        }
        size_t initialKCluster = std::min(numClusters, nprobe);
        size_t searchKPerCluster = std::max(static_cast<size_t>(limit),
            static_cast<size_t>(std::ceil(static_cast<float>(limit * 6) / initialKCluster)));

        // Step 3: Use SearchParametersHNSW instead of modifying member
        auto tCentroid0 = std::chrono::high_resolution_clock::now();
        faiss::SearchParametersHNSW centroidParams;
        centroidParams.efSearch = static_cast<int>(efSearch);

        std::vector<float> centroidDists(numClusters);
        std::vector<faiss::idx_t> centroidLabels(numClusters);
        pImpl_->centroidIndex->search(1, queryVector.data(), numClusters,
                                       centroidDists.data(), centroidLabels.data(),
                                       &centroidParams);
        auto tCentroid1 = std::chrono::high_resolution_clock::now();
        auto centroidUs = std::chrono::duration_cast<std::chrono::microseconds>(tCentroid1 - tCentroid0).count();

        // Pure function for unfiltered cluster search (thread-safe)
        auto searchClusterUnfiltered = [&queryVector, dim, efSearch, searchKPerCluster]
            (int clusterId, faiss::IndexHNSWFlat* clusterIndex) -> std::vector<LocalResult>
        {
            std::vector<LocalResult> results;
            if (!clusterIndex || clusterIndex->ntotal == 0) return results;

            size_t k = std::min(searchKPerCluster, static_cast<size_t>(clusterIndex->ntotal));

            faiss::SearchParametersHNSW searchParams;
            searchParams.efSearch = static_cast<int>(efSearch);

            std::vector<float> dists(k);
            std::vector<faiss::idx_t> labels(k);
            clusterIndex->search(1, queryVector.data(), k,
                                 dists.data(), labels.data(), &searchParams);

            results.reserve(k);
            for (size_t j = 0; j < k; j++) {
                if (labels[j] >= 0) {
                    results.push_back({dists[j], clusterId, static_cast<size_t>(labels[j])});
                }
            }
            return results;
        };

        auto tCluster0 = std::chrono::high_resolution_clock::now();
        std::vector<LocalResult> allResults;
        allResults.reserve(initialKCluster * searchKPerCluster);

        // Initial cluster search
        for (size_t i = 0; i < initialKCluster; i++) {
            int clusterId = static_cast<int>(centroidLabels[i]);
            if (clusterId < 0) continue;
            auto* clusterIndex = pImpl_->getClusterIndex(clusterId);
            auto clusterResults = searchClusterUnfiltered(clusterId, clusterIndex);
            allResults.insert(allResults.end(), clusterResults.begin(), clusterResults.end());
        }

        // Adaptive expansion
        size_t nextCluster = initialKCluster;
        while (allResults.size() < limit && nextCluster < numClusters) {
            int clusterId = static_cast<int>(centroidLabels[nextCluster]);
            if (clusterId >= 0) {
                auto* clusterIndex = pImpl_->getClusterIndex(clusterId);
                auto clusterResults = searchClusterUnfiltered(clusterId, clusterIndex);
                allResults.insert(allResults.end(), clusterResults.begin(), clusterResults.end());
            }
            nextCluster++;
        }
        auto tCluster1 = std::chrono::high_resolution_clock::now();
        auto clusterUs = std::chrono::duration_cast<std::chrono::microseconds>(tCluster1 - tCluster0).count();

        // partial_sort top results
        auto tSort0 = std::chrono::high_resolution_clock::now();
        size_t sortLimit = std::min(allResults.size(), static_cast<size_t>(limit * 3));
        if (sortLimit > 0) {
            std::partial_sort(allResults.begin(),
                              allResults.begin() + sortLimit,
                              allResults.end(),
                              [](const LocalResult& a, const LocalResult& b) {
                                  return a.distance < b.distance;
                              });
            allResults.resize(sortLimit);
        }

        // Convert LocalResult → IndexSearchResult
        idResults.reserve(allResults.size());

        int lastClusterId = -1;
        const std::vector<uint64_t>* cachedChunkIds = nullptr;

        for (const auto& lr : allResults) {
            if (lr.clusterId != lastClusterId) {
                auto mapIt = pImpl_->clusterMappings.find(lr.clusterId);
                cachedChunkIds = (mapIt != pImpl_->clusterMappings.end()) ? &mapIt->second : nullptr;
                lastClusterId = lr.clusterId;
            }
            if (!cachedChunkIds || lr.localIndex >= cachedChunkIds->size()) continue;
            idResults.push_back({(*cachedChunkIds)[lr.localIndex], lr.distance});
        }
        auto tSort1 = std::chrono::high_resolution_clock::now();
        auto sortUs = std::chrono::duration_cast<std::chrono::microseconds>(tSort1 - tSort0).count();
        auto totalUs = std::chrono::duration_cast<std::chrono::microseconds>(tSort1 - tTotal0).count();

        LOGD("[TIMING] searchIds Unfiltered: centroid=%lld cluster=%lld sort=%lld total=%lld us, "
             "clusters=%zu results=%zu",
             (long long)centroidUs, (long long)clusterUs, (long long)sortUs, (long long)totalUs,
             initialKCluster, allResults.size());

        return idResults;

    } catch (const std::exception& e) {
        LOGE("Error in EcoVector searchIds: %s", e.what());
        return idResults;
    }
}

// ============================================================================
// Full search (searchIds + hydration + reranking)
// ============================================================================

std::vector<ChunkSearchResult> EcoVectorIndex::search(
    ObxManager* obxManager,
    const std::vector<float>& queryVector,
    uint32_t limit,
    RerankerType rerankerType,
    const std::vector<int32_t>& queryKiwiTokens,
    const std::unordered_set<uint64_t>* allowedChunkIds,
    size_t nprobe,
    size_t efSearch) {

    auto searchStartTime = std::chrono::high_resolution_clock::now();

    std::vector<ChunkSearchResult> results;

    if (!obxManager) {
        LOGE("ObxManager is null");
        return results;
    }

    // Check kiwiTokens availability for reranking
    if (rerankerType != RerankerType::NONE) {
        if (queryKiwiTokens.empty()) {
            LOGW("Query kiwiTokens empty but reranking requested, disabling reranking");
            rerankerType = RerankerType::NONE;
        }
    }

    // 1. Index-level search: ID+distance only
    auto idResults = searchIds(queryVector, limit, allowedChunkIds, nprobe, efSearch);
    if (idResults.empty()) return results;

    // 2. Hydrate: batch fetch chunks from ObxManager
    std::vector<uint64_t> chunkIds;
    chunkIds.reserve(idResults.size());
    for (const auto& r : idResults) {
        chunkIds.push_back(r.chunkId);
    }

    auto chunks = obxManager->getChunksByIds(chunkIds, true, true);

    // 3. Build ChunkSearchResult (preserve ID order from searchIds)
    std::unordered_map<uint64_t, size_t> idToIndex;
    for (size_t i = 0; i < chunks.size(); i++) {
        idToIndex[chunks[i].id] = i;
    }

    std::vector<ChunkSearchResult> preRankResults;
    preRankResults.reserve(idResults.size());
    for (const auto& ir : idResults) {
        auto it = idToIndex.find(ir.chunkId);
        if (it == idToIndex.end()) continue;

        ChunkSearchResult sr;
        sr.chunk = std::move(chunks[it->second]);
        sr.distance = ir.score;
        preRankResults.push_back(std::move(sr));
    }

    // 4. Apply reranking if requested
    if (rerankerType != RerankerType::NONE) {
        LOGD("Applying %s reranking with %zu kiwi tokens",
             rerankerTypeToString(rerankerType), queryKiwiTokens.size());
        results = rerank_lcs(queryKiwiTokens, preRankResults);
    } else {
        results = std::move(preRankResults);
    }

    // 5. Truncate to limit
    if (results.size() > limit) {
        results.resize(limit);
    }

    // 6. Log total search time
    auto searchEndTime = std::chrono::high_resolution_clock::now();
    auto totalDuration = std::chrono::duration_cast<std::chrono::microseconds>(searchEndTime - searchStartTime);
    LOGD("[BENCHMARK] Total search: limit=%u, results=%zu, reranker=%s, time=%lld us (%.3f ms)",
         limit, results.size(), rerankerTypeToString(rerankerType),
         (long long)totalDuration.count(), totalDuration.count() / 1000.0);

    return results;
}

// ============================================================================
// Overload: String-based reranker type specification
// ============================================================================

std::vector<ChunkSearchResult> EcoVectorIndex::search(
    ObxManager* obxManager,
    const std::vector<float>& queryVector,
    uint32_t limit,
    const std::string& rerankerTypeStr,
    const std::vector<int32_t>& queryKiwiTokens,
    const std::unordered_set<uint64_t>* allowedChunkIds,
    size_t nprobe,
    size_t efSearch) {

    RerankerType rerankerType;
    try {
        rerankerType = stringToRerankerType(rerankerTypeStr);
    } catch (const std::invalid_argument& e) {
        LOGE("Failed to parse reranker type: %s", e.what());
        rerankerType = RerankerType::NONE;
    }

    return search(obxManager, queryVector, limit, rerankerType, queryKiwiTokens, allowedChunkIds, nprobe, efSearch);
}

bool EcoVectorIndex::isIndexReady() const {
    std::string centroidPath = getCentroidIndexPath(pImpl_->basePath);
    std::string mappingPath = getClusterMappingPath(pImpl_->basePath);
    std::string metadataPath = getMetadataPath(pImpl_->basePath);

    return fs::exists(centroidPath) && fs::exists(mappingPath) && fs::exists(metadataPath);
}

bool EcoVectorIndex::preloadIndexes() {
    if (!isIndexReady()) {
        LOGE("EcoVector indexes not ready");
        return false;
    }

    LOGI("Preloading EcoVector indexes...");

    try {
        // 1. Load metadata
        if (!pImpl_->loadMetadata()) {
            LOGE("Failed to load metadata");
            return false;
        }

        // 2. Load centroid index
        if (!pImpl_->ensureCentroidIndexLoaded()) {
            LOGE("Failed to load centroid index");
            return false;
        }

        // 3. Load cluster mappings
        if (!pImpl_->loadClusterMappings()) {
            LOGE("Failed to load cluster mappings");
            return false;
        }

        // 4. Preload all cluster indexes
        for (size_t i = 0; i < pImpl_->numClusters; i++) {
            pImpl_->getClusterIndex(static_cast<int>(i));
        }

        LOGI("Preloaded %zu cluster indexes", pImpl_->numClusters);
        return true;

    } catch (const std::exception& e) {
        LOGE("Error preloading indexes: %s", e.what());
        return false;
    }
}

void EcoVectorIndex::removeIndexes() {
    std::string indexDir = pImpl_->basePath + "/ecovector";
    if (fs::exists(indexDir)) {
        fs::remove_all(indexDir);
        LOGI("Removed EcoVector indexes at %s", indexDir.c_str());
    }

    pImpl_->centroidIndex.reset();
    pImpl_->clusterIndices.clear();
    pImpl_->clusterMappings.clear();
    pImpl_->dimension = 0;
    pImpl_->numClusters = 0;
}

size_t EcoVectorIndex::getClusterCount() const {
    return pImpl_->numClusters;
}

size_t EcoVectorIndex::getDimension() const {
    return pImpl_->dimension;
}

} // namespace ecovector
