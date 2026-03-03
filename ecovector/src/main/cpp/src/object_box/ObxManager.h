#ifndef ECOVECTOR_OBX_MANAGER_H
#define ECOVECTOR_OBX_MANAGER_H

#include <string>
#include <vector>
#include <cstdint>
#include <memory>
#include <optional>
#include <functional>
#include <unordered_set>
namespace ecovector {

// Forward declarations
class EcoVectorIndex;
class BM25Index;
class Tokenizer;
class KiwiTokenizer;


// ============================================================================
// Public API Data structures
// Use different names to avoid conflict with ObjectBox generated types
// ============================================================================

enum class SourceType : int16_t {
    CALL = 0,
    SMS = 1,
    MMS = 2,
    IMAGE = 3,
    FILE = 4
};

struct DocData {
    uint64_t id = 0;
    std::string externalId;     // _id
    std::string description;
    std::string content;
    int64_t createdAt = 0;      // unix timestamp ms
    SourceType sourceType = SourceType::CALL;
    std::string sender;
};

struct ChunkData {
    uint64_t id = 0;
    uint64_t documentId = 0;
    int32_t chunkIndex = 0;
    std::string content;
    std::vector<float> vector;
    std::vector<int32_t> tokenIds;
    std::vector<int32_t> kiwiTokens;  // hashed Kiwi morphemes
    int64_t createdAt = 0;
    SourceType sourceType = SourceType::CALL;
    std::string sender;
    std::string documentExternalId;
};

// Vector search result with distance score and benchmark info
struct ChunkSearchResult {
    ChunkData chunk;
    float distance = 0.0f;  // Distance to query vector (lower = more similar)

    // Benchmark information (shared across all results in a search)
    int64_t benchmarkNoReranker = 0;      // Time without reranker in microseconds
    int64_t benchmarkLCS = 0;             // Time with LCS reranker in microseconds
    int64_t benchmarkTokenIntersection = 0; // Time with token intersection reranker in microseconds
};

// ============================================================================
// ObxManager - ObjectBox Database Manager
// ============================================================================

class ObxManager {
private:
    class Impl;
    std::unique_ptr<Impl> pImpl_;

public:
    explicit ObxManager(const std::string& dbPath);
    ~ObxManager();

    // Prevent copy/move
    ObxManager(const ObxManager&) = delete;
    ObxManager& operator=(const ObxManager&) = delete;
    ObxManager(ObxManager&&) = delete;
    ObxManager& operator=(ObxManager&&) = delete;

    // Initialize ObjectBox database (must be called before first use)
    // kiwiModelPath가 비어있으면 Kiwi 없이 초기화 (BM25 사용 불가)
    bool initialize(const std::string& kiwiModelPath = "");

    // 외부 Tokenizer 설정 (리랭킹용, non-owning)
    void setTokenizer(Tokenizer* tokenizer);

    // 외부 EcoVectorIndex 설정 (non-owning, EcoVectorStore가 소유)
    void setEcoVectorIndex(EcoVectorIndex* ecoVectorIndex);

    // 외부 BM25Index 설정 (non-owning, EcoVectorStore가 소유)
    void setBM25Index(BM25Index* bm25Index);

    // 외부 KiwiTokenizer 설정 (non-owning, EcoVectorStore가 소유)
    void setKiwiTokenizer(KiwiTokenizer* kiwiTokenizer);

    // ==================== Read Operations ====================

    // Bulk read (all records)
    std::vector<DocData>   getAllDocuments();
    std::vector<std::string> getAllDocumentExternalIds();
    std::vector<ChunkData> getAllChunks(bool excludeVectors = true);
    void forEachChunkBatch(size_t batchSize, bool excludeVectors,
                           const std::function<void(const std::vector<ChunkData>&)>& callback);
    // Count-only queries (no data loading, O(1))
    uint32_t getDocumentCount();
    uint32_t getChunkCount();

    // Single-item lookup by internal ObjectBox id (O(1))
    std::optional<DocData>   getDocumentById(uint64_t id);
    std::optional<ChunkData> getChunkById(uint64_t id, bool excludeVector = true);

    // Lookup by external string id (uses query index, not full scan)
    std::optional<DocData>   getDocumentByExternalId(const std::string& externalId);

    // First chunk of a document (chunkIndex=0 or smallest), vector excluded by default
    std::optional<ChunkData> getFirstChunkByDocumentId(uint64_t docId, bool excludeVector = true);

    // Batch lookup by IDs
    std::vector<ChunkData> getChunksByIds(const std::vector<uint64_t>& ids,
                                          bool excludeVectors = true,
                                          bool excludeTokenIds = false,
                                          bool excludeKiwiTokens = false);

    // Sampled chunks (stride-based uniform sampling for memory-efficient clustering)
    std::vector<ChunkData> getSampledChunks(size_t sampleSize, bool excludeVectors = true);

    // ==================== Vector Search Operations ====================

    struct VectorSearchResult {
        uint64_t chunkId;
        float distance;  // L2 Euclidean distance (lower = more similar)
    };

    std::vector<VectorSearchResult> vectorSearch(
        const float* queryVector, size_t dimensions,
        size_t maxResultCount, uint32_t topK);

    // ==================== Metadata Filter Operations ====================

    // JSON 필터 → 매칭 chunk ID set 변환
    // filterJson 형식: {"source_type": 1, "sender": "user", "created_at": {"gte": 123}}
    std::unordered_set<uint64_t> resolveFilter(const std::string& filterJson);

    // Bulk update source_type on chunks by inferring from document_external_id prefix.
    // Returns number of chunks updated.
    int bulkUpdateSourceTypeFromPrefix();
    bool isSourceTypePatched();

    // Bulk update created_at on chunks by document external ID.
    // Input: map of docExternalId → createdAtMs (Unix milliseconds)
    // Returns number of chunks updated.
    int bulkUpdateCreatedAt(const std::unordered_map<std::string, int64_t>& docDateMap);

    // Bulk update source_type on chunks by document external ID.
    // Input: map of docExternalId → sourceType (int16_t)
    // Only updates chunks where source_type differs. Returns number of chunks updated.
    int bulkUpdateSourceType(const std::unordered_map<std::string, int16_t>& docSourceTypeMap);

    // ==================== Integrity Operations ====================

    // Find and remove documents that have zero chunks (crash recovery).
    // Returns external IDs of removed documents.
    std::vector<std::string> removeOrphanDocuments();

    // ==================== Write Operations ====================

    void removeAll();

    // ---- Single-item Create (returns new id, 0 on failure) ----
    uint64_t insertDocument(const DocData& doc);
    uint64_t insertChunk(const ChunkData& chunk);

    // ---- Single-item Update (id must be > 0) ----
    bool updateDocument(const DocData& doc);
    bool updateChunk(const ChunkData& chunk);

    // ---- Single-item Delete ----
    bool removeDocumentById(uint64_t id);
    bool removeChunkById(uint64_t id);

    // ---- Batch Create (id fields must be 0; auto-assigned) ----
    std::vector<uint64_t> insertAllDocuments(const std::vector<DocData>& documents);
    std::vector<uint64_t> insertAllChunks(const std::vector<ChunkData>& chunks);

    // ---- Batch Update (id == 0 entries skipped with warning) ----
    bool updateAllDocuments(const std::vector<DocData>& documents);
    bool updateAllChunks(const std::vector<ChunkData>& chunks);

    // ---- Batch Delete ----
    void removeDocumentsByIds(const std::vector<uint64_t>& ids);
    void removeChunksByIds(const std::vector<uint64_t>& ids);

    // Remove all chunks (documents are preserved)
    void removeAllChunks();

    // Stream chunks that have empty vector, in batches
    void forEachChunkWithoutVector(size_t batchSize,
        const std::function<bool(std::vector<ChunkData>&)>& callback);

    // Re-tokenize all stored chunks with current Kiwi dictionary,
    // then rebuild BM25 index. Embeddings are preserved unchanged.
    bool reTokenizeAllKiwiTokens();

    // ==================== Kiwi Tokenize (for diagnostics) ====================

    // Kiwi 형태소 분석 (검색용 — 쿼리 토큰화)
    std::vector<std::string> kiwiTokenize(const std::string& text) const;

    // Kiwi 형태소 분석 (인덱싱용 — 문서 토큰화)
    std::vector<std::string> kiwiTokenizeForIndexing(const std::string& text) const;

};

} // namespace ecovector

#endif // ECOVECTOR_OBX_MANAGER_H
