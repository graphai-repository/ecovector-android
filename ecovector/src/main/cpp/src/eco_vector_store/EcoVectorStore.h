#pragma once

#include <memory>
#include <string>
#include <vector>
#include <cstdint>
#include "VectorRetriever.h"
#include "ObxVectorRetriever.h"
#include "BM25Retriever.h"
#include "EnsembleRetriever.h"

namespace ecovector { class EcoVectorIndex; class BM25Index; }

// Forward declarations
class OnnxRuntime;

namespace ecovector {

class ObxManager;
class Tokenizer;
class KiwiTokenizer;
class Embedder;

class EcoVectorStore {
public:
    EcoVectorStore();
    ~EcoVectorStore();

    // Prevent copy/move
    EcoVectorStore(const EcoVectorStore&) = delete;
    EcoVectorStore& operator=(const EcoVectorStore&) = delete;

    // Lifecycle
    bool initialize(const std::string& dbPath,
                    const std::string& tokenizerPath,
                    const std::string& modelPath,
                    const std::string& kiwiModelDir);
    void close();

    // Chunk prefix — prepended to each chunk after splitting
    void setChunkPrefix(const std::string& prefix) { chunkPrefix_ = prefix; }
    void clearChunkPrefix() { chunkPrefix_.clear(); }

    // Document management (auto chunk + tokenize + embed + save)
    int64_t addDocument(const std::string& text, const std::string& title,
                        const std::string& externalId = "");
    std::vector<int64_t> addDocuments(const std::vector<std::string>& texts,
                                      const std::vector<std::string>& titles,
                                      const std::vector<std::string>& externalIds = {},
                                      const std::vector<int64_t>& createdAts = {},
                                      const std::vector<std::string>& senders = {},
                                      const std::vector<int16_t>& sourceTypes = {});
    void removeAll();
    std::string removeOrphanDocuments();

    // Index management
    bool buildIndex(int centroidCount = 0);
    bool isIndexReady();

    // Re-tokenize all stored chunks with the current Kiwi dictionary
    // and rebuild BM25 index. Embeddings remain unchanged.
    // Use after updating user_dict.tsv / synonyms.tsv without full reindex.
    bool reTokenizeAll();

    // Statistics
    int getDocumentCount();
    int getChunkCount();

    // Data inspection (paginated JSON)
    std::string getDocumentsJson(int offset, int limit);
    std::string getChunksJson(int offset, int limit);

    // Lightweight ID-only access (for incremental loading)
    std::string getDocumentExternalIdsJson();

    // Raw save (pre-computed vectors, bypasses chunk/tokenize/embed pipeline)
    int64_t saveDocumentRaw(const std::string& externalId,
                            const std::string& description,
                            const std::string& content,
                            int64_t createdAt,
                            int16_t sourceType,
                            const std::string& sender);
    int64_t saveChunkRaw(int64_t documentId, int32_t chunkIndex,
                         const std::string& content,
                         const std::vector<int32_t>& tokenIds,
                         const std::vector<float>& embedding,
                         const std::vector<int32_t>& kiwiTokens,
                         int64_t createdAt, int16_t sourceType,
                         const std::string& sender);

    // Accessors (non-owning)
    ObxManager* getObxManager() { return obxManager_.get(); }
    Tokenizer* getTokenizer() { return tokenizer_.get(); }
    KiwiTokenizer* getKiwiTokenizer() { return kiwiTokenizer_.get(); }
    const std::string& getDbPath() const { return dbPath_; }

    // Public tokenization & embedding
    std::vector<float> embedText(const std::string& text);
    std::vector<int32_t> tokenizeText(const std::string& text);

    // === Retriever 팩토리 ===
    VectorRetriever* createVectorRetriever(
        const VectorRetriever::Params& params = {});
    ObxVectorRetriever* createObxVectorRetriever(
        const ObxVectorRetriever::Params& params = {});
    BM25Retriever* createBM25Retriever(
        const BM25Retriever::Params& params = {});
    EnsembleRetriever* createEnsembleRetriever(
        std::vector<RetrieverConfig> configs,
        const EnsembleRetriever::Params& params = {});

    /** Remove and destroy a retriever from the owned list. */
    bool destroyRetriever(IRetriever* retriever);

    // === 분리된 인덱스 빌드 ===
    bool buildVectorIndex(int centroidCount = 0, int hnswM = 0, int efConstruction = 0,
                          int maxTrainSamples = 0);
    bool buildBM25Index();
    // 기존 buildIndex()는 유지: buildVectorIndex + buildBM25Index

    // === ChunkParams 지원 addDocument ===
    int64_t addDocumentWithChunkParams(
        const std::string& text, const std::string& title,
        int maxTokens, int overlapTokens);
    int addDocumentsWithChunkParams(
        const std::vector<std::string>& texts,
        const std::vector<std::string>& titles,
        int maxTokens, int overlapTokens);

    // === 데이터 관리 ===
    bool removeDocumentById(uint64_t id);
    bool removeChunkById(uint64_t id);

    // === 데이터 접근 ===
    std::string getDocumentJson(uint64_t id);
    std::string getChunksByDocumentJson(uint64_t docId);
    std::string getChunkJson(uint64_t id);

    // === Pipeline Stage Methods ===
    int addDocumentsOnly(
        const std::vector<std::string>& texts,
        const std::vector<std::string>& titles,
        const std::vector<std::string>& externalIds,
        const std::vector<int64_t>& createdAts,
        const std::vector<std::string>& senders,
        const std::vector<int16_t>& sourceTypes);
    int chunkAllDocuments();
    int embedChunks(bool forceAll);
    int tokenizeChunks();

    // === SQLite Export/Import ===
    int exportChunksToSQLite(const std::string& sqlitePath);
    int importEmbeddingsFromSQLite(const std::string& sqlitePath);

    // === Embedder 접근 ===
    Embedder* getEmbedder() { return embedder_.get(); }

    // === Index 접근 ===
    EcoVectorIndex* getEcoVectorIndex() { return ecoVectorIndex_.get(); }
    BM25Index* getBM25Index() { return bm25Index_.get(); }

private:
    std::unique_ptr<ObxManager> obxManager_;
    std::unique_ptr<EcoVectorIndex> ecoVectorIndex_;
    std::unique_ptr<BM25Index> bm25Index_;
    std::unique_ptr<Tokenizer> tokenizer_;
    std::unique_ptr<OnnxRuntime> onnxRuntime_;
    std::unique_ptr<KiwiTokenizer> kiwiTokenizer_;
    std::unique_ptr<Embedder> embedder_;
    std::vector<std::unique_ptr<IRetriever>> ownedRetrievers_;  // retrievers created by factory methods
    std::string dbPath_;
    std::string chunkPrefix_;  // 각 청크에 prepend할 접두어 (빈 문자열이면 비활성)
    size_t kiwiCallCount_ = 0;  // Kiwi tokenization call counter (for periodic reload)

    // Internal helpers
    std::vector<std::string> chunkText(const std::string& text);
    std::vector<std::string> chunkText(const std::string& text, int16_t sourceType);
    std::vector<std::string> chunkTextWithParams(const std::string& text,
                                                  int maxTokens, int overlapTokens);
    static std::string utf8Truncate(const std::string& str, size_t maxBytes);
};

} // namespace ecovector
