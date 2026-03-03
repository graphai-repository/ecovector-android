// Define OBX_CPP_FILE before including objectbox.hpp (required for linking)
#define OBX_CPP_FILE

#include "ObxManager.h"
#include "EcoVectorIndex.h"
#include "BM25Index.h"
#include "../tokenizer/Tokenizer.h"
#include "../kiwi/KiwiTokenizer.h"
#include "../kiwi/KiwiHashUtil.h"
#include "objectbox.h"
#include "objectbox.hpp"
#include "objectbox-model.h"
#include "schema.obx.hpp"
#include <json.hpp>

#include <android/log.h>
#include <algorithm>
#include <stdexcept>

#define LOG_TAG "ObxManager"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)

namespace ecovector {

// ============================================================================
// Helper Functions
// ============================================================================

// Binary conversion helpers for token IDs
// Convert int32_t vector to ubyte vector (binary)
static std::vector<uint8_t> tokenIdsToBinary(const std::vector<int32_t>& tokenIds) {
    std::vector<uint8_t> result;
    if (tokenIds.empty()) {
        return result;
    }
    result.resize(tokenIds.size() * sizeof(int32_t));
    std::memcpy(result.data(), tokenIds.data(), result.size());
    return result;
}

// Convert ubyte vector (binary) to int32_t vector
static std::vector<int32_t> binaryToTokenIds(const std::vector<uint8_t>& binaryData) {
    std::vector<int32_t> result;
    if (binaryData.empty()) {
        return result;
    }
    size_t count = binaryData.size() / sizeof(int32_t);
    result.resize(count);
    std::memcpy(result.data(), binaryData.data(), count * sizeof(int32_t));
    return result;
}


// ============================================================================
// Entity Conversion Helpers  (Data ↔ ObjectBox generated types)
// ============================================================================

// Default write-transaction batch size for mobile devices.
static const size_t WRITE_BATCH_SIZE = 64;

static ecovector::Document toObxDoc(const DocData& d, bool forInsert) {
    ecovector::Document o;
    o.id          = forInsert ? 0 : d.id;
    o.content     = d.content;
    o._id         = d.externalId;
    o.description = d.description;
    o.created_at  = d.createdAt;
    o.source_type = static_cast<int16_t>(d.sourceType);
    o.sender      = d.sender;
    return o;
}

static DocData fromObxDoc(const ecovector::Document& o) {
    DocData d;
    d.id          = o.id;
    d.externalId  = o._id;
    d.description = o.description;
    d.content     = o.content;
    d.createdAt   = o.created_at;
    d.sourceType  = static_cast<SourceType>(o.source_type);
    d.sender      = o.sender;
    return d;
}

static ecovector::Chunk toObxChunk(const ChunkData& c, bool forInsert) {
    ecovector::Chunk o;
    o.id                   = forInsert ? 0 : c.id;
    o.document_id          = c.documentId;
    o.content              = c.content;
    o.vector               = c.vector;
    o.token_ids            = tokenIdsToBinary(c.tokenIds);
    o.chunk_index          = c.chunkIndex;
    o.kiwi_tokens          = tokenIdsToBinary(c.kiwiTokens);
    o.created_at           = c.createdAt;
    o.source_type          = static_cast<int16_t>(c.sourceType);
    o.sender               = c.sender;
    o.document_external_id = c.documentExternalId;
    return o;
}

// includeTokenIds=false: skip HF tokenIds to save memory
// includeKiwiTokens=false: skip kiwi token deserialization when not needed for search
static ChunkData fromObxChunk(const ecovector::Chunk& o,
                               bool includeVector,
                               bool includeTokenIds = true,
                               bool includeKiwiTokens = true) {
    ChunkData c;
    c.id                 = o.id;
    c.documentId         = o.document_id;
    c.content            = o.content;
    c.chunkIndex         = o.chunk_index;
    if (includeKiwiTokens) c.kiwiTokens = binaryToTokenIds(o.kiwi_tokens);
    c.createdAt          = o.created_at;
    c.sourceType         = static_cast<SourceType>(o.source_type);
    c.sender             = o.sender;
    c.documentExternalId = o.document_external_id;
    if (includeTokenIds) c.tokenIds = binaryToTokenIds(o.token_ids);
    if (includeVector)   c.vector   = o.vector;
    return c;
}

// toObxQuery/fromObxQuery removed — Query entity retired, now in BenchmarkObxManager

// ============================================================================
// PIMPL Implementation
// ============================================================================

class ObxManager::Impl {
public:
    std::string dbPath;
    std::unique_ptr<obx::Store> store;

    // Boxes using generated types from schema.obx.hpp (ecovector::Document, etc.)
    std::unique_ptr<obx::Box<ecovector::Document>> boxDocument;
    std::unique_ptr<obx::Box<ecovector::Chunk>> boxChunk;

    // EcoVector for clustered vector search (non-owning, set by EcoVectorStore)
    EcoVectorIndex* ecoVectorIndex = nullptr;

    // BM25 for text-based search (non-owning, set by EcoVectorStore)
    BM25Index* bm25Index = nullptr;

    // Tokenizer for reranking (owned or external)
    std::unique_ptr<Tokenizer> ownedTokenizer;
    Tokenizer* tokenizer = nullptr;  // points to ownedTokenizer or external

    // Kiwi for BM25 morphological analysis (non-owning, set by EcoVectorStore)
    KiwiTokenizer* kiwiTokenizer = nullptr;

    explicit Impl(const std::string& path) : dbPath(path) {}

    void ensureInitialized() const {
        if (!store) {
            throw std::runtime_error("ObxManager not initialized. Call initialize() first.");
        }
    }
};

// ============================================================================
// Construction / Destruction
// ============================================================================

ObxManager::ObxManager(const std::string& dbPath)
    : pImpl_(std::make_unique<Impl>(dbPath)) {
    LOGI("ObxManager constructed with path: %s", dbPath.c_str());
}

void ObxManager::setTokenizer(Tokenizer* tokenizer) {
    pImpl_->tokenizer = tokenizer;
    LOGI("External tokenizer set for ObxManager%s", tokenizer ? "" : " (null)");
}

void ObxManager::setEcoVectorIndex(EcoVectorIndex* ecoVectorIndex) {
    pImpl_->ecoVectorIndex = ecoVectorIndex;
    LOGI("External EcoVectorIndex set for ObxManager%s", ecoVectorIndex ? "" : " (null)");
}

void ObxManager::setBM25Index(BM25Index* bm25Index) {
    pImpl_->bm25Index = bm25Index;
    LOGI("External BM25Index set for ObxManager%s", bm25Index ? "" : " (null)");
}

void ObxManager::setKiwiTokenizer(KiwiTokenizer* kiwiTokenizer) {
    pImpl_->kiwiTokenizer = kiwiTokenizer;
    LOGI("External KiwiTokenizer set for ObxManager%s", kiwiTokenizer ? "" : " (null)");
}

ObxManager::~ObxManager() {
    LOGD("ObxManager destructor called");

    if (pImpl_) {
        // Reset boxes first (they reference the store)
        pImpl_->boxDocument.reset();
        pImpl_->boxChunk.reset();

        // Close store
        if (pImpl_->store) {
            try {
                pImpl_->store->close();
                LOGI("ObjectBox store closed successfully");
            } catch (const std::exception& e) {
                LOGE("Error closing store: %s", e.what());
            }
        }
    }
}

// ============================================================================
// Initialization
// ============================================================================

bool ObxManager::initialize(const std::string& kiwiModelPath) {
    LOGI("Initializing ObjectBox database...");

    if (pImpl_->store) {
        LOGD("Store already initialized");
        return true;
    }

    try {
        // Create model from generated code
        OBX_model* model = create_obx_model();
        if (!model) {
            LOGE("Failed to create ObjectBox model");
            return false;
        }

        // Configure options using C++ API
        obx::Options options;
        options.model(model);  // ownership transferred
        options.directory(pImpl_->dbPath);
        options.maxDbSizeInKb(4 * 1024 * 1024);  // 4GB (default 1GB)

        // Create store
        pImpl_->store = std::make_unique<obx::Store>(options);
        LOGD("ObjectBox store created");

        // Create boxes (using generated types from schema.obx.hpp)
        pImpl_->boxDocument = std::make_unique<obx::Box<ecovector::Document>>(*pImpl_->store);
        pImpl_->boxChunk = std::make_unique<obx::Box<ecovector::Chunk>>(*pImpl_->store);

        LOGI("ObjectBox initialized - Documents: %llu, Chunks: %llu",
             (unsigned long long)pImpl_->boxDocument->count(),
             (unsigned long long)pImpl_->boxChunk->count());

        // KiwiTokenizer and BM25Index are now owned by EcoVectorStore
        // and injected via setKiwiTokenizer() / setBM25Index()

        // Tokenizer for reranking: set externally via setTokenizer() or setTokenizerPath()

        return true;

    } catch (const obx::Exception& e) {
        LOGE("ObjectBox exception: %s (code: %d)", e.what(), e.code());
        pImpl_->store.reset();
        return false;
    } catch (const std::exception& e) {
        LOGE("Exception during initialization: %s", e.what());
        pImpl_->store.reset();
        return false;
    }
}

// ============================================================================
// Read Operations
// ============================================================================

std::vector<DocData> ObxManager::getAllDocuments() {
    std::vector<DocData> result;
    pImpl_->ensureInitialized();
    try {
        auto all = pImpl_->boxDocument->getAll();
        result.reserve(all.size());
        for (const auto& doc : all) result.push_back(fromObxDoc(*doc));
        LOGI("Retrieved %zu documents", result.size());
    } catch (const std::exception& e) {
        LOGE("Error in getAllDocuments: %s", e.what());
    }
    return result;
}

std::vector<std::string> ObxManager::getAllDocumentExternalIds() {
    std::vector<std::string> result;
    pImpl_->ensureInitialized();
    try {
        // Property query: only fetch _id field (property 3), no full document content
        OBX_store* store = pImpl_->store->cPtr();
        OBX_query_builder* qb = obx_query_builder(store, ecovector::Document::_OBX_MetaInfo::entityId());
        OBX_query* query = obx_query(qb);
        obx_qb_close(qb);

        OBX_query_prop* prop = obx_query_prop(query, 3);  // _id property
        OBX_string_array* strings = obx_query_prop_find_strings(prop, nullptr);

        if (strings) {
            result.reserve(strings->count);
            for (size_t i = 0; i < strings->count; i++) {
                if (strings->items[i]) result.emplace_back(strings->items[i]);
            }
            obx_string_array_free(strings);
        }

        obx_query_prop_close(prop);
        obx_query_close(query);
        LOGI("Retrieved %zu document external IDs (property query)", result.size());
    } catch (const std::exception& e) {
        LOGE("Error in getAllDocumentExternalIds: %s", e.what());
    }
    return result;
}

uint32_t ObxManager::getDocumentCount() {
    pImpl_->ensureInitialized();
    return static_cast<uint32_t>(pImpl_->boxDocument->count());
}

uint32_t ObxManager::getChunkCount() {
    pImpl_->ensureInitialized();
    return static_cast<uint32_t>(pImpl_->boxChunk->count());
}

std::optional<DocData> ObxManager::getDocumentById(uint64_t id) {
    pImpl_->ensureInitialized();
    try {
        auto doc = pImpl_->boxDocument->get(id);
        if (!doc) return std::nullopt;
        return fromObxDoc(*doc);
    } catch (const std::exception& e) {
        LOGE("Error in getDocumentById(%llu): %s", (unsigned long long)id, e.what());
        return std::nullopt;
    }
}

std::vector<ChunkData> ObxManager::getAllChunks(bool excludeVectors) {
    std::vector<ChunkData> result;
    pImpl_->ensureInitialized();
    try {
        auto all = pImpl_->boxChunk->getAll();
        result.reserve(all.size());
        for (const auto& chunk : all)
            result.push_back(fromObxChunk(*chunk, !excludeVectors));
        LOGI("Retrieved %zu chunks", result.size());
    } catch (const std::exception& e) {
        LOGE("Error in getAllChunks: %s", e.what());
    }
    return result;
}

void ObxManager::forEachChunkBatch(size_t batchSize, bool excludeVectors,
                                    const std::function<void(const std::vector<ChunkData>&)>& callback) {
    pImpl_->ensureInitialized();
    auto query = pImpl_->boxChunk->query().build();
    uint64_t offset = 0;
    while (true) {
        auto batch = query.offset(offset).limit(batchSize).find();
        if (batch.empty()) break;
        std::vector<ChunkData> chunks;
        chunks.reserve(batch.size());
        for (const auto& chunk : batch)
            chunks.push_back(fromObxChunk(chunk, !excludeVectors));
        callback(chunks);
        offset += batch.size();
    }
}

std::vector<ChunkData> ObxManager::getSampledChunks(size_t sampleSize, bool excludeVectors) {
    std::vector<ChunkData> result;
    pImpl_->ensureInitialized();

    try {
        // 1. Get total count
        uint32_t totalCount = static_cast<uint32_t>(pImpl_->boxChunk->count());
        if (totalCount == 0) {
            LOGW("No chunks in database for sampling");
            return result;
        }

        // 2. Adjust sample size if needed
        if (sampleSize >= totalCount) {
            LOGW("Sample size %zu >= total count %u, returning all chunks", sampleSize, totalCount);
            return getAllChunks(excludeVectors);
        }

        // 3. Load all chunks (metadata only, no vectors) for sampling
        auto allChunks = pImpl_->boxChunk->getAll();
        if (allChunks.size() != totalCount) {
            LOGW("Chunk count mismatch: count()=%u, getAll()=%zu", totalCount, allChunks.size());
        }

        // 4. Stride-based sampling (uniform distribution)
        double stride = static_cast<double>(allChunks.size()) / static_cast<double>(sampleSize);
        std::vector<uint64_t> sampledIds;
        sampledIds.reserve(sampleSize);

        for (size_t i = 0; i < sampleSize; i++) {
            size_t idx = static_cast<size_t>(i * stride);
            if (idx < allChunks.size()) {
                sampledIds.push_back(allChunks[idx]->id);
            }
        }

        LOGI("Sampled %zu chunks (stride=%.2f) from %zu total", sampledIds.size(), stride, allChunks.size());

        // 5. Load sampled chunks with vectors if needed
        result = getChunksByIds(sampledIds, excludeVectors, false);

    } catch (const std::exception& e) {
        LOGE("Error in getSampledChunks: %s", e.what());
    }

    return result;
}

std::optional<ChunkData> ObxManager::getChunkById(uint64_t id, bool excludeVector) {
    pImpl_->ensureInitialized();
    try {
        auto obxChunk = pImpl_->boxChunk->get(id);
        if (!obxChunk) return std::nullopt;
        return fromObxChunk(*obxChunk, !excludeVector);
    } catch (const std::exception& e) {
        LOGE("Error in getChunkById(%llu): %s", (unsigned long long)id, e.what());
        return std::nullopt;
    }
}

std::vector<ChunkData> ObxManager::getChunksByIds(const std::vector<uint64_t>& ids,
                                                    bool excludeVectors,
                                                    bool excludeTokenIds,
                                                    bool excludeKiwiTokens) {
    std::vector<ChunkData> result;
    pImpl_->ensureInitialized();
    if (ids.empty()) return result;
    try {
        auto obxChunks = pImpl_->boxChunk->get(ids);
        result.reserve(obxChunks.size());
        for (const auto& c : obxChunks) {
            if (c) result.push_back(fromObxChunk(*c, !excludeVectors, !excludeTokenIds, !excludeKiwiTokens));
        }
    } catch (const std::exception& e) {
        LOGE("Error in getChunksByIds: %s", e.what());
    }
    return result;
}

// ============================================================================
// Write Operations
// ============================================================================

std::vector<std::string> ObxManager::removeOrphanDocuments() {
    pImpl_->ensureInitialized();
    std::vector<std::string> removedExternalIds;

    try {
        // Optimized O(N) algorithm: collect all document_ids from chunks first
        LOGD("Collecting document IDs from chunks...");
        auto chunks = pImpl_->boxChunk->getAll();
        std::unordered_set<uint64_t> docIdsWithChunks;
        for (const auto& chunk : chunks) {
            docIdsWithChunks.insert(chunk->document_id);
        }
        LOGD("Found %zu documents with chunks (from %zu total chunks)",
             docIdsWithChunks.size(), chunks.size());

        // Find orphan documents (documents without any chunks)
        auto docs = pImpl_->boxDocument->getAll();
        std::vector<uint64_t> orphanIds;

        for (const auto& doc : docs) {
            if (docIdsWithChunks.find(doc->id) == docIdsWithChunks.end()) {
                orphanIds.push_back(doc->id);
                removedExternalIds.push_back(doc->_id);
            }
        }

        if (!orphanIds.empty()) {
            pImpl_->boxDocument->remove(orphanIds);
            LOGI("Removed %zu orphan documents (no chunks)", orphanIds.size());
        } else {
            LOGD("No orphan documents found");
        }

    } catch (const std::exception& e) {
        LOGE("Error in removeOrphanDocuments: %s", e.what());
    }

    return removedExternalIds;
}

void ObxManager::removeAll() {
    pImpl_->ensureInitialized();

    try {
        LOGI("Removing all data...");

        // Use transaction for atomicity
        obx::Transaction tx = pImpl_->store->txWrite();

        pImpl_->boxDocument->removeAll();
        pImpl_->boxChunk->removeAll();

        tx.success();
        LOGI("All ObjectBox data removed");

        // BM25 인덱스 제거 (인메모리 + 디스크 파일)
        if (pImpl_->bm25Index) {
            pImpl_->bm25Index->removeIndex();
        }

        // EcoVectorIndex 인덱스 제거
        if (pImpl_->ecoVectorIndex) {
            pImpl_->ecoVectorIndex->removeIndexes();
        }

        LOGI("All data and indexes removed successfully");

    } catch (const std::exception& e) {
        LOGE("Error in removeAll: %s", e.what());
    }
}

void ObxManager::removeAllChunks() {
    pImpl_->ensureInitialized();
    try {
        obx::Transaction tx = pImpl_->store->txWrite();
        pImpl_->boxChunk->removeAll();
        tx.success();
        LOGI("removeAllChunks: all chunks removed");
    } catch (const std::exception& e) {
        LOGE("Error in removeAllChunks: %s", e.what());
    }
}

void ObxManager::forEachChunkWithoutVector(size_t batchSize,
        const std::function<bool(std::vector<ChunkData>&)>& callback) {
    pImpl_->ensureInitialized();
    auto query = pImpl_->boxChunk->query().build();
    uint64_t offset = 0;
    while (true) {
        auto batch = query.offset(offset).limit(batchSize).find();
        if (batch.empty()) break;
        std::vector<ChunkData> filtered;
        for (const auto& chunk : batch) {
            auto cd = fromObxChunk(chunk, /*includeVectors=*/true);
            if (cd.vector.empty()) {
                filtered.push_back(std::move(cd));
            }
        }
        offset += batch.size();
        if (!filtered.empty()) {
            if (!callback(filtered)) break;
        }
    }
}

std::vector<uint64_t> ObxManager::insertAllDocuments(const std::vector<DocData>& documents) {
    std::vector<uint64_t> insertedIds;
    if (documents.empty()) return insertedIds;
    pImpl_->ensureInitialized();

    try {
        insertedIds.reserve(documents.size());
        for (size_t base = 0; base < documents.size(); base += WRITE_BATCH_SIZE) {
            size_t end = std::min(base + WRITE_BATCH_SIZE, documents.size());
            obx::Transaction tx = pImpl_->store->txWrite();
            for (size_t i = base; i < end; ++i) {
                auto obxDoc = toObxDoc(documents[i], /*forInsert=*/true);
                insertedIds.push_back(pImpl_->boxDocument->put(obxDoc));
            }
            tx.success();
        }
        LOGI("Inserted %zu documents", insertedIds.size());
    } catch (const std::exception& e) {
        LOGE("Error in insertAllDocuments: %s", e.what());
        insertedIds.clear();
    }
    return insertedIds;
}

std::optional<DocData> ObxManager::getDocumentByExternalId(const std::string& externalId) {
    pImpl_->ensureInitialized();
    try {
        auto query = pImpl_->boxDocument->query(Document_::_id.equals(externalId)).build();
        auto results = query.find();
        if (!results.empty()) return fromObxDoc(results[0]);
    } catch (const std::exception& e) {
        LOGE("Error in getDocumentByExternalId: %s", e.what());
    }
    return std::nullopt;
}

std::optional<ChunkData> ObxManager::getFirstChunkByDocumentId(uint64_t docId, bool excludeVector) {
    pImpl_->ensureInitialized();
    try {
        auto qb = pImpl_->boxChunk->query();
        obx_qb_equals_int(qb.cPtr(), Chunk_::document_id.id(), static_cast<int64_t>(docId));
        obx_qb_order(qb.cPtr(), Chunk_::chunk_index.id(), 0);
        auto query = qb.build();
        query.offset(0).limit(1);
        auto results = query.find();
        if (!results.empty()) {
            return fromObxChunk(results[0], !excludeVector);
        }
    } catch (const std::exception& e) {
        LOGE("Error in getFirstChunkByDocumentId(%llu): %s",
             (unsigned long long)docId, e.what());
    }
    return std::nullopt;
}

std::vector<uint64_t> ObxManager::insertAllChunks(const std::vector<ChunkData>& chunks) {
    std::vector<uint64_t> insertedIds;
    if (chunks.empty()) return insertedIds;
    pImpl_->ensureInitialized();

    try {
        insertedIds.reserve(chunks.size());
        for (size_t base = 0; base < chunks.size(); base += WRITE_BATCH_SIZE) {
            size_t end = std::min(base + WRITE_BATCH_SIZE, chunks.size());
            obx::Transaction tx = pImpl_->store->txWrite();
            for (size_t i = base; i < end; ++i) {
                auto obxChunk = toObxChunk(chunks[i], /*forInsert=*/true);
                insertedIds.push_back(pImpl_->boxChunk->put(obxChunk));
            }
            tx.success();
        }
        LOGI("Inserted %zu chunks", insertedIds.size());
    } catch (const std::exception& e) {
        LOGE("Error in insertAllChunks: %s", e.what());
        insertedIds.clear();
    }
    return insertedIds;
}

// insertAllQueries removed — queries now go to BenchmarkObxManager

// ============================================================================
// Update Operations (upsert by existing ID)
// ============================================================================

bool ObxManager::updateAllDocuments(const std::vector<DocData>& documents) {
    if (documents.empty()) return true;
    pImpl_->ensureInitialized();
    try {
        size_t updated = 0;
        for (size_t base = 0; base < documents.size(); base += WRITE_BATCH_SIZE) {
            size_t end = std::min(base + WRITE_BATCH_SIZE, documents.size());
            obx::Transaction tx = pImpl_->store->txWrite();
            for (size_t i = base; i < end; ++i) {
                if (documents[i].id == 0) {
                    LOGW("updateAllDocuments: skipping entry with id=0");
                    continue;
                }
                auto obxDoc = toObxDoc(documents[i], /*forInsert=*/false);
                pImpl_->boxDocument->put(obxDoc);
                updated++;
            }
            tx.success();
        }
        LOGI("Updated %zu documents", updated);
        return true;
    } catch (const std::exception& e) {
        LOGE("Error in updateAllDocuments: %s", e.what());
        return false;
    }
}

bool ObxManager::updateAllChunks(const std::vector<ChunkData>& chunks) {
    if (chunks.empty()) return true;
    pImpl_->ensureInitialized();
    try {
        size_t updated = 0;
        for (size_t base = 0; base < chunks.size(); base += WRITE_BATCH_SIZE) {
            size_t end = std::min(base + WRITE_BATCH_SIZE, chunks.size());
            obx::Transaction tx = pImpl_->store->txWrite();
            for (size_t i = base; i < end; ++i) {
                if (chunks[i].id == 0) {
                    LOGW("updateAllChunks: skipping entry with id=0");
                    continue;
                }
                auto obxChunk = toObxChunk(chunks[i], /*forInsert=*/false);
                pImpl_->boxChunk->put(obxChunk);
                updated++;
            }
            tx.success();
        }
        LOGI("Updated %zu chunks", updated);
        return true;
    } catch (const std::exception& e) {
        LOGE("Error in updateAllChunks: %s", e.what());
        return false;
    }
}

// updateAllQueries removed — queries now go to BenchmarkObxManager

// ============================================================================
// Delete by ID
// ============================================================================

bool ObxManager::removeDocumentById(uint64_t id) {
    pImpl_->ensureInitialized();
    try {
        pImpl_->boxDocument->remove(id);
        return true;
    } catch (const std::exception& e) {
        LOGE("Error in removeDocumentById(%llu): %s", (unsigned long long)id, e.what());
        return false;
    }
}

bool ObxManager::removeChunkById(uint64_t id) {
    pImpl_->ensureInitialized();
    try {
        pImpl_->boxChunk->remove(id);
        return true;
    } catch (const std::exception& e) {
        LOGE("Error in removeChunkById(%llu): %s", (unsigned long long)id, e.what());
        return false;
    }
}

// removeQueryById removed — queries now managed by BenchmarkObxManager

// ============================================================================
// Batch Delete by IDs
// ============================================================================

void ObxManager::removeDocumentsByIds(const std::vector<uint64_t>& ids) {
    if (ids.empty()) return;
    pImpl_->ensureInitialized();
    try {
        pImpl_->boxDocument->remove(ids);
    } catch (const std::exception& e) {
        LOGE("Error in removeDocumentsByIds: %s", e.what());
    }
}

void ObxManager::removeChunksByIds(const std::vector<uint64_t>& ids) {
    if (ids.empty()) return;
    pImpl_->ensureInitialized();
    try {
        pImpl_->boxChunk->remove(ids);
    } catch (const std::exception& e) {
        LOGE("Error in removeChunksByIds: %s", e.what());
    }
}

// removeQueriesByIds removed — queries now managed by BenchmarkObxManager

// ============================================================================
// Single-item Create
// ============================================================================

uint64_t ObxManager::insertDocument(const DocData& doc) {
    pImpl_->ensureInitialized();
    try {
        auto obxDoc = toObxDoc(doc, /*forInsert=*/true);
        return pImpl_->boxDocument->put(obxDoc);
    } catch (const std::exception& e) {
        LOGE("Error in insertDocument: %s", e.what());
        return 0;
    }
}

uint64_t ObxManager::insertChunk(const ChunkData& chunk) {
    pImpl_->ensureInitialized();
    try {
        auto obxChunk = toObxChunk(chunk, /*forInsert=*/true);
        return pImpl_->boxChunk->put(obxChunk);
    } catch (const std::exception& e) {
        LOGE("Error in insertChunk: %s", e.what());
        return 0;
    }
}

// insertQuery removed — queries now go to BenchmarkObxManager

// ============================================================================
// Single-item Update (id must be > 0)
// ============================================================================

bool ObxManager::updateDocument(const DocData& doc) {
    if (doc.id == 0) { LOGE("updateDocument: id must be > 0"); return false; }
    pImpl_->ensureInitialized();
    try {
        auto obxDoc = toObxDoc(doc, /*forInsert=*/false);
        pImpl_->boxDocument->put(obxDoc);
        return true;
    } catch (const std::exception& e) {
        LOGE("Error in updateDocument: %s", e.what());
        return false;
    }
}

bool ObxManager::updateChunk(const ChunkData& chunk) {
    if (chunk.id == 0) { LOGE("updateChunk: id must be > 0"); return false; }
    pImpl_->ensureInitialized();
    try {
        auto obxChunk = toObxChunk(chunk, /*forInsert=*/false);
        pImpl_->boxChunk->put(obxChunk);
        return true;
    } catch (const std::exception& e) {
        LOGE("Error in updateChunk: %s", e.what());
        return false;
    }
}

// updateQuery removed — queries now managed by BenchmarkObxManager

bool ObxManager::reTokenizeAllKiwiTokens() {
    pImpl_->ensureInitialized();

    if (!pImpl_->kiwiTokenizer) {
        LOGE("KiwiTokenizer not initialized, cannot re-tokenize");
        return false;
    }

    try {
        // --- Step 1: Re-tokenize all chunks in batches (preserve embeddings and IDs) ---
        auto obxChunks = pImpl_->boxChunk->getAll();
        LOGI("Re-tokenizing %zu chunks with updated Kiwi dictionary", obxChunks.size());

        // 진단용 코퍼스 키워드
        const std::vector<std::string> debugCorpus = {
            "홍콩반점0410", "베스킨라빈스", "배달의민족", "삼성전자",
            "하이패스", "셀프세차", "LG U+", "노래연습장"
        };

        for (size_t base = 0; base < obxChunks.size(); base += WRITE_BATCH_SIZE) {
            size_t end = std::min(base + WRITE_BATCH_SIZE, obxChunks.size());
            obx::Transaction tx = pImpl_->store->txWrite();
            for (size_t i = base; i < end; ++i) {
                auto& chunk = obxChunks[i];
                auto morphemes = pImpl_->kiwiTokenizer->tokenizeForIndexing(chunk->content);
                chunk->kiwi_tokens = tokenIdsToBinary(hashMorphemes(morphemes));
                pImpl_->boxChunk->put(*chunk);

                // 진단 로그: 특정 문서 토큰화 결과
                for (const auto& dc : debugCorpus) {
                    if (chunk->content.find(dc) != std::string::npos) {
                        std::string tokStr;
                        for (size_t j = 0; j < morphemes.size() && j < 20; ++j) {
                            if (!tokStr.empty()) tokStr += ", ";
                            tokStr += morphemes[j];
                        }
                        if (morphemes.size() > 20) tokStr += "...";
                        LOGD("[DEBUG_TOKEN] C[%llu]: \"%s\" -> [%s] (%zu tokens)",
                             (unsigned long long)chunk->id,
                             chunk->content.substr(0, 80).c_str(),
                             tokStr.c_str(), morphemes.size());
                        break;
                    }
                }
            }
            tx.success();
        }
        LOGI("Chunks re-tokenized: %zu", obxChunks.size());

        // Query re-tokenization moved to BenchmarkObxManager (separate DB)
        // BM25 index rebuild is now the caller's responsibility (EcoVectorStore::reTokenizeAll)
        return true;

    } catch (const std::exception& e) {
        LOGE("Error in reTokenizeAllKiwiTokens: %s", e.what());
        return false;
    }
}

// ============================================================================
// Kiwi Tokenize (for diagnostics)
// ============================================================================

std::vector<std::string> ObxManager::kiwiTokenize(const std::string& text) const {
    if (!pImpl_->kiwiTokenizer) return {};
    return pImpl_->kiwiTokenizer->tokenize(text);
}

std::vector<std::string> ObxManager::kiwiTokenizeForIndexing(const std::string& text) const {
    if (!pImpl_->kiwiTokenizer) return {};
    return pImpl_->kiwiTokenizer->tokenizeForIndexing(text);
}

// ============================================================================
// Vector Search Operations
// ============================================================================

std::vector<ObxManager::VectorSearchResult> ObxManager::vectorSearch(
    const float* queryVector, size_t dimensions,
    size_t maxResultCount, uint32_t topK) {

    pImpl_->ensureInitialized();
    std::vector<VectorSearchResult> results;

    try {
        OBX_store* store = pImpl_->store->cPtr();

        OBX_query_builder* qb = obx_query_builder(store, ecovector::Chunk::_OBX_MetaInfo::entityId());
        if (!qb) {
            LOGE("vectorSearch: failed to create query builder");
            return results;
        }

        obx_qb_cond cond = obx_qb_nearest_neighbors_f32(qb, 4, queryVector, maxResultCount);
        if (cond == 0) {
            LOGE("vectorSearch: nearest_neighbors condition failed");
            obx_qb_close(qb);
            return results;
        }

        OBX_query* query = obx_query(qb);
        obx_qb_close(qb);
        if (!query) {
            LOGE("vectorSearch: failed to build query");
            return results;
        }

        OBX_id_score_array* idScores = obx_query_find_ids_with_scores(query);
        obx_query_close(query);

        if (!idScores) {
            LOGE("vectorSearch: find_ids_with_scores returned null");
            return results;
        }

        size_t count = std::min(static_cast<size_t>(topK), idScores->count);
        results.reserve(count);
        for (size_t i = 0; i < count; i++) {
            results.push_back({
                static_cast<uint64_t>(idScores->ids_scores[i].id),
                static_cast<float>(idScores->ids_scores[i].score)
            });
        }

        obx_id_score_array_free(idScores);

    } catch (const std::exception& e) {
        LOGE("vectorSearch exception: %s", e.what());
    }

    return results;
}

// ============================================================================
// Metadata Filter
// ============================================================================

std::unordered_set<uint64_t> ObxManager::resolveFilter(const std::string& filterJson) {
    pImpl_->ensureInitialized();

    auto filter = nlohmann::json::parse(filterJson);  // throws on invalid JSON
    auto qb = pImpl_->boxChunk->query();

    // source_type: exact match or OR-combined equals (Short type doesn't support IN)
    if (filter.contains("source_type")) {
        const auto& stFilter = filter["source_type"];
        if (stFilter.is_number()) {
            int64_t val = stFilter.get<int64_t>();
            obx_qb_equals_int(qb.cPtr(), Chunk_::source_type.id(), val);
        } else if (stFilter.is_object() && stFilter.contains("in")) {
            auto vals = stFilter["in"].get<std::vector<int64_t>>();
            if (vals.size() == 1) {
                obx_qb_equals_int(qb.cPtr(), Chunk_::source_type.id(), vals[0]);
            } else {
                // Combine multiple equals with OR (obx_qb_any)
                std::vector<obx_qb_cond> conds;
                conds.reserve(vals.size());
                for (auto v : vals) {
                    conds.push_back(obx_qb_equals_int(qb.cPtr(), Chunk_::source_type.id(), v));
                }
                obx_qb_any(qb.cPtr(), conds.data(), conds.size());
            }
        }
    }

    // sender: exact match (string)
    if (filter.contains("sender")) {
        std::string val = filter["sender"].get<std::string>();
        obx_qb_equals_string(qb.cPtr(), Chunk_::sender.id(), val.c_str(), true);
    }

    // document_id: exact or set match
    if (filter.contains("document_id")) {
        const auto& docFilter = filter["document_id"];
        if (docFilter.is_number()) {
            int64_t val = docFilter.get<int64_t>();
            obx_qb_equals_int(qb.cPtr(), Chunk_::document_id.id(), val);
        } else if (docFilter.is_object() && docFilter.contains("in")) {
            auto ids = docFilter["in"].get<std::vector<int64_t>>();
            obx_qb_in_int64s(qb.cPtr(), Chunk_::document_id.id(), ids.data(), ids.size());
        }
    }

    // created_at: exact or range match
    if (filter.contains("created_at")) {
        const auto& caFilter = filter["created_at"];
        if (caFilter.is_number()) {
            int64_t val = caFilter.get<int64_t>();
            obx_qb_equals_int(qb.cPtr(), Chunk_::created_at.id(), val);
        } else if (caFilter.is_object()) {
            if (caFilter.contains("gte")) {
                obx_qb_greater_or_equal_int(qb.cPtr(), Chunk_::created_at.id(),
                                             caFilter["gte"].get<int64_t>());
            }
            if (caFilter.contains("lte")) {
                obx_qb_less_or_equal_int(qb.cPtr(), Chunk_::created_at.id(),
                                          caFilter["lte"].get<int64_t>());
            }
            if (caFilter.contains("gt")) {
                obx_qb_greater_than_int(qb.cPtr(), Chunk_::created_at.id(),
                                         caFilter["gt"].get<int64_t>());
            }
            if (caFilter.contains("lt")) {
                obx_qb_less_than_int(qb.cPtr(), Chunk_::created_at.id(),
                                      caFilter["lt"].get<int64_t>());
            }
        }
    }

    auto query = qb.build();
    auto ids = query.findIds();

    std::unordered_set<uint64_t> result;
    result.reserve(ids.size());
    for (auto id : ids) {
        result.insert(id);
    }

    LOGD("resolveFilter: %zu chunks matched", result.size());
    return result;
}

int ObxManager::bulkUpdateCreatedAt(const std::unordered_map<std::string, int64_t>& docDateMap) {
    pImpl_->ensureInitialized();
    int updated = 0;
    try {
        // Process in batches by document external ID
        for (auto& [docExtId, createdAtMs] : docDateMap) {
            if (createdAtMs == 0) continue;

            // Find all chunks for this document
            auto qb = pImpl_->boxChunk->query(
                Chunk_::document_external_id.equals(docExtId));
            auto query = qb.build();
            auto chunks = query.find();

            if (chunks.empty()) continue;

            // Batch update within a transaction
            obx::Transaction tx = pImpl_->store->txWrite();
            for (auto& chunk : chunks) {
                chunk.created_at = createdAtMs;
                pImpl_->boxChunk->put(chunk);
                updated++;
            }
            tx.success();
        }
        LOGI("bulkUpdateCreatedAt: updated %d chunks across %zu documents", updated, docDateMap.size());
    } catch (const std::exception& e) {
        LOGE("Error in bulkUpdateCreatedAt: %s", e.what());
    }
    return updated;
}

int ObxManager::bulkUpdateSourceType(const std::unordered_map<std::string, int16_t>& docSourceTypeMap) {
    pImpl_->ensureInitialized();
    int updated = 0;
    size_t processed = 0;
    size_t total = docSourceTypeMap.size();
    try {
        for (auto& [docExtId, newType] : docSourceTypeMap) {
            processed++;
            if (processed % 1000 == 0) {
                LOGI("bulkUpdateSourceType: %zu/%zu docs scanned, %d chunks updated", processed, total, updated);
            }

            auto qb = pImpl_->boxChunk->query(
                Chunk_::document_external_id.equals(docExtId));
            auto query = qb.build();
            auto chunks = query.find();

            if (chunks.empty()) continue;

            bool needsUpdate = false;
            for (const auto& chunk : chunks) {
                if (chunk.source_type != newType) {
                    needsUpdate = true;
                    break;
                }
            }
            if (!needsUpdate) continue;

            obx::Transaction tx = pImpl_->store->txWrite();
            for (auto& chunk : chunks) {
                if (chunk.source_type != newType) {
                    chunk.source_type = newType;
                    pImpl_->boxChunk->put(chunk);
                    updated++;
                }
            }
            tx.success();
        }
        LOGI("bulkUpdateSourceType: done — %d chunks updated across %zu documents", updated, total);
    } catch (const std::exception& e) {
        LOGE("Error in bulkUpdateSourceType: %s", e.what());
    }
    return updated;
}

int ObxManager::bulkUpdateSourceTypeFromPrefix() {
    pImpl_->ensureInitialized();
    int updated = 0;
    try {
        auto allChunks = pImpl_->boxChunk->getAll();
        LOGI("bulkUpdateSourceTypeFromPrefix: scanning %zu chunks", allChunks.size());

        // Batch by inferred source_type
        constexpr int TX_BATCH = 5000;
        int pending = 0;
        obx::Transaction tx = pImpl_->store->txWrite();

        for (auto& chunk : allChunks) {
            const auto& extId = chunk->document_external_id;
            int16_t newType = -1;
            if (extId.rfind("call_", 0) == 0)          newType = 0; // CALL
            else if (extId.rfind("sms_", 0) == 0)       newType = 1; // SMS
            else if (extId.rfind("mms_", 0) == 0)       newType = 2; // MMS
            else if (extId.rfind("img_", 0) == 0)       newType = 3; // IMAGE
            else if (extId.rfind("document_", 0) == 0)  newType = 4; // FILE
            else continue;

            if (chunk->source_type == newType) continue; // already correct

            chunk->source_type = newType;
            pImpl_->boxChunk->put(*chunk);
            updated++;
            pending++;

            if (pending >= TX_BATCH) {
                tx.success();
                tx = pImpl_->store->txWrite();
                pending = 0;
            }
        }
        if (pending > 0) tx.success();

        LOGI("bulkUpdateSourceTypeFromPrefix: updated %d chunks", updated);
    } catch (const std::exception& e) {
        LOGE("Error in bulkUpdateSourceTypeFromPrefix: %s", e.what());
    }
    return updated;
}

bool ObxManager::isSourceTypePatched() {
    pImpl_->ensureInitialized();
    try {
        auto allChunks = pImpl_->boxChunk->getAll();
        if (allChunks.empty()) return true;
        // Check both call/sms and document prefixes
        bool callSmsOk = false, documentOk = false;
        bool hasCallSms = false, hasDocument = false;
        for (const auto& chunk : allChunks) {
            const auto& extId = chunk->document_external_id;
            if (!hasCallSms && (extId.rfind("call_", 0) == 0 || extId.rfind("sms_", 0) == 0)) {
                hasCallSms = true;
                callSmsOk = chunk->source_type >= 0;
            }
            if (!hasDocument && extId.rfind("document_", 0) == 0) {
                hasDocument = true;
                documentOk = chunk->source_type >= 0;
            }
            if (hasCallSms && hasDocument) break;
        }
        return (!hasCallSms || callSmsOk) && (!hasDocument || documentOk);
    } catch (...) {}
    return false;
}

} // namespace ecovector
