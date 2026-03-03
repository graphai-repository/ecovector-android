#include "EcoVectorStore.h"
#include "ObxManager.h"
#include "../SearchUtils.h"
#include "Tokenizer.h"
#include "OnnxRuntime.h"
#include "TokenAwareChunker.h"
#include "KiwiTokenizer.h"
#include "KiwiHashUtil.h"
#include "Embedder.h"
#include "VectorRetriever.h"
#include "ObxVectorRetriever.h"
#include "BM25Retriever.h"
#include "EnsembleRetriever.h"
#include "QueryBundle.h"
#include "EcoVectorIndex.h"
#include "BM25Index.h"
#include <json.hpp>
#include <android/log.h>
#include <algorithm>
#include <sqlite3.h>
#include <unordered_set>
#include <unordered_map>
#include <regex>
#include <malloc.h>       // mallopt, M_PURGE
#include <fstream>        // /proc/self/status
#include <sstream>
#include <chrono>
#include <future>

#define LOG_TAG "EcoVectorStore"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace ecovector {

static const size_t EMBED_BATCH_SIZE = 8;   // ONNX embedding batch (모바일 메모리 고려)
static const size_t DB_BATCH_SIZE = 32;     // DB insert batch (트랜잭션 묶기)

// 네이티브 메모리 RSS(KB)를 /proc/self/status에서 읽기
static long getNativeRssKb() {
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            long kb = 0;
            sscanf(line.c_str(), "VmRSS: %ld", &kb);
            return kb;
        }
    }
    return -1;
}

// freed 페이지를 OS에 즉시 반환
static void purgeNativeMemory() {
#if __ANDROID_API__ >= 28
    mallopt(M_PURGE, 0);
#endif
}

EcoVectorStore::EcoVectorStore() = default;
EcoVectorStore::~EcoVectorStore() { close(); }

// ============================================================================
// Lifecycle
// ============================================================================

bool EcoVectorStore::initialize(const std::string& dbPath,
                                 const std::string& tokenizerPath,
                                 const std::string& modelPath,
                                 const std::string& kiwiModelDir) {
    try {
        dbPath_ = dbPath;

        // 1. Tokenizer (used for token ID extraction)
        tokenizer_ = std::make_unique<Tokenizer>();
        if (!tokenizer_->load(tokenizerPath)) {
            LOGE("Failed to load tokenizer: %s", tokenizerPath.c_str());
            return false;
        }
        LOGI("Tokenizer loaded: %s", tokenizerPath.c_str());

        // 2. ONNX Runtime (embedding model + internal tokenizer)
        onnxRuntime_ = std::make_unique<OnnxRuntime>();
        if (!onnxRuntime_->loadTokenizer(tokenizerPath)) {
            LOGE("Failed to load ONNX tokenizer");
            return false;
        }
        if (!onnxRuntime_->load(modelPath)) {
            LOGE("Failed to load ONNX model: %s", modelPath.c_str());
            return false;
        }
        LOGI("ONNX Runtime loaded: %s", modelPath.c_str());

        // 3. Kiwi tokenizer (for BM25 kiwiTokens)
        if (!kiwiModelDir.empty()) {
            kiwiTokenizer_ = std::make_unique<KiwiTokenizer>();
            if (kiwiTokenizer_->load(kiwiModelDir)) {
                LOGI("Kiwi tokenizer loaded for pipeline");
            } else {
                LOGE("Failed to load Kiwi tokenizer, BM25 will be unavailable");
                kiwiTokenizer_.reset();
            }
        }

        // 4. ObxManager (ObjectBox DB + CRUD)
        obxManager_ = std::make_unique<ObxManager>(dbPath);
        if (!obxManager_->initialize(kiwiModelDir)) {
            LOGE("Failed to initialize ObxManager");
            obxManager_.reset();
            return false;
        }
        LOGI("ObxManager initialized: %s", dbPath.c_str());

        // 5. EcoVectorIndex (owned by EcoVectorStore)
        ecoVectorIndex_ = std::make_unique<EcoVectorIndex>(dbPath);
        if (ecoVectorIndex_->isIndexReady()) {
            LOGI("EcoVectorIndex is ready");
        } else {
            LOGI("EcoVectorIndex not found - call buildVectorIndex() to build");
        }

        // 6. BM25Index (owned by EcoVectorStore, shared with ObxManager)
        bm25Index_ = std::make_unique<BM25Index>(dbPath);
        if (kiwiTokenizer_) {
            bm25Index_->setTokenizer(kiwiTokenizer_.get());
        }
        if (bm25Index_->loadIndex()) {
            LOGI("BM25 index loaded from disk");
        } else {
            LOGI("BM25 index not found - call buildBM25Index() to build");
        }
        obxManager_->setKiwiTokenizer(kiwiTokenizer_.get());

        // 7. Share tokenizer with ObxManager for reranking
        obxManager_->setTokenizer(tokenizer_.get());

        // 7. Embedder (wraps OnnxRuntime + Tokenizer for retriever use)
        embedder_ = std::make_unique<Embedder>(onnxRuntime_.get(), tokenizer_.get());
        LOGI("Embedder created");

        return true;
    } catch (const std::exception& e) {
        LOGE("Initialize failed: %s", e.what());
        return false;
    }
}

void EcoVectorStore::close() {
    // Destroy owned retrievers in reverse order (ensemble before sub-retrievers)
    while (!ownedRetrievers_.empty()) {
        ownedRetrievers_.pop_back();  // unique_ptr auto-destructs, last (newest) first
    }

    embedder_.reset();
    ecoVectorIndex_.reset();
    bm25Index_.reset();
    obxManager_.reset();
    onnxRuntime_.reset();
    tokenizer_.reset();
    kiwiTokenizer_.reset();
}

// ============================================================================
// Internal helpers
// ============================================================================

std::vector<std::string> EcoVectorStore::chunkText(const std::string& text) {
    return chunkText(text, -1);
}

std::vector<std::string> EcoVectorStore::chunkText(const std::string& text, int16_t /*sourceType*/) {
    // Unified: TokenAwareChunker for all source types (512 tokens, 216 overlap)
    TokenAwareChunker chunker(tokenizer_.get());
    auto chunks = chunker.chunk(text);

    // prefix가 있으면 각 청크 앞에 붙임 (Call summary 등)
    if (!chunkPrefix_.empty()) {
        for (auto& chunk : chunks) {
            chunk = chunkPrefix_ + "\n" + chunk;
        }
    }
    return chunks;
}

std::vector<std::string> EcoVectorStore::chunkTextWithParams(const std::string& text,
                                                              int maxTokens, int overlapTokens) {
    TokenAwareChunker chunker(tokenizer_.get(), maxTokens, overlapTokens);
    return chunker.chunk(text);
}

std::vector<float> EcoVectorStore::embedText(const std::string& text) {
    return onnxRuntime_->embed(text);
}

std::vector<int32_t> EcoVectorStore::tokenizeText(const std::string& text) {
    return tokenizer_->encode(text);
}


std::string EcoVectorStore::utf8Truncate(const std::string& str, size_t maxBytes) {
    if (str.size() <= maxBytes) return str;
    size_t pos = maxBytes;
    while (pos > 0 && (static_cast<unsigned char>(str[pos]) & 0xC0) == 0x80) {
        --pos;
    }
    return str.substr(0, pos);
}

// ============================================================================
// Document Management
// ============================================================================

int64_t EcoVectorStore::addDocument(const std::string& text, const std::string& title,
                                     const std::string& externalId) {
    if (!obxManager_ || !tokenizer_ || !onnxRuntime_) return -1;

    try {
        // 1. Save document
        DocData doc;
        doc.id = 0;
        doc.content = text;
        doc.externalId = externalId;

        auto docIds = obxManager_->insertAllDocuments({doc});
        if (docIds.empty()) return -1;
        int64_t docId = static_cast<int64_t>(docIds[0]);

        // 2. Chunk text (strategy-dependent)
        auto chunks = chunkText(text);
        if (chunks.empty()) return docId;

        // 3. Process chunks: embed in EMBED_BATCH_SIZE, accumulate, save in DB_BATCH_SIZE
        std::vector<ChunkData> pendingInserts;
        pendingInserts.reserve(chunks.size());

        for (size_t i = 0; i < chunks.size(); i += EMBED_BATCH_SIZE) {
            size_t batchEnd = std::min(i + EMBED_BATCH_SIZE, chunks.size());
            std::vector<std::string> batchTexts(chunks.begin() + i, chunks.begin() + batchEnd);

            auto embeddings = onnxRuntime_->embedBatch(batchTexts);

            for (size_t j = 0; j < batchTexts.size(); j++) {
                ChunkData cd;
                cd.documentId = static_cast<uint64_t>(docId);
                cd.documentExternalId = externalId;
                cd.content = std::move(batchTexts[j]);
                cd.tokenIds = tokenizer_->encode(cd.content);
                cd.vector = std::move(embeddings[j]);
                if (kiwiTokenizer_) {
                    cd.kiwiTokens = hashMorphemes(kiwiTokenizer_->tokenizeForIndexing(cd.content));
                }
                pendingInserts.push_back(std::move(cd));
            }

            // Flush DB when accumulated enough
            if (pendingInserts.size() >= DB_BATCH_SIZE) {
                obxManager_->insertAllChunks(pendingInserts);
                pendingInserts.clear();
            }
        }

        // Flush remaining
        if (!pendingInserts.empty()) {
            obxManager_->insertAllChunks(pendingInserts);
        }

        LOGI("Document added: id=%lld, chunks=%zu", (long long)docId, chunks.size());
        return docId;
    } catch (const std::exception& e) {
        LOGE("addDocument failed: %s", e.what());
        return -1;
    }
}

std::vector<int64_t> EcoVectorStore::addDocuments(const std::vector<std::string>& texts,
                                                   const std::vector<std::string>& titles,
                                                   const std::vector<std::string>& externalIds,
                                                   const std::vector<int64_t>& createdAts,
                                                   const std::vector<std::string>& senders,
                                                   const std::vector<int16_t>& sourceTypes) {
    if (!obxManager_ || !tokenizer_ || !onnxRuntime_) return {};

    try {
        // 1. Batch save all documents at once
        std::vector<DocData> docs;
        docs.reserve(texts.size());
        for (size_t i = 0; i < texts.size(); i++) {
            DocData doc;
            doc.id = 0;
            doc.content = texts[i];
            if (i < externalIds.size()) doc.externalId = externalIds[i];
            if (i < createdAts.size()) doc.createdAt = createdAts[i];
            if (i < senders.size()) doc.sender = senders[i];
            if (i < sourceTypes.size()) doc.sourceType = static_cast<SourceType>(sourceTypes[i]);
            docs.push_back(std::move(doc));
        }

        auto rawIds = obxManager_->insertAllDocuments(docs);

        std::vector<int64_t> docIds;
        docIds.reserve(rawIds.size());
        for (auto id : rawIds) {
            docIds.push_back(static_cast<int64_t>(id));
        }

        // 2. Two-tier batching: embed in EMBED_BATCH_SIZE, DB insert in DB_BATCH_SIZE
        struct PendingChunk {
            uint64_t documentId;
            std::string text;
            std::string documentExternalId;
            int64_t createdAt = 0;
            std::string sender;
            int16_t sourceType = 0;
        };
        std::vector<PendingChunk> chunkQueue;
        std::vector<ChunkData> pendingInserts;
        size_t queueHead = 0;
        size_t totalChunks = 0;
        static constexpr size_t KIWI_RELOAD_INTERVAL = 1000;

        // Embed one batch from queue, append results to pendingInserts
        auto embedBatch = [&](size_t count) {
            std::vector<std::string> batchTexts;
            batchTexts.reserve(count);
            for (size_t k = queueHead; k < queueHead + count; k++) {
                batchTexts.push_back(chunkQueue[k].text);
            }

            auto embeddings = onnxRuntime_->embedBatch(batchTexts);

            for (size_t k = 0; k < count; k++) {
                ChunkData cd;
                cd.documentId = chunkQueue[queueHead + k].documentId;
                cd.documentExternalId = chunkQueue[queueHead + k].documentExternalId;
                cd.content = std::move(chunkQueue[queueHead + k].text);
                cd.tokenIds = tokenizer_->encode(cd.content);
                cd.vector = std::move(embeddings[k]);
                cd.createdAt = chunkQueue[queueHead + k].createdAt;
                cd.sender = chunkQueue[queueHead + k].sender;
                cd.sourceType = static_cast<SourceType>(chunkQueue[queueHead + k].sourceType);
                if (kiwiTokenizer_) {
                    cd.kiwiTokens = hashMorphemes(kiwiTokenizer_->tokenizeForIndexing(cd.content));
                    kiwiCallCount_++;
                    // Kiwi 내부 메모리 누적 방지: 주기적 reload
                    if (kiwiCallCount_ % KIWI_RELOAD_INTERVAL == 0) {
                        long rss = getNativeRssKb();
                        LOGI("Kiwi reload at %zu calls (RSS=%ldKB=%.0fMB)", kiwiCallCount_, rss, rss/1024.0);
                        kiwiTokenizer_->reload();
                        purgeNativeMemory();
                        long rssAfter = getNativeRssKb();
                        LOGI("After reload+purge: RSS=%ldKB=%.0fMB (freed %.0fMB)", rssAfter, rssAfter/1024.0, (rss-rssAfter)/1024.0);
                    }
                }
                pendingInserts.push_back(std::move(cd));
            }
            queueHead += count;
        };

        // Flush pendingInserts to DB
        auto flushDb = [&]() {
            if (pendingInserts.empty()) return;
            obxManager_->insertAllChunks(pendingInserts);
            totalChunks += pendingInserts.size();
            pendingInserts.clear();
            purgeNativeMemory();  // DB flush 후 freed 페이지 OS 반환
        };

        // 3. For each document: chunk → enqueue → embed when full → DB when full
        for (size_t i = 0; i < texts.size(); i++) {
            if (i >= docIds.size() || docIds[i] <= 0) continue;

            int16_t st = (i < sourceTypes.size()) ? sourceTypes[i] : -1;
            auto chunks = chunkText(texts[i], st);
            std::string docExternalId = (i < externalIds.size()) ? externalIds[i] : "";
            for (auto& chunk : chunks) {
                chunkQueue.push_back({
                    static_cast<uint64_t>(docIds[i]), std::move(chunk), docExternalId,
                    (i < createdAts.size()) ? createdAts[i] : 0,
                    (i < senders.size()) ? senders[i] : "",
                    (i < sourceTypes.size()) ? sourceTypes[i] : int16_t(0)
                });
            }

            // Embed full batches
            while (chunkQueue.size() - queueHead >= EMBED_BATCH_SIZE) {
                embedBatch(EMBED_BATCH_SIZE);

                // Flush DB when accumulated enough
                if (pendingInserts.size() >= DB_BATCH_SIZE) {
                    flushDb();
                }
            }
        }

        // 4. Embed remaining chunks in queue
        size_t remaining = chunkQueue.size() - queueHead;
        if (remaining > 0) {
            embedBatch(remaining);
        }

        // 5. Final DB flush
        flushDb();

        // Reclaim queue memory
        chunkQueue.clear();

        long rss = getNativeRssKb();
        LOGI("addDocuments: %zu docs, %zu chunks, RSS=%.0fMB, kiwiCalls=%zu",
             docIds.size(), totalChunks, rss/1024.0, kiwiCallCount_);
        return docIds;
    } catch (const std::exception& e) {
        LOGE("addDocuments failed: %s", e.what());
        return {};
    }
}

void EcoVectorStore::removeAll() {
    if (obxManager_) obxManager_->removeAll();
    if (bm25Index_) bm25Index_->removeIndex();
    if (ecoVectorIndex_) ecoVectorIndex_->removeIndexes();
}

std::string EcoVectorStore::removeOrphanDocuments() {
    if (!obxManager_) return "[]";
    auto orphans = obxManager_->removeOrphanDocuments();
    nlohmann::json arr(orphans);
    return arr.dump();
}

// ============================================================================
// Index Management
// ============================================================================

bool EcoVectorStore::buildIndex(int centroidCount) {
    if (!obxManager_) return false;
    try {
        bool vectorOk = buildVectorIndex(centroidCount);
        bool bm25Ok = buildBM25Index();
        return vectorOk && bm25Ok;
    } catch (const std::exception& e) {
        LOGE("buildIndex failed: %s", e.what());
        return false;
    }
}

bool EcoVectorStore::isIndexReady() {
    bool vectorReady = ecoVectorIndex_ && ecoVectorIndex_->isIndexReady();
    bool bm25Ready = bm25Index_ && bm25Index_->isIndexReady();
    return vectorReady && bm25Ready;
}

bool EcoVectorStore::reTokenizeAll() {
    if (!obxManager_) {
        LOGE("reTokenizeAll: ObxManager not initialized");
        return false;
    }
    if (!obxManager_->reTokenizeAllKiwiTokens()) {
        return false;
    }
    // Rebuild BM25 index with new token data
    if (bm25Index_) {
        bool ok = bm25Index_->buildIndex(obxManager_.get());
        LOGI("BM25 index rebuilt after re-tokenization: %s", ok ? "OK" : "FAILED");
        return ok;
    }
    LOGW("BM25Index not available, skipping rebuild");
    return true;
}

// ============================================================================
// Pipeline Stage Methods
// ============================================================================

int EcoVectorStore::addDocumentsOnly(
        const std::vector<std::string>& texts,
        const std::vector<std::string>& titles,
        const std::vector<std::string>& externalIds,
        const std::vector<int64_t>& createdAts,
        const std::vector<std::string>& senders,
        const std::vector<int16_t>& sourceTypes) {
    if (!obxManager_) return -1;
    try {
        std::vector<DocData> docs;
        docs.reserve(texts.size());
        for (size_t i = 0; i < texts.size(); i++) {
            DocData doc;
            doc.id = 0;
            doc.content = texts[i];
            if (i < externalIds.size()) doc.externalId = externalIds[i];
            if (i < createdAts.size()) doc.createdAt = createdAts[i];
            if (i < senders.size()) doc.sender = senders[i];
            if (i < sourceTypes.size()) doc.sourceType = static_cast<SourceType>(sourceTypes[i]);
            docs.push_back(std::move(doc));
        }
        auto insertedIds = obxManager_->insertAllDocuments(docs);
        LOGI("addDocumentsOnly: inserted %zu documents", insertedIds.size());
        return static_cast<int>(insertedIds.size());
    } catch (const std::exception& e) {
        LOGE("addDocumentsOnly failed: %s", e.what());
        return -1;
    }
}

int EcoVectorStore::chunkAllDocuments() {
    if (!obxManager_) return -1;
    try {
        obxManager_->removeAllChunks();
        LOGI("chunkAllDocuments: cleared existing chunks");

        auto docs = obxManager_->getAllDocuments();
        LOGI("chunkAllDocuments: chunking %zu documents", docs.size());

        int totalChunks = 0;
        std::vector<ChunkData> chunkBatch;

        for (const auto& doc : docs) {
            // CALL 문서(sourceType 0): description(=summary)를 각 청크에 prepend
            if (doc.sourceType == SourceType::CALL && !doc.description.empty()) {
                chunkPrefix_ = doc.description;
            } else {
                chunkPrefix_.clear();
            }
            auto chunkTexts = chunkText(doc.content, static_cast<int16_t>(doc.sourceType));
            for (int ci = 0; ci < static_cast<int>(chunkTexts.size()); ci++) {
                ChunkData cd;
                cd.id = 0;
                cd.documentId = doc.id;
                cd.chunkIndex = ci;
                cd.content = chunkTexts[ci];
                cd.vector = {};
                cd.tokenIds = {};
                cd.kiwiTokens = {};
                cd.createdAt = doc.createdAt;
                cd.sourceType = doc.sourceType;
                cd.sender = doc.sender;
                cd.documentExternalId = doc.externalId;
                chunkBatch.push_back(std::move(cd));
            }

            if (chunkBatch.size() >= DB_BATCH_SIZE) {
                obxManager_->insertAllChunks(chunkBatch);
                totalChunks += static_cast<int>(chunkBatch.size());
                chunkBatch.clear();
            }
        }

        if (!chunkBatch.empty()) {
            obxManager_->insertAllChunks(chunkBatch);
            totalChunks += static_cast<int>(chunkBatch.size());
        }
        chunkPrefix_.clear();  // 루프 종료 후 정리

        LOGI("chunkAllDocuments: created %d chunks from %zu documents", totalChunks, docs.size());
        return totalChunks;
    } catch (const std::exception& e) {
        LOGE("chunkAllDocuments failed: %s", e.what());
        return -1;
    }
}

int EcoVectorStore::embedChunks(bool forceAll) {
    if (!obxManager_ || !onnxRuntime_ || !tokenizer_) return -1;
    LOGI("embedChunks: forceAll=%d", forceAll);

    // 1. Collect all target chunks
    std::vector<ChunkData> allChunks;
    if (forceAll) {
        obxManager_->forEachChunkBatch(256, false,
            [&](const std::vector<ChunkData>& batch) {
                for (const auto& c : batch) allChunks.push_back(c);
            }
        );
    } else {
        obxManager_->forEachChunkWithoutVector(256,
            [&](std::vector<ChunkData>& batch) -> bool {
                for (auto& c : batch) allChunks.push_back(std::move(c));
                return true;
            }
        );
    }

    LOGI("embedChunks: collected %zu chunks, sorting by content length", allChunks.size());

    // 2. Global sort by content length — minimizes padding waste across all batches
    std::sort(allChunks.begin(), allChunks.end(),
        [](const ChunkData& a, const ChunkData& b) {
            return a.content.size() < b.content.size();
        });

    // 3. Embed in sorted order, DB update in batches
    int embeddedCount = 0;
    std::vector<ChunkData> pendingUpdate;
    pendingUpdate.reserve(DB_BATCH_SIZE);

    for (size_t i = 0; i < allChunks.size(); i += EMBED_BATCH_SIZE) {
        size_t end = std::min(i + EMBED_BATCH_SIZE, allChunks.size());
        std::vector<std::string> batchTexts;
        batchTexts.reserve(end - i);
        for (size_t j = i; j < end; j++) {
            batchTexts.push_back(allChunks[j].content);
        }

        auto embeddings = onnxRuntime_->embedBatch(batchTexts);
        for (size_t j = 0; j < embeddings.size(); j++) {
            allChunks[i + j].vector = std::move(embeddings[j]);
            allChunks[i + j].tokenIds = tokenizer_->encode(allChunks[i + j].content);
            pendingUpdate.push_back(std::move(allChunks[i + j]));
        }

        if (pendingUpdate.size() >= DB_BATCH_SIZE) {
            obxManager_->updateAllChunks(pendingUpdate);
            embeddedCount += static_cast<int>(pendingUpdate.size());
            pendingUpdate.clear();
        }
    }

    if (!pendingUpdate.empty()) {
        obxManager_->updateAllChunks(pendingUpdate);
        embeddedCount += static_cast<int>(pendingUpdate.size());
    }

    LOGI("embedChunks: embedded %d chunks", embeddedCount);
    return embeddedCount;
}

int EcoVectorStore::tokenizeChunks() {
    if (!obxManager_ || !kiwiTokenizer_) return -1;
    LOGI("tokenizeChunks: re-tokenizing all chunks with Kiwi");

    int count = 0;
    obxManager_->forEachChunkBatch(DB_BATCH_SIZE, false,
        [&](const std::vector<ChunkData>& batch) {
            auto mutableBatch = batch;
            for (auto& chunk : mutableBatch) {
                chunk.kiwiTokens = hashMorphemes(kiwiTokenizer_->tokenizeForIndexing(chunk.content));
            }
            obxManager_->updateAllChunks(mutableBatch);
            count += static_cast<int>(mutableBatch.size());
        }
    );

    LOGI("tokenizeChunks: tokenized %d chunks", count);
    return count;
}

// ============================================================================
// SQLite Export/Import
// ============================================================================

int EcoVectorStore::exportChunksToSQLite(const std::string& sqlitePath) {
    if (!obxManager_) return -1;

    std::remove(sqlitePath.c_str());

    sqlite3* db = nullptr;
    int rc = sqlite3_open(sqlitePath.c_str(), &db);
    if (rc != SQLITE_OK) {
        LOGE("exportChunksToSQLite: failed to open %s: %s", sqlitePath.c_str(), sqlite3_errmsg(db));
        sqlite3_close(db);
        return -1;
    }

    sqlite3_exec(db, "PRAGMA journal_mode=WAL; PRAGMA synchronous=OFF;", nullptr, nullptr, nullptr);

    const char* createSql = "CREATE TABLE chunks ("
                            "chunk_id INTEGER PRIMARY KEY, "
                            "content TEXT NOT NULL, "
                            "embedding BLOB)";
    rc = sqlite3_exec(db, createSql, nullptr, nullptr, nullptr);
    if (rc != SQLITE_OK) {
        LOGE("exportChunksToSQLite: CREATE TABLE failed: %s", sqlite3_errmsg(db));
        sqlite3_close(db);
        return -1;
    }

    sqlite3_stmt* stmt = nullptr;
    rc = sqlite3_prepare_v2(db, "INSERT INTO chunks (chunk_id, content) VALUES (?, ?)", -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        LOGE("exportChunksToSQLite: prepare failed: %s", sqlite3_errmsg(db));
        sqlite3_close(db);
        return -1;
    }

    int count = 0;
    sqlite3_exec(db, "BEGIN TRANSACTION", nullptr, nullptr, nullptr);

    obxManager_->forEachChunkBatch(DB_BATCH_SIZE, true,
        [&](const std::vector<ChunkData>& batch) {
            for (const auto& chunk : batch) {
                sqlite3_bind_int64(stmt, 1, static_cast<int64_t>(chunk.id));
                sqlite3_bind_text(stmt, 2, chunk.content.c_str(), -1, SQLITE_TRANSIENT);
                sqlite3_step(stmt);
                sqlite3_reset(stmt);
                count++;
            }
            if (count % 10000 < DB_BATCH_SIZE) {
                sqlite3_exec(db, "COMMIT; BEGIN TRANSACTION", nullptr, nullptr, nullptr);
            }
        }
    );

    sqlite3_exec(db, "COMMIT", nullptr, nullptr, nullptr);
    sqlite3_finalize(stmt);
    sqlite3_close(db);

    LOGI("exportChunksToSQLite: exported %d chunks to %s", count, sqlitePath.c_str());
    return count;
}

int EcoVectorStore::importEmbeddingsFromSQLite(const std::string& sqlitePath) {
    if (!obxManager_) return -1;

    sqlite3* db = nullptr;
    int rc = sqlite3_open_v2(sqlitePath.c_str(), &db, SQLITE_OPEN_READONLY, nullptr);
    if (rc != SQLITE_OK) {
        LOGE("importEmbeddingsFromSQLite: failed to open %s: %s", sqlitePath.c_str(), sqlite3_errmsg(db));
        sqlite3_close(db);
        return -1;
    }

    sqlite3_stmt* stmt = nullptr;
    rc = sqlite3_prepare_v2(db,
        "SELECT chunk_id, embedding FROM chunks WHERE embedding IS NOT NULL", -1, &stmt, nullptr);
    if (rc != SQLITE_OK) {
        LOGE("importEmbeddingsFromSQLite: prepare failed: %s", sqlite3_errmsg(db));
        sqlite3_close(db);
        return -1;
    }

    const size_t EXPECTED_DIM = 768;
    const size_t EXPECTED_BLOB_SIZE = EXPECTED_DIM * sizeof(float);

    int count = 0;
    int errorCount = 0;
    std::vector<ChunkData> updateBatch;
    updateBatch.reserve(DB_BATCH_SIZE);

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        uint64_t chunkId = static_cast<uint64_t>(sqlite3_column_int64(stmt, 0));
        const void* blobData = sqlite3_column_blob(stmt, 1);
        int blobSize = sqlite3_column_bytes(stmt, 1);

        if (!blobData || static_cast<size_t>(blobSize) != EXPECTED_BLOB_SIZE) {
            LOGW("importEmbeddingsFromSQLite: chunk_id=%llu invalid blob size=%d (expected %zu)",
                 (unsigned long long)chunkId, blobSize, EXPECTED_BLOB_SIZE);
            errorCount++;
            continue;
        }

        std::vector<float> vec(EXPECTED_DIM);
        std::memcpy(vec.data(), blobData, EXPECTED_BLOB_SIZE);

        auto existingChunk = obxManager_->getChunkById(chunkId, false);
        if (!existingChunk) {
            LOGW("importEmbeddingsFromSQLite: chunk_id=%llu not found in ObjectBox", (unsigned long long)chunkId);
            errorCount++;
            continue;
        }

        existingChunk->vector = std::move(vec);
        updateBatch.push_back(std::move(*existingChunk));

        if (updateBatch.size() >= DB_BATCH_SIZE) {
            obxManager_->updateAllChunks(updateBatch);
            count += static_cast<int>(updateBatch.size());
            updateBatch.clear();
        }
    }

    if (!updateBatch.empty()) {
        obxManager_->updateAllChunks(updateBatch);
        count += static_cast<int>(updateBatch.size());
    }

    sqlite3_finalize(stmt);
    sqlite3_close(db);

    LOGI("importEmbeddingsFromSQLite: updated %d chunks (%d errors) from %s",
         count, errorCount, sqlitePath.c_str());
    return count;
}

// ============================================================================
// Statistics
// ============================================================================

int EcoVectorStore::getDocumentCount() {
    if (!obxManager_) return 0;
    return static_cast<int>(obxManager_->getDocumentCount());
}

int EcoVectorStore::getChunkCount() {
    if (!obxManager_) return 0;
    return static_cast<int>(obxManager_->getChunkCount());
}

// ============================================================================
// Data Inspection (paginated JSON)
// ============================================================================

std::string EcoVectorStore::getDocumentsJson(int offset, int limit) {
    if (!obxManager_) return "[]";
    try {
        auto docs = obxManager_->getAllDocuments();
        nlohmann::json arr = nlohmann::json::array();

        int start = std::min(static_cast<int>(docs.size()), offset);
        int end = std::min(static_cast<int>(docs.size()), offset + limit);

        for (int i = start; i < end; ++i) {
            const auto& d = docs[i];
            arr.push_back({
                {"id", d.id},
                {"externalId", d.externalId},
                {"content", utf8Truncate(d.content, 300)}
            });
        }
        return arr.dump();
    } catch (const std::exception& e) {
        LOGE("getDocumentsJson error: %s", e.what());
        return "[]";
    }
}

std::string EcoVectorStore::getChunksJson(int offset, int limit) {
    if (!obxManager_) return "[]";
    try {
        auto chunks = obxManager_->getAllChunks(true);
        nlohmann::json arr = nlohmann::json::array();

        int start = std::min(static_cast<int>(chunks.size()), offset);
        int end = std::min(static_cast<int>(chunks.size()), offset + limit);

        for (int i = start; i < end; ++i) {
            const auto& c = chunks[i];
            arr.push_back({
                {"id", c.id},
                {"documentId", c.documentId},
                {"content", utf8Truncate(c.content, 300)},
                {"tokenCount", c.tokenIds.size()}
            });
        }
        return arr.dump();
    } catch (const std::exception& e) {
        LOGE("getChunksJson error: %s", e.what());
        return "[]";
    }
}

// ============================================================================
// Lightweight ID-only access
// ============================================================================

std::string EcoVectorStore::getDocumentExternalIdsJson() {
    if (!obxManager_) return "[]";
    try {
        auto ids = obxManager_->getAllDocumentExternalIds();
        nlohmann::json arr(ids);
        return arr.dump();
    } catch (const std::exception& e) {
        LOGE("getDocumentExternalIdsJson error: %s", e.what());
        return "[]";
    }
}

// ============================================================================
// Raw Save (pre-computed vectors, bypasses chunk/tokenize/embed pipeline)
// ============================================================================

int64_t EcoVectorStore::saveDocumentRaw(const std::string& externalId,
                                         const std::string& description,
                                         const std::string& content,
                                         int64_t createdAt,
                                         int16_t sourceType,
                                         const std::string& sender) {
    if (!obxManager_) return -1;
    try {
        DocData doc;
        doc.id = 0;
        doc.externalId = externalId;
        doc.description = description;
        doc.content = content;
        doc.createdAt = createdAt;
        doc.sourceType = static_cast<SourceType>(sourceType);
        doc.sender = sender;
        auto ids = obxManager_->insertAllDocuments({doc});
        return ids.empty() ? -1 : static_cast<int64_t>(ids[0]);
    } catch (const std::exception& e) {
        LOGE("saveDocumentRaw failed: %s", e.what());
        return -1;
    }
}

int64_t EcoVectorStore::saveChunkRaw(int64_t documentId, int32_t chunkIndex,
                                      const std::string& content,
                                      const std::vector<int32_t>& tokenIds,
                                      const std::vector<float>& embedding,
                                      const std::vector<int32_t>& kiwiTokens,
                                      int64_t createdAt, int16_t sourceType,
                                      const std::string& sender) {
    if (!obxManager_) return -1;
    try {
        ChunkData cd;
        cd.id = 0;
        cd.documentId = static_cast<uint64_t>(documentId);
        cd.chunkIndex = chunkIndex;
        cd.content = content;
        cd.tokenIds = tokenIds;
        cd.vector = embedding;
        cd.kiwiTokens = kiwiTokens;
        cd.createdAt = createdAt;
        cd.sourceType = static_cast<SourceType>(sourceType);
        cd.sender = sender;
        auto ids = obxManager_->insertAllChunks({cd});
        return ids.empty() ? -1 : static_cast<int64_t>(ids[0]);
    } catch (const std::exception& e) {
        LOGE("saveChunkRaw failed: %s", e.what());
        return -1;
    }
}

// saveQueryRaw and saveGroundTruths removed — queries/GT now go to BenchmarkObxManager

// ============================================================================
// [TEMPORARY] Strip URLs from SMS/FILE chunks and re-embed
// ============================================================================

// Fast URL presence check using string::find (avoids regex overhead for non-URL chunks)
static bool containsUrl(const std::string& text) {
    return text.find("http://") != std::string::npos
        || text.find("https://") != std::string::npos
        || text.find("www.") != std::string::npos;
}

// Strip URLs from text using regex + collapse multiple whitespace
static std::string stripUrls(const std::string& text) {
    static const std::regex urlPattern(R"(https?://[^\s]+|www\.[^\s]+)");
    std::string result = std::regex_replace(text, urlPattern, "");
    // collapse multiple spaces/newlines into single space
    static const std::regex multiSpace(R"(\s{2,})");
    result = std::regex_replace(result, multiSpace, " ");
    // trim leading/trailing whitespace
    size_t start = result.find_first_not_of(" \t\n\r");
    size_t end = result.find_last_not_of(" \t\n\r");
    if (start == std::string::npos) return "";
    return result.substr(start, end - start + 1);
}

// ============================================================================
// Retriever Factories
// ============================================================================

VectorRetriever* EcoVectorStore::createVectorRetriever(
        const VectorRetriever::Params& params) {
    if (!obxManager_ || !embedder_) {
        LOGE("createVectorRetriever: store not initialized");
        return nullptr;
    }

    if (!ecoVectorIndex_) {
        LOGE("createVectorRetriever: EcoVectorIndex not available");
        return nullptr;
    }

    auto retriever = std::make_unique<VectorRetriever>(ecoVectorIndex_.get(), obxManager_.get(), embedder_.get());
    retriever->params = params;
    auto* ptr = retriever.get();
    ownedRetrievers_.push_back(std::move(retriever));
    LOGI("VectorRetriever created (owned by EcoVectorStore)");
    return ptr;
}

ObxVectorRetriever* EcoVectorStore::createObxVectorRetriever(
        const ObxVectorRetriever::Params& params) {
    if (!obxManager_ || !embedder_) {
        LOGE("createObxVectorRetriever: store not initialized");
        return nullptr;
    }

    auto retriever = std::make_unique<ObxVectorRetriever>(obxManager_.get(), embedder_.get());
    retriever->params = params;
    auto* ptr = retriever.get();
    ownedRetrievers_.push_back(std::move(retriever));
    LOGI("ObxVectorRetriever created (topK=%u, maxRes=%u)", params.topK, params.maxResultCount);
    return ptr;
}

BM25Retriever* EcoVectorStore::createBM25Retriever(
        const BM25Retriever::Params& params) {
    if (!obxManager_) {
        LOGE("createBM25Retriever: store not initialized");
        return nullptr;
    }

    if (!bm25Index_) {
        LOGE("createBM25Retriever: BM25Index not available");
        return nullptr;
    }

    auto retriever = std::make_unique<BM25Retriever>(bm25Index_.get(), obxManager_.get(), kiwiTokenizer_.get());
    retriever->params = params;
    auto* ptr = retriever.get();
    ownedRetrievers_.push_back(std::move(retriever));
    LOGI("BM25Retriever created (k1=%.2f, b=%.2f, topK=%u)", params.k1, params.b, params.topK);
    return ptr;
}

EnsembleRetriever* EcoVectorStore::createEnsembleRetriever(
        std::vector<RetrieverConfig> configs,
        const EnsembleRetriever::Params& params) {
    if (!embedder_) {
        LOGE("createEnsembleRetriever: Embedder not available");
        return nullptr;
    }

    auto retriever = std::make_unique<EnsembleRetriever>(std::move(configs), embedder_.get(), kiwiTokenizer_.get());
    retriever->params = params;
    auto* ptr = retriever.get();
    ownedRetrievers_.push_back(std::move(retriever));
    LOGI("EnsembleRetriever created (owned by EcoVectorStore)");
    return ptr;
}

bool EcoVectorStore::destroyRetriever(IRetriever* retriever) {
    auto it = std::find_if(ownedRetrievers_.begin(), ownedRetrievers_.end(),
        [retriever](const std::unique_ptr<IRetriever>& p) { return p.get() == retriever; });
    if (it != ownedRetrievers_.end()) {
        LOGI("destroyRetriever: removing retriever %p", retriever);
        ownedRetrievers_.erase(it);
        return true;
    }
    LOGW("destroyRetriever: retriever %p not found in owned list", retriever);
    return false;
}


// ============================================================================
// Split Index Build
// ============================================================================

bool EcoVectorStore::buildVectorIndex(int centroidCount, int hnswM, int efConstruction,
                                      int maxTrainSamples) {
    if (!ecoVectorIndex_ || !obxManager_) return false;
    try {
        if (hnswM > 0 || efConstruction > 0 || maxTrainSamples > 0) {
            EcoVectorConfig cfg;
            cfg.nCluster = (centroidCount > 0) ? static_cast<size_t>(centroidCount) : 0;
            cfg.hnswM = (hnswM > 0) ? static_cast<size_t>(hnswM) : 16;
            cfg.hnswEfConstruction = (efConstruction > 0) ? static_cast<size_t>(efConstruction) : 100;
            if (maxTrainSamples > 0) cfg.maxTrainSamples = static_cast<size_t>(maxTrainSamples);
            ecoVectorIndex_->setConfig(cfg);
        }
        return ecoVectorIndex_->createIndexes(obxManager_.get(), static_cast<size_t>(centroidCount));
    } catch (const std::exception& e) {
        LOGE("buildVectorIndex failed: %s", e.what());
        return false;
    }
}

bool EcoVectorStore::buildBM25Index() {
    if (!bm25Index_ || !obxManager_) return false;
    try {
        return bm25Index_->buildIndex(obxManager_.get());
    } catch (const std::exception& e) {
        LOGE("buildBM25Index failed: %s", e.what());
        return false;
    }
}

// ============================================================================
// ChunkParams-aware Document Addition
// ============================================================================

int64_t EcoVectorStore::addDocumentWithChunkParams(
        const std::string& text, const std::string& title,
        int maxTokens, int overlapTokens) {
    if (!obxManager_ || !tokenizer_ || !onnxRuntime_) return -1;

    try {
        // 1. Save document
        DocData doc;
        doc.id = 0;
        doc.content = text;

        auto docIds = obxManager_->insertAllDocuments({doc});
        if (docIds.empty()) return -1;
        int64_t docId = static_cast<int64_t>(docIds[0]);

        // 2. Chunk text with custom params
        auto chunks = chunkTextWithParams(text, maxTokens, overlapTokens);
        if (chunks.empty()) return docId;

        // 3. Process chunks: embed in EMBED_BATCH_SIZE, accumulate, save in DB_BATCH_SIZE
        std::vector<ChunkData> pendingInserts;
        pendingInserts.reserve(chunks.size());

        for (size_t i = 0; i < chunks.size(); i += EMBED_BATCH_SIZE) {
            size_t batchEnd = std::min(i + EMBED_BATCH_SIZE, chunks.size());
            std::vector<std::string> batchTexts(chunks.begin() + i, chunks.begin() + batchEnd);

            auto embeddings = onnxRuntime_->embedBatch(batchTexts);

            for (size_t j = 0; j < batchTexts.size(); j++) {
                ChunkData cd;
                cd.documentId = static_cast<uint64_t>(docId);
                cd.content = std::move(batchTexts[j]);
                cd.tokenIds = tokenizer_->encode(cd.content);
                cd.vector = std::move(embeddings[j]);
                if (kiwiTokenizer_) {
                    cd.kiwiTokens = hashMorphemes(kiwiTokenizer_->tokenizeForIndexing(cd.content));
                }
                pendingInserts.push_back(std::move(cd));
            }

            if (pendingInserts.size() >= DB_BATCH_SIZE) {
                obxManager_->insertAllChunks(pendingInserts);
                pendingInserts.clear();
            }
        }

        if (!pendingInserts.empty()) {
            obxManager_->insertAllChunks(pendingInserts);
        }

        LOGI("addDocumentWithChunkParams: id=%lld, chunks=%zu (maxTokens=%d, overlap=%d)",
             (long long)docId, chunks.size(), maxTokens, overlapTokens);
        return docId;
    } catch (const std::exception& e) {
        LOGE("addDocumentWithChunkParams failed: %s", e.what());
        return -1;
    }
}

int EcoVectorStore::addDocumentsWithChunkParams(
        const std::vector<std::string>& texts,
        const std::vector<std::string>& titles,
        int maxTokens, int overlapTokens) {
    int count = 0;
    for (size_t i = 0; i < texts.size(); i++) {
        const std::string& title = (i < titles.size()) ? titles[i] : "";
        int64_t id = addDocumentWithChunkParams(texts[i], title,
                                                 maxTokens, overlapTokens);
        if (id > 0) count++;
    }
    return count;
}

// ============================================================================
// Data Management (remove by ID)
// ============================================================================

bool EcoVectorStore::removeDocumentById(uint64_t id) {
    if (!obxManager_) return false;
    try {
        return obxManager_->removeDocumentById(id);
    } catch (const std::exception& e) {
        LOGE("removeDocumentById failed: %s", e.what());
        return false;
    }
}

bool EcoVectorStore::removeChunkById(uint64_t id) {
    if (!obxManager_) return false;
    try {
        return obxManager_->removeChunkById(id);
    } catch (const std::exception& e) {
        LOGE("removeChunkById failed: %s", e.what());
        return false;
    }
}

// ============================================================================
// Data Access (single-item JSON)
// ============================================================================

std::string EcoVectorStore::getDocumentJson(uint64_t id) {
    if (!obxManager_) return "{}";
    try {
        auto doc = obxManager_->getDocumentById(id);
        if (!doc.has_value()) return "{}";

        nlohmann::json j;
        j["id"] = doc->id;
        j["externalId"] = doc->externalId;
        j["description"] = doc->description;
        j["content"] = doc->content;
        j["createdAt"] = doc->createdAt;
        j["sourceType"] = static_cast<int16_t>(doc->sourceType);
        j["sender"] = doc->sender;
        return j.dump();
    } catch (const std::exception& e) {
        LOGE("getDocumentJson failed: %s", e.what());
        return "{}";
    }
}

std::string EcoVectorStore::getChunksByDocumentJson(uint64_t docId) {
    if (!obxManager_) return "[]";
    try {
        // Get all chunks (without vectors for efficiency) and filter by docId
        auto allChunks = obxManager_->getAllChunks(/* excludeVectors= */ true);
        nlohmann::json arr = nlohmann::json::array();

        for (const auto& c : allChunks) {
            if (c.documentId != docId) continue;
            arr.push_back({
                {"id", c.id},
                {"documentId", c.documentId},
                {"chunkIndex", c.chunkIndex},
                {"content", c.content},
                {"tokenCount", c.tokenIds.size()},
                {"kiwiTokenCount", c.kiwiTokens.size()}
            });
        }
        return arr.dump();
    } catch (const std::exception& e) {
        LOGE("getChunksByDocumentJson failed: %s", e.what());
        return "[]";
    }
}

std::string EcoVectorStore::getChunkJson(uint64_t id) {
    if (!obxManager_) return "{}";
    try {
        auto chunk = obxManager_->getChunkById(id, /* excludeVector= */ true);
        if (!chunk.has_value()) return "{}";

        nlohmann::json j;
        j["id"] = chunk->id;
        j["documentId"] = chunk->documentId;
        j["chunkIndex"] = chunk->chunkIndex;
        j["content"] = chunk->content;
        j["tokenCount"] = chunk->tokenIds.size();
        j["kiwiTokenCount"] = chunk->kiwiTokens.size();
        j["createdAt"] = chunk->createdAt;
        j["sourceType"] = static_cast<int16_t>(chunk->sourceType);
        j["sender"] = chunk->sender;
        j["documentExternalId"] = chunk->documentExternalId;
        return j.dump();
    } catch (const std::exception& e) {
        LOGE("getChunkJson failed: %s", e.what());
        return "{}";
    }
}

} // namespace ecovector
