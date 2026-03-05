#define OBX_CPP_FILE

#include "BenchmarkObxManager.h"
#include "objectbox.h"
#include "objectbox.hpp"
#include "objectbox-benchmark-model.h"
#include "schema-benchmark.obx.hpp"
#include <json.hpp>

#include <android/log.h>
#include <sqlite3.h>
#include <stdexcept>
#include <algorithm>
#include <cstring>
#include <unordered_map>

#define LOG_TAG "BenchmarkObxManager"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace ecovector {

static const size_t WRITE_BATCH_SIZE = 64;

// Binary conversion helpers (same as ObxManager)
static std::vector<uint8_t> tokenIdsToBinary(const std::vector<int32_t>& tokenIds) {
    std::vector<uint8_t> result;
    if (tokenIds.empty()) return result;
    result.resize(tokenIds.size() * sizeof(int32_t));
    std::memcpy(result.data(), tokenIds.data(), result.size());
    return result;
}

static std::vector<int32_t> binaryToTokenIds(const std::vector<uint8_t>& binaryData) {
    std::vector<int32_t> result;
    if (binaryData.empty()) return result;
    size_t count = binaryData.size() / sizeof(int32_t);
    result.resize(count);
    std::memcpy(result.data(), binaryData.data(), count * sizeof(int32_t));
    return result;
}

// Conversion: QueryData ↔ ecovector_bench::Query
static ecovector_bench::Query toObxQuery(const QueryData& q, bool forInsert) {
    ecovector_bench::Query o;
    o.id            = forInsert ? 0 : q.id;
    o.content       = q.content;
    o.refined_query = q.refinedQuery;
    o._id           = q.externalId;
    o.vector        = q.vector;
    o.token_ids     = tokenIdsToBinary(q.tokenIds);
    o.kiwi_tokens   = tokenIdsToBinary(q.kiwiTokens);
    o.created_at    = q.createdAt;
    o.target_types  = q.targetTypes;
    o.categories    = q.categories;
    o.split         = q.split;
    o.eval_top_k    = q.evalTopK;
    return o;
}

static QueryData fromObxQuery(const ecovector_bench::Query& o, bool includeVector) {
    QueryData q;
    q.id           = o.id;
    q.externalId   = o._id;
    q.content      = o.content;
    q.refinedQuery = o.refined_query;
    q.tokenIds     = binaryToTokenIds(o.token_ids);
    q.kiwiTokens   = binaryToTokenIds(o.kiwi_tokens);
    q.createdAt    = o.created_at;
    q.targetTypes  = o.target_types;
    q.categories   = o.categories;
    q.split        = o.split;
    q.evalTopK     = o.eval_top_k;
    if (includeVector) q.vector = o.vector;
    return q;
}

// Conversion: GroundTruthData ↔ ecovector_bench::GroundTruth
static ecovector_bench::GroundTruth toObxGT(const GroundTruthData& g, bool forInsert) {
    ecovector_bench::GroundTruth o;
    o.id       = forInsert ? 0 : g.id;
    o.query_id = g.queryId;
    o.doc_id   = g.docId;
    return o;
}

static GroundTruthData fromObxGT(const ecovector_bench::GroundTruth& o) {
    GroundTruthData g;
    g.id      = o.id;
    g.queryId = o.query_id;
    g.docId   = o.doc_id;
    return g;
}

// PIMPL
class BenchmarkObxManager::Impl {
public:
    std::unique_ptr<obx::Store> store;
    std::unique_ptr<obx::Box<ecovector_bench::Query>> boxQuery;
    std::unique_ptr<obx::Box<ecovector_bench::GroundTruth>> boxGroundTruth;
};

BenchmarkObxManager::BenchmarkObxManager(const std::string& dbPath)
    : pImpl_(std::make_unique<Impl>()), dbPath_(dbPath) {
    LOGI("BenchmarkObxManager constructed with path: %s", dbPath.c_str());
}

BenchmarkObxManager::~BenchmarkObxManager() {
    if (pImpl_) {
        pImpl_->boxQuery.reset();
        pImpl_->boxGroundTruth.reset();
        if (pImpl_->store) {
            try { pImpl_->store->close(); } catch (...) {}
        }
    }
}

bool BenchmarkObxManager::initialize() {
    if (pImpl_->store) return true;
    try {
        OBX_model* model = create_benchmark_obx_model();
        if (!model) { LOGE("Failed to create benchmark model"); return false; }

        obx::Options options;
        options.model(model);
        options.directory(dbPath_);
        pImpl_->store = std::make_unique<obx::Store>(options);

        pImpl_->boxQuery = std::make_unique<obx::Box<ecovector_bench::Query>>(*pImpl_->store);
        pImpl_->boxGroundTruth = std::make_unique<obx::Box<ecovector_bench::GroundTruth>>(*pImpl_->store);

        LOGI("Benchmark DB initialized — Queries: %llu, GT: %llu",
             (unsigned long long)pImpl_->boxQuery->count(),
             (unsigned long long)pImpl_->boxGroundTruth->count());
        return true;
    } catch (const std::exception& e) {
        LOGE("Benchmark DB init error: %s", e.what());
        pImpl_->store.reset();
        return false;
    }
}

// ==================== Query Read ====================

std::vector<QueryData> BenchmarkObxManager::getAllQueries(bool excludeVectors) {
    std::vector<QueryData> result;
    auto all = pImpl_->boxQuery->getAll();
    result.reserve(all.size());
    for (const auto& q : all) result.push_back(fromObxQuery(*q, !excludeVectors));
    return result;
}

std::vector<QueryData> BenchmarkObxManager::getQueriesBySplit(
        const std::string& split, bool excludeVectors) {
    auto all = getAllQueries(excludeVectors);
    if (split.empty()) return all;
    std::vector<QueryData> result;
    for (auto& q : all) {
        if (q.split == split) result.push_back(std::move(q));
    }
    return result;
}

std::vector<std::string> BenchmarkObxManager::getAllQueryExternalIds() {
    std::vector<std::string> result;
    auto all = pImpl_->boxQuery->getAll();
    result.reserve(all.size());
    for (const auto& q : all) result.push_back(q->_id);
    return result;
}

uint32_t BenchmarkObxManager::getQueryCount() {
    return static_cast<uint32_t>(pImpl_->boxQuery->count());
}

std::string BenchmarkObxManager::getQueriesJson(int offset, int limit) {
    try {
        auto queries = getAllQueries(true);
        nlohmann::json arr = nlohmann::json::array();
        int start = std::min(static_cast<int>(queries.size()), offset);
        int end = std::min(static_cast<int>(queries.size()), offset + limit);
        for (int i = start; i < end; ++i) {
            const auto& q = queries[i];
            arr.push_back({
                {"id", q.id},
                {"content", q.content.substr(0, 300)},
                {"externalId", q.externalId},
                {"tokenCount", q.tokenIds.size()}
            });
        }
        return arr.dump();
    } catch (const std::exception& e) {
        LOGE("getQueriesJson error: %s", e.what());
        return "[]";
    }
}

std::string BenchmarkObxManager::getQueryExternalIdsJson() {
    try {
        auto ids = getAllQueryExternalIds();
        nlohmann::json arr(ids);
        return arr.dump();
    } catch (const std::exception& e) {
        LOGE("getQueryExternalIdsJson error: %s", e.what());
        return "[]";
    }
}

std::optional<QueryData> BenchmarkObxManager::getQueryByExternalId(const std::string& externalId) {
    auto all = pImpl_->boxQuery->getAll();
    for (const auto& q : all) {
        if (q->_id == externalId) return fromObxQuery(*q, true);
    }
    return std::nullopt;
}

// ==================== Query Write ====================

uint64_t BenchmarkObxManager::insertQuery(const QueryData& query) {
    auto obxQ = toObxQuery(query, true);
    return pImpl_->boxQuery->put(obxQ);
}

std::vector<uint64_t> BenchmarkObxManager::insertAllQueries(const std::vector<QueryData>& queries) {
    std::vector<uint64_t> ids;
    ids.reserve(queries.size());
    for (size_t i = 0; i < queries.size(); i += WRITE_BATCH_SIZE) {
        size_t end = std::min(i + WRITE_BATCH_SIZE, queries.size());
        obx::Transaction txn(pImpl_->store->txWrite());
        for (size_t j = i; j < end; j++) {
            auto obxQ = toObxQuery(queries[j], true);
            ids.push_back(pImpl_->boxQuery->put(obxQ));
        }
        txn.success();
    }
    return ids;
}

void BenchmarkObxManager::removeAllQueries() {
    pImpl_->boxQuery->removeAll();
}

uint64_t BenchmarkObxManager::insertQueryTextOnly(const QueryData& query) {
    QueryData q = query;
    q.vector = {};
    q.tokenIds = {};
    q.kiwiTokens = {};
    return insertQuery(q);
}

int BenchmarkObxManager::updateAllQueryEmbeddings(
        const std::function<std::pair<std::vector<float>, std::vector<int32_t>>(
            const std::string&)>& embedFn) {
    auto all = pImpl_->boxQuery->getAll();
    int count = 0;
    for (size_t i = 0; i < all.size(); i += WRITE_BATCH_SIZE) {
        size_t end = std::min(i + WRITE_BATCH_SIZE, all.size());
        obx::Transaction txn(pImpl_->store->txWrite());
        for (size_t j = i; j < end; j++) {
            auto& q = *all[j];
            std::string embedTarget = q.refined_query.empty() ? q.content : q.refined_query;
            auto [vec, tokIds] = embedFn(embedTarget);
            q.vector = std::move(vec);
            q.token_ids = tokenIdsToBinary(tokIds);
            pImpl_->boxQuery->put(q);
            count++;
        }
        txn.success();
    }
    LOGI("updateAllQueryEmbeddings: updated %d queries", count);
    return count;
}

bool BenchmarkObxManager::reTokenizeAllKiwiTokens(
        const std::function<std::vector<int32_t>(const std::string&)>& tokenizeFn) {
    auto all = pImpl_->boxQuery->getAll();
    size_t updated = 0;
    for (size_t i = 0; i < all.size(); i += WRITE_BATCH_SIZE) {
        size_t end = std::min(i + WRITE_BATCH_SIZE, all.size());
        obx::Transaction txn(pImpl_->store->txWrite());
        for (size_t j = i; j < end; j++) {
            auto& q = *all[j];
            auto newTokens = tokenizeFn(q.content);
            q.kiwi_tokens = tokenIdsToBinary(newTokens);
            pImpl_->boxQuery->put(q);
            updated++;
        }
        txn.success();
    }
    LOGI("Re-tokenized %zu queries", updated);
    return true;
}

int BenchmarkObxManager::importQueryEmbeddingsFromSQLite(const std::string& dbPath) {
    sqlite3* db = nullptr;
    int rc = sqlite3_open_v2(dbPath.c_str(), &db, SQLITE_OPEN_READONLY, nullptr);
    if (rc != SQLITE_OK) {
        LOGE("importQueryEmbeddings: cannot open %s: %s", dbPath.c_str(), sqlite3_errmsg(db));
        if (db) sqlite3_close(db);
        return -1;
    }

    // query_embeddings 테이블 존재 확인
    sqlite3_stmt* probe = nullptr;
    rc = sqlite3_prepare_v2(db, "SELECT external_id, embedding FROM query_embeddings LIMIT 1", -1, &probe, nullptr);
    if (rc != SQLITE_OK) {
        LOGI("importQueryEmbeddings: no query_embeddings table, skipping");
        sqlite3_close(db);
        return 0;
    }
    sqlite3_finalize(probe);

    // externalId → obx id 매핑 빌드
    auto all = pImpl_->boxQuery->getAll();
    std::unordered_map<std::string, obx_id> extIdMap;
    for (auto& q : all) {
        extIdMap[q->_id] = q->id;
    }

    // 임포트
    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(db, "SELECT external_id, embedding FROM query_embeddings WHERE embedding IS NOT NULL", -1, &stmt, nullptr);

    int count = 0, errorCount = 0;
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        const char* extId = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
        const void* blobData = sqlite3_column_blob(stmt, 1);
        int blobSize = sqlite3_column_bytes(stmt, 1);

        if (!extId || !blobData || blobSize <= 0 || blobSize % sizeof(float) != 0) {
            errorCount++;
            continue;
        }

        auto it = extIdMap.find(extId);
        if (it == extIdMap.end()) {
            continue;  // 쿼리 미매칭 (다른 split 등)
        }

        size_t dim = blobSize / sizeof(float);
        std::vector<float> vec(dim);
        std::memcpy(vec.data(), blobData, blobSize);

        // ObjectBox 업데이트
        auto existing = pImpl_->boxQuery->get(it->second);
        if (existing) {
            existing->vector = std::move(vec);
            obx::Transaction txn(pImpl_->store->txWrite());
            pImpl_->boxQuery->put(*existing);
            txn.success();
            count++;
        }
    }

    sqlite3_finalize(stmt);
    sqlite3_close(db);
    LOGI("importQueryEmbeddings: imported %d queries (%d errors)", count, errorCount);
    return count;
}

// ==================== GroundTruth ====================

std::vector<GroundTruthData> BenchmarkObxManager::getAllGroundTruths() {
    std::vector<GroundTruthData> result;
    auto all = pImpl_->boxGroundTruth->getAll();
    result.reserve(all.size());
    for (const auto& gt : all) result.push_back(fromObxGT(*gt));
    return result;
}

std::vector<uint64_t> BenchmarkObxManager::insertAllGroundTruths(
        const std::vector<GroundTruthData>& entries) {
    std::vector<uint64_t> ids;
    ids.reserve(entries.size());
    for (size_t i = 0; i < entries.size(); i += WRITE_BATCH_SIZE) {
        size_t end = std::min(i + WRITE_BATCH_SIZE, entries.size());
        obx::Transaction txn(pImpl_->store->txWrite());
        for (size_t j = i; j < end; j++) {
            auto obxGT = toObxGT(entries[j], true);
            ids.push_back(pImpl_->boxGroundTruth->put(obxGT));
        }
        txn.success();
    }
    return ids;
}

void BenchmarkObxManager::removeAllGroundTruths() {
    pImpl_->boxGroundTruth->removeAll();
}

void BenchmarkObxManager::removeAll() {
    removeAllQueries();
    removeAllGroundTruths();
    LOGI("Benchmark DB cleared");
}

} // namespace ecovector
