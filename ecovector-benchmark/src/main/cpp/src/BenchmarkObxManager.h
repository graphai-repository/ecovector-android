#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <memory>
#include <optional>
#include <functional>
#include "BenchmarkTypes.h"

namespace ecovector {

/**
 * Benchmark DB manager — Query + GroundTruth 전용.
 * Core DB(Document/Chunk)와 분리된 별도 ObjectBox 스토어 관리.
 */
class BenchmarkObxManager {
public:
    explicit BenchmarkObxManager(const std::string& dbPath);
    ~BenchmarkObxManager();

    BenchmarkObxManager(const BenchmarkObxManager&) = delete;
    BenchmarkObxManager& operator=(const BenchmarkObxManager&) = delete;

    bool initialize();

    // ==================== Query Read ====================
    std::vector<QueryData> getAllQueries(bool excludeVectors = true);
    std::vector<QueryData> getQueriesBySplit(const std::string& split, bool excludeVectors = true);
    std::vector<std::string> getAllQueryExternalIds();
    uint32_t getQueryCount();
    std::string getQueriesJson(int offset, int limit);
    std::string getQueryExternalIdsJson();
    std::optional<QueryData> getQueryByExternalId(const std::string& externalId);

    // ==================== Query Write ====================
    uint64_t insertQuery(const QueryData& query);
    std::vector<uint64_t> insertAllQueries(const std::vector<QueryData>& queries);

    // Text-only insert (empty vector/tokenIds/kiwiTokens)
    uint64_t insertQueryTextOnly(const QueryData& query);

    // Update embeddings for all queries using provided function
    int updateAllQueryEmbeddings(
        const std::function<std::pair<std::vector<float>, std::vector<int32_t>>(
            const std::string&)>& embedFn);

    bool reTokenizeAllKiwiTokens(
        const std::function<std::vector<int32_t>(const std::string&)>& tokenizeFn);
    void removeAllQueries();

    // ==================== GroundTruth ====================
    std::vector<GroundTruthData> getAllGroundTruths();
    std::vector<uint64_t> insertAllGroundTruths(const std::vector<GroundTruthData>& entries);
    void removeAllGroundTruths();

    // ==================== Lifecycle ====================
    void removeAll();

    /**
     * SQLite에서 쿼리 임베딩을 임포트한다.
     * 테이블: query_embeddings(external_id TEXT, embedding BLOB)
     * @param dbPath SQLite 파일 경로
     * @return 임포트한 쿼리 수
     */
    int importQueryEmbeddingsFromSQLite(const std::string& dbPath);

    const std::string& getDbPath() const { return dbPath_; }

private:
    class Impl;
    std::unique_ptr<Impl> pImpl_;
    std::string dbPath_;
};

} // namespace ecovector
