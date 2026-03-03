#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <functional>
#include "IRetriever.h"
#include "ObxManager.h"
#include "BenchmarkTypes.h"
#include "GroundTruthResolver.h"

namespace ecovector {

struct BenchmarkMetrics {
    uint32_t totalQueries;
    uint32_t correctResults;
    double hitRate;       // Hit@K (%)
    double recall;        // Recall@K (%)
    double avgLatencyMs;
    double totalLatencySec;
};

struct QueryResult {
    std::vector<ChunkSearchResult> retrievedChunks;  // dedup 후 결과
    bool isHit = false;
    double recall = 0.0;
    double latencyMs = 0.0;
};

struct EvaluationResult {
    BenchmarkMetrics metrics;
    std::vector<QueryResult> queryResults;  // queries와 동일 인덱스
};

/**
 * 검색 결과를 GT와 비교하여 Hit@K, Recall@K, 레이턴시를 계산.
 */
class MetricsEvaluator {
public:
    explicit MetricsEvaluator(const GroundTruthResolver& gt);

    /**
     * 단일 retriever/메서드에 대한 벤치마크 실행 및 평가.
     * @param methodName 메서드 이름 (로그용)
     * @param queries    쿼리 목록
     * @param topK       평가 기준 K
     * @param searchFn   검색 함수 (쿼리 인덱스 → 검색 결과)
     * @return 메트릭 결과
     */
    /**
     * @param topK       평가 기준 K (Hit@K, Recall@K)
     * @param detailTopK JSONL 출력용 결과 수 (0 = topK와 동일)
     */
    EvaluationResult evaluate(
        const std::string& methodName,
        const std::vector<QueryData>& queries,
        uint32_t topK,
        std::function<std::vector<ChunkSearchResult>(size_t queryIdx)> searchFn,
        uint32_t detailTopK = 0);

private:
    const GroundTruthResolver& gt_;

    bool isCorrectResult(const std::string& queryExternalId,
                         const std::vector<ChunkSearchResult>& results) const;
};

} // namespace ecovector
