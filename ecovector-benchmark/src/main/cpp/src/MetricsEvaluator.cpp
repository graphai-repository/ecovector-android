#include "MetricsEvaluator.h"
#include "SearchUtils.h"
#include <android/log.h>
#include <chrono>
#include <algorithm>

#define LOG_TAG "MetricsEvaluator"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)

namespace ecovector {

MetricsEvaluator::MetricsEvaluator(const GroundTruthResolver& gt)
    : gt_(gt) {}

bool MetricsEvaluator::isCorrectResult(const std::string& queryExternalId,
                                        const std::vector<ChunkSearchResult>& results) const {
    const auto& targetDocIds = gt_.getTargetDocIds(queryExternalId);
    if (targetDocIds.empty()) return false;

    for (const auto& result : results) {
        if (std::find(targetDocIds.begin(), targetDocIds.end(),
                      result.chunk.documentId) != targetDocIds.end()) {
            return true;
        }
    }
    return false;
}

EvaluationResult MetricsEvaluator::evaluate(
    const std::string& methodName,
    const std::vector<QueryData>& queries,
    uint32_t topK,
    std::function<std::vector<ChunkSearchResult>(size_t queryIdx)> searchFn,
    uint32_t detailTopK) {

    uint32_t outputTopK = detailTopK > 0 ? detailTopK : topK;
    uint32_t totalQueries = static_cast<uint32_t>(queries.size());

    // Phase 1: 검색만 실행 — 캐시 오염 없이 순수 검색 시간 측정
    std::vector<std::vector<ChunkSearchResult>> allResults(totalQueries);
    std::vector<double> perQueryLatency(totalQueries, 0.0);
    double totalLatencyMs = 0.0;
    uint32_t validQueries = 0;

    for (uint32_t i = 0; i < totalQueries; ++i) {
        if (queries[i].vector.empty()) continue;

        validQueries++;
        auto startTime = std::chrono::high_resolution_clock::now();
        allResults[i] = searchFn(i);
        auto endTime = std::chrono::high_resolution_clock::now();
        double duration = std::chrono::duration<double, std::milli>(endTime - startTime).count();
        perQueryLatency[i] = duration;
        totalLatencyMs += duration;
    }

    // Phase 2: dedup → 평가 + per-query 결과 수집
    // GT <= topK → Recall@K, GT > topK → Recall@50 (적응형)
    constexpr uint32_t LARGE_K = 50;
    uint32_t correctResults = 0;
    double totalRecall = 0.0;
    std::vector<QueryResult> queryResults(totalQueries);

    for (uint32_t i = 0; i < totalQueries; ++i) {
        if (queries[i].vector.empty()) continue;

        const auto& targetDocIds = gt_.getTargetDocIds(queries[i].externalId);

        // GT 수에 따라 평가 K 결정: GT > topK이면 LARGE_K(50) 사용
        uint32_t evalK = (!targetDocIds.empty() && targetDocIds.size() > topK) ? LARGE_K : topK;
        uint32_t dedupLimit = std::max(outputTopK, evalK);
        auto dedupedResults = deduplicateByDocument(std::move(allResults[i]), dedupLimit);

        uint32_t evalLimit = std::min(evalK, static_cast<uint32_t>(dedupedResults.size()));
        bool isHit = false;
        double queryRecall = 0.0;
        if (!targetDocIds.empty()) {
            uint32_t hits = 0;
            for (uint32_t r = 0; r < evalLimit; ++r) {
                if (std::find(targetDocIds.begin(), targetDocIds.end(),
                              dedupedResults[r].chunk.documentId) != targetDocIds.end()) {
                    isHit = true;
                    hits++;
                }
            }
            queryRecall = (double)hits / targetDocIds.size();
            totalRecall += queryRecall;
        }
        if (isHit) correctResults++;

        queryResults[i].retrievedChunks = std::move(dedupedResults);
        queryResults[i].isHit = isHit;
        queryResults[i].recall = queryRecall;
        queryResults[i].latencyMs = perQueryLatency[i];
    }

    double hitRate = (validQueries > 0) ? (correctResults * 100.0 / validQueries) : 0.0;
    double recall = (validQueries > 0) ? (totalRecall * 100.0 / validQueries) : 0.0;
    double avgLatency = (validQueries > 0) ? (totalLatencyMs / validQueries) : 0.0;
    double totalLatencySec = totalLatencyMs / 1000.0;

    LOGD("%s Results: Hit@%u=%.2f%% Recall@%u(50)=%.2f%% AvgLat=%.3fms",
         methodName.c_str(), topK, hitRate, topK, recall, avgLatency);

    return {
        {validQueries, correctResults, hitRate, recall, avgLatency, totalLatencySec},
        std::move(queryResults)
    };
}

} // namespace ecovector
