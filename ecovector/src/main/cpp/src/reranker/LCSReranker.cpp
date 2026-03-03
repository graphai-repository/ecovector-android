// ecovector/src/main/cpp/src/reranker/LCSReranker.cpp
#define LOG_TAG "LCSReranker"
#include "../common/Logging.h"
#include "LCSReranker.h"
#include "../object_box/ObxManager.h"

#include <algorithm>

namespace ecovector {

// ============================================================================
// LCS (Longest Common Subsequence) calculation
// ============================================================================

/**
 * LCS 길이 계산 (2-row DP)
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
// LCSReranker 구현
// ============================================================================

LCSReranker::LCSReranker() {
    LOGD("LCSReranker constructed");
}

LCSReranker::~LCSReranker() {
    LOGD("LCSReranker destructor");
}

std::vector<ChunkSearchResult> LCSReranker::rerank(
    const std::vector<int32_t>& queryKiwiTokens,
    std::vector<ChunkSearchResult>& results) {

    if (queryKiwiTokens.empty() || results.empty()) {
        return results;
    }

    try {
        // 각 결과에 대해 LCS 점수 계산
        struct RankItem {
            size_t index;
            uint32_t lcsScore;
        };

        std::vector<RankItem> rankedItems;
        rankedItems.reserve(results.size());

        for (size_t i = 0; i < results.size(); ++i) {
            const auto& sr = results[i];

            // kiwiTokens는 저장 시점에 이미 내용어만 포함
            uint32_t lcsLength = calculateLCSLength(queryKiwiTokens, sr.chunk.kiwiTokens);
            rankedItems.push_back({i, lcsLength});
        }

        // LCS 점수 내림차순 정렬
        std::stable_sort(rankedItems.begin(), rankedItems.end(),
                         [](const RankItem& a, const RankItem& b) {
                             return a.lcsScore > b.lcsScore;
                         });

        // 정렬된 결과 반환
        std::vector<ChunkSearchResult> rerankedResults;
        rerankedResults.reserve(results.size());
        for (const auto& item : rankedItems) {
            rerankedResults.push_back(results[item.index]);
        }

        LOGD("LCS reranking completed: %zu results", rerankedResults.size());
        return rerankedResults;

    } catch (const std::exception& e) {
        LOGE("Error in LCS reranking: %s", e.what());
        return results;
    }
}

} // namespace ecovector
