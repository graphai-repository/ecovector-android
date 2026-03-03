// ecovector/src/main/cpp/src/reranker/LCSReranker.h
#ifndef ECOVECTOR_RERANKER_LCS_RERANKER_H
#define ECOVECTOR_RERANKER_LCS_RERANKER_H

#include "IReranker.h"

namespace ecovector {

/**
 * LCSReranker - Kiwi 형태소 토큰 기반 LCS 리랭커
 *
 * 쿼리와 청크의 kiwiTokens (내용어 해시)를 직접 비교하여 LCS 계산.
 * kiwiTokens는 저장 시점에 이미 내용어만 필터링되어 있으므로
 * 런타임 필터링 불필요.
 */
class LCSReranker : public IReranker {
public:
    LCSReranker();
    ~LCSReranker() override;

    std::vector<ChunkSearchResult> rerank(
        const std::vector<int32_t>& queryKiwiTokens,
        std::vector<ChunkSearchResult>& results) override;

    const char* getName() const override { return "LCS"; }
};

} // namespace ecovector

#endif // ECOVECTOR_RERANKER_LCS_RERANKER_H
