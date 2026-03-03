// ecovector/src/main/cpp/src/reranker/IReranker.h
#ifndef ECOVECTOR_RERANKER_IRERANKER_H
#define ECOVECTOR_RERANKER_IRERANKER_H

#include <vector>
#include <cstdint>

namespace ecovector {

// Forward declaration
struct ChunkSearchResult;

/**
 * IReranker - 리랭커 인터페이스
 *
 * 검색 결과를 재정렬하는 다양한 알고리즘을 추상화
 */
class IReranker {
public:
    virtual ~IReranker() = default;

    /**
     * 검색 결과 리랭킹
     *
     * @param queryKiwiTokens 쿼리의 Kiwi 형태소 해시 토큰 (내용어만 포함)
     * @param results 리랭킹할 검색 결과 (in-place 수정 가능)
     * @return 리랭킹된 결과
     */
    virtual std::vector<ChunkSearchResult> rerank(
        const std::vector<int32_t>& queryKiwiTokens,
        std::vector<ChunkSearchResult>& results) = 0;

    /**
     * 리랭커 이름 반환 (로깅/디버깅용)
     */
    virtual const char* getName() const = 0;
};

} // namespace ecovector

#endif // ECOVECTOR_RERANKER_IRERANKER_H
