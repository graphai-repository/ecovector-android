#ifndef ECOVECTOR_KIWI_TOKENIZER_H
#define ECOVECTOR_KIWI_TOKENIZER_H

#include <string>
#include <vector>
#include <memory>

namespace ecovector {

/**
 * KiwiTokenizer - Kiwi 한국어 형태소 분석기 래퍼
 *
 * BM25 인덱싱/검색용. 내용어(명사, 동사, 형용사, 부사, 외국어, 숫자)만
 * 추출하고 기능어(조사, 어미, 기호)는 필터링한다.
 */
class KiwiTokenizer {
public:
    KiwiTokenizer();
    ~KiwiTokenizer();

    bool load(const std::string& modelDirPath, int numThreads = 2);
    /** Kiwi 엔진을 close 후 재로드하여 내부 메모리를 해제한다. */
    bool reload();
    std::vector<std::string> tokenize(const std::string& text) const;
    std::vector<std::string> tokenizeForIndexing(const std::string& text) const;
    bool isLoaded() const;

private:
    struct Impl;
    std::unique_ptr<Impl> pImpl_;
};

} // namespace ecovector

#endif // ECOVECTOR_KIWI_TOKENIZER_H
