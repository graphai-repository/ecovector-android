// app/src/main/cpp/src/common/TokenConstants.h
#ifndef ECOVECTOR_COMMON_TOKEN_CONSTANTS_H
#define ECOVECTOR_COMMON_TOKEN_CONSTANTS_H

#include <cstdint>

namespace ecovector {
namespace tokens {

// BERT 토크나이저 특수 토큰 ID
// tokenizer.json 기반
constexpr int32_t PAD = 0;   // [PAD] - 패딩
constexpr int32_t UNK = 1;   // [UNK] - 알 수 없는 토큰
constexpr int32_t CLS = 2;   // [CLS] - 문장 시작
constexpr int32_t SEP = 3;   // [SEP] - 문장 구분

// 특수 토큰 여부 확인
inline bool isSpecialToken(int32_t tokenId) {
    return tokenId <= SEP;
}

} // namespace tokens
} // namespace ecovector

#endif // ECOVECTOR_COMMON_TOKEN_CONSTANTS_H
