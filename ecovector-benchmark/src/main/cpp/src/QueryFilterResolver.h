#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include "ObxManager.h"

namespace ecovector {

/**
 * 쿼리별 필터(시간 범위, 도메인)를 JSONL 파일에서 파싱하고
 * ObxManager를 통해 허용 chunk ID 집합으로 해석.
 */
class QueryFilterResolver {
public:
    explicit QueryFilterResolver(ObxManager& obxManager);

    /** 필터 파일을 파싱하고 chunk ID 집합으로 해석 */
    void load(const std::string& filterPath);

    /** 특정 docExternalId에 대한 필터 해석 결과 (없으면 nullptr) */
    const std::unordered_set<uint64_t>* getFilterForDoc(const std::string& docExternalId) const;

    bool hasFilters() const { return !resolvedFilterMap_.empty(); }

    // 쿼리별 원본 필터 정보 (JSON 출력용)
    struct RawFilterInfo {
        std::vector<std::string> sourceTypes;  // domain 문자열: "call", "sms", ...
        std::string timeGte;                   // ISO 8601: "2026-01-10" (원본 그대로)
        std::string timeLte;                   // ISO 8601: "2026-01-20" (원본 그대로)
    };

    /** docExternalId에 대한 원본 필터 정보 (없으면 nullptr) */
    const RawFilterInfo* getRawFilterForDoc(const std::string& docExternalId) const;

    /** 메타데이터 패치: JSONL 파일에서 created_at 일괄 업데이트 */
    void patchCreatedAtFromFile(const std::string& datePatchPath);

    /** 메타데이터 패치: externalId prefix → sourceType 일괄 업데이트 */
    void patchSourceTypeFromPrefix();

private:
    ObxManager& obxManager_;
    std::unordered_map<std::string, std::unordered_set<uint64_t>> resolvedFilterMap_;
    std::unordered_map<std::string, RawFilterInfo> rawFilterMap_;

    static int64_t isoDateToUnixMs(const std::string& dateStr, bool endOfDay);
    static int16_t domainToSourceType(const std::string& domain);
    static std::vector<std::string> readLinesC(const std::string& path);
};

} // namespace ecovector
