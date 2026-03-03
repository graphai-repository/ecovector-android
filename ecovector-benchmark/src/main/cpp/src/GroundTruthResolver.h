#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <unordered_map>
#include "ObxManager.h"
#include "BenchmarkTypes.h"

namespace ecovector {

class BenchmarkObxManager;  // forward decl

/**
 * Ground truth 데이터 로딩 및 쿼리-문서 매핑 관리.
 * queryExternalId → [docInternalIds] 매핑을 빌드.
 */
class GroundTruthResolver {
public:
    /** Benchmark DB에서 GT 로드, Core DB에서 Document 룩업 */
    GroundTruthResolver(ObxManager& obxManager, BenchmarkObxManager& benchmarkObxManager);
    /** 기존 호환: Core DB에서 GT + Document 모두 로드 */
    explicit GroundTruthResolver(ObxManager& obxManager);

    /** GT 테이블을 로드하여 매핑 빌드 */
    void load();

    /** 쿼리의 GT target document ID 목록 반환 (없으면 빈 벡터) */
    const std::vector<uint64_t>& getTargetDocIds(const std::string& queryExternalId) const;

    /** 전체 GT 맵 참조 */
    const std::unordered_map<std::string, std::vector<uint64_t>>& getMap() const { return groundTruthMap_; }

    /** GT 엔트리가 있는 쿼리 수 */
    size_t size() const { return groundTruthMap_.size(); }

private:
    ObxManager& obxManager_;
    BenchmarkObxManager* benchmarkObxManager_ = nullptr;
    std::unordered_map<std::string, std::vector<uint64_t>> groundTruthMap_;
    static const std::vector<uint64_t> emptyVec_;
};

} // namespace ecovector
