#include "GroundTruthResolver.h"
#include "BenchmarkObxManager.h"
#include <android/log.h>

#define LOG_TAG "GroundTruthResolver"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

namespace ecovector {

const std::vector<uint64_t> GroundTruthResolver::emptyVec_;

GroundTruthResolver::GroundTruthResolver(ObxManager& obxManager, BenchmarkObxManager& benchmarkObxManager)
    : obxManager_(obxManager), benchmarkObxManager_(&benchmarkObxManager) {}

GroundTruthResolver::GroundTruthResolver(ObxManager& obxManager)
    : obxManager_(obxManager) {}

void GroundTruthResolver::load() {
    groundTruthMap_.clear();

    // Document 매핑은 항상 Core DB에서 로드
    std::unordered_map<std::string, uint64_t> docExternalToInternal;
    auto docs = obxManager_.getAllDocuments();
    for (const auto& doc : docs) {
        if (!doc.externalId.empty()) {
            docExternalToInternal[doc.externalId] = doc.id;
        }
    }

    // GT는 BenchmarkObxManager에서 로드 (Core DB tables retired)
    std::vector<GroundTruthData> gts;
    if (benchmarkObxManager_) {
        gts = benchmarkObxManager_->getAllGroundTruths();
    } else {
        LOGI("BenchmarkObxManager not set — no GT available");
    }

    for (const auto& gt : gts) {
        auto it = docExternalToInternal.find(gt.docId);
        if (it != docExternalToInternal.end()) {
            groundTruthMap_[gt.queryId].push_back(it->second);
        }
    }

    LOGI("GroundTruth loaded: %zu queries with targets, %zu doc mappings",
         groundTruthMap_.size(), docExternalToInternal.size());
}

const std::vector<uint64_t>& GroundTruthResolver::getTargetDocIds(
        const std::string& queryExternalId) const {
    auto it = groundTruthMap_.find(queryExternalId);
    return (it != groundTruthMap_.end()) ? it->second : emptyVec_;
}

} // namespace ecovector
