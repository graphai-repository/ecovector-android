#include "QueryFilterResolver.h"
#include <json.hpp>
#include <android/log.h>
#include <cstdio>
#include <ctime>

#define LOG_TAG "QueryFilterResolver"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)

namespace ecovector {

QueryFilterResolver::QueryFilterResolver(ObxManager& obxManager)
    : obxManager_(obxManager) {}

int64_t QueryFilterResolver::isoDateToUnixMs(const std::string& dateStr, bool endOfDay) {
    if (dateStr.size() < 10) return 0;
    int year = std::stoi(dateStr.substr(0, 4));
    int month = std::stoi(dateStr.substr(5, 2));
    int day = std::stoi(dateStr.substr(8, 2));

    struct tm tm = {};
    tm.tm_year = year - 1900;
    tm.tm_mon = month - 1;
    tm.tm_mday = day;
    if (endOfDay) {
        tm.tm_hour = 23;
        tm.tm_min = 59;
        tm.tm_sec = 59;
    }
    // Convert to UTC, then adjust for KST (UTC+9)
    time_t t = timegm(&tm);
    t -= 9 * 3600; // KST to UTC
    return static_cast<int64_t>(t) * 1000LL + (endOfDay ? 999 : 0);
}

// C-style file line reader — avoids std::ifstream locale facet conflict
// (libobjectbox-jni.so ships c++_static whose locale::id clashes with c++_shared)
std::vector<std::string> QueryFilterResolver::readLinesC(const std::string& path) {
    std::vector<std::string> lines;
    FILE* fp = fopen(path.c_str(), "r");
    if (!fp) return lines;
    char buf[8192];
    std::string partial;
    while (fgets(buf, sizeof(buf), fp)) {
        partial += buf;
        if (!partial.empty() && partial.back() == '\n') {
            partial.pop_back();
            if (!partial.empty() && partial.back() == '\r') partial.pop_back();
            lines.push_back(std::move(partial));
            partial.clear();
        }
    }
    if (!partial.empty()) {
        if (partial.back() == '\r') partial.pop_back();
        lines.push_back(std::move(partial));
    }
    fclose(fp);
    return lines;
}

int16_t QueryFilterResolver::domainToSourceType(const std::string& domain) {
    if (domain == "call")     return 0; // CALL
    if (domain == "sms")      return 1; // SMS
    if (domain == "mms")      return 2; // MMS
    if (domain == "image")    return 3; // IMAGE
    if (domain == "document") return 4; // FILE
    return -1;
}

void QueryFilterResolver::load(const std::string& filterPath) {
    resolvedFilterMap_.clear();
    rawFilterMap_.clear();
    auto fileLines = readLinesC(filterPath);
    if (fileLines.empty()) {
        LOGI("No filter file at %s, running without filters", filterPath.c_str());
        return;
    }

    std::unordered_map<std::string, std::string> filterJsonMap;

    int totalFilters = 0;
    int timeFilters = 0;
    int domainFilters = 0;
    for (const auto& line : fileLines) {
        if (line.empty()) continue;
        try {
            auto j = nlohmann::json::parse(line);
            // target_corpus_ids (array) 또는 target_corpus_id (string) 지원
            std::vector<std::string> corpusIds;
            if (j.contains("target_corpus_ids") && j["target_corpus_ids"].is_array()) {
                for (const auto& cid : j["target_corpus_ids"]) {
                    corpusIds.push_back(cid.get<std::string>());
                }
            } else {
                std::string corpusId = j.value("target_corpus_id", "");
                if (!corpusId.empty()) corpusIds.push_back(corpusId);
            }
            if (corpusIds.empty()) continue;

            nlohmann::json filterJson;
            bool hasFilter = false;

            // Time window filter
            if (j.contains("filters") && j["filters"].contains("time_window")
                && !j["filters"]["time_window"].is_null()) {
                auto& tw = j["filters"]["time_window"];
                std::string startDate = tw.value("start_date", "");
                std::string endDate = tw.value("end_date", "");
                if (!startDate.empty() && !endDate.empty()) {
                    int64_t startMs = isoDateToUnixMs(startDate, false);
                    int64_t endMs = isoDateToUnixMs(endDate, true);
                    filterJson["created_at"] = {{"gte", startMs}, {"lte", endMs}};
                    hasFilter = true;
                    timeFilters++;
                }
            }

            // Domain filter
            if (j.contains("domains") && j["domains"].is_array() && !j["domains"].empty()) {
                std::vector<int64_t> sourceTypes;
                for (const auto& d : j["domains"]) {
                    int16_t st = domainToSourceType(d.get<std::string>());
                    if (st >= 0) sourceTypes.push_back(static_cast<int64_t>(st));
                }
                if (!sourceTypes.empty()) {
                    if (sourceTypes.size() == 1) {
                        filterJson["source_type"] = sourceTypes[0];
                    } else {
                        filterJson["source_type"] = {{"in", sourceTypes}};
                    }
                    hasFilter = true;
                    domainFilters++;
                }
            }

            if (hasFilter) {
                std::string filterStr = filterJson.dump();
                // 원본 필터 정보 저장 (JSON 출력용)
                RawFilterInfo raw;
                if (j.contains("domains") && j["domains"].is_array()) {
                    for (const auto& d : j["domains"]) {
                        raw.sourceTypes.push_back(d.get<std::string>());
                    }
                }
                if (j.contains("filters") && j["filters"].contains("time_window")
                    && !j["filters"]["time_window"].is_null()) {
                    auto& tw = j["filters"]["time_window"];
                    raw.timeGte = tw.value("start_date", "");
                    raw.timeLte = tw.value("end_date", "");
                }
                // 모든 corpus ID에 동일 필터 적용
                for (const auto& cid : corpusIds) {
                    filterJsonMap[cid] = filterStr;
                    rawFilterMap_[cid] = raw;
                }
                totalFilters++;
            }
        } catch (const std::exception& e) {
            LOGW("Filter parse error: %s", e.what());
        }
    }

    LOGI("Parsed %d filters from file (time: %d, domain: %d)", totalFilters, timeFilters, domainFilters);

    // Resolve each filter to allowedChunkIds set
    int resolved = 0;
    for (auto& [corpusId, filterStr] : filterJsonMap) {
        auto chunkIds = obxManager_.resolveFilter(filterStr);
        if (!chunkIds.empty()) {
            resolvedFilterMap_[corpusId] = std::move(chunkIds);
            resolved++;
        }
    }

    LOGI("Resolved %d/%d filters to chunk ID sets", resolved, totalFilters);
}

const std::unordered_set<uint64_t>* QueryFilterResolver::getFilterForDoc(
        const std::string& docExternalId) const {
    auto it = resolvedFilterMap_.find(docExternalId);
    return (it != resolvedFilterMap_.end()) ? &it->second : nullptr;
}

const QueryFilterResolver::RawFilterInfo* QueryFilterResolver::getRawFilterForDoc(
        const std::string& docExternalId) const {
    auto it = rawFilterMap_.find(docExternalId);
    return (it != rawFilterMap_.end()) ? &it->second : nullptr;
}

void QueryFilterResolver::patchCreatedAtFromFile(const std::string& datePatchPath) {
    auto lines = readLinesC(datePatchPath);
    if (lines.empty()) {
        LOGI("No date patch file at %s", datePatchPath.c_str());
        return;
    }

    std::unordered_map<std::string, int64_t> docDateMap;
    for (const auto& line : lines) {
        if (line.empty()) continue;
        try {
            auto j = nlohmann::json::parse(line);
            std::string id = j.value("id", "");
            int64_t ts = j.value("ts", 0LL);
            if (!id.empty() && ts > 0) {
                docDateMap[id] = ts;
            }
        } catch (...) {}
    }

    if (!docDateMap.empty()) {
        int updated = obxManager_.bulkUpdateCreatedAt(docDateMap);
        LOGI("Patched created_at from file: %d chunks updated (%zu docs)", updated, docDateMap.size());
    }
}

void QueryFilterResolver::patchSourceTypeFromPrefix() {
    int updated = obxManager_.bulkUpdateSourceTypeFromPrefix();
    LOGI("Patched source_type from prefix: %d chunks updated", updated);
}

} // namespace ecovector
