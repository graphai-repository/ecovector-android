#include "BenchmarkRunner.h"
#include "BenchmarkObxManager.h"
#include "QueryFilterResolver.h"
#include "Tokenizer.h"
#include <json.hpp>
#include <android/log.h>
#include <sstream>
#include <iomanip>
#include <unordered_map>
#include <chrono>
#include <ctime>
#include <fstream>

#define LOG_TAG "BenchmarkRunner"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)

namespace ecovector {

BenchmarkRunner::BenchmarkRunner(ObxManager& obxManager, Tokenizer& tokenizer,
                                 BenchmarkObxManager* benchmarkObxManager)
    : obxManager_(obxManager), tokenizer_(tokenizer),
      benchmarkObxManager_(benchmarkObxManager) {}

std::string BenchmarkRunner::utf8Truncate(const std::string& str, size_t maxBytes) {
    if (str.size() <= maxBytes) return str;
    size_t pos = maxBytes;
    while (pos > 0 && (static_cast<unsigned char>(str[pos]) & 0xC0) == 0x80) {
        --pos;
    }
    return str.substr(0, pos);
}

std::string BenchmarkRunner::sourceTypeToString(SourceType st) {
    switch (st) {
        case SourceType::CALL:  return "call";
        case SourceType::SMS:   return "sms";
        case SourceType::MMS:   return "mms";
        case SourceType::IMAGE: return "image";
        case SourceType::FILE:  return "file";
        default:                return "unknown";
    }
}

std::string BenchmarkRunner::epochMsToIso8601(int64_t epochMs) {
    if (epochMs <= 0) return {};
    time_t sec = static_cast<time_t>(epochMs / 1000);
    sec += 9 * 3600; // KST = UTC+9
    struct tm tm;
    gmtime_r(&sec, &tm);
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%04d-%02d-%02dT%02d:%02d:%02d+09:00",
                  tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
                  tm.tm_hour, tm.tm_min, tm.tm_sec);
    return buf;
}

// ============================================================================
// Unified IRetriever benchmarking
// ============================================================================

void BenchmarkRunner::registerRetriever(IRetriever* retriever) {
    if (retriever) {
        registeredRetrievers_.push_back(retriever);
        LOGI("Registered retriever: %s (total: %zu)", retriever->getName(),
             registeredRetrievers_.size());
    }
}

void BenchmarkRunner::clearRetrievers() {
    registeredRetrievers_.clear();
}

std::string BenchmarkRunner::runRegisteredRetrievers(uint32_t topK,
                                                      const std::string& dbPath,
                                                      const std::string& filterPath,
                                                      const std::string& split) {
    LOGI("========================================");
    LOGI("runRegisteredRetrievers() - %zu retrievers, topK=%u",
         registeredRetrievers_.size(), topK);
    LOGI("========================================");

    if (registeredRetrievers_.empty()) {
        LOGE("No retrievers registered");
        return "{}";
    }

    try {
        // Load queries from BenchmarkObxManager
        std::vector<QueryData> queries;
        if (benchmarkObxManager_) {
            if (split.empty()) {
                queries = benchmarkObxManager_->getAllQueries(false);
            } else {
                queries = benchmarkObxManager_->getQueriesBySplit(split, false);
            }
        } else {
            LOGE("BenchmarkObxManager not set — cannot load queries");
        }
        uint32_t totalQueries = static_cast<uint32_t>(queries.size());
        LOGI("Total queries: %u (split=%s)", totalQueries, split.c_str());
        if (totalQueries == 0) { LOGE("No queries found"); return "{}"; }

        auto totalChunks = obxManager_.getChunkCount();
        LOGI("Total chunks: %u", totalChunks);

        // Load ground truth from BenchmarkObxManager
        if (!benchmarkObxManager_) {
            LOGE("BenchmarkObxManager not set — cannot load ground truth");
            return "{}";
        }
        auto gt = std::make_unique<GroundTruthResolver>(obxManager_, *benchmarkObxManager_);
        gt->load();

        // Load query filters ("NONE" = explicitly disabled, "" = auto-discover)
        QueryFilterResolver filterResolver(obxManager_);
        if (filterPath != "NONE") {
            filterResolver.patchSourceTypeFromPrefix();
            std::string resolvedFilterPath = filterPath;
            if (resolvedFilterPath.empty()) {
                std::string parentDir = dbPath;
                size_t slash = parentDir.find_last_of("/\\");
                if (slash != std::string::npos) parentDir = parentDir.substr(0, slash);
                resolvedFilterPath = parentDir + "/valid_filters.jsonl";
            }
            filterResolver.load(resolvedFilterPath);
        } else {
            LOGI("[FILTER] Filters explicitly disabled (filterPath=NONE)");
        }

        // Warmup all retrievers
        for (auto* retriever : registeredRetrievers_) {
            retriever->warmup();
        }

        // Build doc lookup maps from GT target docs only (avoids loading all 12K docs)
        std::unordered_map<uint64_t, DocData> docCache;
        std::unordered_map<uint64_t, std::string> docIdToExternalId;
        for (const auto& [queryExtId, targetDocIds] : gt->getMap()) {
            for (uint64_t docId : targetDocIds) {
                if (docIdToExternalId.count(docId) == 0) {
                    auto doc = obxManager_.getDocumentById(docId);
                    if (doc) {
                        docIdToExternalId[docId] = doc->externalId;
                        docCache.emplace(docId, std::move(*doc));
                    }
                }
            }
        }
        LOGI("Loaded %zu GT-target docs (instead of all %u)",
             docCache.size(), obxManager_.getDocumentCount());

        // Build QueryBundles (unfiltered)
        std::vector<QueryBundle> bundles;
        bundles.reserve(totalQueries);
        for (const auto& q : queries) {
            QueryBundle bundle;
            bundle.rawText = q.content;
            bundle.embedding = q.vector;
            bundle.kiwiTokens = q.kiwiTokens;
            bundles.push_back(std::move(bundle));
        }

        // Build filtered QueryBundles
        bool hasFilters = filterResolver.hasFilters();
        LOGI("[FILTER_DEBUG] hasFilters=%d, totalQueries=%d", hasFilters, totalQueries);
        std::vector<QueryBundle> filteredBundles;
        int filterAssigned = 0, filterEmpty = 0, filterNull = 0, noDocMapping = 0;
        if (hasFilters) {
            filteredBundles.reserve(totalQueries);
            for (uint32_t i = 0; i < totalQueries; i++) {
                QueryBundle bundle;
                bundle.rawText = queries[i].content;
                bundle.embedding = queries[i].vector;
                bundle.kiwiTokens = queries[i].kiwiTokens;

                const auto& targetIds = gt->getTargetDocIds(queries[i].externalId);
                if (!targetIds.empty()) {
                    auto docExtIt = docIdToExternalId.find(targetIds[0]);
                    if (docExtIt != docIdToExternalId.end()) {
                        auto* filter = filterResolver.getFilterForDoc(docExtIt->second);
                        if (filter && !filter->empty()) {
                            bundle.filterChunkIds = filter;
                            filterAssigned++;
                            if (i < 5) {
                                LOGI("[FILTER_DEBUG] Q[%d] extId=%s -> docExt=%s -> filterSize=%zu",
                                    i, queries[i].externalId.c_str(), docExtIt->second.c_str(),
                                    filter->size());
                            }
                        } else {
                            filterEmpty++;
                        }
                    } else {
                        noDocMapping++;
                    }
                } else {
                    noDocMapping++;
                }
                filteredBundles.push_back(std::move(bundle));
            }
        }
        LOGI("[FILTER_DEBUG] assigned=%d, empty=%d, noMapping=%d, total=%d",
            filterAssigned, filterEmpty, noDocMapping, totalQueries);

        MetricsEvaluator evaluator(*gt);

        // Determine details output directory (next to benchmark DB)
        std::string detailsDir = dbPath;
        {
            size_t slash = detailsDir.find_last_of("/\\");
            if (slash != std::string::npos) detailsDir = detailsDir.substr(0, slash);
        }

        // Per-query detail streaming helper — writes one JSONL line per query
        auto streamQueryDetails = [&](
            const std::string& detailsPath,
            const std::vector<QueryData>& qs,
            const std::vector<QueryResult>& qResults) {

            std::ofstream ofs(detailsPath, std::ios::trunc);
            if (!ofs.is_open()) {
                LOGE("Cannot open details file: %s", detailsPath.c_str());
                return;
            }

            for (uint32_t i = 0; i < qs.size(); i++) {
                if (qs[i].vector.empty()) continue;
                const auto& qr = qResults[i];

                nlohmann::json detail;
                detail["query"] = qs[i].content;
                detail["is_hit"] = qr.isHit;
                detail["recall"] = qr.recall;
                detail["latency_ms"] = qr.latencyMs;

                // filter (per-query)
                const auto& targetIds = gt->getTargetDocIds(qs[i].externalId);
                bool hasQueryFilter = false;
                if (!targetIds.empty()) {
                    auto docExtIt = docIdToExternalId.find(targetIds[0]);
                    if (docExtIt != docIdToExternalId.end()) {
                        auto* rawFilter = filterResolver.getRawFilterForDoc(docExtIt->second);
                        if (rawFilter) {
                            nlohmann::json filterJson;
                            if (!rawFilter->sourceTypes.empty()) {
                                filterJson["source_type"] = rawFilter->sourceTypes;
                            }
                            if (!rawFilter->timeGte.empty() || !rawFilter->timeLte.empty()) {
                                nlohmann::json timeRange;
                                if (!rawFilter->timeGte.empty()) timeRange["gte"] = rawFilter->timeGte;
                                if (!rawFilter->timeLte.empty()) timeRange["lte"] = rawFilter->timeLte;
                                filterJson["created_at"] = timeRange;
                            }
                            detail["filter"] = filterJson;
                            hasQueryFilter = true;
                        }
                    }
                }
                if (!hasQueryFilter) {
                    detail["filter"] = nullptr;
                }

                // ground_truth
                nlohmann::json gtArr = nlohmann::json::array();
                for (uint64_t docId : targetIds) {
                    nlohmann::json gtDoc;
                    gtDoc["id"] = docId;
                    auto docIt = docCache.find(docId);
                    if (docIt != docCache.end()) {
                        const auto& doc = docIt->second;
                        gtDoc["externalId"] = doc.externalId;
                        gtDoc["sourceType"] = sourceTypeToString(doc.sourceType);
                        auto iso = epochMsToIso8601(doc.createdAt);
                        gtDoc["createdAt"] = iso.empty() ? nlohmann::json(nullptr) : nlohmann::json(iso);
                        gtDoc["text"] = utf8Truncate(doc.content, 300);
                    }
                    gtArr.push_back(std::move(gtDoc));
                }
                detail["ground_truth"] = std::move(gtArr);

                // retrieved_docs
                nlohmann::json retArr = nlohmann::json::array();
                for (const auto& chunk : qr.retrievedChunks) {
                    nlohmann::json retDoc;
                    retDoc["id"] = chunk.chunk.documentId;
                    retDoc["externalId"] = chunk.chunk.documentExternalId;
                    retDoc["sourceType"] = sourceTypeToString(chunk.chunk.sourceType);
                    auto iso = epochMsToIso8601(chunk.chunk.createdAt);
                    retDoc["createdAt"] = iso.empty() ? nlohmann::json(nullptr) : nlohmann::json(iso);
                    retDoc["is_correct"] = std::find(targetIds.begin(), targetIds.end(),
                        chunk.chunk.documentId) != targetIds.end();
                    retDoc["chunk"] = utf8Truncate(chunk.chunk.content, 300);
                    retArr.push_back(std::move(retDoc));
                }
                detail["retrieved_docs"] = std::move(retArr);

                // Write as single line (no pretty-print) and flush
                ofs << detail.dump(-1) << '\n';
            }
            ofs.flush();
            LOGI("Streamed %u query details to %s", totalQueries, detailsPath.c_str());
        };

        // Run benchmarks for each retriever
        nlohmann::json result;
        nlohmann::json retrieverResults = nlohmann::json::array();

        for (auto* retriever : registeredRetrievers_) {
            if (!retriever->isReady()) {
                LOGW("Retriever '%s' not ready, skipping", retriever->getName());
                continue;
            }

            std::string methodName = retriever->getName();
            auto paramSummary = retriever->getParamsSummary();
            if (!paramSummary.empty()) methodName += "(" + paramSummary + ")";

            auto& useBundles = hasFilters ? filteredBundles : bundles;
            constexpr uint32_t LARGE_K = 100;
            constexpr uint32_t DETAIL_TOP_K = LARGE_K;
            auto evalResult = evaluator.evaluate(methodName, queries, topK,
                [&useBundles, retriever, &gt, &queries, topK](size_t queryIdx) {
                    // GT > topK → retrieve with LARGE_K for near-miss analysis
                    const auto& targetIds = gt->getTargetDocIds(queries[queryIdx].externalId);
                    if (targetIds.size() > topK) {
                        IRetriever::Params largeParams;
                        largeParams.topK = LARGE_K;
                        return retriever->retrieve(useBundles[queryIdx], largeParams);
                    }
                    return retriever->retrieve(useBundles[queryIdx]);
                }, DETAIL_TOP_K);

            nlohmann::json methodJson;
            methodJson["method"] = methodName;
            methodJson["Hit@5"] = evalResult.metrics.hitRate;
            methodJson["Recall@5"] = evalResult.metrics.recall;
            methodJson["avg_latency_ms"] = evalResult.metrics.avgLatencyMs;
            methodJson["total_latency_sec"] = evalResult.metrics.totalLatencySec;
            methodJson["total_queries"] = evalResult.metrics.totalQueries;
            methodJson["correct_results"] = evalResult.metrics.correctResults;

            // Stream per-query details to JSONL file (O(1) memory)
            std::string detailsFile = detailsDir + "/details_" + methodName + ".jsonl";
            streamQueryDetails(detailsFile, queries, evalResult.queryResults);
            methodJson["details_file"] = detailsFile;

            retrieverResults.push_back(std::move(methodJson));
        }

        // Metadata
        auto now = std::chrono::system_clock::now();
        auto now_c = std::chrono::system_clock::to_time_t(now);
        char timeBuf[100];
        std::strftime(timeBuf, sizeof(timeBuf), "%Y-%m-%d %H:%M:%S", std::localtime(&now_c));
        result["timestamp"] = timeBuf;

        result["doc_count"] = obxManager_.getDocumentCount();
        result["chunk_count"] = totalChunks;
        result["query_count"] = totalQueries;
        result["results"] = retrieverResults;

        // Summary table
        {
            std::ostringstream oss;
            oss << std::fixed;
            oss << "\n========== Retriever Benchmark Summary ==========\n";
            oss << "| Method              | Hit@5  | Recall@5(50) | Avg Latency | Total Latency |\n";
            oss << "|---------------------|--------|--------------|-------------|---------------|\n";
            for (const auto& mr : retrieverResults) {
                std::string name = mr["method"].get<std::string>();
                oss << "| " << std::left << std::setw(19) << name << " | "
                    << std::right << std::setprecision(2)
                    << std::setw(6) << mr["Hit@5"].get<double>() << "% | "
                    << std::setw(10) << mr["Recall@5"].get<double>() << "% | "
                    << std::setw(8) << std::setprecision(3) << mr["avg_latency_ms"].get<double>() << " ms | "
                    << std::setw(10) << std::setprecision(3) << mr["total_latency_sec"].get<double>() << " s |\n";
            }
            oss << "=================================================\n";
            std::string summaryStr = oss.str();
            LOGI("%s", summaryStr.c_str());
            result["summary"] = summaryStr;
        }

        return result.dump(2);

    } catch (const std::exception& e) {
        LOGE("Exception in runRegisteredRetrievers: %s", e.what());
        return "{}";
    }
}

// ============================================================================
// Merge per-retriever detail files into single per-query JSONL
// ============================================================================

std::string BenchmarkRunner::mergeDetailFiles(
    const std::vector<std::string>& methodNames,
    const std::vector<std::string>& detailPaths,
    const std::string& outputPath,
    const std::string& split) {

    LOGI("mergeDetailFiles: %zu methods -> %s", methodNames.size(), outputPath.c_str());

    if (methodNames.size() != detailPaths.size() || methodNames.empty()) {
        LOGE("mergeDetailFiles: invalid args (methods=%zu, paths=%zu)",
             methodNames.size(), detailPaths.size());
        return "error: invalid arguments";
    }

    // Load queries with vectors (needed to identify which queries were evaluated)
    std::vector<QueryData> queries;
    if (benchmarkObxManager_) {
        queries = split.empty()
            ? benchmarkObxManager_->getAllQueries(false)
            : benchmarkObxManager_->getQueriesBySplit(split, false);
    }
    if (queries.empty()) {
        LOGE("mergeDetailFiles: no queries");
        return "error: no queries";
    }
    LOGI("mergeDetailFiles: %zu queries loaded", queries.size());

    // Load GT
    auto gt = std::make_unique<GroundTruthResolver>(obxManager_, *benchmarkObxManager_);
    gt->load();

    // Preload GT docs + first chunks
    std::unordered_map<uint64_t, DocData> docCache;
    std::unordered_map<uint64_t, ChunkData> chunkCache;
    for (const auto& [queryExtId, targetDocIds] : gt->getMap()) {
        for (uint64_t docId : targetDocIds) {
            if (docCache.count(docId) == 0) {
                auto doc = obxManager_.getDocumentById(docId);
                if (doc) docCache.emplace(docId, std::move(*doc));
                auto chunk = obxManager_.getFirstChunkByDocumentId(docId, true);
                if (chunk) chunkCache.emplace(docId, std::move(*chunk));
            }
        }
    }
    LOGI("mergeDetailFiles: preloaded %zu GT docs, %zu chunks",
         docCache.size(), chunkCache.size());

    // Open all intermediate files
    std::vector<std::ifstream> inputs(detailPaths.size());
    for (size_t m = 0; m < detailPaths.size(); m++) {
        inputs[m].open(detailPaths[m]);
        if (!inputs[m].is_open()) {
            LOGE("mergeDetailFiles: cannot open %s", detailPaths[m].c_str());
            return "error: cannot open " + detailPaths[m];
        }
    }

    // Open output
    std::ofstream ofs(outputPath, std::ios::trunc);
    if (!ofs.is_open()) {
        LOGE("mergeDetailFiles: cannot open output %s", outputPath.c_str());
        return "error: cannot open output";
    }

    // Per-query merge loop
    uint32_t writtenQueries = 0;
    for (const auto& query : queries) {
        if (query.vector.empty()) continue;

        // Read one line from each intermediate file
        std::vector<nlohmann::json> perMethodDetails(methodNames.size());
        bool allRead = true;
        for (size_t m = 0; m < methodNames.size(); m++) {
            std::string line;
            if (std::getline(inputs[m], line) && !line.empty()) {
                try {
                    perMethodDetails[m] = nlohmann::json::parse(line);
                } catch (...) {
                    LOGW("mergeDetailFiles: parse error method=%s query=%s",
                         methodNames[m].c_str(), query.externalId.c_str());
                    allRead = false;
                }
            } else {
                allRead = false;
            }
        }

        // Build merged JSON for this query
        nlohmann::json merged;
        merged["query_id"] = query.externalId;
        merged["query"] = query.content;
        merged["refined_query"] = query.refinedQuery;

        // kiwi tokens
        merged["query_kiwi_tokens"] = query.kiwiTokens;

        // filter — take from first method's detail (same for all)
        if (!perMethodDetails.empty() && perMethodDetails[0].contains("filter")) {
            merged["filter"] = perMethodDetails[0]["filter"];
        } else {
            merged["filter"] = nullptr;
        }

        // Ground truth with doc/chunk content
        const auto& targetDocIds = gt->getTargetDocIds(query.externalId);
        nlohmann::json gtArr = nlohmann::json::array();
        for (uint64_t docId : targetDocIds) {
            nlohmann::json gtDoc;
            auto docIt = docCache.find(docId);
            if (docIt != docCache.end()) {
                const auto& doc = docIt->second;
                gtDoc["doc_id"] = doc.externalId;
                gtDoc["source_type"] = sourceTypeToString(doc.sourceType);
                gtDoc["created_at"] = epochMsToIso8601(doc.createdAt);
                gtDoc["content"] = utf8Truncate(doc.content, 300);
            } else {
                gtDoc["doc_id"] = docId;
            }
            auto chunkIt = chunkCache.find(docId);
            if (chunkIt != chunkCache.end()) {
                const auto& chunk = chunkIt->second;
                gtDoc["chunk_content"] = utf8Truncate(chunk.content, 300);
                gtDoc["chunk_kiwi_tokens"] = chunk.kiwiTokens;
            }
            gtArr.push_back(std::move(gtDoc));
        }
        merged["ground_truth"] = std::move(gtArr);

        // Per-retriever results
        nlohmann::json results = nlohmann::json::object();
        for (size_t m = 0; m < methodNames.size(); m++) {
            nlohmann::json methodResult;
            const auto& detail = perMethodDetails[m];
            methodResult["is_hit"] = detail.value("is_hit", false);
            methodResult["recall"] = detail.value("recall", 0.0);
            methodResult["latency_ms"] = detail.value("latency_ms", 0.0);

            // Compact retrieved docs (★ prefix for correct matches)
            nlohmann::json retArr = nlohmann::json::array();
            if (detail.contains("retrieved_docs")) {
                for (const auto& rd : detail["retrieved_docs"]) {
                    nlohmann::json retDoc;
                    bool correct = rd.value("is_correct", false);
                    std::string docId = rd.value("externalId", "");
                    std::string chunk = rd.value("chunk", "");
                    retDoc["doc_id"] = correct ? ("★" + docId) : docId;
                    retDoc["chunk_content"] = correct ? ("★ " + chunk) : chunk;
                    retDoc["source_type"] = rd.value("sourceType", "");
                    retDoc["is_correct"] = correct;
                    retArr.push_back(std::move(retDoc));
                }
            }
            methodResult["retrieved"] = std::move(retArr);
            results[methodNames[m]] = std::move(methodResult);
        }
        merged["results"] = std::move(results);

        ofs << merged.dump(-1) << '\n';
        writtenQueries++;
    }
    ofs.flush();
    ofs.close();

    // Close inputs
    for (auto& f : inputs) f.close();

    std::string summary = "merged " + std::to_string(writtenQueries) + " queries, "
        + std::to_string(methodNames.size()) + " methods -> " + outputPath;
    LOGI("mergeDetailFiles: %s", summary.c_str());
    return summary;
}

} // namespace ecovector
