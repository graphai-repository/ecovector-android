#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include "ObxManager.h"
#include "IRetriever.h"
#include "GroundTruthResolver.h"
#include "MetricsEvaluator.h"

namespace ecovector {

class Tokenizer;
class BenchmarkObxManager;

/**
 * Benchmark orchestrator.
 * Separated from core EcoVectorStore to keep library lean.
 * Receives ObxManager and Tokenizer by reference (non-owning).
 */
class BenchmarkRunner {
public:
    BenchmarkRunner(ObxManager& obxManager, Tokenizer& tokenizer,
                    BenchmarkObxManager* benchmarkObxManager = nullptr);

    void registerRetriever(IRetriever* retriever);
    void clearRetrievers();

    /**
     * Run benchmark on all registered IRetriever instances.
     * @param split  "valid"/"test"/"" — empty = all queries
     */
    std::string runRegisteredRetrievers(uint32_t topK, const std::string& dbPath,
                                        const std::string& filterPath = "",
                                        const std::string& split = "");

    /**
     * Merge per-retriever detail JSONL files into a single per-query JSONL.
     * Each output line = 1 query with all retriever results merged.
     */
    std::string mergeDetailFiles(
        const std::vector<std::string>& methodNames,
        const std::vector<std::string>& detailPaths,
        const std::string& outputPath,
        const std::string& split = "");

private:
    ObxManager& obxManager_;
    Tokenizer& tokenizer_;
    BenchmarkObxManager* benchmarkObxManager_ = nullptr;
    std::vector<IRetriever*> registeredRetrievers_;

    static std::string utf8Truncate(const std::string& str, size_t maxBytes);
    static std::string sourceTypeToString(SourceType st);
    static std::string epochMsToIso8601(int64_t epochMs);
};

} // namespace ecovector
