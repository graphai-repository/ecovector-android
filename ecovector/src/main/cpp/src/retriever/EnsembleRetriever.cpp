#define LOG_TAG "EnsembleRetriever"
#include "../common/Logging.h"
#include "EnsembleRetriever.h"
#include "../object_box/ObxManager.h"
#include "../embedder/Embedder.h"
#include "../kiwi/KiwiTokenizer.h"
#include "../kiwi/KiwiHashUtil.h"

#include <unordered_map>
#include <algorithm>
#include <chrono>
#include <limits>
#include <thread>
#include <stdexcept>

namespace ecovector {

// ============================================================================
// ThreadPool implementation
// ============================================================================

ThreadPool::ThreadPool(size_t numThreads) {
    workers_.reserve(numThreads);
    for (size_t i = 0; i < numThreads; ++i) {
        workers_.emplace_back([this]() {
            while (true) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(mutex_);
                    cv_.wait(lock, [this]() { return stop_ || !tasks_.empty(); });
                    if (stop_ && tasks_.empty()) return;
                    task = std::move(tasks_.front());
                    tasks_.pop();
                }
                task();
            }
        });
    }
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
    }
    cv_.notify_all();
    for (auto& w : workers_) {
        if (w.joinable()) w.join();
    }
}

// ============================================================================
// EnsembleRetriever
// ============================================================================

EnsembleRetriever::EnsembleRetriever(std::vector<RetrieverConfig> configs,
                                     Embedder* embedder, KiwiTokenizer* kiwiTokenizer)
    : configs(std::move(configs))
    , embedder_(embedder)
    , kiwiTokenizer_(kiwiTokenizer) {
    LOGD("EnsembleRetriever constructed with %zu retrievers (embedder=%p, kiwiTokenizer=%p)",
         this->configs.size(), embedder, kiwiTokenizer);
}

EnsembleRetriever::~EnsembleRetriever() {
    // ThreadPool must be destroyed before configs (joins worker threads)
    threadPool_.reset();
    LOGD("EnsembleRetriever destructor");
}

bool EnsembleRetriever::isReady() const {
    if (configs.empty()) return false;
    for (const auto& cfg : configs) {
        if (!cfg.retriever || !cfg.retriever->isReady()) return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// L2: IRetriever interface — dispatch to each sub-retriever, then fuse
// ---------------------------------------------------------------------------

std::vector<ChunkSearchResult> EnsembleRetriever::retrieve(
    const QueryBundle& query, const IRetriever::Params& overrideParams) {

    auto startTime = std::chrono::high_resolution_clock::now();
    const auto* ep = dynamic_cast<const Params*>(&overrideParams);
    const uint32_t topK = overrideParams.topK;
    const FusionMethod fusionMethodVal = ep ? ep->fusionMethod : params.fusionMethod;
    const float rrfKVal = ep ? ep->rrfK : params.rrfK;
    const bool parallelVal = ep ? ep->parallel : params.parallel;
    const uint32_t maxThreadsVal = ep ? ep->maxThreads : params.maxThreads;
    // Sub-retriever params: from override if provided, else from instance defaults
    const auto& subParams = (ep && !ep->subRetrieverParams.empty())
        ? ep->subRetrieverParams : params.subRetrieverParams;
    const bool hasSubParams = !subParams.empty();

    if (configs.empty()) {
        LOGE("No retrievers configured");
        return {};
    }

    // Validate subRetrieverParams size
    if (hasSubParams && subParams.size() != configs.size()) {
        LOGE("subRetrieverParams size (%zu) != configs size (%zu)",
             subParams.size(), configs.size());
        throw std::invalid_argument(
            "subRetrieverParams.size() must match configs.size()");
    }

    // Helper: call sub-retriever with per-sub params override.
    // subRetrieverParams[i] != nullptr → use that params
    // subRetrieverParams[i] == nullptr → use sub-retriever's default
    // subRetrieverParams empty → forward ensemble's overrideParams to sub-retriever
    auto callSubRetriever = [&](size_t idx, const QueryBundle& q) {
        if (hasSubParams) {
            if (subParams[idx]) {
                return configs[idx].retriever->retrieve(q, *subParams[idx]);
            }
            // nullptr = sub-retriever uses its own default
            return configs[idx].retriever->retrieve(q);
        }
        // No subRetrieverParams at all → forward ensemble's overrideParams
        return configs[idx].retriever->retrieve(q, overrideParams);
    };

    // Collect results from each sub-retriever
    std::vector<WeightedResults> allResults;
    allResults.reserve(configs.size());

    auto mainTid = std::hash<std::thread::id>{}(std::this_thread::get_id()) % 10000;

    if (parallelVal && configs.size() > 1) {
        // Lazy-init thread pool (created once, reused across queries)
        if (!threadPool_) {
            size_t poolSize = std::min(static_cast<size_t>(maxThreadsVal),
                                       configs.size());
            threadPool_ = std::make_unique<ThreadPool>(poolSize);
            LOGD("ThreadPool created with %zu threads", poolSize);
        }

        struct AsyncResult {
            std::future<std::vector<ChunkSearchResult>> future;
            float weight;
            bool isDistance;
            const char* name;
        };

        std::vector<AsyncResult> asyncResults;
        asyncResults.reserve(configs.size());

        auto submitTime = std::chrono::high_resolution_clock::now();

        for (size_t ci = 0; ci < configs.size(); ci++) {
            const auto& cfg = configs[ci];
            if (!cfg.retriever || !cfg.retriever->isReady()) {
                LOGW("Skipping retriever '%s' (not ready)",
                     cfg.retriever ? cfg.retriever->getName() : "null");
                continue;
            }
            asyncResults.push_back({
                threadPool_->submit(
                    [&callSubRetriever, ci, &query, &cfg, submitTime]() {
                        auto tid = std::hash<std::thread::id>{}(std::this_thread::get_id()) % 10000;
                        auto dispatchDelay = std::chrono::duration_cast<std::chrono::microseconds>(
                            std::chrono::high_resolution_clock::now() - submitTime).count();
                        LOGD("[TIMING] %s thread=%zu dispatch_delay=%lld us",
                             cfg.retriever->getName(), tid, (long long)dispatchDelay);
                        return callSubRetriever(ci, query);
                    }),
                cfg.weight,
                cfg.retriever->returnsDistance(),
                cfg.retriever->getName()
            });
        }

        for (size_t i = 0; i < asyncResults.size(); i++) {
            auto& ar = asyncResults[i];
            auto waitStart = std::chrono::high_resolution_clock::now();
            auto results = ar.future.get();
            auto waitEnd = std::chrono::high_resolution_clock::now();
            auto waitUs = std::chrono::duration_cast<std::chrono::microseconds>(waitEnd - waitStart).count();
            LOGD("[TIMING] Ensemble wait %s: %lld us (%zu results)",
                 ar.name, (long long)waitUs, results.size());
            allResults.push_back({std::move(results), ar.weight, ar.isDistance});
        }
    } else {
        // Sequential
        for (size_t ci = 0; ci < configs.size(); ci++) {
            const auto& cfg = configs[ci];
            if (!cfg.retriever || !cfg.retriever->isReady()) {
                LOGW("Skipping retriever '%s' (not ready)",
                     cfg.retriever ? cfg.retriever->getName() : "null");
                continue;
            }

            auto t0 = std::chrono::high_resolution_clock::now();
            auto results = callSubRetriever(ci, query);
            auto t1 = std::chrono::high_resolution_clock::now();
            auto us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
            LOGD("[TIMING] Ensemble seq %s: %lld us (%zu results, topK=%u)",
                 cfg.retriever->getName(), (long long)us, results.size(),
                 cfg.retriever->getTopK());
            allResults.push_back({std::move(results), cfg.weight, cfg.retriever->returnsDistance()});
        }
    }

    if (allResults.empty()) {
        LOGE("All retrievers failed or returned empty");
        return {};
    }

    // If only one retriever produced results, return directly (no fusion needed)
    if (allResults.size() == 1) {
        auto& single = allResults[0].results;
        if (single.size() > topK) {
            single.resize(topK);
        }
        return std::move(single);
    }

    // Fuse using configured method
    auto fusionStart = std::chrono::high_resolution_clock::now();
    std::vector<ChunkSearchResult> fused;
    if (fusionMethodVal == FusionMethod::RRF) {
        fused = fuseRRF(allResults, topK, rrfKVal);
    } else {
        fused = fuseByScore(allResults, topK);
    }
    auto fusionEnd = std::chrono::high_resolution_clock::now();
    auto fusionUs = std::chrono::duration_cast<std::chrono::microseconds>(fusionEnd - fusionStart).count();

    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
        endTime - startTime);

    LOGD("[TIMING] Ensemble total=%lld us, fusion=%lld us, mode=%s, thread=%zu",
         (long long)duration.count(), (long long)fusionUs,
         parallelVal ? "parallel" : "sequential", mainTid);

    LOGD("[BENCHMARK] EnsembleRetriever: topK=%u, inputs=%zu, fused=%zu, time=%lld us",
         topK, allResults.size(), fused.size(), (long long)duration.count());

    return fused;
}

// ---------------------------------------------------------------------------
// L1: stateless fusion
// ---------------------------------------------------------------------------

std::vector<ChunkSearchResult> EnsembleRetriever::fuse(
    std::vector<WeightedResults>& inputs,
    const Params* overrideParams) {

    const Params& p = overrideParams ? *overrideParams : params;
    uint32_t limit = p.topK;

    if (p.fusionMethod == FusionMethod::RRF) {
        return fuseRRF(inputs, limit, p.rrfK);
    } else {
        return fuseByScore(inputs, limit);
    }
}

// ---------------------------------------------------------------------------
// N-way RRF fusion (generalized from HybridRetriever::fuse)
// ---------------------------------------------------------------------------

std::vector<ChunkSearchResult> EnsembleRetriever::fuseRRF(
    std::vector<WeightedResults>& inputs, uint32_t limit, float rrfK) {

    if (rrfK <= 0) rrfK = params.rrfK;  // fallback to instance default

    std::vector<ChunkSearchResult> results;

    // Check if all inputs are empty
    bool allEmpty = true;
    for (const auto& wr : inputs) {
        if (!wr.results.empty()) { allEmpty = false; break; }
    }
    if (allEmpty) {
        LOGD("All input result lists are empty");
        return results;
    }

    // 1단계: 점수 + 원본 위치만 추적 (ChunkSearchResult 복사 없음)
    struct ScoreEntry { float score; size_t inputIdx; size_t resultIdx; };
    std::unordered_map<uint64_t, ScoreEntry> rrfScores;

    for (size_t i = 0; i < inputs.size(); ++i) {
        for (size_t rank = 0; rank < inputs[i].results.size(); ++rank) {
            uint64_t chunkId = inputs[i].results[rank].chunk.id;
            float contrib = inputs[i].weight / (rrfK + static_cast<float>(rank + 1));

            auto it = rrfScores.find(chunkId);
            if (it != rrfScores.end()) {
                it->second.score += contrib;
            } else {
                rrfScores[chunkId] = {contrib, i, rank};
            }
        }
    }

    // 2단계: partial_sort + move
    struct ScoredRef { float score; size_t inputIdx; size_t resultIdx; };
    std::vector<ScoredRef> entries;
    entries.reserve(rrfScores.size());
    for (auto& [id, e] : rrfScores) {
        entries.push_back({e.score, e.inputIdx, e.resultIdx});
    }

    size_t sortLimit = std::min(entries.size(), static_cast<size_t>(limit));
    std::partial_sort(entries.begin(), entries.begin() + sortLimit, entries.end(),
                      [](const ScoredRef& a, const ScoredRef& b) { return a.score > b.score; });

    results.reserve(sortLimit);
    for (size_t i = 0; i < sortLimit; ++i) {
        auto& e = entries[i];
        auto& src = inputs[e.inputIdx].results[e.resultIdx];
        ChunkSearchResult finalResult = std::move(src);
        finalResult.distance = e.score;
        results.push_back(std::move(finalResult));
    }

    LOGD("RRF N-way fusion: %zu inputs -> %zu fused results (k=%.1f)",
         inputs.size(), results.size(), rrfK);

    return results;
}

// ---------------------------------------------------------------------------
// N-way Score fusion (generalized from HybridRetriever::fuseByScore)
// Min-max normalization per input, then weighted summation
// ---------------------------------------------------------------------------

std::vector<ChunkSearchResult> EnsembleRetriever::fuseByScore(
    std::vector<WeightedResults>& inputs, uint32_t limit) {

    std::vector<ChunkSearchResult> results;

    // Check if all inputs are empty
    bool allEmpty = true;
    for (const auto& wr : inputs) {
        if (!wr.results.empty()) { allEmpty = false; break; }
    }
    if (allEmpty) {
        return results;
    }

    // 1단계: 점수 + 원본 위치만 추적 (ChunkSearchResult 복사 없음)
    struct ScoreEntry { float score; size_t inputIdx; size_t resultIdx; };
    std::unordered_map<uint64_t, ScoreEntry> combinedScores;

    for (size_t i = 0; i < inputs.size(); ++i) {
        const auto& wr = inputs[i];
        if (wr.results.empty()) continue;

        // Min-max normalization per input
        float minScore = std::numeric_limits<float>::max();
        float maxScore = std::numeric_limits<float>::lowest();
        for (const auto& r : wr.results) {
            minScore = std::min(minScore, r.distance);
            maxScore = std::max(maxScore, r.distance);
        }
        float range = maxScore - minScore;
        bool hasRange = (range > 1e-6f);

        for (size_t rank = 0; rank < wr.results.size(); ++rank) {
            const auto& r = wr.results[rank];
            float normalized;
            if (wr.isDistance) {
                normalized = hasRange ? (maxScore - r.distance) / range : 1.0f;
            } else {
                normalized = hasRange ? (r.distance - minScore) / range : 1.0f;
            }
            float weighted = wr.weight * normalized;

            auto it = combinedScores.find(r.chunk.id);
            if (it != combinedScores.end()) {
                it->second.score += weighted;
            } else {
                combinedScores[r.chunk.id] = {weighted, i, rank};
            }
        }
    }

    // 2단계: partial_sort + move
    struct ScoredRef { float score; size_t inputIdx; size_t resultIdx; };
    std::vector<ScoredRef> entries;
    entries.reserve(combinedScores.size());
    for (auto& [id, e] : combinedScores) {
        entries.push_back({e.score, e.inputIdx, e.resultIdx});
    }

    size_t sortLimit = std::min(entries.size(), static_cast<size_t>(limit));
    std::partial_sort(entries.begin(), entries.begin() + sortLimit, entries.end(),
                      [](const ScoredRef& a, const ScoredRef& b) { return a.score > b.score; });

    results.reserve(sortLimit);
    for (size_t i = 0; i < sortLimit; ++i) {
        auto& e = entries[i];
        auto& src = inputs[e.inputIdx].results[e.resultIdx];
        ChunkSearchResult finalResult = std::move(src);
        finalResult.distance = e.score;
        results.push_back(std::move(finalResult));
    }

    LOGD("Score N-way fusion: %zu inputs -> %zu fused results",
         inputs.size(), results.size());

    return results;
}

// ---------------------------------------------------------------------------
// L3: convenience — build QueryBundle from text, then retrieve
// ---------------------------------------------------------------------------

std::vector<ChunkSearchResult> EnsembleRetriever::retrieve(
    const std::string& queryText,
    const Params* overrideParams) {

    QueryBundle query;
    query.rawText = queryText;

    const bool doEmbed = needsEmbedding() && embedder_;
    const bool doKiwi  = needsKiwiTokens() && kiwiTokenizer_;

    if (doEmbed && doKiwi) {
        // Both needed: run in parallel
        if (threadPool_) {
            auto embFuture = threadPool_->submit(
                [this, &queryText]() { return embedder_->embed(queryText); });
            auto morphemes = kiwiTokenizer_->tokenize(queryText);
            if (!morphemes.empty()) {
                query.kiwiTokens = hashMorphemes(morphemes);
            }
            query.embedding = embFuture.get();
        } else {
            auto embFuture = std::async(std::launch::async,
                [this, &queryText]() { return embedder_->embed(queryText); });
            auto morphemes = kiwiTokenizer_->tokenize(queryText);
            if (!morphemes.empty()) {
                query.kiwiTokens = hashMorphemes(morphemes);
            }
            query.embedding = embFuture.get();
        }
    } else if (doEmbed) {
        query.embedding = embedder_->embed(queryText);
    } else if (doKiwi) {
        auto morphemes = kiwiTokenizer_->tokenize(queryText);
        if (!morphemes.empty()) {
            query.kiwiTokens = hashMorphemes(morphemes);
        }
    }

    const Params& p = overrideParams ? *overrideParams : params;
    return retrieve(query, p);
}

} // namespace ecovector
