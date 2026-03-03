#ifndef ECOVECTOR_RETRIEVER_ENSEMBLE_RETRIEVER_H
#define ECOVECTOR_RETRIEVER_ENSEMBLE_RETRIEVER_H

#include "IRetriever.h"
#include <vector>
#include <string>
#include <future>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <memory>
#include <cstdint>

namespace ecovector {

class Embedder;
class KiwiTokenizer;

struct WeightedResults {
    std::vector<ChunkSearchResult> results;
    float weight;
    bool isDistance = false;  // true: lower = more similar (L2), needs inversion
};

struct RetrieverConfig {
    IRetriever* retriever;
    float weight;
};

enum class FusionMethod {
    RSF,   // Relative Score Fusion (min-max normalization + weighted sum)
    RRF    // Reciprocal Rank Fusion (rank-based)
};

// Lightweight persistent thread pool to avoid per-query thread creation overhead
class ThreadPool {
public:
    explicit ThreadPool(size_t numThreads);
    ~ThreadPool();

    template<typename F>
    std::future<std::invoke_result_t<F>> submit(F&& task) {
        using ReturnType = std::invoke_result_t<F>;
        auto packaged = std::make_shared<std::packaged_task<ReturnType()>>(std::forward<F>(task));
        auto future = packaged->get_future();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            tasks_.push([packaged]() { (*packaged)(); });
        }
        cv_.notify_all();
        return future;
    }

private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool stop_ = false;
};

class EnsembleRetriever : public IRetriever {
public:
    struct Params : IRetriever::Params {
        FusionMethod fusionMethod = FusionMethod::RRF;
        float rrfK = 30.0f;        // RRF constant (only used when fusionMethod == RRF)
        bool parallel = false;     // sequential is faster when sub-retrievers are sub-ms
        uint32_t maxThreads = 4;   // max concurrent retriever threads
        // Per-sub-retriever params override (index matches configs order).
        // Empty = each sub-retriever uses its own defaults.
        // If provided, size must match configs.size().
        std::vector<std::shared_ptr<IRetriever::Params>> subRetrieverParams;
        Params() { topK = 15; }
    };

    EnsembleRetriever(std::vector<RetrieverConfig> configs,
                      Embedder* embedder, KiwiTokenizer* kiwiTokenizer);
    ~EnsembleRetriever() override;

    // Bring base class convenience overload into scope (name hiding prevention)
    using IRetriever::retrieve;

    // IRetriever (L2)
    std::vector<ChunkSearchResult> retrieve(
        const QueryBundle& query, const IRetriever::Params& overrideParams) override;
    const char* getName() const override { return "Ensemble"; }
    bool isReady() const override;
    const IRetriever::Params& getDefaultParams() const override { return params; }
    void warmup() override {
        for (auto& cfg : configs) {
            if (cfg.retriever) cfg.retriever->warmup();
        }
    }
    bool needsEmbedding() const override {
        for (const auto& cfg : configs) {
            if (cfg.retriever && cfg.retriever->needsEmbedding()) return true;
        }
        return false;
    }
    bool needsKiwiTokens() const override {
        for (const auto& cfg : configs) {
            if (cfg.retriever && cfg.retriever->needsKiwiTokens()) return true;
        }
        return false;
    }
    std::string getParamsSummary() const override {
        std::string s;
        s += (params.fusionMethod == FusionMethod::RRF) ? "RRF" : "RSF";
        if (params.fusionMethod == FusionMethod::RRF && params.rrfK != 30.0f)
            s += ",k=" + std::to_string(static_cast<int>(params.rrfK));
        s += ",top=" + std::to_string(params.topK);
        // Weights
        s += ",w=[";
        for (size_t i = 0; i < configs.size(); i++) {
            if (i > 0) s += ",";
            char buf[8]; std::snprintf(buf, sizeof(buf), "%.1f", configs[i].weight);
            s += buf;
        }
        s += "]";
        // Sub-retriever params (compact)
        for (const auto& cfg : configs) {
            s += " | ";
            s += cfg.retriever->getName();
            auto sub = cfg.retriever->getParamsSummary();
            if (!sub.empty()) s += "(" + sub + ")";
        }
        return s;
    }

    // L1: stateless fusion (inputs are moved-from during fusion)
    std::vector<ChunkSearchResult> fuse(
        std::vector<WeightedResults>& inputs,
        const Params* overrideParams = nullptr);

    // L3: convenience
    std::vector<ChunkSearchResult> retrieve(
        const std::string& queryText,
        const Params* overrideParams = nullptr);

    Params params;
    std::vector<RetrieverConfig> configs;

private:
    Embedder* embedder_;
    KiwiTokenizer* kiwiTokenizer_;
    std::unique_ptr<ThreadPool> threadPool_;

    std::vector<ChunkSearchResult> fuseRRF(
        std::vector<WeightedResults>& inputs, uint32_t limit, float rrfK = 0);
    std::vector<ChunkSearchResult> fuseByScore(
        std::vector<WeightedResults>& inputs, uint32_t limit);
};

} // namespace ecovector

#endif // ECOVECTOR_RETRIEVER_ENSEMBLE_RETRIEVER_H
