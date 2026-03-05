#include <jni.h>
#include "JniArrayUtils.h"
#include "BenchmarkRunner.h"
#include "BenchmarkObxManager.h"
#include "EcoVectorStore.h"
#include "ObxManager.h"
#include "IRetriever.h"
#include "KiwiTokenizer.h"
#include "KiwiHashUtil.h"
#include <android/log.h>

using ecovector::hashMorphemes;

#define LOG_TAG "NativeBenchmarkRunner"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

// Defined in NativeEcoVectorStore.cpp (same .so)
extern "C" ecovector::EcoVectorStore* getGlobalEcoVectorStore();

// Global BenchmarkObxManager (separate DB for Query/GroundTruth)
static ecovector::BenchmarkObxManager* gBenchmarkObxManager = nullptr;

extern "C" {

// ============================================================================
// Benchmark DB Lifecycle
// ============================================================================

JNIEXPORT jboolean JNICALL
Java_io_graphai_ecovector_benchmark_NativeBenchmarkRunner_initBenchmarkDb(
        JNIEnv* env, jobject, jstring dbPath) {
    std::string path = jni_utils::toString(env, dbPath);
    if (gBenchmarkObxManager) {
        delete gBenchmarkObxManager;
        gBenchmarkObxManager = nullptr;
    }
    gBenchmarkObxManager = new ecovector::BenchmarkObxManager(path);
    if (!gBenchmarkObxManager->initialize()) {
        LOGE("Failed to initialize BenchmarkObxManager at %s", path.c_str());
        delete gBenchmarkObxManager;
        gBenchmarkObxManager = nullptr;
        return JNI_FALSE;
    }
    LOGI("BenchmarkObxManager initialized at %s", path.c_str());
    return JNI_TRUE;
}

JNIEXPORT void JNICALL
Java_io_graphai_ecovector_benchmark_NativeBenchmarkRunner_closeBenchmarkDb(
        JNIEnv*, jobject) {
    delete gBenchmarkObxManager;
    gBenchmarkObxManager = nullptr;
    LOGI("BenchmarkObxManager closed");
}

// ============================================================================
// Benchmark Data Management
// ============================================================================

JNIEXPORT jlong JNICALL
Java_io_graphai_ecovector_benchmark_NativeBenchmarkRunner_saveQueryRaw(
        JNIEnv* env, jobject,
        jstring externalId, jstring text, jstring refinedQueryJ,
        jintArray tokenIds, jfloatArray embedding,
        jintArray kiwiTokens,
        jlong createdAt, jstring targetTypes, jstring categories,
        jstring splitJ, jint evalTopK) {
    if (!gBenchmarkObxManager) {
        LOGE("saveQueryRaw: BenchmarkObxManager not initialized");
        return -1;
    }
    ecovector::QueryData q;
    q.externalId = jni_utils::toString(env, externalId);
    q.content = jni_utils::toString(env, text);
    q.refinedQuery = refinedQueryJ ? jni_utils::toString(env, refinedQueryJ) : "";
    q.tokenIds = jni_utils::toIntVector(env, tokenIds);
    q.vector = jni_utils::toFloatVector(env, embedding);
    q.kiwiTokens = jni_utils::toIntVector(env, kiwiTokens);
    q.createdAt = createdAt;
    q.targetTypes = jni_utils::toString(env, targetTypes);
    q.categories = jni_utils::toString(env, categories);
    q.split = splitJ ? jni_utils::toString(env, splitJ) : "";
    q.evalTopK = evalTopK;
    return static_cast<jlong>(gBenchmarkObxManager->insertQuery(q));
}

JNIEXPORT jlongArray JNICALL
Java_io_graphai_ecovector_benchmark_NativeBenchmarkRunner_saveGroundTruths(
        JNIEnv* env, jobject,
        jobjectArray queryIds, jobjectArray docIds) {
    if (!gBenchmarkObxManager) {
        LOGE("saveGroundTruths: BenchmarkObxManager not initialized");
        return env->NewLongArray(0);
    }
    auto queryIdVec = jni_utils::toStringVector(env, queryIds);
    auto docIdVec = jni_utils::toStringVector(env, docIds);

    std::vector<ecovector::GroundTruthData> entries;
    entries.reserve(queryIdVec.size());
    for (size_t i = 0; i < queryIdVec.size(); i++) {
        entries.push_back({0, queryIdVec[i], docIdVec[i]});
    }
    auto ids = gBenchmarkObxManager->insertAllGroundTruths(entries);
    std::vector<int64_t> jniIds(ids.begin(), ids.end());
    return jni_utils::toLongArray(env, jniIds);
}

JNIEXPORT jint JNICALL
Java_io_graphai_ecovector_benchmark_NativeBenchmarkRunner_getQueryCount(
        JNIEnv*, jobject) {
    if (gBenchmarkObxManager) {
        return static_cast<jint>(gBenchmarkObxManager->getQueryCount());
    }
    return 0;
}

JNIEXPORT jstring JNICALL
Java_io_graphai_ecovector_benchmark_NativeBenchmarkRunner_getQueriesJson(
        JNIEnv* env, jobject, jint offset, jint limit) {
    if (!gBenchmarkObxManager) return env->NewStringUTF("[]");
    return env->NewStringUTF(gBenchmarkObxManager->getQueriesJson(offset, limit).c_str());
}

JNIEXPORT jstring JNICALL
Java_io_graphai_ecovector_benchmark_NativeBenchmarkRunner_getQueryExternalIdsJson(
        JNIEnv* env, jobject) {
    if (!gBenchmarkObxManager) return env->NewStringUTF("[]");
    return env->NewStringUTF(gBenchmarkObxManager->getQueryExternalIdsJson().c_str());
}

// ============================================================================
// Unified IRetriever benchmarking
// ============================================================================

JNIEXPORT jstring JNICALL
Java_io_graphai_ecovector_benchmark_NativeBenchmarkRunner_runRegisteredRetrievers(
        JNIEnv* env, jobject, jlongArray retrieverHandles, jint topK,
        jstring filterPathJ, jstring splitJ) {
    auto* store = getGlobalEcoVectorStore();
    if (!store || !store->getObxManager() || !store->getTokenizer()) {
        LOGE("EcoVectorStore not initialized");
        return env->NewStringUTF("{}");
    }

    ecovector::BenchmarkRunner runner(
        *store->getObxManager(), *store->getTokenizer(), gBenchmarkObxManager);

    // Convert handle array to IRetriever pointers
    jsize len = env->GetArrayLength(retrieverHandles);
    jlong* handles = env->GetLongArrayElements(retrieverHandles, nullptr);
    for (jsize i = 0; i < len; i++) {
        auto* retriever = reinterpret_cast<ecovector::IRetriever*>(handles[i]);
        if (retriever) {
            runner.registerRetriever(retriever);
        }
    }
    env->ReleaseLongArrayElements(retrieverHandles, handles, 0);

    std::string filterPath = jni_utils::toString(env, filterPathJ);
    std::string split = splitJ ? jni_utils::toString(env, splitJ) : "";

    auto result = runner.runRegisteredRetrievers(
        static_cast<uint32_t>(topK), store->getDbPath(), filterPath, split);
    return env->NewStringUTF(result.c_str());
}

// === Pipeline Stage JNI Methods ===

JNIEXPORT void JNICALL
Java_io_graphai_ecovector_benchmark_NativeBenchmarkRunner_clearBenchmarkQueries(
        JNIEnv*, jobject) {
    if (!gBenchmarkObxManager) {
        LOGE("clearBenchmarkQueries: BenchmarkObxManager not initialized");
        return;
    }
    gBenchmarkObxManager->removeAll();
    LOGI("clearBenchmarkQueries: all queries and ground truths removed");
}

JNIEXPORT jlong JNICALL
Java_io_graphai_ecovector_benchmark_NativeBenchmarkRunner_saveQueryTextOnly(
        JNIEnv* env, jobject,
        jstring externalId, jstring text, jstring refinedQueryJ,
        jlong createdAt, jstring targetTypes, jstring categories,
        jstring splitJ, jint evalTopK) {
    if (!gBenchmarkObxManager) {
        LOGE("saveQueryTextOnly: BenchmarkObxManager not initialized");
        return -1;
    }
    ecovector::QueryData q;
    q.externalId = jni_utils::toString(env, externalId);
    q.content = jni_utils::toString(env, text);
    q.refinedQuery = refinedQueryJ ? jni_utils::toString(env, refinedQueryJ) : "";
    q.createdAt = createdAt;
    q.targetTypes = targetTypes ? jni_utils::toString(env, targetTypes) : "";
    q.categories = categories ? jni_utils::toString(env, categories) : "";
    q.split = splitJ ? jni_utils::toString(env, splitJ) : "";
    q.evalTopK = evalTopK;
    return static_cast<jlong>(gBenchmarkObxManager->insertQueryTextOnly(q));
}

JNIEXPORT jint JNICALL
Java_io_graphai_ecovector_benchmark_NativeBenchmarkRunner_embedAllQueries(
        JNIEnv*, jobject) {
    if (!gBenchmarkObxManager) return -1;
    auto* store = getGlobalEcoVectorStore();
    if (!store) return -1;

    return gBenchmarkObxManager->updateAllQueryEmbeddings(
        [&](const std::string& text) -> std::pair<std::vector<float>, std::vector<int32_t>> {
            auto vec = store->embedText(text);
            auto tokIds = store->tokenizeText(text);
            return {std::move(vec), std::move(tokIds)};
        }
    );
}

JNIEXPORT jint JNICALL
Java_io_graphai_ecovector_benchmark_NativeBenchmarkRunner_importQueryEmbeddingsFromSQLite(
        JNIEnv* env, jobject, jstring jDbPath) {
    if (!gBenchmarkObxManager) return -1;
    const char* dbPath = env->GetStringUTFChars(jDbPath, nullptr);
    int result = gBenchmarkObxManager->importQueryEmbeddingsFromSQLite(dbPath);
    env->ReleaseStringUTFChars(jDbPath, dbPath);
    return result;
}

JNIEXPORT jint JNICALL
Java_io_graphai_ecovector_benchmark_NativeBenchmarkRunner_tokenizeAllQueries(
        JNIEnv*, jobject) {
    if (!gBenchmarkObxManager) return -1;
    auto* store = getGlobalEcoVectorStore();
    if (!store || !store->getKiwiTokenizer()) return -1;

    return gBenchmarkObxManager->reTokenizeAllKiwiTokens(
        [&](const std::string& text) -> std::vector<int32_t> {
            return hashMorphemes(store->getKiwiTokenizer()->tokenizeForIndexing(text));
        }
    ) ? gBenchmarkObxManager->getQueryCount() : -1;
}

// ============================================================================
// Retriever lifecycle (benchmark-specific: create & destroy without Kotlin wrappers)
// ============================================================================

JNIEXPORT jlong JNICALL
Java_io_graphai_ecovector_benchmark_NativeBenchmarkRunner_createVectorRetriever(
        JNIEnv*, jobject, jint efSearch, jint nprobe, jint topK) {
    auto* store = getGlobalEcoVectorStore();
    if (!store) { LOGE("createVectorRetriever: store is null"); return 0; }
    ecovector::VectorRetriever::Params params;
    params.efSearch = static_cast<size_t>(efSearch);
    params.nprobe = static_cast<size_t>(nprobe);
    params.topK = static_cast<uint32_t>(topK);
    auto* r = store->createVectorRetriever(params);
    return reinterpret_cast<jlong>(r);
}

JNIEXPORT jlong JNICALL
Java_io_graphai_ecovector_benchmark_NativeBenchmarkRunner_createObxVectorRetriever(
        JNIEnv*, jobject, jint maxResultCount, jint topK) {
    auto* store = getGlobalEcoVectorStore();
    if (!store) { LOGE("createObxVectorRetriever: store is null"); return 0; }
    ecovector::ObxVectorRetriever::Params params;
    params.maxResultCount = static_cast<uint32_t>(maxResultCount);
    params.topK = static_cast<uint32_t>(topK);
    auto* r = store->createObxVectorRetriever(params);
    return reinterpret_cast<jlong>(r);
}

JNIEXPORT jlong JNICALL
Java_io_graphai_ecovector_benchmark_NativeBenchmarkRunner_createBM25Retriever(
        JNIEnv*, jobject,
        jfloat k1, jfloat b, jint topK,
        jfloat idfThreshold, jint maxSeedTerms, jint candidateMultiplier,
        jint minCandidates, jfloat minScore,
        jboolean rm3Enabled, jint rm3FbDocs, jint rm3FbTerms,
        jfloat rm3OrigWeight, jint rm3MinDf) {
    auto* store = getGlobalEcoVectorStore();
    if (!store) { LOGE("createBM25Retriever: store is null"); return 0; }
    ecovector::BM25Retriever::Params params;
    params.k1 = k1;
    params.b = b;
    params.topK = static_cast<uint32_t>(topK);
    params.idfThreshold = idfThreshold;
    params.maxSeedTerms = static_cast<size_t>(maxSeedTerms);
    params.candidateMultiplier = static_cast<size_t>(candidateMultiplier);
    params.minCandidates = static_cast<size_t>(minCandidates);
    params.minScore = minScore;
    params.rm3Enabled = rm3Enabled;
    params.rm3FbDocs = static_cast<uint32_t>(rm3FbDocs);
    params.rm3FbTerms = static_cast<uint32_t>(rm3FbTerms);
    params.rm3OrigWeight = rm3OrigWeight;
    params.rm3MinDf = static_cast<uint32_t>(rm3MinDf);
    auto* r = store->createBM25Retriever(params);
    return reinterpret_cast<jlong>(r);
}

JNIEXPORT jlong JNICALL
Java_io_graphai_ecovector_benchmark_NativeBenchmarkRunner_createEnsembleRetriever(
        JNIEnv* env, jobject,
        jlongArray retrieverHandles, jfloatArray weights,
        jint fusionMethod, jfloat rrfK, jint topK, jboolean parallel) {
    auto* store = getGlobalEcoVectorStore();
    if (!store) { LOGE("createEnsembleRetriever: store is null"); return 0; }

    jsize count = env->GetArrayLength(retrieverHandles);
    jlong* handles = env->GetLongArrayElements(retrieverHandles, nullptr);
    jfloat* w = env->GetFloatArrayElements(weights, nullptr);

    std::vector<ecovector::RetrieverConfig> configs;
    configs.reserve(count);
    for (jsize i = 0; i < count; i++) {
        auto* retriever = reinterpret_cast<ecovector::IRetriever*>(handles[i]);
        configs.push_back({retriever, w[i]});
    }
    env->ReleaseLongArrayElements(retrieverHandles, handles, 0);
    env->ReleaseFloatArrayElements(weights, w, 0);

    ecovector::EnsembleRetriever::Params params;
    params.fusionMethod = (fusionMethod == 1)
        ? ecovector::FusionMethod::RRF : ecovector::FusionMethod::RSF;
    params.rrfK = rrfK;
    params.topK = static_cast<uint32_t>(topK);
    params.parallel = parallel;

    auto* r = store->createEnsembleRetriever(std::move(configs), params);
    return reinterpret_cast<jlong>(r);
}

JNIEXPORT jboolean JNICALL
Java_io_graphai_ecovector_benchmark_NativeBenchmarkRunner_destroyRetriever(
        JNIEnv*, jobject, jlong handle) {
    auto* store = getGlobalEcoVectorStore();
    if (!store) {
        LOGE("destroyRetriever: EcoVectorStore not initialized");
        return JNI_FALSE;
    }
    auto* retriever = reinterpret_cast<ecovector::IRetriever*>(handle);
    return store->destroyRetriever(retriever) ? JNI_TRUE : JNI_FALSE;
}

// ============================================================================
// Merge detail files
// ============================================================================

JNIEXPORT jstring JNICALL
Java_io_graphai_ecovector_benchmark_NativeBenchmarkRunner_mergeDetails(
        JNIEnv* env, jobject,
        jobjectArray methodNamesArr, jobjectArray detailPathsArr,
        jstring outputPathJ, jstring splitJ) {
    auto* store = getGlobalEcoVectorStore();
    if (!store || !store->getObxManager() || !store->getTokenizer()) {
        LOGE("mergeDetails: EcoVectorStore not initialized");
        return env->NewStringUTF("error: store not initialized");
    }
    if (!gBenchmarkObxManager) {
        LOGE("mergeDetails: BenchmarkObxManager not initialized");
        return env->NewStringUTF("error: benchmark db not initialized");
    }

    auto methodNames = jni_utils::toStringVector(env, methodNamesArr);
    auto detailPaths = jni_utils::toStringVector(env, detailPathsArr);
    std::string outputPath = jni_utils::toString(env, outputPathJ);
    std::string split = splitJ ? jni_utils::toString(env, splitJ) : "";

    ecovector::BenchmarkRunner runner(
        *store->getObxManager(), *store->getTokenizer(), gBenchmarkObxManager);

    auto result = runner.mergeDetailFiles(methodNames, detailPaths, outputPath, split);
    return env->NewStringUTF(result.c_str());
}

} // extern "C"
