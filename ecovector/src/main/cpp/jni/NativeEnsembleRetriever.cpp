#include <jni.h>
#include <string>
#include <vector>
#include "EcoVectorStore.h"
#include "EnsembleRetriever.h"
#include "JniSearchUtils.h"
#include <android/log.h>

#define LOG_TAG "NativeEnsembleRetriever"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// Defined in NativeEcoVectorStore.cpp
extern "C" ecovector::EcoVectorStore* getGlobalEcoVectorStore();

extern "C" {

JNIEXPORT jlong JNICALL
Java_io_graphai_ecovector_NativeEnsembleRetriever_createFromHandles(
        JNIEnv* env, jobject,
        jlongArray retrieverHandles, jfloatArray weights,
        jint fusionMethod, jfloat rrfK, jint topK, jboolean parallel) {
    auto* store = getGlobalEcoVectorStore();
    if (!store) {
        LOGE("createFromHandles: g_store is null");
        return 0;
    }

    try {
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

        auto* ensemble = store->createEnsembleRetriever(std::move(configs), params);
        return reinterpret_cast<jlong>(ensemble);
    } catch (const std::exception& e) {
        LOGE("createFromHandles failed: %s", e.what());
        return 0;
    }
}

JNIEXPORT jstring JNICALL
Java_io_graphai_ecovector_NativeEnsembleRetriever_retrieveByText(
        JNIEnv* env, jobject, jlong handle, jstring queryText) {
    auto* retriever = reinterpret_cast<ecovector::EnsembleRetriever*>(handle);
    if (!retriever) return env->NewStringUTF("[]");

    try {
        std::string text = jstringToStdString(env, queryText);
        auto results = retriever->retrieve(text);
        return env->NewStringUTF(resultsToJson(results).c_str());
    } catch (const std::exception& e) {
        LOGE("retrieveByText failed: %s", e.what());
        return env->NewStringUTF("[]");
    }
}

JNIEXPORT jstring JNICALL
Java_io_graphai_ecovector_NativeEnsembleRetriever_retrieveByBundle(
        JNIEnv* env, jobject, jlong handle,
        jfloatArray embedding, jintArray kiwiTokens) {
    auto* retriever = reinterpret_cast<ecovector::EnsembleRetriever*>(handle);
    if (!retriever) return env->NewStringUTF("[]");

    try {
        ecovector::QueryBundle bundle;

        // Embedding (nullable)
        if (embedding) {
            jsize embLen = env->GetArrayLength(embedding);
            jfloat* embElements = env->GetFloatArrayElements(embedding, nullptr);
            bundle.embedding.assign(embElements, embElements + embLen);
            env->ReleaseFloatArrayElements(embedding, embElements, 0);
        }

        // Kiwi tokens (nullable)
        if (kiwiTokens) {
            jsize tokLen = env->GetArrayLength(kiwiTokens);
            jint* tokElements = env->GetIntArrayElements(kiwiTokens, nullptr);
            bundle.kiwiTokens.assign(tokElements, tokElements + tokLen);
            env->ReleaseIntArrayElements(kiwiTokens, tokElements, 0);
        }

        auto results = retriever->retrieve(bundle);
        return env->NewStringUTF(resultsToJson(results).c_str());
    } catch (const std::exception& e) {
        LOGE("retrieveByBundle failed: %s", e.what());
        return env->NewStringUTF("[]");
    }
}

JNIEXPORT jstring JNICALL
Java_io_graphai_ecovector_NativeEnsembleRetriever_fuse(
        JNIEnv* env, jobject, jlong handle, jstring resultsJson) {
    auto* retriever = reinterpret_cast<ecovector::EnsembleRetriever*>(handle);
    if (!retriever) return env->NewStringUTF("[]");

    try {
        std::string jsonStr = jstringToStdString(env, resultsJson);
        auto parsed = nlohmann::json::parse(jsonStr);

        // Expected format:
        // [ { "weight": 1.0, "results": [ {"chunkId": 1, "documentId": 2, "content": "...", "score": 0.9}, ... ] }, ... ]
        std::vector<ecovector::WeightedResults> inputs;
        for (const auto& group : parsed) {
            ecovector::WeightedResults wr;
            wr.weight = group.value("weight", 1.0f);
            wr.isDistance = group.value("isDistance", false);
            for (const auto& item : group["results"]) {
                ecovector::ChunkSearchResult csr;
                csr.chunk.id = item.value("chunkId", static_cast<uint64_t>(0));
                csr.chunk.documentId = item.value("documentId", static_cast<uint64_t>(0));
                csr.chunk.content = item.value("content", std::string(""));
                csr.distance = item.value("score", 0.0f);
                wr.results.push_back(std::move(csr));
            }
            inputs.push_back(std::move(wr));
        }

        auto results = retriever->fuse(inputs);
        return env->NewStringUTF(resultsToJson(results).c_str());
    } catch (const std::exception& e) {
        LOGE("fuse failed: %s", e.what());
        return env->NewStringUTF("[]");
    }
}

JNIEXPORT jboolean JNICALL
Java_io_graphai_ecovector_NativeEnsembleRetriever_isReady(
        JNIEnv*, jobject, jlong handle) {
    auto* retriever = reinterpret_cast<ecovector::EnsembleRetriever*>(handle);
    if (!retriever) return JNI_FALSE;
    return retriever->isReady() ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_io_graphai_ecovector_NativeEnsembleRetriever_destroy(
        JNIEnv*, jobject, jlong handle) {
    if (!handle) return;
    auto* store = getGlobalEcoVectorStore();
    if (store) {
        store->destroyRetriever(reinterpret_cast<ecovector::IRetriever*>(handle));
    }
}

} // extern "C"
