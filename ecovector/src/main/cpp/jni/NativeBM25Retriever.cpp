#include <jni.h>
#include <string>
#include <vector>
#include "EcoVectorStore.h"
#include "BM25Retriever.h"
#include "JniSearchUtils.h"
#include <android/log.h>

#define LOG_TAG "NativeBM25Retriever"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// Defined in NativeEcoVectorStore.cpp
extern "C" ecovector::EcoVectorStore* getGlobalEcoVectorStore();

extern "C" {

JNIEXPORT jlong JNICALL
Java_io_graphai_ecovector_NativeBM25Retriever_create(
        JNIEnv* env, jobject, jfloat k1, jfloat b, jint topK,
        jfloat idfThreshold, jint maxSeedTerms, jint candidateMultiplier,
        jint minCandidates, jfloat minScore,
        jboolean rm3Enabled, jint rm3FbDocs, jint rm3FbTerms,
        jfloat rm3OrigWeight, jint rm3MinDf) {
    auto* store = getGlobalEcoVectorStore();
    if (!store) {
        LOGE("create: g_store is null");
        return 0;
    }
    try {
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
        auto* retriever = store->createBM25Retriever(params);
        return reinterpret_cast<jlong>(retriever);
    } catch (const std::exception& e) {
        LOGE("create failed: %s", e.what());
        return 0;
    }
}

JNIEXPORT jstring JNICALL
Java_io_graphai_ecovector_NativeBM25Retriever_retrieveByTokens(
        JNIEnv* env, jobject, jlong handle, jintArray kiwiTokens) {
    auto* retriever = reinterpret_cast<ecovector::BM25Retriever*>(handle);
    if (!retriever) return env->NewStringUTF("[]");

    try {
        jsize len = env->GetArrayLength(kiwiTokens);
        jint* elements = env->GetIntArrayElements(kiwiTokens, nullptr);
        std::vector<int32_t> tokens(elements, elements + len);
        env->ReleaseIntArrayElements(kiwiTokens, elements, 0);

        auto results = retriever->retrieve(tokens);
        return env->NewStringUTF(resultsToJson(results).c_str());
    } catch (const std::exception& e) {
        LOGE("retrieveByTokens failed: %s", e.what());
        return env->NewStringUTF("[]");
    }
}

JNIEXPORT jstring JNICALL
Java_io_graphai_ecovector_NativeBM25Retriever_retrieveByText(
        JNIEnv* env, jobject, jlong handle, jstring queryText) {
    auto* retriever = reinterpret_cast<ecovector::BM25Retriever*>(handle);
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

JNIEXPORT jboolean JNICALL
Java_io_graphai_ecovector_NativeBM25Retriever_isReady(
        JNIEnv*, jobject, jlong handle) {
    auto* retriever = reinterpret_cast<ecovector::BM25Retriever*>(handle);
    if (!retriever) return JNI_FALSE;
    return retriever->isReady() ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_io_graphai_ecovector_NativeBM25Retriever_destroy(
        JNIEnv*, jobject, jlong handle) {
    if (!handle) return;
    auto* store = getGlobalEcoVectorStore();
    if (store) {
        store->destroyRetriever(reinterpret_cast<ecovector::IRetriever*>(handle));
    }
}

} // extern "C"
