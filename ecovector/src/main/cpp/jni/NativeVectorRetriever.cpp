#include <jni.h>
#include <string>
#include <vector>
#include "EcoVectorStore.h"
#include "VectorRetriever.h"
#include "JniSearchUtils.h"
#include <android/log.h>

#define LOG_TAG "NativeVectorRetriever"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// Defined in NativeEcoVectorStore.cpp
extern "C" ecovector::EcoVectorStore* getGlobalEcoVectorStore();

extern "C" {

JNIEXPORT jlong JNICALL
Java_io_graphai_ecovector_NativeVectorRetriever_create(
        JNIEnv* env, jobject, jint efSearch, jint nprobe, jint topK) {
    auto* store = getGlobalEcoVectorStore();
    if (!store) {
        LOGE("create: g_store is null");
        return 0;
    }
    try {
        ecovector::VectorRetriever::Params params;
        params.efSearch = static_cast<size_t>(efSearch);
        params.nprobe = static_cast<size_t>(nprobe);
        params.topK = static_cast<uint32_t>(topK);
        auto* retriever = store->createVectorRetriever(params);
        return reinterpret_cast<jlong>(retriever);
    } catch (const std::exception& e) {
        LOGE("create failed: %s", e.what());
        return 0;
    }
}

JNIEXPORT jstring JNICALL
Java_io_graphai_ecovector_NativeVectorRetriever_retrieveByEmbedding(
        JNIEnv* env, jobject, jlong handle, jfloatArray embedding) {
    auto* retriever = reinterpret_cast<ecovector::VectorRetriever*>(handle);
    if (!retriever) return env->NewStringUTF("[]");

    try {
        jsize len = env->GetArrayLength(embedding);
        jfloat* elements = env->GetFloatArrayElements(embedding, nullptr);
        auto results = retriever->retrieve(elements, static_cast<size_t>(len));
        env->ReleaseFloatArrayElements(embedding, elements, 0);
        return env->NewStringUTF(resultsToJson(results).c_str());
    } catch (const std::exception& e) {
        LOGE("retrieveByEmbedding failed: %s", e.what());
        return env->NewStringUTF("[]");
    }
}

JNIEXPORT jstring JNICALL
Java_io_graphai_ecovector_NativeVectorRetriever_retrieveByText(
        JNIEnv* env, jobject, jlong handle, jstring queryText) {
    auto* retriever = reinterpret_cast<ecovector::VectorRetriever*>(handle);
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
Java_io_graphai_ecovector_NativeVectorRetriever_isReady(
        JNIEnv*, jobject, jlong handle) {
    auto* retriever = reinterpret_cast<ecovector::VectorRetriever*>(handle);
    if (!retriever) return JNI_FALSE;
    return retriever->isReady() ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_io_graphai_ecovector_NativeVectorRetriever_destroy(
        JNIEnv*, jobject, jlong handle) {
    if (!handle) return;
    auto* store = getGlobalEcoVectorStore();
    if (store) {
        store->destroyRetriever(reinterpret_cast<ecovector::IRetriever*>(handle));
    }
}

} // extern "C"
