#include <jni.h>
#include <string>
#include "EcoVectorStore.h"
#include "ObxVectorRetriever.h"
#include <android/log.h>

#define LOG_TAG "NativeObxVectorRetriever"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

extern "C" ecovector::EcoVectorStore* getGlobalEcoVectorStore();

extern "C" {

JNIEXPORT jlong JNICALL
Java_io_graphai_ecovector_NativeObxVectorRetriever_create(
        JNIEnv* env, jobject, jint maxResultCount, jint topK) {
    auto* store = getGlobalEcoVectorStore();
    if (!store) {
        LOGE("create: g_store is null");
        return 0;
    }
    try {
        ecovector::ObxVectorRetriever::Params params;
        params.maxResultCount = static_cast<uint32_t>(maxResultCount);
        params.topK = static_cast<uint32_t>(topK);
        auto* retriever = store->createObxVectorRetriever(params);
        return reinterpret_cast<jlong>(retriever);
    } catch (const std::exception& e) {
        LOGE("create failed: %s", e.what());
        return 0;
    }
}

JNIEXPORT jboolean JNICALL
Java_io_graphai_ecovector_NativeObxVectorRetriever_isReady(
        JNIEnv*, jobject, jlong handle) {
    auto* retriever = reinterpret_cast<ecovector::ObxVectorRetriever*>(handle);
    if (!retriever) return JNI_FALSE;
    return retriever->isReady() ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT void JNICALL
Java_io_graphai_ecovector_NativeObxVectorRetriever_destroy(
        JNIEnv*, jobject, jlong handle) {
    if (!handle) return;
    auto* store = getGlobalEcoVectorStore();
    if (store) {
        store->destroyRetriever(reinterpret_cast<ecovector::IRetriever*>(handle));
    }
}

} // extern "C"
