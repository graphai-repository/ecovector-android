#include <jni.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include "EcoVectorStore.h"
#include "KiwiTokenizer.h"
#include "KiwiHashUtil.h"
#include "JniSearchUtils.h"
#include <android/log.h>

#define LOG_TAG "NativeEcoVectorStore"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

static std::unique_ptr<ecovector::EcoVectorStore> g_store;

extern "C" {

// ============================================================================
// Lifecycle
// ============================================================================

JNIEXPORT jboolean JNICALL
Java_io_graphai_ecovector_NativeEcoVectorStore_initialize(
        JNIEnv* env, jobject,
        jstring dbPath, jstring tokenizerPath,
        jstring modelPath, jstring kiwiModelDir) {
    g_store = std::make_unique<ecovector::EcoVectorStore>();
    return g_store->initialize(
        jstringToStdString(env, dbPath),
        jstringToStdString(env, tokenizerPath),
        jstringToStdString(env, modelPath),
        jstringToStdString(env, kiwiModelDir)
    );
}

JNIEXPORT void JNICALL
Java_io_graphai_ecovector_NativeEcoVectorStore_close(JNIEnv*, jobject) {
    g_store.reset();
}

JNIEXPORT void JNICALL
Java_io_graphai_ecovector_NativeEcoVectorStore_setChunkPrefix(
        JNIEnv* env, jobject, jstring prefix) {
    if (g_store && prefix) {
        const char* str = env->GetStringUTFChars(prefix, nullptr);
        g_store->setChunkPrefix(str);
        env->ReleaseStringUTFChars(prefix, str);
    }
}

JNIEXPORT void JNICALL
Java_io_graphai_ecovector_NativeEcoVectorStore_clearChunkPrefix(
        JNIEnv*, jobject) {
    if (g_store) {
        g_store->clearChunkPrefix();
    }
}

// ============================================================================
// Document Management
// ============================================================================

JNIEXPORT jlong JNICALL
Java_io_graphai_ecovector_NativeEcoVectorStore_addDocument(
        JNIEnv* env, jobject, jstring text, jstring title) {
    if (!g_store) return -1;
    return g_store->addDocument(
        jstringToStdString(env, text),
        jstringToStdString(env, title)
    );
}

JNIEXPORT jlongArray JNICALL
Java_io_graphai_ecovector_NativeEcoVectorStore_addDocuments(
        JNIEnv* env, jobject, jobjectArray texts, jobjectArray titles) {
    if (!g_store) return env->NewLongArray(0);

    int len = env->GetArrayLength(texts);
    std::vector<std::string> textVec, titleVec;
    textVec.reserve(len);
    titleVec.reserve(len);

    for (int i = 0; i < len; i++) {
        auto jText = (jstring)env->GetObjectArrayElement(texts, i);
        auto jTitle = (jstring)env->GetObjectArrayElement(titles, i);
        textVec.push_back(jstringToStdString(env, jText));
        titleVec.push_back(jstringToStdString(env, jTitle));
        env->DeleteLocalRef(jText);
        env->DeleteLocalRef(jTitle);
    }

    auto ids = g_store->addDocuments(textVec, titleVec);
    jlongArray result = env->NewLongArray(static_cast<jsize>(ids.size()));
    env->SetLongArrayRegion(result, 0, static_cast<jsize>(ids.size()), ids.data());
    return result;
}

JNIEXPORT jlongArray JNICALL
Java_io_graphai_ecovector_NativeEcoVectorStore_addDocumentsWithIds(
        JNIEnv* env, jobject, jobjectArray texts, jobjectArray titles, jobjectArray externalIds) {
    if (!g_store) return env->NewLongArray(0);

    int len = env->GetArrayLength(texts);
    std::vector<std::string> textVec, titleVec, idVec;
    textVec.reserve(len);
    titleVec.reserve(len);
    idVec.reserve(len);

    for (int i = 0; i < len; i++) {
        auto jText = (jstring)env->GetObjectArrayElement(texts, i);
        auto jTitle = (jstring)env->GetObjectArrayElement(titles, i);
        auto jId = (jstring)env->GetObjectArrayElement(externalIds, i);
        textVec.push_back(jstringToStdString(env, jText));
        titleVec.push_back(jstringToStdString(env, jTitle));
        idVec.push_back(jstringToStdString(env, jId));
        env->DeleteLocalRef(jText);
        env->DeleteLocalRef(jTitle);
        env->DeleteLocalRef(jId);
    }

    auto ids = g_store->addDocuments(textVec, titleVec, idVec);
    jlongArray result = env->NewLongArray(static_cast<jsize>(ids.size()));
    env->SetLongArrayRegion(result, 0, static_cast<jsize>(ids.size()), ids.data());
    return result;
}

JNIEXPORT jlongArray JNICALL
Java_io_graphai_ecovector_NativeEcoVectorStore_addDocumentsWithMetadata(
        JNIEnv* env, jobject,
        jobjectArray texts, jobjectArray titles, jobjectArray externalIds,
        jlongArray createdAts, jobjectArray senders, jshortArray sourceTypes) {
    if (!g_store) return env->NewLongArray(0);

    int len = env->GetArrayLength(texts);
    std::vector<std::string> textVec, titleVec, idVec, senderVec;
    std::vector<int64_t> dateVec;
    textVec.reserve(len);
    titleVec.reserve(len);
    idVec.reserve(len);

    for (int i = 0; i < len; i++) {
        auto jText = (jstring)env->GetObjectArrayElement(texts, i);
        auto jTitle = (jstring)env->GetObjectArrayElement(titles, i);
        auto jId = (jstring)env->GetObjectArrayElement(externalIds, i);
        textVec.push_back(jstringToStdString(env, jText));
        titleVec.push_back(jstringToStdString(env, jTitle));
        idVec.push_back(jstringToStdString(env, jId));
        env->DeleteLocalRef(jText);
        env->DeleteLocalRef(jTitle);
        env->DeleteLocalRef(jId);
    }

    if (createdAts) {
        jlong* dateElements = env->GetLongArrayElements(createdAts, nullptr);
        dateVec.assign(dateElements, dateElements + len);
        env->ReleaseLongArrayElements(createdAts, dateElements, 0);
    }

    if (senders) {
        senderVec.reserve(len);
        for (int i = 0; i < len; i++) {
            auto jSender = (jstring)env->GetObjectArrayElement(senders, i);
            senderVec.push_back(jstringToStdString(env, jSender));
            env->DeleteLocalRef(jSender);
        }
    }

    std::vector<int16_t> sourceTypeVec;
    if (sourceTypes) {
        jshort* stElements = env->GetShortArrayElements(sourceTypes, nullptr);
        sourceTypeVec.assign(stElements, stElements + len);
        env->ReleaseShortArrayElements(sourceTypes, stElements, 0);
    }

    auto ids = g_store->addDocuments(textVec, titleVec, idVec, dateVec, senderVec, sourceTypeVec);
    jlongArray result = env->NewLongArray(static_cast<jsize>(ids.size()));
    env->SetLongArrayRegion(result, 0, static_cast<jsize>(ids.size()), ids.data());
    return result;
}

JNIEXPORT void JNICALL
Java_io_graphai_ecovector_NativeEcoVectorStore_removeAll(JNIEnv*, jobject) {
    if (g_store) g_store->removeAll();
}

JNIEXPORT jstring JNICALL
Java_io_graphai_ecovector_NativeEcoVectorStore_removeOrphanDocuments(JNIEnv* env, jobject) {
    if (!g_store) return env->NewStringUTF("[]");
    auto result = g_store->removeOrphanDocuments();
    return env->NewStringUTF(result.c_str());
}

// ============================================================================
// Tokenization & Embedding (public SDK API)
// ============================================================================

JNIEXPORT jintArray JNICALL
Java_io_graphai_ecovector_NativeEcoVectorStore_tokenize(
        JNIEnv* env, jobject, jstring text) {
    if (!g_store || !g_store->getTokenizer()) return env->NewIntArray(0);
    try {
        auto ids = g_store->tokenizeText(jstringToStdString(env, text));
        jintArray result = env->NewIntArray(static_cast<jsize>(ids.size()));
        env->SetIntArrayRegion(result, 0, static_cast<jsize>(ids.size()), ids.data());
        return result;
    } catch (const std::exception& e) {
        LOGE("tokenize failed: %s", e.what());
        return env->NewIntArray(0);
    }
}

JNIEXPORT jfloatArray JNICALL
Java_io_graphai_ecovector_NativeEcoVectorStore_embed(
        JNIEnv* env, jobject, jstring text) {
    if (!g_store) return env->NewFloatArray(0);
    try {
        auto embedding = g_store->embedText(jstringToStdString(env, text));
        jfloatArray result = env->NewFloatArray(static_cast<jsize>(embedding.size()));
        env->SetFloatArrayRegion(result, 0, static_cast<jsize>(embedding.size()), embedding.data());
        return result;
    } catch (const std::exception& e) {
        LOGE("embed failed: %s", e.what());
        return env->NewFloatArray(0);
    }
}

JNIEXPORT jintArray JNICALL
Java_io_graphai_ecovector_NativeEcoVectorStore_tokenizeKiwi(
        JNIEnv* env, jobject, jstring text) {
    if (!g_store) return env->NewIntArray(0);
    try {
        auto* kiwi = g_store->getKiwiTokenizer();
        if (!kiwi) return env->NewIntArray(0);
        auto morphemes = kiwi->tokenize(jstringToStdString(env, text));
        auto hashes = ecovector::hashMorphemes(morphemes);
        jintArray result = env->NewIntArray(static_cast<jsize>(hashes.size()));
        env->SetIntArrayRegion(result, 0, static_cast<jsize>(hashes.size()), hashes.data());
        return result;
    } catch (const std::exception& e) {
        LOGE("tokenizeKiwi failed: %s", e.what());
        return env->NewIntArray(0);
    }
}

// ============================================================================
// Index Management
// ============================================================================

JNIEXPORT jboolean JNICALL
Java_io_graphai_ecovector_NativeEcoVectorStore_buildIndex(
        JNIEnv*, jobject, jint centroidCount) {
    if (!g_store) return JNI_FALSE;
    return g_store->buildIndex(centroidCount);
}

JNIEXPORT jboolean JNICALL
Java_io_graphai_ecovector_NativeEcoVectorStore_isIndexReady(JNIEnv*, jobject) {
    if (!g_store) return JNI_FALSE;
    return g_store->isIndexReady();
}

// ============================================================================
// Statistics
// ============================================================================

JNIEXPORT jint JNICALL
Java_io_graphai_ecovector_NativeEcoVectorStore_getDocumentCount(JNIEnv*, jobject) {
    return g_store ? g_store->getDocumentCount() : 0;
}

JNIEXPORT jint JNICALL
Java_io_graphai_ecovector_NativeEcoVectorStore_getChunkCount(JNIEnv*, jobject) {
    return g_store ? g_store->getChunkCount() : 0;
}

// ============================================================================
// Data Inspection
// ============================================================================

JNIEXPORT jstring JNICALL
Java_io_graphai_ecovector_NativeEcoVectorStore_getDocumentsJson(
        JNIEnv* env, jobject, jint offset, jint limit) {
    if (!g_store) return env->NewStringUTF("[]");
    return env->NewStringUTF(g_store->getDocumentsJson(offset, limit).c_str());
}

JNIEXPORT jstring JNICALL
Java_io_graphai_ecovector_NativeEcoVectorStore_getChunksJson(
        JNIEnv* env, jobject, jint offset, jint limit) {
    if (!g_store) return env->NewStringUTF("[]");
    return env->NewStringUTF(g_store->getChunksJson(offset, limit).c_str());
}

// ============================================================================
// Lightweight ID-only access
// ============================================================================

JNIEXPORT jstring JNICALL
Java_io_graphai_ecovector_NativeEcoVectorStore_getDocumentExternalIdsJson(JNIEnv* env, jobject) {
    if (!g_store) return env->NewStringUTF("[]");
    return env->NewStringUTF(g_store->getDocumentExternalIdsJson().c_str());
}

// ============================================================================
// Raw Save (pre-computed vectors, bypasses pipeline)
// ============================================================================

JNIEXPORT jlong JNICALL
Java_io_graphai_ecovector_NativeEcoVectorStore_importDocument(
        JNIEnv* env, jobject,
        jstring externalId, jstring description, jstring content,
        jlong createdAt, jint sourceType, jstring sender) {
    if (!g_store) return -1;
    return g_store->saveDocumentRaw(
        jstringToStdString(env, externalId),
        jstringToStdString(env, description),
        jstringToStdString(env, content),
        createdAt,
        static_cast<int16_t>(sourceType),
        jstringToStdString(env, sender)
    );
}

JNIEXPORT jlong JNICALL
Java_io_graphai_ecovector_NativeEcoVectorStore_importChunk(
        JNIEnv* env, jobject,
        jlong documentId, jint chunkIndex, jstring content,
        jintArray tokenIds, jfloatArray embedding,
        jintArray kiwiTokens,
        jlong createdAt, jint sourceType, jstring sender) {
    if (!g_store) return -1;

    std::string contentStr = jstringToStdString(env, content);

    jsize tokenLen = env->GetArrayLength(tokenIds);
    jint* tokenElements = env->GetIntArrayElements(tokenIds, nullptr);
    std::vector<int32_t> tokenIdsVec(tokenElements, tokenElements + tokenLen);
    env->ReleaseIntArrayElements(tokenIds, tokenElements, 0);

    jsize embLen = env->GetArrayLength(embedding);
    jfloat* embElements = env->GetFloatArrayElements(embedding, nullptr);
    std::vector<float> embeddingVec(embElements, embElements + embLen);
    env->ReleaseFloatArrayElements(embedding, embElements, 0);

    jsize kiwiLen = env->GetArrayLength(kiwiTokens);
    jint* kiwiElements = env->GetIntArrayElements(kiwiTokens, nullptr);
    std::vector<int32_t> kiwiTokensVec(kiwiElements, kiwiElements + kiwiLen);
    env->ReleaseIntArrayElements(kiwiTokens, kiwiElements, 0);

    return g_store->saveChunkRaw(documentId, chunkIndex, contentStr,
                                  tokenIdsVec, embeddingVec, kiwiTokensVec,
                                  createdAt, static_cast<int16_t>(sourceType),
                                  jstringToStdString(env, sender));
}

// ============================================================================
// Re-Tokenize
// ============================================================================

JNIEXPORT jboolean JNICALL
Java_io_graphai_ecovector_NativeEcoVectorStore_reTokenizeAll(JNIEnv*, jobject) {
    if (!g_store) return JNI_FALSE;
    return g_store->reTokenizeAll() ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jint JNICALL
Java_io_graphai_ecovector_NativeEcoVectorStore_bulkUpdateCreatedAt(
        JNIEnv* env, jobject,
        jobjectArray docExternalIds, jlongArray createdAts) {
    if (!g_store) return 0;

    int len = env->GetArrayLength(docExternalIds);
    jlong* dates = env->GetLongArrayElements(createdAts, nullptr);

    std::unordered_map<std::string, int64_t> docDateMap;
    docDateMap.reserve(len);
    for (int i = 0; i < len; i++) {
        auto jId = (jstring)env->GetObjectArrayElement(docExternalIds, i);
        docDateMap[jstringToStdString(env, jId)] = static_cast<int64_t>(dates[i]);
        env->DeleteLocalRef(jId);
    }
    env->ReleaseLongArrayElements(createdAts, dates, 0);

    return static_cast<jint>(g_store->getObxManager()->bulkUpdateCreatedAt(docDateMap));
}

JNIEXPORT jint JNICALL
Java_io_graphai_ecovector_NativeEcoVectorStore_bulkUpdateSourceType(
        JNIEnv* env, jobject,
        jobjectArray docExternalIds, jshortArray sourceTypes) {
    if (!g_store) return 0;

    int len = env->GetArrayLength(docExternalIds);
    jshort* types = env->GetShortArrayElements(sourceTypes, nullptr);

    std::unordered_map<std::string, int16_t> docTypeMap;
    docTypeMap.reserve(len);
    for (int i = 0; i < len; i++) {
        auto jId = (jstring)env->GetObjectArrayElement(docExternalIds, i);
        docTypeMap[jstringToStdString(env, jId)] = static_cast<int16_t>(types[i]);
        env->DeleteLocalRef(jId);
    }
    env->ReleaseShortArrayElements(sourceTypes, types, 0);

    return static_cast<jint>(g_store->getObxManager()->bulkUpdateSourceType(docTypeMap));
}

// ============================================================================
// Split Index Build
// ============================================================================

JNIEXPORT jboolean JNICALL
Java_io_graphai_ecovector_NativeEcoVectorStore_buildVectorIndex(
        JNIEnv*, jobject, jint centroidCount, jint hnswM, jint efConstruction, jint maxTrainSamples) {
    if (!g_store) return JNI_FALSE;
    try {
        return g_store->buildVectorIndex(centroidCount, hnswM, efConstruction, maxTrainSamples) ? JNI_TRUE : JNI_FALSE;
    } catch (const std::exception& e) {
        LOGE("buildVectorIndex failed: %s", e.what());
        return JNI_FALSE;
    }
}

JNIEXPORT jboolean JNICALL
Java_io_graphai_ecovector_NativeEcoVectorStore_buildBM25Index(
        JNIEnv*, jobject) {
    if (!g_store) return JNI_FALSE;
    try {
        return g_store->buildBM25Index() ? JNI_TRUE : JNI_FALSE;
    } catch (const std::exception& e) {
        LOGE("buildBM25Index failed: %s", e.what());
        return JNI_FALSE;
    }
}

// ============================================================================
// ChunkParams Support
// ============================================================================

JNIEXPORT jlong JNICALL
Java_io_graphai_ecovector_NativeEcoVectorStore_addDocumentWithChunkParams(
        JNIEnv* env, jobject,
        jstring text, jstring title,
        jint maxTokens, jint overlapTokens) {
    if (!g_store) return -1;
    try {
        return g_store->addDocumentWithChunkParams(
            jstringToStdString(env, text),
            jstringToStdString(env, title),
            maxTokens, overlapTokens
        );
    } catch (const std::exception& e) {
        LOGE("addDocumentWithChunkParams failed: %s", e.what());
        return -1;
    }
}

JNIEXPORT jint JNICALL
Java_io_graphai_ecovector_NativeEcoVectorStore_addDocumentsWithChunkParams(
        JNIEnv* env, jobject,
        jobjectArray texts, jobjectArray titles,
        jint maxTokens, jint overlapTokens) {
    if (!g_store) return 0;
    try {
        int len = env->GetArrayLength(texts);
        std::vector<std::string> textVec, titleVec;
        textVec.reserve(len);
        titleVec.reserve(len);

        for (int i = 0; i < len; i++) {
            auto jText = (jstring)env->GetObjectArrayElement(texts, i);
            auto jTitle = (jstring)env->GetObjectArrayElement(titles, i);
            textVec.push_back(jstringToStdString(env, jText));
            titleVec.push_back(jstringToStdString(env, jTitle));
            env->DeleteLocalRef(jText);
            env->DeleteLocalRef(jTitle);
        }

        return g_store->addDocumentsWithChunkParams(
            textVec, titleVec,
            maxTokens, overlapTokens
        );
    } catch (const std::exception& e) {
        LOGE("addDocumentsWithChunkParams failed: %s", e.what());
        return 0;
    }
}

// (retrieve/retrieveWithFilter removed — use IRetriever factory methods instead)

// ============================================================================
// Data Access
// ============================================================================

JNIEXPORT jstring JNICALL
Java_io_graphai_ecovector_NativeEcoVectorStore_getDocumentJson(
        JNIEnv* env, jobject, jlong id) {
    if (!g_store) return env->NewStringUTF("{}");
    try {
        auto result = g_store->getDocumentJson(static_cast<uint64_t>(id));
        return env->NewStringUTF(result.c_str());
    } catch (const std::exception& e) {
        LOGE("getDocumentJson failed: %s", e.what());
        return env->NewStringUTF("{}");
    }
}

JNIEXPORT jstring JNICALL
Java_io_graphai_ecovector_NativeEcoVectorStore_getChunksByDocumentJson(
        JNIEnv* env, jobject, jlong docId) {
    if (!g_store) return env->NewStringUTF("[]");
    try {
        auto result = g_store->getChunksByDocumentJson(static_cast<uint64_t>(docId));
        return env->NewStringUTF(result.c_str());
    } catch (const std::exception& e) {
        LOGE("getChunksByDocumentJson failed: %s", e.what());
        return env->NewStringUTF("[]");
    }
}

JNIEXPORT jstring JNICALL
Java_io_graphai_ecovector_NativeEcoVectorStore_getChunkJson(
        JNIEnv* env, jobject, jlong id) {
    if (!g_store) return env->NewStringUTF("{}");
    try {
        auto result = g_store->getChunkJson(static_cast<uint64_t>(id));
        return env->NewStringUTF(result.c_str());
    } catch (const std::exception& e) {
        LOGE("getChunkJson failed: %s", e.what());
        return env->NewStringUTF("{}");
    }
}

// ============================================================================
// Data Management (remove by ID)
// ============================================================================

JNIEXPORT jboolean JNICALL
Java_io_graphai_ecovector_NativeEcoVectorStore_removeDocument(
        JNIEnv*, jobject, jlong id) {
    if (!g_store) return JNI_FALSE;
    try {
        return g_store->removeDocumentById(static_cast<uint64_t>(id)) ? JNI_TRUE : JNI_FALSE;
    } catch (const std::exception& e) {
        LOGE("removeDocument failed: %s", e.what());
        return JNI_FALSE;
    }
}

JNIEXPORT jboolean JNICALL
Java_io_graphai_ecovector_NativeEcoVectorStore_removeChunk(
        JNIEnv*, jobject, jlong id) {
    if (!g_store) return JNI_FALSE;
    try {
        return g_store->removeChunkById(static_cast<uint64_t>(id)) ? JNI_TRUE : JNI_FALSE;
    } catch (const std::exception& e) {
        LOGE("removeChunk failed: %s", e.what());
        return JNI_FALSE;
    }
}

// === Pipeline Stage JNI Methods ===

JNIEXPORT jint JNICALL
Java_io_graphai_ecovector_NativeEcoVectorStore_addDocumentsOnly(
        JNIEnv* env, jobject,
        jobjectArray texts, jobjectArray titles, jobjectArray externalIds,
        jlongArray createdAts, jobjectArray senders, jintArray sourceTypes) {
    if (!g_store) return -1;
    int len = env->GetArrayLength(texts);
    std::vector<std::string> textVec, titleVec, idVec, senderVec;
    std::vector<int64_t> dateVec;
    std::vector<int16_t> sourceTypeVec;
    textVec.reserve(len); titleVec.reserve(len); idVec.reserve(len);

    for (int i = 0; i < len; i++) {
        auto jText = (jstring)env->GetObjectArrayElement(texts, i);
        auto jTitle = (jstring)env->GetObjectArrayElement(titles, i);
        auto jId = (jstring)env->GetObjectArrayElement(externalIds, i);
        textVec.push_back(jstringToStdString(env, jText));
        titleVec.push_back(jstringToStdString(env, jTitle));
        idVec.push_back(jstringToStdString(env, jId));
        env->DeleteLocalRef(jText); env->DeleteLocalRef(jTitle); env->DeleteLocalRef(jId);
    }
    if (createdAts) {
        jlong* e = env->GetLongArrayElements(createdAts, nullptr);
        dateVec.assign(e, e + len);
        env->ReleaseLongArrayElements(createdAts, e, 0);
    }
    if (senders) {
        senderVec.reserve(len);
        for (int i = 0; i < len; i++) {
            auto js = (jstring)env->GetObjectArrayElement(senders, i);
            senderVec.push_back(jstringToStdString(env, js));
            env->DeleteLocalRef(js);
        }
    }
    if (sourceTypes) {
        jint* st = env->GetIntArrayElements(sourceTypes, nullptr);
        sourceTypeVec.reserve(len);
        for (int i = 0; i < len; i++) sourceTypeVec.push_back(static_cast<int16_t>(st[i]));
        env->ReleaseIntArrayElements(sourceTypes, st, 0);
    }
    return g_store->addDocumentsOnly(textVec, titleVec, idVec, dateVec, senderVec, sourceTypeVec);
}

JNIEXPORT jint JNICALL
Java_io_graphai_ecovector_NativeEcoVectorStore_chunkAllDocuments(
        JNIEnv*, jobject) {
    if (!g_store) return -1;
    return g_store->chunkAllDocuments();
}

JNIEXPORT jint JNICALL
Java_io_graphai_ecovector_NativeEcoVectorStore_embedChunks(
        JNIEnv*, jobject, jboolean forceAll) {
    if (!g_store) return -1;
    return g_store->embedChunks(static_cast<bool>(forceAll));
}

JNIEXPORT jint JNICALL
Java_io_graphai_ecovector_NativeEcoVectorStore_tokenizeChunks(
        JNIEnv*, jobject) {
    if (!g_store) return -1;
    return g_store->tokenizeChunks();
}

JNIEXPORT jint JNICALL
Java_io_graphai_ecovector_NativeEcoVectorStore_exportChunksToSQLite(
        JNIEnv* env, jobject, jstring sqlitePath) {
    if (!g_store) return -1;
    const char* path = env->GetStringUTFChars(sqlitePath, nullptr);
    int result = g_store->exportChunksToSQLite(path);
    env->ReleaseStringUTFChars(sqlitePath, path);
    return result;
}

JNIEXPORT jint JNICALL
Java_io_graphai_ecovector_NativeEcoVectorStore_importEmbeddingsFromSQLite(
        JNIEnv* env, jobject, jstring sqlitePath) {
    if (!g_store) return -1;
    const char* path = env->GetStringUTFChars(sqlitePath, nullptr);
    int result = g_store->importEmbeddingsFromSQLite(path);
    env->ReleaseStringUTFChars(sqlitePath, path);
    return result;
}

// Provides global EcoVectorStore pointer for NativeBenchmarkRunner.cpp (same .so)
ecovector::EcoVectorStore* getGlobalEcoVectorStore() {
    return g_store.get();
}

} // extern "C"
