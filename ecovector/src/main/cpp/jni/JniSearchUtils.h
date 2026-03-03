#pragma once
#include <jni.h>
#include <string>
#include <vector>
#include <json.hpp>
#include "ObxManager.h"

inline std::string resultsToJson(const std::vector<ecovector::ChunkSearchResult>& results) {
    nlohmann::json arr = nlohmann::json::array();
    for (const auto& r : results) {
        arr.push_back({
            {"documentId", r.chunk.documentId},
            {"chunkId",    r.chunk.id},
            {"content",    r.chunk.content},
            {"score",      r.distance}
        });
    }
    return arr.dump();
}

inline std::string jstringToStdString(JNIEnv* env, jstring jstr) {
    if (!jstr) return "";
    const char* chars = env->GetStringUTFChars(jstr, nullptr);
    std::string result(chars);
    env->ReleaseStringUTFChars(jstr, chars);
    return result;
}
