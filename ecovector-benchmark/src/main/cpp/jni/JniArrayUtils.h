#pragma once

#include <jni.h>
#include <string>
#include <vector>

namespace jni_utils {

inline std::string toString(JNIEnv* env, jstring jstr) {
    if (!jstr) return "";
    const char* chars = env->GetStringUTFChars(jstr, nullptr);
    std::string result(chars);
    env->ReleaseStringUTFChars(jstr, chars);
    return result;
}

inline std::vector<int32_t> toIntVector(JNIEnv* env, jintArray arr) {
    if (!arr) return {};
    jsize len = env->GetArrayLength(arr);
    jint* elements = env->GetIntArrayElements(arr, nullptr);
    std::vector<int32_t> result(elements, elements + len);
    env->ReleaseIntArrayElements(arr, elements, 0);
    return result;
}

inline std::vector<float> toFloatVector(JNIEnv* env, jfloatArray arr) {
    if (!arr) return {};
    jsize len = env->GetArrayLength(arr);
    jfloat* elements = env->GetFloatArrayElements(arr, nullptr);
    std::vector<float> result(elements, elements + len);
    env->ReleaseFloatArrayElements(arr, elements, 0);
    return result;
}

inline std::vector<std::string> toStringVector(JNIEnv* env, jobjectArray arr) {
    if (!arr) return {};
    int len = env->GetArrayLength(arr);
    std::vector<std::string> result;
    result.reserve(len);
    for (int i = 0; i < len; i++) {
        auto jStr = (jstring)env->GetObjectArrayElement(arr, i);
        result.push_back(toString(env, jStr));
        env->DeleteLocalRef(jStr);
    }
    return result;
}

inline jlongArray toLongArray(JNIEnv* env, const std::vector<int64_t>& vec) {
    jlongArray result = env->NewLongArray(static_cast<jsize>(vec.size()));
    env->SetLongArrayRegion(result, 0, static_cast<jsize>(vec.size()),
                            reinterpret_cast<const jlong*>(vec.data()));
    return result;
}

} // namespace jni_utils
