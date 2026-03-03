// app/src/main/cpp/src/common/Logging.h
#ifndef ECOVECTOR_COMMON_LOGGING_H
#define ECOVECTOR_COMMON_LOGGING_H

#include <android/log.h>

// 각 파일에서 LOG_TAG를 정의한 후 이 매크로들을 사용
// 예: #define LOG_TAG "MyClass"
//     #include "common/Logging.h"

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

#endif // ECOVECTOR_COMMON_LOGGING_H
