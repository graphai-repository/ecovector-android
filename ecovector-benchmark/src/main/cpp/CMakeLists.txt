cmake_minimum_required(VERSION 3.18)
project(ecovector-benchmark)

set(CMAKE_CXX_STANDARD 17)

# FlatBuffers (for BenchmarkObxManager schema serialization)
include(FetchContent)
FetchContent_Declare(
    flatbuffers
    URL https://github.com/google/flatbuffers/archive/refs/tags/v23.5.26.tar.gz
)
FetchContent_MakeAvailable(flatbuffers)

# ObjectBox headers and pre-built .so (for BenchmarkObxManager)
set(OBJECTBOX_INCLUDE "${ECOVECTOR_CPP_DIR}/third_party/objectbox/include")
set(OBJECTBOX_LIB_PATH "${ECOVECTOR_CPP_DIR}/third_party/objectbox/lib/${ANDROID_ABI}")

# --- ecovector SDK headers & pre-built .so ---
# ECOVECTOR_CPP_DIR: ecovector 모듈의 C++ 소스 루트 (헤더 참조용)
# ECOVECTOR_SO_DIR:  ecovector 빌드 출력의 libecovector.so 위치

add_library(ecovector_lib SHARED IMPORTED)
set_target_properties(ecovector_lib PROPERTIES
    IMPORTED_LOCATION "${ECOVECTOR_SO_DIR}/libecovector.so"
)

# Third-party headers (transitive dependencies from ecovector headers)
set(NLOHMANN_INCLUDE "${ECOVECTOR_CPP_DIR}/third_party/nlohmann")
set(TOKENIZERS_CPP_INCLUDE "${ECOVECTOR_CPP_DIR}/third_party/tokenizers-cpp/include")

# --- benchmark library ---
add_library(
    ecovector-benchmark
    SHARED
    jni/NativeBenchmarkRunner.cpp
    src/BenchmarkRunner.cpp
    src/GroundTruthResolver.cpp
    src/QueryFilterResolver.cpp
    src/MetricsEvaluator.cpp
    src/BenchmarkObxManager.cpp
    src/object_box/schema-benchmark.obx.cpp
)

target_include_directories(
    ecovector-benchmark
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/jni
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    # ecovector SDK headers
    ${ECOVECTOR_CPP_DIR}/src/eco_vector_store
    ${ECOVECTOR_CPP_DIR}/src/object_box
    ${ECOVECTOR_CPP_DIR}/src/tokenizer
    ${ECOVECTOR_CPP_DIR}/src/eco_vector
    ${ECOVECTOR_CPP_DIR}/src/bm25
    ${ECOVECTOR_CPP_DIR}/src/reranker
    ${ECOVECTOR_CPP_DIR}/src/retriever
    ${ECOVECTOR_CPP_DIR}/src/kiwi
    ${ECOVECTOR_CPP_DIR}/src/common
    ${ECOVECTOR_CPP_DIR}/src              # SearchUtils.h
    ${ECOVECTOR_CPP_DIR}/jni              # EcoVectorStore forward decl
    ${CMAKE_CURRENT_SOURCE_DIR}/src/object_box
    ${OBJECTBOX_INCLUDE}
    ${flatbuffers_SOURCE_DIR}/include
    ${NLOHMANN_INCLUDE}
    ${TOKENIZERS_CPP_INCLUDE}
)

target_link_libraries(
    ecovector-benchmark
    ecovector_lib
    flatbuffers
    ${OBJECTBOX_LIB_PATH}/libobjectbox-jni.so
    android
    log
)
