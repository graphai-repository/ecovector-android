cmake_minimum_required(VERSION 3.18)
project(ecovector)

set(CMAKE_CXX_STANDARD 17)

# FlatBuffers
include(FetchContent)
FetchContent_Declare(
    flatbuffers
    URL https://github.com/google/flatbuffers/archive/refs/tags/v23.5.26.tar.gz
)
FetchContent_MakeAvailable(flatbuffers)

# ObjectBox C API include and lib paths (similar to ONNX Runtime)
set(OBJECTBOX_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/objectbox/include)
set(OBJECTBOX_LIB_PATH ${CMAKE_CURRENT_SOURCE_DIR}/third_party/objectbox/lib/${ANDROID_ABI})

# FAISS library paths (for EcoVector)
set(FAISS_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/faiss/include)
set(FAISS_LIB_PATH ${CMAKE_CURRENT_SOURCE_DIR}/third_party/faiss/lib/${ANDROID_ABI})

# Tokenizers-cpp (pre-built)
set(TOKENIZERS_CPP_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tokenizers-cpp/include)
set(TOKENIZERS_CPP_LIB_PATH ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tokenizers-cpp/lib/${ANDROID_ABI})

# ONNX Runtime include and lib paths
set(ONNX_RUNTIME_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime/include)
set(ONNX_RUNTIME_LIB_PATH ${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime/lib/${ANDROID_ABI})

# Kiwi Korean Morphological Analyzer (BM25용)
set(KIWI_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/kiwi/include)
set(KIWI_LIB_PATH ${CMAKE_CURRENT_SOURCE_DIR}/third_party/kiwi/lib/${ANDROID_ABI})

# 1. native library
add_library(
        ecovector
        SHARED
        jni/NativeEcoVectorStore.cpp
        jni/NativeVectorRetriever.cpp
        jni/NativeObxVectorRetriever.cpp
        jni/NativeBM25Retriever.cpp
        jni/NativeEnsembleRetriever.cpp
        src/eco_vector_store/EcoVectorStore.cpp
        src/tokenizer/Tokenizer.cpp
        src/chunker/TokenAwareChunker.cpp
        src/onnx/OnnxRuntime.cpp
        src/object_box/ObxManager.cpp
        src/object_box/schema.obx.cpp
        src/eco_vector/EcoVectorIndex.cpp
        src/bm25/BM25Index.cpp

        src/reranker/LCSReranker.cpp
        src/retriever/IRetriever.cpp
        src/retriever/BM25Retriever.cpp
        src/retriever/VectorRetriever.cpp
        src/retriever/ObxVectorRetriever.cpp
        src/retriever/EnsembleRetriever.cpp
        src/kiwi/KiwiTokenizer.cpp
        src/embedder/Embedder.cpp
        third_party/sqlite3/sqlite3.c
)


# 2. include headers
target_include_directories(
    ecovector
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/jni
    ${CMAKE_CURRENT_SOURCE_DIR}/src/tokenizer
    ${CMAKE_CURRENT_SOURCE_DIR}/src/chunker
    ${CMAKE_CURRENT_SOURCE_DIR}/src/onnx
    ${CMAKE_CURRENT_SOURCE_DIR}/src/object_box
    ${CMAKE_CURRENT_SOURCE_DIR}/src/eco_vector
    ${CMAKE_CURRENT_SOURCE_DIR}/src/bm25

    ${CMAKE_CURRENT_SOURCE_DIR}/src/common
    ${CMAKE_CURRENT_SOURCE_DIR}/src/reranker
    ${CMAKE_CURRENT_SOURCE_DIR}/src/retriever
    ${CMAKE_CURRENT_SOURCE_DIR}/src/kiwi
    ${CMAKE_CURRENT_SOURCE_DIR}/src/eco_vector_store
    ${CMAKE_CURRENT_SOURCE_DIR}/src/embedder
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${KIWI_INCLUDE}
    ${TOKENIZERS_CPP_INCLUDE}
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/nlohmann
    ${ONNX_RUNTIME_INCLUDE}
    ${OBJECTBOX_INCLUDE}
    ${FAISS_INCLUDE}
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/sqlite3
    ${flatbuffers_SOURCE_DIR}/include
)

# 3. link libraries
target_link_libraries(
        ecovector
        ${TOKENIZERS_CPP_LIB_PATH}/libtokenizers_cpp.a
        ${TOKENIZERS_CPP_LIB_PATH}/libtokenizers_c.a
        ${TOKENIZERS_CPP_LIB_PATH}/libsentencepiece.a
        flatbuffers
        ${FAISS_LIB_PATH}/libfaiss.a
        ${FAISS_LIB_PATH}/libopenblas.a
        ${FAISS_LIB_PATH}/libomp.a
        ${OBJECTBOX_LIB_PATH}/libobjectbox-jni.so
        ${ONNX_RUNTIME_LIB_PATH}/libonnxruntime.so
        ${KIWI_LIB_PATH}/libkiwi_static.a
        ${KIWI_LIB_PATH}/libstreamvbyte.a
        android
        log
)
