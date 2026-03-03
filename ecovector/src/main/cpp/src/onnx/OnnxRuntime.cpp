#include "OnnxRuntime.h"
#include "Tokenizer.h"
#include <onnxruntime_cxx_api.h>
#include <fstream>
#include <sstream>
#include <cstring>
#include <cmath>
#include <chrono>
#include <android/log.h>

#define LOG_TAG "OnnxRuntime"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// Global ONNX Runtime environment (must be created before any sessions)
static Ort::Env g_ort_env(ORT_LOGGING_LEVEL_WARNING, "akasic_db_edge");

OnnxRuntime::OnnxRuntime() : session_(nullptr) {
    tokenizer_ = std::make_unique<ecovector::Tokenizer>();
    LOGI("OnnxRuntime instance created");
}

OnnxRuntime::~OnnxRuntime() {
    // Session is cleaned up automatically via unique_ptr
    LOGI("OnnxRuntime instance destroyed");
}

bool OnnxRuntime::load(const std::string& model_path) {
    LOGI("Loading ONNX model from: %s", model_path.c_str());

    // Check if model file exists
    std::ifstream model_file(model_path, std::ios::binary);
    if (!model_file.is_open()) {
        LOGE("Model file not found: %s", model_path.c_str());
        return false;
    }
    model_file.close();

    try {
        // Create session options
        // Match Python onnxruntime default settings for consistent inference results
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(2);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);

        // Create session
        Ort::Session* raw_session = new Ort::Session(
            g_ort_env,
            model_path.c_str(),
            session_options
        );

        session_ = std::unique_ptr<Ort::Session>(raw_session);

        // Get model metadata
        size_t num_input_nodes = session_->GetInputCount();
        size_t num_output_nodes = session_->GetOutputCount();

        LOGD("Model has %zu input nodes, %zu output nodes",
             num_input_nodes, num_output_nodes);

        LOGI("ONNX model loaded successfully");
        return true;

    } catch (const Ort::Exception& e) {
        LOGE("ONNX Runtime error: %s", e.what());
        return false;
    } catch (const std::exception& e) {
        LOGE("Error: %s", e.what());
        return false;
    }
}

std::vector<float> OnnxRuntime::embed(const std::string& text, std::vector<int32_t>* outTokenIds) {
    auto start_time = std::chrono::high_resolution_clock::now();
    LOGI("[EMBED] Starting embedding process");
    LOGD("[EMBED] Input text (first 100 chars): %s", text.substr(0, 100).c_str());

    std::vector<float> embedding;

    // Check if session is initialized
    if (!session_) {
        LOGE("[EMBED] ERROR: Session not initialized");
        return embedding;
    }

    // Step 1: Tokenize input text
    if (!tokenizer_) {
        LOGE("[EMBED] ERROR: Tokenizer not initialized");
        return embedding;
    }

    auto tokenize_start = std::chrono::high_resolution_clock::now();
    std::vector<int32_t> token_ids = tokenizer_->encode(text);
    auto tokenize_end = std::chrono::high_resolution_clock::now();
    double tokenize_ms = std::chrono::duration<double, std::milli>(tokenize_end - tokenize_start).count();

    if (token_ids.empty()) {
        LOGE("[EMBED] ERROR: Failed to tokenize input text");
        return embedding;
    }

    // Return token IDs if requested (before padding removal)
    if (outTokenIds != nullptr) {
        *outTokenIds = token_ids;
    }

    LOGD("[EMBED] Step 1 - Tokenization: %zu tokens in %.2fms", token_ids.size(), tokenize_ms);

    try {
        // Convert token_ids to int64 for ONNX input
        std::vector<int64_t> input_ids(token_ids.begin(), token_ids.end());

        // Prepare attention_mask and token_type_ids
        // No padding in single inference — all tokens are real, so attention_mask is all 1s
        std::vector<int64_t> attention_mask(input_ids.size(), 1);
        std::vector<int64_t> token_type_ids(input_ids.size(), 0);

        // Create input tensor shape: [1, sequence_length]
        std::vector<int64_t> input_shape = {1, static_cast<int64_t>(input_ids.size())};

        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        // Create input tensors
        Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info,
            input_ids.data(),
            input_ids.size(),
            input_shape.data(),
            input_shape.size()
        );

        Ort::Value token_type_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info,
            token_type_ids.data(),
            token_type_ids.size(),
            input_shape.data(),
            input_shape.size()
        );

        Ort::Value attention_mask_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info,
            attention_mask.data(),
            attention_mask.size(),
            input_shape.data(),
            input_shape.size()
        );

        LOGD("[EMBED] Step 2 - Input tensors prepared: shape [1, %zu]", input_ids.size());

        // Get input/output names dynamically from model to ensure correct order
        Ort::AllocatorWithDefaultOptions allocator;
        std::vector<std::string> input_names_str;
        for (size_t i = 0; i < session_->GetInputCount(); ++i) {
            std::string name = session_->GetInputNameAllocated(i, allocator).get();
            input_names_str.push_back(name);
            LOGD("[EMBED] Model Input[%zu]: %s", i, name.c_str());
        }

        // Build input tensors in model's expected order
        std::vector<Ort::Value> input_tensors;
        std::vector<const char*> input_names;

        for (const auto& name : input_names_str) {
            input_names.push_back(name.c_str());
            if (name == "input_ids") {
                input_tensors.push_back(std::move(input_tensor));
            } else if (name == "token_type_ids") {
                input_tensors.push_back(std::move(token_type_tensor));
            } else if (name == "attention_mask") {
                input_tensors.push_back(std::move(attention_mask_tensor));
            } else {
                LOGE("[EMBED] ERROR: Unexpected input name '%s'", name.c_str());
            }
        }

        LOGD("[EMBED] Building tensors in model order: %zu inputs", input_names.size());

        // Get output names dynamically as well
        std::vector<std::string> output_names_str;
        for (size_t i = 0; i < session_->GetOutputCount(); ++i) {
            std::string name = session_->GetOutputNameAllocated(i, allocator).get();
            output_names_str.push_back(name);
            LOGD("[EMBED] Model Output[%zu]: %s", i, name.c_str());
        }

        // Build output names array
        std::vector<const char*> output_names_cstr;
        for (const auto& name : output_names_str) {
            output_names_cstr.push_back(name.c_str());
        }

        // Run inference
        auto inference_start = std::chrono::high_resolution_clock::now();
        std::vector<Ort::Value> output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            input_names.data(), input_tensors.data(), input_tensors.size(),
            output_names_cstr.data(), output_names_cstr.size()
        );
        auto inference_end = std::chrono::high_resolution_clock::now();
        double inference_ms = std::chrono::duration<double, std::milli>(inference_end - inference_start).count();

        LOGD("[EMBED] Step 3 - ONNX inference completed in %.2fms", inference_ms);

        // Extract output embedding
        if (output_tensors.size() > 0) {
            Ort::Value& output_tensor = output_tensors[0];

            // Get output shape: [1, sequence_length, hidden_size]
            auto output_info = output_tensor.GetTensorTypeAndShapeInfo();
            std::vector<int64_t> output_shape = output_info.GetShape();

            if (output_shape.size() != 3) {
                LOGE("Unexpected output shape size: %zu (expected 3)", output_shape.size());
                // Log shape for debugging
                std::string shape_str = "[";
                for (size_t i = 0; i < output_shape.size(); i++) {
                    if (i > 0) shape_str += ", ";
                    shape_str += std::to_string(output_shape[i]);
                }
                shape_str += "]";
                LOGE("Output shape: %s", shape_str.c_str());
            } else {
                // Extract shape as size_t (matching MobileRAG)
                size_t batch = output_shape[0];
                size_t seq_len = output_shape[1];
                size_t hidden_size = output_shape[2];

                LOGD("Output shape: batch=%zu, seq_len=%zu, hidden_size=%zu", batch, seq_len, hidden_size);

                // Get output data (const pointer, matching MobileRAG)
                const float *raw = output_tensor.GetTensorData<float>();

                embedding.assign(hidden_size, 0.0f);
                float mask_sum = 0.0f;

                // Mean pooling: sum embeddings of non-padding tokens
                const int64_t *mask = attention_mask.data();
                for (size_t t = 0; t < seq_len; ++t) {
                    if (mask[t] == 0)
                        continue;

                    const float *cur_outv = raw + t * hidden_size;

                    for (size_t h = 0; h < hidden_size; ++h) {
                        embedding[h] += mask[t] * cur_outv[h];
                    }
                    mask_sum += mask[t];
                }

                // Divide by sum of masks to get mean
                for (size_t h = 0; h < hidden_size; ++h)
                    embedding[h] /= mask_sum;
            }
        }

    } catch (const Ort::Exception& e) {
        LOGE("[EMBED] ERROR - ONNX Runtime: %s", e.what());
        return embedding;
    } catch (const std::exception& e) {
        LOGE("[EMBED] ERROR - Exception: %s", e.what());
        return embedding;
    }

    // Auto-normalize for consistent similarity search
    auto normalize_start = std::chrono::high_resolution_clock::now();
    this->normalize(embedding);
    auto normalize_end = std::chrono::high_resolution_clock::now();
    double normalize_ms = std::chrono::duration<double, std::milli>(normalize_end - normalize_start).count();

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(total_end - start_time).count();

    if (!embedding.empty()) {
        LOGD("[EMBED] Step 5 - L2 Normalization: done (%.2fms)", normalize_ms);
        LOGI("[EMBED] COMPLETED | Dimensions: %zu | Total time: %.2fms", embedding.size(), total_ms);
    } else {
        LOGE("[EMBED] ERROR: Embedding is empty");
    }

    return embedding;
}

std::vector<std::vector<float>> OnnxRuntime::embedBatch(const std::vector<std::string>& texts,
                                                       std::vector<std::vector<int32_t>>* outTokenIds) {
    auto batch_start_time = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<float>> embeddings;

    // Validation
    if (texts.empty()) {
        LOGE("[BATCH] ERROR: Batch texts is empty");
        return embeddings;
    }

    if (!session_) {
        LOGE("[BATCH] ERROR: Session not initialized");
        return embeddings;
    }

    if (!tokenizer_) {
        LOGE("[BATCH] ERROR: Tokenizer not initialized");
        return embeddings;
    }

    try {
        LOGD("[BATCH] Starting batch embedding | batch_size=%zu", texts.size());

        // Step 1: Tokenize all texts
        auto tokenize_batch_start = std::chrono::high_resolution_clock::now();
        std::vector<std::vector<int64_t>> batch_input_ids;
        std::vector<std::vector<int64_t>> batch_attention_masks;
        std::vector<std::vector<int32_t>> allTokenIds;  // Store token IDs before padding
        int64_t max_seq_len = 0;

        if (outTokenIds != nullptr) {
            allTokenIds.reserve(texts.size());
        }

        for (size_t i = 0; i < texts.size(); i++) {
            std::vector<int32_t> token_ids = tokenizer_->encode(texts[i]);
            if (token_ids.empty()) {
                LOGE("[BATCH] ERROR: Failed to tokenize text at index %zu", i);
                return embeddings;
            }

            // Store original token IDs (before padding)
            if (outTokenIds != nullptr) {
                allTokenIds.push_back(token_ids);
            }

            // Convert to int64
            std::vector<int64_t> input_ids(token_ids.begin(), token_ids.end());
            int64_t seq_len = static_cast<int64_t>(input_ids.size());
            if (seq_len > max_seq_len) {
                max_seq_len = seq_len;
            }

            batch_input_ids.push_back(input_ids);
        }

        auto tokenize_batch_end = std::chrono::high_resolution_clock::now();
        double tokenize_batch_ms = std::chrono::duration<double, std::milli>(tokenize_batch_end - tokenize_batch_start).count();
        LOGD("[BATCH] Step 1 - Tokenized all texts | max_seq_len=%lld | %.2fms", (long long)max_seq_len, tokenize_batch_ms);

        // Step 2: Dynamic padding — pad all sequences to max_seq_len in batch
        std::vector<int64_t> padded_input_ids;
        std::vector<int64_t> padded_token_type_ids;
        std::vector<int64_t> padded_attention_mask;

        for (size_t i = 0; i < batch_input_ids.size(); i++) {
            std::vector<int64_t>& input_ids = batch_input_ids[i];

            // Add input_ids with padding
            for (int64_t j = 0; j < max_seq_len; j++) {
                if (j < static_cast<int64_t>(input_ids.size())) {
                    padded_input_ids.push_back(input_ids[j]);
                } else {
                    padded_input_ids.push_back(0);  // PAD token
                }
            }

            // Add token_type_ids (all 0 for single sequence)
            for (int64_t j = 0; j < max_seq_len; j++) {
                padded_token_type_ids.push_back(0);
            }

            // Add attention_mask: 1 for real tokens, 0 for padding
            for (int64_t j = 0; j < max_seq_len; j++) {
                if (j < static_cast<int64_t>(input_ids.size())) {
                    padded_attention_mask.push_back(1);
                } else {
                    padded_attention_mask.push_back(0);
                }
            }
        }

        LOGD("[BATCH] Step 2 - Padded batch created | batch_size=%zu | max_seq_len=%lld",
             texts.size(), (long long)max_seq_len);

        // Step 3: Create batch tensors
        std::vector<int64_t> input_shape = {static_cast<int64_t>(texts.size()), max_seq_len};

        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info,
            padded_input_ids.data(),
            padded_input_ids.size(),
            input_shape.data(),
            input_shape.size()
        );

        Ort::Value token_type_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info,
            padded_token_type_ids.data(),
            padded_token_type_ids.size(),
            input_shape.data(),
            input_shape.size()
        );

        Ort::Value attention_mask_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info,
            padded_attention_mask.data(),
            padded_attention_mask.size(),
            input_shape.data(),
            input_shape.size()
        );

        LOGD("[BATCH] Step 3 - Input tensors created | shape [%lld, %lld]",
             (long long)texts.size(), (long long)max_seq_len);

        // Step 4: Run batch inference
        // Get input/output names dynamically from model to ensure correct order
        Ort::AllocatorWithDefaultOptions allocator;
        std::vector<std::string> batch_input_names_str;
        for (size_t i = 0; i < session_->GetInputCount(); ++i) {
            std::string name = session_->GetInputNameAllocated(i, allocator).get();
            batch_input_names_str.push_back(name);
            LOGD("[BATCH] Model Input[%zu]: %s", i, name.c_str());
        }

        // Build input tensors in model's expected order
        auto inference_batch_start = std::chrono::high_resolution_clock::now();
        std::vector<Ort::Value> input_tensors;
        std::vector<const char*> input_names;

        for (const auto& name : batch_input_names_str) {
            input_names.push_back(name.c_str());
            if (name == "input_ids") {
                input_tensors.push_back(std::move(input_tensor));
            } else if (name == "token_type_ids") {
                input_tensors.push_back(std::move(token_type_tensor));
            } else if (name == "attention_mask") {
                input_tensors.push_back(std::move(attention_mask_tensor));
            } else {
                LOGE("[BATCH] ERROR: Unexpected input name '%s'", name.c_str());
            }
        }

        // Get output names dynamically
        std::vector<std::string> batch_output_names_str;
        for (size_t i = 0; i < session_->GetOutputCount(); ++i) {
            std::string name = session_->GetOutputNameAllocated(i, allocator).get();
            batch_output_names_str.push_back(name);
            LOGD("[BATCH] Model Output[%zu]: %s", i, name.c_str());
        }

        std::vector<const char*> output_names;
        for (const auto& name : batch_output_names_str) {
            output_names.push_back(name.c_str());
        }

        std::vector<Ort::Value> output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            input_names.data(), input_tensors.data(), input_tensors.size(),
            output_names.data(), output_names.size()
        );
        auto inference_batch_end = std::chrono::high_resolution_clock::now();
        double inference_batch_ms = std::chrono::duration<double, std::milli>(inference_batch_end - inference_batch_start).count();

        LOGD("[BATCH] Step 4 - Batch inference completed | %.2fms", inference_batch_ms);

        // Step 5: Perform Mean Pooling for each batch item
        if (output_tensors.size() > 0) {
            Ort::Value& output_tensor = output_tensors[0];

            auto output_info = output_tensor.GetTensorTypeAndShapeInfo();
            std::vector<int64_t> output_shape = output_info.GetShape();

            // output_shape should be [batch_size, max_seq_len, hidden_size]
            if (output_shape.size() != 3) {
                LOGE("Batch: Unexpected output shape size: %zu (expected 3)", output_shape.size());
                std::string shape_str = "[";
                for (size_t i = 0; i < output_shape.size(); i++) {
                    if (i > 0) shape_str += ", ";
                    shape_str += std::to_string(output_shape[i]);
                }
                shape_str += "]";
                LOGE("Batch output shape: %s", shape_str.c_str());
            } else {
                // Extract shape variables as size_t
                size_t batch = output_shape[0];
                size_t seq_len = output_shape[1];
                size_t hidden_size = output_shape[2];

                // Get output data (const pointer)
                const float *raw = output_tensor.GetTensorData<float>();

                // Perform Mean Pooling for each batch item
                for (size_t batch_idx = 0; batch_idx < batch; batch_idx++) {
                    std::vector<float> embedding(hidden_size, 0.0f);

                    float mask_sum = 0.0f;
                    const int64_t *mask = padded_attention_mask.data();

                    for (size_t t = 0; t < seq_len; t++) {
                        int64_t mask_index = batch_idx * seq_len + t;
                        if (mask[mask_index] == 0)
                            continue;

                        const float *cur_outv = raw + batch_idx * seq_len * hidden_size + t * hidden_size;

                        for (size_t h = 0; h < hidden_size; h++) {
                            embedding[h] += mask[mask_index] * cur_outv[h];
                        }
                        mask_sum += mask[mask_index];
                    }

                    for (size_t h = 0; h < hidden_size; h++)
                        embedding[h] /= mask_sum;

                    embeddings.push_back(embedding);
                }
            }
        }

        // Auto-normalize all embeddings for consistent similarity search
        auto normalize_batch_start = std::chrono::high_resolution_clock::now();
        this->normalizeBatch(embeddings);
        auto normalize_batch_end = std::chrono::high_resolution_clock::now();
        double normalize_batch_ms = std::chrono::duration<double, std::milli>(normalize_batch_end - normalize_batch_start).count();

        // Return token IDs if requested (convert int32_t to uint32_t)
        if (outTokenIds != nullptr) {
            outTokenIds->clear();
            outTokenIds->reserve(allTokenIds.size());
            for (const auto& tokens : allTokenIds) {
                outTokenIds->push_back(tokens);  // Already int32_t, no conversion needed
            }
        }

        auto batch_end_time = std::chrono::high_resolution_clock::now();
        double total_batch_ms = std::chrono::duration<double, std::milli>(batch_end_time - batch_start_time).count();

        if (embeddings.size() == texts.size()) {
            LOGD("[BATCH] Step 6 - L2 Normalization completed | %.2fms", normalize_batch_ms);
            LOGD("[BATCH] COMPLETED | Items: %zu | Dimensions: %zu | Total time: %.2fms | Avg: %.2fms/item",
                 embeddings.size(), embeddings.empty() ? 0 : embeddings[0].size(),
                 total_batch_ms, total_batch_ms / texts.size());
        } else {
            LOGE("[BATCH] ERROR: Embedding count mismatch | Expected: %zu, Got: %zu",
                 texts.size(), embeddings.size());
        }

    } catch (const Ort::Exception& e) {
        LOGE("[BATCH] ERROR - ONNX Runtime: %s", e.what());
    } catch (const std::exception& e) {
        LOGE("[BATCH] ERROR - Exception: %s", e.what());
    }

    return embeddings;
}

bool OnnxRuntime::loadTokenizer(const std::string& tokenizer_path) {
    LOGI("Loading tokenizer from: %s", tokenizer_path.c_str());

    if (!tokenizer_) {
        tokenizer_ = std::make_unique<ecovector::Tokenizer>();
    }

    bool success = tokenizer_->load(tokenizer_path);
    if (success) {
        LOGI("Tokenizer loaded successfully");
    } else {
        LOGE("Failed to init tokenizer");
    }
    return success;
}

void OnnxRuntime::normalize(std::vector<float>& vector) {
    if (vector.empty()) {
        LOGD("[NORMALIZE] ERROR: Cannot normalize empty vector");
        return;
    }

    // Calculate L2 norm: sqrt(sum(v[i]^2))
    float norm = 0.0f;
    for (float v : vector) {
        norm += v * v;
    }
    norm = std::sqrt(norm);

    // Avoid division by zero (epsilon: 1e-10)
    if (norm < 1e-10f) {
        LOGD("[NORMALIZE] Norm too small (%.2e), skipping", norm);
        return;
    }

    // Normalize: v = v / ||v||
    for (float& v : vector) {
        v /= norm;
    }
    LOGD("[NORMALIZE] Applied L2 normalization (norm=%.6f, dims=%zu)", norm, vector.size());
}

void OnnxRuntime::normalizeBatch(std::vector<std::vector<float>>& embeddings) {
    if (embeddings.empty()) {
        LOGD("[NORMALIZE_BATCH] WARNING: No embeddings to normalize");
        return;
    }

    size_t normalized_count = 0;
    for (auto& embedding : embeddings) {
        if (!embedding.empty()) {
            float norm = 0.0f;
            for (float v : embedding) {
                norm += v * v;
            }
            norm = std::sqrt(norm);

            if (norm >= 1e-10f) {
                for (float& v : embedding) {
                    v /= norm;
                }
                normalized_count++;
            }
        }
    }
    LOGD("[NORMALIZE_BATCH] Normalized %zu/%zu embeddings", normalized_count, embeddings.size());
}

