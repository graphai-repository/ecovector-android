// Benchmark ObjectBox model — Query + GroundTruth (separate DB)
#pragma once

#ifdef __cplusplus
#include <cstdbool>
#include <cstdint>
extern "C" {
#else
#include <stdbool.h>
#include <stdint.h>
#endif
#include "objectbox.h"

static inline OBX_model* create_benchmark_obx_model() {
    OBX_model* model = obx_model();
    if (!model) return NULL;

    obx_model_entity(model, "Query", 1, 1001000000000000001ULL);
    obx_model_property(model, "id", OBXPropertyType_Long, 1, 1001000000000000002ULL);
    obx_model_property_flags(model, OBXPropertyFlags_ID);
    obx_model_property(model, "content", OBXPropertyType_String, 2, 1001000000000000003ULL);
    obx_model_property(model, "vector", OBXPropertyType_FloatVector, 3, 1001000000000000004ULL);
    obx_model_property(model, "token_ids", OBXPropertyType_ByteVector, 4, 1001000000000000005ULL);
    obx_model_property(model, "_id", OBXPropertyType_String, 5, 1001000000000000006ULL);
    obx_model_property(model, "kiwi_tokens", OBXPropertyType_ByteVector, 6, 1001000000000000007ULL);
    obx_model_property(model, "created_at", OBXPropertyType_Long, 7, 1001000000000000008ULL);
    obx_model_property(model, "target_types", OBXPropertyType_String, 8, 1001000000000000009ULL);
    obx_model_property(model, "categories", OBXPropertyType_String, 9, 1001000000000000010ULL);
    obx_model_property(model, "split", OBXPropertyType_String, 10, 1001000000000000011ULL);
    obx_model_property(model, "document_external_id", OBXPropertyType_String, 11, 1001000000000000012ULL);
    obx_model_property(model, "refined_query", OBXPropertyType_String, 12, 1001000000000000013ULL);
    obx_model_property(model, "eval_top_k", OBXPropertyType_Int, 13, 1001000000000000014ULL);
    obx_model_entity_last_property_id(model, 13, 1001000000000000014ULL);

    obx_model_entity(model, "GroundTruth", 2, 1002000000000000001ULL);
    obx_model_property(model, "id", OBXPropertyType_Long, 1, 1002000000000000002ULL);
    obx_model_property_flags(model, OBXPropertyFlags_ID);
    obx_model_property(model, "query_id", OBXPropertyType_String, 2, 1002000000000000003ULL);
    obx_model_property(model, "doc_id", OBXPropertyType_String, 3, 1002000000000000004ULL);
    obx_model_entity_last_property_id(model, 3, 1002000000000000004ULL);

    obx_model_last_entity_id(model, 2, 1002000000000000001ULL);
    obx_model_last_index_id(model, 0, 0);
    return model;
}

#ifdef __cplusplus
}
#endif
