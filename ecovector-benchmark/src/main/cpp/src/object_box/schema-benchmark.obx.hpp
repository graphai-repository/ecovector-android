// Benchmark ObjectBox schema — Query + GroundTruth
#pragma once

#include <cstdbool>
#include <cstdint>
#include "flatbuffers/flatbuffers.h"
#include "objectbox.h"
#include "objectbox.hpp"

namespace ecovector_bench {

struct Query_;

struct Query {
    obx_id id;
    std::string content;
    std::vector<float> vector;
    std::vector<uint8_t> token_ids;
    std::string _id;
    std::vector<uint8_t> kiwi_tokens;
    int64_t created_at = 0;
    std::string target_types;
    std::string categories;
    std::string split;
    std::string document_external_id;
    std::string refined_query;

    struct _OBX_MetaInfo {
        static constexpr obx_schema_id entityId() { return 1; }
        static void setObjectId(Query& object, obx_id newId) { object.id = newId; }
        static void toFlatBuffer(flatbuffers::FlatBufferBuilder& fbb, const Query& object);
        static Query fromFlatBuffer(const void* data, size_t size);
        static std::unique_ptr<Query> newFromFlatBuffer(const void* data, size_t size);
        static void fromFlatBuffer(const void* data, size_t size, Query& outObject);
    };
};

struct Query_ {
    static const obx::Property<Query, OBXPropertyType_Long> id;
    static const obx::Property<Query, OBXPropertyType_String> content;
    static const obx::Property<Query, OBXPropertyType_FloatVector> vector;
    static const obx::Property<Query, OBXPropertyType_ByteVector> token_ids;
    static const obx::Property<Query, OBXPropertyType_String> _id;
    static const obx::Property<Query, OBXPropertyType_ByteVector> kiwi_tokens;
    static const obx::Property<Query, OBXPropertyType_Long> created_at;
    static const obx::Property<Query, OBXPropertyType_String> target_types;
    static const obx::Property<Query, OBXPropertyType_String> categories;
    static const obx::Property<Query, OBXPropertyType_String> split;
    static const obx::Property<Query, OBXPropertyType_String> document_external_id;
    static const obx::Property<Query, OBXPropertyType_String> refined_query;
};

struct GroundTruth_;

struct GroundTruth {
    obx_id id;
    std::string query_id;
    std::string doc_id;

    struct _OBX_MetaInfo {
        static constexpr obx_schema_id entityId() { return 2; }
        static void setObjectId(GroundTruth& object, obx_id newId) { object.id = newId; }
        static void toFlatBuffer(flatbuffers::FlatBufferBuilder& fbb, const GroundTruth& object);
        static GroundTruth fromFlatBuffer(const void* data, size_t size);
        static std::unique_ptr<GroundTruth> newFromFlatBuffer(const void* data, size_t size);
        static void fromFlatBuffer(const void* data, size_t size, GroundTruth& outObject);
    };
};

struct GroundTruth_ {
    static const obx::Property<GroundTruth, OBXPropertyType_Long> id;
    static const obx::Property<GroundTruth, OBXPropertyType_String> query_id;
    static const obx::Property<GroundTruth, OBXPropertyType_String> doc_id;
};

}  // namespace ecovector_bench
