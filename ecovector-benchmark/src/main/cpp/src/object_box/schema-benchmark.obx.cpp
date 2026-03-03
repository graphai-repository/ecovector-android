#include "schema-benchmark.obx.hpp"

// ==================== Query ====================

const obx::Property<ecovector_bench::Query, OBXPropertyType_Long> ecovector_bench::Query_::id(1);
const obx::Property<ecovector_bench::Query, OBXPropertyType_String> ecovector_bench::Query_::content(2);
const obx::Property<ecovector_bench::Query, OBXPropertyType_FloatVector> ecovector_bench::Query_::vector(3);
const obx::Property<ecovector_bench::Query, OBXPropertyType_ByteVector> ecovector_bench::Query_::token_ids(4);
const obx::Property<ecovector_bench::Query, OBXPropertyType_String> ecovector_bench::Query_::_id(5);
const obx::Property<ecovector_bench::Query, OBXPropertyType_ByteVector> ecovector_bench::Query_::kiwi_tokens(6);
const obx::Property<ecovector_bench::Query, OBXPropertyType_Long> ecovector_bench::Query_::created_at(7);
const obx::Property<ecovector_bench::Query, OBXPropertyType_String> ecovector_bench::Query_::target_types(8);
const obx::Property<ecovector_bench::Query, OBXPropertyType_String> ecovector_bench::Query_::categories(9);
const obx::Property<ecovector_bench::Query, OBXPropertyType_String> ecovector_bench::Query_::split(10);
const obx::Property<ecovector_bench::Query, OBXPropertyType_String> ecovector_bench::Query_::document_external_id(11);
const obx::Property<ecovector_bench::Query, OBXPropertyType_String> ecovector_bench::Query_::refined_query(12);

void ecovector_bench::Query::_OBX_MetaInfo::toFlatBuffer(
        flatbuffers::FlatBufferBuilder& fbb, const ecovector_bench::Query& object) {
    fbb.Clear();
    auto offsetcontent = fbb.CreateString(object.content);
    auto offsetvector = fbb.CreateVector(object.vector);
    auto offsettoken_ids = fbb.CreateVector(object.token_ids);
    auto offset_id = fbb.CreateString(object._id);
    auto offsetkiwi_tokens = fbb.CreateVector(object.kiwi_tokens);
    auto offsettarget_types = fbb.CreateString(object.target_types);
    auto offsetcategories = fbb.CreateString(object.categories);
    auto offsetsplit = fbb.CreateString(object.split);
    auto offsetdocument_external_id = fbb.CreateString(object.document_external_id);
    auto offsetrefined_query = fbb.CreateString(object.refined_query);
    flatbuffers::uoffset_t fbStart = fbb.StartTable();
    fbb.AddElement(4, object.id);           // prop 1: id
    fbb.AddOffset(6, offsetcontent);         // prop 2: content
    fbb.AddOffset(8, offsetvector);          // prop 3: vector
    fbb.AddOffset(10, offsettoken_ids);      // prop 4: token_ids
    fbb.AddOffset(12, offset_id);            // prop 5: _id
    fbb.AddOffset(14, offsetkiwi_tokens);    // prop 6: kiwi_tokens
    fbb.AddElement(16, object.created_at);   // prop 7: created_at
    fbb.AddOffset(18, offsettarget_types);   // prop 8: target_types
    fbb.AddOffset(20, offsetcategories);     // prop 9: categories
    fbb.AddOffset(22, offsetsplit);          // prop 10: split
    fbb.AddOffset(24, offsetdocument_external_id); // prop 11: document_external_id
    fbb.AddOffset(26, offsetrefined_query);  // prop 12: refined_query
    flatbuffers::Offset<flatbuffers::Table> offset;
    offset.o = fbb.EndTable(fbStart);
    fbb.Finish(offset);
}

ecovector_bench::Query ecovector_bench::Query::_OBX_MetaInfo::fromFlatBuffer(const void* data, size_t size) {
    ecovector_bench::Query object;
    fromFlatBuffer(data, size, object);
    return object;
}

std::unique_ptr<ecovector_bench::Query> ecovector_bench::Query::_OBX_MetaInfo::newFromFlatBuffer(const void* data, size_t size) {
    auto object = std::make_unique<ecovector_bench::Query>();
    fromFlatBuffer(data, size, *object);
    return object;
}

void ecovector_bench::Query::_OBX_MetaInfo::fromFlatBuffer(
        const void* data, size_t, ecovector_bench::Query& o) {
    const auto* table = flatbuffers::GetRoot<flatbuffers::Table>(data);
    assert(table);
    o.id = table->GetField<obx_id>(4, 0);
    { auto* p = table->GetPointer<const flatbuffers::String*>(6);
      if (p) o.content.assign(p->c_str(), p->size()); else o.content.clear(); }
    { auto* p = table->GetPointer<const flatbuffers::Vector<float>*>(8);
      if (p) o.vector.assign(p->begin(), p->end()); else o.vector.clear(); }
    { auto* p = table->GetPointer<const flatbuffers::Vector<uint8_t>*>(10);
      if (p) o.token_ids.assign(p->begin(), p->end()); else o.token_ids.clear(); }
    { auto* p = table->GetPointer<const flatbuffers::String*>(12);
      if (p) o._id.assign(p->c_str(), p->size()); else o._id.clear(); }
    { auto* p = table->GetPointer<const flatbuffers::Vector<uint8_t>*>(14);
      if (p) o.kiwi_tokens.assign(p->begin(), p->end()); else o.kiwi_tokens.clear(); }
    o.created_at = table->GetField<int64_t>(16, 0);
    { auto* p = table->GetPointer<const flatbuffers::String*>(18);
      if (p) o.target_types.assign(p->c_str(), p->size()); else o.target_types.clear(); }
    { auto* p = table->GetPointer<const flatbuffers::String*>(20);
      if (p) o.categories.assign(p->c_str(), p->size()); else o.categories.clear(); }
    { auto* p = table->GetPointer<const flatbuffers::String*>(22);
      if (p) o.split.assign(p->c_str(), p->size()); else o.split.clear(); }
    { auto* p = table->GetPointer<const flatbuffers::String*>(24);
      if (p) o.document_external_id.assign(p->c_str(), p->size()); else o.document_external_id.clear(); }
    { auto* p = table->GetPointer<const flatbuffers::String*>(26);
      if (p) o.refined_query.assign(p->c_str(), p->size()); else o.refined_query.clear(); }
}

// ==================== GroundTruth ====================

const obx::Property<ecovector_bench::GroundTruth, OBXPropertyType_Long> ecovector_bench::GroundTruth_::id(1);
const obx::Property<ecovector_bench::GroundTruth, OBXPropertyType_String> ecovector_bench::GroundTruth_::query_id(2);
const obx::Property<ecovector_bench::GroundTruth, OBXPropertyType_String> ecovector_bench::GroundTruth_::doc_id(3);

void ecovector_bench::GroundTruth::_OBX_MetaInfo::toFlatBuffer(
        flatbuffers::FlatBufferBuilder& fbb, const ecovector_bench::GroundTruth& object) {
    fbb.Clear();
    auto offsetquery_id = fbb.CreateString(object.query_id);
    auto offsetdoc_id = fbb.CreateString(object.doc_id);
    flatbuffers::uoffset_t fbStart = fbb.StartTable();
    fbb.AddElement(4, object.id);
    fbb.AddOffset(6, offsetquery_id);
    fbb.AddOffset(8, offsetdoc_id);
    flatbuffers::Offset<flatbuffers::Table> offset;
    offset.o = fbb.EndTable(fbStart);
    fbb.Finish(offset);
}

ecovector_bench::GroundTruth ecovector_bench::GroundTruth::_OBX_MetaInfo::fromFlatBuffer(const void* data, size_t size) {
    ecovector_bench::GroundTruth object;
    fromFlatBuffer(data, size, object);
    return object;
}

std::unique_ptr<ecovector_bench::GroundTruth> ecovector_bench::GroundTruth::_OBX_MetaInfo::newFromFlatBuffer(const void* data, size_t size) {
    auto object = std::make_unique<ecovector_bench::GroundTruth>();
    fromFlatBuffer(data, size, *object);
    return object;
}

void ecovector_bench::GroundTruth::_OBX_MetaInfo::fromFlatBuffer(
        const void* data, size_t, ecovector_bench::GroundTruth& o) {
    const auto* table = flatbuffers::GetRoot<flatbuffers::Table>(data);
    assert(table);
    o.id = table->GetField<obx_id>(4, 0);
    { auto* p = table->GetPointer<const flatbuffers::String*>(6);
      if (p) o.query_id.assign(p->c_str(), p->size()); else o.query_id.clear(); }
    { auto* p = table->GetPointer<const flatbuffers::String*>(8);
      if (p) o.doc_id.assign(p->c_str(), p->size()); else o.doc_id.clear(); }
}
