#pragma once

#include <string>
#include <vector>
#include <cstdint>

namespace ecovector {

struct QueryData {
    uint64_t id = 0;
    std::string externalId;     // _id
    std::string content;        // original query (사용자 원문)
    std::string refinedQuery;   // LLM refined query (임베딩용)
    std::vector<float> vector;
    std::vector<int32_t> tokenIds;
    std::vector<int32_t> kiwiTokens;
    int64_t createdAt = 0;
    std::string targetTypes;    // comma-separated: "call,sms"
    std::string categories;     // comma-separated
    std::string split;          // "valid" or "test" (benchmark split)
};

struct GroundTruthData {
    uint64_t id = 0;
    std::string queryId;    // external _id reference
    std::string docId;      // external _id reference
};

} // namespace ecovector
