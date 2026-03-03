#include "IRetriever.h"
#include "../object_box/ObxManager.h"  // ChunkSearchResult complete type

namespace ecovector {

std::vector<ChunkSearchResult> IRetriever::retrieve(const QueryBundle& query) {
    return retrieve(query, getDefaultParams());
}

} // namespace ecovector
