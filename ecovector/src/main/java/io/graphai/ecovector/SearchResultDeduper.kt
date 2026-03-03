package io.graphai.ecovector

/**
 * Document-level deduplication for search results.
 *
 * Retriever results are chunk-level: the same document may appear multiple times
 * via different chunks. This deduper keeps only the highest-scoring chunk per
 * document, preserving the original relevance order.
 *
 * Typical usage:
 * ```
 * val chunks = retriever.retrieve("query")        // raw chunks
 * val docs = SearchResultDeduper.dedup(chunks)     // unique documents
 * val top5 = docs.take(5)                          // slice
 * ```
 */
object SearchResultDeduper {

    /**
     * Remove duplicate documents, keeping the first (highest-scored) chunk per document.
     *
     * Input must be pre-sorted by relevance (as returned by retrievers).
     *
     * @param results Raw chunk-level search results
     * @return Deduplicated results with at most one chunk per document
     */
    fun dedup(results: List<SearchResult>): List<SearchResult> {
        val seen = mutableSetOf<Long>()
        return results.filter { seen.add(it.documentId) }
    }
}
