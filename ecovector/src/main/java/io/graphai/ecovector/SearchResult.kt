package io.graphai.ecovector

/**
 * A single search result from EcoVectorStore.
 *
 * @param documentId The ID of the source document
 * @param chunkId The ID of the matched chunk
 * @param content The text content of the matched chunk
 * @param score Relevance score (higher is better)
 */
data class SearchResult(
    val documentId: Long,
    val chunkId: Long,
    val content: String,
    val score: Float,
)
