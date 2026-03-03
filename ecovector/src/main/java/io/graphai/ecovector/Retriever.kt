package io.graphai.ecovector

/**
 * Unified retriever interface for all search strategies.
 *
 * Implementations include vector search ([VectorRetrieverParams]),
 * BM25 full-text search ([BM25RetrieverParams]),
 * and ensemble/hybrid search ([EnsembleRetriever]).
 */
interface Retriever : AutoCloseable {
    /**
     * Retrieve relevant chunks by raw query text.
     * Uses the retriever's configured topK from its Params.
     *
     * @param query Raw query text (will be tokenized/embedded internally)
     * @return List of [SearchResult] ordered by relevance
     */
    fun retrieve(query: String): List<SearchResult>

    /**
     * Retrieve relevant chunks using a pre-computed [QueryBundle].
     * Uses the retriever's configured topK from its Params.
     *
     * @param bundle Pre-computed query bundle (embedding, kiwiTokens, etc.)
     * @return List of [SearchResult] ordered by relevance
     */
    fun retrieve(bundle: QueryBundle): List<SearchResult>

    /** Human-readable name of this retriever (e.g., "VectorRetriever", "BM25Retriever"). */
    val name: String

    /** Whether the retriever is initialized and ready for queries. */
    val isReady: Boolean

}
