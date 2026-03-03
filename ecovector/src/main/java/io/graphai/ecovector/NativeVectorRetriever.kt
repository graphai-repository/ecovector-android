package io.graphai.ecovector

/**
 * JNI bridge for the native vector retriever.
 *
 * Manages a native handle to a C++ VectorRetriever instance that performs
 * FAISS cluster probing + HNSW search.
 *
 * All methods accept/return a native handle (Long pointer) that must be
 * created via [create] and released via [destroy].
 */
internal object NativeVectorRetriever {
    /**
     * Create a native VectorRetriever instance.
     *
     * @param efSearch HNSW ef_search parameter
     * @param nprobe Number of FAISS clusters to probe
     * @param topK Default number of results
     * @return Native handle (pointer)
     */
    external fun create(efSearch: Int, nprobe: Int, topK: Int): Long

    /**
     * Retrieve by pre-computed embedding vector.
     * Uses the retriever's configured params.topK.
     *
     * @param handle Native handle from [create]
     * @param embedding Query embedding vector
     * @return JSON array of search results
     */
    external fun retrieveByEmbedding(handle: Long, embedding: FloatArray): String

    /**
     * Retrieve by raw query text (embeds internally).
     * Uses the retriever's configured params.topK.
     *
     * @param handle Native handle from [create]
     * @param queryText Raw query text
     * @return JSON array of search results
     */
    external fun retrieveByText(handle: Long, queryText: String): String

    /**
     * Check if the retriever is initialized and ready for queries.
     *
     * @param handle Native handle from [create]
     * @return true if ready
     */
    external fun isReady(handle: Long): Boolean

    /**
     * Destroy the native VectorRetriever instance and free resources.
     *
     * @param handle Native handle from [create]
     */
    external fun destroy(handle: Long)
}
