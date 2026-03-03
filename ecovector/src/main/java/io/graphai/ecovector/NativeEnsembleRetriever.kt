package io.graphai.ecovector

/**
 * JNI bridge for the native ensemble (hybrid) retriever.
 *
 * Manages a native handle to a C++ EnsembleRetriever that combines
 * multiple retrievers using Reciprocal Rank Fusion (RRF) or score-based fusion.
 *
 * All methods accept/return a native handle (Long pointer) that must be
 * created via [createFromHandles] and released via [destroy].
 */
internal object NativeEnsembleRetriever {
    /**
     * Create a native EnsembleRetriever from existing retriever handles.
     *
     * @param retrieverHandles Native handles of sub-retrievers
     * @param weights Corresponding weights for each sub-retriever
     * @param fusionMethod 0=RSF, 1=RRF
     * @param rrfK RRF constant
     * @param topK Number of final results
     * @param parallel Execute sub-retrievers in parallel
     * @return Native handle (pointer)
     */
    external fun createFromHandles(
        retrieverHandles: LongArray,
        weights: FloatArray,
        fusionMethod: Int,
        rrfK: Float,
        topK: Int,
        parallel: Boolean
    ): Long

    /**
     * Retrieve by raw query text (delegates to sub-retrievers internally).
     * Uses the retriever's configured params.topK.
     *
     * @param handle Native handle from [createFromHandles]
     * @param queryText Raw query text
     * @return JSON array of fused search results
     */
    external fun retrieveByText(handle: Long, queryText: String): String

    /**
     * Retrieve using a pre-computed query bundle.
     * Uses the retriever's configured params.topK.
     *
     * @param handle Native handle from [createFromHandles]
     * @param embedding Pre-computed embedding vector (nullable, for vector retriever)
     * @param kiwiTokens Pre-computed Kiwi tokens (nullable, for BM25 retriever)
     * @return JSON array of fused search results
     */
    external fun retrieveByBundle(
        handle: Long,
        embedding: FloatArray?,
        kiwiTokens: IntArray?
    ): String

    /**
     * Fuse pre-computed results from multiple retrievers.
     *
     * @param handle Native handle from [createFromHandles]
     * @param resultsJson JSON string containing weighted result sets
     * @return JSON array of fused search results
     */
    external fun fuse(handle: Long, resultsJson: String): String

    /**
     * Check if the ensemble retriever and all sub-retrievers are ready.
     *
     * @param handle Native handle from [createFromHandles]
     * @return true if all sub-retrievers are ready
     */
    external fun isReady(handle: Long): Boolean

    /**
     * Destroy the native EnsembleRetriever instance and free resources.
     * Does NOT destroy sub-retrievers — those must be destroyed separately.
     *
     * @param handle Native handle from [createFromHandles]
     */
    external fun destroy(handle: Long)
}
