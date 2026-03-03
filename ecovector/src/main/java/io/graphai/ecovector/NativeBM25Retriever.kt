package io.graphai.ecovector

/**
 * JNI bridge for the native BM25 full-text retriever.
 *
 * Manages a native handle to a C++ BM25Retriever instance that performs
 * Okapi BM25 scoring with Kiwi morphological tokenization.
 *
 * All methods accept/return a native handle (Long pointer) that must be
 * created via [create] and released via [destroy].
 */
internal object NativeBM25Retriever {
    /**
     * Create a native BM25Retriever instance.
     *
     * @param k1 Term frequency saturation parameter
     * @param b Document length normalization
     * @param topK Default number of results
     * @return Native handle (pointer)
     */
    external fun create(
        k1: Float, b: Float, topK: Int,
        idfThreshold: Float, maxSeedTerms: Int,
        candidateMultiplier: Int, minCandidates: Int, minScore: Float,
        rm3Enabled: Boolean, rm3FbDocs: Int, rm3FbTerms: Int,
        rm3OrigWeight: Float, rm3MinDf: Int
    ): Long

    /**
     * Retrieve by pre-computed Kiwi morpheme tokens.
     * Uses the retriever's configured params.topK.
     *
     * @param handle Native handle from [create]
     * @param kiwiTokens Pre-computed Kiwi morpheme hash tokens
     * @return JSON array of search results
     */
    external fun retrieveByTokens(handle: Long, kiwiTokens: IntArray): String

    /**
     * Retrieve by raw query text (tokenizes internally via Kiwi).
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
     * Destroy the native BM25Retriever instance and free resources.
     *
     * @param handle Native handle from [create]
     */
    external fun destroy(handle: Long)
}
