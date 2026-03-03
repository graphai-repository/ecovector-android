package io.graphai.ecovector

class BM25Retriever internal constructor(
    params: BM25RetrieverParams = BM25RetrieverParams(),
    override val name: String = "BM25"
) : Retriever {

    private val handle: Long
    var params: BM25RetrieverParams = params
        private set

    init {
        handle = NativeBM25Retriever.create(
            params.k1, params.b, params.topK,
            params.idfThreshold, params.maxSeedTerms,
            params.candidateMultiplier, params.minCandidates, params.minScore,
            params.rm3Enabled, params.rm3FbDocs, params.rm3FbTerms,
            params.rm3OrigWeight, params.rm3MinDf
        )
        require(handle != 0L) { "Failed to create BM25Retriever" }
    }

    override fun retrieve(query: String): List<SearchResult> {
        val json = NativeBM25Retriever.retrieveByText(handle, query)
        return parseSearchResults(json)
    }

    override fun retrieve(bundle: QueryBundle): List<SearchResult> {
        val kiwiTokens = bundle.kiwiTokens
            ?: throw IllegalArgumentException("BM25Retriever requires kiwiTokens in QueryBundle")
        return retrieve(kiwiTokens)
    }

    fun retrieve(kiwiTokens: IntArray): List<SearchResult> {
        val json = NativeBM25Retriever.retrieveByTokens(handle, kiwiTokens)
        return parseSearchResults(json)
    }

    override val isReady: Boolean
        get() = NativeBM25Retriever.isReady(handle)

    /** Native handle for advanced use (e.g., benchmark registration). */
    val nativeHandle: Long get() = handle

    override fun close() {
        NativeBM25Retriever.destroy(handle)
    }
}
