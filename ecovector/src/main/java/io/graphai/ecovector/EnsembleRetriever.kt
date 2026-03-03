package io.graphai.ecovector

class EnsembleRetriever internal constructor(
    private val handle: Long,
    override val name: String = "Ensemble"
) : Retriever {

    init {
        require(handle != 0L) { "Failed to create EnsembleRetriever" }
    }

    override fun retrieve(query: String): List<SearchResult> {
        val json = NativeEnsembleRetriever.retrieveByText(handle, query)
        return parseSearchResults(json)
    }

    override fun retrieve(bundle: QueryBundle): List<SearchResult> {
        val json = NativeEnsembleRetriever.retrieveByBundle(
            handle, bundle.embedding, bundle.kiwiTokens
        )
        return parseSearchResults(json)
    }

    fun fuse(inputs: List<WeightedResults>): List<SearchResult> {
        val json = NativeEnsembleRetriever.fuse(
            handle, serializeWeightedResults(inputs)
        )
        return parseSearchResults(json)
    }

    override val isReady: Boolean
        get() = NativeEnsembleRetriever.isReady(handle)

    /** Native handle for advanced use (e.g., benchmark registration). */
    val nativeHandle: Long get() = handle

    override fun close() {
        NativeEnsembleRetriever.destroy(handle)
    }
}
