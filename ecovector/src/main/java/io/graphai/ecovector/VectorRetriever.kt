package io.graphai.ecovector

class VectorRetriever internal constructor(
    params: VectorRetrieverParams = VectorRetrieverParams(),
    override val name: String = "Vector"
) : Retriever {

    private val handle: Long
    var params: VectorRetrieverParams = params
        private set

    init {
        handle = NativeVectorRetriever.create(
            params.efSearch, params.nprobe, params.topK
        )
        require(handle != 0L) { "Failed to create VectorRetriever" }
    }

    override fun retrieve(query: String): List<SearchResult> {
        val json = NativeVectorRetriever.retrieveByText(handle, query)
        return parseSearchResults(json)
    }

    override fun retrieve(bundle: QueryBundle): List<SearchResult> {
        val embedding = bundle.embedding
            ?: throw IllegalArgumentException("VectorRetriever requires embedding in QueryBundle")
        return retrieve(embedding)
    }

    fun retrieve(embedding: FloatArray): List<SearchResult> {
        val json = NativeVectorRetriever.retrieveByEmbedding(handle, embedding)
        return parseSearchResults(json)
    }

    override val isReady: Boolean
        get() = NativeVectorRetriever.isReady(handle)

    /** Native handle for advanced use (e.g., benchmark registration). */
    val nativeHandle: Long get() = handle

    override fun close() {
        NativeVectorRetriever.destroy(handle)
    }
}
