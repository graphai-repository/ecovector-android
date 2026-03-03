package io.graphai.ecovector

class ObxVectorRetriever internal constructor(
    params: ObxVectorRetrieverParams = ObxVectorRetrieverParams(),
    override val name: String = "ObxVector"
) : Retriever {

    private val handle: Long
    var params: ObxVectorRetrieverParams = params
        private set

    init {
        handle = NativeObxVectorRetriever.create(params.maxResultCount, params.topK)
        require(handle != 0L) { "Failed to create ObxVectorRetriever" }
    }

    override fun retrieve(query: String): List<SearchResult> {
        throw UnsupportedOperationException(
            "ObxVectorRetriever requires embedding. Use via benchmark pipeline."
        )
    }

    override fun retrieve(bundle: QueryBundle): List<SearchResult> {
        throw UnsupportedOperationException(
            "ObxVectorRetriever is benchmark-only. Use nativeHandle for benchmark registration."
        )
    }

    override val isReady: Boolean
        get() = NativeObxVectorRetriever.isReady(handle)

    val nativeHandle: Long get() = handle

    override fun close() {
        NativeObxVectorRetriever.destroy(handle)
    }
}
