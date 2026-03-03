package io.graphai.ecovector

import android.content.Context
import android.util.Log
import java.io.Closeable

/**
 * EcoVectorStore -- Hybrid vector search SDK for Android.
 *
 * Combines neural embedding (ONNX), vector search (FAISS + HNSW),
 * full-text search (BM25 + Kiwi morphology), and RRF ranking.
 *
 * Usage:
 * ```kotlin
 * val eco = EcoVectorStore.create(context)
 * eco.addDocument("document content", "title")
 * eco.buildIndex()
 *
 * val vector = eco.createVectorRetriever(VectorRetrieverParams(efSearch = 64, nprobe = 4, topK = 50))
 * val bm25 = eco.createBM25Retriever(BM25RetrieverParams(k1 = 0.9f, topK = 50))
 * val ensemble = eco.createEnsembleRetriever(
 *     components = listOf(vector weighted 0.7f, bm25 weighted 0.3f),
 *     topK = 10
 * )
 * val results = ensemble.retrieve("query")
 *
 * eco.close()
 * ```
 */
class EcoVectorStore private constructor(
    private val context: Context,
    private val paths: ModelPaths
) : Closeable {

    private val ownedRetrievers = mutableListOf<Retriever>()

    companion object {
        private const val TAG = "EcoVectorStore"

        @Volatile
        private var instance: EcoVectorStore? = null

        /**
         * Create and initialize an EcoVectorStore instance.
         *
         * Copies model files from assets on first run, then initializes
         * the native engine (tokenizer, embedding model, database, morphological analyzer).
         *
         * @param context Android context (Application context recommended)
         * @param config Configuration options (optional, uses defaults)
         * @return Initialized EcoVectorStore instance
         * @throws IllegalStateException if initialization fails
         */
        fun create(
            context: Context,
            config: EcoVectorConfig = EcoVectorConfig()
        ): EcoVectorStore {
            return instance ?: synchronized(this) {
                instance ?: run {
                    val paths = ModelPaths.fromContext(
                        context, config.modelAssetDir, config.dbName, config.kiwiModelAssetDir
                    )

                    val copySuccess = AssetCopier.copyAssetDirectory(
                        context.assets,
                        config.modelAssetDir,
                        paths.modelDir
                    )
                    if (!copySuccess) {
                        throw IllegalStateException("Failed to copy model assets from ${config.modelAssetDir}")
                    }

                    val kiwiCopySuccess = AssetCopier.copyAssetDirectory(
                        context.assets,
                        config.kiwiModelAssetDir,
                        paths.kiwiModelDir,
                        overwrite = true  // 사전 파일 변경 시 반드시 덮어쓰기
                    )
                    if (!kiwiCopySuccess) {
                        Log.w(TAG, "Failed to copy Kiwi model assets - BM25 will use fallback")
                    }

                    PdfTextExtractor.init(context)

                    val initSuccess = NativeEcoVectorStore.initialize(
                        paths.obxDbDir,
                        paths.tokenizerPath,
                        paths.onnxModelPath,
                        paths.kiwiModelDir
                    )
                    if (!initSuccess) {
                        throw IllegalStateException("EcoVectorStore initialization failed")
                    }

                    EcoVectorStore(context, paths).also { instance = it }
                }
            }
        }

        fun getInstance(): EcoVectorStore {
            return instance ?: throw IllegalStateException("EcoVectorStore not initialized. Call create() first.")
        }
    }

    /** Whether this EcoVectorStore instance is initialized and ready. */
    val isInitialized: Boolean get() = instance != null

    // =========================================================================
    // Retriever Factory Methods
    // =========================================================================

    /**
     * Create a [VectorRetriever] bound to this EcoVectorStore instance.
     *
     * The retriever uses FAISS clustering + HNSW for vector similarity search.
     * It is owned by this EcoVectorStore instance and will be destroyed when [close] is called.
     *
     * @param params Vector retriever parameters (efSearch, nprobe, topK)
     * @return A new [VectorRetriever] instance
     */
    fun createVectorRetriever(
        params: VectorRetrieverParams = VectorRetrieverParams(),
        name: String = "Vector"
    ): VectorRetriever {
        check(isInitialized) { "EcoVectorStore not initialized. Call create() first." }
        val retriever = VectorRetriever(params, name)
        ownedRetrievers.add(retriever)
        return retriever
    }

    fun createObxVectorRetriever(
        params: ObxVectorRetrieverParams = ObxVectorRetrieverParams(),
        name: String = "ObxVector"
    ): ObxVectorRetriever {
        check(isInitialized) { "EcoVectorStore not initialized. Call create() first." }
        val retriever = ObxVectorRetriever(params, name)
        ownedRetrievers.add(retriever)
        return retriever
    }

    /**
     * Create a [BM25Retriever] bound to this EcoVectorStore instance.
     *
     * The retriever uses BM25 full-text search with Kiwi morphological analysis.
     * It is owned by this EcoVectorStore instance and will be destroyed when [close] is called.
     *
     * @param params BM25 retriever parameters (k1, b, topK)
     * @return A new [BM25Retriever] instance
     */
    fun createBM25Retriever(
        params: BM25RetrieverParams = BM25RetrieverParams(),
        name: String = "BM25"
    ): BM25Retriever {
        check(isInitialized) { "EcoVectorStore not initialized. Call create() first." }
        val retriever = BM25Retriever(params, name)
        ownedRetrievers.add(retriever)
        return retriever
    }

    /**
     * Create an [EnsembleRetriever] that combines existing retrievers.
     *
     * ```kotlin
     * val vector = eco.createVectorRetriever(VectorRetrieverParams(efSearch = 64, nprobe = 4, topK = 50))
     * val bm25 = eco.createBM25Retriever(BM25RetrieverParams(k1 = 0.9f, topK = 50))
     * val ensemble = eco.createEnsembleRetriever(
     *     components = listOf(vector weighted 0.7f, bm25 weighted 0.3f),
     *     topK = 10
     * )
     * ```
     *
     * @param components List of [WeightedRetriever] (use [weighted] infix function)
     * @param fusionMethod Fusion strategy: [FusionMethod.RRF] or [FusionMethod.RSF]
     * @param rrfK RRF constant (higher = flatter rank distribution)
     * @param topK Number of final results after fusion
     * @param parallel Execute sub-retrievers in parallel
     * @return A new [EnsembleRetriever] instance owned by this EcoVectorStore
     */
    fun createEnsembleRetriever(
        components: List<WeightedRetriever>,
        fusionMethod: FusionMethod = FusionMethod.RRF,
        rrfK: Float = 60f,
        topK: Int = 10,
        parallel: Boolean = false,
        name: String = "Ensemble"
    ): EnsembleRetriever {
        check(isInitialized) { "EcoVectorStore not initialized. Call create() first." }
        require(components.isNotEmpty()) {
            "EnsembleRetriever requires at least one component."
        }

        val handles = LongArray(components.size) {
            when (val r = components[it].retriever) {
                is VectorRetriever -> r.nativeHandle
                is BM25Retriever -> r.nativeHandle
                is EnsembleRetriever -> r.nativeHandle
                else -> throw IllegalArgumentException("Unknown retriever type: ${r::class}")
            }
        }
        val weights = FloatArray(components.size) { components[it].weight }

        val handle = NativeEnsembleRetriever.createFromHandles(
            handles, weights, fusionMethod.value, rrfK, topK, parallel
        )
        val retriever = EnsembleRetriever(handle, name)
        ownedRetrievers.add(retriever)
        return retriever
    }

    // =========================================================================
    // Document Management
    // =========================================================================

    /**
     * Add a document. Automatically chunks, tokenizes, embeds, and stores.
     *
     * @param content Document text content
     * @param title Document title (for identification)
     * @return Document ID (positive on success, -1 on failure)
     */
    fun addDocument(content: String, title: String = ""): Long {
        return NativeEcoVectorStore.addDocument(content, title)
    }

    /**
     * Add a document with custom chunking parameters.
     *
     * @param content Document text content
     * @param title Document title (for identification)
     * @param chunkParams Chunking parameters (strategy, chunkSize, chunkOverlap)
     * @return Document ID (positive on success, -1 on failure)
     */
    fun addDocument(
        content: String,
        title: String = "",
        chunkParams: ChunkParams
    ): Long {
        check(isInitialized) { "EcoVectorStore not initialized. Call create() first." }
        return NativeEcoVectorStore.addDocumentWithChunkParams(
            content, title,
            chunkParams.strategy.value, chunkParams.chunkSize, chunkParams.chunkOverlap
        )
    }

    /**
     * Add a document from a [DocumentSource].
     * Extracts text, then processes via the native pipeline.
     */
    suspend fun addDocument(source: DocumentSource): Long {
        val text = source.extractText()
        if (text.isBlank()) return -1
        return NativeEcoVectorStore.addDocument(text, source.title)
    }

    /**
     * Add multiple documents in batch.
     *
     * @param documents List of pairs (content, title)
     * @param progressCallback Optional progress callback (0.0 to 1.0)
     * @return Number of documents successfully added
     */
    fun addDocuments(
        documents: List<Pair<String, String>>,
        progressCallback: ((Float) -> Unit)? = null
    ): Int {
        var addedCount = 0
        documents.forEachIndexed { index, (content, title) ->
            if (addDocument(content, title) > 0) {
                addedCount++
            }
            progressCallback?.invoke((index + 1).toFloat() / documents.size)
        }
        return addedCount
    }

    /**
     * Add multiple documents in batch with custom chunking parameters.
     *
     * @param documents List of pairs (content, title)
     * @param chunkParams Chunking parameters (strategy, chunkSize, chunkOverlap)
     * @param progressCallback Optional progress callback (0.0 to 1.0)
     * @return Number of documents successfully added
     */
    fun addDocuments(
        documents: List<Pair<String, String>>,
        chunkParams: ChunkParams,
        progressCallback: ((Float) -> Unit)? = null
    ): Int {
        check(isInitialized) { "EcoVectorStore not initialized. Call create() first." }
        val texts = documents.map { it.first }.toTypedArray()
        val titles = documents.map { it.second }.toTypedArray()
        return NativeEcoVectorStore.addDocumentsWithChunkParams(
            texts, titles,
            chunkParams.strategy.value, chunkParams.chunkSize, chunkParams.chunkOverlap
        )
    }

    /**
     * Add multiple [DocumentSource]s in batch.
     */
    suspend fun addDocuments(
        sources: List<DocumentSource>,
        progressCallback: ((Float) -> Unit)? = null
    ): Int {
        var addedCount = 0
        sources.forEachIndexed { index, source ->
            if (addDocument(source) > 0) {
                addedCount++
            }
            progressCallback?.invoke((index + 1).toFloat() / sources.size)
        }
        return addedCount
    }

    // =========================================================================
    // Tokenization & Embedding
    // =========================================================================

    /** Tokenize text to HuggingFace token IDs. */
    fun tokenize(text: String): IntArray {
        check(isInitialized) { "EcoVectorStore not initialized. Call create() first." }
        return NativeEcoVectorStore.tokenize(text)
    }

    /** Embed text to a float vector using the ONNX model. */
    fun embed(text: String): FloatArray {
        check(isInitialized) { "EcoVectorStore not initialized. Call create() first." }
        return NativeEcoVectorStore.embed(text)
    }

    /** Tokenize text to Kiwi morpheme hashes (Korean morphological analysis for BM25). */
    fun tokenizeKiwi(text: String): IntArray {
        check(isInitialized) { "EcoVectorStore not initialized. Call create() first." }
        return NativeEcoVectorStore.tokenizeKiwi(text)
    }

    // =========================================================================
    // Index Management
    // =========================================================================

    /**
     * Build both the vector (FAISS + HNSW) and BM25 search indices.
     * Required after adding documents.
     *
     * @param centroidCount Number of cluster centroids (0 = auto: max(8, totalChunks/100))
     * @return true if index was built successfully
     */
    fun buildIndex(centroidCount: Int = 0): Boolean {
        return NativeEcoVectorStore.buildIndex(centroidCount)
    }

    /**
     * Build only the vector search index (FAISS clustering + HNSW).
     *
     * Use this when you only need vector retrieval, or when you want to build
     * vector and BM25 indices separately.
     *
     * @param centroidCount Number of cluster centroids (0 = auto: max(8, totalChunks/100))
     * @return true if index was built successfully
     */
    fun buildVectorIndex(centroidCount: Int = 0): Boolean {
        check(isInitialized) { "EcoVectorStore not initialized. Call create() first." }
        return NativeEcoVectorStore.buildVectorIndex(centroidCount)
    }

    /**
     * Build vector index with custom parameters.
     * @param params Index build parameters (nCluster, hnswM, efConstruction)
     * @return true if successful
     */
    fun buildVectorIndex(params: EcoVectorIndexParams): Boolean {
        check(isInitialized) { "EcoVectorStore not initialized. Call create() first." }
        return NativeEcoVectorStore.buildVectorIndex(params.nCluster, params.hnswM, params.efConstruction, params.maxTrainSamples)
    }

    /**
     * Build only the BM25 full-text search index.
     *
     * Use this when you only need BM25 retrieval, or when you want to build
     * vector and BM25 indices separately.
     *
     * @return true if index was built successfully
     */
    fun buildBM25Index(): Boolean {
        check(isInitialized) { "EcoVectorStore not initialized. Call create() first." }
        return NativeEcoVectorStore.buildBM25Index()
    }

    /** Check if the search index is built and ready. */
    fun isIndexReady(): Boolean {
        return NativeEcoVectorStore.isIndexReady()
    }

    // =========================================================================
    // Data Management
    // =========================================================================

    /** Remove all documents and chunks from the database. */
    fun removeAll() {
        NativeEcoVectorStore.removeAll()
    }

    /**
     * Remove a single document and all its chunks.
     *
     * @param id Document ID to remove
     * @return true if the document was successfully removed
     */
    fun removeDocument(id: Long): Boolean {
        check(isInitialized) { "EcoVectorStore not initialized. Call create() first." }
        return NativeEcoVectorStore.removeDocument(id)
    }

    /**
     * Remove a single chunk.
     *
     * @param id Chunk ID to remove
     * @return true if the chunk was successfully removed
     */
    fun removeChunk(id: Long): Boolean {
        check(isInitialized) { "EcoVectorStore not initialized. Call create() first." }
        return NativeEcoVectorStore.removeChunk(id)
    }

    /** Get the number of documents in the database. */
    val documentCount: Int get() = NativeEcoVectorStore.getDocumentCount()

    /** Get the number of chunks in the database. */
    val chunkCount: Int get() = NativeEcoVectorStore.getChunkCount()

    // =========================================================================
    // Lifecycle
    // =========================================================================

    /** Release native resources. Call when done using EcoVectorStore. */
    override fun close() {
        synchronized(Companion) {
            for (r in ownedRetrievers) {
                r.close()
            }
            ownedRetrievers.clear()
            NativeEcoVectorStore.close()
            instance = null
        }
    }
}
