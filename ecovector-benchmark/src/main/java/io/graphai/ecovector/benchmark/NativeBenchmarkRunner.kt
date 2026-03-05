package io.graphai.ecovector.benchmark

/**
 * JNI interface for benchmark operations and benchmark data management.
 * Requires EcoVector (NativeEcoVectorStore) to be initialized first.
 */
object NativeBenchmarkRunner {
    init {
        // 1. ecovector SDK 네이티브 라이브러리 로드 (EcoVectorStore 초기화 필수)
        io.graphai.ecovector.NativeEcoVectorStore
        // 2. benchmark 네이티브 라이브러리 로드 (JNI 함수들이 여기에 있음)
        System.loadLibrary("ecovector-benchmark")
    }

    // --- Benchmark DB Lifecycle ---

    /** Benchmark DB 초기화 (Query+GroundTruth 전용 분리 DB) */
    external fun initBenchmarkDb(dbPath: String): Boolean

    /** Benchmark DB 닫기 */
    external fun closeBenchmarkDb()

    // --- IRetriever Benchmarking ---

    /**
     * Run benchmark on registered IRetriever instances.
     * @param retrieverHandles Array of native IRetriever* pointer handles
     * @param topK Number of top results to evaluate
     * @param filterPath Path to filter file (optional)
     * @param split Query split to evaluate: "valid", "test", or null for all
     * @return JSON result string with metrics per retriever
     */
    external fun runRegisteredRetrievers(
        retrieverHandles: LongArray, topK: Int,
        filterPath: String? = null, split: String? = null
    ): String

    // --- Benchmark Data Management ---

    /** Get the number of benchmark queries in the database. */
    external fun getQueryCount(): Int

    /** Get paginated benchmark queries as JSON. */
    external fun getQueriesJson(offset: Int, limit: Int): String

    /** Get all query external IDs as JSON array. */
    external fun getQueryExternalIdsJson(): String

    /** Save a query with pre-computed embedding and tokens to Benchmark DB. */
    external fun saveQueryRaw(
        externalId: String,
        text: String,
        refinedQuery: String,
        tokenIds: IntArray,
        embedding: FloatArray,
        kiwiTokens: IntArray,
        createdAt: Long,
        targetTypes: String,
        categories: String,
        split: String = "",
        evalTopK: Int = 0
    ): Long

    /** Save ground truth entries for benchmark accuracy evaluation. */
    external fun saveGroundTruths(queryIds: Array<String>, docIds: Array<String>): LongArray

    // --- Pipeline Stage Methods ---

    /** 벤치마크 DB 쿼리+GT 전체 삭제 */
    external fun clearBenchmarkQueries()

    /** 질의 텍스트만 저장 (임베딩/토큰 없이) */
    external fun saveQueryTextOnly(
        externalId: String, text: String, refinedQuery: String,
        createdAt: Long, targetTypes: String, categories: String,
        split: String = "", evalTopK: Int = 0
    ): Long

    /** 전체 질의 임베딩 (EcoVectorStore의 embedder 사용) */
    external fun embedAllQueries(): Int

    /** SQLite에서 쿼리 임베딩 임포트 (query_embeddings 테이블) */
    external fun importQueryEmbeddingsFromSQLite(dbPath: String): Int

    /** 전체 질의 Kiwi 토큰화 */
    external fun tokenizeAllQueries(): Int

    // --- Retriever Lifecycle (benchmark-specific) ---

    /** Create VectorRetriever via EcoVectorStore and return native handle. */
    external fun createVectorRetriever(efSearch: Int, nprobe: Int, topK: Int): Long

    /** Create ObxVectorRetriever via EcoVectorStore and return native handle. */
    external fun createObxVectorRetriever(maxResultCount: Int, topK: Int): Long

    /** Create BM25Retriever via EcoVectorStore and return native handle. */
    external fun createBM25Retriever(
        k1: Float, b: Float, topK: Int,
        idfThreshold: Float, maxSeedTerms: Int, candidateMultiplier: Int,
        minCandidates: Int, minScore: Float,
        rm3Enabled: Boolean, rm3FbDocs: Int, rm3FbTerms: Int,
        rm3OrigWeight: Float, rm3MinDf: Int
    ): Long

    /** Create EnsembleRetriever from component handles. */
    external fun createEnsembleRetriever(
        retrieverHandles: LongArray, weights: FloatArray,
        fusionMethod: Int, rrfK: Float, topK: Int, parallel: Boolean
    ): Long

    /** Destroy a retriever (remove from EcoVectorStore's owned list and free memory). */
    external fun destroyRetriever(handle: Long): Boolean

    // --- Detail Merge ---

    /** Merge per-retriever detail JSONL files into single per-query JSONL. */
    external fun mergeDetails(
        methodNames: Array<String>,
        detailPaths: Array<String>,
        outputPath: String,
        split: String? = null
    ): String
}
