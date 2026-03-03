package io.graphai.ecovector


/**
 * Parameters for vector retrieval (FAISS clustering + HNSW).
 *
 * @param efSearch HNSW ef_search parameter (higher = more accurate, slower)
 * @param nprobe Number of FAISS clusters to probe (higher = more accurate, slower)
 * @param topK Default number of results to return
 */
data class ObxVectorRetrieverParams(
    val maxResultCount: Int = 100,
    val topK: Int = 11
)

data class VectorRetrieverParams(
    val efSearch: Int = 20,
    val nprobe: Int = 4,
    val topK: Int = 11
)

/**
 * Parameters for BM25 full-text retrieval.
 *
 * @param k1 Term frequency saturation parameter (typically 1.2-2.0)
 * @param b Document length normalization (0.0 = no normalization, 1.0 = full)
 * @param topK Default number of results to return
 * @param idfThreshold Seed token IDF filter ratio (fraction of top IDF)
 * @param maxSeedTerms Maximum number of seed tokens for candidate collection
 * @param candidateMultiplier Full-scoring candidates = topK × multiplier
 * @param minCandidates Minimum number of full-scoring candidates
 * @param minScore BM25 score lower bound (results below are filtered out)
 */
data class BM25RetrieverParams(
    val k1: Float = 0.9f,
    val b: Float = 0.25f,
    val topK: Int = 15,
    val idfThreshold: Float = 0.6f,
    val maxSeedTerms: Int = 5,
    val candidateMultiplier: Int = 10,
    val minCandidates: Int = 50,
    val minScore: Float = 0.0f,
    // RM3 (Relevance Model 3) query expansion
    val rm3Enabled: Boolean = false,
    val rm3FbDocs: Int = 10,       // feedback documents
    val rm3FbTerms: Int = 20,      // expansion terms
    val rm3OrigWeight: Float = 0.6f,  // λ: original query weight
    val rm3MinDf: Int = 2          // min document frequency filter
)

/**
 * Fusion method for combining results from multiple retrievers.
 *
 * - [RSF]: Relative Score Fusion — min-max normalizes scores per retriever, then weighted sum.
 *   Preserves score magnitude differences.
 * - [RRF]: Reciprocal Rank Fusion — uses only rank positions, ignoring scores.
 *   Formula: weight / (k + rank). More robust when score distributions differ widely.
 */
enum class FusionMethod(val value: Int) {
    RSF(0),
    RRF(1)
}

/**
 * A list of search results with an associated weight for ensemble fusion.
 *
 * @param results Search results from a single retriever
 * @param weight Weight for this retriever's contribution in fusion
 * @param isDistance true if scores are L2 distances (lower = more similar), needs inversion in score fusion
 */
data class WeightedResults(
    val results: List<SearchResult>,
    val weight: Float,
    val isDistance: Boolean = false
)

/**
 * A retriever paired with its weight for ensemble composition.
 * Use the [weighted] infix function: `retriever weighted 0.7f`
 */
data class WeightedRetriever(
    val retriever: Retriever,
    val weight: Float
)

infix fun Retriever.weighted(weight: Float): WeightedRetriever =
    WeightedRetriever(this, weight)

/**
 * Parameters for EcoVector index building (FAISS clustering + HNSW).
 *
 * @param nCluster Number of K-means clusters (0 = auto: max(8, chunks/100))
 * @param hnswM HNSW connections per node (higher = more accurate, more memory)
 * @param efConstruction HNSW build quality (higher = more accurate, slower build)
 * @param maxTrainSamples Max FAISS training samples (0 = default 100K, auto: min(nCluster*100, this))
 */
data class EcoVectorIndexParams(
    val nCluster: Int = 0,
    val hnswM: Int = 16,
    val efConstruction: Int = 100,
    val maxTrainSamples: Int = 0
) {
    fun label(): String {
        val parts = mutableListOf<String>()
        if (nCluster > 0) parts.add("nc=$nCluster")
        parts.add("M=$hnswM")
        parts.add("efc=$efConstruction")
        if (maxTrainSamples > 0) parts.add("train=$maxTrainSamples")
        return parts.joinToString(",")
    }
}
