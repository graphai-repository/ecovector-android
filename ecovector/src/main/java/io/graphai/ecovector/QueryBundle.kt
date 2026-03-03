package io.graphai.ecovector

/**
 * Pre-computed query representation for multi-modal retrieval.
 *
 * A QueryBundle bundles the different representations of a query
 * so that retrieval can skip redundant computation when multiple
 * retrievers share the same pipeline.
 *
 * @param rawText Original query text (optional if embedding/kiwiTokens are provided)
 * @param embedding Pre-computed float embedding vector (for vector retrieval)
 * @param kiwiTokens Pre-computed Kiwi morpheme hash tokens (for BM25 retrieval)
 */
data class QueryBundle(
    val rawText: String? = null,
    val embedding: FloatArray? = null,
    val kiwiTokens: IntArray? = null
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is QueryBundle) return false
        if (rawText != other.rawText) return false
        if (embedding != null) {
            if (other.embedding == null) return false
            if (!embedding.contentEquals(other.embedding)) return false
        } else if (other.embedding != null) return false
        if (kiwiTokens != null) {
            if (other.kiwiTokens == null) return false
            if (!kiwiTokens.contentEquals(other.kiwiTokens)) return false
        } else if (other.kiwiTokens != null) return false
        return true
    }

    override fun hashCode(): Int {
        var result = rawText?.hashCode() ?: 0
        result = 31 * result + (embedding?.contentHashCode() ?: 0)
        result = 31 * result + (kiwiTokens?.contentHashCode() ?: 0)
        return result
    }
}
