package io.graphai.ecovector

/**
 * Parameters for text chunking.
 *
 * @param strategy Chunking strategy to use
 * @param chunkSize Maximum chunk size in tokens
 * @param chunkOverlap Overlap between consecutive chunks in tokens
 */
data class ChunkParams(
    val strategy: ChunkStrategy = ChunkStrategy.SENTENCE,
    val chunkSize: Int = 256,
    val chunkOverlap: Int = 64
)

/**
 * Text chunking strategy.
 *
 * @param value Native integer value passed to C++ layer
 */
enum class ChunkStrategy(val value: Int) {
    /** Sliding window over word boundaries. */
    WORD_SLIDING_WINDOW(0),

    /** Sentence-aware chunking that respects sentence boundaries. */
    SENTENCE(1)
}
