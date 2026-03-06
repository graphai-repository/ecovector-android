package io.graphai.ecovector

/**
 * Parameters for token-aware text chunking.
 *
 * @param maxTokens Maximum tokens per chunk (default: 256)
 * @param overlapTokens Token overlap between consecutive chunks (default: 128)
 */
data class ChunkParams(
    val maxTokens: Int = 256,
    val overlapTokens: Int = 128
)
