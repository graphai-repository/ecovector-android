package io.graphai.ecovector

/**
 * Parameters for token-aware text chunking.
 *
 * @param maxTokens Maximum tokens per chunk (default: 512)
 * @param overlapTokens Token overlap between consecutive chunks (default: 216)
 */
data class ChunkParams(
    val maxTokens: Int = 216,
    val overlapTokens: Int = 128
)
