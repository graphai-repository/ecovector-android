package io.graphai.ecovector

internal object NativeObxVectorRetriever {
    external fun create(maxResultCount: Int, topK: Int): Long
    external fun isReady(handle: Long): Boolean
    external fun destroy(handle: Long)
}
