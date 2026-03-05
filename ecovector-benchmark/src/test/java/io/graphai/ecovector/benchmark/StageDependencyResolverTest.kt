package io.graphai.ecovector.benchmark

import org.junit.Assert.*
import org.junit.Test

class StageDependencyResolverTest {
    private val resolver = StageDependencyResolver()

    @Test
    fun `eco_load cascades to all eco stages`() {
        val result = resolver.resolve(setOf(PipelineStage.ECO_LOAD))
        assertEquals(
            listOf(
                PipelineStage.ECO_LOAD, PipelineStage.ECO_CHUNK,
                PipelineStage.ECO_EMBED, PipelineStage.ECO_TOKENIZE,
                PipelineStage.ECO_VECTOR_INDEX, PipelineStage.ECO_BM25_INDEX
            ),
            result
        )
    }

    @Test
    fun `eco_embed cascades to tokenize and vector_index and bm25_index`() {
        val result = resolver.resolve(setOf(PipelineStage.ECO_EMBED))
        assertEquals(
            listOf(
                PipelineStage.ECO_EMBED, PipelineStage.ECO_TOKENIZE,
                PipelineStage.ECO_VECTOR_INDEX, PipelineStage.ECO_BM25_INDEX
            ),
            result
        )
    }

    @Test
    fun `eco_import_embed cascades to tokenize and vector_index and bm25_index`() {
        val result = resolver.resolve(setOf(PipelineStage.ECO_IMPORT_EMBED))
        assertEquals(
            listOf(
                PipelineStage.ECO_IMPORT_EMBED, PipelineStage.ECO_TOKENIZE,
                PipelineStage.ECO_VECTOR_INDEX, PipelineStage.ECO_BM25_INDEX
            ),
            result
        )
    }

    @Test
    fun `eco_export_chunks adds load and chunk without cascade`() {
        val result = resolver.resolve(setOf(PipelineStage.ECO_EXPORT_CHUNKS))
        assertEquals(
            listOf(
                PipelineStage.ECO_LOAD, PipelineStage.ECO_CHUNK,
                PipelineStage.ECO_EXPORT_CHUNKS
            ),
            result
        )
    }

    @Test
    fun `eco_tokenize cascades to bm25_index only`() {
        val result = resolver.resolve(setOf(PipelineStage.ECO_TOKENIZE))
        assertEquals(
            listOf(PipelineStage.ECO_TOKENIZE, PipelineStage.ECO_BM25_INDEX),
            result
        )
    }

    @Test
    fun `eco_vector_index has no cascade`() {
        val result = resolver.resolve(setOf(PipelineStage.ECO_VECTOR_INDEX))
        assertEquals(listOf(PipelineStage.ECO_VECTOR_INDEX), result)
    }

    @Test
    fun `bench_load cascades to tokenize only`() {
        val result = resolver.resolve(setOf(PipelineStage.BENCH_LOAD))
        assertEquals(
            listOf(PipelineStage.BENCH_LOAD, PipelineStage.BENCH_TOKENIZE),
            result
        )
    }

    @Test
    fun `cross-phase combination preserves phase order`() {
        val result = resolver.resolve(
            setOf(PipelineStage.ECO_TOKENIZE, PipelineStage.BENCH_EMBED)
        )
        assertEquals(
            listOf(PipelineStage.ECO_TOKENIZE, PipelineStage.ECO_BM25_INDEX, PipelineStage.BENCH_EMBED),
            result
        )
    }

    @Test
    fun `empty request returns empty`() {
        val result = resolver.resolve(emptySet())
        assertTrue(result.isEmpty())
    }

    @Test
    fun `eco_chunk and eco_embed deduplicates shared downstream`() {
        val result = resolver.resolve(
            setOf(PipelineStage.ECO_CHUNK, PipelineStage.ECO_EMBED)
        )
        assertEquals(
            listOf(
                PipelineStage.ECO_CHUNK, PipelineStage.ECO_EMBED,
                PipelineStage.ECO_TOKENIZE, PipelineStage.ECO_VECTOR_INDEX,
                PipelineStage.ECO_BM25_INDEX
            ),
            result
        )
    }
}
