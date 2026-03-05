package io.graphai.ecovector.benchmark

enum class PipelineStage(val intentKey: String, val phase: Int) {
    ECO_LOAD("eco_load", 1),
    ECO_CHUNK("eco_chunk", 1),
    ECO_EXPORT_CHUNKS("eco_export_chunks", 1),
    ECO_EMBED("eco_embed", 1),
    ECO_IMPORT_EMBED("eco_import_embed", 1),
    ECO_TOKENIZE("eco_tokenize", 1),
    ECO_VECTOR_INDEX("eco_vector_index", 1),
    ECO_BM25_INDEX("eco_bm25_index", 1),
    BENCH_LOAD("bench_load", 2),
    BENCH_EMBED("bench_embed", 2),
    BENCH_IMPORT_EMBED("bench_import_embed", 2),
    BENCH_EXPORT_QUERIES("bench_export_queries", 2),
    BENCH_TOKENIZE("bench_tokenize", 2);

    companion object {
        fun fromIntent(key: String): PipelineStage? =
            entries.find { it.intentKey == key }
    }
}
