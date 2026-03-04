package io.graphai.ecovector.benchmark

class StageDependencyResolver {

    private val downstream: Map<PipelineStage, Set<PipelineStage>> = mapOf(
        PipelineStage.ECO_LOAD to setOf(PipelineStage.ECO_CHUNK),
        PipelineStage.ECO_CHUNK to setOf(PipelineStage.ECO_EMBED),
        PipelineStage.ECO_EMBED to setOf(PipelineStage.ECO_VECTOR_INDEX, PipelineStage.ECO_TOKENIZE),
        PipelineStage.ECO_IMPORT_EMBED to setOf(PipelineStage.ECO_VECTOR_INDEX, PipelineStage.ECO_TOKENIZE),
        PipelineStage.ECO_TOKENIZE to setOf(PipelineStage.ECO_BM25_INDEX),
        PipelineStage.BENCH_LOAD to setOf(PipelineStage.BENCH_EMBED, PipelineStage.BENCH_TOKENIZE),
    )

    /** cascade 없이 단순 추가되는 선행 단계 */
    private val prerequisites: Map<PipelineStage, Set<PipelineStage>> = mapOf(
        PipelineStage.ECO_EXPORT_CHUNKS to setOf(PipelineStage.ECO_LOAD, PipelineStage.ECO_CHUNK),
    )

    fun resolve(requested: Set<PipelineStage>): List<PipelineStage> {
        val resolved = mutableSetOf<PipelineStage>()
        for (stage in requested) {
            prerequisites[stage]?.let { resolved.addAll(it) }
            addWithDownstream(stage, resolved)
        }
        return resolved.sortedWith(compareBy<PipelineStage> { it.phase }.thenBy { it.ordinal })
    }

    private fun addWithDownstream(stage: PipelineStage, result: MutableSet<PipelineStage>) {
        if (!result.add(stage)) return
        downstream[stage]?.forEach { addWithDownstream(it, result) }
    }
}
