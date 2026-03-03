package io.graphai.ecovector.benchmark

class StageDependencyResolver {

    private val downstream: Map<PipelineStage, Set<PipelineStage>> = mapOf(
        PipelineStage.ECO_LOAD to setOf(PipelineStage.ECO_CHUNK),
        PipelineStage.ECO_CHUNK to setOf(PipelineStage.ECO_TOKENIZE),
        PipelineStage.ECO_EMBED to setOf(PipelineStage.ECO_VECTOR_INDEX),
        PipelineStage.ECO_IMPORT_EMBED to setOf(PipelineStage.ECO_VECTOR_INDEX),
        PipelineStage.ECO_TOKENIZE to setOf(PipelineStage.ECO_BM25_INDEX),
        PipelineStage.BENCH_LOAD to setOf(PipelineStage.BENCH_EMBED, PipelineStage.BENCH_TOKENIZE),
    )

    /** eco_vector_index가 요청되면 eco_embed 또는 eco_import_embed가 필요 (명시적 요청 없으면 eco_embed 자동 추가) */
    private val upstream: Map<PipelineStage, (Set<PipelineStage>) -> Set<PipelineStage>> = mapOf(
        PipelineStage.ECO_VECTOR_INDEX to { requested ->
            if (PipelineStage.ECO_IMPORT_EMBED in requested || PipelineStage.ECO_EMBED in requested) {
                emptySet()
            } else {
                setOf(PipelineStage.ECO_EMBED)
            }
        },
    )

    fun resolve(requested: Set<PipelineStage>): List<PipelineStage> {
        val resolved = mutableSetOf<PipelineStage>()
        for (stage in requested) {
            addWithDownstream(stage, resolved)
        }
        // upstream 의존성 처리 (예: eco_vector_index → eco_embed 자동 추가)
        val extra = mutableSetOf<PipelineStage>()
        for (stage in resolved.toSet()) {
            upstream[stage]?.invoke(requested)?.let { deps ->
                for (dep in deps) addWithDownstream(dep, extra)
            }
        }
        resolved.addAll(extra)
        return resolved.sortedWith(compareBy<PipelineStage> { it.phase }.thenBy { it.ordinal })
    }

    private fun addWithDownstream(stage: PipelineStage, result: MutableSet<PipelineStage>) {
        if (!result.add(stage)) return
        downstream[stage]?.forEach { addWithDownstream(it, result) }
    }
}
