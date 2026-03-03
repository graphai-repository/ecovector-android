package io.graphai.ecovector.benchmark

import io.graphai.ecovector.*

/**
 * 벤치마크에서 사용할 retriever 조합 설정.
 * 코드 변경 없이 파라미터 실험 가능.
 */
data class BenchmarkConfig(
    val retrievers: List<RetrieverSpec>,
    val evaluationTopK: Int = 5,
    val indexConfigs: List<IndexSpec>? = null  // null = 인덱스 재구축 안 함
)

sealed class RetrieverSpec {
    abstract val name: String?

    /** 기본값과 다른 파라미터만 반환 */
    fun nonDefaultParams(): List<String> = when (this) {
        is ObxVector -> {
            val d = ObxVectorRetrieverParams()
            buildList {
                if (params.maxResultCount != d.maxResultCount) add("maxRes=${params.maxResultCount}")
                if (params.topK != d.topK) add("topK=${params.topK}")
            }
        }
        is Vector -> {
            val d = VectorRetrieverParams()
            buildList {
                if (params.efSearch != d.efSearch) add("efSearch=${params.efSearch}")
                if (params.nprobe != d.nprobe) add("nprobe=${params.nprobe}")
                if (params.topK != d.topK) add("topK=${params.topK}")
            }
        }
        is BM25 -> {
            val d = BM25RetrieverParams()
            buildList {
                if (params.k1 != d.k1) add("k1=${params.k1}")
                if (params.b != d.b) add("b=${params.b}")
                if (params.topK != d.topK) add("topK=${params.topK}")
                if (params.idfThreshold != d.idfThreshold) add("idf=${params.idfThreshold}")
                if (params.maxSeedTerms != d.maxSeedTerms) add("seed=${params.maxSeedTerms}")
                if (params.candidateMultiplier != d.candidateMultiplier) add("candMul=${params.candidateMultiplier}")
                if (params.minCandidates != d.minCandidates) add("minCand=${params.minCandidates}")
                if (params.minScore != d.minScore) add("minScore=${params.minScore}")
                if (params.rm3Enabled) add("rm3=${params.rm3FbDocs}d/${params.rm3FbTerms}t,λ=${params.rm3OrigWeight}")
            }
        }
        is Ensemble -> buildList {
            if (fusionMethod != FusionMethod.RRF) add("fusion=$fusionMethod")
            if (rrfK != 30f) add("rrfK=$rrfK")
            if (topK != 15) add("topK=$topK")
        }
    }

    /** 표시용 이름 (name이 없으면 타입명) */
    fun displayName(): String = name ?: when (this) {
        is ObxVector -> "ObxVector"
        is Vector -> "Vector"
        is BM25 -> "BM25"
        is Ensemble -> "Ensemble"
    }

    data class ObxVector(
        val params: ObxVectorRetrieverParams,
        override val name: String? = null
    ) : RetrieverSpec()

    data class Vector(
        val params: VectorRetrieverParams,
        override val name: String? = null
    ) : RetrieverSpec()

    data class BM25(
        val params: BM25RetrieverParams,
        override val name: String? = null
    ) : RetrieverSpec()

    data class Ensemble(
        val components: List<RetrieverSpec>,
        val weights: List<Float>,
        val fusionMethod: FusionMethod,
        val rrfK: Float = 30f,
        val topK: Int = 15,
        val parallel: Boolean = false,
        override val name: String? = null
    ) : RetrieverSpec()
}

sealed class IndexSpec {
    abstract val name: String?
    data class Vector(val params: EcoVectorIndexParams, override val name: String? = null) : IndexSpec()
}
