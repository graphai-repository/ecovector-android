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
) {
    companion object {
        /** 2-way Ensemble 스펙 헬퍼 */
        private fun ens(
            vw: Float, bw: Float,
            method: FusionMethod = FusionMethod.RRF,
            k: Float = 30f,
            label: String? = null
        ) = RetrieverSpec.Ensemble(
            components = listOf(
                RetrieverSpec.Vector(VectorRetrieverParams()),
                RetrieverSpec.BM25(BM25RetrieverParams())
            ),
            weights = listOf(vw, bw),
            fusionMethod = method,
            rrfK = k,
            name = label ?: "${method.name} ${vw}:${bw}" + if (k != 30f) " k=$k" else ""
        )

        private val benchmarkIndexParams = EcoVectorIndexParams(nCluster = 200)

        /** RM3 2-way 앙상블 헬퍼 (Vector + RM3-BM25) */
        private fun ensRm3(
            fbDocs: Int = 10, fbTerms: Int = 10,
            origWeight: Float = 0.5f, minDf: Int = 2,
            k: Float = 30f,
            label: String? = null
        ) = RetrieverSpec.Ensemble(
            components = listOf(
                RetrieverSpec.Vector(VectorRetrieverParams()),
                RetrieverSpec.BM25(BM25RetrieverParams(
                    rm3Enabled = true,
                    rm3FbDocs = fbDocs, rm3FbTerms = fbTerms,
                    rm3OrigWeight = origWeight, rm3MinDf = minDf
                ))
            ),
            weights = listOf(1.0f, 1.0f),
            fusionMethod = FusionMethod.RRF,
            rrfK = k,
            name = label ?: "Vec+RM3(${fbDocs}d,${fbTerms}t,λ=$origWeight)"
        )

        /** 3-way 앙상블 헬퍼 (Vector + BM25 + BM25(다른 파라미터)) */
        private fun ens3bm25(
            bm25Params2: BM25RetrieverParams,
            vw: Float = 1.0f, bw1: Float = 1.0f, bw2: Float = 1.0f,
            k: Float = 30f,
            label: String
        ) = RetrieverSpec.Ensemble(
            components = listOf(
                RetrieverSpec.Vector(VectorRetrieverParams()),
                RetrieverSpec.BM25(BM25RetrieverParams()),
                RetrieverSpec.BM25(bm25Params2)
            ),
            weights = listOf(vw, bw1, bw2),
            fusionMethod = FusionMethod.RRF,
            rrfK = k,
            name = label
        )

        /** 3-way 앙상블 헬퍼 (Vector + BM25 + RM3-BM25) */
        private fun ens3bm25rm3(
            rm3FbDocs: Int = 10, rm3FbTerms: Int = 20,
            rm3OrigWeight: Float = 0.5f, rm3MinDf: Int = 2,
            k: Float = 30f,
            label: String
        ) = RetrieverSpec.Ensemble(
            components = listOf(
                RetrieverSpec.Vector(VectorRetrieverParams()),
                RetrieverSpec.BM25(BM25RetrieverParams()),
                RetrieverSpec.BM25(BM25RetrieverParams(
                    rm3Enabled = true,
                    rm3FbDocs = rm3FbDocs, rm3FbTerms = rm3FbTerms,
                    rm3OrigWeight = rm3OrigWeight, rm3MinDf = rm3MinDf
                ))
            ),
            weights = listOf(1.0f, 1.0f, 1.0f),
            fusionMethod = FusionMethod.RRF,
            rrfK = k,
            name = label
        )

        /** 3-way 앙상블 헬퍼 (Vector + RM3-BM25(A) + RM3-BM25(B)) */
        private fun ens3rm3rm3(
            rm3A: BM25RetrieverParams,
            rm3B: BM25RetrieverParams,
            k: Float = 30f,
            label: String
        ) = RetrieverSpec.Ensemble(
            components = listOf(
                RetrieverSpec.Vector(VectorRetrieverParams()),
                RetrieverSpec.BM25(rm3A),
                RetrieverSpec.BM25(rm3B)
            ),
            weights = listOf(1.0f, 1.0f, 1.0f),
            fusionMethod = FusionMethod.RRF,
            rrfK = k,
            name = label
        )

        val DEFAULT = BenchmarkConfig(
            retrievers = listOf(
                RetrieverSpec.ObxVector(ObxVectorRetrieverParams(maxResultCount = 100, topK = 11)),
                RetrieverSpec.Vector(VectorRetrieverParams(), name = "Vector1"),
                RetrieverSpec.BM25(BM25RetrieverParams(), name = "BM25-1"),
//                RetrieverSpec.BM25(BM25RetrieverParams(k1 = 1.2f, b = 0.5f, idfThreshold = 0.4f),
//                    name = "BM25-2"),
                ens(1.0f, 1.0f, label = "Ensemble1-Vec+BM25-1"),
//                RetrieverSpec.Ensemble(
//                    components = listOf(
//                        RetrieverSpec.Vector(VectorRetrieverParams()),
//                        RetrieverSpec.BM25(BM25RetrieverParams(k1 = 1.2f, b = 0.5f, idfThreshold = 0.4f))
//                    ),
//                    weights = listOf(1.0f, 1.0f),
//                    fusionMethod = FusionMethod.RRF,
//                    name = "Ensemble2-Vec+BM25-2"
//                ),
//                ens3bm25(BM25RetrieverParams(k1 = 1.2f, b = 0.5f, idfThreshold = 0.4f),
//                    label = "Ensemble3-3way"),
            ),
            indexConfigs = null  // 기존 인덱스 사용
        )
    }
}

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
