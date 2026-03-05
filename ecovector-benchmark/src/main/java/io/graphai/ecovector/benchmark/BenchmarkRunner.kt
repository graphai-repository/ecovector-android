package io.graphai.ecovector.benchmark

import android.content.Context
import android.util.Log
import io.graphai.ecovector.*
import java.io.File

/**
 * Reusable benchmark pipeline using IRetriever-based benchmarking:
 * 1. Init EcoVectorStore → 2. (Optional) Clear DB
 * → 3. Load dataset → 4. Create retrievers → 5. Run benchmark
 */
class BenchmarkRunner(
    private val context: Context
) {
    fun interface ProgressCallback {
        fun onProgress(progress: Float, status: String)
    }

    fun interface ResultCallback {
        fun onResult(result: BenchmarkMethodResult)
    }

    private fun saveResults(result: String) {
        val tag = "BenchmarkRunner"
        val extBase = context.getExternalFilesDir(null) ?: return

        // 최신 결과 전체 저장 (타임스탬프 포함 파일명)
        try {
            val ts = java.text.SimpleDateFormat("yyyyMMdd_HHmm", java.util.Locale.US).format(java.util.Date())
            val fileName = "benchmark_results_$ts.json"
            extBase.resolve(fileName).writeText(result, Charsets.UTF_8)
            Log.i(tag, "JSON saved to external: ${extBase.resolve(fileName).absolutePath}")
        } catch (e: Exception) {
            Log.w(tag, "External save failed: ${e.message}")
        }

        // 누적 히스토리 추가
        try {
            val parsed = org.json.JSONObject(result)

            val compactResults = org.json.JSONArray()
            val rawResults = parsed.optJSONArray("results")
            if (rawResults != null) {
                for (i in 0 until rawResults.length()) {
                    val m = rawResults.getJSONObject(i)
                    compactResults.put(org.json.JSONObject().apply {
                        put("method",           m.optString("method"))
                        put("Hit@5",            m.optDouble("Hit@5"))
                        put("Recall@5",         m.optDouble("Recall@5"))
                        put("avg_latency_ms",   m.optDouble("avg_latency_ms"))
                        put("total_latency_sec",m.optDouble("total_latency_sec"))
                    })
                }
            }

            val entry = org.json.JSONObject().apply {
                put("timestamp",   parsed.optString("timestamp", ""))
                put("dataset",     parsed.optString("dataset", "unknown"))
                put("doc_count",   parsed.optInt("doc_count"))
                put("chunk_count", parsed.optInt("chunk_count"))
                put("query_count", parsed.optInt("query_count"))
                put("results",     compactResults)
            }

            val historyDir = extBase.resolve("datasets").also { it.mkdirs() }
            val historyFile = historyDir.resolve("benchmark_history.jsonl")
            historyFile.appendText(entry.toString() + "\n", Charsets.UTF_8)
            Log.i(tag, "History appended: ${historyFile.absolutePath}")
        } catch (e: Exception) {
            Log.w(tag, "History append failed: ${e.message}")
        }
    }

    private fun ensureBenchmarkDb() {
        val tag = "BenchmarkRunner"
        val benchmarkDbPath = File(context.filesDir, "objectbox-benchmark-db").absolutePath
        if (!NativeBenchmarkRunner.initBenchmarkDb(benchmarkDbPath)) {
            Log.e(tag, "Failed to init benchmark DB at $benchmarkDbPath")
            return
        }
    }

    /** Tracks main handle + component handles for proper cleanup */
    private data class RetrieverHandleSet(
        val mainHandle: Long,
        val componentHandles: List<Long> = emptyList()
    )

    /** Create retriever handle(s) via benchmark JNI — no Kotlin wrapper pollution */
    private fun createRetrieverHandleSet(spec: RetrieverSpec): RetrieverHandleSet = when (spec) {
        is RetrieverSpec.ObxVector -> RetrieverHandleSet(
            NativeBenchmarkRunner.createObxVectorRetriever(spec.params.maxResultCount, spec.params.topK)
        )
        is RetrieverSpec.Vector -> RetrieverHandleSet(
            NativeBenchmarkRunner.createVectorRetriever(spec.params.efSearch, spec.params.nprobe, spec.params.topK)
        )
        is RetrieverSpec.BM25 -> RetrieverHandleSet(
            NativeBenchmarkRunner.createBM25Retriever(
                spec.params.k1, spec.params.b, spec.params.topK,
                spec.params.idfThreshold, spec.params.maxSeedTerms, spec.params.candidateMultiplier,
                spec.params.minCandidates, spec.params.minScore, spec.params.rm3Enabled,
                spec.params.rm3FbDocs, spec.params.rm3FbTerms, spec.params.rm3OrigWeight, spec.params.rm3MinDf
            )
        )
        is RetrieverSpec.Ensemble -> {
            val compHandles = mutableListOf<Long>()
            val compWeights = mutableListOf<Float>()
            for ((i, compSpec) in spec.components.withIndex()) {
                val h = when (compSpec) {
                    is RetrieverSpec.ObxVector -> NativeBenchmarkRunner.createObxVectorRetriever(
                        compSpec.params.maxResultCount, compSpec.params.topK)
                    is RetrieverSpec.Vector -> NativeBenchmarkRunner.createVectorRetriever(
                        compSpec.params.efSearch, compSpec.params.nprobe, compSpec.params.topK)
                    is RetrieverSpec.BM25 -> NativeBenchmarkRunner.createBM25Retriever(
                        compSpec.params.k1, compSpec.params.b, compSpec.params.topK,
                        compSpec.params.idfThreshold, compSpec.params.maxSeedTerms, compSpec.params.candidateMultiplier,
                        compSpec.params.minCandidates, compSpec.params.minScore, compSpec.params.rm3Enabled,
                        compSpec.params.rm3FbDocs, compSpec.params.rm3FbTerms, compSpec.params.rm3OrigWeight, compSpec.params.rm3MinDf
                    )
                    is RetrieverSpec.Ensemble -> throw IllegalArgumentException("Nested ensemble not supported")
                }
                compHandles.add(h)
                compWeights.add(spec.weights[i])
            }
            val ensHandle = NativeBenchmarkRunner.createEnsembleRetriever(
                compHandles.toLongArray(), compWeights.toFloatArray(),
                spec.fusionMethod.value, spec.rrfK, spec.topK, spec.parallel
            )
            RetrieverHandleSet(ensHandle, compHandles)
        }
    }

    /** Destroy C++ retrievers: ensemble first (holds raw pointers to components), then components */
    private fun destroyRetrieverHandleSet(handleSet: RetrieverHandleSet) {
        NativeBenchmarkRunner.destroyRetriever(handleSet.mainHandle)
        for (h in handleSet.componentHandles) {
            NativeBenchmarkRunner.destroyRetriever(h)
        }
    }

    /**
     * Create → Run → Destroy each retriever individually to minimize peak memory.
     * Handles are created directly via JNI (bypasses Kotlin ownedRetrievers).
     */
    private fun executeBenchmark(
        config: BenchmarkConfig,
        filterPath: String?,
        progressCallback: ProgressCallback? = null,
        resultCallback: ResultCallback? = null
    ): String {
        val tag = "BenchmarkRunner"
        val allMethodResults = org.json.JSONArray()
        var baseJson: org.json.JSONObject? = null
        val methodNames = mutableListOf<String>()
        val detailPaths = mutableListOf<String>()

        for ((i, spec) in config.retrievers.withIndex()) {
            val specName = spec.displayName()
            progressCallback?.onProgress(
                0.35f + (i.toFloat() / config.retrievers.size) * 0.6f,
                "벤치마크: $specName (${i + 1}/${config.retrievers.size})"
            )

            val handleSet = createRetrieverHandleSet(spec)
            try {
                val singleResult = NativeBenchmarkRunner.runRegisteredRetrievers(
                    longArrayOf(handleSet.mainHandle), config.evaluationTopK, filterPath
                )
                val parsed = org.json.JSONObject(singleResult)
                if (baseJson == null) baseJson = parsed

                val arr = parsed.optJSONArray("results") ?: continue
                for (j in 0 until arr.length()) {
                    val m = arr.getJSONObject(j)
                    m.put("method", specName)
                    allMethodResults.put(m)

                    // Collect detail file path for merge
                    val detailFile = m.optString("details_file", "")
                    if (detailFile.isNotEmpty() && j == 0) {
                        methodNames.add(specName)
                        detailPaths.add(detailFile)
                    }

                    resultCallback?.onResult(BenchmarkMethodResult(
                        method = m.optString("method"),
                        hit5 = m.optDouble("Hit@5"),
                        recall5 = m.optDouble("Recall@5"),
                        avgLatencyMs = m.optDouble("avg_latency_ms"),
                        correct = m.optInt("correct_results"),
                        total = m.optInt("total_queries")
                    ))
                }
            } finally {
                destroyRetrieverHandleSet(handleSet)
            }
            Log.i(tag, "Completed + destroyed: $specName (${i + 1}/${config.retrievers.size})")
        }

        val merged = baseJson ?: org.json.JSONObject()
        merged.put("results", allMethodResults)
        val result = merged.toString()
        Log.i(tag, "All ${config.retrievers.size} retrievers completed")
        saveResults(result)

        // Phase 2: merge per-retriever details into single per-query JSONL
        if (methodNames.isNotEmpty()) {
            try {
                val ts = java.text.SimpleDateFormat("yyyyMMdd_HHmm", java.util.Locale.US)
                    .format(java.util.Date())
                val mergedPath = "${context.getExternalFilesDir(null)}/benchmark_detail_$ts.jsonl"
                val mergeResult = NativeBenchmarkRunner.mergeDetails(
                    methodNames.toTypedArray(),
                    detailPaths.toTypedArray(),
                    mergedPath
                )
                Log.i(tag, "Merged detail JSONL: $mergeResult")
            } catch (e: Exception) {
                Log.w(tag, "Detail merge failed: ${e.message}")
            }
        }

        return result
    }

    /** 결과 JSON의 method 이름을 RetrieverSpec.name으로 교체 */
    private fun applyRetrieverNames(resultJson: String, config: BenchmarkConfig): String {
        return try {
            val parsed = org.json.JSONObject(resultJson)
            val arr = parsed.optJSONArray("results") ?: return resultJson
            for (i in 0 until arr.length()) {
                if (i >= config.retrievers.size) break
                val spec = config.retrievers[i]
                val name = spec.displayName()
                val m = arr.getJSONObject(i)
                m.put("method", name)
            }
            parsed.toString()
        } catch (_: Exception) {
            resultJson
        }
    }

    /**
     * 전체 파이프라인: 초기화 → 데이터 로딩 → 인덱스 빌드 → IRetriever 벤치마크
     */
    suspend fun run(
        datasetLoader: BenchmarkDatasetLoader,
        loadTarget: BenchmarkDatasetLoader.LoadTarget = BenchmarkDatasetLoader.LoadTarget.ALL,
        progressCallback: ProgressCallback,
        clearBeforeLoad: Boolean = true,
        filterPath: String? = null,
        config: BenchmarkConfig
    ): String {
        val tag = "BenchmarkRunner"

        progressCallback.onProgress(0.01f, "EcoVectorStore 초기화 중...")
        val eco = EcoVectorStore.create(context)
        ensureBenchmarkDb()
        Log.i(tag, "EcoVectorStore initialized")

        if (clearBeforeLoad) {
            progressCallback.onProgress(0.05f, "기존 데이터 클리어 중...")
            NativeEcoVectorStore.removeAll()
            Log.i(tag, "Database cleared")
        }

        progressCallback.onProgress(0.06f, "데이터셋 로딩 중...")
        val loadResult = datasetLoader.load(loadTarget) { p, msg ->
            progressCallback.onProgress(0.06f + p * 0.84f, msg)
        }
        if (loadResult.documentsLoaded == 0 && loadResult.queriesLoaded == 0
            && NativeEcoVectorStore.getDocumentCount() == 0) {
            throw IllegalStateException("데이터셋 로딩 실패")
        }

        val docCount = eco.documentCount
        val chunkCount = eco.chunkCount
        val queryCount = NativeBenchmarkRunner.getQueryCount()
        Log.i(tag, "Loaded: $docCount docs, $chunkCount chunks, $queryCount queries")

        progressCallback.onProgress(0.90f, "벤치마크 실행 중...")
        val result = executeBenchmark(config, filterPath)

        progressCallback.onProgress(1.0f, "완료! (문서: $docCount, 청크: $chunkCount, 쿼리: $queryCount)")
        return result
    }

    /**
     * 벤치마크 전용: 기존 DB 재활용
     */
    suspend fun runBenchmarkOnly(
        progressCallback: ProgressCallback,
        filterPath: String? = null,
        config: BenchmarkConfig,
        resultCallback: ResultCallback? = null
    ): String {
        val tag = "BenchmarkRunner"

        progressCallback.onProgress(0.01f, "EcoVectorStore 초기화 중...")
        val eco = EcoVectorStore.create(context)
        ensureBenchmarkDb()

        val docCount = eco.documentCount
        val chunkCount = eco.chunkCount
        val queryCount = NativeBenchmarkRunner.getQueryCount()
        Log.i(tag, "Existing data: $docCount docs, $chunkCount chunks, $queryCount queries")

        progressCallback.onProgress(0.1f, "벤치마크 실행 중... ($chunkCount chunks, $queryCount queries)")
        val result = executeBenchmark(config, filterPath, progressCallback, resultCallback)

        progressCallback.onProgress(1.0f, "완료! (문서: $docCount, 청크: $chunkCount, 쿼리: $queryCount)")
        return result
    }

    /**
     * 재토큰화 후 벤치마크
     */
    suspend fun runAfterReTokenize(
        progressCallback: ProgressCallback,
        filterPath: String? = null,
        config: BenchmarkConfig,
        resultCallback: ResultCallback? = null
    ): String {
        val tag = "BenchmarkRunner"

        progressCallback.onProgress(0.01f, "EcoVectorStore 초기화 중...")
        val eco = EcoVectorStore.create(context)
        ensureBenchmarkDb()

        val chunkCount = eco.chunkCount
        val queryCount = NativeBenchmarkRunner.getQueryCount()
        Log.i(tag, "Existing data: $chunkCount chunks, $queryCount queries")

        progressCallback.onProgress(0.1f, "Kiwi 재토큰화 중... ($chunkCount chunks, $queryCount queries)")
        NativeEcoVectorStore.reTokenizeAll()
        Log.i(tag, "Re-tokenization complete")

        progressCallback.onProgress(0.5f, "BM25 인덱스 재구축 중...")
        eco.buildBM25Index()
        Log.i(tag, "BM25 index rebuilt after re-tokenization")

        progressCallback.onProgress(0.6f, "벤치마크 실행 중...")
        val result = executeBenchmark(config, filterPath, progressCallback, resultCallback)

        progressCallback.onProgress(1.0f, "완료! ($chunkCount chunks, $queryCount queries)")
        return result
    }

    /**
     * 인덱스 재구축 스윕: 각 IndexSpec별로 벡터 인덱스를 재구축 후 벤치마크 실행.
     * 결과 JSON의 method 이름에 인덱스 파라미터가 포함됨.
     */
    suspend fun runWithIndexSweep(
        progressCallback: ProgressCallback,
        filterPath: String? = null,
        config: BenchmarkConfig,
        resultCallback: ResultCallback? = null
    ): String {
        val tag = "BenchmarkRunner"
        val indexConfigs = config.indexConfigs
            ?: return runBenchmarkOnly(progressCallback, filterPath, config, resultCallback)

        progressCallback.onProgress(0.01f, "EcoVectorStore 초기화 중...")
        val eco = EcoVectorStore.create(context)
        ensureBenchmarkDb()

        val chunkCount = eco.chunkCount
        val queryCount = NativeBenchmarkRunner.getQueryCount()
        Log.i(tag, "Index sweep: ${indexConfigs.size} configs, $chunkCount chunks, $queryCount queries")

        val allResults = mutableListOf<Pair<String, IndexSpec>>()

        for ((idx, idxSpec) in indexConfigs.withIndex()) {
            val label = idxSpec.name ?: when (idxSpec) {
                is IndexSpec.Vector -> idxSpec.params.label()
            }
            val pct = idx.toFloat() / indexConfigs.size

            progressCallback.onProgress(
                0.05f + pct * 0.90f,
                "인덱스 재구축 중 (${idx + 1}/${indexConfigs.size}): $label"
            )

            when (idxSpec) {
                is IndexSpec.Vector -> {
                    Log.i(tag, "Rebuilding vector index: $label")
                    val rebuildOk = NativeEcoVectorStore.buildVectorIndex(
                        idxSpec.params.nCluster, idxSpec.params.hnswM, idxSpec.params.efConstruction,
                        idxSpec.params.maxTrainSamples
                    )
                    if (!rebuildOk) {
                        Log.e(tag, "Vector index rebuild failed for $label")
                        continue
                    }
                }
            }

            // BM25 인덱스 빌드 (인덱스 스윕은 벡터만 재구축하므로 BM25도 보장)
            Log.i(tag, "Building BM25 index...")
            eco.buildBM25Index()

            progressCallback.onProgress(
                0.05f + pct * 0.90f + 0.45f / indexConfigs.size,
                "벤치마크 실행 중: $label"
            )

            // Run each retriever one at a time with create/destroy
            val indexMethodResults = org.json.JSONArray()
            var indexBaseJson: org.json.JSONObject? = null
            for (spec in config.retrievers) {
                val handleSet = createRetrieverHandleSet(spec)
                try {
                    val rawResult = NativeBenchmarkRunner.runRegisteredRetrievers(
                        longArrayOf(handleSet.mainHandle), config.evaluationTopK, filterPath
                    )
                    val parsed = org.json.JSONObject(rawResult)
                    if (indexBaseJson == null) indexBaseJson = parsed
                    val arr = parsed.optJSONArray("results") ?: continue
                    for (j in 0 until arr.length()) {
                        val m = arr.getJSONObject(j)
                        m.put("method", spec.displayName())
                        indexMethodResults.put(m)
                    }
                } finally {
                    destroyRetrieverHandleSet(handleSet)
                }
            }
            val indexMerged = indexBaseJson ?: org.json.JSONObject()
            indexMerged.put("results", indexMethodResults)
            allResults.add(indexMerged.toString() to idxSpec)
            Log.i(tag, "Completed: $label")
        }

        val merged = mergeIndexSweepResults(allResults)
        saveResults(merged)

        progressCallback.onProgress(1.0f, "완료! (${indexConfigs.size}개 인덱스 설정)")
        return merged
    }

    /**
     * 파이프라인 단계별 실행 + 벤치마크.
     * stages가 비어있으면 벤치마크만 실행 (기존 DB 재활용).
     */
    suspend fun runPipeline(
        stages: List<PipelineStage>,
        datasetLoader: BenchmarkDatasetLoader? = null,
        forceEmbedAll: Boolean = false,
        progressCallback: ProgressCallback? = null,
        filterPath: String? = null,
        config: BenchmarkConfig,
        resultCallback: ResultCallback? = null,
        importPath: String? = null
    ): String {
        val tag = "BenchmarkRunner"

        progressCallback?.onProgress(0.01f, "EcoVectorStore 초기화 중...")
        val eco = EcoVectorStore.create(context)
        ensureBenchmarkDb()

        if (stages.isNotEmpty()) {
            progressCallback?.onProgress(0.02f, "Pipeline: ${stages.joinToString { it.intentKey }}")
        } else {
            progressCallback?.onProgress(0.02f, "No stages — benchmark only")
        }

        val stageCount = stages.size.coerceAtLeast(1).toFloat()
        var stageIdx = 0

        fun stageProgress(label: String) {
            val p = 0.05f + (stageIdx / stageCount) * 0.55f
            progressCallback?.onProgress(p, label)
            stageIdx++
        }

        // === Phase 1: EcoVector DB ===
        if (PipelineStage.ECO_LOAD in stages) {
            stageProgress("[Phase 1] eco_load: 문서 적재")
            requireNotNull(datasetLoader) { "eco_load requires datasetLoader" }
            val count = datasetLoader.loadDocumentsOnly { p, msg ->
                progressCallback?.onProgress(0.05f + p * 0.1f, msg)
            }
            Log.i(tag, "eco_load: $count documents loaded")
        }

        if (PipelineStage.ECO_CHUNK in stages) {
            stageProgress("[Phase 1] eco_chunk: 청크 분할")
            val chunkCount = NativeEcoVectorStore.chunkAllDocuments()
            Log.i(tag, "eco_chunk: $chunkCount chunks created")
        }

        if (PipelineStage.ECO_EXPORT_CHUNKS in stages) {
            stageProgress("[Phase 1] eco_export_chunks: 청크 SQLite export")
            val exportDir = context.getExternalFilesDir(null)?.absolutePath ?: context.filesDir.absolutePath
            val exportPath = "$exportDir/chunks_export.db"
            val exportCount = NativeEcoVectorStore.exportChunksToSQLite(exportPath)
            Log.i(tag, "eco_export_chunks: $exportCount chunks exported to $exportPath")
        }

        // eco_embed와 eco_import_embed는 상호 배타적 — import가 우선
        if (PipelineStage.ECO_IMPORT_EMBED in stages) {
            stageProgress("[Phase 1] eco_import_embed: SQLite에서 임베딩 import")
            val defaultPath = "${context.getExternalFilesDir(null)?.absolutePath}/chunks_export.db"
            val effectiveImportPath = importPath ?: defaultPath
            val importCount = NativeEcoVectorStore.importEmbeddingsFromSQLite(effectiveImportPath)
            Log.i(tag, "eco_import_embed: $importCount chunks updated from $effectiveImportPath")
        } else if (PipelineStage.ECO_EMBED in stages) {
            stageProgress("[Phase 1] eco_embed: 임베딩 (forceAll=$forceEmbedAll)")
            val embedCount = NativeEcoVectorStore.embedChunks(forceEmbedAll)
            Log.i(tag, "eco_embed: $embedCount chunks embedded")
        }

        if (PipelineStage.ECO_TOKENIZE in stages) {
            stageProgress("[Phase 1] eco_tokenize: Kiwi 토큰화")
            val tokCount = NativeEcoVectorStore.tokenizeChunks()
            Log.i(tag, "eco_tokenize: $tokCount chunks tokenized")
        }

        if (PipelineStage.ECO_VECTOR_INDEX in stages) {
            stageProgress("[Phase 1] eco_vector_index: 벡터 인덱스 빌드")
            NativeEcoVectorStore.buildVectorIndex(0, 0, 0, 0)
            Log.i(tag, "eco_vector_index: done")
        }

        if (PipelineStage.ECO_BM25_INDEX in stages) {
            stageProgress("[Phase 1] eco_bm25_index: BM25 인덱스 빌드")
            NativeEcoVectorStore.buildBM25Index()
            Log.i(tag, "eco_bm25_index: done")
        }

        // === Phase 2: Benchmark DB ===
        if (PipelineStage.BENCH_LOAD in stages) {
            stageProgress("[Phase 2] bench_load: 질의 적재")
            NativeBenchmarkRunner.clearBenchmarkQueries()
            Log.i(tag, "bench_load: cleared existing queries and ground truths")
            requireNotNull(datasetLoader) { "bench_load requires datasetLoader" }
            val count = datasetLoader.loadQueriesTextOnly { p, msg ->
                progressCallback?.onProgress(0.4f + p * 0.1f, msg)
            }
            Log.i(tag, "bench_load: $count queries loaded (text-only)")
        }

        // 쿼리 임베딩 임포트 (같은 SQLite에 query_embeddings 테이블이 있으면 자동 임포트)
        if (PipelineStage.ECO_IMPORT_EMBED in stages && importPath != null) {
            val queryImportCount = NativeBenchmarkRunner.importQueryEmbeddingsFromSQLite(importPath)
            Log.i(tag, "Query embedding import: $queryImportCount queries")
            stageProgress("쿼리 임베딩 임포트: ${queryImportCount}개")
        }

        if (PipelineStage.BENCH_EMBED in stages) {
            stageProgress("[Phase 2] bench_embed: 질의 임베딩")
            val count = NativeBenchmarkRunner.embedAllQueries()
            Log.i(tag, "bench_embed: $count queries embedded")
        }

        if (PipelineStage.BENCH_TOKENIZE in stages) {
            stageProgress("[Phase 2] bench_tokenize: 질의 Kiwi 토큰화")
            val count = NativeBenchmarkRunner.tokenizeAllQueries()
            Log.i(tag, "bench_tokenize: $count queries tokenized")
        }

        // === Phase 3: 벤치마크 실행 ===
        val docCount = eco.documentCount
        val chunkCount = eco.chunkCount
        val queryCount = NativeBenchmarkRunner.getQueryCount()
        progressCallback?.onProgress(0.65f, "벤치마크 실행 중... ($docCount docs, $chunkCount chunks, $queryCount queries)")

        // filterPath=null → "NONE" (explicitly disable filters)
        // filterPath="path" → use that path
        // Old code paths pass null for auto-discover, but runPipeline uses explicit semantics
        val effectiveFilterPath = filterPath ?: "NONE"
        val result = executeBenchmark(config, effectiveFilterPath, progressCallback, resultCallback)

        progressCallback?.onProgress(1.0f, "완료! (문서: $docCount, 청크: $chunkCount, 쿼리: $queryCount)")
        return result
    }

    private fun mergeIndexSweepResults(
        results: List<Pair<String, IndexSpec>>
    ): String {
        val merged = org.json.JSONObject()
        val mergedResults = org.json.JSONArray()

        var timestamp = ""
        var dataset = "unknown"
        var docCount = 0
        var chunkCount = 0
        var queryCount = 0

        for ((i, pair) in results.withIndex()) {
            val (resultJson, idxSpec) = pair
            val parsed = org.json.JSONObject(resultJson)
            if (i == 0) {
                timestamp = parsed.optString("timestamp", "")
                dataset = parsed.optString("dataset", "unknown")
                docCount = parsed.optInt("doc_count")
                chunkCount = parsed.optInt("chunk_count")
                queryCount = parsed.optInt("query_count")
            }

            val label = idxSpec.name ?: when (idxSpec) {
                is IndexSpec.Vector -> idxSpec.params.label()
            }
            val rawResults = parsed.optJSONArray("results") ?: continue
            for (j in 0 until rawResults.length()) {
                val m = rawResults.getJSONObject(j)
                val origMethod = m.optString("method")
                m.put("method", "[$label] $origMethod")
                mergedResults.put(m)
            }
        }

        merged.put("timestamp", timestamp)
        merged.put("dataset", dataset)
        merged.put("doc_count", docCount)
        merged.put("chunk_count", chunkCount)
        merged.put("query_count", queryCount)
        merged.put("results", mergedResults)
        return merged.toString()
    }
}
