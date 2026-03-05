package io.graphai.ecovector.benchmark

import android.content.res.AssetManager
import android.util.Log
import io.graphai.ecovector.NativeEcoVectorStore
import io.graphai.ecovector.benchmark.loaders.*
import org.json.JSONArray
import org.json.JSONObject

/**
 * 통합 벤치마크 데이터셋 로더.
 * dataset 디렉토리에서 document, query, ground truth를 적재 (임베딩 포함).
 * 증분 로딩 지원: DB에 이미 있는 데이터는 건너뛰고 없는 것만 적재.
 */
class BenchmarkDatasetLoader(
    private val assets: AssetManager,
    private val datasetDir: String = "datasets/uplus_dataset/data",
    private val gtPath: String = "datasets/uplus_dataset/quries/valid_filters.jsonl",
    private val gtFilePath: String? = null,
    private val domainLoaders: List<DomainLoader> = listOf(
        PdfDocumentLoader(), MmsDocumentLoader(), SmsDocumentLoader(), CallDocumentLoader()
    )
) {
    companion object {
        private const val TAG = "BenchmarkDatasetLoader"
        private const val QUERY_BATCH_LOG_INTERVAL = 20
    }

    enum class LoadTarget { DOCUMENTS, QUERIES, ALL }

    data class LoadResult(
        val documentsLoaded: Int = 0,
        val queriesLoaded: Int = 0,
        val groundTruthsLoaded: Int = 0
    )

    /** domain → (rawId.lowercase → compositeId) 매핑 (GT 해석용) */
    private val rawIdToComposite = mutableMapOf<String, MutableMap<String, String>>()
    private lateinit var existingDocIds: Set<String>
    private lateinit var existingQueryIds: Set<String>

    private val domainProgressWeights = floatArrayOf(0.02f, 0.04f, 0.12f, 0.75f)
    private val gtProgressWeight = 0.07f

    suspend fun load(
        target: LoadTarget = LoadTarget.ALL,
        progressCallback: ((Float, String) -> Unit)? = null
    ): LoadResult {
        rawIdToComposite.clear()
        loadExistingIds()
        cleanupOrphanDocuments()

        var docsLoaded = 0
        if (target == LoadTarget.DOCUMENTS || target == LoadTarget.ALL) {
            docsLoaded = loadDocuments { p, msg -> progressCallback?.invoke(p, msg) }
        }

        var queriesLoaded = 0
        var gtsLoaded = 0
        if (target == LoadTarget.QUERIES || target == LoadTarget.ALL) {
            val (q, g) = loadQueries { p, msg ->
                val offset = if (target == LoadTarget.ALL) 1f - gtProgressWeight else 0f
                progressCallback?.invoke(offset + p * gtProgressWeight, msg)
            }
            queriesLoaded = q
            gtsLoaded = g
        }

        progressCallback?.invoke(1f, "완료")
        Log.i(TAG, "=== Complete: $docsLoaded docs, $queriesLoaded queries, $gtsLoaded GTs ===")
        return LoadResult(docsLoaded, queriesLoaded, gtsLoaded)
    }

    // ==================== 기존 ID 조회 (증분 로딩) ====================

    private fun loadExistingIds() {
        val docCount = NativeEcoVectorStore.getDocumentCount()
        val queryCount = NativeBenchmarkRunner.getQueryCount()

        if (docCount == 0 && queryCount == 0) {
            existingDocIds = emptySet()
            existingQueryIds = emptySet()
            Log.i(TAG, "Empty DB — full load mode")
            return
        }

        val docIds = mutableSetOf<String>()
        val docIdsJson = NativeEcoVectorStore.getDocumentExternalIdsJson()
        val docIdsArr = JSONArray(docIdsJson)
        for (i in 0 until docIdsArr.length()) {
            val extId = docIdsArr.getString(i)
            if (extId.isNotEmpty()) {
                docIds.add(extId)
                val underscore = extId.indexOf('_')
                if (underscore > 0) {
                    val domain = extId.substring(0, underscore)
                    val rawId = extId.substring(underscore + 1)
                    rawIdToComposite.getOrPut(domain) { mutableMapOf() }[rawId.lowercase()] = extId
                }
            }
        }
        existingDocIds = docIds

        val qIds = mutableSetOf<String>()
        if (queryCount > 0) {
            val queryIdsJson = NativeBenchmarkRunner.getQueryExternalIdsJson()
            val queryIdsArr = JSONArray(queryIdsJson)
            for (i in 0 until queryIdsArr.length()) {
                val extId = queryIdsArr.getString(i)
                if (extId.isNotEmpty()) qIds.add(extId)
            }
        }
        existingQueryIds = qIds
        Log.i(TAG, "Incremental mode: ${existingDocIds.size} docs, ${existingQueryIds.size} queries in DB")
    }

    // ==================== Orphan 정리 ====================

    private fun cleanupOrphanDocuments() {
        try {
            val orphansJson = NativeEcoVectorStore.removeOrphanDocuments()
            val orphanArray = JSONArray(orphansJson)
            if (orphanArray.length() > 0) {
                val orphanIds = (0 until orphanArray.length()).map { orphanArray.getString(it) }
                Log.w(TAG, "Removed ${orphanIds.size} orphan documents: ${orphanIds.take(5)}")
                existingDocIds = existingDocIds.filterNot { it in orphanIds }.toSet()
                orphanIds.forEach { compositeId ->
                    val underscore = compositeId.indexOf('_')
                    if (underscore > 0) {
                        val domain = compositeId.substring(0, underscore)
                        val rawId = compositeId.substring(underscore + 1)
                        rawIdToComposite[domain]?.remove(rawId.lowercase())
                    }
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Orphan cleanup failed: ${e.message}")
        }
    }

    // ==================== 문서 적재 (document-only, no chunk/embed/tokenize) ====================

    suspend fun loadDocumentsOnly(
        progressCallback: ((Float, String) -> Unit)? = null
    ): Int {
        rawIdToComposite.clear()
        loadExistingIds()
        cleanupOrphanDocuments()

        var totalLoaded = 0
        var progressOffset = 0f

        domainLoaders.forEachIndexed { i, loader ->
            val weight = domainProgressWeights.getOrElse(i) { 0.1f }
            val offset = progressOffset
            val rawMap = rawIdToComposite.getOrPut(loader.domain) { mutableMapOf() }
            progressCallback?.invoke(offset, "${loader.domain} 로딩 중 (문서만)...")
            val count = loader.load(assets, datasetDir, existingDocIds, rawMap, documentOnly = true) { phase ->
                progressCallback?.invoke(offset + phase * weight, "${loader.domain} 로딩 중 (문서만)...")
            }
            totalLoaded += count
            progressOffset += weight
        }

        Log.i(TAG, "Documents loaded (document-only): $totalLoaded new")
        return totalLoaded
    }

    // ==================== 쿼리 적재 (text-only, no embed/tokenize) ====================

    fun loadQueriesTextOnly(
        progressCallback: ((Float, String) -> Unit)? = null
    ): Int {
        // 항상 재로드 (bench_load 전에 clearBenchmarkQueries 호출될 수 있으므로)
        loadExistingIds()

        val lines = readJsonlAsset(gtPath)
        Log.i(TAG, "GT (text-only): ${lines.size} entries found")

        var savedQueryCount = 0
        val gtQueryIds = mutableListOf<String>()
        val gtDocIds = mutableListOf<String>()
        var skippedImage = 0; var skippedOrphan = 0; var skippedExisting = 0

        lines.forEachIndexed { i, line ->
            try {
                val json = JSONObject(line)
                val queryId = json.getInt("query_id")
                val queryExternalId = "q_$queryId"

                if (queryExternalId in existingQueryIds) {
                    skippedExisting++
                    return@forEachIndexed
                }

                val queryText = JsonlDomainLoader.sanitizeText(json.getString("query"))
                val originalQuery = JsonlDomainLoader.sanitizeText(json.optString("original_query", ""))
                    .ifEmpty { queryText }
                val refinedQuery = if (json.has("original_query")) queryText else queryText
                val targetsArray = json.getJSONArray("target_corpus_ids")
                val domainArray = if (json.isNull("domains")) null else json.optJSONArray("domains")

                val domains = if (domainArray != null) {
                    (0 until domainArray.length()).map {
                        domainArray.getString(it).lowercase().let { d ->
                            if (d == "message") "sms" else d
                        }
                    }.toSet()
                } else emptySet()

                if (domains.size == 1 && domains.first() == "image") {
                    skippedImage++
                    return@forEachIndexed
                }

                val resolvedTargets = mutableListOf<String>()
                for (j in 0 until targetsArray.length()) {
                    resolveCompositeIds(targetsArray.getString(j)).let { resolvedTargets.addAll(it) }
                }

                if (resolvedTargets.isEmpty()) {
                    skippedOrphan++
                    // orphan이어도 스킵하지 않음
                }

                val evalTopK = json.optInt("top_k", 0)

                val id = NativeBenchmarkRunner.saveQueryTextOnly(
                    externalId = queryExternalId,
                    text = originalQuery,
                    refinedQuery = refinedQuery,
                    createdAt = 0L,
                    targetTypes = domains.filter { it != "image" }.sorted().joinToString(","),
                    categories = json.optString("catogory", ""),
                    split = "",
                    evalTopK = evalTopK
                )

                if (id > 0) {
                    savedQueryCount++
                    resolvedTargets.forEach { docId ->
                        gtQueryIds.add(queryExternalId)
                        gtDocIds.add(docId)
                    }
                }

                if ((i + 1) % 20 == 0) {
                    progressCallback?.invoke(
                        (i + 1).toFloat() / lines.size,
                        "쿼리 텍스트 저장 중 (${i + 1}/${lines.size})"
                    )
                }
            } catch (e: Exception) {
                Log.w(TAG, "GT (text-only): parse failed: ${e.message}")
            }
        }

        if (gtQueryIds.isNotEmpty()) {
            NativeBenchmarkRunner.saveGroundTruths(
                gtQueryIds.toTypedArray(), gtDocIds.toTypedArray()
            )
        }

        Log.i(TAG, "GT (text-only): $savedQueryCount new, $skippedExisting existing, $skippedImage image, $skippedOrphan orphan, ${gtDocIds.size} GT pairs")
        return savedQueryCount
    }

    // ==================== 문서 적재 (기존 — 전체 파이프라인) ====================

    private suspend fun loadDocuments(
        progressCallback: ((Float, String) -> Unit)?
    ): Int {
        var totalLoaded = 0
        var progressOffset = 0f

        domainLoaders.forEachIndexed { i, loader ->
            val weight = domainProgressWeights.getOrElse(i) { 0.1f }
            val offset = progressOffset
            val rawMap = rawIdToComposite.getOrPut(loader.domain) { mutableMapOf() }
            progressCallback?.invoke(offset, "${loader.domain} 로딩 중...")
            val count = loader.load(assets, datasetDir, existingDocIds, rawMap) { phase ->
                progressCallback?.invoke(offset + phase * weight, "${loader.domain} 로딩 중...")
            }
            totalLoaded += count
            progressOffset += weight
        }

        Log.i(TAG, "Documents loaded: ${domainLoaders.mapIndexed { i, l ->
            "${l.domain}=${if (i < domainLoaders.size) "ok" else "?"}"
        }.joinToString(", ")} (total new: $totalLoaded)")
        return totalLoaded
    }

    // ==================== 쿼리 + GT 적재 ====================

    private fun loadQueries(
        progressCallback: ((Float, String) -> Unit)?
    ): Pair<Int, Int> {
        val lines = readJsonlAsset(gtPath)
        Log.i(TAG, "GT: ${lines.size} entries found")

        // 1. GT 파싱 — 쿼리 텍스트, ID, 도메인 매핑
        data class ParsedQuery(
            val text: String,            // original_query (DB에 저장할 텍스트)
            val embeddingText: String,   // LLM refined query (임베딩/토큰화용)
            val externalId: String,
            val targetCompositeIds: List<String>,
            val targetTypes: String,
            val categories: String,
            val evalTopK: Int = 0        // LLM 결정 평가 top-K (5 또는 50)
        )

        val parsedQueries = mutableListOf<ParsedQuery>()
        var skippedImage = 0; var skippedOrphan = 0; var skippedExisting = 0

        lines.forEach { line ->
            try {
                val json = JSONObject(line)
                val queryId = json.getInt("query_id")
                val queryExternalId = "q_$queryId"

                if (queryExternalId in existingQueryIds) {
                    skippedExisting++
                    return@forEach
                }

                val queryText = JsonlDomainLoader.sanitizeText(json.getString("query"))
                val originalQuery = JsonlDomainLoader.sanitizeText(json.optString("original_query", ""))
                    .ifEmpty { queryText }
                val refinedQuery = if (json.has("original_query")) queryText else queryText
                val targetsArray = json.getJSONArray("target_corpus_ids")
                val domainArray = if (json.isNull("domains")) null else json.optJSONArray("domains")

                val domains = if (domainArray != null) {
                    (0 until domainArray.length()).map {
                        domainArray.getString(it).lowercase().let { d ->
                            if (d == "message") "sms" else d
                        }
                    }.toSet()
                } else emptySet()

                if (domains.size == 1 && domains.first() == "image") {
                    skippedImage++
                    return@forEach
                }

                val resolvedTargets = mutableListOf<String>()
                for (i in 0 until targetsArray.length()) {
                    val rawId = targetsArray.getString(i)
                    resolveCompositeIds(rawId).let { resolvedTargets.addAll(it) }
                }

                if (resolvedTargets.isEmpty()) {
                    skippedOrphan++
                    // orphan이어도 스킵하지 않음 — 검색은 수행, GT만 없음
                }

                parsedQueries.add(ParsedQuery(
                    text = originalQuery,
                    embeddingText = refinedQuery,
                    externalId = queryExternalId,
                    targetCompositeIds = resolvedTargets,
                    targetTypes = domains.filter { it != "image" }.sorted().joinToString(","),
                    categories = json.optString("catogory", ""),
                    evalTopK = json.optInt("top_k", 0)
                ))
            } catch (e: Exception) {
                Log.w(TAG, "GT: parse failed: ${e.message}")
            }
        }

        Log.i(TAG, "GT: ${parsedQueries.size} new, $skippedExisting existing, $skippedImage image, $skippedOrphan orphan")

        // 2. 쿼리 임베딩 + 저장 (Kotlin 레벨)
        var savedQueryCount = 0
        val allSavedQueryIds = mutableListOf<String>()
        val allSavedTargets = mutableListOf<List<String>>()

        parsedQueries.forEachIndexed { i, pq ->
            val embedding = NativeEcoVectorStore.embed(pq.embeddingText)       // refined query → 벡터 검색용
            val tokenIds = NativeEcoVectorStore.tokenize(pq.embeddingText)   // refined query → 임베딩 토큰
            val kiwiTokens = NativeEcoVectorStore.tokenizeKiwi(pq.text)      // original query → BM25용

            val queryId = NativeBenchmarkRunner.saveQueryRaw(
                pq.externalId, pq.text, pq.embeddingText,
                tokenIds, embedding, kiwiTokens,
                0L, pq.targetTypes, pq.categories, "",
                pq.evalTopK
            )

            if (queryId > 0) {
                savedQueryCount++
                allSavedQueryIds.add(pq.externalId)
                allSavedTargets.add(pq.targetCompositeIds)
            }

            if ((i + 1) % QUERY_BATCH_LOG_INTERVAL == 0) {
                Log.i(TAG, "Queries: ${i + 1}/${parsedQueries.size} embedded")
                progressCallback?.invoke(
                    (i + 1).toFloat() / parsedQueries.size,
                    "쿼리 임베딩 중 (${i + 1}/${parsedQueries.size})"
                )
            }
        }

        // 3. GroundTruth 저장
        val gtQueryIds = mutableListOf<String>()
        val gtDocIds = mutableListOf<String>()
        allSavedQueryIds.forEachIndexed { i, queryId ->
            allSavedTargets[i].forEach { docCompositeId ->
                gtQueryIds.add(queryId)
                gtDocIds.add(docCompositeId)
            }
        }

        var gtsSaved = 0
        if (gtQueryIds.isNotEmpty()) {
            val gtIds = NativeBenchmarkRunner.saveGroundTruths(
                gtQueryIds.toTypedArray(), gtDocIds.toTypedArray()
            )
            gtsSaved = gtIds.count { it > 0 }
            Log.i(TAG, "GT: saved $gtsSaved entries (${allSavedQueryIds.size} queries, ${gtDocIds.size} pairs)")
        }

        return Pair(savedQueryCount, gtsSaved)
    }

    // ==================== Helpers ====================

    private fun readJsonlAsset(assetPath: String): List<String> {
        val inputStream = if (gtFilePath != null && assetPath == gtPath) {
            Log.i(TAG, "Reading GT from filesystem: $gtFilePath")
            java.io.File(gtFilePath).inputStream()
        } else {
            assets.open(assetPath)
        }
        return inputStream.bufferedReader().useLines { lines ->
            lines.filter { it.isNotBlank() && it.trim().startsWith("{") }.toList()
        }
    }

    private fun resolveCompositeIds(rawId: String): List<String> {
        // composite ID (e.g. "call_0000106")가 이미 DB에 있으면 그대로 사용
        if (rawId in existingDocIds) return listOf(rawId)

        val key = rawId.lowercase()
        val results = mutableListOf<String>()
        for ((_, rawMap) in rawIdToComposite) {
            rawMap[key]?.let { results.add(it) }
        }
        return results
    }
}
