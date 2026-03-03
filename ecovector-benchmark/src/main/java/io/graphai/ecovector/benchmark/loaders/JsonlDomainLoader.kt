package io.graphai.ecovector.benchmark.loaders

import android.content.res.AssetManager
import android.util.Log
import io.graphai.ecovector.NativeEcoVectorStore

/**
 * JSONL 기반 도메인 로더 공통 로직 (SMS, MMS).
 * 서브클래스는 [assetSubPath]와 [parseRecord]만 구현.
 */
abstract class JsonlDomainLoader : DomainLoader {
    companion object {
        private const val TAG = "JsonlDomainLoader"
        const val BATCH_SIZE = 50

        fun sanitizeText(text: String): String =
            text.replace("\u0000", "")
                .replace(Regex("[\\x01-\\x08\\x0B\\x0C\\x0E-\\x1F]"), "")

        fun parseIsoDatetimeToMs(datetime: String): Long {
            return try {
                val parts = datetime.split(" ", "T")
                val dateParts = parts[0].split("-")
                val year = dateParts[0].toInt()
                val month = dateParts[1].toInt()
                val day = dateParts[2].toInt()
                var hour = 0; var min = 0; var sec = 0
                if (parts.size > 1) {
                    val timeParts = parts[1].split(":")
                    hour = timeParts[0].toInt()
                    min = timeParts.getOrNull(1)?.toInt() ?: 0
                    sec = timeParts.getOrNull(2)?.substringBefore(".")?.toInt() ?: 0
                }
                val cal = java.util.Calendar.getInstance(java.util.TimeZone.getTimeZone("UTC"))
                cal.set(year, month - 1, day, hour - 9, min, sec)
                cal.set(java.util.Calendar.MILLISECOND, 0)
                cal.timeInMillis
            } catch (e: Exception) { 0L }
        }
    }

    /** 데이터셋 디렉토리 기준 상대 경로 (예: "sms/sms.jsonl") */
    protected abstract val assetSubPath: String

    /** JSON 한 줄을 파싱하여 DocumentRecord 반환. 파싱 실패/빈 텍스트면 null */
    protected abstract fun parseRecord(json: org.json.JSONObject): DocumentRecord?

    data class DocumentRecord(
        val rawId: String,
        val text: String,
        val title: String,
        val createdAt: Long,
        val sender: String
    )

    override suspend fun load(
        assets: AssetManager,
        datasetDir: String,
        existingDocIds: Set<String>,
        rawIdMap: MutableMap<String, String>,
        documentOnly: Boolean,
        progressCallback: ((Float) -> Unit)?
    ): Int {
        val fullPath = "$datasetDir/$assetSubPath"
        val lines = assets.open(fullPath).bufferedReader().useLines { lines ->
            lines.filter { it.isNotBlank() && it.trim().startsWith("{") }.toList()
        }
        Log.i(TAG, "${domain.uppercase()}: ${lines.size} records found")

        var loadedCount = 0
        var skippedCount = 0

        val batches = lines.chunked(BATCH_SIZE)
        batches.forEachIndexed { batchIdx, batch ->
            val texts = mutableListOf<String>()
            val titles = mutableListOf<String>()
            val externalIds = mutableListOf<String>()
            val createdAts = mutableListOf<Long>()
            val senders = mutableListOf<String>()

            batch.forEach { line ->
                try {
                    val json = org.json.JSONObject(line)
                    val record = parseRecord(json) ?: return@forEach
                    val compositeId = "${domain}_${record.rawId}"

                    if (compositeId in existingDocIds) {
                        skippedCount++
                        rawIdMap[record.rawId.lowercase()] = compositeId
                        return@forEach
                    }

                    texts.add(record.text)
                    titles.add(record.title)
                    externalIds.add(compositeId)
                    createdAts.add(record.createdAt)
                    senders.add(record.sender)
                    rawIdMap[record.rawId.lowercase()] = compositeId
                } catch (e: Exception) {
                    Log.w(TAG, "${domain.uppercase()}: failed to parse line: ${e.message}")
                }
            }

            if (texts.isNotEmpty()) {
                if (documentOnly) {
                    val count = NativeEcoVectorStore.addDocumentsOnly(
                        texts.toTypedArray(), titles.toTypedArray(), externalIds.toTypedArray(),
                        createdAts.toLongArray(), senders.toTypedArray(),
                        IntArray(texts.size) { sourceType.toInt() }
                    )
                    loadedCount += count
                } else {
                    val sourceTypes = ShortArray(texts.size) { sourceType }
                    val docIds = NativeEcoVectorStore.addDocumentsWithMetadata(
                        texts.toTypedArray(), titles.toTypedArray(), externalIds.toTypedArray(),
                        createdAts.toLongArray(), senders.toTypedArray(), sourceTypes
                    )
                    loadedCount += docIds.count { it > 0 }
                }
            }

            progressCallback?.invoke((batchIdx + 1).toFloat() / batches.size)
        }

        Log.i(TAG, "${domain.uppercase()}: loaded $loadedCount new, skipped $skippedCount existing (total ${lines.size})")
        return loadedCount
    }
}
