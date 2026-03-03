package io.graphai.ecovector.benchmark.loaders

import android.content.res.AssetManager
import android.util.Log
import io.graphai.ecovector.NativeEcoVectorStore
import io.graphai.ecovector.TextCleaner
import org.json.JSONObject

class CallDocumentLoader : DomainLoader {
    override val domain = "call"
    override val sourceType: Short = 0

    companion object {
        private const val TAG = "CallDocumentLoader"

        private fun extractCallText(json: JSONObject): String {
            val conversation = json.getJSONArray("conversation")
            val sb = StringBuilder()
            for (i in 0 until conversation.length()) {
                val line = conversation.getString(i)
                val textStart = line.indexOf(':')
                if (textStart >= 0 && textStart < line.length - 1) {
                    if (sb.isNotEmpty()) sb.append('\n')
                    sb.append(line.substring(textStart + 1))
                }
            }
            return sb.toString()
        }
    }

    override suspend fun load(
        assets: AssetManager,
        datasetDir: String,
        existingDocIds: Set<String>,
        rawIdMap: MutableMap<String, String>,
        documentOnly: Boolean,
        progressCallback: ((Float) -> Unit)?
    ): Int {
        val callDir = "$datasetDir/call"
        val files = assets.list(callDir)?.filter { it.endsWith(".json") }?.sorted() ?: emptyList()
        Log.i(TAG, "Call: ${files.size} files found")

        var loadedCount = 0
        var skippedCount = 0

        files.forEachIndexed { index, fileName ->
            try {
                val rawId = fileName.removeSuffix(".json")
                val compositeId = "call_$rawId"

                if (compositeId in existingDocIds) {
                    skippedCount++
                    rawIdMap[rawId.lowercase()] = compositeId
                    progressCallback?.invoke((index + 1).toFloat() / files.size)
                    return@forEachIndexed
                }

                val jsonStr = assets.open("$callDir/$fileName").use { it.bufferedReader().readText() }
                val json = JSONObject(jsonStr)
                val text = TextCleaner.cleanDocument(extractCallText(json))
                val summary = json.optString("summary_1line", "").trim()

                if (text.isNotBlank()) {
                    if (documentOnly) {
                        val count = NativeEcoVectorStore.addDocumentsOnly(
                            arrayOf(text), arrayOf(summary.ifEmpty { "Call $rawId" }),
                            arrayOf(compositeId),
                            longArrayOf(JsonlDomainLoader.parseIsoDatetimeToMs(json.optString("call_start_datetime", ""))),
                            arrayOf(""), intArrayOf(sourceType.toInt())
                        )
                        if (count > 0) loadedCount++
                    } else {
                        // summary를 chunk prefix로 설정
                        if (summary.isNotEmpty()) {
                            NativeEcoVectorStore.setChunkPrefix(summary)
                        }
                        val docIds = NativeEcoVectorStore.addDocumentsWithMetadata(
                            arrayOf(text), arrayOf(summary.ifEmpty { "Call $rawId" }),
                            arrayOf(compositeId),
                            longArrayOf(JsonlDomainLoader.parseIsoDatetimeToMs(json.optString("call_start_datetime", ""))),
                            arrayOf(""), shortArrayOf(sourceType)
                        )
                        if (summary.isNotEmpty()) {
                            NativeEcoVectorStore.clearChunkPrefix()
                        }
                        if (docIds[0] > 0) loadedCount++
                    }
                    rawIdMap[rawId.lowercase()] = compositeId
                }
            } catch (e: Exception) {
                Log.w(TAG, "Call: failed to parse $fileName: ${e.message}")
            }

            if ((index + 1) % 50 == 0) {
                System.gc()
                val rt = Runtime.getRuntime()
                val usedMB = (rt.totalMemory() - rt.freeMemory()) / 1024 / 1024
                val maxMB = rt.maxMemory() / 1024 / 1024
                Log.i(TAG, "Call: ${index + 1}/${files.size}, loaded $loadedCount, skipped $skippedCount, heap=${usedMB}MB/${maxMB}MB")
            }
            progressCallback?.invoke((index + 1).toFloat() / files.size)
        }

        Log.i(TAG, "Call: loaded $loadedCount new, skipped $skippedCount existing (total ${files.size})")
        return loadedCount
    }
}
