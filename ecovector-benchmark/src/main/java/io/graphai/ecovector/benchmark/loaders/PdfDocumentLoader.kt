package io.graphai.ecovector.benchmark.loaders

import android.content.res.AssetManager
import android.util.Log
import io.graphai.ecovector.NativeEcoVectorStore
import io.graphai.ecovector.PdfDocument

class PdfDocumentLoader : DomainLoader {
    override val domain = "document"
    override val sourceType: Short = 4

    companion object {
        private const val TAG = "PdfDocumentLoader"
    }

    override suspend fun load(
        assets: AssetManager,
        datasetDir: String,
        existingDocIds: Set<String>,
        rawIdMap: MutableMap<String, String>,
        documentOnly: Boolean,
        progressCallback: ((Float) -> Unit)?
    ): Int {
        val filesDir = "$datasetDir/files"
        val pdfFiles = assets.list(filesDir)?.filter { it.endsWith(".pdf") }?.sorted() ?: emptyList()
        Log.i(TAG, "Document: ${pdfFiles.size} PDF files found")

        var loadedCount = 0
        var skippedCount = 0
        var failedCount = 0

        pdfFiles.forEachIndexed { index, fileName ->
            val rawId = fileName.removeSuffix(".pdf")
            val compositeId = "document_$rawId"

            if (compositeId in existingDocIds) {
                skippedCount++
                rawIdMap[rawId.lowercase()] = compositeId
                progressCallback?.invoke((index + 1).toFloat() / pdfFiles.size)
                return@forEachIndexed
            }

            try {
                val doc = PdfDocument(
                    inputStreamProvider = { assets.open("$filesDir/$fileName") },
                    title = rawId
                )
                val text = JsonlDomainLoader.sanitizeText(doc.extractText())
                if (text.isNotBlank()) {
                    if (documentOnly) {
                        val count = NativeEcoVectorStore.addDocumentsOnly(
                            arrayOf(text), arrayOf(rawId), arrayOf(compositeId),
                            longArrayOf(0L), arrayOf(""), intArrayOf(sourceType.toInt())
                        )
                        if (count > 0) {
                            loadedCount++
                            rawIdMap[rawId.lowercase()] = compositeId
                        }
                    } else {
                        val sourceTypes = shortArrayOf(sourceType)
                        val docIds = NativeEcoVectorStore.addDocumentsWithMetadata(
                            arrayOf(text), arrayOf(rawId), arrayOf(compositeId),
                            longArrayOf(0L), arrayOf(""), sourceTypes
                        )
                        if (docIds[0] > 0) {
                            loadedCount++
                            rawIdMap[rawId.lowercase()] = compositeId
                        }
                    }
                } else {
                    Log.w(TAG, "Document: empty PDF skipped: $fileName")
                    failedCount++
                }
            } catch (e: OutOfMemoryError) {
                Log.e(TAG, "Document: OOM on $fileName - skipping and requesting GC")
                failedCount++
                System.gc()
            } catch (e: Exception) {
                Log.w(TAG, "Document: failed to extract $fileName: ${e.message}")
                failedCount++
            }

            if ((index + 1) % 10 == 0) {
                System.gc()
            }

            progressCallback?.invoke((index + 1).toFloat() / pdfFiles.size)
        }

        Log.i(TAG, "Document: loaded $loadedCount new, skipped $skippedCount existing, failed $failedCount (total ${pdfFiles.size})")
        return loadedCount
    }
}
