package io.graphai.ecovector.benchmark.loaders

import org.json.JSONObject

class MmsDocumentLoader : JsonlDomainLoader() {
    override val domain = "mms"
    override val sourceType: Short = 2
    override val assetSubPath = "mms/mms.jsonl"

    override fun parseRecord(json: JSONObject): DocumentRecord? {
        val rawId = json.getString("_id")
        val text = sanitizeText(json.optString("text_combined", ""))
        if (text.isBlank()) return null

        var createdAt = 0L
        if (json.has("date")) {
            var ms = json.getLong("date")
            if (ms < 1_000_000_000_000L) ms *= 1000L  // seconds → ms
            createdAt = ms
        }

        return DocumentRecord(
            rawId = rawId,
            text = text,
            title = json.optString("sub", "MMS $rawId"),
            createdAt = createdAt,
            sender = json.optString("address", "")
        )
    }
}
