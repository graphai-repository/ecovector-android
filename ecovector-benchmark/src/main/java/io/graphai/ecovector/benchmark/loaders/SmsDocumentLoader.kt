package io.graphai.ecovector.benchmark.loaders

import org.json.JSONObject

class SmsDocumentLoader : JsonlDomainLoader() {
    override val domain = "sms"
    override val sourceType: Short = 1
    override val assetSubPath = "sms/sms.jsonl"

    override fun parseRecord(json: JSONObject): DocumentRecord? {
        val rawId = json.getString("_id")
        val body = sanitizeText(json.getString("body"))
        if (body.isBlank()) return null

        val createdAt = if (json.has("datetime")) {
            parseIsoDatetimeToMs(json.getString("datetime"))
        } else if (json.has("date")) {
            json.getLong("date")
        } else 0L

        return DocumentRecord(
            rawId = rawId,
            text = body,
            title = "SMS $rawId",
            createdAt = createdAt,
            sender = json.optString("address", "")
        )
    }

}
