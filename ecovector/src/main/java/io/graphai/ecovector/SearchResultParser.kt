package io.graphai.ecovector

import org.json.JSONArray
import org.json.JSONObject

internal fun parseSearchResults(json: String): List<SearchResult> {
    if (json.isEmpty() || json == "[]") return emptyList()
    val arr = JSONArray(json)
    return (0 until arr.length()).map { i ->
        val obj = arr.getJSONObject(i)
        SearchResult(
            documentId = obj.getLong("documentId"),
            chunkId = obj.getLong("chunkId"),
            content = obj.optString("content", ""),
            score = obj.getDouble("score").toFloat()
        )
    }
}

internal fun serializeWeightedResults(inputs: List<WeightedResults>): String {
    val arr = JSONArray()
    for (wr in inputs) {
        val obj = JSONObject()
        obj.put("weight", wr.weight.toDouble())
        if (wr.isDistance) obj.put("isDistance", true)
        val resultsArr = JSONArray()
        for (r in wr.results) {
            val ro = JSONObject()
            ro.put("documentId", r.documentId)
            ro.put("chunkId", r.chunkId)
            ro.put("content", r.content)
            ro.put("score", r.score.toDouble())
            resultsArr.put(ro)
        }
        obj.put("results", resultsArr)
        arr.put(obj)
    }
    return arr.toString()
}
