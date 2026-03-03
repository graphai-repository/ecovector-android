package io.graphai.ecovector.benchmark

import org.json.JSONObject

data class BenchmarkDbStats(val docs: Int, val chunks: Int, val queries: Int)

data class BenchmarkMethodResult(
    val method: String,
    val hit5: Double,
    val recall5: Double,
    val avgLatencyMs: Double,
    val correct: Int,
    val total: Int
) {
    val indexLabel: String?
        get() = Regex("""^\[(.+?)]""").find(method)?.groupValues?.get(1)

    val baseName: String
        get() = method.replace(Regex("""^\[.+?]\s*"""), "")
}

fun parseBenchmarkResults(json: String): List<BenchmarkMethodResult>? {
    return try {
        val obj = JSONObject(json)
        val arr = obj.optJSONArray("results") ?: return null
        (0 until arr.length()).map { i ->
            val m = arr.getJSONObject(i)
            BenchmarkMethodResult(
                method = m.optString("method"),
                hit5 = m.optDouble("Hit@5"),
                recall5 = m.optDouble("Recall@5"),
                avgLatencyMs = m.optDouble("avg_latency_ms"),
                correct = m.optInt("correct_results"),
                total = m.optInt("total_queries")
            )
        }
    } catch (_: Exception) {
        null
    }
}
