package io.graphai.ecovector.benchmark.ui

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import io.graphai.ecovector.benchmark.BenchmarkConfig
import io.graphai.ecovector.benchmark.BenchmarkMethodResult
import io.graphai.ecovector.benchmark.RetrieverSpec

internal fun findSpec(baseName: String, config: BenchmarkConfig): RetrieverSpec? {
    return config.retrievers.firstOrNull { it.displayName() == baseName }
}

@Composable
internal fun RetrieverDetailCard(
    result: BenchmarkMethodResult,
    spec: RetrieverSpec?,
    isBest: Boolean = false,
    isDark: Boolean = false
) {
    val containerColor = if (isBest) {
        if (isDark) Color(0xFF1B3A1B) else Color(0xFFE8F5E9)
    } else {
        MaterialTheme.colorScheme.surfaceContainerLow
    }
    val accentColor = if (isBest) {
        if (isDark) GreenLight else GreenDark
    } else {
        MaterialTheme.colorScheme.onSurface
    }

    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = containerColor),
        shape = RoundedCornerShape(10.dp)
    ) {
        Column(
            modifier = Modifier.padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(4.dp)
        ) {
            result.indexLabel?.let { IndexBadge(it) }
            Text(
                result.baseName,
                style = MaterialTheme.typography.titleSmall,
                fontWeight = FontWeight.Bold,
                color = accentColor
            )

            if (spec != null) {
                val diffs = spec.nonDefaultParams()
                if (diffs.isNotEmpty()) {
                    diffs.forEach { ParamRow(it) }
                }

                if (spec is RetrieverSpec.Ensemble) {
                    Spacer(Modifier.height(2.dp))
                    SubRetrieverSection(spec)
                }
            }

            Spacer(Modifier.height(4.dp))

            MetricRow("Recall@5", "%.2f%%".format(result.recall5))
            MetricRow("Hit@5", "%.2f%%".format(result.hit5), "${result.correct}/${result.total}")
            MetricRow("Latency", "%.2f ms".format(result.avgLatencyMs))
        }
    }
}

@Composable
private fun SubRetrieverSection(ensemble: RetrieverSpec.Ensemble) {
    ensemble.components.forEachIndexed { i, comp ->
        val weight = ensemble.weights.getOrNull(i) ?: 0f
        SubRetrieverRow(comp, weight)
    }
}

@Composable
private fun SubRetrieverRow(spec: RetrieverSpec, weight: Float) {
    Surface(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 8.dp),
        shape = RoundedCornerShape(6.dp),
        color = MaterialTheme.colorScheme.surfaceContainerHigh.copy(alpha = 0.5f)
    ) {
        Column(
            modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp),
            verticalArrangement = Arrangement.spacedBy(1.dp)
        ) {
            Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                Text(
                    spec.displayName(),
                    style = MaterialTheme.typography.bodySmall,
                    fontWeight = FontWeight.Medium
                )
                Text(
                    "w=%.1f".format(weight),
                    style = MaterialTheme.typography.labelSmall,
                    fontFamily = FontFamily.Monospace,
                    color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f)
                )
            }
            val diffs = spec.nonDefaultParams()
            if (diffs.isNotEmpty()) {
                Text(
                    diffs.joinToString(", "),
                    style = MaterialTheme.typography.labelSmall,
                    fontFamily = FontFamily.Monospace,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
            }
        }
    }
}
