package io.graphai.ecovector.benchmark.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import io.graphai.ecovector.benchmark.BenchmarkMethodResult

@Composable
internal fun ResultsTable(results: List<BenchmarkMethodResult>, isDark: Boolean) {
    val bestRecall = results.maxOfOrNull { it.recall5 } ?: 0.0

    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(12.dp)
    ) {
        Column {
            // Header
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(start = 12.dp, end = 12.dp, top = 12.dp, bottom = 8.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    "Method",
                    modifier = Modifier.weight(1f),
                    style = MaterialTheme.typography.labelMedium,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                TableHeader("Recall", Modifier.width(56.dp))
                TableHeader("Hit@5", Modifier.width(56.dp))
                TableHeader("ms", Modifier.width(52.dp))
            }

            HorizontalDivider(
                modifier = Modifier.padding(horizontal = 12.dp),
                color = MaterialTheme.colorScheme.outlineVariant
            )

            // Rows
            results.forEachIndexed { index, r ->
                val isBest = r.recall5 == bestRecall
                val rowBg = when {
                    isBest && isDark -> Color(0xFF1B3A1B).copy(alpha = 0.4f)
                    isBest && !isDark -> Color(0xFFE8F5E9).copy(alpha = 0.6f)
                    index % 2 == 1 -> MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.3f)
                    else -> Color.Transparent
                }

                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .background(rowBg)
                        .padding(vertical = 6.dp, horizontal = 12.dp),
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Column(modifier = Modifier.weight(1f)) {
                        r.indexLabel?.let {
                            IndexBadge(it)
                            Spacer(Modifier.height(1.dp))
                        }
                        Text(
                            r.baseName,
                            style = MaterialTheme.typography.bodySmall,
                            fontWeight = if (isBest) FontWeight.Bold else FontWeight.Normal,
                            maxLines = 1,
                            overflow = TextOverflow.Ellipsis
                        )
                    }

                    Text(
                        "%.1f".format(r.recall5),
                        modifier = Modifier.width(56.dp),
                        style = MaterialTheme.typography.bodySmall,
                        fontFamily = FontFamily.Monospace,
                        textAlign = TextAlign.End,
                        fontWeight = if (isBest) FontWeight.Bold else FontWeight.Normal,
                        color = if (isBest) {
                            if (isDark) GreenLight else GreenDark
                        } else {
                            MaterialTheme.colorScheme.onSurface
                        }
                    )

                    Text(
                        "%.1f".format(r.hit5),
                        modifier = Modifier.width(56.dp),
                        style = MaterialTheme.typography.bodySmall,
                        fontFamily = FontFamily.Monospace,
                        textAlign = TextAlign.End,
                        fontWeight = if (isBest) FontWeight.Bold else FontWeight.Normal
                    )

                    Text(
                        "%.2f".format(r.avgLatencyMs),
                        modifier = Modifier.width(52.dp),
                        style = MaterialTheme.typography.bodySmall,
                        fontFamily = FontFamily.Monospace,
                        textAlign = TextAlign.End,
                        color = latencyColor(r.avgLatencyMs, isDark)
                    )
                }

                if (index < results.lastIndex) {
                    HorizontalDivider(
                        modifier = Modifier.padding(horizontal = 12.dp),
                        color = MaterialTheme.colorScheme.outlineVariant.copy(alpha = 0.3f)
                    )
                }
            }

            // Footer
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(start = 12.dp, end = 12.dp, top = 6.dp, bottom = 12.dp),
                horizontalArrangement = Arrangement.End
            ) {
                Text(
                    "단위: %  |  ms",
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.5f)
                )
            }
        }
    }
}

@Composable
private fun TableHeader(text: String, modifier: Modifier) {
    Text(
        text,
        modifier = modifier,
        style = MaterialTheme.typography.labelMedium,
        fontWeight = FontWeight.Bold,
        textAlign = TextAlign.End,
        color = MaterialTheme.colorScheme.onSurfaceVariant
    )
}
