package io.graphai.ecovector.benchmark.ui

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import io.graphai.ecovector.benchmark.BenchmarkMethodResult

@Composable
internal fun BestResultBanner(best: BenchmarkMethodResult, isDark: Boolean) {
    val bannerBg = if (isDark) Color(0xFF1B3A1B) else Color(0xFFE8F5E9)
    val textColor = if (isDark) GreenLight else GreenDark

    Card(
        modifier = Modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = bannerBg),
        shape = RoundedCornerShape(12.dp)
    ) {
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 16.dp, vertical = 12.dp),
            verticalArrangement = Arrangement.spacedBy(4.dp)
        ) {
            Text(
                "Best Recall@5",
                style = MaterialTheme.typography.labelMedium,
                color = textColor.copy(alpha = 0.8f)
            )
            best.indexLabel?.let { IndexBadge(it) }
            Text(
                best.baseName,
                style = MaterialTheme.typography.bodyMedium,
                fontWeight = FontWeight.Medium,
                color = textColor
            )
            BannerMetricRow("Recall@5", "%.2f%%".format(best.recall5), "${best.correct}/${best.total}", textColor)
            BannerMetricRow("Hit@5", "%.2f%%".format(best.hit5), "", textColor)
            BannerMetricRow("Latency", "%.2f ms".format(best.avgLatencyMs), "", textColor)
        }
    }
}

@Composable
private fun BannerMetricRow(label: String, value: String, extra: String, textColor: Color) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(
            label,
            style = MaterialTheme.typography.bodySmall,
            color = textColor.copy(alpha = 0.7f)
        )
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            if (extra.isNotEmpty()) {
                Text(
                    extra,
                    style = MaterialTheme.typography.labelSmall,
                    color = textColor.copy(alpha = 0.5f)
                )
            }
            Text(
                value,
                style = MaterialTheme.typography.bodyMedium,
                fontWeight = FontWeight.Bold,
                fontFamily = FontFamily.Monospace,
                color = textColor
            )
        }
    }
}
