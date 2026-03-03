package io.graphai.ecovector.benchmark.ui

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp

// ── Colors ──────────────────────────────────────────────────────────────

internal val GreenDark = Color(0xFF2E7D32)
internal val GreenLight = Color(0xFF81C784)
internal val RedDark = Color(0xFFC62828)
internal val RedLight = Color(0xFFEF9A9A)

internal fun latencyColor(ms: Double, isDark: Boolean): Color {
    return when {
        ms < 1.0 -> if (isDark) GreenLight else GreenDark
        ms < 3.0 -> Color.Unspecified
        else -> if (isDark) RedLight else RedDark
    }
}

// ── Shared Components ───────────────────────────────────────────────────

@Composable
internal fun IndexBadge(label: String) {
    Surface(
        shape = RoundedCornerShape(4.dp),
        color = MaterialTheme.colorScheme.tertiary.copy(alpha = 0.15f)
    ) {
        Text(
            text = label,
            modifier = Modifier.padding(horizontal = 6.dp, vertical = 2.dp),
            style = MaterialTheme.typography.labelSmall,
            fontFamily = FontFamily.Monospace,
            fontWeight = FontWeight.Medium,
            color = MaterialTheme.colorScheme.tertiary
        )
    }
}

@Composable
internal fun MetricRow(label: String, value: String, extra: String = "") {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(
            label,
            style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            if (extra.isNotEmpty()) {
                Text(
                    extra,
                    style = MaterialTheme.typography.labelSmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = 0.6f)
                )
            }
            Text(
                value,
                style = MaterialTheme.typography.bodyMedium,
                fontWeight = FontWeight.Bold,
                fontFamily = FontFamily.Monospace
            )
        }
    }
}

@Composable
internal fun ParamRow(text: String) {
    Text(
        text,
        modifier = Modifier.padding(vertical = 1.dp),
        style = MaterialTheme.typography.bodySmall,
        fontFamily = FontFamily.Monospace,
        color = MaterialTheme.colorScheme.onSurfaceVariant
    )
}
