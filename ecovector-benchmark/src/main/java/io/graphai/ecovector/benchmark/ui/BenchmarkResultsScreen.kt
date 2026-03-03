package io.graphai.ecovector.benchmark.ui

import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.luminance
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import io.graphai.ecovector.benchmark.BenchmarkConfig
import io.graphai.ecovector.benchmark.BenchmarkDbStats
import io.graphai.ecovector.benchmark.BenchmarkMethodResult

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun BenchmarkResultsScreen(
    isRunning: Boolean,
    progress: Float,
    status: String,
    dbStats: BenchmarkDbStats?,
    results: List<BenchmarkMethodResult>,
    config: BenchmarkConfig
) {
    val isDark = MaterialTheme.colorScheme.background.luminance() < 0.5f

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text("EcoVector Benchmark", fontWeight = FontWeight.Bold)
                        if (isRunning) {
                            Spacer(Modifier.width(8.dp))
                            PulsingDot()
                        }
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.primaryContainer,
                    titleContentColor = MaterialTheme.colorScheme.onPrimaryContainer
                )
            )
        }
    ) { innerPadding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 16.dp, vertical = 12.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            if (isRunning) {
                ProgressCard(progress, status)
            } else if (status.startsWith("오류")) {
                ErrorCard(status)
            }

            dbStats?.let { DbStatsRow(it) }

            if (results.isNotEmpty()) {
                ResultsTable(results, isDark)
            }

            if (!isRunning && results.isNotEmpty()) {
                val bestRecall = results.maxOfOrNull { it.recall5 } ?: 0.0
                Text(
                    "Retriever Details",
                    style = MaterialTheme.typography.titleSmall,
                    fontWeight = FontWeight.Bold,
                    color = MaterialTheme.colorScheme.onSurfaceVariant
                )
                results.forEach { r ->
                    val spec = findSpec(r.baseName, config)
                    RetrieverDetailCard(r, spec, isBest = r.recall5 == bestRecall, isDark = isDark)
                }
            }

            Spacer(Modifier.height(8.dp))
        }
    }
}
