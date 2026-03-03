package io.graphai.ecovector

/**
 * Configuration for EcoVectorStore initialization.
 *
 * @param modelAssetDir Model directory name under assets/models/ (default: "models/KoSimCSE-bert-QInt8")
 * @param dbName ObjectBox database directory name (default: "ecovector-db")
 * @param kiwiModelAssetDir Kiwi morphological analyzer model directory in assets (default: "kiwi_model")
 */
data class EcoVectorConfig(
    val modelAssetDir: String = ModelPaths.DEFAULT_MODEL_ASSET_DIR,
    val dbName: String = "ecovector-db",
    val kiwiModelAssetDir: String = "kiwi_model",
)
