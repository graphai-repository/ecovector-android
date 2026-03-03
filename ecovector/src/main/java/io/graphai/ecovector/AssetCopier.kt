package io.graphai.ecovector

import android.content.Context
import android.content.res.AssetManager
import android.util.Log
import java.io.File
import java.io.FileOutputStream

/**
 * Asset 파일을 내부 저장소로 복사하는 유틸리티
 */
object AssetCopier {

    /**
     * Asset 파일을 지정된 경로로 복사
     * @param assets AssetManager
     * @param assetPath Asset 내 경로 (예: "models/embedding/tokenizer.json")
     * @param destPath 복사 대상 절대 경로
     * @param overwrite true면 기존 파일 덮어쓰기
     * @return 복사 성공 여부
     */
    fun copyAsset(
        assets: AssetManager,
        assetPath: String,
        destPath: String,
        overwrite: Boolean = false
    ): Boolean {
        val destFile = File(destPath)

        // 이미 존재하고 덮어쓰기 안 함
        if (destFile.exists() && !overwrite) {
            return true
        }

        // 부모 디렉토리 생성
        destFile.parentFile?.mkdirs()

        return try {
            assets.open(assetPath).use { input ->
                FileOutputStream(destFile).use { output ->
                    input.copyTo(output, bufferSize = 8192)
                }
            }
            true
        } catch (e: Exception) {
            e.printStackTrace()
            false
        }
    }

    /**
     * 여러 Asset 파일을 복사
     * @return 모든 복사 성공 시 true
     */
    fun copyAssets(
        assets: AssetManager,
        assetDestPairs: List<Pair<String, String>>,
        overwrite: Boolean = false
    ): Boolean {
        return assetDestPairs.all { (assetPath, destPath) ->
            copyAsset(assets, assetPath, destPath, overwrite)
        }
    }

    /**
     * Asset 디렉토리의 모든 파일을 대상 디렉토리로 복사
     *
     * @param assets AssetManager
     * @param assetDir Asset 내 디렉토리 경로 (예: "models/KoSimCSE-bert-QInt8")
     * @param destDir 복사 대상 디렉토리 절대 경로
     * @param overwrite true면 기존 파일 덮어쓰기
     * @return 모든 파일 복사 성공 시 true
     */
    fun copyAssetDirectory(
        assets: AssetManager,
        assetDir: String,
        destDir: String,
        overwrite: Boolean = false
    ): Boolean {
        val files = assets.list(assetDir)
        if (files == null || files.isEmpty()) {
            // Asset not bundled — check if destination already has files from a previous install
            val destFiles = File(destDir).listFiles()
            if (destFiles != null && destFiles.isNotEmpty()) {
                Log.i("AssetCopier", "Asset directory not in APK but already exists on device: $destDir (${destFiles.size} files)")
                return true
            }
            Log.e("AssetCopier", "Asset directory not found or empty: $assetDir")
            return false
        }

        File(destDir).mkdirs()

        return files.all { fileName ->
            copyAsset(assets, "$assetDir/$fileName", "$destDir/$fileName", overwrite)
        }
    }
}

/**
 * 모델 파일 경로 데이터 클래스
 *
 * @param modelDir 디바이스 내 모델 파일들이 저장된 디렉토리 경로
 * @param tokenizerPath tokenizer.json 절대 경로
 * @param onnxModelPath model.onnx 절대 경로
 * @param obxDbDir ObjectBox 데이터베이스 디렉토리 경로
 */
data class ModelPaths(
    val modelDir: String,
    val tokenizerPath: String,
    val onnxModelPath: String,
    val obxDbDir: String,
    val kiwiModelDir: String
) {
    companion object {
        const val DEFAULT_MODEL_ASSET_DIR = "models/KoSimCSE-bert-QInt8"

        /**
         * Context와 asset 모델 디렉토리로부터 경로 생성
         *
         * @param context Android Context
         * @param modelAssetDir asset 내 모델 디렉토리 경로 (예: "models/KoSimCSE-bert-QInt8")
         */
        fun fromContext(
            context: Context,
            modelAssetDir: String = DEFAULT_MODEL_ASSET_DIR,
            dbName: String = "ecovector-db",
            kiwiModelName: String = "kiwi_model"
        ): ModelPaths {
            val modelDirName = modelAssetDir.substringAfterLast("/")
            val modelDir = "${context.filesDir}/models/$modelDirName"

            return ModelPaths(
                modelDir = modelDir,
                tokenizerPath = "$modelDir/tokenizer.json",
                onnxModelPath = "$modelDir/model.onnx",
                obxDbDir = "${context.filesDir}/$dbName",
                kiwiModelDir = "${context.filesDir}/$kiwiModelName"
            )
        }
    }
}
