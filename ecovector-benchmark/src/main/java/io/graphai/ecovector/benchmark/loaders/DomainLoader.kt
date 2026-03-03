package io.graphai.ecovector.benchmark.loaders

import android.content.res.AssetManager

/**
 * 단일 도메인(call/sms/mms/pdf)의 문서 로딩 전략.
 */
interface DomainLoader {
    /** 도메인 이름 (composite ID 접두사로 사용) */
    val domain: String

    /** 도메인의 sourceType 값 */
    val sourceType: Short

    /**
     * 도메인 문서를 배치 로딩.
     * @param existingDocIds DB에 이미 있는 문서 ID 집합 (스킵용)
     * @param rawIdMap 로딩된 rawId → compositeId 매핑 (GT 해석용, 호출자가 관리)
     * @param documentOnly true이면 문서만 저장 (청크/임베딩/토큰화 없이)
     * @return 새로 로딩된 문서 수
     */
    suspend fun load(
        assets: AssetManager,
        datasetDir: String,
        existingDocIds: Set<String>,
        rawIdMap: MutableMap<String, String>,
        documentOnly: Boolean = false,
        progressCallback: ((Float) -> Unit)? = null
    ): Int
}
