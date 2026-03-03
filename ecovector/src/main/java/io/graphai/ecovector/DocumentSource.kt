package io.graphai.ecovector

import java.io.InputStream

/**
 * 문서 소스 추상화.
 * 다양한 포맷(PDF, 텍스트, HTML 등)의 문서를 통일된 인터페이스로 제공한다.
 */
interface DocumentSource {
    val title: String

    /**
     * 문서에서 텍스트를 추출하고 정제하여 반환한다.
     * 반환값은 TextCleaner가 적용된 클린 텍스트.
     */
    suspend fun extractText(): String
}

/**
 * PDF 문서 소스.
 * PDFBox-Android를 사용하여 텍스트를 추출하고 TextCleaner로 정제한다.
 *
 * @param inputStreamProvider InputStream 팩토리 (일회성 스트림을 재생성 가능하게)
 * @param title 문서 제목 (DB 저장 시 사용)
 */
class PdfDocument(
    private val inputStreamProvider: () -> InputStream,
    override val title: String
) : DocumentSource {

    override suspend fun extractText(): String {
        val raw = inputStreamProvider().use { PdfTextExtractor.extractText(it) }
        return TextCleaner.cleanDocument(raw)
    }
}

/**
 * 텍스트 문서 소스.
 * 이미 추출된 텍스트를 TextCleaner로 정제만 한다.
 */
class TextDocument(
    private val text: String,
    override val title: String
) : DocumentSource {

    override suspend fun extractText(): String {
        return TextCleaner.cleanDocument(text)
    }
}
