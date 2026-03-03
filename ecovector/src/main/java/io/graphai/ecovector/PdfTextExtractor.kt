package io.graphai.ecovector

import android.content.Context
import com.tom_roush.pdfbox.android.PDFBoxResourceLoader
import com.tom_roush.pdfbox.pdmodel.PDDocument
import com.tom_roush.pdfbox.text.PDFTextStripper
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.InputStream

/**
 * PDF 텍스트 추출 유틸리티
 *
 * PdfBox-Android를 사용하여 PDF에서 텍스트를 추출한다.
 * 페이지별 병렬 처리로 속도를 개선한다.
 */
object PdfTextExtractor {
    private var initialized = false

    /**
     * PdfBox 리소스 초기화 (앱 시작 시 1회 호출)
     */
    fun init(context: Context) {
        if (!initialized) {
            PDFBoxResourceLoader.init(context.applicationContext)
            initialized = true
        }
    }

    /**
     * PDF InputStream에서 텍스트 추출
     *
     * @param inputStream PDF 파일 스트림
     * @return 추출된 텍스트
     */
    suspend fun extractText(inputStream: InputStream): String = withContext(Dispatchers.IO) {
        val document = PDDocument.load(inputStream)
        try {
            if (document.numberOfPages == 1) {
                // 단일 페이지: 직접 추출
                PDFTextStripper().getText(document)
            } else {
                // 다중 페이지: 순차 처리 (메모리 절약)
                extractTextSequential(document)
            }
        } finally {
            document.close()
        }
    }

    private fun extractTextSequential(document: PDDocument): String {
        val sb = StringBuilder()
        for (pageNum in 1..document.numberOfPages) {
            val stripper = PDFTextStripper().apply {
                startPage = pageNum
                endPage = pageNum
            }
            if (sb.isNotEmpty()) sb.append('\n')
            sb.append(stripper.getText(document))
        }
        return sb.toString()
    }
}
