package io.graphai.ecovector

import java.text.Normalizer

/**
 * PDF 추출 텍스트 전처리 유틸리티
 *
 * 벡터 검색 정확도를 높이기 위해 노이즈를 제거하고 텍스트를 정규화한다.
 * 문서와 쿼리 양쪽에 동일하게 적용하여 임베딩 매칭 일관성을 유지한다.
 */
object TextCleaner {

    // 장식/불릿 기호 (의미 없이 벡터를 왜곡하는 문자들)
    private val DECORATIVE_SYMBOLS = Regex("[●○◎◇◆□■△▲▽▼☆★♣♠♥♦►▶◀◄→←↑↓↔⇒⇐⇑⇓※☎✔✖✕✓✗✘⊙⊚⊛⊜⊝]")

    // 제어문자 (\t, \n, \r 제외)
    private val CONTROL_CHARS = Regex("[\\x00-\\x08\\x0B\\x0C\\x0E-\\x1F\\x7F]")

    // 페이지 번호 패턴: "- 1 -", "- 12 -", "1 / 10", "[ 3 ]", "(5)" 등 (줄 단위)
    private val PAGE_NUMBER_LINE = Regex(
        """^[\s]*(?:-\s*\d+\s*-|\d+\s*/\s*\d+|\[\s*\d+\s*]|\(\s*\d+\s*\)|페이지\s*\d+|page\s*\d+)[\s]*$""",
        setOf(RegexOption.IGNORE_CASE, RegexOption.MULTILINE)
    )

    // 연속 마침표 (3개 이상)
    private val REPEATED_DOTS = Regex("\\.{3,}")

    // 연속 하이픈/대시 (3개 이상)
    private val REPEATED_DASHES = Regex("[-–—]{3,}")

    // 연속 밑줄 (3개 이상)
    private val REPEATED_UNDERSCORES = Regex("_{3,}")

    // 연속 공백 (2개 이상, 개행 제외)
    private val MULTIPLE_SPACES = Regex("""[^\S\n]{2,}""")

    // 연속 빈 줄 (3줄 이상 → 1줄)
    private val MULTIPLE_BLANK_LINES = Regex("\n{3,}")

    /**
     * 텍스트 전처리 (문서용)
     *
     * PDF에서 추출한 본문 텍스트에 모든 정제 단계를 적용한다.
     */
    fun cleanDocument(text: String): String {
        if (text.isBlank()) return ""

        var result = text

        // 1. Unicode NFC 정규화 (한글 조합형→완성형 통일)
        result = Normalizer.normalize(result, Normalizer.Form.NFC)

        // 2. 제어문자 제거
        result = CONTROL_CHARS.replace(result, "")

        // 3. 전각 ASCII → 반각 변환 (！→!, ０→0, Ａ→A 등)
        result = fullwidthToHalfwidth(result)

        // 4. 장식 기호 제거
        result = DECORATIVE_SYMBOLS.replace(result, "")

        // 5. 페이지 번호 줄 제거
        result = PAGE_NUMBER_LINE.replace(result, "")

        // 6. 반복 문장부호 축소
        result = REPEATED_DOTS.replace(result, ".")
        result = REPEATED_DASHES.replace(result, "-")
        result = REPEATED_UNDERSCORES.replace(result, "")

        // 7. 공백 정규화
        result = MULTIPLE_SPACES.replace(result, " ")
        result = MULTIPLE_BLANK_LINES.replace(result, "\n\n")

        // 8. 각 줄 앞뒤 공백 제거
        result = result.lines().joinToString("\n") { it.trim() }

        // 9. 앞뒤 공백/빈줄 제거
        result = result.trim()

        return result
    }

    /**
     * 텍스트 전처리 (쿼리용)
     *
     * 쿼리 텍스트에 경량 정제를 적용한다.
     * 문서와 동일한 정규화를 적용하되, 구조적 정제(헤더/푸터 등)는 생략한다.
     */
    fun cleanQuery(text: String): String {
        if (text.isBlank()) return ""

        var result = text

        result = Normalizer.normalize(result, Normalizer.Form.NFC)
        result = CONTROL_CHARS.replace(result, "")
        result = fullwidthToHalfwidth(result)
        result = DECORATIVE_SYMBOLS.replace(result, "")
        result = MULTIPLE_SPACES.replace(result, " ")

        result = result.trim()

        return result
    }

    /**
     * 전각 ASCII 문자를 반각으로 변환
     *
     * U+FF01(！) ~ U+FF5E(～) → U+0021(!) ~ U+007E(~)
     * U+3000(전각 공백) → U+0020(반각 공백)
     */
    private fun fullwidthToHalfwidth(text: String): String {
        val sb = StringBuilder(text.length)
        for (ch in text) {
            when {
                ch in '\uFF01'..'\uFF5E' -> sb.append((ch.code - 0xFEE0).toChar())
                ch == '\u3000' -> sb.append(' ')
                else -> sb.append(ch)
            }
        }
        return sb.toString()
    }
}
