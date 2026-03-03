#include "KiwiTokenizer.h"
#include <kiwi/capi.h>
#include <android/log.h>
#include <algorithm>
#include <unordered_map>
#include <fstream>
#define LOG_TAG "KiwiTokenizer"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace ecovector {

namespace {
    // 영어 대문자를 소문자로 변환 (한글/기타 문자는 그대로 유지)
    std::string toLowerCase(const std::string& str) {
        std::string result;
        result.reserve(str.size());
        for (unsigned char c : str) {
            if (c >= 'A' && c <= 'Z') {
                result.push_back(c + 32);  // A-Z -> a-z
            } else {
                result.push_back(c);
            }
        }
        return result;
    }

}

struct KiwiTokenizer::Impl {
    kiwi_h kiwi = nullptr;
    std::string modelDirPath;
    int numThreads = 2;
    std::unordered_map<std::string, std::vector<std::string>> synonymsMap;

    ~Impl() {
        if (kiwi) {
            kiwi_close(kiwi);
            kiwi = nullptr;
        }
    }

    // 이형태의 원형(lemma)을 반환. 이형태가 아니면 표면형 그대로 반환.
    std::string getLemma(kiwi_res_h res, int index, int num) const {
        const char* surfaceForm = kiwi_res_form(res, index, num);
        if (!surfaceForm || surfaceForm[0] == '\0') return {};

        int morphId = kiwi_res_morpheme_id(res, index, num, kiwi);
        if (morphId < 0) return surfaceForm;

        kiwi_morpheme_t info = kiwi_get_morpheme_info(kiwi, (unsigned int)morphId);
        if (info.orig_morpheme_id != 0 && info.orig_morpheme_id != (uint32_t)morphId) {
            const char* origForm = kiwi_get_morpheme_form(kiwi, info.orig_morpheme_id);
            if (origForm && origForm[0] != '\0') {
                std::string result(origForm);
                kiwi_free_morpheme_form(origForm);
                return result;
            }
        }
        return surfaceForm;
    }

    // BM25에 유용한 내용어 POS 태그인지 확인
    // Kiwi Types.h POSTag enum 기반
    static bool isContentPOS(const char* tag) {
        if (!tag || tag[0] == '\0') return false;
        // NNG(일반명사), NNP(고유명사), NNB(의존명사)
        if (tag[0] == 'N' && tag[1] == 'N') return true;
        // VV(동사), VA(형용사)
        if (tag[0] == 'V' && (tag[1] == 'V' || tag[1] == 'A')) return true;
        // MAG(일반부사)
        if (tag[0] == 'M' && tag[1] == 'A' && tag[2] == 'G') return true;
        // XR(어근)
        if (tag[0] == 'X' && tag[1] == 'R') return true;
        // SL(외국어), SH(한자), SN(숫자)
        if (tag[0] == 'S' && (tag[1] == 'L' || tag[1] == 'H' || tag[1] == 'N')) return true;
        return false;
    }
};

KiwiTokenizer::KiwiTokenizer() : pImpl_(std::make_unique<Impl>()) {}
KiwiTokenizer::~KiwiTokenizer() = default;

bool KiwiTokenizer::reload() {
    if (pImpl_->modelDirPath.empty()) {
        LOGE("Cannot reload: model path not set (call load() first)");
        return false;
    }
    LOGI("Reloading Kiwi to reclaim memory...");
    return load(pImpl_->modelDirPath, pImpl_->numThreads);
}

bool KiwiTokenizer::load(const std::string& modelDirPath, int numThreads) {
    LOGI("Loading Kiwi model from: %s", modelDirPath.c_str());
    pImpl_->modelDirPath = modelDirPath;
    pImpl_->numThreads = numThreads;

    if (pImpl_->kiwi) {
        kiwi_close(pImpl_->kiwi);
        pImpl_->kiwi = nullptr;
    }

    int options = KIWI_BUILD_INTEGRATE_ALLOMORPH | KIWI_BUILD_LOAD_DEFAULT_DICT;
    kiwi_builder_h builder = kiwi_builder_init(modelDirPath.c_str(), numThreads, options, 0);

    if (!builder) {
        const char* err = kiwi_error();
        LOGE("Kiwi builder init failed: %s", err ? err : "unknown");
        return false;
    }

    // Load user dictionary if exists
    std::string userDictPath = modelDirPath + "/user_dict.tsv";
    int dictResult = kiwi_builder_load_dict(builder, userDictPath.c_str());
    if (dictResult > 0) {
        LOGI("User dictionary loaded: %d entries from %s", dictResult, userDictPath.c_str());
    } else if (dictResult == 0) {
        LOGI("User dictionary found but 0 entries loaded: %s", userDictPath.c_str());
    } else {
        const char* err = kiwi_error();
        LOGI("No user dictionary at %s (result=%d, err=%s)", userDictPath.c_str(), dictResult, err ? err : "none");
    }

    // Load synonyms dictionary (en -> ko mapping)
    std::string synonymsPath = modelDirPath + "/synonyms.tsv";
    std::ifstream synonymsFile(synonymsPath);
    if (synonymsFile.is_open()) {
        std::string line;
        int count = 0;
        while (std::getline(synonymsFile, line)) {
            if (line.empty() || line[0] == '#') continue;
            auto tab = line.find('\t');
            if (tab == std::string::npos) continue;
            std::string eng = line.substr(0, tab);
            std::string kor = line.substr(tab + 1);
            // trim trailing whitespace/CR
            while (!kor.empty() && (kor.back() == '\r' || kor.back() == ' ')) {
                kor.pop_back();
            }
            if (!eng.empty() && !kor.empty()) {
                pImpl_->synonymsMap[eng].push_back(kor);
                count++;
            }
        }
        LOGI("Synonyms dictionary loaded: %d entries from %s", count, synonymsPath.c_str());
    } else {
        LOGI("No synonyms dictionary at %s, synonym expansion disabled", synonymsPath.c_str());
    }

    pImpl_->kiwi = kiwi_builder_build(builder, nullptr, 0);
    kiwi_builder_close(builder);

    if (!pImpl_->kiwi) {
        const char* err = kiwi_error();
        LOGE("Kiwi build failed: %s", err ? err : "unknown");
        return false;
    }

    // 이형태 통합 활성화: kiwi_res_form()이 표면형 대신 원형을 반환하도록
    kiwi_config_t config = kiwi_get_global_config(pImpl_->kiwi);
    config.integrate_allomorph = 1;
    kiwi_set_global_config(pImpl_->kiwi, config);

    LOGI("Kiwi loaded successfully (integrate_allomorph=1)");
    return true;
}

std::vector<std::string> KiwiTokenizer::tokenize(const std::string& text) const {
    std::vector<std::string> morphemes;
    if (!pImpl_->kiwi || text.empty()) return morphemes;

    kiwi_analyze_option_t option = {};
    option.match_options = KIWI_MATCH_ALL;
    option.blocklist = nullptr;
    option.open_ending = 0;
    option.allowed_dialects = 0;
    option.dialect_cost = 3.0f;

    kiwi_res_h res = kiwi_analyze(pImpl_->kiwi, text.c_str(), 1, option, nullptr);
    if (!res) return morphemes;

    int wordNum = kiwi_res_word_num(res, 0);
    morphemes.reserve(wordNum);

    for (int i = 0; i < wordNum; i++) {
        const char* tag = kiwi_res_tag(res, 0, i);

        if (Impl::isContentPOS(tag)) {
            // 이형태 정규화: 표면형 대신 원형(lemma) 사용
            std::string lemma = pImpl_->getLemma(res, 0, i);
            if (!lemma.empty()) {
                std::string token = toLowerCase(lemma);
                morphemes.emplace_back(token);
                // 동의어 확장: 토큰에 동의어 추가 (multi-value 지원)
                auto it = pImpl_->synonymsMap.find(token);
                if (it != pImpl_->synonymsMap.end()) {
                    for (const auto& syn : it->second) {
                        morphemes.emplace_back(syn);
                    }
                }
            }
        }
    }

    kiwi_res_close(res);
    return morphemes;
}

std::vector<std::string> KiwiTokenizer::tokenizeForIndexing(const std::string& text) const {
    std::vector<std::string> morphemes;
    if (!pImpl_->kiwi || text.empty()) return morphemes;

    kiwi_analyze_option_t option = {};
    option.match_options = KIWI_MATCH_ALL;
    option.blocklist = nullptr;
    option.open_ending = 0;
    option.allowed_dialects = 0;
    option.dialect_cost = 3.0f;

    kiwi_res_h res = kiwi_analyze(pImpl_->kiwi, text.c_str(), 1, option, nullptr);
    if (!res) return morphemes;

    int wordNum = kiwi_res_word_num(res, 0);
    morphemes.reserve(wordNum * 2);  // 영어 토큰 시 한글 동의어도 추가될 수 있으므로 여유 공간

    for (int i = 0; i < wordNum; i++) {
        const char* tag = kiwi_res_tag(res, 0, i);

        if (Impl::isContentPOS(tag)) {
            // 이형태 정규화: 표면형 대신 원형(lemma) 사용
            std::string lemma = pImpl_->getLemma(res, 0, i);
            if (!lemma.empty()) {
                std::string token = toLowerCase(lemma);
                morphemes.emplace_back(token);

                // 동의어 확장: 토큰에 동의어 추가 (multi-value 지원)
                auto it = pImpl_->synonymsMap.find(token);
                if (it != pImpl_->synonymsMap.end()) {
                    for (const auto& syn : it->second) {
                        morphemes.emplace_back(syn);
                    }
                }
            }
        }
    }

    kiwi_res_close(res);
    return morphemes;
}

bool KiwiTokenizer::isLoaded() const {
    return pImpl_->kiwi != nullptr;
}

} // namespace ecovector
