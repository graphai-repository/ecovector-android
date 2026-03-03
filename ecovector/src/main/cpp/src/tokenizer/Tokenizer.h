#pragma once
#include <string>
#include <vector>
#include <memory>
#include <tokenizers_cpp.h>

namespace ecovector {

class Tokenizer {
public:
    Tokenizer() = default;
    ~Tokenizer();

    bool load(const std::string& tokenizer_json_path);
    std::vector<int32_t> encode(const std::string& text);
    std::string decode(const std::vector<int32_t>& ids);

private:
    std::unique_ptr<tokenizers::Tokenizer> tokenizer_;
};

} // namespace ecovector
