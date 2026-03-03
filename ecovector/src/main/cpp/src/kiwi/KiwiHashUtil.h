#pragma once
#include <string>
#include <vector>
#include <cstdint>

namespace ecovector {

/// Deterministic 32-bit hash for Kiwi morphemes (MurmurHash3)
inline int32_t hashMorpheme(const std::string& morpheme) {
    const uint8_t* data = reinterpret_cast<const uint8_t*>(morpheme.data());
    int len = static_cast<int>(morpheme.size());
    const uint32_t seed = 0x9E3779B9; // golden ratio

    uint32_t h = seed ^ len;
    const int nblocks = len / 4;
    const auto* blocks = reinterpret_cast<const uint32_t*>(data);

    for (int i = 0; i < nblocks; i++) {
        uint32_t k = blocks[i];
        k *= 0xcc9e2d51;
        k = (k << 15) | (k >> 17);
        k *= 0x1b873593;
        h ^= k;
        h = (h << 13) | (h >> 19);
        h = h * 5 + 0xe6546b64;
    }

    const uint8_t* tail = data + nblocks * 4;
    uint32_t k1 = 0;
    switch (len & 3) {
        case 3: k1 ^= tail[2] << 16; [[fallthrough]];
        case 2: k1 ^= tail[1] << 8; [[fallthrough]];
        case 1: k1 ^= tail[0];
                k1 *= 0xcc9e2d51;
                k1 = (k1 << 15) | (k1 >> 17);
                k1 *= 0x1b873593;
                h ^= k1;
    }

    h ^= len;
    h ^= h >> 16;
    h *= 0x85ebca6b;
    h ^= h >> 13;
    h *= 0xc2b2ae35;
    h ^= h >> 16;

    return static_cast<int32_t>(h);
}

/// Hash a vector of morpheme strings to int32 array
inline std::vector<int32_t> hashMorphemes(const std::vector<std::string>& morphemes) {
    std::vector<int32_t> result;
    result.reserve(morphemes.size());
    for (const auto& m : morphemes) {
        result.push_back(hashMorpheme(m));
    }
    return result;
}

} // namespace ecovector
