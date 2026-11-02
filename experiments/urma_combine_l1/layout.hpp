#pragma once
#include <cstdint>
namespace Demo {
constexpr uint32_t cores=28, cvRows=256, words=16, copyBytes=65536, slots=4;
struct Args {
    uint64_t source, destination, flags, l1Source, records, workspace, ffts;
    uint64_t l1Bytes;
    uint32_t rank, world, bs, hidden, topk, l1Enable, l1Repeat, gapCycles;
};
inline uint64_t L1Bytes(uint32_t bs,uint32_t h,uint32_t i,uint32_t k) {
    // A is FP8 in both GMMs. Each N256 projection reads its input again.
    return uint64_t(bs)*k*(uint64_t(h)*((2*i+255)/256)+uint64_t(i)*((h+255)/256));
}
inline uint8_t Pattern(uint32_t src,uint32_t dst,uint64_t p) {
    return uint8_t(src*17+dst*31+p+(p>>8)*13+(p>>16)*7);
}
} // namespace Demo
