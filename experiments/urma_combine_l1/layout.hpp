#pragma once
#include <cstdint>
#include "../megamoe/csrc/kernels/mega_moe/urma/abi.hpp"
namespace Demo {
constexpr uint32_t cores=28, cvRows=256, words=16;
struct Args {
    uint64_t source, sourceScale, dispatchDestination, destination, records, workspace, ffts;
    uint32_t rank, world, bs, hidden, topk, intermediate, sharedIntermediate, sharedExperts, computeEnable, communicationEnable;
    MegaMoeUrma::Arguments compute;
};
inline uint8_t Pattern(uint32_t src,uint32_t dst,uint64_t p) {
    return uint8_t(src*17+dst*31+p+(p>>8)*13+(p>>16)*7);
}
} // namespace Demo
