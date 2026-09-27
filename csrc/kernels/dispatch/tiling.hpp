#pragma once
#include <cstdint>
#include "../notify/tiling.hpp"

namespace ascend_deepep {
struct DispatchTiling {
    uint32_t rank, world, tokens, topk, rows, address_stride, fp8;
    uint64_t sync, book, input, output, weights, scales, inbox, inbox_stride, workspace_bytes;
};

// Caller validates bounds before narrowing to this device ABI. All byte
// calculations stay uint64_t, including receive capacity and scale inboxes.
inline DispatchTiling MakeDispatchTiling(uint32_t rank, uint32_t world, uint32_t tokens, uint32_t topk,
                                         uint32_t experts, uint32_t rows, bool fp8)
{
    DispatchTiling t{};
    t.rank = rank;
    t.world = world;
    t.tokens = tokens;
    t.topk = topk;
    t.rows = rows;
    t.fp8 = fp8;
    t.address_stride = 1;
    while (t.address_stride < rows) {
        t.address_stride <<= 1;
    }
    uint64_t cursor = MakeNotifyTiling(rank, world, tokens, topk, experts, 256, 7168, 32).workspace_bytes;
    auto reserve = [&cursor](uint64_t bytes) {
        const auto offset = cursor;
        cursor += (bytes + 511) / 512 * 512;
        return offset;
    };
    const uint64_t row_bytes = fp8 ? 7168 : 14336;
    t.sync = reserve(uint64_t(world) * 49152 + 65536);
    t.book = reserve(uint64_t(world) * 16);
    t.input = reserve(uint64_t(tokens) * row_bytes);
    t.output = reserve(uint64_t(rows) * row_bytes);
    t.weights = reserve(uint64_t(rows) * 4);
    t.scales = reserve(fp8 ? uint64_t(rows) * 224 : 0);
    t.inbox_stride = fp8 ? (512 + uint64_t(tokens) * topk * 256 + 511) / 512 * 512 : 0;
    t.inbox = reserve(uint64_t(world) * t.inbox_stride);
    t.workspace_bytes = cursor;
    return t;
}
}  // namespace ascend_deepep
