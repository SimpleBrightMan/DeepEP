// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once

// Host-only fixed Pack rounds; no dynamic task claiming is needed on device.
#include "abi.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace MegaMoeUrma {
inline std::vector<PackRound> BuildPackRounds(const std::vector<Chunk>& sendChunks, uint32_t maxPeerChunks)
{
    uint32_t count = 0U;
    for (const auto& chunk : sendChunks) {
        if (chunk.peer >= kWorld || chunk.cv >= maxPeerChunks || chunk.rows == 0U) {
            throw std::invalid_argument("Invalid URMA Pack chunk");
        }
        count = std::max(count, chunk.cv + 1U);
    }
    std::vector<PackRound> rounds(count);
    for (uint32_t round = 0U; round < count; ++round) {
        for (uint32_t peer = 0U; peer < kWorld; ++peer) {
            rounds[round].chunks[peer].peer = peer;
            rounds[round].chunks[peer].cv = round;
        }
    }
    for (const auto& chunk : sendChunks) {
        auto& round = rounds[chunk.cv];
        if (round.chunks[chunk.peer].rows != 0U) {
            throw std::invalid_argument("Duplicate URMA Pack peer/CV");
        }
        round.chunks[chunk.peer] = chunk;
        round.rows += chunk.rows;
    }
    return rounds;
}
}  // namespace MegaMoeUrma
