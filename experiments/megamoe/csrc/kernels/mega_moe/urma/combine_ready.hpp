// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#pragma once

#include "abi.hpp"
#include "combine_ready_layout.hpp"
#include "kernel_operator.h"

namespace MegaMoeUrma {

// The dependency table remains source*maxPeerChunks+cv. Only its readiness
// encoding changes; positive observations are cached by Reduce as before.
__aicore__ inline bool CombineCvReady(const Arguments& args, uint32_t dependencyIndex)
{
    const uint32_t source = dependencyIndex / args.config.maxPeerChunks;
    const uint32_t cv = dependencyIndex % args.config.maxPeerChunks;
    auto* flag = reinterpret_cast<__gm__ uint64_t*>(
        args.symmetric + args.config.combineReadyOffset +
        CombineReadyWordIndex(args.config.maxPeerChunks, source, cv) * kFlagBytes);
    return CombineReadyBit(AscendC::ReadGmByPassDCache(flag), cv);
}

} // namespace MegaMoeUrma
