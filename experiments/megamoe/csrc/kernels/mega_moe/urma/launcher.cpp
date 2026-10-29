// Copyright (c) 2026, Lu Lu
// Modified by SimpleBright_Man 2026
#if __has_include("version/asc_devkit_version.h")
#include "version/asc_devkit_version.h"
#endif
#include "basic_api/kernel_basic_intf.h"
#include "abi.hpp"
#include "compute_contract.hpp"
#if defined(__NPU_DEVICE__)
#include "kernel.hpp"
#endif

using namespace AscendC;

__schedmode__(1) __global__ __mix__(1, 2) void mega_moe_urma_kernel(MegaMoeUrma::Arguments args)
{
#if defined(__NPU_DEVICE__)
    InitSocState();
    util_set_ffts_config(args.ffts);
    SetSysWorkspaceForce(reinterpret_cast<GM_ADDR>(args.systemWorkspace));
    MegaMoeUrma::Process<false>(args);
#endif
}

// Separate entry isolates M128 register/stack allocation from the M256 kernel.
__schedmode__(1) __global__ __mix__(1, 2) void mega_moe_urma_m128_kernel(MegaMoeUrma::Arguments args)
{
#if defined(__NPU_DEVICE__)
    InitSocState();
    util_set_ffts_config(args.ffts);
    SetSysWorkspaceForce(reinterpret_cast<GM_ADDR>(args.systemWorkspace));
    MegaMoeUrma::Process<true>(args);
#endif
}

extern "C" __attribute__((visibility("default"))) void mega_moe_urma_kernel_do(uint32_t block_dim, uint32_t ub_bytes,
                                                                               void* stream,
                                                                               const MegaMoeUrma::Arguments* arguments)
{
    if (MegaMoeUrma::UseFusedGateUp(arguments->config)) {
        mega_moe_urma_m128_kernel<<<block_dim, ub_bytes, stream>>>(*arguments);
    } else {
        mega_moe_urma_kernel<<<block_dim, ub_bytes, stream>>>(*arguments);
    }
}
