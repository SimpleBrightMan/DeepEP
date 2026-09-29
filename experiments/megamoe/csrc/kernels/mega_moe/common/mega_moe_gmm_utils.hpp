/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under the CANN Open Software License Agreement Version 2.0.
 */
#pragma once

#include <type_traits>
#include "kernel_operator.h"

namespace MegaMoeImpl::GmmKernel {
// Both Ascend FP4 encodings pack two logical elements into one byte.
template <typename T>
__aicore__ inline constexpr bool IsFp4()
{
    // CANN compiler aliases may be global; do not require AscendC aliases.
    using namespace AscendC;
    return std::is_same_v<T, fp4x2_e2m1_t> || std::is_same_v<T, fp4x2_e1m2_t>;
}

template <typename T>
__aicore__ inline constexpr T ScalarMin(T a, T b)
{
    return a < b ? a : b;
}
}  // namespace MegaMoeImpl::GmmKernel
