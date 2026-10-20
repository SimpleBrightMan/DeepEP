// Adapted from DeepSeek MegaMoE's routed linear-task lookahead schedule.
// Copyright (c) 2026 DeepSeek. MIT License:
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#pragma once
#include <cstdint>

#if defined(__NPU_DEVICE__)
#define MEGAMOE_LINEAR_INLINE __aicore__ inline
#else
#define MEGAMOE_LINEAR_INLINE inline
#endif

namespace MegaMoeImpl {
struct SmallLinearTask {
    bool gmm1;
    uint32_t index;
};

// 两层分别保持单调task游标；按AIC数量批量分配，并保留一批GMM1 lookahead。
// 它只描述顺序，不表示输入/FIX已完成；调用者仍必须执行原生数据依赖。
struct SmallLinearTaskSchedule {
    uint32_t blocks;
    uint32_t gmm1N;
    uint32_t gmm2N;
    uint32_t cores;
    uint32_t gmm1Base = 0U;
    uint32_t gmm2Base = 0U;

    MEGAMOE_LINEAR_INLINE bool Next(uint32_t core, SmallLinearTask& task)
    {
        if (cores == 0U || gmm1N == 0U || gmm2N == 0U || core >= cores) return false;
        const uint32_t total1 = blocks * gmm1N, total2 = blocks * gmm2N;
        while (gmm1Base < total1 || gmm2Base < total2) {
            bool first = gmm2Base >= total2;
            if (!first && gmm1Base < total1) {
                const uint32_t end = gmm2Base + cores < total2 ? gmm2Base + cores : total2;
                const uint32_t required = (end + gmm2N - 1U) / gmm2N * gmm1N;
                first = gmm1Base < required + cores;
            }
            auto& base = first ? gmm1Base : gmm2Base;
            const uint32_t index = base + core;
            base += cores;
            if (index < (first ? total1 : total2)) {
                task = {first, index};
                return true;
            }
        }
        return false;
    }
};
} // namespace MegaMoeImpl
#undef MEGAMOE_LINEAR_INLINE
