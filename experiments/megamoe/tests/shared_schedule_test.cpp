// Copyright (c) 2026, Lu Lu
// Modified by SimpleBrightMan 2026
#include <cassert>
#include <cstdint>
#include <vector>
#include "kernels/mega_moe_urma/common/mega_moe_shared_schedule.hpp"

using namespace MegaMoeUrma;

int main()
{
    static_assert(!UseSplitSharedGmm(4095) && UseSplitSharedGmm(4096));
    static_assert(SharedGroupCount(12288) == 48 && SharedHeadGroupCount(12288) == 24);
    static_assert(SharedGroupCount(12289) == 49 && SharedHeadGroupCount(12289) == 25);
    for (uint32_t bs : {1U, 127U, 128U, 129U, 255U, 256U, 257U, 1024U, 2048U, 4095U,
                        4096U, 4097U, 8192U, 12288U, 12289U, 16383U, 16384U}) {
        const auto groups = SharedGroupCount(bs), split = SharedHeadGroupCount(bs);
        assert(split + (groups - split) == groups && split >= groups - split);
        for (uint32_t width : {256U, 1280U, 2304U, 16384U}) {
            const uint32_t cols = width / 256U, total = SharedTaskCount(bs, width);
            std::vector<uint32_t> visits(total), ready(groups), producers(groups * cols);
            // Model either phase under arbitrary per-core completion order.
            // Splitting must preserve original global tile ownership and IDs.
            for (uint32_t phase = 0; phase < 2; ++phase) {
                const uint32_t begin = phase ? split : 0, end = phase ? groups : split;
                for (int core = 27; core >= 0; --core) {
                    for (uint32_t g = begin; g < end; ++g)
                        for (uint32_t col = 0; col < cols; ++col)
                            if ((g * cols + col) % 28 == uint32_t(27 - core)) ++producers[g * cols + col];
                }
            }
            for (auto n : producers) assert(n == 1);
            for (uint32_t worker = 0; worker < 28; ++worker) {
                const uint32_t owned = SharedOwnedTasks(total, worker);
                const uint32_t head = SharedOwnedTasks(split * cols * 2U, worker);
                assert(head <= owned && owned <= 293U);
                // First pass is allowed to stop early for routed prologue;
                // second pass resumes with original IDs, exactly once per task.
                for (uint32_t phase = 0; phase < 2; ++phase) {
                    const uint32_t begin = phase ? head / 2U : 0, end = phase ? owned : head / 2U;
                    for (uint32_t j = begin; j < end; ++j) {
                        const auto id = worker + j * 28U;
                        const auto task = SharedTaskAt(id, width);
                        assert(id < total && task.group < groups && task.tile < cols && task.slice < 2);
                        if (!phase) assert(task.group < split);
                        ++visits[id];
                        const uint32_t remaining = bs - task.group * 256U;
                        const uint32_t rows = remaining < 256U ? remaining : 256U;
                        if (task.slice * 128U < rows) ++ready[task.group];
                    }
                }
            }
            for (auto n : visits) assert(n == 1);
            for (uint32_t g = 0; g < groups; ++g) {
                const uint32_t remaining = bs - g * 256U;
                const uint32_t rows = remaining < 256U ? remaining : 256U;
                assert(ready[g] == cols * ((rows + 127U) / 128U));
            }
        }
    }
}
