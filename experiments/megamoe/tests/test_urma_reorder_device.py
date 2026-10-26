"""Execute production ReorderGmm2Tile using asynchronous CPU DMA queues.

This checks address/stride/event/completion contracts, not NPU timing or
hardware arithmetic. The existing Reduce fake supplies tensors and events.
"""

import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest

from test_urma_reduce_pipeline import FAKE_ASCEND


def fake_ascend():
    fake = (
        FAKE_ASCEND.replace("S_MTE2, MTE2_S };", "S_MTE2, MTE2_S, MTE3_MTE2, MTE2_MTE3, MTE3_S };")
        .replace("events[6][8]", "events[9][8]")
        .replace("explicit bfloat16_t(float value)", "bfloat16_t(float value)")
    )
    fake = fake.replace(
        "case HardEvent::MTE2_S: return {Mte2, Scalar};",
        """case HardEvent::MTE2_S: return {Mte2, Scalar};
    case HardEvent::MTE3_MTE2: return {Mte3, Mte2};
    case HardEvent::MTE2_MTE3: return {Mte2, Mte3};
    case HardEvent::MTE3_S: return {Mte3, Scalar};""",
    )
    # A fast store engine can retire several commands while MTE2 is stalled;
    # this exposes unconsumed cross-pipe credits across successive tile calls.
    fake = fake.replace(
        "if (!queue.empty() && queue.front()()) { queue.pop_front(); progress = true; }",
        """unsigned budget = priority == 2U && 1U + (n + priority) % 3U == Mte3 ? 10000U : 1U;
        while (budget-- && !queue.empty() && queue.front()()) { queue.pop_front(); progress = true; }""",
    )
    return (
        fake
        + r"""
namespace AscendC {
enum class CacheLine { SINGLE_CACHE_LINE };
enum class DcciDst { CACHELINE_OUT };
template <typename T, CacheLine, DcciDst> void DataCacheCleanAndInvalid(GlobalTensor<T>) {}
struct DataCopyExtParams { uint16_t blockCount; uint32_t blockLen, srcStride, dstStride, reserved; };
template <typename T> struct DataCopyPadExtParams { bool isPad; uint8_t left, right; T padding; };
inline uint32_t submittedStores, finishedStores, atomicCalls, tileReads;
inline std::function<void(uintptr_t, uint16_t)> verifyStore;
inline std::function<void()> verifyAtomic;
inline std::array<bool, 2> initialized;
inline std::array<uint32_t, 2> slotSubmissions, slotCompletions;
template <typename T>
void DataCopyPad(LocalTensor<T> destination, GlobalTensor<T> source,
                 DataCopyExtParams p, DataCopyPadExtParams<T> padding) {
    assert(!padding.isPad && p.dstStride == 0U && p.blockLen % 32U == 0U);
    const uint32_t slot = destination.address / 65536U;
    assert(slot < 2U);
    ++tileReads;
    const uint32_t previousStores = slotSubmissions[slot];
    Enqueue(Mte2, [=] {
        assert(slotCompletions[slot] == previousStores && "UB overwritten while stores still read it");
        for (uint32_t row = 0; row < p.blockCount; ++row) {
            const auto* input = reinterpret_cast<const unsigned char*>(source.data) + row * (p.blockLen + p.srcStride);
            auto* output = reinterpret_cast<unsigned char*>(destination.Data()) + row * p.blockLen;
            assert(destination.address + (row + 1U) * p.blockLen <= (slot + 1U) * 65536U);
            std::memcpy(output, input, p.blockLen);
        }
        initialized[slot] = true;
        return true;
    });
}
template <typename T>
void DataCopyPad(GlobalTensor<T> destination, LocalTensor<T> source, DataCopyExtParams p) {
    assert(p.srcStride == 0U && p.blockLen % 32U == 0U);
    const uint32_t slot = source.address / 65536U;
    assert(slot < 2U);
    ++submittedStores;
    ++slotSubmissions[slot];
    Enqueue(Mte3, [=] {
        assert(initialized[slot] && "MTE3 consumed a slot before MTE2 filled it");
        for (uint32_t row = 0; row < p.blockCount; ++row) {
            auto* output = reinterpret_cast<unsigned char*>(destination.data) + row * (p.blockLen + p.dstStride);
            const auto* input = reinterpret_cast<const unsigned char*>(source.Data()) + row * p.blockLen;
            assert(source.address + (row + 1U) * p.blockLen <= (slot + 1U) * 65536U);
            for (uint32_t col = 0; col < p.blockLen / sizeof(T); ++col) {
                const auto value = reinterpret_cast<const T*>(input)[col];
                verifyStore(reinterpret_cast<uintptr_t>(output) + col * sizeof(T), value.bits);
            }
            std::memcpy(output, input, p.blockLen);
        }
        ++slotCompletions[slot];
        ++finishedStores;
        return true;
    });
}
inline void AtomicAdd(int32_t* address, int32_t value) {
    assert(value == 1 && finishedStores == submittedStores);
    verifyAtomic();
    *address += value;
    ++atomicCalls;
}
} // namespace AscendC
"""
    )


DRIVER = r"""
#include "kernel_operator.h"
#include "reorder.hpp"
#include <iostream>
#include <string>
#include <unordered_map>
using namespace AscendC;
using namespace MegaMoeUrma;
int main(int argc, char** argv) {
    assert(argc == 4);
    const uint32_t rows = std::stoul(argv[1]);
    const uint32_t world = std::stoul(argv[2]);
    priority = std::stoul(argv[3]);
    constexpr uint32_t h = 3584U, ntiles = 14U, prefix = 7U, fragmentIndex = 3U;
    const uint16_t sentinel = 0x7fc1;
    std::vector<bfloat16_t> input((prefix + rows + 9U) * h);
    std::vector<bfloat16_t> remote((rows + 300U) * h), self((rows + 500U) * h);
    for (auto& item : remote) item.bits = sentinel;
    for (auto& item : self) item.bits = sentinel;
    for (uint32_t row = 0; row < input.size() / h; ++row)
        for (uint32_t col = 0; col < h; ++col)
            input[row * h + col].bits = uint16_t(1U + (row * 7919U + col * 997U) % 30000U);
    const uint32_t rank = world - 1U;
    std::vector<RowSegment> segments(2); // Test nonzero segmentBegin too.
    std::vector<uint32_t> sourceRow(rows), destinationRow(rows);
    std::vector<bool> local(rows);
    uint32_t segmentId = 0;
    for (uint32_t row = 0; row < rows;) {
        const uint32_t width = std::min(rows - row, segmentId % 3U == 0U ? 43U : 17U);
        const uint32_t peer = segmentId % 3U == 0U ? rank : segmentId % (world - 1U);
        segments.push_back({peer, segmentId, 19U + row, 201U + row, row, width, segmentId, 0U});
        for (uint32_t r = row; r < row + width; ++r) {
            sourceRow[r] = prefix + r;
            destinationRow[r] = (peer == rank ? 201U : 19U) + r;
            local[r] = peer == rank;
        }
        row += width;
        ++segmentId;
    }
    Fragment fragment{5U, 2U, prefix, rows, segmentId, 0U, 0U};
    std::array<int32_t, 6U * 32U> done{};
    Arguments args{};
    args.config.rank = rank; args.config.world = world; args.config.hidden = h;
    args.config.returnYOffset = 128U;
    args.symmetric = reinterpret_cast<uint64_t>(self.data()) - args.config.returnYOffset;
    args.gmm2Contiguous = reinterpret_cast<uint64_t>(input.data());
    args.gmm2Output = reinterpret_cast<uint64_t>(remote.data());
    args.gmm2Done = reinterpret_cast<uint64_t>(done.data());
    args.segments = reinterpret_cast<uint64_t>(segments.data());
    std::unordered_map<uintptr_t, uint16_t> expected;
    std::unordered_map<uintptr_t, uint32_t> seen;
    for (uint32_t row = 0; row < rows; ++row)
        for (uint32_t col = 0; col < h; ++col) {
            auto& plane = local[row] ? self : remote;
            const uintptr_t address = reinterpret_cast<uintptr_t>(&plane[destinationRow[row] * h + col]);
            assert(expected.emplace(address, input[sourceRow[row] * h + col].bits).second);
        }
    uint32_t currentTile = 0U;
    verifyStore = [&](uintptr_t address, uint16_t value) {
        assert(expected.count(address) && expected.at(address) == value);
        assert(++seen[address] == 1U);
    };
    verifyAtomic = [&] {
        assert(done[fragmentIndex * 32U] == int32_t(currentTile));
        for (uint32_t row = 0; row < rows; ++row)
            for (uint32_t col = currentTile * 256U; col < (currentTile + 1U) * 256U; ++col) {
                auto& plane = local[row] ? self : remote;
                const uintptr_t address = reinterpret_cast<uintptr_t>(&plane[destinationRow[row] * h + col]);
                assert(seen[address] == 1U);
                assert(plane[destinationRow[row] * h + col].bits == expected.at(address));
            }
    };
    Timer timer;
    timer.Init(0U);
    for (; currentTile < ntiles; ++currentTile) {
        ReorderGmm2Tile(args, fragment, fragmentIndex, currentTile, timer);
        assert(done[fragmentIndex * 32U] == int32_t(currentTile + 1U));
        // Intentionally do not drain before the next call: production may
        // schedule consecutive reorder tiles with no activation in between.
    }
    Drain();
    assert(atomicCalls == ntiles && tileReads == ntiles * ((rows + 127U) / 128U));
    for (const auto& queue : queues) assert(queue.empty());
    for (const auto& kind : events) for (bool flag : kind) assert(!flag);
    assert(seen.size() == uint64_t(rows) * h);
    for (const auto* plane : {&remote, &self})
        for (uint64_t index = 0; index < plane->size(); ++index) {
            const uintptr_t address = reinterpret_cast<uintptr_t>(&(*plane)[index]);
            assert((*plane)[index].bits == (expected.count(address) ? expected.at(address) : sentinel));
        }
    for (uint32_t index = 0; index < done.size(); ++index)
        assert(done[index] == (index == fragmentIndex * 32U ? int32_t(ntiles) : 0));
}
"""


@pytest.fixture(scope="module")
def reorder_device_binary(tmp_path_factory):
    command = shlex.split(os.environ.get("CXX", "c++"))
    if not command or not shutil.which(command[0]):
        pytest.skip("A host C++ compiler is required for production reorder checks")
    temporary = tmp_path_factory.mktemp("urma_reorder_device")
    (temporary / "kernel_operator.h").write_text(fake_ascend(), encoding="utf-8")
    source = temporary / "reorder.cpp"
    source.write_text(DRIVER, encoding="utf-8")
    binary = temporary / ("reorder.exe" if os.name == "nt" else "reorder")
    root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [
            *command,
            "-std=c++17",
            "-O1",
            "-UNDEBUG",
            "-I" + str(temporary),
            "-I" + str(root / "experiments/megamoe/csrc/kernels/mega_moe/urma"),
            str(source),
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return binary


@pytest.mark.parametrize("rows", [1, 43, 128, 170, 256])
@pytest.mark.parametrize("world", [4, 8, 16])
@pytest.mark.parametrize("engine_priority", [0, 1, 2])
def test_production_reorder_copies_all_columns_and_publishes_after_stores(
    reorder_device_binary, rows, world, engine_priority
):
    result = subprocess.run(
        [str(reorder_device_binary), str(rows), str(world), str(engine_priority)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
