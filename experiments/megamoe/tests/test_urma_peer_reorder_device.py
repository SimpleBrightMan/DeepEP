"""Execute the peer/CV full-row reorder with asynchronous host DMA queues."""

import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest

from test_urma_reorder_device import fake_ascend


def peer_reorder_fake():
    # Reuse the existing FIFO/event/slot ownership model, adapting its value
    # observer to byte copies. Production copies every BF16 row as raw bytes.
    return (
        fake_ascend()
        .replace("constexpr TEventID EVENT_ID0 = 0;", "constexpr TEventID EVENT_ID0 = 0, EVENT_ID1 = 1;")
        .replace(
            "inline uint32_t submittedStores,",
            """template <typename T> uint16_t RawBits(T value) { return value.bits; }
template <> inline uint16_t RawBits<uint8_t>(uint8_t value) { return value; }
inline uint32_t submittedStores,""",
        )
        .replace("col * sizeof(T), value.bits);", "col * sizeof(T), RawBits(value));")
    )


DRIVER = r"""
#include "kernel_operator.h"
#include "peer_reorder.hpp"
#include <string>
using namespace AscendC;
using namespace MegaMoeUrma;
int main(int argc, char** argv) {
    assert(argc == 3);
    const uint32_t rows = std::stoul(argv[1]);
    priority = std::stoul(argv[2]);
    constexpr uint32_t h = 3584U, rowBytes = h * sizeof(bfloat16_t), rank = 3U;
    constexpr uint32_t rowsPerSlot = 65536U / rowBytes;
    constexpr uint8_t sentinel = 254U;
    const uint32_t outputRows = 3U * (rows + 23U) + 64U;
    std::vector<uint8_t> input((rows + 300U) * rowBytes);
    std::vector<uint8_t> remote(outputRows * rowBytes, sentinel), self(remote.size(), sentinel);
    std::vector<int64_t> remoteSources(outputRows, -1), selfSources(outputRows, -1);
    for (uint32_t row = 0; row < input.size() / rowBytes; ++row)
        for (uint32_t column = 0; column < rowBytes; ++column)
            input[row * rowBytes + column] = uint8_t((row * 79U + column * 31U) % 251U);

    std::vector<PeerReorderSpan> spans(2); // Nonzero spanBegin and chunkIndex.
    std::vector<PeerReorderRange> ranges(3);
    std::array<Chunk, 3> chunks{};
    std::array<uint32_t, 3> copies{};
    constexpr uint32_t widths[] = {1U, 9U, 17U, 43U, 128U, 211U};
    for (uint32_t call = 0; call < chunks.size(); ++call) {
        const uint32_t destinationBase = 17U + call * (rows + 23U);
        const uint32_t begin = spans.size();
        uint32_t spanIndex = 0U;
        for (uint32_t offset = 0U; offset < rows; ++spanIndex) {
            const uint32_t width = std::min(rows - offset, widths[spanIndex % 6U]);
            const uint64_t source = 11U + (spanIndex * 67U + call * 13U) % rows;
            spans.push_back({source, destinationBase + offset, spanIndex + 5U, width});
            auto& sourceRows = call == 1U ? selfSources : remoteSources;
            for (uint32_t row = 0; row < width; ++row) {
                assert(sourceRows[destinationBase + offset + row] == -1);
                sourceRows[destinationBase + offset + row] = source + row;
            }
            copies[call] += (width + rowsPerSlot - 1U) / rowsPerSlot;
            offset += width;
        }
        ranges.push_back({begin, uint32_t(spans.size()) - begin});
        chunks[call].peer = call == 1U ? rank : 1U;
        chunks[call].cv = call + 7U;
        chunks[call].localRow = destinationBase;
        chunks[call].rows = rows;
    }
    Arguments args{};
    args.config.rank = rank; args.config.world = 4U; args.config.hidden = h;
    args.config.returnYOffset = 128U;
    args.symmetric = reinterpret_cast<uint64_t>(self.data()) - args.config.returnYOffset;
    args.gmm2Contiguous = reinterpret_cast<uint64_t>(input.data());
    args.gmm2Output = reinterpret_cast<uint64_t>(remote.data());
    args.peerReorderRanges = reinterpret_cast<uint64_t>(ranges.data());
    args.peerReorderSpans = reinterpret_cast<uint64_t>(spans.data());
    Timer timer;
    timer.Init(0U);
    uint32_t expectedCopies = 0U;
    for (uint32_t call = 0; call < chunks.size(); ++call) {
        auto& destination = call == 1U ? self : remote;
        auto& sourceRows = call == 1U ? selfSources : remoteSources;
        const uint64_t firstByte = chunks[call].localRow * rowBytes;
        const uint64_t bytes = uint64_t(rows) * rowBytes;
        const uintptr_t base = reinterpret_cast<uintptr_t>(destination.data());
        std::vector<uint8_t> seen(bytes, 0);
        verifyStore = [&](uintptr_t address, uint16_t value) {
            assert(address >= base + firstByte && address < base + firstByte + bytes);
            const uint64_t index = address - base;
            const uint64_t row = index / rowBytes, column = index % rowBytes;
            assert(sourceRows[row] >= 0);
            assert(value == input[uint64_t(sourceRows[row]) * rowBytes + column]);
            assert(++seen[index - firstByte] == 1U);
        };
        const auto beforeSlots = slotSubmissions;
        ReorderPeerChunk(args, call + 3U, chunks[call], timer);
        expectedCopies += copies[call];
        // No external Drain(): the next action in production can submit a WQE
        // or reuse the same UB/event IDs for another peer or activation.
        assert(finishedStores == expectedCopies && submittedStores == expectedCopies);
        assert(tileReads == expectedCopies && atomicCalls == 0U);
        for (const auto& queue : queues) assert(queue.empty());
        for (const auto& kind : events) for (bool flag : kind) assert(!flag);
        for (uint32_t slot = 0U; slot < 2U; ++slot) {
            assert(slotSubmissions[slot] == slotCompletions[slot]);
            if (rows >= 256U) assert(slotSubmissions[slot] - beforeSlots[slot] > 2U);
        }
        for (uint64_t index = 0U; index < bytes; ++index) {
            assert(seen[index] == 1U);
            const uint64_t destinationIndex = firstByte + index;
            assert(destination[destinationIndex] ==
                input[uint64_t(sourceRows[destinationIndex / rowBytes]) * rowBytes + index % rowBytes]);
        }
    }
    assert(bypass);
    for (uint32_t planeIndex = 0U; planeIndex < 2U; ++planeIndex) {
        const auto& plane = planeIndex == 0U ? remote : self;
        const auto& sourceRows = planeIndex == 0U ? remoteSources : selfSources;
        for (uint64_t index = 0; index < plane.size(); ++index) {
            const int64_t sourceRow = sourceRows[index / rowBytes];
            const uint8_t expected = sourceRow < 0 ? sentinel : input[uint64_t(sourceRow) * rowBytes + index % rowBytes];
            assert(plane[index] == expected);
        }
    }
}
"""


@pytest.fixture(scope="module")
def peer_reorder_device_binary(tmp_path_factory):
    command = shlex.split(os.environ.get("CXX", "c++"))
    if not command or not shutil.which(command[0]):
        pytest.skip("A host C++ compiler is required for production peer reorder checks")
    temporary = tmp_path_factory.mktemp("urma_peer_reorder_device")
    (temporary / "kernel_operator.h").write_text(peer_reorder_fake(), encoding="utf-8")
    source = temporary / "peer_reorder.cpp"
    source.write_text(DRIVER, encoding="utf-8")
    binary = temporary / ("peer_reorder.exe" if os.name == "nt" else "peer_reorder")
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


@pytest.mark.parametrize("rows", [1, 43, 256, 499, 512, 4083, 4096])
@pytest.mark.parametrize("engine_priority", [0, 1, 2])
def test_peer_reorder_full_rows_reuses_slots_and_closes_events_before_return(
    peer_reorder_device_binary, rows, engine_priority
):
    result = subprocess.run(
        [str(peer_reorder_device_binary), str(rows), str(engine_priority)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
