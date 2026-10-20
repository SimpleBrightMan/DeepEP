"""Execute the row-offset contract used by the device segmented CATLASS IO.

This checks byte coverage and ownership, not hardware DMA legality or NPU
numerics. The device uses exact-N ND2NZ (only D padded to a 32-byte C0),
and an explicitly fenced sequence of FIX stores. MX scales are prepacked into
contiguous rows; a two-byte Compact L1 destination is deliberately not used.
"""

import os
from pathlib import Path
import shlex
import shutil
import subprocess

import pytest


@pytest.fixture(scope="module")
def segmented_io_binary(tmp_path_factory):
    command = shlex.split(os.environ.get("CXX", "c++"))
    if not command or not shutil.which(command[0]):
        pytest.skip("A host C++ compiler is required for segmented IO tests")
    temporary = tmp_path_factory.mktemp("urma_segmented_io")
    root = Path(__file__).resolve().parents[3]
    source = temporary / "segmented_io.cpp"
    source.write_text(
        r"""
#include "segmented_row_layout.hpp"
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <vector>
using namespace MegaMoeImpl::GmmKernel::SegmentedRows;
struct Segment {
    uint32_t tileRow, rows, rank;
    uint64_t receiveRow, returnRow;
};
int main(int argc, char** argv) {
    assert(argc == 4);
    uint32_t segmentRows = std::stoul(argv[1]), m = std::stoul(argv[2]), k = std::stoul(argv[3]);
    std::vector<Segment> segments;
    std::vector<uint64_t> expectedRows(m);
    uint32_t begin = 0, index = 0;
    while (begin < m) {
        const uint32_t rows = std::min(segmentRows, m - begin);
        const uint64_t receive = 1000 + index * 777, output = 7000 + index * 555;
        segments.push_back({begin, rows, index % 3, receive, output});
        for (uint32_t row = 0; row < rows; ++row) expectedRows[begin + row] = receive + row;
        begin += rows;
        ++index;
    }
    // All window edges, including partial tiles that begin inside a segment.
    for (uint32_t tileBegin : {0U, 1U, 15U, 16U, 31U, 32U, 43U, 85U}) {
        if (tileBegin >= m) continue;
        const uint32_t tileRows = m - tileBegin, physicalM = RoundUp16(tileRows);
        for (uint32_t kBegin = 0; kBegin < k; kBegin += 256) {
            const uint32_t columns = std::min(256U, k - kBegin);
            const uint32_t physicalK = (columns + 63) / 64 * 64;
            std::vector<uint64_t> a(physicalM * physicalK, UINT64_MAX);
            std::vector<uint32_t> writes(a.size(), 0);
            for (const auto& segment : segments) {
                auto part = Intersect(segment.tileRow, segment.rows, tileBegin, tileRows);
                // One production ND2NZ per overlap: nValue=part.rows,
                // dValue=columns, dstNzC0Stride=physicalM, dstNzNStride=1.
                const uint32_t paddedD = (columns + 31) / 32 * 32;
                assert(AOffset(part.tileRow, 0, tileRows) % 32 == 0);
                for (uint32_t row = 0; row < part.rows; ++row) {
                    for (uint32_t column = 0; column < paddedD; ++column) {
                        uint32_t local = part.tileRow + row;
                        uint64_t address = AOffset(local, column, tileRows);
                        assert(address < a.size());
                        a[address] = column < columns
                            ? (segment.receiveRow + part.segmentRow + row) * k + kBegin + column : 0;
                        ++writes[address];
                    }
                }
            }
            for (uint32_t row = 0; row < tileRows; ++row)
                for (uint32_t col = 0; col < columns; ++col) {
                    auto pos = AOffset(row, col, tileRows);
                    assert(writes[pos] == 1);
                    assert(a[pos] == expectedRows[tileBegin + row] * k + kBegin + col);
                }
            for (uint32_t row = 0; row < tileRows; ++row)
                for (uint32_t col = columns; col < (columns + 31) / 32 * 32; ++col) {
                    auto pos = AOffset(row, col, tileRows);
                    assert(writes[pos] == 1 && a[pos] == 0);
                }
            for (auto visits : writes) assert(visits <= 1); // no padded overlapping writes
        }
        // GMM2: original row identity, self return remap, and each N-tile
        // offset are preserved; different segments never release/reuse L0C.
        for (uint32_t nBegin = 0; nBegin < 3584; nBegin += 256) {
            std::vector<uint32_t> visited(tileRows, 0);
            for (const auto& segment : segments) {
                auto part = Intersect(segment.tileRow, segment.rows, tileBegin, tileRows);
                if (!part.rows) continue;
                assert(L0COffset(part.tileRow, 0, tileRows) * sizeof(float) % 64 == 0);
                uint64_t outputRow = (segment.rank == 1 ? segment.returnRow : segment.receiveRow) +
                                     part.segmentRow;
                for (uint32_t row = 0; row < part.rows; ++row) {
                    ++visited[part.tileRow + row];
                    for (uint32_t col = 0; col < 256; ++col) {
                        uint64_t dst = OutputOffset(outputRow + row, nBegin + col, 3584);
                        assert(dst / 3584 == outputRow + row && dst % 3584 == nBegin + col);
                        uint64_t src = L0COffset(part.tileRow + row, col, tileRows);
                        assert(src < static_cast<uint64_t>(physicalM) * 256);
                    }
                }
            }
            for (auto visits : visited) assert(visits == 1);
        }
    }
    // MX scale rows use an independent, contiguous GM staging plane. Verify
    // the ordinary zZ shape for both 4096-K windows and partial last windows.
    for (uint32_t kBegin = 0; kBegin < k; kBegin += 4096) {
        uint32_t scaleColumns = (std::min(4096U, k - kBegin) + 63) / 64 * 2;
        std::vector<uint64_t> scale(RoundUp16(m) * scaleColumns, UINT64_MAX);
        std::vector<uint32_t> writes(scale.size());
        for (uint32_t row = 0; row < m; ++row)
            for (uint32_t col = 0; col < scaleColumns; ++col) {
                uint64_t address = ScaleOffset(row, col, scaleColumns);
                assert(address < scale.size());
                scale[address] = expectedRows[row] * ((k + 63) / 64 * 2) + kBegin / 32 + col;
                ++writes[address];
            }
        for (uint32_t row = 0; row < m; ++row)
            for (uint32_t col = 0; col < scaleColumns; ++col)
                assert(writes[ScaleOffset(row, col, scaleColumns)] == 1);
        for (uint32_t row = 0; row < m; row += 16)
            assert(ScaleOffset(row, 0, scaleColumns) % 32 == 0);
    }
    std::cout << "OK\n";
}
"""
    )
    binary = temporary / ("segmented_io.exe" if os.name == "nt" else "segmented_io")
    include = root / "experiments/megamoe/csrc/kernels/mega_moe/catlass_gmm"
    subprocess.run(
        [*command, "-std=c++17", "-O2", "-UNDEBUG", "-I", str(include), str(source), "-o", str(binary)],
        check=True,
        capture_output=True,
        text=True,
    )
    return binary


@pytest.mark.parametrize("segment_rows", [1, 15, 16, 17, 31, 32, 43, 85, 171, 256])
@pytest.mark.parametrize("rows,k", [(256, 3584), (171, 6144), (43, 4096), (85, 8192), (85, 4129)])
def test_segmented_a_scale_and_scatter_layout(segmented_io_binary, segment_rows, rows, k):
    completed = subprocess.run(
        [str(segmented_io_binary), str(segment_rows), str(rows), str(k)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "OK"
