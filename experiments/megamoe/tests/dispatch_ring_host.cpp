// Execute the production ring and owner-finish protocol; DMA leaves are CPU mocks.
#include <array>
#include <cassert>
#include <cstdint>
#include <vector>
#define __aicore__
#define __gm__
using GM_ADDR = uint8_t*;
using TEventID = uint32_t;
constexpr uint32_t MAX_DISPATCH_BUFFER_COUNT = 6;
namespace AscendC { enum class HardEvent { MTE3_MTE2, MTE3_S, MTE2_S }; }
using MegaMoeDispatchBufferConfig = struct { int32_t bufferCount; };
struct TokenDispatchConfig { MegaMoeDispatchBufferConfig bufferConfig; uint32_t quantWinOffset = 0; };
struct Params {};
struct MoeSyncWorkspaceLayout {};
template <typename T> struct GlobalTensor { void SetGlobalBuffer(T*) {} };
struct Indices { int32_t GetValue(uint32_t i) const { return static_cast<int32_t>(i); } };
template <typename T> struct TokenDispatchScratch { Indices validTopkIndexTensor; };
#include "dispatch_ring_types.inc"

std::array<bool, 6> fetched{}, reusable{}, finished{};
std::array<int32_t, 6> token{};
std::vector<int32_t> copied;
uint32_t readyCount = 0;

template <AscendC::HardEvent Event> void WaitFlag(TEventID slot) {
    if constexpr (Event == AscendC::HardEvent::MTE3_MTE2) {
        assert(reusable[slot]);
        reusable[slot] = false;
    } else if constexpr (Event == AscendC::HardEvent::MTE3_S) {
        assert(finished[slot]);
        finished[slot] = false;
    }
}
template <AscendC::HardEvent Event, uint32_t Id> void SyncFuncStatic() {}
template <bool Reuse, bool Prefetch, typename T>
void FetchDispatchTokenAndMetaInfo(const TokenDispatchConfig&, const Params&, TokenDispatchScratch<T>&,
                                   uint32_t slot, int32_t index, int32_t, GlobalTensor<T>&) {
    if constexpr (Reuse) {
        WaitFlag<AscendC::HardEvent::MTE3_MTE2>(slot);
        WaitFlag<AscendC::HardEvent::MTE3_S>(slot);
    }
    assert(!fetched[slot] && !reusable[slot] && !finished[slot]);
    fetched[slot] = true;
    token[slot] = index;
}
template <typename T, typename S, bool Prefetch>
void StoreDispatchTokenAndMetaInfo(const TokenDispatchConfig&, const Params&, TokenDispatchScratch<T>&,
                                   uint32_t slot, int32_t row) {
    assert(fetched[slot] && !reusable[slot] && !finished[slot]);
    assert(row == static_cast<int32_t>(copied.size()) && token[slot] == row);
    fetched[slot] = false;
    reusable[slot] = finished[slot] = true;
    copied.push_back(row);
}
void PublishGmm1TileReady(const MoeSyncWorkspaceLayout&, const Params&, uint32_t expert,
                          int32_t tile, int32_t begin, int32_t end) {
    for (uint32_t i = 0; i < 6; ++i) assert(!fetched[i] && !reusable[i] && !finished[i]);
    assert(expert == 2 && tile == 256 && begin == 0 && end == static_cast<int32_t>(copied.size()));
    ++readyCount;
}
#include "dispatch_ring_methods.inc"

int main() {
    uint8_t dummy = 0;
    GM_ADDR peers[] = {&dummy, &dummy};
    Params params;
    MoeSyncWorkspaceLayout sync;
    TokenDispatchScratch<uint8_t> scratch;
    for (int32_t slots = 1; slots <= 6; ++slots) {
        for (int32_t rows : {0, 1, 2, 5, 6, 7, 33}) {
            copied.clear();
            readyCount = 0;
            TokenDispatchConfig config{{slots}};
            DispatchRouteBatch batch{};
            DispatchCopyProgress progress{};
            // One-row peers and repeated batch flushes must share the same live ring.
            for (int32_t row = 0; row < rows; ++row) {
                if (batch.count == MAX_DISPATCH_BUFFER_COUNT) {
                    ConsumeRouteBatch<uint8_t, int32_t, false>(config, params, peers, scratch, batch, progress);
                }
                batch.segments[batch.count++] = {static_cast<uint32_t>(row % 2), row,
                                                static_cast<uint32_t>(row), 1};
                ++batch.usedItems;
            }
            FinishDispatchOwnedRows<uint8_t, int32_t, 256, false>(
                config, sync, params, peers, scratch, 2, {0, 0, static_cast<uint32_t>(rows)}, batch, progress);
            assert(copied.size() == static_cast<uint32_t>(rows) && readyCount == 1);
            assert(batch.count == 0 && batch.usedItems == 0);
        }
    }
}
