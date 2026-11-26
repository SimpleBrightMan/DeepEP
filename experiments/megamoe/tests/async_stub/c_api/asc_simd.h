#pragma once
#include <vector>
#define __aicore__
enum Pipe { PIPE_FIX, PIPE_MTE2, PIPE_MTE3 };
struct Event { Pipe pipe; bool arrive; unsigned flag; };
inline std::vector<Event> recorded;
inline unsigned drains = 0;
inline void asc_sync_intra_arrive(Pipe pipe, unsigned flag) { recorded.push_back({pipe, true, flag}); }
inline void asc_sync_intra_wait(Pipe pipe, unsigned flag) { recorded.push_back({pipe, false, flag}); }
namespace AscendC {
enum class HardEvent { FIX_S };
template<HardEvent> void SetFlag(unsigned) { ++drains; }
template<HardEvent> void WaitFlag(unsigned) { ++drains; }
}
