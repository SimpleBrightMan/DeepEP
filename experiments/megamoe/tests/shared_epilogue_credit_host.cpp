// 执行生产完成通道；本模型检查credit配对，不冒充跨核可见性的原生证明。
#include <array>
#include <cassert>
#define __aicore__
enum { PIPE_FIX, PIPE_V, PIPE_S, PIPE_MTE3, PIPE_MTE2 };
std::array<bool,16> flags{};
int lastWait=-1,lastArrive=-1;
void asc_sync_intra_arrive(int pipe,unsigned id) {
    assert(id&16U);id&=15U;assert(!flags[id]);flags[id]=true;lastArrive=pipe;
}
void asc_sync_intra_wait(int pipe,unsigned id) {
    id&=15U;assert(flags[id]);flags[id]=false;lastWait=pipe;
}
namespace AscendC {
enum class HardEvent { FIX_S };
template<HardEvent> void SetFlag(unsigned) {}
template<HardEvent> void WaitFlag(unsigned) {}
}
#include "common/gmm1_async_completion.hpp"
int main() {
    using Channel=MegaMoeImpl::Gmm1AsyncCompletion;
    Channel::InitConsumer();
    for(unsigned sequence=0;sequence<101;++sequence) {
        Channel::AcquireProducer(sequence);assert(lastWait==PIPE_FIX);
        Channel::PublishCompleted(sequence);assert(lastArrive==PIPE_FIX);
        Channel::WaitConsumer(sequence);assert(lastWait==PIPE_MTE2);
        Channel::ReleaseConsumer(sequence);assert(lastArrive==PIPE_MTE3);
    }
    Channel::DrainProducer();
    for(bool flag:flags) assert(!flag);
}
