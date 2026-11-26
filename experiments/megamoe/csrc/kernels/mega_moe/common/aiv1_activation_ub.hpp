// 激活的UB credit跨任务存活；通信拥有不同UB，不消费这个credit。
#ifndef MEGAMOE_AIV1_ACTIVATION_UB_HPP
#define MEGAMOE_AIV1_ACTIVATION_UB_HPP
#include "aiv1_async_events.hpp"
namespace MegaMoeImpl {
struct Aiv1ActivationUb {
    __aicore__ static inline void Init()
    {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(Aiv1AsyncEvents::ActivationUb);
    }
    __aicore__ static inline void Acquire()
    {
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(Aiv1AsyncEvents::ActivationUb);
    }
    __aicore__ static inline void Release()
    {
        // 量化输出和下轮输入重叠：MTE3最后一次读完 -> V -> 下轮MTE2覆盖。
        // 依赖排在硬件队列，不能改成MTE3_S或在Scalar提前归还。
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(Aiv1AsyncEvents::ActivationUb);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(Aiv1AsyncEvents::ActivationUb);
        AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(Aiv1AsyncEvents::ActivationUb);
    }
    __aicore__ static inline void Finish()
    {
        // 包括从未使用的初始credit；保证不遗留到下次kernel/最终Unpermute。
        Acquire();
    }
};
} // namespace MegaMoeImpl
#endif
