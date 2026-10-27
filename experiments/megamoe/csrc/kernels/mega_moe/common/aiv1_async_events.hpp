// AIV1常驻任务的本地事件分区；同一HardEvent内的ID必须互不重叠。
#ifndef MEGAMOE_AIV1_ASYNC_EVENTS_HPP
#define MEGAMOE_AIV1_ASYNC_EVENTS_HPP

namespace MegaMoeImpl {
struct Aiv1AsyncEvents {
    static constexpr unsigned LocalEventCount = 8;
    // GM常驻通信环：Dispatch占0..5，Combine占6..7；不同HardEvent域独立编号。
    static constexpr unsigned DispatchSlots = 6;
    static constexpr unsigned CombineBase = DispatchSlots;
    static constexpr unsigned CombineSlots = LocalEventCount - CombineBase;
    // MTE3->S：0..5归Dispatch metadata；两种完成发布各有独立ID。
    static constexpr unsigned ActivationDone = 6;
    static constexpr unsigned DispatchDone = 7;
    // MTE3->V->MTE2：激活输出读完才归还下一轮输入UB，无须占MTE3_MTE2的ID。
    static constexpr unsigned ActivationUb = 0;
};
static_assert(Aiv1AsyncEvents::CombineBase + Aiv1AsyncEvents::CombineSlots <= Aiv1AsyncEvents::LocalEventCount);
} // namespace MegaMoeImpl
#endif
