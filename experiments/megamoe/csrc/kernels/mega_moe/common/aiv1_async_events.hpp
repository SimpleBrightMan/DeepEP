// AIV1常驻任务的本地事件分区；同一HardEvent内的ID必须互不重叠。
#ifndef MEGAMOE_AIV1_ASYNC_EVENTS_HPP
#define MEGAMOE_AIV1_ASYNC_EVENTS_HPP

namespace MegaMoeImpl {
struct Aiv1AsyncEvents {
    static constexpr unsigned LocalEventCount = 8;
    // DS式通信环：pull 0..1、push 2..5；6..7不占用，便于最终归约独立使用。
    static constexpr unsigned DispatchSlots = 2;
    static constexpr unsigned CombineBase = DispatchSlots;
    static constexpr unsigned CombineSlots = 4;
    // MTE3->S：0..5归Dispatch metadata；两种完成发布各有独立ID。
    static constexpr unsigned ActivationDone = 6;
    static constexpr unsigned DispatchDone = 7;
    // MTE3->V->MTE2：激活输出读完才归还下一轮输入UB，无须占MTE3_MTE2的ID。
    static constexpr unsigned ActivationUb = 0;
};
static_assert(Aiv1AsyncEvents::CombineBase + Aiv1AsyncEvents::CombineSlots <= Aiv1AsyncEvents::LocalEventCount);
} // namespace MegaMoeImpl
#endif
