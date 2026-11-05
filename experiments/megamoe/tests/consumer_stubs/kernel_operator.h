#pragma once
#include <cstdint>
#define __aicore__
namespace CreditModel { void Drain(unsigned pipe); }
namespace AscendC {
using TEventID=uint16_t;
enum class HardEvent { MTE2_S, FIX_S };
template<HardEvent E> inline void SetFlag(TEventID) {}
template<HardEvent E> inline void WaitFlag(TEventID) {
    CreditModel::Drain(E==HardEvent::MTE2_S ? 1U : 0U);
}
}
