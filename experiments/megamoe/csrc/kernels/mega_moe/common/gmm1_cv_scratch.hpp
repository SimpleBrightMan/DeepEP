// Copyright (c) 2026, Lu Lu
#ifndef MEGAMOE_GMM1_CV_SCRATCH_HPP
#define MEGAMOE_GMM1_CV_SCRATCH_HPP
namespace MegaMoeImpl {
// V写scratch -> MTE3读scratch -> V复用。CV输入槽另由AIC/AIV full/empty保护。
struct Gmm1CvScratch {
    __aicore__ static inline void Init() { AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0); }
    __aicore__ static inline void Acquire() { AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(0); }
    __aicore__ static inline void Release() { AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(0); }
    __aicore__ static inline void Finish() { Acquire(); }
};
}
#endif
