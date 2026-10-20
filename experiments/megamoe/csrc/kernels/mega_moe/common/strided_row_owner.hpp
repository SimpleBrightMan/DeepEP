#pragma once
#include <cstdint>
#if defined(__NPU_DEVICE__)
#define MEGAMOE_ROW_INLINE __aicore__ inline
#else
#define MEGAMOE_ROW_INLINE inline
#endif
namespace MegaMoeImpl {
// DS的物理core c负责 c,c+cores,...；仅映射owner，不改变目标行及其源token。
struct StridedRowOwner {
    uint32_t row, rows, cores;
    MEGAMOE_ROW_INLINE bool Next(uint32_t& result) {
        if (cores==0U || row>=rows) return false;
        result=row;
        row+=cores;
        return true;
    }
};
} // namespace MegaMoeImpl
#undef MEGAMOE_ROW_INLINE
