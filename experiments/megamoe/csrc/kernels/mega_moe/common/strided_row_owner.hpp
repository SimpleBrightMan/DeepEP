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
// 与DS一致，用互素步长打散拼接source段；映射是双射，目标行的metadata必须跟随真实源行。
struct CoprimeRowPermutation {
    uint32_t rows, stride;
    static MEGAMOE_ROW_INLINE uint32_t Gcd(uint32_t a,uint32_t b) {
        while(b!=0U) { const uint32_t r=a%b;a=b;b=r; }
        return a;
    }
    static MEGAMOE_ROW_INLINE CoprimeRowPermutation Make(uint32_t count,uint32_t sources) {
        if(count==0U || sources==0U) return {count,0U};
        uint32_t step=(count+sources-1U)/sources;
        while(Gcd(step,count)!=1U)++step;
        return {count,step%count};
    }
    MEGAMOE_ROW_INLINE uint32_t Map(uint32_t row) const {
        return rows==0U?0U:static_cast<uint32_t>(static_cast<uint64_t>(row)*stride%rows);
    }
};
} // namespace MegaMoeImpl
#undef MEGAMOE_ROW_INLINE
