#pragma once
#include <type_traits>
#include "../megamoe/csrc/kernels/mega_moe/urma/tile_completion.hpp"
#if defined(__NPU_DEVICE__)
#define DEMO_COMPLETION_INLINE __aicore__ inline
#else
#define DEMO_COMPLETION_INLINE inline
#endif
namespace Demo {
template<class Backend> struct NoTileCompletion {
    DEMO_COMPLETION_INLINE explicit NoTileCompletion(Backend&) {}
    DEMO_COMPLETION_INLINE void BeforeStage(bool) {}
    DEMO_COMPLETION_INLINE void AfterSubmit(bool,uint64_t,const MegaMoeUrma::TraceMeta&) {}
    DEMO_COMPLETION_INLINE void Flush() {}
};
template<bool Enabled,class Backend>
using OptionalTileCompletion = std::conditional_t<Enabled,MegaMoeUrma::TileCompletion<Backend>,NoTileCompletion<Backend>>;
}

#undef DEMO_COMPLETION_INLINE
