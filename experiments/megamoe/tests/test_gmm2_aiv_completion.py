"""Exercise production cross-core completion methods with delayed pipeline queues."""
from pathlib import Path
from test_device_schedule import ROOT, compile_and_run
from test_dispatch import function_source


def test_fix_to_aiv1_completion_credit_and_ready_order(tmp_path):
    (tmp_path / "kernel_operator.h").write_text(r"""
#pragma once
#include <cstdint>
#define __aicore__
#define __gm__
enum Pipe { PIPE_S, PIPE_FIX };
constexpr int AIC = 0, AIV = 1;
extern unsigned activeCore, activeSub;
void Record(unsigned pipe, unsigned kind, unsigned flag);
namespace AscendC {
constexpr int AIC = ::AIC, AIV = ::AIV;
enum class HardEvent { FIX_S };
inline unsigned GetSubBlockIdx() { return activeSub; }
template <unsigned Mode, Pipe P> void CrossCoreSetFlag(unsigned f) {
    static_assert(Mode == 4); Record(P, 0, f);
}
template <unsigned Mode, Pipe P> void CrossCoreWaitFlag(unsigned f) {
    static_assert(Mode == 4); Record(P, 1, f);
}
template <HardEvent E> void SetFlag(unsigned f) { Record(PIPE_FIX, 3, f); }
template <HardEvent E> void WaitFlag(unsigned f) { Record(PIPE_S, 4, f); }
void AtomicAdd(int32_t*, int);
void WriteGmByPassDCache(int32_t*, int32_t);
inline int32_t ReadGmByPassDCache(int32_t* p) { return *p; }
}
""")
    kernel = ROOT / "experiments/megamoe/csrc/kernels/mega_moe"
    source = (kernel / "stage/mega_moe_gmm2_combine.hpp").read_text()
    (tmp_path / "notify_slice_ready.inc").write_text(function_source(source, "NotifyTokenCombineSliceReady"))
    compile_and_run(Path(__file__).with_name("gmm2_aiv_completion_host.cpp"), tmp_path)


def test_aiv1_replays_owned_slices_and_publishes_one_ready_per_full(tmp_path):
    from test_device_schedule import cpp_body
    kernel = ROOT / "experiments/megamoe/csrc/kernels/mega_moe"
    wave = (kernel / "mega_moe_wave_a8w4.hpp").read_text()
    utils = (kernel / "common/mega_moe_utils.hpp").read_text()
    methods = function_source(utils, "PlanNextExpertTokenRangeInWave")
    methods += "\nbool NextAivSlice(AivTaskCursor& cursor, AivTaskKind kind)"
    methods += cpp_body(wave, "::NextAivSlice(")
    methods += "\nbool NextGmm2CompletionSlice(AivTaskCursor& cursor)"
    methods += cpp_body(wave, "::NextGmm2CompletionSlice(")
    methods += "\nbool TryReceiveGmm2Completion(const AivTaskCursor& cursor, int32_t& completedSequence)"
    methods += cpp_body(wave, "::TryReceiveGmm2Completion(")
    (tmp_path / "gmm2_wave_receive.inc").write_text(methods)
    compile_and_run(Path(__file__).with_name("gmm2_wave_receive_host.cpp"), tmp_path)
