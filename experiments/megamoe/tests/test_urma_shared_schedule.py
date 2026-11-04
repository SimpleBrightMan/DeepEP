from pathlib import Path
import hashlib

def body(source,name):
    start=source.index("inline void "+name+"(");a=source.index("{",start);depth=1;b=a+1
    while depth:
        depth+=(source[b]=="{")-(source[b]=="}");b+=1
    return source[a:b]

def test_s0_keeps_legacy_pipeline_body():
    k=Path(__file__).resolve().parents[1]/"csrc/kernels/mega_moe/urma"
    source=(k/"kernel.hpp").read_text(encoding="utf-8")
    assert hashlib.sha256(body(source,"ProcessRouted").replace("UseLegacyReduce(args.config)", "args.config.hidden==3584U").encode()).hexdigest()=="5fa6f9bcd8b9c33610bacf5bdf0e2bb7adcac68ad387b0572de5509abe427aea"
    assert hashlib.sha256(body(source,"CommunicatePeer").encode()).hexdigest()=="5360707b9bdff990efef688b82394d830d77261e07863675287964a983e88d21"


def test_shared_mode_is_isolated_and_static_activation_is_bounded():
    k=Path(__file__).resolve().parents[1]/"csrc/kernels/mega_moe/urma"
    kernel=(k/"kernel.hpp").read_text(encoding="utf-8")
    shared=(k/"shared.hpp").read_text(encoding="utf-8")
    launcher=(k/"launcher.cpp").read_text(encoding="utf-8")
    assert "if(arguments->sharedExperts==0)" in launcher
    assert "MegaMoeUrma::ProcessRouted(args)" in launcher
    assert "MegaMoeUrma::ProcessShared(args)" in launcher
    assert "class SharedActivationState" in shared
    assert "SharedOwnedTasks(total_,worker)" in shared
    assert "RunSharedGmm<true>" in kernel and "RunSharedGmm<false>" in kernel
    assert "ContiguousRowIO" in shared
    assert "kSubmittedOffset = ActivationUbContract::inputEnd" in body(kernel,"CommunicatePeerShared")

def test_profiler_accepts_shared_entry_but_not_launcher():
    from deep_ep_experimental.megamoe.urma_timing import _URMA_KERNEL
    assert _URMA_KERNEL.search("mega_moe_urma_shared_kernel(Arguments)")
    assert _URMA_KERNEL.search("_Z27mega_moe_urma_shared_kernelN11MegaMoeUrma9ArgumentsE")
    assert not _URMA_KERNEL.search("mega_moe_urma_shared_kernel_do")
