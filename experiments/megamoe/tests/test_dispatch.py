"""Execute production Dispatch planning against the frozen pre-refactor layout."""

from pathlib import Path

import pytest

from test_device_schedule import ROOT, compile_and_run, cpp_body


def function_source(source, name):
    position = source.index(name + "(")
    declaration = source.rfind("__aicore__", 0, position)
    template = source.rfind("template <", 0, declaration)
    start = template if template > source.rfind("}", 0, declaration) else declaration
    begin = source.index("{", position)
    return source[start:begin] + cpp_body(source, name + "(")


@pytest.mark.parametrize("fast_path", [0, 1])
@pytest.mark.parametrize("core", [0, 1])
def test_dispatch_layout_and_direct_slice_match_legacy(tmp_path, fast_path, core):
    kernel = ROOT / "experiments/megamoe/csrc/kernels/mega_moe"
    source = (kernel / "stage/mega_moe_token_dispatch.hpp").read_text(encoding="utf-8")
    utils = (kernel / "common/mega_moe_utils.hpp").read_text(encoding="utf-8")
    types = "struct WorkRange" + cpp_body(utils, "struct WorkRange") + ";\n"
    types += "struct ExpertDispatchCoreRange" + cpp_body(source, "struct ExpertDispatchCoreRange") + ";\n"
    types += "struct DispatchPeerCounts" + cpp_body(source, "struct DispatchPeerCounts") + ";\n"
    (tmp_path / "dispatch_plan_types.inc").write_text(types, encoding="utf-8")
    methods = [function_source(utils, name) for name in ("GetBalancedTokenRange", "GetRotatedBalancedTokenRange")]
    methods += [function_source(source, name) for name in (
        "FindDispatchSourceRank", "FinishDispatchOwnedRows", "DispatchOwnedExpertRows", "DispatchPeerBalancedRows",
        "DispatchOwnedExpertSlice", "DispatchExpertSlice", "DispatchTokenRange",
    )]
    (tmp_path / "dispatch_plan_methods.inc").write_text("\n".join(methods), encoding="utf-8")
    fixture = Path(__file__).with_name("dispatch_execution_host.cpp").read_text(encoding="utf-8")
    translation_unit = tmp_path / "dispatch_execution.cpp"
    translation_unit.write_text(
        f"#define MEGAMOE_PREFILL_PEER_FAST_PATH {fast_path}\n#define TEST_CORE {core}\n" + fixture,
        encoding="utf-8",
    )
    compile_and_run(translation_unit, tmp_path)


def test_dispatch_ring_drains_before_ready(tmp_path):
    kernel = ROOT / "experiments/megamoe/csrc/kernels/mega_moe"
    source = (kernel / "stage/mega_moe_token_dispatch.hpp").read_text(encoding="utf-8")
    types = "\n".join("struct " + name + cpp_body(source, "struct " + name) + ";" for name in (
        "ExpertDispatchCoreRange", "DispatchCopyProgress", "DispatchRouteSegment", "DispatchRouteBatch",
    ))
    (tmp_path / "dispatch_ring_types.inc").write_text(types, encoding="utf-8")
    methods = "\n".join(function_source(source, name) for name in (
        "IssuePeerTokenCopies", "DrainDispatchCopyBuffers", "ConsumeRouteBatch", "FinishDispatchOwnedRows",
    ))
    (tmp_path / "dispatch_ring_methods.inc").write_text(methods, encoding="utf-8")
    compile_and_run(Path(__file__).with_name("dispatch_ring_host.cpp"), tmp_path)
