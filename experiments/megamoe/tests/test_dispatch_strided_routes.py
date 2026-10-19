# Copyright (c) 2026, SimpleBright_Man

"""Execute Dispatch's actual route batching and strided row selection."""

from pathlib import Path

from test_combine_row_pipeline import _function
from test_comm_split import compile_and_run


def test_actual_dispatch_strided_routes(tmp_path):
    tests = Path(__file__).parent
    source = (tests.parent / "csrc/kernels/mega_moe/stage/mega_moe_token_dispatch.hpp").read_text(encoding="utf-8")
    begin = source.index("struct DispatchCopyProgress {")
    end = source.index("// 由 tiling", begin)
    declarations = source[begin:end]
    functions = [
        _function(source, name, template=True)
        for name in (
            "CopyTokensAndMetaForDispatch",
            "DrainDispatchCopyBuffers",
            "FlushDispatchRouteBatch",
            "DispatchRankTokens",
        )
    ]
    fixture = (tests / "dispatch_strided_routes_test.cpp").read_text(encoding="utf-8")
    generated = tmp_path / "dispatch_strided_routes_production.cpp"
    generated.write_text(
        fixture.replace("// INSERT_ROUTE_DECLARATIONS", declarations).replace(
            "// INSERT_ROUTE_FUNCTIONS", "\n\n".join(functions)
        ),
        encoding="utf-8",
    )
    completed = compile_and_run(str(generated), tmp_path)
    assert (
        "Dispatch actual route batching: strides 1..27, bounded spans, holes, peers, partial tails passed"
        in completed.stdout
    )
