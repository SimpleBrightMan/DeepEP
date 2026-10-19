# Copyright (c) 2026, SimpleBright_Man

from pathlib import Path

from test_prefill_integration_contract import _compile_and_run


def test_actual_timer_writer_survives_independent_cache_writebacks(tmp_path):
    completed = _compile_and_run(Path(__file__).with_suffix(".cpp"), tmp_path)
    assert "96 private cache writers, repeated publication, disabled timer passed" in completed.stdout
