"""Execute the production AIV1 loop with deterministic readiness (not a timing model)."""
from pathlib import Path

from test_device_schedule import compile_and_run, cpp_body


def test_production_aiv1_fair_loop(tmp_path):
    kernel = Path(__file__).resolve().parents[1] / 'csrc/kernels/mega_moe'
    source = (kernel / 'mega_moe_wave_a8w4.hpp').read_text(encoding='utf-8')
    body = cpp_body(source, '::ProcessAiv1PriorityStages(')
    (tmp_path / 'aiv1_loop.inc').write_text(
        'void ProcessAiv1PriorityStages(int32_t& readySequence)' + body, encoding='utf-8')
    compile_and_run(Path(__file__).with_name('aiv1_fair_scheduler_host.cpp'), tmp_path)
