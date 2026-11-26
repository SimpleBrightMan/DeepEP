"""Run production activation ownership/retirement on independent queued engines."""
from pathlib import Path

from test_device_schedule import compile_and_run, cpp_body


def test_queued_activation_credit_and_completion(tmp_path):
    kernel = Path(__file__).resolve().parents[1] / 'csrc/kernels/mega_moe'
    source = (kernel / 'mega_moe_wave_a8w4.hpp').read_text(encoding='utf-8')
    (tmp_path / 'activation_retire.inc').write_text(
        'bool RetireActivation()' + cpp_body(source, '::RetireActivation('), encoding='utf-8')
    combine = (kernel / 'stage/mega_moe_gmm2_combine.hpp').read_text(encoding='utf-8')
    (tmp_path / 'combine_drain.inc').write_text(
        'void DrainCombineRowBuffers(uint32_t& issuedRowCount, uint32_t rowBufferCount, uint32_t eventBase)' +
        cpp_body(combine, 'void DrainCombineRowBuffers('), encoding='utf-8')
    compile_and_run(Path(__file__).with_name('aiv1_async_events_host.cpp'), tmp_path)
