from pathlib import Path
from test_device_schedule import compile_and_run


def test_shared_real_completion_channel(tmp_path):
    (tmp_path/'c_api').mkdir()
    (tmp_path/'c_api/asc_simd.h').write_text('// no device implementation in the host protocol model\n')
    compile_and_run(Path(__file__).with_name('shared_epilogue_credit_host.cpp'),tmp_path)
