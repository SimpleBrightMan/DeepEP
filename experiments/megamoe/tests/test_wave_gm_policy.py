"""执行真实host/device共享策略；GM常驻UB及容量边界，不替代NPU运行。"""
from pathlib import Path
from test_device_schedule import compile_and_run


def test_wave_gm_policy_and_resident_capacity(tmp_path):
    compile_and_run(Path(__file__).with_name('wave_gm_policy_host.cpp'),tmp_path)
