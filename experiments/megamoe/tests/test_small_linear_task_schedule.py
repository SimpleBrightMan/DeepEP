from pathlib import Path
from test_device_schedule import compile_and_run


def test_production_linear_schedule(tmp_path):
    compile_and_run(Path(__file__).with_name('small_linear_task_schedule_host.cpp'), tmp_path)
