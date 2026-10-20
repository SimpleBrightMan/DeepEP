"""Execute the production row owner and verify exact one-block readiness contribution."""
from pathlib import Path
from test_device_schedule import compile_and_run


def test_every_row_once_and_ready_sum(tmp_path):
    compile_and_run(Path(__file__).with_name('strided_row_owner_host.cpp'),tmp_path)
