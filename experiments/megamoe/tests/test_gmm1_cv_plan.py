"""Execute production CV task stepping, capacity and queued scratch credit ownership."""
from pathlib import Path
from test_device_schedule import compile_and_run


def test_cv_tails_slots_and_scratch_ownership(tmp_path):
    compile_and_run(Path(__file__).with_name('gmm1_cv_plan_host.cpp'), tmp_path)
