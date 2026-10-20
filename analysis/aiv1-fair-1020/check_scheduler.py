"""Check expanded CPU readiness cases against the unchanged native-build scheduler body."""
from pathlib import Path
import sys
import tempfile

repo, fixture = map(Path, sys.argv[1:])
sys.path.insert(0, str(repo / 'experiments/megamoe/tests'))
from test_device_schedule import compile_and_run, cpp_body

source = (repo / 'experiments/megamoe/csrc/kernels/mega_moe/mega_moe_wave_a8w4.hpp').read_text()
with tempfile.TemporaryDirectory(prefix='aiv1-host-') as directory:
    out = Path(directory)
    (out / 'aiv1_loop.inc').write_text('void ProcessAiv1PriorityStages(int32_t& readySequence)' +
                                     cpp_body(source, '::ProcessAiv1PriorityStages('))
    compile_and_run(fixture, out)
print('SCHEDULER_READY_FALLBACK_REPLAY_EMPTY_PASS')
