"""Use an isolated, hash-checked candidate wheel without changing installed packages."""
from pathlib import Path
import os
import runpy
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'package'))
module = sys.argv.pop(1)
if module not in ('benchmark', 'golden', 'timer'):
    raise ValueError('Unknown experiment entry point')
if module == 'golden':
    # CPU Golden 不占满整机线程，不改变其他进程的 affinity。
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:16])
runpy.run_module('deep_ep_experimental.megamoe.' + module, run_name='__main__')
