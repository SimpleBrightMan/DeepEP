"""用假ps/grep/npu-smi执行实际门禁，不占用设备；覆盖全组启动的本机失败边界。"""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class GateTest(unittest.TestCase):
    def test_actual_gate(self):
        bash=shutil.which('bash')
        if bash is None:self.skipTest('Linux build host required')
        gate=Path(__file__).with_name('cohort_gate.sh').resolve()
        with tempfile.TemporaryDirectory(prefix='r31-wave-gate-') as directory:
            lock=Path(directory)/'launch.lock'
            for external,idle,smi_ok,go,expected in ((1,8,1,'GO',4),(0,7,1,'GO',4),
                    (0,8,0,'GO',4),(0,8,1,'CANCEL',5),(0,8,1,'GO',0)):
                command=r'''
flock() { return 0; }
ps() { printf '4242 python\n'; }
grep() { if [[ $1 == -qE ]]; then (( external == 1 )); else command grep "$@"; fi; }
npu-smi() { (( smi_ok == 1 )) || return 1; for ((i=0;i<idle;i++)); do echo "No running processes found in NPU $i"; done; }
source "$gate" "$lock"
'''
                variables=f'external={external};idle={idle};smi_ok={smi_ok};gate={str(gate)!r};lock={str(lock)!r};\n'
                result=subprocess.run([bash,'-c',variables+command],input=go+'\n',text=True,capture_output=True,timeout=10)
                self.assertEqual(result.returncode,expected,result.stdout+result.stderr)
                self.assertEqual('COHORT_READY' in result.stdout,external==0 and idle==8 and smi_ok==1)


if __name__=='__main__':unittest.main()
