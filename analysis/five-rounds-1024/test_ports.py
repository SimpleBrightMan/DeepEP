"""执行实际启动器端口门禁，覆盖冲突、探测失败及临时端口边界；不访问NPU。"""
from pathlib import Path
import shutil
import subprocess
import unittest


class PortTests(unittest.TestCase):
    def test_actual_port_guard(self):
        bounds=Path('/proc/sys/net/ipv4/ip_local_port_range')
        if shutil.which('bash') is None or not bounds.is_file():
            self.skipTest('Run on Linux build host')
        first,last=map(int,bounds.read_text().split())
        if first<2048:self.skipTest('No suitable below-ephemeral fixture range')
        script=Path(__file__).with_name('run_node.sh').read_text()
        body=script[script.index('read -r ephemeral_first'):script.index('source "$root/cohort_gate.sh"')]
        for port,occupied,probe_ok,expected in ((first-100,0,1,0),(first-100,1,1,3),
                (first-100,0,0,3),(first,0,1,3),(first-3,0,1,3),(last,0,1,3)):
            stub=f'port={port};occupied={occupied};probe_ok={probe_ok};\n'+r'''
ss() { (( probe_ok == 1 )) || return 1; if (( occupied == 1 )); then echo occupied; fi; }
'''
            result=subprocess.run(['bash','-c',stub+body],text=True,capture_output=True,timeout=10)
            self.assertEqual(result.returncode,expected,result.stdout+result.stderr)


if __name__=='__main__':unittest.main()
