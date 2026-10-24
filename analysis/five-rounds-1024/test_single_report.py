"""单版本统测仍从每个rank的真实CSV重算，拒绝缺rank和指纹漂移。"""
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


class SingleVersionReport(unittest.TestCase):
    def make_case(self, root):
        def write(path, value):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value), encoding='utf-8')
        manifest = dict(ep=8, bs_values=[32], schedule=['control-a', 'control-b'],
            completed=['bs32-control-a', 'bs32-control-b'], timer_requested=False,
            packages={'control': {'native': {'kernel.so': 'frozen'}}},
            scope='test fixture', accuracy='not_run_user_requested', rack='test', nodes=['test'])
        write(root/'experiment.json', manifest)
        events=[]
        for label in manifest['completed']:
            events.append(dict(stage='node_exit', label=label, exit=0))
            write(root/label/'performance.json', dict(ep=8, bs=32, timer=False,
                model={'num_experts':384}, mean_us=100., rank_max_mean_us=107.))
            for rank in range(8):
                folder=root/f'node0/runs/{label}/rank_{rank}'
                write(folder/'identity.json', dict(native={'kernel.so':'frozen'},rank=rank))
                write(folder/'inputs.json', {'same':'inputs'})
                write(folder/'output.json', {'finite':True,'sha256':str(rank)})
                file=folder/'profiler/kernel_details.csv';file.parent.mkdir()
                with file.open('w',newline='') as stream:
                    writer=csv.DictWriter(stream,fieldnames=['Name','Start Time(us)','Duration(us)'])
                    writer.writeheader()
                    writer.writerows(dict(Name='MegaMoe',**{'Start Time(us)':i,'Duration(us)':100+rank}) for i in range(20))
        (root/'events.jsonl').write_text('\n'.join(json.dumps(e) for e in events))

    def run_report(self,root):
        return subprocess.run([sys.executable,str(Path(__file__).with_name('report.py')),str(root)],
            capture_output=True,text=True)

    def test_single_version(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);self.make_case(root)
            result=self.run_report(root);self.assertEqual(result.returncode,0,result.stderr)
            case=json.loads((root/'summary.json').read_text())['cases'][0]
            self.assertIsNone(case['reduction_pct'])
            self.assertEqual(set(case['methods']),{'control'})
            self.assertEqual(case['methods']['control']['starts'],2)
            self.assertEqual(case['methods']['control']['rank_max_mean_us'],107.)

    def test_missing_rank_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);self.make_case(root)
            (root/'node0/runs/bs32-control-b/rank_7/profiler/kernel_details.csv').unlink()
            self.assertNotEqual(self.run_report(root).returncode,0)

    def test_existing_abba_still_supported(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);self.make_case(root)
            manifest=json.loads((root/'experiment.json').read_text())
            manifest['packages']['combined']=manifest['packages']['control']
            manifest['schedule']=['control-a','combined-a','combined-b','control-b']
            for repeat in ('a','b'):
                old,new=f'bs32-control-{repeat}',f'bs32-combined-{repeat}'
                shutil.copytree(root/old,root/new)
                shutil.copytree(root/f'node0/runs/{old}',root/f'node0/runs/{new}')
                manifest['completed'].append(new)
                with (root/'events.jsonl').open('a') as log:
                    log.write('\n'+json.dumps(dict(stage='node_exit',label=new,exit=0)))
            (root/'experiment.json').write_text(json.dumps(manifest))
            result=self.run_report(root);self.assertEqual(result.returncode,0,result.stderr)
            case=json.loads((root/'summary.json').read_text())['cases'][0]
            self.assertEqual(set(case['methods']),{'control','combined'})
            self.assertEqual(case['reduction_pct'],0.)

    def test_changed_output_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);self.make_case(root)
            (root/'node0/runs/bs32-control-b/rank_7/output.json').write_text(json.dumps({'finite':True,'sha256':'different'}))
            self.assertNotEqual(self.run_report(root).returncode,0)


if __name__=='__main__':unittest.main()
