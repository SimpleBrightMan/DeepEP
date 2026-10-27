"""CPU检查快速筛选的实际计划与报告状态；不替代远端门禁或NPU验证。"""
import importlib.util
import json
from pathlib import Path
import sys

import pytest


def module(name):
    spec=importlib.util.spec_from_file_location('quick_'+name,Path(__file__).with_name(name+'.py'))
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


def test_one_b_and_legacy_schedules():
    plan=module('campaign').measurement_schedule
    assert plan(True)==['combined-a']
    assert plan(False)==['control-a','combined-a','combined-b','control-b']
    assert plan(False,'combined')==['combined-a','combined-b']
    with pytest.raises(ValueError):plan(True,'unknown')


def run_report(tmp_path,monkeypatch,single,schedule,completed):
    report=module('report')
    manifest=dict(ep=8,bs_values=[32],schedule=schedule,completed=completed,packages={},timer_requested=False,
        scope='kernel Duration',accuracy='not_run_user_requested',rack='05',nodes=['CPU7'],single_start=single,
        comparison='historical A only' if single else 'same-window')
    (tmp_path/'experiment.json').write_text(json.dumps(manifest),encoding='utf-8')
    monkeypatch.setattr(sys,'argv',['report.py',str(tmp_path)])
    monkeypatch.setattr(report,'audit_run',lambda root,label,*args:(dict(method=label.split('-')[1],
        mean_us=600,rank_max_mean_us=601),['same-input'],['same-output']))
    return report


def test_single_start_is_not_repeated_acceptance(tmp_path,monkeypatch):
    report=run_report(tmp_path,monkeypatch,True,['combined-a'],['bs32-combined-a'])
    report.main()
    result=json.loads((tmp_path/'summary.json').read_text())
    assert result['status']=='audited_screening'
    assert result['cases'][0]['methods']['combined']['starts']==1
    assert result['cases'][0]['reduction_pct'] is None


def test_repeated_acceptance_still_requires_two_starts(tmp_path,monkeypatch):
    report=run_report(tmp_path,monkeypatch,False,['combined-a'],['bs32-combined-a'])
    with pytest.raises(ValueError,match='expected 2'):report.main()


def test_missing_case_cannot_publish_complete(tmp_path,monkeypatch):
    report=run_report(tmp_path,monkeypatch,True,['combined-a'],[])
    with pytest.raises(ValueError,match='Matrix incomplete'):report.main()
