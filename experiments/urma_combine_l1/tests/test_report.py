import csv
import importlib.util
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('demo_report',Path(__file__).resolve().parents[1]/'report.py')
report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)


def write(tmp,rank,rows):
    with (tmp/f'rank{rank}.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['iteration','kind','worker','peer','start_tick','end_tick','bytes']);w.writerows(rows)


def case(tmp,loads):
    for rank in range(4):
        rows=[]
        for iteration in range(2):
            # Clocks deliberately differ between ranks by a large offset.
            offset=rank*10000000+iteration*100000
            for peer in range(4):
                if peer!=rank:rows.append([iteration,'combine',28+(peer if peer<rank else peer-1),peer,offset,offset+10000,100000])
            if loads:
                for core in range(28):rows.append([iteration,'l1',core,-1,offset+1000,offset+6000,1000])
        write(tmp,rank,rows)


def test_local_envelope_one_direction_and_actual_overlap(tmp_path):
    case(tmp_path,True)
    r=report.analyze(tmp_path,[0,1,4,5],2,4,1000,True)
    assert r['classes']['TOTAL']['mean_gbps']==30
    assert r['classes']['INTRA']['mean_gbps']==10
    assert r['classes']['CROSS']['mean_gbps']==20
    assert r['classes']['TOTAL']['round_max_mean_us']==10
    assert r['classes']['L1']['mean_gbps']==5.6
    assert r['aic_active_coverage_mean']==.5


def test_off_and_missing_rank_records(tmp_path):
    case(tmp_path,False)
    r=report.analyze(tmp_path,[0,1,2,3],2,4,1000,False)
    assert r['classes']['CROSS'] is None and r['classes']['L1'] is None
    assert r['aic_active_coverage_mean']==0
    write(tmp_path,0,[])
    with pytest.raises(ValueError,match='peer'):report.analyze(tmp_path,[0,1,2,3],2)


def test_8192_volume_contract():
    bs,h,i,k=8192,3584,1280,8
    assert bs*k//16==4096
    assert 256*h*2==1835008
    assert (bs*k//16)*h*2==28*1024**2
    assert bs*k*(h*10+i*14)==3523215360
    assert (bs*k//16)*h*2*15==420*1024**2
