import csv
import importlib.util
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('demo_report',Path(__file__).resolve().parents[1]/'report.py')
report=importlib.util.module_from_spec(spec);spec.loader.exec_module(report)

def write(tmp,rank,rows):
    with (tmp/f'rank{rank}.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['iteration','kind','worker','peer','start_tick','end_tick','bytes']);w.writerows(rows)

def case(tmp,compute):
    for rank in range(4):
        rows=[]
        for iteration in range(2):
            offset=rank*10000000+iteration*100000
            for peer in range(4):
                if peer==rank:continue
                worker=56+(peer if peer<rank else peer-1)
                rows.extend([[iteration,'total',worker,peer,offset,offset+10000,300000],
                    [iteration,'dispatch_submit',worker,peer,offset,offset+2000,200000],
                    [iteration,'combine_shared_tail',worker,peer,offset+2000,offset+10000,100000]])
            if compute:
                for core in range(28):
                    rows.append([iteration,'gmm',core,-1,offset+1000,offset+6000,0])
                    rows.append([iteration,'dequant',core+28,-1,offset,offset+8000,0])
        write(tmp,rank,rows)

def test_completion_vs_submission_and_compute_overlap(tmp_path):
    case(tmp_path,True)
    r=report.analyze(tmp_path,[0,1,4,5],2,4,1000,True)
    assert r['classes']['TOTAL']['mean_gbps']==90
    assert r['classes']['INTRA']['mean_gbps']==30
    assert r['classes']['CROSS']['mean_gbps']==60
    assert r['classes']['COMBINE_TOTAL']['mean_gbps']==37.5
    assert r['classes']['DISPATCH_SUBMIT_TOTAL']['mean_gbps'] is None
    assert r['classes']['GMM']['round_max_mean_us']==5
    assert r['classes']['DEQUANT']['round_max_mean_us']==8
    assert r['gmm_coverage_combine']==.5

def test_off_and_missing_rank_records(tmp_path):
    case(tmp_path,False)
    r=report.analyze(tmp_path,[0,1,2,3],2,4,1000,False)
    assert r['classes']['CROSS'] is None and r['classes']['GMM'] is None
    assert r['gmm_coverage_total']==0
    write(tmp_path,0,[])
    with pytest.raises(ValueError,match='peer'):report.analyze(tmp_path,[0,1,2,3],2)

def test_no_external_ready_protocol_or_forced_l2_bypass():
    root=Path(__file__).resolve().parents[1]
    kernel=(root/'kernel.cpp').read_text(encoding='utf-8')
    compute=(root/'compute.hpp').read_text(encoding='utf-8')
    assert 'PublishCombineRange(' not in kernel and 'PublishOrdered(' not in kernel
    assert 'CACHE_MODE_DISABLE' not in kernel
    assert 'WaitComputeFlag(' not in compute and 'PendingTile<' not in compute
    assert 'gmm.Compute(' in compute and 'gmm.Dequantize(' in compute
    assert 'CompleteComputeContext()' in compute


def test_mfu_uses_rank_local_aic_envelope_and_round_max_mean(tmp_path,capsys):
    model=dict(bs=8192,hidden=5120,intermediate=2304,topk=6,shared_experts=1,shared_intermediate=2304)
    durations=[[100,400],[200,300],[300,200],[400,100]]
    for rank in range(4):
        rows=[]
        for iteration in range(2):
            origin=rank*1000000000+iteration*10000000
            end=origin+durations[rank][iteration]*1000
            for core in range(28):
                # Individual durations are smaller than the whole-card envelope.
                start=origin if core==0 else origin+1000
                stop=end-1000 if core==0 else end
                rows.append([iteration,'gmm',core,-1,start,stop,0])
                rows.append([iteration,'dequant',28+core,-1,origin,end+10000,0])
        write(tmp_path,rank,rows)
    r=report.analyze(tmp_path,[0,1,2,3],2,compute_enable=True,communication_enable=False,model=model,peak_tflops=756)
    assert r['classes']['TOTAL'] is None and r['gmm_coverage_combine'] is None
    assert r['classes']['GMM']['round_max_mean_us']==400
    assert r['classes']['GMM']['round_min_mean_us']==100
    assert r['mfu']['flops_per_card']==4058744094720
    assert r['mfu']['percent']==pytest.approx(4058744094720/(400*756*1e6)*100)
    # Diagnostic may exceed100% with synthetic durations; do not silently clip.
    assert r['mfu']['percent']>100
    report.display(r,'on')
    assert '[MFU]' in capsys.readouterr().out
    rows=[row for row in rows if not (row[1]=='gmm' and row[2]==27)]
    write(tmp_path,3,rows)
    with pytest.raises(ValueError,match='compute'):report.analyze(tmp_path,[0,1,2,3],2,compute_enable=True,communication_enable=False,model=model)


def test_mfu_shared_width_independent_and_compute_off_is_missing(tmp_path):
    case(tmp_path,True)
    model=dict(bs=1024,hidden=5120,intermediate=2304,topk=6,shared_experts=1,shared_intermediate=1280)
    r=report.analyze(tmp_path,[0,1,2,3],2,compute_enable=True,model=model)
    assert r['mfu']['flops_per_card']==6*1024*5120*(6*2304+1280)
    model['shared_experts']=0
    r=report.analyze(tmp_path,[0,1,2,3],2,compute_enable=True,model=model)
    assert r['mfu']['flops_per_card']==6*1024*5120*6*2304
    case(tmp_path,False)
    r=report.analyze(tmp_path,[0,1,2,3],2,model=model)
    assert r['mfu'] is None
    with pytest.raises(ValueError,match='Peak'):report.analyze(tmp_path,[0,1,2,3],2,peak_tflops=float('nan'))
