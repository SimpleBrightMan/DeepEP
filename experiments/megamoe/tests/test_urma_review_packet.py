"""Offline comment packets: bounded size, multi-BS discovery and capture identity."""
import importlib.util
import json
from pathlib import Path
import pytest

@pytest.fixture
def packet(monkeypatch):
    scripts=Path(__file__).resolve().parents[1]/"scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec=importlib.util.spec_from_file_location("review_packet_test",scripts/"urma_review_packet.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def document(bs=1024, mode="coarse", repeats=1):
    out=dict(metadata=dict(rank_count=4,per_rank=[]),traceEvents=[])
    ids=([(62,"RoutePrepare",2),(63,"RouteSummary",1),(46,"CoarseGMM",0),(47,"CoarseDequant",1),
          (48,"CoarseDispatch",2),(49,"CoarseCombine",2),(51,"CoarseReduce",1),(52,"CoarseLocalSyncEnd",0)]
         if mode=="coarse" else [(15,"Gmm1Tile",0),(19,"Gmm2Tile",0),(1,"DispatchPutX",2),
                                 (10,"CombinePut",2),(40,"Gmm2Reorder",2),(25,"Reduce",1)])
    for rank in range(4):
        out["metadata"]["per_rank"].append(dict(schema=1,rank=rank,world=4,bs=bs,instrumentation=mode,
            envelope_us=100*repeats+rank,model=dict(hidden=3584,intermediate_hidden=1280,num_experts=384,num_topk=8)))
        for n in range(repeats):
            for i,(eid,name,engine) in enumerate(ids):
                out["traceEvents"].append(dict(ph="X",pid=rank,tid=n%28*3+engine,name=name,cat="test",
                    ts=n*100+i*8,dur=5+rank,args=dict(event_id=eid,peer=(rank+1)%4,cv=n,fragment=n,expert=n%384,rows=256,bytes=917504)))
    return out

def save(path, obj):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj),encoding="utf-8")

def run_files(root, schedule="unitflag-shared-fixub-both-v5"):
    save(root/"rank_0/manifest.json",dict(rank=0,world=4,compute_schedule=schedule,cv_rows=256))
    save(root/"summary.json",[dict(bs=1024,profile=dict(rank_max_mean_us=946,rank_min_mean_us=941,rounds=20))])
    for bs in (2048,1024):save(root/f"bs_{bs}/coarse/allrank.json",document(bs))
    save(root/"bs_1024/fine/allrank.json",document(mode="fine",repeats=400))

@pytest.mark.parametrize("limit",[16000,60000])
def test_ascii_bound_and_route_details(packet,limit):
    text=packet.render(document(),document(mode="fine",repeats=400),label="中文",limit=limit)
    assert len(text)==len(text.encode())<=limit
    assert "RouteSummary" in text and "END URMA_RANK_PAIR_V1" in text

def test_run_directory_optional_fine_and_formal_profile(packet,tmp_path):
    run_files(tmp_path)
    outputs=packet.export_run(tmp_path,expected_ep=4,expected_schedule="unitflag-shared-fixub-both-v5")
    assert [p.name for p,t in outputs]==["ep4_bs1024.txt","ep4_bs2048.txt"]
    assert '"rank_max_mean_us":946' in outputs[0][1]
    assert "NO_FINE" in outputs[1][1]
    assert "not inferred from current checkout" in outputs[0][1]
    for path,text in outputs:assert path.read_text(encoding="ascii")==text

def test_refuses_wrong_native_before_writing(packet,tmp_path):
    run_files(tmp_path,"unitflag-shared-aiv0-gmm2-v7")
    with pytest.raises(ValueError,match="Capture schedule"):
        packet.export_run(tmp_path,expected_schedule="unitflag-shared-fixub-both-v5")
    assert not (tmp_path/"comment_packets").exists()

def test_refuses_wrong_ep_and_mixed_bs(packet,tmp_path):
    run_files(tmp_path)
    with pytest.raises(ValueError,match="expected EP8"):packet.export_run(tmp_path,expected_ep=8)
    save(tmp_path/"bs_1024/fine/allrank.json",document(2048,"fine"))
    with pytest.raises(ValueError,match="bs mismatch"):packet.export_run(tmp_path)
    assert not (tmp_path/"comment_packets").exists()
