"""Generate full CPU Golden; no torch/NPU initialization and bounded BLAS threads."""
import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import CASES,CONFIG,package,write

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--package-root',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--ep',type=int,choices=(8,32),required=True)
p.add_argument('--ranks',type=int,nargs='+',required=True)
p.add_argument('--workers',type=int,default=4)
p.add_argument('--blas-threads',type=int,default=4)
a=p.parse_args()
if a.workers<1 or a.blas_threads<1 or len(set(a.ranks))!=len(a.ranks):raise ValueError('Invalid worker/rank settings')
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[key]=str(a.blas_threads)
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
identity=package(a.package_root)
from deep_ep_experimental.megamoe import golden

records=[]
for routing,bs in CASES:
    print(f'GOLDEN_START EP{a.ep} {routing} BS{bs} ranks={a.ranks}',flush=True)
    spec=golden.prepare(CONFIG,a.ep,bs,a.ranks,a.output/'cache',input_kind='fp8',routing=routing,
        check_tokens=0,workers=a.workers,chunk_tokens=256)
    records.append(dict(routing=routing,bs=bs,contract=spec,
        caches={str(rank):str(golden.cache_path(a.output/'cache',spec,rank)) for rank in a.ranks}))
    print(f'GOLDEN_READY EP{a.ep} {routing} BS{bs}',flush=True)
write(a.output/f'prepared-ep{a.ep}-r{a.ranks[0]}.json',dict(ep=a.ep,ranks=a.ranks,identity=identity,
    workers=a.workers,blas_threads=a.blas_threads,cases=records,status='cpu_reference_ready'))
