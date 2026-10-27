"""核对单轮B与已保存历史A的原始证据，明确跨窗口而非同窗因果收益。"""
import argparse
import importlib.util
import json
from pathlib import Path
import statistics


def load(path):return json.loads(path.read_text(encoding='utf-8'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--current',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--reference-method',choices=('control','combined'),required=True)
    p.add_argument('--reference-repeats',nargs='+',choices=('a','b'),required=True)
    a=p.parse_args()
    spec=importlib.util.spec_from_file_location('audit',Path(__file__).with_name('quick')/'report.py')
    audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
    current,ref=load(a.current/'experiment.json'),load(a.reference/'experiment.json')
    if current['status']!='audited_screening' or current['ep']!=ref['ep']:
        raise ValueError('Current evidence must be audited single-start, same EP')
    ep=current['ep'];rows=[]
    def get(root,manifest,label):
        return audit.audit_run(root,label,ep,manifest['packages'],manifest['timer_requested'],
            manifest['execution_mode'],manifest['buffer_capacity'])
    for bs in current['bs_values']:
        b,ins,outs=get(a.current,current,f'bs{bs}-combined-a')
        refs=[]
        for repeat in a.reference_repeats:
            old,old_ins,old_outs=get(a.reference,ref,f'bs{bs}-{a.reference_method}-{repeat}')
            if old_ins!=ins or old_outs!=outs:raise ValueError(f'EP{ep} BS{bs}: history/current input or output mismatch')
            refs.append(old)
        mean=statistics.mean(r['mean_us'] for r in refs)
        rows.append(dict(bs=bs,reference_us=mean,candidate_us=b['mean_us'],
            historical_reduction_pct=100*(1-b['mean_us']/mean),
            reference_runs=refs,candidate=b,all_rank_inputs_outputs_equal=True))
    result=dict(ep=ep,status='audited_historical_comparison',comparison='Different windows; screening, not causal acceptance',
        current=str(a.current.resolve()),reference=str(a.reference.resolve()),
        candidate_revision=current['packages']['combined']['revision'],
        reference_revision=ref['packages'][a.reference_method]['revision'],
        candidate_placement=dict(rack=current['rack'],nodes=current['nodes']),
        reference_placement=dict(rack=ref['rack'],nodes=ref['nodes']),rows=rows,
        accuracy='No independent Golden; full per-rank inputs and output fingerprints equal')
    (a.current/'historical-comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    for r in rows:print(f"EP{ep} BS{r['bs']}: {r['reference_us']:.2f} -> {r['candidate_us']:.2f} us, historical {r['historical_reduction_pct']:+.2f}%")


if __name__=='__main__':main()
