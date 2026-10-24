"""CPU检查：区间并集不重复扣除重叠ISSUE，累计量与窗口量保持不同语义。"""
import unittest
from analyze_trace import analyze, union_length


class TraceTests(unittest.TestCase):
    def test_union(self):
        self.assertEqual(union_length([]),0)
        self.assertEqual(union_length([(10,20),(15,25),(30,40)]),25)

    def test_actual_analysis(self):
        events=[]
        for group in range(32):
            def add(name,start,end,task=None):
                events.append(dict(name=name,group_id=group,start_cycle=start,end_cycle=end,task=task or {}))
            add('KERNEL_TIMING',1,51)
            add('AIC_GMM1_ISSUE_0',10,20);add('AIC_GMM1_ISSUE_1',15,25)
            add('AIC_GMM2_ISSUE_0',30,40)
            for family in ('AIC_GMM1_WAIT','AIC_GMM2_WAIT','AIV_SWIGLU_ISSUE','AIV_DISPATCH_ISSUE'):
                add(family+'_0',5,6)
            add('AIV_COMBINE_ISSUE_0',35,39,dict(row_count=4))
        metrics=analyze(events,1)
        self.assertEqual(metrics['aic_issue_window_mean_us'],30)
        self.assertEqual(metrics['aic_outside_issue_mean_us'],5)
        self.assertEqual(metrics['AIC_GMM1_ISSUE']['sum_per_lane_us'],20)
        self.assertEqual(metrics['combine_total_owned_rows'],128)
        self.assertEqual(metrics['native_kernel_envelope_us'],50)


if __name__=='__main__':unittest.main()
