import unittest
from validate_trace_coverage import REQUIRED, validate


def fixture(groups=32):
    return [dict(group_id=g,core_type=r,sub_id=s,name=n) for g in range(groups) for r,s,n in REQUIRED]


class TraceCoverage(unittest.TestCase):
    def test_full_native(self):
        self.assertEqual(validate(fixture())['tracks'],96)

    def test_full_ds_aliases(self):
        events=[]
        for e in fixture():
            offset=0 if e['core_type']=='AIC' else e['sub_id']+1
            name=e['name']+'_ISSUE' if 'PROLOGUE' in e['name'] else e['name']
            events.append(dict(core=e['group_id']*3+offset,family=name))
        self.assertEqual(validate(events)['dynamic_groups'],32)

    def test_fixed_96_tracks_do_not_prove_dynamic_coverage(self):
        fixed=[dict(group_id=g,core_type=r,sub_id=s,name='KERNEL_TIMING') for g in range(32)
               for r,s in [('AIC',0),('AIV',0),('AIV',1)]]
        with self.assertRaisesRegex(ValueError,'Sparse dynamic'):
            validate(fixed+fixture(6))

    def test_missing_one_prologue_group_is_rejected(self):
        events=fixture()
        events=[e for e in events if not(e['group_id']==31 and e['name']=='AIV_GMM2_PROLOGUE')]
        with self.assertRaisesRegex(ValueError,'g31/AIV0/AIV_GMM2_PROLOGUE'):validate(events)

    def test_unexpected_core_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'Unexpected group'):validate([dict(core=96,family='KERNEL_TIMING')])


if __name__=='__main__':unittest.main()
