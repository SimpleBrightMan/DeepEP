from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parent))
from audit import validate_case


def sample():
    reports=[dict(rank=rank,checked_token_count=32,total_tokens=32,check_tokens=0,elements=32*7168,
        diff_thd=0.01,pct_thd=0.05,max_diff=0.1,passed=True,counts_equal=True,
        cumulative_counts_equal=True,finite=True,bad_elements=0,max_mixed_error=0.002,
        max_abs_error=0.002,relative_rmse=0.001) for rank in range(8)]
    return dict(reports=reports,repeat_equal=[True]*8)


def test_complete_full_case():
    assert validate_case(sample(),8,32)['checked_elements']==8*32*7168


@pytest.mark.parametrize('key,value',[('check_tokens',8),('counts_equal',False),
    ('cumulative_counts_equal',False),('diff_thd',0.02),('max_mixed_error',0.11),('relative_rmse',float('nan'))])
def test_no_sample_or_relaxation(key,value):
    data=sample();data['reports'][3][key]=value
    with pytest.raises(ValueError):validate_case(data,8,32)


def test_missing_rank_and_repeat_failure():
    data=sample();data['reports'].pop()
    with pytest.raises(ValueError):validate_case(data,8,32)
    data=sample();data['repeat_equal'][2]=False
    with pytest.raises(ValueError):validate_case(data,8,32)
