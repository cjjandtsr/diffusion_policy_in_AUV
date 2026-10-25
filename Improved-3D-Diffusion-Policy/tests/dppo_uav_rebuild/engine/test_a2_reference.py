import numpy as np
from scripts.routea_paired_support.a2_reference import select_prefixes, check_execution


def test_selection_keeps_final_arrival_and_is_not_success_resampling():
    assert select_prefixes(1) == [0]
    selected = select_prefixes(60)
    assert len(selected) == len(set(selected)) == 8
    assert selected[0] == 0 and selected[-1] == 59


def test_partial_arrival_path_must_match_saved_plan():
    import pytest
    a=np.zeros((16,11),np.float32); a[2:6,0]=1
    before=np.array([0,0,2],np.float32)
    info=dict(done_reason='reached',event_segment=1,event_fraction=.5,
              executed_path=np.array([[0,0,2],[1,0,2],[1.5,0,2]],np.float32))
    check_execution(a,before,info)
    info['executed_path'][-1,0]=1.8
    with pytest.raises(ValueError): check_execution(a,before,info)
