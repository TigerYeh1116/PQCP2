from copy import deepcopy
from math import comb

import pytest

from experiments.parallel_two_half_remaining import closed_parent_prefix


def parent_fixture():
    count=12*sum(comb(43,r) for r in range(5))
    rows=[dict(center_index=i,generated_triples=count,content_excluded=count-19,
               identical_triples=8,eligible_jobs=11,witnesses=0) for i in range(2)]
    return dict(radius=4,completed_centers=2,centers=8,rows=rows,eligible_jobs=22),dict(seeds=[None]*8)


def test_only_contiguous_fully_decided_parent_prefix_can_be_credited():
    parent,inputs=parent_fixture();assert closed_parent_prefix(parent,inputs)==2
    for field,value in (('generated_triples',0),('center_index',4),('eligible_jobs',12)):
        altered=deepcopy(parent);altered['rows'][1][field]=value
        with pytest.raises(RuntimeError):closed_parent_prefix(altered,inputs)
    altered=deepcopy(parent);altered['eligible_jobs']+=2000
    with pytest.raises(RuntimeError):closed_parent_prefix(altered,inputs)
    altered=deepcopy(parent);altered['rows'][0]['witnesses']=1
    with pytest.raises(ValueError):closed_parent_prefix(altered,inputs)
