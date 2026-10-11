import pytest

from experiments.prepare_coarse_resume import closed_target_stats


def _data():
    rows=[dict(anchor=0,parent=1,side=0,flips=[7],target_fibers_decided=84,statuses={'UNSAT_FIXED_SIGNED_C_A':84})]
    records=[dict(anchor=0,parent=1,side=0,flips=[7],k=k,eta=eta,complete=True,
                  status='UNSAT_FIXED_SIGNED_C_A',words=None,unknown_sign_fibers=0,
                  support_orbits=2,sign_fibers=3,assignments=5,coarse_witness_supports=0)
             for k in range(1,43) for eta in (-1,1)]
    return rows,records


def test_only_whole_closed_prefix_is_credited():
    rows,records=_data()
    # Partially visited following anchor remains wholly uncredited.
    records.append(dict(anchor=1,status='UNKNOWN_FIXED_SIGNED_C_A'))
    stats=closed_target_stats(rows,records)
    assert stats==dict(target_fibers_decided=84,support_orbits=168,sign_fibers=252,assignments=420,coarse_witness_supports=0)


def test_inherited_closed_anchor_stats_are_combined_without_replaying_signs():
    rows,records=_data();original=closed_target_stats(rows,records)
    old=dict(rows[0],closed_prefix_parent='/audited/parent',proof_target_stats=dict(original))
    fresh=dict(rows[0],anchor=1);fresh.pop('proof_target_stats')
    for record in records:record['anchor']=1
    stats=closed_target_stats([old,fresh],records)
    assert stats=={key:2*value for key,value in original.items()}
    old.pop('proof_target_stats')
    with pytest.raises(ValueError,match='audited original'):closed_target_stats([old,fresh],records)


@pytest.mark.parametrize('kind',('unknown','partial','sat','missing','reordered','literal'))
def test_invalid_prefix_evidence_is_rejected(kind):
    rows,records=_data()
    if kind=='unknown':records[10]['status']='UNKNOWN_FIXED_SIGNED_C_A'
    elif kind=='partial':records[10]['complete']=False
    elif kind=='sat':records[10]['words']=((0,),(1,))
    elif kind=='missing':records.pop()
    elif kind=='reordered':records[10],records[11]=records[11],records[10]
    else:records[10]['flips']=[8]
    with pytest.raises(ValueError):closed_target_stats(rows,records)
