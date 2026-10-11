"""The modulated difference norm is an exact transformed coarse norm."""
import pytest
from dataclasses import replace
import gzip
import json
from pathlib import Path
import subprocess
import sys

from experiments.extract_difference_carrier_anchors import difference_coarse_target
from solver.structured_energy import fold_profile,target_profile
from solver.odd_lift_native import difference_sign_problem
from solver.projection_group_completion import complete_projection_group
from solver.target_profiles import TargetContentProfile
from solver.ternary_support_search import support_mask


@pytest.mark.parametrize('d',(3,5,11,43,47))
def test_every_strict_shift_and_sign_has_the_exact_transformed_profile(d):
    for k in range(1,d):
        for eta in (-1,1):
            full=target_profile(2*d,k,eta)
            wanted=tuple((-1 if u%2 else 1)*(full[u]-full[u+d])//4 for u in range(d))
            transformed=difference_coarse_target(2*d,k,eta)
            coarse=tuple(v//4 for v in fold_profile(target_profile(transformed['L'],transformed['k'],transformed['eta']),2))
            assert wanted==coarse
            assert coarse==(d,)+tuple(transformed['eta']*int(u in (k,d-k)) for u in range(1,d))


@pytest.mark.parametrize('args',((4,1,1),(86,43,1),(86,0,1),(86,1,0),(86,True,1),(86,1,True)))
def test_invalid_and_half_shift_cases_are_not_silently_transferred(args):
    with pytest.raises(ValueError):difference_coarse_target(*args)


def test_actual_g_positive_c_negative_control_is_harvested_and_reaudited(tmp_path):
    # Literal audited C witness; alternating full binary modulation swaps
    # C and the modulated difference H. This is a control, not a discovery.
    symbols=('00-0-+--+0-00+0++00+00-++0-0-++++0++0-00-+0',
             '0+-000+0--0000+0000++--0+0++0+-0000000+-+00')
    words=tuple(tuple({'0':0,'+':1,'-':-1}[v] for v in row) for row in symbols)
    target=TargetContentProfile(86,2,-1,17,21,17,22)
    original_masks=tuple(support_mask(w) for w in words)
    masks,_,negative=difference_sign_problem(original_masks,target)
    alt=replace(target,a_odd_ones=43-target.a_odd_ones,b_odd_ones=43-target.b_odd_ones,
                eta=target.eta*(-1 if target.k%2 else 1))
    refs=tuple(tuple([-1]*n+[0]*(43-bin(m).count('1'))+[1]*(bin(m).count('1')-n)) for m,n in zip(masks,negative))
    result=complete_projection_group(*refs,(alt,),masks,difference_first=True)
    assert result['status']=='UNSAT_BINARY_SUPPORT_GROUP'
    assert result['attempts']['coarse']['status']=='UNSAT_FIXED_SUPPORT'
    assert result['attempts']['lifts'][0]['result']['status']=='SAT_COMPRESSED'
    source=tmp_path/'source';source.mkdir()
    record=dict(trial=0,u=masks[0],sampling=dict(mask=masks[1]),result=result)
    with gzip.open(source/'literal_trials.jsonl.gz','wt') as stream:stream.write(json.dumps(record)+'\n')
    (source/'report.json').write_text(json.dumps(dict(complete=True,all_sampled_supports_decided=True,
        tested=1,trace_file='literal_trials.jsonl.gz',scope='Constructed mathematical positive control only.')))
    root=Path(__file__).resolve().parents[1];output=tmp_path/'harvest'
    subprocess.run([sys.executable,str(root/'experiments/extract_difference_carrier_anchors.py'),
                    '--input',str(source),'--output',str(output)],check=True,capture_output=True)
    report=json.loads((output/'report.json').read_text())
    assert report['actual_frozen_execution'] and report['complete']
    assert report['audited_g_carriers']==report['original_c_unsat_carriers']==1
    assert report['new_radius_zero_anchor_points']==2
    from experiments.complete_l86_full_rows import audited_coarse_centers
    seeds=audited_coarse_centers(output/'coarse_witnesses.json',0)['seeds']
    assert len(seeds)==2
