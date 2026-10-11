from itertools import product
from threading import Event

from solver.correlation import full_correlation_profile
from solver.hamming8_join import exact_four_four_join
from solver.structured_energy import target_profile
from solver.target_profiles import pair_content,target_content_profiles


def test_prefix_filtered_four_four_join_retains_all_literal_small_solutions():
    rows=tuple(product((0,1),repeat=6))
    for center in (((0,0,0,0,1,1),(0,0,1,1,0,1)), ((0,1,0,1,1,0),(1,0,1,0,0,1))):
        content=pair_content(*center)
        desired={target_profile(6,t.k,t.eta) for t in target_content_profiles(6) if
            (t.a_even_ones,t.a_odd_ones,t.b_even_ones,t.b_odd_ones)==content}
        possible={(a,b) for a,b in product(rows,repeat=2) if pair_content(a,b)==content
            and all(sum(x!=y for x,y in zip(old,new))==4 for old,new in zip(center,(a,b)))
            and tuple(full_correlation_profile(a,b)) in desired}
        for prefix in (1,2,3):
            out=exact_four_four_join(*center,prefix=prefix,stop_after_hit=False)
            assert out['complete'] and {w['words'] for w in out['witnesses']}==possible
    stop=Event();stop.set()
    out=exact_four_four_join((0,0,0,0,1,1),(0,0,1,1,0,1),prefix=2,stop_event=stop)
    assert out['status']=='UNKNOWN' and not out['complete']
