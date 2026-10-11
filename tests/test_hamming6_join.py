from itertools import product
from threading import Event

from solver.correlation import full_correlation_profile
from solver.hamming6_join import exact_four_two_join,row_four_flip_count,row_four_vectors
from solver.interleaving import periodic_sign_profile
from solver.structured_energy import target_profile
from solver.target_profiles import pair_content,target_content_profiles


def test_stream_preserves_every_small_four_flip_preimage_and_exact_delta():
    rows = tuple(product((0,1),repeat=6))
    for row in rows:
        base = periodic_sign_profile(tuple(1-2*v for v in row))
        expected = {r for r in rows if sum(a!=b for a,b in zip(row,r))==4
                    and (sum(r[::2]),sum(r[1::2]))==(sum(row[::2]),sum(row[1::2]))}
        actual=set()
        for mask,delta in row_four_vectors(row,batch_size=2):
            word=tuple((mask>>i)&1 for i in range(6));actual.add(word)
            profile=periodic_sign_profile(tuple(1-2*v for v in word))
            assert delta==bytes(128+(x-y)//4 for x,y in zip(profile[1:4],base[1:4]))
        assert actual==expected


def test_four_two_join_matches_literal_small_neighborhood_and_cancellation():
    rows=tuple(product((0,1),repeat=6))
    for center in (((0,0,0,0,1,1),(0,0,1,1,0,1)), ((0,1,0,1,1,0),(1,0,1,0,0,1))):
        content=pair_content(*center)
        desired={target_profile(6,t.k,t.eta) for t in target_content_profiles(6) if
            (t.a_even_ones,t.a_odd_ones,t.b_even_ones,t.b_odd_ones)==content}
        possible={(a,b) for a,b in product(rows,repeat=2) if pair_content(a,b)==content
            and tuple(sum(x!=y for x,y in zip(old,new)) for old,new in zip(center,(a,b))) in ((2,4),(4,2))
            and tuple(full_correlation_profile(a,b)) in desired}
        out=exact_four_two_join(*center)
        assert {w['words'] for w in out['witnesses']}==possible and out['complete']
    stop=Event();stop.set()
    out=exact_four_two_join((0,0,0,0,1,1),(0,0,1,1,0,1),stop_event=stop)
    assert out['status']=='UNKNOWN' and not out['complete']


def test_large_stream_first_last_masks_and_deltas_cross_native_word_boundary():
    for length in (66,86,94):
        row=tuple(int(i%4<2) for i in range(length))
        base=periodic_sign_profile(tuple(1-2*v for v in row))
        total=row_four_flip_count(row)
        selected=[]
        count=0
        for index,(mask,delta) in enumerate(row_four_vectors(row,batch_size=8192)):
            count+=1
            if index<3 or index>=total-3:selected.append((mask,delta))
        assert count==total
        for mask,delta in selected:
            word=tuple((mask>>i)&1 for i in range(length))
            assert sum(a!=b for a,b in zip(row,word))==4
            assert (sum(word[::2]),sum(word[1::2]))==(sum(row[::2]),sum(row[1::2]))
            profile=periodic_sign_profile(tuple(1-2*v for v in word))
            assert delta==bytes(128+(x-y)//4 for x,y in zip(profile[1:length//2+1],base[1:length//2+1]))
