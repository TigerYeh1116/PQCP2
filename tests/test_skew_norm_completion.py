from collections import defaultdict
from itertools import product

from solver.sds_norm_completion import cyclic_binary_norm
from solver.skew_norm_completion import skew_norm_preimages


def test_all_literal_small_skew_rows_and_all_symmetric_norm_targets():
    for d in (3,5,11):
        groups=defaultdict(set)
        for half in product((0,1),repeat=d//2):
            bits=(0,)+half+tuple(1-v for v in half[::-1])
            mask=sum(v<<i for i,v in enumerate(bits))
            groups[cyclic_binary_norm(mask,d)].add(mask)
        # Every symmetric F2 target, including empty fibers and both parities.
        for half in product((0,1),repeat=d//2):
            for origin in (0,1):
                desired=origin+sum(v*((1<<i)|(1<<(d-i))) for i,v in enumerate(half,1))
                out=skew_norm_preimages(desired,d)
                assert out['complete'] and set(out['masks'])==groups.get(desired,set())


def test_l43_nonempty_fibers_keep_all_coordinates_and_exact_norm():
    for mask in (sum(1<<i for i in range(1,22)),sum(1<<i for i in range(1,43) if pow(i,21,43)!=1)):
        desired=cyclic_binary_norm(mask,43)
        out=skew_norm_preimages(desired)
        assert out['complete'] and mask in out['masks'] and len(out['masks'])==8
        assert not out['rotation_quotient']
        for v in out['masks']:
            assert v&1==0 and bin(v).count('1')==21
            assert all(((v>>i)^(v>>(43-i)))&1==1 for i in range(1,43))
            assert cyclic_binary_norm(v,43)==desired
