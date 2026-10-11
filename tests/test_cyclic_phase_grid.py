"""Literal weighted ranks, rotation phases and SIMD fibre word boundaries."""
import ctypes
from itertools import product
from random import Random

import pytest

from solver.cyclic_support import iter_family, parity_support_family
from solver.cyclic_support_native import _library
from solver.cyclic_support_sampling import sample_weighted_support


@pytest.mark.parametrize('nc',(1,4,5,63,64,65,128,129))
def test_complete_native_grid_and_translated_ranks_against_literal_products(nc):
    rng=Random(860200+nc)
    groups=tuple(tuple(rng.getrandbits(43) for _ in range(n)) for n in (3,5,nc))
    base=rng.getrandbits(43);weight=21;na,nb=3,5
    expected=[(i,j,k) for i,j,k in product(range(na),range(nb),range(nc))
              if bin(base^groups[0][i]^groups[1][j]^groups[2][k]).count('1')==weight]
    u=ctypes.c_uint64;up=ctypes.POINTER(u);integer=ctypes.c_int
    grid=(u*(na*nb*((nc+63)//64)))();count=u()
    fn=_library().pqcp_weighted_support_grid
    fn.argtypes=[up,integer]*3+[u,integer,up,u,up];fn.restype=integer
    arrays=tuple((u*len(g))(*g) for g in groups)
    assert fn(arrays[0],na,arrays[1],nb,arrays[2],nc,base,weight,grid,len(grid),ctypes.byref(count))==0
    actual=[(i,j,k) for i,j,k in product(range(na),range(nb),range(nc))
            if grid[(i*nb+j)*((nc+63)//64)+k//64]>>(k%64)&1]
    assert actual==expected and count.value==len(expected)
    select=_library().pqcp_weighted_grid_at
    select.argtypes=[up,integer,integer,integer,integer,integer,integer,u,up];select.restype=integer
    for sa,sb,sc in product((0,2),(0,4),sorted({0,nc-1,min(63,nc-1),min(64,nc-1)})):
        translated=sorted(expected,key=lambda x:((x[0]+sa)%na,(x[1]+sb)%nb,(x[2]+sc)%nc))
        for rank,indices in enumerate(translated):
            ordinal=u()
            assert select(grid,na,nb,nc,sa,sb,sc,rank,ctypes.byref(ordinal))==1
            assert ordinal.value==(indices[0]*nb+indices[1])*nc+indices[2]
        assert select(grid,na,nb,nc,sa,sb,sc,len(expected),ctypes.byref(u()))==0


@pytest.mark.parametrize('d',(3,5,11))
def test_all_literal_small_weighted_ranks_and_metadata_are_unchanged(d):
    for u in range(1<<d) if d<11 else range(0,1<<d,127):
        for k in range(1,d//2+1):
            family=parity_support_family(u,d,k);literal=list(iter_family(family))
            for weight in range(d+1):
                expected=[v for v in literal if bin(v).count('1')==weight]
                class Pick:
                    def __init__(self,index):self.index=index
                    def randrange(self,count):assert count==len(expected);return self.index
                for index,mask in enumerate(expected):
                    old=sample_weighted_support(family,weight,Pick(index),phase_count_pruning=True,phase_rank_pruning=False)
                    new=sample_weighted_support(family,weight,Pick(index),phase_count_pruning=True,phase_rank_pruning=True)
                    assert new==old and new['mask']==mask
                if not expected:
                    assert sample_weighted_support(family,weight,Pick(0),phase_count_pruning=True,phase_rank_pruning=True)['mask'] is None


@pytest.mark.parametrize('cancel',(0,1,2,3))
def test_l43_zero_norm_components_first_phase_position_and_rng(cancel):
    initial=parity_support_family(0,43,1)
    u=0
    for group in initial.choices[:cancel]:u^=group[0]
    family=parity_support_family(u,43,1)
    assert sum(len(g)==1 for g in family.choices)==cancel
    for weight in (0,7,15,19,23,27,35,43):
        for seed in (0,43,97):
            old_rng,new_rng=Random(seed),Random(seed)
            old=sample_weighted_support(family,weight,old_rng,phase_count_pruning=True,phase_rank_pruning=False)
            new=sample_weighted_support(family,weight,new_rng,phase_count_pruning=True,phase_rank_pruning=True)
            assert new==old and old_rng.getstate()==new_rng.getstate()


def test_explicit_invalid_phase_grid_controls_are_rejected():
    family=parity_support_family(0,43,1)
    with pytest.raises(ValueError):sample_weighted_support(family,19,Random(1),phase_rank_pruning=True)
    for weight in (-1,44,True):
        with pytest.raises(ValueError):sample_weighted_support(family,weight,Random(1),phase_count_pruning=True,phase_rank_pruning=True)
