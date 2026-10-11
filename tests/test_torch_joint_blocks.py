from itertools import product

import torch

from solver.correlation import full_correlation_profile
from solver.target_profiles import TargetContentProfile,canonical_target_content_profiles,pair_content
from solver.torch_joint_blocks import exact_joint_block,joint_integer_energy
from solver.torch_joint_search import search_joint
from threading import Event


def literal_block_candidates(center,target,positions):
    d=target.L//2
    locations=[(side,p) for side,ps in enumerate(positions) for i in ps for p in (i,i+d)]
    content=pair_content(*center)
    for bits in product((0,1),repeat=len(locations)):
        rows=[list(w) for w in center]
        for (side,p),value in zip(locations,bits):rows[side][p]=value
        words=tuple(map(tuple,rows))
        if pair_content(*words)!=content or full_correlation_profile(*words)[d]!=0:continue
        yield words


def audit_matrix(residual,da,db,scores,*args):
    assert da.dtype==db.dtype==scores.dtype==torch.int32
    da,db=da.cpu().to(torch.int64),db.cpu().to(torch.int64)
    values=da[:,None,:]+db[None,:,:]+torch.tensor(residual,dtype=torch.int64)
    expected=values.square().sum(-1)
    assert torch.equal(scores.cpu(),expected.to(torch.int32))


def test_full_small_blocks_cover_literal_binary_space_and_all_zero_candidates():
    center=((0,0,0,0,1,1),(0,0,1,1,0,1))
    target=TargetContentProfile(6,2,-1,*pair_content(*center))
    positions=((0,1,2),(0,1,2))
    literal=list(literal_block_candidates(center,target,positions))
    expected=min(joint_integer_energy(w,target) for w in literal)
    zeros={w for w in literal if joint_integer_energy(w,target)==0}
    for temperature in (0.,1.):
        out=exact_joint_block(center,target,positions,device=torch.device('cpu'),temperature=temperature,matrix_audit=audit_matrix)
        assert out.complete and out.configurations==len(literal)
        assert out.minimum_joint_energy==expected and out.words in literal
        assert set(out.zero_words)==zeros


def test_partial_blocks_preserve_fixed_outside_bits_and_exact_gram_energy():
    profiles=canonical_target_content_profiles(14)
    stop=Event();stop.set()
    initial=search_joint(profiles,device=torch.device('cpu'),seed=14912,steps=1,stop_event=stop)
    for item in initial.candidates:
        center=(item['A'],item['B']);target=profiles[item['lane']]
        positions=((0,5),(1,6))
        literal=list(literal_block_candidates(center,target,positions))
        out=exact_joint_block(center,target,positions,device=torch.device('cpu'),matrix_audit=audit_matrix,temperature=2.)
        assert out.configurations==len(literal)
        assert out.minimum_joint_energy==min(joint_integer_energy(w,target) for w in literal)
        assert out.words in literal
