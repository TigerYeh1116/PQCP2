from collections import defaultdict
from itertools import product
from pathlib import Path
from random import Random

from experiments.complete_l86_full_rows import audited_coarse_centers
from solver.coarse_row_completion import coarse_target,complete_fixed_coarse_row_target
from solver.cyclic_support import cyclic_norm_family,rotation_representatives
from solver.cyclic_support_native import audit_literal_support_norms,signed_norm_supports
from solver.sds_norm_completion import cyclic_binary_norm


def test_all_small_supports_match_full_independent_python_norm_and_weight():
    for d in (3,5,11):
        groups=defaultdict(list)
        for mask in range(1<<d):groups[cyclic_binary_norm(mask,d),bin(mask).count('1')].append(mask)
        for (norm,weight),masks in groups.items():
            assert audit_literal_support_norms(masks,d,norm,weight)
            assert not audit_literal_support_norms(masks,d,norm^1,weight)
            if weight<d:assert not audit_literal_support_norms(masks,d,norm,weight+1)


def test_degree_84_product_and_all_high_bits_are_audited_at_l43():
    rng=Random(20261011)
    masks=[0,(1<<43)-1,1<<42,(1<<42)|1]+[rng.getrandbits(43) for _ in range(128)]
    for mask in masks:
        norm=cyclic_binary_norm(mask,43);weight=bin(mask).count('1')
        assert audit_literal_support_norms((mask,),43,norm,weight)
        assert not audit_literal_support_norms((mask,),43,norm^1,weight)


def test_actual_c_b_norm_families_retain_identical_complete_outputs_and_exclusions():
    root=Path(__file__).resolve().parents[1]
    seeds=audited_coarse_centers(root/'results/L86_phase_grid_default_original_ten_million_20261009/coarse_witnesses.json',2)['seeds']
    for seed in seeds:
        for row in seed['coarse_rows']:
            target=seed['original_target'];spec=coarse_target(row,target['k'],target['eta'])
            assert spec is not None
            wanted=spec['wanted_c_b'];desired=sum((v%2)<<u for u,v in enumerate(wanted))
            family=rotation_representatives(cyclic_norm_family(desired,43))
            a=signed_norm_supports(family,wanted,spec['c_b_negative'])
            b=signed_norm_supports(family,wanted,spec['c_b_negative'],native_audit=True)
            assert a==b and a['masks']


def test_all_new_completion_options_preserve_every_l6_decision_and_first_model():
    for row in product((-1,0,1),repeat=3):
        for k,eta in product((1,2),(-1,1)):
            args=dict(k=k,eta=eta,support_bounds=True,dihedral_support=True,unary_batch_free=12)
            a=complete_fixed_coarse_row_target(row,**args)
            b=complete_fixed_coarse_row_target(row,prebatch_dihedral=True,unary_linear_certificate=True,native_support_audit=True,**args)
            assert (a['status'],a['complete'],a['words'])==(b['status'],b['complete'],b['words'])
