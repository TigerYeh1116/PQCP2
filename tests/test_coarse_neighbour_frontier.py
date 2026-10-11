from itertools import product
import json
from pathlib import Path

import pytest

from experiments.complete_l86_full_rows import audited_coarse_points, audited_coarse_centers, anchor_candidates
from experiments.run_l86_coarse_neighbours import neighbour_frontier
from solver.coarse_character import admissible_coarse_characters, coarse_all_target_psd_obstruction
from solver.coarse_row_completion import coarse_target
from solver.coarse_row_completion import complete_fixed_coarse_row_target
from solver.structured_energy import integer_periodic_autocorrelation_profile as pacf


@pytest.mark.parametrize('d', (3, 5))
def test_character_filter_equals_existing_target_obstructions_for_all_small_trits(d):
    for row in product((-1, 0, 1), repeat=d):
        allowed = admissible_coarse_characters(row)
        for k in range(1, d):
            for eta in (-1, 1):
                assert ((eta, k % 2) in allowed) == (coarse_target(row, k, eta) is not None)


def test_no_full_length_six_solution_is_lost_by_character_filter():
    # Independent binary words and exact PACF, no completion solver.
    words = tuple(product((-1, 1), repeat=6))
    hits = 0
    for a in words:
        for b in words:
            profile = tuple(x+y for x, y in zip(pacf(a), pacf(b)))
            for k in (1, 2):
                for eta in (-1, 1):
                    if profile == (12,)+tuple(4*eta*int(u in (k, 6-k)) for u in range(1, 6)):
                        c = tuple((a[2*j % 6]+a[(2*j+3) % 6])//2 for j in range(3))
                        assert (eta, k % 2) in admissible_coarse_characters(c)
                        assert coarse_all_target_psd_obstruction(c) is None
                        hits += 1
    assert hits > 0


def test_spectral_exclusions_agree_with_independent_complex_fourier_values():
    from cmath import exp
    from math import pi
    from random import Random
    rng = Random(20261011)
    rows = list(product((-1, 0, 1), repeat=5))
    rows += [tuple(rng.choice((-1, 0, 1)) for _ in range(43)) for _ in range(128)]
    excluded = 0
    for row in rows:
        proof = coarse_all_target_psd_obstruction(row)
        if proof:
            f, d = proof['frequency'], len(row)
            psd = abs(sum(v*exp(2j*pi*f*j/d) for j, v in enumerate(row)))**2
            assert psd > d+2+1e-8
            assert proof['upper_bound_numerator'] < 0
            assert proof['maximum_pair_psd'] == d+2
            excluded += 1
    assert excluded > 0


def test_all_actual_compressed_carriers_survive_universal_spectral_bound():
    root = Path(__file__).resolve().parents[1]
    bank = root/'results/L86_phase_grid_default_original_ten_million_20261009/coarse_witnesses.json'
    seeds = audited_coarse_centers(bank, 0)['seeds']
    assert len(seeds) == 427
    for seed in seeds:
        for row in seed['coarse_rows']:
            assert coarse_all_target_psd_obstruction(row) is None


def test_shared_solver_envelope_preserves_all_l6_decisions_and_models():
    for row in product((-1, 0, 1), repeat=3):
        for k, eta in product((1, 2), (-1, 1)):
            a = complete_fixed_coarse_row_target(row, k=k, eta=eta)
            b = complete_fixed_coarse_row_target(row, k=k, eta=eta, all_target_psd=True)
            assert (a['status'], a['complete'], a['words']) == (b['status'], b['complete'], b['words'])


def test_cached_spectral_certificate_cannot_be_changed_by_caller():
    row = (1,)*21+(-1,)*19+(0,)*3
    proof = coarse_all_target_psd_obstruction(row)
    assert proof is not None
    proof['upper_bound_numerator'] = 0
    assert coarse_all_target_psd_obstruction(list(row))['upper_bound_numerator'] < 0


def test_character_l43_row_sums_and_parities_are_exact():
    seen = set()
    for weight in range(44):
        for minus in range(weight+1):
            row = (-1,)*minus+(1,)*(weight-minus)+(0,)*(43-weight)
            allowed = admissible_coarse_characters(row)
            for k in (1, 2):
                for eta in (-1, 1):
                    assert ((eta, k % 2) in allowed) == (coarse_target(row, k, eta) is not None)
            if allowed:
                seen.add((abs(sum(row)), allowed[0][0]))
    assert seen == {(3, 1), (6, 1), (4, -1), (5, -1)}


def test_every_literal_shell_leaf_generated_before_closed_and_equivalence_filtering():
    # A deliberately coarse key maps the two centres together but retains
    # distinct leaf sums. Deduplicating centres first would lose sum 2.
    centres = ((-1, 0, 0), (0, 0, 1))
    def key(row):
        return 0 if row in centres else row
    points, stats = neighbour_frontier([dict(coarse_rows=centres)], {0: 'closed'}, key_function=key)
    expected = {row for centre in centres for row in product((-1, 0, 1), repeat=3)
                if sum(abs(a-b) for a, b in zip(row, centre)) == 1
                and admissible_coarse_characters(row) and key(row) != 0}
    assert {tuple(p['row']) for p in points} == expected
    assert any(sum(p['row']) == 2 for p in points)
    assert stats['literal_centres'] == 2
    # Adding a closed leaf only removes that leaf, never its neighbours.
    leaf = tuple(points[0]['row'])
    remaining, _ = neighbour_frontier([dict(coarse_rows=centres)], {0: 'closed', key(leaf): 'closed'}, key_function=key)
    assert {tuple(p['row']) for p in remaining} == expected-{leaf}


def test_literal_point_input_needs_no_fictitious_compressed_pair(tmp_path):
    path = tmp_path/'points.json'
    row = (1,)*3+(0,)*40
    path.write_text(json.dumps(dict(format='literal_coarse_points_v1', planned_radius=0,
                                   points=[dict(row=row, provenance={'shell': 1})])))
    seeds = audited_coarse_points(path, 0)['seeds']
    anchors = list(anchor_candidates(seeds[0], 'coarse-row', 0, 0))
    assert len(anchors) == 1 and anchors[0][3] == row
    packet = json.loads(path.read_text())
    packet['points'][0]['row'][0] = True
    path.write_text(json.dumps(packet))
    with pytest.raises(ValueError):
        audited_coarse_points(path, 0)


def test_two_step_shell_equals_all_literal_small_rows_at_exact_distance_two():
    for centre in product((-1,0,1),repeat=3):
        points,_=neighbour_frontier([dict(coarse_rows=(centre,))],{},key_function=lambda row:row,
                                    spectral_pruning=False,radius=2)
        expected={row for row in product((-1,0,1),repeat=3)
                  if sum(abs(a-b) for a,b in zip(row,centre))==2 and admissible_coarse_characters(row)}
        assert {tuple(x['row']) for x in points}==expected
        assert all(sum(abs(old-new) for _,old,new in p['provenance']['edits'])==2 for p in points)
