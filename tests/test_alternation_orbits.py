from itertools import product
import random

import pytest

from solver.alternation_orbits import alternate_canonical_profile, canonical_counts, reduce_alternation_profile_orbits
from solver.correlation import full_correlation_profile
from solver.target_profiles import canonical_target_content_profiles, pair_content
from solver.torch_compressed_search import search_paired_contents


def test_profile_action_is_an_involution_and_never_removes_an_unrepresented_sign():
    for length in (6, 14, 44, 46, 68, 86, 94):
        originals = canonical_target_content_profiles(length)
        retained, mapping = reduce_alternation_profile_orbits(originals)
        for profile in originals:
            partner = alternate_canonical_profile(profile)
            assert alternate_canonical_profile(partner) == profile
            assert mapping[profile] in retained
            assert mapping[profile] in (profile, partner)
            assert reduce_alternation_profile_orbits((profile,))[0] == (profile,)
    old = canonical_target_content_profiles(86)
    new, mapping = reduce_alternation_profile_orbits(old)
    assert len(old) == 4 and len(new) == 3
    assert sum(map(lambda t: len(search_paired_contents(t)), old)) == 53
    assert sum(map(lambda t: len(search_paired_contents(t)), new)) == 40


def test_modulation_identity_on_every_literal_l6_binary_pair():
    words = tuple(product((0, 1), repeat=6))
    for a in words:
        aa = tuple(v ^ (i % 2) for i, v in enumerate(a))
        for b in words:
            bb = tuple(v ^ (i % 2) for i, v in enumerate(b))
            original = tuple(full_correlation_profile(a, b))
            modulated = tuple(full_correlation_profile(aa, bb))
            assert modulated == tuple((-1 if u % 2 else 1)*v for u, v in enumerate(original))
            counts = pair_content(a, b)
            assert pair_content(aa, bb) == (counts[0], 3-counts[1], counts[2], 3-counts[3])


def test_l86_binary_content_gauges_preserve_every_original_pacf():
    rng = random.Random(123)
    for target in canonical_target_content_profiles(86):
        counts = (target.a_even_ones, target.a_odd_ones, target.b_even_ones, target.b_odd_ones)
        words = []
        for even, odd in (counts[:2], counts[2:]):
            a, b = [1]*even+[0]*(43-even), [1]*odd+[0]*(43-odd)
            rng.shuffle(a); rng.shuffle(b)
            words.append(tuple(v for pair in zip(a, b) for v in pair))
        modulated = tuple(tuple(v ^ (i % 2) for i, v in enumerate(w)) for w in words)
        expected = tuple((-1 if u % 2 else 1)*v for u, v in enumerate(full_correlation_profile(*words)))
        assert tuple(full_correlation_profile(*modulated)) == expected
        reached = set()
        for ca, cb, ra, rb, swap in product((False, True), repeat=5):
            rows = [tuple(v ^ int(complement) for v in w) for w, complement in zip(modulated, (ca, cb))]
            rows = [w[1:]+w[:1] if rotate else w for w, rotate in zip(rows, (ra, rb))]
            if swap: rows.reverse()
            assert tuple(full_correlation_profile(*rows)) == expected
            reached.add(pair_content(*rows))
        partner = alternate_canonical_profile(target)
        partner_counts = (partner.a_even_ones, partner.a_odd_ones, partner.b_even_ones, partner.b_odd_ones)
        assert partner_counts == min(reached)


def test_odd_l86_retained_contents_match_under_the_literal_c_g_swap():
    negative, positive = canonical_target_content_profiles(86)[:2]
    transformed = set()
    for contents in search_paired_contents(negative):
        transformed.add(tuple((e-n, 43-z, o-n) for (n, z, p), (e, o) in zip(contents, ((16, 22), (18, 21)))))
    assert transformed == set(search_paired_contents(positive))


def test_real_production_profiles_and_rollback_keep_their_declared_target_pools(tmp_path):
    from solver.torch_search import TorchSearch, TorchSearchConfig
    from solver.torch_hybrid_runner import TorchHybridConfig, _new_continuous_search
    from main import parse_args
    old = TorchSearch(TorchSearchConfig(86, device='cpu', batch_size=4, symmetry_pruning=True), tmp_path, initialize=False)
    new = TorchSearch(TorchSearchConfig(86, device='cpu', batch_size=4, symmetry_pruning=True,
                                      alternation_orbit_pruning=True), tmp_path, initialize=False)
    assert len(old.available) == 4 and len(new.available) == 3
    args = parse_args(['--L', '86', '--torch-alternation-orbit-pruning'])
    assert args.torch_alternation_orbit_pruning
    hybrid = _new_continuous_search(TorchHybridConfig(86, device='cpu', root=tmp_path,
        continuous_batch_size=4, continuous_alternation_orbit_pruning=True))
    assert len(hybrid.available) == 3
    with pytest.raises(ValueError):
        TorchSearchConfig(86, alternation_orbit_pruning=True, symmetry_pruning=False)


def test_enabled_checkpoint_replays_and_missing_legacy_flag_keeps_the_old_pool(tmp_path):
    import json
    import torch
    from solver.torch_search import TorchSearch, TorchSearchConfig
    for enabled in (False, True):
        config=TorchSearchConfig(86,device='cpu',batch_size=4,symmetry_pruning=True,
                                 alternation_orbit_pruning=enabled,compression=False,
                                 steps_per_restart=20,stagnation_steps=20)
        search=TorchSearch(config,tmp_path)
        search.step()
        path=tmp_path/('checkpoint_'+str(enabled)+'.json')
        search.checkpoint(path)
        if not enabled:
            data=json.loads(path.read_text())
            del data['config']['alternation_orbit_pruning']
            path.write_text(json.dumps(data))
        replay=TorchSearch.resume(path,tmp_path)
        assert replay.config.alternation_orbit_pruning is enabled
        assert len(replay.available)==(3 if enabled else 4)
        assert replay.model.profiles==search.model.profiles
        search.step(); replay.step()
        assert torch.equal(search.theta,replay.theta)
