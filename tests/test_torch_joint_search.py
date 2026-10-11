"""Literal binary spaces, exact local increments, and complete zero auditing."""
from itertools import product
from threading import Event

import torch

from solver.correlation import full_correlation_profile
from solver.structured_energy import target_profile
from solver.target_profiles import TargetContentProfile, canonical_target_content_profiles, pair_content
from solver.torch_joint_search import binary_joint_rows, joint_binary_words, joint_swap_trials, joint_transfer_trials, search_joint
from solver.verifier import verify_pqcp


def encode(words):
    d = len(words[0])//2
    c, g = [], []
    for word in words:
        signs = tuple(1-2*v for v in word)
        c.append(tuple((signs[i]+signs[i+d])//2 for i in range(d)))
        g.append(tuple((-1 if i % 2 else 1)*(signs[i]-signs[i+d])//2 for i in range(d)))
    return tuple(c), tuple(g)


def ac(row):
    return tuple(sum(row[i]*row[(i+u) % len(row)] for i in range(len(row))) for u in range(len(row)))


def audit_pair(c, g, target):
    c, g = tuple(tuple(map(int, row)) for row in c), tuple(tuple(map(int, row)) for row in g)
    words = joint_binary_words(c, g, target)
    full = tuple(full_correlation_profile(*words))
    pc, pg = (tuple(x+y for x, y in zip(*(ac(w) for w in pair))) for pair in (c, g))
    d = target.L//2
    assert full[d] == 0
    for u in range(d):
        assert full[u] == 2*pc[u]+2*(-1 if u % 2 else 1)*pg[u]
        assert full[u+d] == 2*pc[u]-2*(-1 if u % 2 else 1)*pg[u]
    return pc, pg


def test_four_colors_parameterize_every_literal_half_shift_zero_binary_pair():
    rows = tuple(product((0, 1), repeat=6))
    count = 0
    for words in product(rows, repeat=2):
        full = full_correlation_profile(*words)
        if full[3] != 0:
            continue
        target = TargetContentProfile(6, 2, 1, *pair_content(*words))
        c, g = encode(words)
        assert binary_joint_rows(words, target) == (c, g)
        assert joint_binary_words(c, g, target) == words
        audit_pair(c, g, target)
        count += 1
    assert count == 1280


def test_all_small_color_swaps_preserve_contents_and_exact_profiles():
    words = ((0, 1, 0, 0, 1, 1), (0, 1, 1, 0, 0, 0))
    # These words satisfy half shift zero, but the test uses the literal PACF
    # rather than assuming they already match an assigned target.
    c, g = encode(words)
    target = TargetContentProfile(6, 2, 1, *pair_content(*words))
    pc, pg = audit_pair(c, g, target)
    moves = tuple(product(range(2), range(3), range(3)))
    draws = torch.tensor([[(side+.1)/2, (p+.1)/3, (q+.1)/3, .5] for side, p, q in moves])[None]
    out_c, out_g, dc, dg, legal = joint_swap_trials(torch.tensor([c], dtype=torch.float32),
        torch.tensor([g], dtype=torch.float32), draws)
    for i, (side, p, q) in enumerate(moves):
        assert bool(legal[0, i]) == ((c[side][p], g[side][p]) != (c[side][q], g[side][q]))
        new_c, new_g = out_c[0, i].tolist(), out_g[0, i].tolist()
        new_pc, new_pg = audit_pair(new_c, new_g, target)
        assert tuple(x+y for x, y in zip(pc, dc[0, i].tolist())) == new_pc
        assert tuple(x+y for x, y in zip(pg, dg[0, i].tolist())) == new_pg


def test_all_small_opposite_content_transfers_preserve_exact_binary_constraints():
    c, g = ((1, -1, 0), (1, 0, 0)), ((0, 0, 1), (0, 1, -1))
    target = TargetContentProfile(6, 2, -1, 1, 2, 1, 1)
    pc, pg = audit_pair(c, g, target)
    moves = tuple(product((0, 1), range(3), range(3), range(3), range(3), (0, 1), (0, 1)))
    draws = torch.tensor([[(create+.1)/2, (p+.1)/3, (q+.1)/3, .5,
                          (pp+.1)/3, (qq+.1)/3, (sc+.1)/2, (sg+.1)/2]
                         for create, p, q, pp, qq, sc, sg in moves])[None]
    out_c, out_g, dc, dg, legal = joint_transfer_trials(torch.tensor([c], dtype=torch.float32),
        torch.tensor([g], dtype=torch.float32), draws, torch.tensor([[0, 0]]), torch.tensor([[3, 3]]))
    checked = 0
    for i in range(len(moves)):
        if not bool(legal[0, i]):
            continue
        new_pc, new_pg = audit_pair(out_c[0, i].tolist(), out_g[0, i].tolist(), target)
        assert tuple(x+y for x, y in zip(pc, dc[0, i].tolist())) == new_pc
        assert tuple(x+y for x, y in zip(pg, dg[0, i].tolist())) == new_pg
        checked += 1
    assert checked > 0


def test_search_trace_preserves_constraints_and_zero_lanes_escape_export_cap():
    targets = canonical_target_content_profiles(6)
    profiles = tuple(targets[i % len(targets)] for i in range(32))
    trace = []
    out = search_joint(profiles, device=torch.device('cpu'), seed=6386, steps=16,
                       transfer_interval=2, export_limit=0, trace=trace)
    assert out.steps == 16 and len(trace) == 16
    for cs, gs, pc, pg in trace:
        for c, g, ac_c, ac_g, target in zip(cs, gs, pc, pg, profiles):
            exact_c, exact_g = audit_pair(c, g, target)
            assert tuple(ac_c) == exact_c and tuple(ac_g) == exact_g
    assert out.candidates
    for candidate in out.candidates:
        assert candidate['verified'] and verify_pqcp(candidate['A'], candidate['B']).is_valid
        target = profiles[candidate['lane']]
        assert candidate['profile'] == target_profile(6, target.k, target.eta)
    stop = Event(); stop.set()
    cancelled = search_joint(profiles, device=torch.device('cpu'), seed=6386, steps=16, stop_event=stop)
    assert cancelled.steps == 0 and cancelled.proposals == 0


def test_seeded_joint_search_keeps_verified_initial_zeros_even_when_cancelled():
    targets = canonical_target_content_profiles(6)
    initial = search_joint(targets*8, device=torch.device('cpu'), seed=6386, steps=16)
    zeros = [candidate for candidate in initial.candidates if candidate['verified']]
    assert zeros
    profiles = tuple((targets*8)[candidate['lane']] for candidate in zeros)
    words = tuple((candidate['A'], candidate['B']) for candidate in zeros)
    stop = Event(); stop.set()
    out = search_joint(profiles, device=torch.device('cpu'), seed=14, steps=16,
        initial_words=words, stop_event=stop, export_limit=0, initial_temperature=.5, final_temperature=.1)
    assert out.steps == 0 and len(out.candidates) == len(zeros)
    assert all(candidate['verified'] for candidate in out.candidates)
