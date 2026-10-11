from itertools import product

import torch

from solver.torch_joint_search import joint_transfer_trials
from solver.torch_joint_transfer import direct_transfer_draw


def test_direct_sampler_covers_every_feasible_small_transfer_without_invalid_positions():
    c, g = ((1, -1, 1, 0, 0), (1, -1, 0, 0, 0)), ((0, 0, 0, 1, -1), (0, 0, 1, 1, -1))
    tensors = tuple(torch.tensor([row], dtype=torch.float32) for row in (c, g))
    lower, upper = torch.tensor([[0, 0]]), torch.tensor([[2, 2]])
    # Literal position source plus both new sign orientations.
    raw = torch.tensor([[(di+.1)/2, (p+.1)/5, (q+.1)/5, .5,
                         (pp+.1)/5, (qq+.1)/5, (sc+.1)/2, (sg+.1)/2]
        for di, p, q, pp, qq, sc, sg in product(range(2), range(5), range(5), range(5), range(5), range(2), range(2))])[None]
    original = joint_transfer_trials(*tensors, raw, lower, upper)
    direct = joint_transfer_trials(*tensors, direct_transfer_draw(*tensors, raw, lower, upper), lower, upper)
    def states(out):
        cs, gs, _, _, legal = out
        return {(tuple(map(tuple, a)), tuple(map(tuple, b))) for a, b, valid
                in zip(cs[0].int().tolist(), gs[0].int().tolist(), legal[0].tolist()) if valid}
    assert bool(direct[-1].all())
    assert states(direct) == states(original)


def test_direct_sampler_respects_content_boundary_and_empty_move_graph():
    c = torch.tensor([[[1., -1., 0.], [1., 0., 0.]]])
    g = torch.tensor([[[0., 0., 1.], [0., 1., -1.]]])
    draw = torch.rand(1, 32, 8, generator=torch.Generator().manual_seed(14))
    for lower, upper, expected in (([[0, 0]], [[1, 1]], True), ([[1, 0]], [[1, 0]], False)):
        lo, hi = torch.tensor(lower), torch.tensor(upper)
        out = joint_transfer_trials(c, g, direct_transfer_draw(c, g, draw, lo, hi), lo, hi)
        assert bool(out[-1].all()) if expected else not bool(out[-1].any())
