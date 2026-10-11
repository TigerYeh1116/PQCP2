"""Direct legal source-color sampling for existing opposite content moves.

This retains the same move graph: source colors are c+/c- and g+/g-, rather
than arbitrary positions subsequently rejected for having the wrong colors.
Proposal probabilities are heuristic; no Gibbs/stationarity claim is made.
"""
import torch


def direct_transfer_draw(c, g, draw, lower, upper):
    batch, _, d = c.shape
    count = draw.shape[1]
    nc = (c == -1).sum(-1)
    create_a = (nc[:, 0] < upper[:, 0]) & (nc[:, 1] > lower[:, 1])
    create_b = (nc[:, 0] > lower[:, 0]) & (nc[:, 1] < upper[:, 1])
    # Within a valid contiguous content range, a feasible direction has a
    # g+/g- donor in the creator and c+/c- donor in the other row.
    for words, feasible, side in ((g, create_a, 0), (c, create_a, 1),
                                   (c, create_b, 0), (g, create_b, 1)):
        feasible &= (words[:, side] == 1).any(-1) & (words[:, side] == -1).any(-1)
    selected_a = torch.where(create_a[:, None] & create_b[:, None], draw[..., 0] < .5, create_a[:, None])
    create = torch.stack((selected_a, ~selected_a), -1)
    source = torch.where(create[..., None], g[:, None], c[:, None])
    transformed = draw.clone()
    transformed[..., 0] = torch.where(selected_a, .25, .75)
    for sign, columns in ((1, (1, 4)), (-1, (2, 5))):
        mask = source == sign
        size = mask.sum(-1)
        uniform = torch.stack(tuple(draw[..., column] for column in columns), -1)
        rank = (uniform*size).long()
        selected = mask & (mask.long().cumsum(-1) == rank[..., None]+1)
        position = selected.long().argmax(-1)
        for side, column in enumerate(columns):
            transformed[..., column] = (position[..., side].to(draw.dtype)+.5)/d
    # If neither direction exists, source selection remains a no-op/illegal;
    # the original exact legality checks still decide every proposed move.
    return transformed
