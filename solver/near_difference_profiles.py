"""Complete smallest nonzero error shell with fixed sum and support parity.

For a fixed C support/content, G errors are even and their independent
coordinates sum to zero. The smallest nonzero squared norm is 8: +2 at one
coordinate and -2 at another. Enumerating this shell can find full energy-4
starts, or certify energy >=8 after exact G zero/shell UNSAT. Not a PQCP test.
"""


def minimum_difference_error_shell(wanted):
    wanted = tuple(wanted)
    d = len(wanted)
    if (not 3 <= d <= 47 or d % 2 != 1 or any(type(v) is not int for v in wanted)
            or any(wanted[u] != wanted[-u % d] for u in range(d))):
        raise ValueError('symmetric integral odd-length target required')
    for plus in range(1, d//2+1):
        for minus in range(1, d//2+1):
            if plus == minus:
                continue
            candidate = list(wanted)
            for u in (plus, d-plus):
                candidate[u] += 2
            for u in (minus, d-minus):
                candidate[u] -= 2
            yield plus, minus, tuple(candidate)
