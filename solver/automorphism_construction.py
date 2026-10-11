"""Exact sufficient construction: y_i=(-1)^i x_(v*i) at positive even target.

On independent nonzero shifts of odd d, v must have odd order (d-1)/2.
Solve epsilon(u)+epsilon(v*u)=2*delta_{+/-k}(u). Its unique +/-1
pattern gives unary trit targets P_c=(epsilon+1)/2, P_h=(epsilon-1)/2.
Complement supports make E=c+h and O'=c-h binary. Re-indexing gives
full x,y with precisely two +4 sidelobes. This is a restricted construction;
failure never rules out other L86 pairs. No known PQCPs are read.
"""
from collections import Counter
from dataclasses import dataclass
from time import perf_counter

from .cyclic_support import cyclic_norm_family, iter_family, rotation_representatives
from .odd_lift_native import reconstruct_odd_pair
from .structured_energy import integer_periodic_autocorrelation_profile as pacf
from .target_profiles import TargetContentProfile
from .ternary_support_search import support_mask, support_signature
from .ternary_system_native import prepare_native_system, run_native_system


def projective_order(v, d):
    value = 1
    for exponent in range(1, d):
        value = value*v % d
        if value in (1, d-1):
            return exponent
    raise ValueError('unit modulo supported odd prime required')


@dataclass(frozen=True)
class AutomorphismTarget:
    d: int
    v: int
    k: int
    sigma_c: int
    sigma_h: int
    weight_c: int
    negative_c: int
    negative_h: int
    epsilon: tuple
    wanted_c: tuple
    wanted_h: tuple


def positive_automorphism_target(d, v, k, sigma_c, sigma_h):
    if (type(d) is not int or d not in (3, 11, 43) or any(type(z) is not int for z in (v, k, sigma_c, sigma_h))
            or not 1 <= v < d or not 1 <= k <= d//2 or sigma_c < 0 or sigma_h < 0
            or sigma_c*sigma_c+sigma_h*sigma_h != d+2 or sigma_c+sigma_h > d
            or (d-sigma_c-sigma_h) % 2 or projective_order(v, d) != (d-1)//2):
        raise ValueError('positive character sums and transitive projective multiplier required')
    weight = sigma_c*sigma_c-(d+1)//2
    if (not sigma_c <= weight <= d-sigma_h or (weight-sigma_c) % 2
            or (d-weight-sigma_h) % 2):
        raise ValueError('unary target contents cannot be realized')
    epsilon = [0]*d
    epsilon[k] = epsilon[d-k] = 1
    position, sign = v*k % d, 1
    for _ in range(1, (d-1)//2):
        epsilon[position] = epsilon[d-position] = sign
        position = v*position % d
        sign = -sign
    if any(epsilon[u]+epsilon[v*u % d] != (2 if u in (k, d-k) else 0) for u in range(1, d)):
        raise RuntimeError('multiplicative-cycle recurrence failed exact audit')
    wanted_c = (weight,)+tuple((z+1)//2 for z in epsilon[1:])
    wanted_h = (d-weight,)+tuple((z-1)//2 for z in epsilon[1:])
    if sum(wanted_c) != sigma_c*sigma_c or sum(wanted_h) != sigma_h*sigma_h:
        raise RuntimeError('unary targets violate character identity')
    return AutomorphismTarget(d, v, k, sigma_c, sigma_h, weight, (weight-sigma_c)//2,
                              (d-weight-sigma_h)//2, tuple(epsilon), wanted_c, wanted_h)


def assemble_automorphism_pair(c, h, spec):
    c, h = tuple(c), tuple(h)
    d = spec.d
    if (len(c) != d or len(h) != d or any(type(z) is not int or z not in (-1, 0, 1) for z in c+h)
            or tuple(pacf(c)) != spec.wanted_c or tuple(pacf(h)) != spec.wanted_h
            or c.count(-1) != spec.negative_c or h.count(-1) != spec.negative_h
            or any(int(a != 0)+int(b != 0) != 1 for a, b in zip(c, h))):
        raise ValueError('complementary unary witnesses required')
    inverse2 = (d+1)//2
    ca = tuple(c[i*inverse2 % d] for i in range(d))
    cb = tuple(h[spec.v*i*inverse2 % d] for i in range(d))
    ga = tuple(h[i*inverse2 % d] for i in range(d))
    gb = tuple(c[spec.v*i*inverse2 % d] for i in range(d))
    e = (d-spec.sigma_c-spec.sigma_h)//2
    target = TargetContentProfile(2*d, 2*spec.k, 1, e, e+spec.sigma_h, e, e+spec.sigma_c)
    words = reconstruct_odd_pair((ca, cb), (ga, gb), target)
    multiplier = spec.v if spec.v % 2 else spec.v+d
    if words[1] != tuple(words[0][multiplier*i % (2*d)] ^ (i % 2) for i in range(2*d)):
        raise RuntimeError('constructed full pair violates automorphism identity')
    return words, target


def search_automorphism_target(spec, *, max_supports=None, max_assignments=2**23, stop_event=None):
    if ((max_supports is not None and (type(max_supports) is not int or max_supports < 1))
            or type(max_assignments) is not int or not 1 <= max_assignments <= 2**31):
        raise ValueError('positive support and assignment budgets required')
    started = perf_counter()
    rhs = sum((z & 1) << i for i, z in enumerate(spec.wanted_c))
    family = cyclic_norm_family(rhs, spec.d)
    representatives = rotation_representatives(family)
    statuses, rows = Counter(), []
    scanned = tested = 0
    unknown = False
    full = (1 << spec.d)-1
    expected_signature = sum((spec.wanted_c[u] & 1) << (u-1) for u in range(1, spec.d//2+1))

    def finish(status, words=None, target=None, complete=False):
        return dict(status=status, words=words, target=target.__dict__ if target else None,
            verified=words is not None, complete=complete, scanned_orbits=scanned, tested_supports=tested,
            literal_family_size=family.size, orbit_family_size=representatives.size,
            statuses=dict(statuses), retained_cases=rows, elapsed=perf_counter()-started,
            scope='this unary-profile automorphism construction only')

    for mask in iter_family(representatives):
        scanned += 1
        if stop_event is not None and stop_event.is_set():
            return finish('UNKNOWN')
        if bin(mask).count('1') != spec.weight_c:
            continue
        if max_supports is not None and tested >= max_supports:
            return finish('UNKNOWN')
        if support_signature(mask, spec.d) != expected_signature:
            raise RuntimeError('generic norm support failed independent parity audit')
        tested += 1
        attempts = {}
        for name, support, wanted, negative in (
            ('c', mask, spec.wanted_c, spec.negative_c),
            ('h', full ^ mask, spec.wanted_h, spec.negative_h)):
            plan = prepare_native_system((support, 0), spec.d, wanted, (negative, 0), conditioning=False)
            out = run_native_system(plan, max_assignments)
            attempts[name] = out
            statuses[name+'_'+out['status']] += 1
            if out['status'] == 'UNSAT_FIXED_SUPPORT':
                break
            if out['status'] == 'UNKNOWN':
                unknown = True
                rows.append(dict(mask=mask, attempts=attempts))
                break
            if (tuple(pacf(out['words'][0])) != wanted or any(out['words'][1])
                    or out['words'][0].count(-1) != negative or support_mask(out['words'][0]) != support):
                raise RuntimeError('unary witness failed independent integer audit')
        else:
            c, h = attempts['c']['words'][0], attempts['h']['words'][0]
            words, target = assemble_automorphism_pair(c, h, spec)
            rows.append(dict(mask=mask, attempts=attempts))
            return finish('SAT_PQCP', words, target)
        if (attempts.get('c', {}).get('status') == 'SAT_COMPRESSED'
                and attempts.get('h', {}).get('status') != 'UNKNOWN'):
            rows.append(dict(mask=mask, attempts=attempts))
    return finish('UNKNOWN' if unknown else 'UNSAT_CONSTRUCTOR_TARGET', complete=not unknown)
