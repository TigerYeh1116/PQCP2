"""Exact full PQCP completion on one literal support/content combination.

Solve coarse signs and alternated difference signs independently. Either
complete UNSAT excludes the whole combination; two witnesses reconstruct a
full binary pair. This is no claim about other supports/content/targets.
"""
from time import perf_counter

from .compressed_repair import _validated_reference
from .odd_lift_native import difference_sign_problem, reconstruct_odd_pair
from .ternary_signs_native import prepare_sign_enumeration, solve_sign_system
from .ternary_support_search import _audit_sign_words, support_sign_system
from .structured_energy import integer_periodic_autocorrelation_profile as pacf
from .ternary_support_search import support_mask


def complete_pqcp_supports(a, b, target, masks, *, max_assignments=2**23, conditioning=True,
                          native_preprocessing=True, binary_content_pruning=True, cardinality_pruning=True,
                          orbit_pruning=False, modulation_orbit_pruning=False,
                          sat_unknown_completion=False,sat_resource_limit=None,even_sign_orbit=False):
    started = perf_counter()
    a, b, d, wanted = _validated_reference(a, b, target)
    masks = tuple(masks)
    if (d > 47 or len(masks) != 2 or any(type(m) is not int or not 0 <= m < 1 << d for m in masks)
            or any(bin(m).count('1') != sum(v != 0 for v in w) for m, w in zip(masks, (a, b)))
            or type(max_assignments) is not int or not 1 <= max_assignments <= 2**31
            or type(conditioning) is not bool or type(native_preprocessing) is not bool
            or type(binary_content_pruning) is not bool):
        raise ValueError('odd-half supports preserving reference contents and valid budget required')
    if type(cardinality_pruning) is not bool or type(orbit_pruning) is not bool:
        raise ValueError('cardinality/orbit pruning must be boolean')
    if type(modulation_orbit_pruning) is not bool or (modulation_orbit_pruning and not orbit_pruning):
        raise ValueError('modulation orbit pruning requires boolean flags and enabled orbit cache')
    if (type(sat_unknown_completion) is not bool or (sat_resource_limit is not None
            and (type(sat_resource_limit) is not int or not 1<=sat_resource_limit<2**32))):
        raise ValueError('boolean SAT fallback and valid resource count required')
    if type(even_sign_orbit) is not bool:raise ValueError('boolean even-row sign orbit pruning required')
    other_masks, other_wanted, other_negative = difference_sign_problem(masks, target)
    problems = dict(coarse=(masks, wanted, (a.count(-1), b.count(-1))),
                    difference=(other_masks, other_wanted, other_negative))
    attempts = {}
    first_negative = a.count(-1)
    if binary_content_pruning and d == 43 and (first_negative <= 1 or first_negative >= 15):
        from .l86_binary_content import binary_content_bounds
        bounds = binary_content_bounds(target)
        if bounds is not None and any(not lo <= w.count(-1) <= hi for w, (lo, hi) in zip((a, b), bounds)):
            return dict(status='UNSAT_BINARY_SUPPORT', words=None, verified=False, attempts=attempts,
                        excluded_by='binary_content', elapsed=perf_counter()-started,
                        reason='recorded complete L86 binary-content proof')

    def finish(status, words=None, excluded_by=None):
        if orbit_key is not None and status=='UNSAT_BINARY_SUPPORT':
            FULL_SUPPORT_ORBITS.remember(orbit_key,masks,excluded_by,target=target,
                                        negative_counts=(a.count(-1),b.count(-1)))
        return dict(status=status, words=words, verified=words is not None,
                    attempts=attempts, excluded_by=excluded_by, elapsed=perf_counter()-started)

    orbit_key=None
    if orbit_pruning and d==43:
        from .support_orbit_cache import FULL_SUPPORT_ORBITS
        orbit_key=FULL_SUPPORT_ORBITS.key(masks,target,(a.count(-1),b.count(-1)),
                                         modulation=modulation_orbit_pruning)
        certificate=FULL_SUPPORT_ORBITS.get(orbit_key)
        if certificate is not None:
            return dict(status='UNSAT_BINARY_SUPPORT',words=None,verified=False,attempts={},
                excluded_by='dihedral_orbit',orbit_certificate=dict(source_masks=certificate[0],
                source_excluded_by=certificate[1],source_target=vars(certificate[2]) if certificate[2] is not None else None,
                source_negative_counts=certificate[3],
                scope=('independent dihedral plus simultaneous alternation and its canonical content gauge'
                       if modulation_orbit_pruning else 'same target/counts, independent row rotations/reflections')),
                elapsed=perf_counter()-started)

    if native_preprocessing and d == 43:
        from .ternary_system_native import prepare_native_system, run_native_system
        prepared = {name: prepare_native_system(m, d, w, n, conditioning=conditioning, cardinality_pruning=cardinality_pruning)
                    for name, (m, w, n) in problems.items()}
        inconsistent = [name for name, plan in prepared.items() if not plan['consistent']]
        ordered = sorted(problems, key=lambda name: (prepared[name]['estimated_assignments'], name != 'coarse'))
        def run(name):
            return run_native_system(prepared[name], max_assignments,even_sign_orbit=even_sign_orbit)
    else:
        systems = {name: support_sign_system(m, d, w, n) for name, (m, w, n) in problems.items()}
        inconsistent = [name for name, system in systems.items() if not system.consistent]
        prepared = {name: prepare_sign_enumeration(systems[name], m, n, conditioning=conditioning, cardinality_pruning=cardinality_pruning)
                    for name, (m, _, n) in problems.items() if systems[name].consistent}
        ordered = sorted(prepared, key=lambda name: (prepared[name][1], name != 'coarse'))
        def run(name):
            m, w, n = problems[name]
            return solve_sign_system(systems[name], m, d, w, n, max_assignments,
                                     conditioning=conditioning, prepared=prepared[name], cardinality_pruning=cardinality_pruning)
    for name in inconsistent:
        attempts[name] = dict(status='UNSAT_FIXED_SUPPORT', reason='GF2 / integer obstruction',
                              assignments=0, words=None)
        return finish('UNSAT_BINARY_SUPPORT', excluded_by=name)
    for name in ordered:
        out = run(name)
        attempts[name] = out
        if out['status'] == 'UNSAT_FIXED_SUPPORT':
            return finish('UNSAT_BINARY_SUPPORT', excluded_by=name)
        if out['status'] == 'UNKNOWN':
            if sat_unknown_completion:
                from .joint_support_sat import build_joint_support_model,check_joint_support_model
                joint=check_joint_support_model(build_joint_support_model(a,b,target,fixed_supports=masks,
                    known_coarse_words=attempts.get('coarse',{}).get('words'),
                    known_difference_words=attempts.get('difference',{}).get('words'),
                    affine_projection=True,mod8_projection=True),
                                                resource_limit=sat_resource_limit)
                attempts['joint_sat']=joint
                if joint['status']=='SAT_PQCP':return finish('SAT_PQCP',joint['words'])
                if joint['status']=='UNSAT_CONTENT_DOMAIN':return finish('UNSAT_BINARY_SUPPORT',excluded_by='joint_sat')
            return finish('UNKNOWN')
        # Audit intermediate witnesses too, including those whose other
        # problem later proves UNSAT. They remain useful replay evidence.
        if name == 'coarse':
            _audit_sign_words(out['words'], a, b, target, wanted, masks)
        else:
            words = out['words']
            if (tuple(x+y for x, y in zip(pacf(words[0]), pacf(words[1]))) != other_wanted
                    or tuple(w.count(-1) for w in words) != other_negative
                    or tuple(support_mask(w) for w in words) != other_masks):
                raise RuntimeError('difference sign witness failed independent integer audit')
    coarse = attempts['coarse']['words']
    _audit_sign_words(coarse, a, b, target, wanted, masks)
    words = reconstruct_odd_pair(coarse, attempts['difference']['words'], target)
    return finish('SAT_PQCP', words)
