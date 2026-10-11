"""Bounded exact radius-four repair feeding the existing compressed lift jobs.

Candidate selection is heuristic; local UNSAT never discards a content family.
No full PQCP is certified here. Native results are independently integer-audited.
"""
from dataclasses import asdict

from .compressed_completion import CompressedLiftJob
from .ternary_repair_native import repair_two_swaps
from .torch_compressed_search import representative_lift


def repair_search_result(result, profiles, *, top, stop_event=None):
    """Append deduplicated exact jobs and return reproducible repair records."""
    if type(top) is not int or top < 0:
        raise ValueError('top must be a nonnegative integer')
    if top == 0 or (stop_event is not None and stop_event.is_set()):
        return []
    energies = result.energies.int().cpu().tolist()
    words = result.words.int().cpu().tolist()
    known = {job.canonical_key for job in result.jobs}
    references = set()
    records = []
    for lane in sorted(range(len(profiles)), key=lambda i: (energies[i], i)):
        if len(records) >= top or (stop_event is not None and stop_event.is_set()):
            break
        if energies[lane] == 0:
            continue  # Already exported by the GPU search's exact audit.
        a, b = (tuple(w) for w in words[lane])
        target = profiles[lane]
        key = (a, b, target)
        if key in references:
            continue
        references.add(key)
        repaired, evaluations = repair_two_swaps(a, b, target)
        record = dict(lane=lane, original_energy=energies[lane], reference=[a, b],
                      target=asdict(target), result=asdict(repaired), evaluations=evaluations,
                      new_job=False)
        records.append(record)
        if repaired.words is None:
            continue
        ca, cb = (tuple(2*v for v in word) for word in repaired.words)
        sa = representative_lift(ca, target.a_even_ones, target.a_odd_ones)
        sb = representative_lift(cb, target.b_even_ones, target.b_odd_ones)
        job = CompressedLiftJob(ca, cb, target, sa, sb)
        if job.canonical_key not in known:
            known.add(job.canonical_key)
            result.jobs.append(job)
            record['new_job'] = True
    return records


def advanced_repair_search_result(result, profiles, *, transfer_top=0, mitm_top=0,
                                 max_results=32, stop_event=None):
    """Optional extra neighborhoods; each outcome has an explicit scope."""
    from .ternary_content_transfer import repair_content_transfer
    from .ternary_mitm import repair_mitm
    energies = result.energies.int().cpu().tolist()
    words = result.words.int().cpu().tolist()
    known = {job.canonical_key for job in result.jobs}
    records = []
    for method, top in (('content_transfer', transfer_top), ('mitm_two_per_side', mitm_top)):
        seen = set()
        used = 0
        for lane in sorted(range(len(profiles)), key=lambda i: (energies[i], i)):
            if used >= top or (stop_event is not None and stop_event.is_set()):
                break
            if energies[lane] == 0:
                continue
            a, b = (tuple(w) for w in words[lane])
            target = profiles[lane]
            if (a, b, target) in seen:
                continue
            seen.add((a, b, target))
            used += 1
            if method == 'content_transfer':
                repaired, work = repair_content_transfer(a, b, target)
                outcome = asdict(repaired)
                pairs = [repaired.words] if repaired.words is not None else []
            else:
                outcome = repair_mitm(a, b, target, max_results=max_results)
                pairs = outcome['words']
                work = dict(entries=outcome['entries'], probes=outcome['probes'])
            new_jobs = 0
            for ca, cb in pairs:
                ca, cb = tuple(2*v for v in ca), tuple(2*v for v in cb)
                sa = representative_lift(ca, target.a_even_ones, target.a_odd_ones)
                sb = representative_lift(cb, target.b_even_ones, target.b_odd_ones)
                job = CompressedLiftJob(ca, cb, target, sa, sb)
                if job.canonical_key not in known:
                    known.add(job.canonical_key)
                    result.jobs.append(job)
                    new_jobs += 1
            records.append(dict(method=method, lane=lane, original_energy=energies[lane],
                                reference=[a,b], target=asdict(target), result=outcome,
                                work=work, new_jobs=new_jobs))
    return records


def refine_search_result_native(result, profiles, *, top=4, moves=131072, seed=123, stop_event=None):
    """Continue fresh GPU candidates using bounded integer navigation.

    First select one best lane per target, then fill by energy. No content or
    target is mathematically excluded. Only exact audited hits become jobs.
    """
    from .ternary_native_search import search_ternary_native
    if type(top) is not int or top < 0:
        raise ValueError('top must be a nonnegative integer')
    if not top or not moves or (stop_event is not None and stop_event.is_set()):
        return []
    energies=result.energies.int().cpu().tolist()
    words=result.words.int().cpu().tolist()
    ordered=sorted(range(len(profiles)),key=lambda i:(energies[i],i))
    groups=set();priority=[]
    for lane in ordered:
        if energies[lane] and profiles[lane] not in groups:
            groups.add(profiles[lane]);priority.append(lane)
    priority+= [i for i in ordered if i not in priority and energies[i]]
    known={job.canonical_key for job in result.jobs}
    records=[]
    for lane in priority[:top]:
        if stop_event is not None and stop_event.is_set():
            break
        target=profiles[lane]
        out=search_ternary_native(*words[lane],target,seed=seed+1000003*lane,moves=moves)
        # Device tensors keep the existing repair API; writes contain small exact integers.
        result.words[lane]=result.words.new_tensor(out['words'])
        result.profiles[lane]=result.profiles.new_tensor(out['profile'])
        result.energies[lane]=out['energy']
        new_job=False
        if out['energy']==0:
            ca,cb=(tuple(2*v for v in w) for w in out['words'])
            sa=representative_lift(ca,target.a_even_ones,target.a_odd_ones)
            sb=representative_lift(cb,target.b_even_ones,target.b_odd_ones)
            job=CompressedLiftJob(ca,cb,target,sa,sb)
            if job.canonical_key not in known:
                known.add(job.canonical_key);result.jobs.append(job);new_job=True
        records.append(dict(method='native_navigation',lane=lane,reference=words[lane],
            target=asdict(target),result=out,new_job=new_job))
    return records


def repair_search_result_support(result, profiles, *, top=4, max_supports=8,
                                 max_assignments=2**23, stop_event=None, method='local', full_completion=None):
    """Local or cyclic-norm support scheduling with exact fixed-count signs.

    No bounded miss excludes the lane from any of the existing repairs.
    Every hit is independently audited before entering the binary lift pool.
    """
    from .ternary_support_search import repair_support_signs
    from .cyclic_support_search import repair_cyclic_supports
    from .structured_energy import integer_periodic_autocorrelation_profile as pacf
    if type(top) is not int or top < 0:
        raise ValueError('top must be a nonnegative integer')
    if method not in ('local', 'cyclic_norm'):
        raise ValueError('support method must be local or cyclic_norm')
    if full_completion is not None and type(full_completion) is not bool:
        raise ValueError('full_completion must be boolean or None (automatic)')
    if not top or (stop_event is not None and stop_event.is_set()):
        return []
    energies = result.energies.int().cpu().tolist()
    words = result.words.int().cpu().tolist()
    known = {job.canonical_key for job in result.jobs}
    records = []; references = set()
    for lane in sorted(range(len(profiles)), key=lambda i: (energies[i], i)):
        if len(records) >= top or (stop_event is not None and stop_event.is_set()):
            break
        binary = method == 'cyclic_norm' and (full_completion if full_completion is not None else profiles[lane].L == 86)
        if energies[lane] == 0 and not binary:
            continue
        a, b = (tuple(w) for w in words[lane]); target = profiles[lane]
        key = (a, b, target)
        if key in references:
            continue
        references.add(key)
        repair = repair_support_signs if method == 'local' else repair_cyclic_supports
        options = dict(full_completion=full_completion) if method == 'cyclic_norm' else {}
        out = repair(a, b, target, max_supports=max_supports,
                     max_assignments=max_assignments, stop_event=stop_event, **options)
        new_job = False
        upgraded = False
        if out['words'] is not None:
            ca, cb = (tuple(2*v for v in w) for w in out['words'])
            if out.get('full_words') is not None:
                from .structured_energy import compress_binary_sequence, target_profile
                from .target_profiles import pair_content
                from .verifier import verify_pqcp
                sa, sb = out['full_words']
                check = verify_pqcp(sa, sb)
                if (not check.is_valid or check.profile != target_profile(target.L, target.k, target.eta)
                        or pair_content(sa, sb) != (target.a_even_ones, target.a_odd_ones,
                                                   target.b_even_ones, target.b_odd_ones)
                        or (compress_binary_sequence(sa, 2), compress_binary_sequence(sb, 2)) != (ca, cb)):
                    raise RuntimeError('full support handoff failed independent binary audit')
            else:
                sa = representative_lift(ca, target.a_even_ones, target.a_odd_ones)
                sb = representative_lift(cb, target.b_even_ones, target.b_odd_ones)
            job = CompressedLiftJob(ca, cb, target, sa, sb)
            if job.canonical_key not in known:
                known.add(job.canonical_key); result.jobs.append(job); new_job = True
            elif out.get('full_words') is not None:
                # Compressed-job deduplication must not discard a fully
                # verified source when an arbitrary lift was queued first.
                for index, prior in enumerate(result.jobs):
                    if prior.canonical_key == job.canonical_key:
                        result.jobs[index] = job
                        upgraded = True
                        break
            result.words[lane] = result.words.new_tensor(out['words'])
            profile = tuple(x+y for x, y in zip(*(pacf(w) for w in out['words'])))
            result.profiles[lane] = result.profiles.new_tensor(profile)
            result.energies[lane] = 0
        records.append(dict(method='support_signs' if method == 'local' else 'cyclic_norm',
                            lane=lane, original_energy=energies[lane],
                            reference=words[lane], target=asdict(target), result=out, new_job=new_job,
                            full_source_upgraded=upgraded))
    return records
