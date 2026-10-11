"""Complete replacement of one parity half-row by certified norm preimages.

Fix three ACTUAL binary parity rows, allow every bit/count in the fourth.
DC/ALT contents enumerate all necessary k,eta,weight choices. Exact even-lag
profiles decide all remaining supports; every relative rotation is restored,
then all odd lags and the full original-length verifier certify solutions.
No template, reflection, homometric or fixed-content assumption is imposed.
"""
from collections import defaultdict
from functools import lru_cache
from time import perf_counter

from .correlation import normalize_binary_sequence
from .interleaving import interleave_bits
from .sds_norm_completion import (complete_binary_intersection_profile,intersection_profile,
                                 native_odd_cross_vectors)
from .structured_energy import target_profile
from .target_profiles import target_content_profiles
from .verifier import verify_pqcp
from .norm_trace_completion import norm_trace_preimages


@lru_cache(maxsize=4)
def _all_targets(length):return target_content_profiles(length,decimation_reduced=False)


@lru_cache(maxsize=16)
def _target_index(length,free):
    index=defaultdict(list)
    for target in _all_targets(length):
        counts=(target.a_even_ones,target.a_odd_ones,target.b_even_ones,target.b_odd_ones)
        index[tuple(v for i,v in enumerate(counts) if i!=free)].append((target,counts))
    return dict(index)


@lru_cache(maxsize=512)
def _profile_templates(length,free,fixed_counts):
    """Group exactly identical even equations; retain ALL odd targets."""
    groups=defaultdict(set)
    for target,counts in _target_index(length,free).get(fixed_counts,()):
        full=target_profile(length,target.k,target.eta)
        key=(counts[free],sum(counts)-length//2,tuple(full[2*u]//4 for u in range(1,length//2)))
        groups[key].add((target.k,target.eta))
    return tuple((weight,lam,offsets,tuple(sorted(targets)))
                 for (weight,lam,offsets),targets in groups.items())


def _prepare_center(a,b):
    a,b=tuple(normalize_binary_sequence(a)),tuple(normalize_binary_sequence(b));length=len(a);d=length//2
    if length!=len(b) or d not in (3,5,11,43) or length!=2*d:
        raise ValueError('equal binary rows of supported twice-prime lengths required')
    if verify_pqcp(a,b).is_valid:raise ValueError('completion centers must be non-solutions')
    rows=(a[::2],a[1::2],b[::2],b[1::2])
    masks=tuple(sum(v<<i for i,v in enumerate(row)) for row in rows)
    return (length,d,rows,masks,tuple(map(sum,rows)),tuple(intersection_profile(m,d) for m in masks))


def _validate_options(free_component,certified_psd,trace_backend,joint_gram=False,finite_trace=False,batched_psd=False,affine_finite=False):
    if (type(free_component) is not int or not 0<=free_component<4
            or type(certified_psd) is not bool or type(trace_backend) is not bool or type(joint_gram) is not bool
            or type(finite_trace) is not bool or type(batched_psd) is not bool or type(affine_finite) is not bool):
        raise ValueError('parity component 0..3 and boolean backend switches required')


def complete_parity_half_row(a,b,*,free_component=3,certified_psd=True,trace_backend=False,joint_gram=False,finite_trace=False,batched_psd=False,affine_finite=False):
    """Enumerate ALL full PQCPs sharing the other three literal parity rows.

    Components are A-even, A-odd, B-even, B-odd (0..3). Remaining weight can
    change, all strict target positions/signs are retained. Every support
    rotation omitted in norm enumeration is restored in the odd equations.
    """
    _validate_options(free_component,certified_psd,trace_backend,joint_gram,finite_trace,batched_psd,affine_finite)
    return _complete_prepared(_prepare_center(a,b),free_component,certified_psd,trace_backend,joint_gram,finite_trace,batched_psd,affine_finite)


def complete_parity_half_rows(a,b,*,free_components=(0,1,2,3),certified_psd=True,trace_backend=False,joint_gram=False,finite_trace=False,batched_psd=False,affine_finite=False):
    """Complete selected fibers with one validated immutable center preparation."""
    free_components=tuple(free_components)
    for free in free_components:_validate_options(free,certified_psd,trace_backend,joint_gram,finite_trace,batched_psd,affine_finite)
    if not free_components or len(set(free_components))!=len(free_components):
        raise ValueError('nonempty distinct parity components required')
    prepared=_prepare_center(a,b)
    return tuple(_complete_prepared(prepared,free,certified_psd,trace_backend,joint_gram,finite_trace,batched_psd,affine_finite) for free in free_components)


def complete_three_parity_rows(halves,*,free_component=3,certified_psd=True,trace_backend=False,joint_gram=False,finite_trace=False,batched_psd=False,affine_finite=False):
    """ALL PQCPs with three specified literal halves; None marks the unknown.

    This input is a partial assignment, not a non-solution center. Every
    complete output is independently verified, including all-zero unknown
    rows when they are valid at small lengths. No input validation bypass.
    """
    _validate_options(free_component,certified_psd,trace_backend,joint_gram,finite_trace,batched_psd,affine_finite)
    halves=tuple(halves)
    if len(halves)!=4 or halves[free_component] is not None or any(
            row is None for i,row in enumerate(halves) if i!=free_component):
        raise ValueError('four physical parity slots with exactly the chosen slot None required')
    fixed=tuple(tuple(normalize_binary_sequence(row)) for i,row in enumerate(halves) if i!=free_component)
    d=len(fixed[0])
    if d not in (3,5,11,43) or any(len(row)!=d for row in fixed):
        raise ValueError('three equal binary half-rows of supported prime lengths required')
    rows=list(halves);at=0
    for i in range(4):
        if i==free_component:rows[i]=(0,)*d
        else:rows[i]=fixed[at];at+=1
    rows=tuple(rows);masks=tuple(sum(v<<i for i,v in enumerate(row)) for row in rows)
    prepared=(2*d,d,rows,masks,tuple(map(sum,rows)),tuple(intersection_profile(m,d) for m in masks))
    out=_complete_prepared(prepared,free_component,certified_psd,trace_backend,joint_gram,finite_trace,batched_psd,affine_finite)
    out['input_kind']='THREE_LITERAL_PARITY_ROWS'
    return out


def _batched_obstructions(profile_groups):
    """Same range/sum/PSD decisions, sharing only affine spectral arithmetic."""
    from .certified_profile_psd import profile_psd_obstruction,sparse_profile_psd_obstructions
    results={};groups=defaultdict(list)
    for wanted,targets in profile_groups.items():
        filtered=complete_binary_intersection_profile(wanted,certified_psd=False,obstruction_only=True)
        results[wanted]=filtered
        if filtered['complete']:continue
        k,eta=min(targets);d=len(wanted)
        if k%2:
            certificate=profile_psd_obstruction(wanted)
            if certificate is not None:
                filtered.update(complete=True,reason='certified negative PSD upper bound',psd_certificate=certificate)
                filtered.pop('status',None)
        else:
            tau=k//2;base=list(wanted);base[tau]-=eta;base[-tau]-=eta
            groups[tuple(base)].append((wanted,tau,eta))
    for base,items in groups.items():
        certificates=sparse_profile_psd_obstructions(base,tuple((t,e) for w,t,e in items))
        for (wanted,tau,eta),certificate in zip(items,certificates):
            if certificate is not None:
                results[wanted].update(complete=True,reason='certified negative PSD upper bound',psd_certificate=certificate)
                results[wanted].pop('status',None)
    return results


def _affine_field_obstructions(profile_groups,partner,other_cross,fixed_counts):
    from .finite_field_trace_pruning import finite_trace_affine_filter
    filters={};groups=defaultdict(list);origins={};field_results={}
    for wanted,targets in profile_groups.items():
        filters[wanted]=complete_binary_intersection_profile(wanted,certified_psd=False,obstruction_only=True)
        if filters[wanted]['complete']:continue
        k,eta=min(targets);d=len(wanted);base=list(wanted)
        # Odd targets share one norm profile but require many distinct trace
        # certificates. Keep the single PSD check first for these groups;
        # affine early evaluation is only for sparse even-norm targets.
        if k%2:continue
        if not k%2:base[k//2]-=eta;base[-(k//2)]-=eta
        origin=tuple(base),sum(fixed_counts)+wanted[0]-d;origins[wanted]=origin
        groups[origin].extend(sorted(targets))
    decided={}
    for (base,lam),targets in groups.items():
        if len(set(targets))!=len(targets):raise RuntimeError('affine grouping duplicated an original target')
        cross=tuple(lam-v for v in other_cross)
        out=finite_trace_affine_filter(base,partner,cross,targets)
        decided[base,lam]=set(out['allowed_targets']),{tuple(c['target']):c for c in out['certificates']}
    for wanted,origin in origins.items():
        allowed,certificates=decided[origin];targets=tuple(sorted(profile_groups[wanted]))
        proofs=tuple(certificates[t] for t in targets if t not in allowed)
        if any(tuple(c['wanted'])!=wanted for c in proofs):raise RuntimeError('affine proof changed the actual integer norm')
        field_results[wanted]=dict(complete=True,allowed_targets=tuple(t for t in targets if t in allowed),
                                  certificates=proofs,tested_targets=len(targets),excluded_targets=len(proofs),
                                  scope='Same finite-field necessary condition, affine dots shared before PSD.')
    return filters,field_results


def _complete_prepared(prepared,free_component,certified_psd,trace_backend,joint_gram=False,finite_trace=False,batched_psd=False,affine_finite=False):
    length,d,rows,masks,weights,profiles=prepared
    started=perf_counter()
    fixed_counts=tuple(v for i,v in enumerate(weights) if i!=free_component)
    fixed_profiles=tuple(p for i,p in enumerate(profiles) if i!=free_component)
    fixed_sum=tuple(sum(p[u] for p in fixed_profiles) for u in range(d))
    profile_groups=defaultdict(set)
    for weight,lam,offsets,targets in _profile_templates(length,free_component,fixed_counts):
        wanted=(weight,)+tuple(lam+offset-fixed_sum[u] for u,offset in enumerate(offsets,1))
        profile_groups[wanted].update(targets)
    prefiltered=_batched_obstructions(profile_groups) if batched_psd and certified_psd and not (affine_finite and finite_trace) else None
    # Actual physical odd rows are used, including their signs. Odd PACFs ADD.
    # No O_B negation or canonical-profile gauge is silently introduced.
    free_side=free_component//2
    other=1-free_side
    fixed_odd=None
    witnesses=[];seen=set();records=[]
    phase_candidates=odd_checks=trace_candidates=integer_profile_checks=0
    gram_tested=gram_excluded=0
    field_tested=field_excluded=0
    companion=other_cross=None
    def prepare_cross():
        centered=list(masks);center_shift=(d+1)//2;all_bits=(1<<d)-1
        for index in (1,3):centered[index]=((masks[index]<<center_shift)|(masks[index]>>(d-center_shift)))&all_bits
        companion=centered[2*free_side+(0 if free_component%2 else 1)]
        def rotated(mask,u):return ((mask<<u)|(mask>>(d-u)))&all_bits if u else mask
        e_other,o_other=centered[2*other],centered[2*other+1]
        other_cross=tuple(bin(e_other&rotated(o_other,u)).count('1')
                          +bin(e_other&rotated(o_other,(-u)%d)).count('1') for u in range(d))
        return companion,other_cross,center_shift
    affine_results=post_field_psd=None
    if affine_finite and finite_trace:
        companion,other_cross,center_shift=prepare_cross()
        prefiltered,affine_results=_affine_field_obstructions(profile_groups,companion,other_cross,fixed_counts)
        if batched_psd and certified_psd:
            remaining={w:targets for w,targets in profile_groups.items() if not prefiltered[w]['complete']
                       and (w not in affine_results or affine_results[w]['allowed_targets'])}
            post_field_psd=_batched_obstructions(remaining)
    for wanted,targets in profile_groups.items():
        original_targets=tuple(sorted(targets));gram_result=field_result=None
        if trace_backend or joint_gram or finite_trace or prefiltered is not None:
            filtered=(prefiltered[wanted] if prefiltered is not None else
                      complete_binary_intersection_profile(wanted,certified_psd=certified_psd,obstruction_only=True))
            if affine_results is not None and wanted not in affine_results and not filtered['complete'] and certified_psd:
                filtered=(post_field_psd[wanted] if post_field_psd is not None else
                          complete_binary_intersection_profile(wanted,certified_psd=True,obstruction_only=True))
            if filtered['complete']:
                records.append(dict(wanted=wanted,targets=tuple(sorted(targets)),result=filtered));continue
        if finite_trace:
            from .finite_field_trace_pruning import finite_trace_target_filter
            if other_cross is None:companion,other_cross,center_shift=prepare_cross()
            lam=sum(fixed_counts)+wanted[0]-d;base_cross=tuple(lam-v for v in other_cross)
            field_result=(affine_results[wanted] if affine_results is not None and wanted in affine_results else
                          finite_trace_target_filter(wanted,companion,base_cross,original_targets))
            field_tested+=field_result['tested_targets'];field_excluded+=field_result['excluded_targets']
            targets=field_result['allowed_targets'];filtered['finite_trace_result']=field_result
            if not targets:
                out=dict(filtered,complete=True,status='UNSAT_FINITE_FIELD_TRACE_TARGETS',
                         scope='All reported FULL target fibers excluded by finite-field conjugate-root certificates; '
                               'not an exclusion of unconstrained periodic norm preimages.')
                records.append(dict(wanted=wanted,targets=original_targets,retained_targets=(),result=out));continue
            if affine_results is not None and wanted in affine_results and certified_psd:
                filtered=(post_field_psd[wanted] if post_field_psd is not None else
                          complete_binary_intersection_profile(wanted,certified_psd=True,obstruction_only=True))
                filtered['finite_trace_result']=field_result
                if filtered['complete']:
                    records.append(dict(wanted=wanted,targets=original_targets,retained_targets=tuple(sorted(targets)),result=filtered));continue
        if joint_gram:
            from .joint_gram_pruning import joint_gram_target_filter
            if other_cross is None:companion,other_cross,center_shift=prepare_cross()
            lam=sum(fixed_counts)+wanted[0]-d
            base_cross=tuple(lam-v for v in other_cross)
            gram_result=joint_gram_target_filter(wanted,companion,base_cross,tuple(sorted(targets)))
            gram_tested+=gram_result['tested_targets'];gram_excluded+=gram_result['excluded_targets']
            targets=gram_result['allowed_targets']
            if not targets:
                out=dict(filtered,complete=True,status='UNSAT_JOINT_GRAM_TARGETS',joint_gram_result=gram_result,
                         scope='All reported FULL target fibers excluded by exact joint Gram certificates; '
                               'not an exclusion of unconstrained periodic norm preimages.')
                records.append(dict(wanted=wanted,targets=original_targets,retained_targets=(),result=out));continue
        if trace_backend:
            if other_cross is None:companion,other_cross,center_shift=prepare_cross()
            desired_norm=sum((v%2)<<i for i,v in enumerate(wanted))
            allowed={};trace_records=[];lam=sum(fixed_counts)+wanted[0]-d
            w_companion=bin(companion).count('1')
            lower=2*max(0,wanted[0]+w_companion-d);upper=2*min(wanted[0],w_companion)
            for k,eta in sorted(targets):
                desired_full=target_profile(length,k,eta)
                required_cross=tuple(lam+desired_full[(2*u+d)%length]//4-other_cross[u] for u in range(d))
                if required_cross[0]%2 or any(not lower<=v<=upper for v in required_cross):
                    trace_records.append(dict(k=k,eta=eta,range_excluded=True));continue
                trace=sum((v%2)<<i for i,v in enumerate(required_cross))
                roots=norm_trace_preimages(desired_norm,trace,companion,d)
                if not roots['complete']:raise RuntimeError('norm/trace fiber incomplete')
                trace_records.append(dict(k=k,eta=eta,result=roots))
                trace_candidates+=len(roots['masks'])
                for mask in roots['masks']:allowed.setdefault(mask,set()).add((k,eta))
            matching_masks=[]
            for mask,mask_targets in allowed.items():
                if bin(mask).count('1')!=wanted[0]:continue
                integer_profile_checks+=1
                if intersection_profile(mask,d)!=wanted:continue
                matching_masks.append(mask)
                changed=tuple((mask>>((i+center_shift)%d if free_component%2 else i))&1 for i in range(d))
                trial=list(rows);trial[free_component]=changed
                words=tuple(interleave_bits(tuple(1-2*v for v in trial[2*side]),
                                            tuple(1-2*v for v in trial[2*side+1])) for side in range(2))
                odd_checks+=1;check=verify_pqcp(*words)
                if not check.is_valid:continue
                k=check.nonzero_shifts[0];eta=check.profile[k]//4
                if (k,eta) not in mask_targets or tuple(check.profile)!=target_profile(length,k,eta):
                    raise RuntimeError('norm/trace hit outside necessary target set')
                changed_rows=(words[0][::2],words[0][1::2],words[1][::2],words[1][1::2])
                if any(changed_rows[i]!=rows[i] for i in range(4) if i!=free_component):
                    raise RuntimeError('norm/trace completion changed fixed parity rows')
                if words in seen:continue
                seen.add(words);witnesses.append(dict(words=words,k=k,eta=eta,free_component=free_component,
                                                      remaining_weight=wanted[0],support_phase=None))
            out=dict(filtered,complete=True,status='COMPLETE_NORM_TRACE_FIBERS',masks=tuple(matching_masks),
                     trace_records=trace_records,reason='All norm/trace masks integer-audited',trace_backend=True)
            if gram_result is not None:out['joint_gram_result']=gram_result
            records.append(dict(wanted=wanted,targets=original_targets,retained_targets=tuple(sorted(targets)),result=out));continue
        out=complete_binary_intersection_profile(wanted,certified_psd=certified_psd and prefiltered is None)
        if out['masks'] and fixed_odd is None:
            other_vectors=native_odd_cross_vectors(masks[2*other],masks[2*other+1],d)
            fixed_odd=next(vector for vector,phases in other_vectors.items() if 0 in phases)
        odd_targets=defaultdict(list)
        for k,eta in targets:
            desired=target_profile(length,k,eta)
            odd_targets[tuple(desired[2*r+1] for r in range(d//2+1))].append((k,eta))
        for candidate in out['masks']:
            if free_component%2:
                variable=native_odd_cross_vectors(masks[2*free_side],candidate,d)
            else:
                variable=native_odd_cross_vectors(candidate,masks[2*free_side+1],d)
            for vector,phases in variable.items():
                odd_checks+=1;phase_candidates+=len(phases)
                summed=tuple(x+y for x,y in zip(vector,fixed_odd))
                matching=odd_targets.get(summed,())
                if not matching:continue
                for phase in phases:
                    # Native phase shifts the O row. When E is free, shift E
                    # by -phase instead, preserving the fixed O row literally.
                    shift=phase if free_component%2 else (-phase)%d
                    changed=tuple((candidate>>((i+shift)%d))&1 for i in range(d))
                    trial=list(rows);trial[free_component]=changed
                    words=tuple(interleave_bits(tuple(1-2*v for v in trial[2*side]),
                                                tuple(1-2*v for v in trial[2*side+1])) for side in range(2))
                    check=verify_pqcp(*words)
                    if not check.is_valid:raise RuntimeError('half-row completion failed independent full verifier')
                    for k,eta in matching:
                        if tuple(check.profile)!=target_profile(length,k,eta):
                            raise RuntimeError('half-row completion failed full target profile audit')
                    changed_rows=(words[0][::2],words[0][1::2],words[1][::2],words[1][1::2])
                    if any(changed_rows[i]!=rows[i] for i in range(4) if i!=free_component):
                        raise RuntimeError('half-row completion changed a fixed parity row')
                    if words in seen:continue
                    seen.add(words);k,eta=matching[0]
                    witnesses.append(dict(words=words,k=k,eta=eta,free_component=free_component,
                                          remaining_weight=wanted[0],support_phase=shift))
        if gram_result is not None:out['joint_gram_result']=gram_result
        if field_result is not None:out['finite_trace_result']=field_result
        records.append(dict(wanted=wanted,targets=original_targets,retained_targets=tuple(sorted(targets)),result=out))
    return dict(complete=True,status='SAT' if witnesses else 'UNSAT_FIXED_OTHER_PARITY_ROWS',
                witnesses=witnesses,profiles_decided=len(records),profile_records=records,
                certified_psd=certified_psd,
                batched_psd=batched_psd,
                affine_finite=affine_finite,
                joint_gram=joint_gram,joint_gram_targets_tested=gram_tested,joint_gram_targets_excluded=gram_excluded,
                finite_trace=finite_trace,finite_trace_targets_tested=field_tested,finite_trace_targets_excluded=field_excluded,
                trace_backend=trace_backend,norm_trace_candidates=trace_candidates,
                integer_profile_checks=integer_profile_checks,
                free_component=free_component,raw_half_assignments=1<<d,
                norm_visited=sum(r['result']['visited'] for r in records),
                weight_matches=sum(r['result']['weight_matches'] for r in records),
                intersection_checks=sum(r['result']['correlation_checks'] for r in records),
                completed_support_orbits=0 if trace_backend else sum(len(r['result']['masks']) for r in records),
                completed_trace_support_masks=sum(len(r['result']['masks']) for r in records) if trace_backend else 0,
                restored_phase_candidates=phase_candidates,odd_profile_checks=odd_checks,
                elapsed=perf_counter()-started,
                scope='ALL arbitrary bits and necessary counts in the chosen physical parity half-row, '
                      'all strict k/eta, other three literal parity rows fixed. Not general PQCP nonexistence.')
