"""Fix signed c_A only; all g_A signs and all of binary B remain free.

Unary integer c_B norm completion supplies every necessary support class.
Given a c_B support, its signed models do not affect the g_A/g_B equations:
one c_B witness suffices for existence, then an exact two-side G solve.
Never apply a SINGLE-norm obstruction to the sum of two unknown G norms.
"""
from collections import OrderedDict
from functools import lru_cache
from itertools import combinations,product
from math import isqrt
from time import perf_counter

from .cyclic_support import cyclic_norm_family,rotation_representatives
from .cyclic_support_native import all_weighted_norm_supports,signed_norm_supports,dihedral_support_key
from .full_row_completion import reconstruct_crt_binary,signed_norm_psd_obstruction,unary_sign_completion
from .prime_norm_obstruction import prime_norm_obstruction
from .structured_energy import integer_periodic_autocorrelation_profile as pacf
from .ternary_system_native import prepare_native_system,run_native_system
from .verifier import verify_pqcp
from .unary_norm_carrier_cache import UnaryNormCarrierCache


class CompleteUnaryNormUnsatCache:
    """In-process, bounded proofs of SINGLE C_B norm/count exhaustion only.

    No checkpoint/dictionary import and no SAT/G-only/UNKNOWN entries.
    Different original k parities may share C_B but require different G
    equations. Only proven C_B emptiness can be reused across them.
    """
    def __init__(self,capacity=4096):
        if type(capacity) is not int or not 1<=capacity<=65536:raise ValueError('bounded proof cache capacity 1..65536 required')
        self.capacity=capacity;self._items=OrderedDict();self.hits=0

    def _get(self,key):
        if key not in self._items:return None
        value=self._items.pop(key);self._items[key]=value;self.hits+=1
        return value

    def _remember(self,key,result):
        if (not result['c_b_norm_unsat'] or not result['complete'] or result['words'] is not None
                or result['coarse_witness_supports'] or result['unknown_sign_fibers']):
            raise RuntimeError('only certified whole C_B norm emptiness may enter the proof cache')
        self._items[key]=dict(source_k=result['k'],source_eta=result['eta'],
            wanted_c_b=key[0],negative_c_b=key[1],source_support_orbits=result['support_orbits'],
            source_sign_fibers=result['sign_fibers'],source_assignments=result['assignments'],
            scope='All signed C_B supports/models for this exact SINGLE integer norm/count exhausted. No G-only exclusion reused.')
        if len(self._items)>self.capacity:self._items.popitem(last=False)


def validate_coarse_row(row):
    row=tuple(row)
    if len(row) not in (3,5,11,43) or any(type(v) is not int or v not in (-1,0,1) for v in row):
        raise ValueError('literal signed ternary row of supported prime length required')
    return row


def coarse_l1_anchors(row,radius):
    """Every literal trit row at L1 distance <= radius (currently 0..2)."""
    row=validate_coarse_row(row)
    if type(radius) is not int or not 0<=radius<=2:raise ValueError('exact radius 0..2 required')
    yield row,()
    neighbors=tuple(tuple(v for v in (-1,0,1) if abs(v-old)==1) for old in row)
    for distance in range(1,radius+1):
        for positions in combinations(range(len(row)),distance):
            for values in product(*(neighbors[j] for j in positions)):
                new=list(row);edits=[]
                for j,value in zip(positions,values):new[j]=value;edits.append((j,row[j],value))
                yield tuple(new),tuple(edits)
        if distance==2:
            for j,old in enumerate(row):
                if old:
                    new=list(row);new[j]=-old;yield tuple(new),((j,old,-old),)


@lru_cache(maxsize=64)
def _prepared(c):return tuple(pacf(c)),sum((v!=0)<<i for i,v in enumerate(c)),sum(c)


def coarse_target(c,k,eta):
    c=validate_coarse_row(c);d=len(c)
    if type(k) is not int or not 1<=k<d or type(eta) is not int or eta not in (-1,1):raise ValueError('strict original target required')
    nc,mask,sigma=_prepared(c);weight=d-nc[0];square=d+2*eta-sigma*sigma
    if square<0 or isqrt(square)**2!=square:return None
    other_sigma=isqrt(square)
    if other_sigma>weight or (weight-other_sigma)%2:return None
    tau=k*((d+1)//2)%d
    wanted=(weight,)+tuple(eta*int(u in (tau,-tau%d))-nc[u] for u in range(1,d))
    g_target=(d,)+tuple(eta*((-1)**k)*int(u in (tau,-tau%d)) for u in range(1,d))
    negative=[];value=d+2*eta*((-1)**k)
    for left in range(isqrt(max(0,value))+1):
        remainder=value-left*left
        if remainder<0 or isqrt(remainder)**2!=remainder:continue
        right=isqrt(remainder)
        if left<=weight and right<=nc[0] and (weight-left)%2==0 and (nc[0]-right)%2==0:
            negative.append(((weight-left)//2,(nc[0]-right)//2))
    if not negative:return None
    return dict(c_mask=mask,wanted_c_b=wanted,c_b_negative=(weight-other_sigma)//2,
                wanted_g_pair=g_target,g_negative_pairs=tuple(negative),k=k,eta=eta)


def complete_fixed_coarse_row_target(c,*,k,eta,certified_psd=True,prime_norm=True,support_bounds=False,
                                     max_assignments=2**31,on_support=None,two_adic=False,norm_unsat_cache=None,dihedral_support=False,cut_projection=False,norm_carrier_cache=None,unary_batch_free=-1,all_target_psd=False,prebatch_dihedral=False,unary_affine_audit=False,unary_linear_certificate=False,native_support_audit=False):
    c=validate_coarse_row(c);d=len(c);tick=perf_counter();spec=coarse_target(c,k,eta)
    if (any(type(v) is not bool for v in (certified_psd,prime_norm,support_bounds,two_adic,dihedral_support,cut_projection,all_target_psd,prebatch_dihedral,unary_affine_audit,unary_linear_certificate,native_support_audit))
            or (unary_affine_audit and unary_linear_certificate)
            or type(max_assignments) is not int or not 1<=max_assignments<=2**31
            or (norm_unsat_cache is not None and not isinstance(norm_unsat_cache,CompleteUnaryNormUnsatCache))
            or (norm_carrier_cache is not None and not isinstance(norm_carrier_cache,UnaryNormCarrierCache))
            or type(unary_batch_free) is not int or not -1<=unary_batch_free<=12
            or (on_support is not None and not callable(on_support))):
        raise ValueError('boolean switches, assignment budget 1..2**31 and optional callback required')
    out=dict(k=k,eta=eta,status='UNSAT_FIXED_SIGNED_C_A',complete=True,words=None,support_orbits=0,
             sign_fibers=0,assignments=0,unknown_sign_fibers=0,coarse_witness_supports=0,profile=spec,
             two_adic=two_adic,
             cut_projection=cut_projection,
             c_b_norm_unsat=False,norm_unsat_cache_hit=False,
             norm_carrier_cache_hit=False,reused_c_b_carriers=0,
             unary_batch_free=unary_batch_free,batch_c_b_witnesses_used=0,
             dihedral_support=dihedral_support,dihedral_support_hits=0,
             prebatch_dihedral=prebatch_dihedral,prebatch_dihedral_hits=0,
             unary_affine_audit=unary_affine_audit,
             unary_linear_certificate=unary_linear_certificate,
             native_support_audit=native_support_audit,
             scope='Signed C_A fixed; all G_A signs and all of binary B free. All C_B support rotations represented; first full model when SAT.')
    if spec is None:return dict(out,reason='exact positive-scalar gauge content obstruction',elapsed=perf_counter()-tick)
    wanted=spec['wanted_c_b']
    key=wanted,spec['c_b_negative']
    if norm_unsat_cache is not None:
        proof=norm_unsat_cache._get(key)
        if proof is not None:return dict(out,c_b_norm_unsat=True,norm_unsat_cache_hit=True,
            norm_unsat_certificate=proof,reason='cached exact SINGLE C_B norm/count exhaustion',elapsed=perf_counter()-tick)
    def excluded(reason,**proofs):
        result=dict(out,c_b_norm_unsat=True,reason=reason,elapsed=perf_counter()-tick,**proofs)
        if norm_unsat_cache is not None:norm_unsat_cache._remember(key,result)
        return result
    if all_target_psd:
        from .coarse_character import coarse_all_target_psd_obstruction
        proof=coarse_all_target_psd_obstruction(c)
        if proof is not None:return excluded(proof['reason'],all_target_psd_certificate=proof)
    if any(abs(v)>wanted[0] for v in wanted[1:]):return excluded('single C_B norm magnitude obstruction')
    if certified_psd:
        proof=signed_norm_psd_obstruction(wanted)
        if proof is not None:return excluded(proof['reason'],psd_certificate=proof)
    if prime_norm:
        proof=prime_norm_obstruction(wanted)
        if proof is not None:return excluded('SINGLE C_B norm p-squared obstruction',prime_certificate=proof)
    if two_adic:
        from .two_adic_norm_completion import two_adic_norm_obstruction
        proof=two_adic_norm_obstruction(wanted)
        if proof is not None:return excluded('SINGLE C_B norm mod-four obstruction',two_adic_certificate=proof)
    cached=norm_carrier_cache._get(key) if norm_carrier_cache is not None else None
    batch_orbit_cache=None
    if cached is None:
        desired=sum((v%2)<<i for i,v in enumerate(wanted))
        family=rotation_representatives(cyclic_norm_family(desired,d))
        supports=(signed_norm_supports(family,wanted,spec['c_b_negative'],native_audit=native_support_audit) if support_bounds else all_weighted_norm_supports(family,wanted[0]))
        candidates=((mask,None,None) for mask in supports['masks'])
        eligible=len(supports['masks'])
        out.update(norm_family_size=family.size,norm_visited=supports['visited'],raw_weight_support_orbits=supports['weight_matches'],
                   eligible_support_orbits=eligible,support_exclusions=supports.get('exclusions',()),support_bounds=support_bounds)
        if unary_batch_free>=0:
            from .unary_norm_batch import complete_unary_norm_batch,CompleteUnarySupportOrbitUnsatCache
            batch_orbit_cache=CompleteUnarySupportOrbitUnsatCache() if prebatch_dihedral else None
            totals=dict(input_supports=0,unsat_supports=0,sat_supports=0,deferred_supports=0,
                        native_input_supports=0,dihedral_unsat_supports=0,
                        raw_candidates_inspected=0,auditor_candidates_inspected=0,elapsed=0.0)
            out['unary_batch']=dict(totals,chunk_size=256,all_planned_supports_processed=eligible==0)
            def batched_candidates():
                # Bound latency before the next G attempt. No whole norm
                # family has to finish before a first full model is tested.
                for offset in range(0,eligible,256):
                    batch=complete_unary_norm_batch(supports['masks'][offset:offset+256],d,wanted,
                        spec['c_b_negative'],max_assignments,free_limit=unary_batch_free,dihedral_unsat_cache=batch_orbit_cache,affine_audit=unary_affine_audit,linear_certificate=unary_linear_certificate)
                    out['prebatch_dihedral_hits']+=batch['dihedral_unsat_supports']
                    for name in totals:totals[name]+=batch[name]
                    out['unary_batch'].update(totals,all_planned_supports_processed=totals['input_supports']==eligible)
                    out['support_orbits']+=batch['unsat_supports'];out['sign_fibers']+=batch['input_supports']
                    out['assignments']+=batch['raw_candidates_inspected']+batch['auditor_candidates_inspected']
                    summary={name:value for name,value in batch.items() if name not in ('candidates','certificate_base64')}
                    if on_support is not None:on_support(dict(kind='C_B_BATCH',support_offset=offset,wanted=wanted,
                        negative=spec['c_b_negative'],**summary,certificate_base64=batch['certificate_base64']))
                    for mask,word,seen in batch['candidates']:yield mask,word,'batch' if word is not None else None
            candidates=batched_candidates()
    else:
        candidates=((mask,word,'cache' if word is not None else None) for mask,word in cached['candidates']);proof=cached['proof'];eligible=proof['eligible_support_orbits']
        out.update(norm_carrier_cache_hit=True,norm_carrier_certificate=proof,
                   norm_family_size=proof['norm_family_size'],norm_visited=0,
                   raw_weight_support_orbits=proof['raw_weight_support_orbits'],eligible_support_orbits=eligible,
                   support_exclusions=(),support_bounds=support_bounds,
                   cached_support_candidates=len(cached['candidates']),
                   skipped_complete_c_b_support_domains=eligible-len(cached['candidates']))
    all_bits=(1<<d)-1;closed_orbits={};retained=[]
    def keep(mask,word):
        nonlocal retained
        if norm_carrier_cache is None or retained is None:return
        if len(retained)>=norm_carrier_cache.slot_budget:retained=None;return
        retained.append((mask,word))
    for mask,cached_word,origin in candidates:
        orbit=dihedral_support_key(mask,d) if dihedral_support else None
        if orbit is not None and orbit in closed_orbits:
            out['dihedral_support_hits']+=1
            source_mask,c_b_empty=closed_orbits[orbit]
            if not c_b_empty:keep(mask,cached_word) # G-only emptiness cannot remove a C_B candidate at another target.
            elif cached_word is not None:raise RuntimeError('cached norm carrier contradicts a complete C_B orbit exclusion')
            if on_support is not None:on_support(dict(mask=mask,dihedral_unsat_source=source_mask,c_b_norm_orbit_empty=c_b_empty,
                scope='Independent B row rotation/reflection preserves C_B/G_B norms and exact counts. No UNKNOWN or SAT orbit skipped.'))
            continue
        out['support_orbits']+=1;attempts=[]
        if cached_word is None:
            coarse=unary_sign_completion(mask,d,wanted,spec['c_b_negative'],max_assignments,two_adic=two_adic,cut_projection=cut_projection)
        else:
            coarse=dict(status='SAT_COMPRESSED',words=(cached_word,(0,)*d),assignments=0,
                        reason=('audited SINGLE C_B norm carrier reused' if origin=='cache' else 'complete native small-affine C_B batch witness')+'; G equations freshly solved')
            out['reused_c_b_carriers' if origin=='cache' else 'batch_c_b_witnesses_used']+=1
        if origin!='batch':out['sign_fibers']+=1
        out['assignments']+=coarse['assignments'];attempts.append(dict(kind='C_B',result=coarse))
        if coarse['status']=='UNKNOWN':out['unknown_sign_fibers']+=1;out['complete']=False;keep(mask,None)
        if coarse['status']!='SAT_COMPRESSED':
            if batch_orbit_cache is not None and coarse['status']=='UNSAT_FIXED_SUPPORT' and coarse.get('complete',True):
                # Original scalar kernels encode terminal exhaustion in the
                # UNSAT status; some omit a separate complete field.
                batch_orbit_cache.remember_full_exclusion(mask,dict(coarse,complete=True),d)
            if orbit is not None and coarse['status']=='UNSAT_FIXED_SUPPORT':closed_orbits[orbit]=(mask,True)
            if on_support is not None:on_support(dict(mask=mask,attempts=attempts))
            continue
        cb=tuple(coarse['words'][0]);out['coarse_witness_supports']+=1
        if tuple(pacf(cb))!=wanted or cb.count(-1)!=spec['c_b_negative'] or sum((v!=0)<<i for i,v in enumerate(cb))!=mask:
            raise RuntimeError('C_B unary norm witness failed independent all-shift audit')
        keep(mask,cb)
        g_masks=all_bits^spec['c_mask'],all_bits^mask
        g_unknown=False
        for negative in spec['g_negative_pairs']:
            plan=prepare_native_system(g_masks,d,spec['wanted_g_pair'],negative)
            if cut_projection and plan['consistent'] and plan['estimated_assignments']>=4096:
                from .support_cut_projection import support_cut_obstruction
                proof=support_cut_obstruction(g_masks,d,spec['wanted_g_pair'],negative)
                if proof is not None:
                    out['sign_fibers']+=1
                    attempts.append(dict(kind='G_PAIR',negative=negative,result=dict(status='UNSAT_FIXED_SUPPORT',complete=True,
                        words=None,assignments=0,cut_certificate=proof)))
                    continue
            difference=run_native_system(plan,max_assignments)
            out['sign_fibers']+=1;out['assignments']+=difference['assignments'];attempts.append(dict(kind='G_PAIR',negative=negative,result=difference))
            if difference['status']=='UNKNOWN':out['unknown_sign_fibers']+=1;out['complete']=False;g_unknown=True;continue
            if difference['status']!='SAT_COMPRESSED':continue
            ga,gb=map(tuple,difference['words'])
            if (tuple(x+y for x,y in zip(pacf(ga),pacf(gb)))!=spec['wanted_g_pair']
                    or tuple(r.count(-1) for r in (ga,gb))!=negative
                    or tuple(sum((v!=0)<<i for i,v in enumerate(r)) for r in (ga,gb))!=g_masks):
                raise RuntimeError('G pair signed witness failed independent full integer audit')
            words=reconstruct_crt_binary(c,ga),reconstruct_crt_binary(cb,gb);check=verify_pqcp(*words)
            if not check.is_valid or check.nonzero_shifts!=(k,2*d-k) or check.profile[k]!=4*eta:
                raise RuntimeError('coarse anchored binary pair failed original full verifier')
            if on_support is not None:on_support(dict(mask=mask,attempts=attempts,full_words=words))
            return dict(out,status='SAT_FULL_PQCP',complete=False,words=words,elapsed=perf_counter()-tick)
        if orbit is not None and not g_unknown:closed_orbits[orbit]=(mask,False)
        if on_support is not None:on_support(dict(mask=mask,attempts=attempts))
    if out['unknown_sign_fibers']:out['status']='UNKNOWN_FIXED_SIGNED_C_A'
    result=dict(out,c_b_norm_unsat=out['complete'] and out['coarse_witness_supports']==0,elapsed=perf_counter()-tick)
    if norm_unsat_cache is not None and result['c_b_norm_unsat']:norm_unsat_cache._remember(key,result)
    if norm_carrier_cache is not None and retained is not None and retained:
        source_proof=cached['proof'] if cached is not None else dict(source_k=k,source_eta=eta,
            wanted_c_b=wanted,negative_c_b=spec['c_b_negative'],norm_family_size=out['norm_family_size'],
            raw_weight_support_orbits=out['raw_weight_support_orbits'],eligible_support_orbits=eligible,
            all_c_b_supports_accounted=True,omitted_unknown_supports=0,
            scope='Every C_B support either completely C_B-UNSAT or retained as SAT/deferred. No G-only exclusion reused across targets.')
        norm_carrier_cache._remember(key,retained,source_proof)
    return result
