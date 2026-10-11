"""Exact cyclic-window repair with unrestricted changes to parity contents.

Rotate each row independently to put a chosen free block at the tail. The
boundary projection then checks ALL assignments and strict target positions.
This can cross between canonical content families, unlike fixed-content swaps.
No low score substitutes for full verification; misses exclude this union only.
"""
from time import perf_counter

from .boundary_tail_search import search_boundary_tail_masks
from .correlation import normalize_binary_sequence
from .structured_energy import target_profile
from .verifier import verify_pqcp
from .support_orbit_cache import row_dihedral_mask


def cyclic_window_center_key(a,b):
    """Center quotient preserving contiguous-window neighborhoods EXACTLY.

    Independent dihedral actions, row complementation and row exchange do
    preserve windows. General unit decimation does not and is NOT used.
    """
    a,b=tuple(normalize_binary_sequence(a)),tuple(normalize_binary_sequence(b))
    if not a or len(a)!=len(b):raise ValueError('equal nonempty binary rows required')
    length=len(a);all_bits=(1<<length)-1
    values=[]
    for row in (a,b):
        mask=sum(v << i for i,v in enumerate(row))
        values.append(min(row_dihedral_mask(mask,length),row_dihedral_mask(all_bits^mask,length)))
    return (length,*sorted(values))


def exact_cyclic_window_repair(a,b,*,block_size=4,rotations=None):
    a,b=tuple(normalize_binary_sequence(a)),tuple(normalize_binary_sequence(b))
    length=len(a)
    if (length!=len(b) or not 4<=length<=94 or length%2 or type(block_size) is not int
            or not 1<=block_size<=4 or block_size>=length//2):
        raise ValueError('equal even binary rows and bounded free cyclic blocks required')
    if verify_pqcp(a,b).is_valid:
        raise ValueError('repair centers must be non-solutions, not archived PQCPs')
    rotations=(tuple(range(length)),)*2 if rotations is None else tuple(map(tuple,rotations))
    if (len(rotations)!=2 or any(not r or len(set(r))!=len(r)
            or any(type(v) is not int or not 0<=v<length for v in r) for r in rotations)):
        raise ValueError('two nonempty literal rotation lists required')
    tick=perf_counter()
    all_bits=(1<<length)-1
    prefix_bits=(1<<(length-block_size))-1
    masks=tuple(sum(v << i for i,v in enumerate(row)) for row in (a,b))
    prefixes=tuple({s:((m >> s)|(m << (length-s))) & all_bits & prefix_bits for s in rs}
                   for m,rs in zip(masks,rotations))
    witnesses=[]
    seen=set()
    queries=covered=survivors=checks=0
    for sa in rotations[0]:
        for sb in rotations[1]:
            out=search_boundary_tail_masks((prefixes[0][sa],prefixes[1][sb]),length,block_size)
            if not out['complete']:raise RuntimeError('cyclic-window repair did not complete this prefix')
            queries+=1;covered+=out['covered_tail_assignments']
            survivors+=out['surviving_tail_assignments'];checks+=out['correlation_checks']
            for hit in out['witnesses']:
                words=tuple(tuple(w[(i-s)%length] for i in range(length)) for w,s in zip(hit['words'],(sa,sb)))
                check=verify_pqcp(*words)
                if not check.is_valid or tuple(check.profile)!=target_profile(length,hit['k'],hit['eta']):
                    raise RuntimeError('inverse row rotations failed full original-length target verification')
                windows=tuple({(s+length-block_size+j)%length for j in range(block_size)} for s in (sa,sb))
                if any(old[i]!=new[i] for old,new,window in zip((a,b),words,windows)
                       for i in range(length) if i not in window):
                    raise RuntimeError('window reconstruction changed an outside coordinate')
                if words in seen:continue
                seen.add(words)
                witnesses.append(dict(words=words,k=hit['k'],eta=hit['eta'],rotations=(sa,sb),
                                      hamming_distance=sum(x!=y for old,new in zip((a,b),words) for x,y in zip(old,new))))
    return dict(complete=True,witnesses=witnesses,queries=queries,covered_assignments=covered,
                projected_assignments=survivors,correlation_checks=checks,block_size=block_size,
                elapsed=perf_counter()-tick,
                scope='ALL bit assignments in the selected independent cyclic windows; parity contents can change. '
                      'All strict target lags/signs. Not all arbitrary Hamming-radius configurations or global PQCPs.')
