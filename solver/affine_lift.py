"""Exact, resumable factor-2 lift via necessary GF(2) orientation equations.

For L=2d, c[i]=x[i]+x[i+d], h[i]=(x[i]-x[i+d])/2, and
P_x(u)=P_c(u)/2+2*N_h(u), P_x(u+d)=P_c(u)/2-2*N_h(u).
N is NEGACYCLIC (wrap changes sign), not periodic autocorrelation.
With d even, residual R(u)=(T(u)-T(u+d))/4 gives mod-4 linear
conditions on the zero-entry orientations. These are necessary only.
UNSAT requires inconsistency or complete exact full-PACF enumeration.
Limits are UNKNOWN; this module does not call Z3 or inspect known PQCPs.
"""

from dataclasses import dataclass
from time import perf_counter
from typing import Iterator, Optional, Sequence, Tuple

from .compressed_lift import (
    CompressedLiftResult, _compressed_word, _validate_content,
)
from .structured_energy import (
    compress_binary_sequence, fold_profile,
    integer_periodic_autocorrelation_profile as pacf, target_profile,
)
from .target_profiles import TargetContentProfile, pair_content
from .verifier import verify_pqcp

ORDER_VERSION = 1
popcount = int.bit_count if hasattr(int, "bit_count") else lambda value: bin(value).count("1")


@dataclass(frozen=True)
class AffineLiftResult(CompressedLiftResult):
    """The cursor is the next untested assignment, not a success estimate."""

    rank: int = 0
    free_bits: int = 0
    assignments: int = 0
    next_index: int = 0
    examined: int = 0
    deferred: bool = False
    scan_backend: str = "python"


def orientation_parity_system(ma: Sequence[int], mb: Sequence[int], target: Sequence[int]) -> dict:
    """Exact necessary mod-4 equations on the remaining sign choices.

    Each residual equation is sum_e s_e*y_i*y_j = R, with M terms, s_e=+/-1
    known from the negacyclic wrap and y_i=(-1)^b_i. It must have exactly
    (M-R)/2 negative terms. Multiplying all terms yields a LINEAR GF(2)
    equation: XOR of incident b_i with odd multiplicity equals
    ((M-R)/2 mod 2) XOR (number of negative s_e mod 2).
    Gaussian elimination is exact. Inconsistency proves this FIXED middle
    problem impossible. Consistency is necessary, not sufficient for PACF.
    The rank counts independent parity equations, not quadratic equations.
    """
    ma,mb=tuple(ma),tuple(mb)
    d=len(ma)
    if not d or d%2 or len(mb)!=d or len(target)!=d or any(type(x) is not int or x not in (-2,0,2) for x in ma+mb):
        raise ValueError('equal even-length factor-2 words and residual target required')
    variables=[(s,i) for s,w in enumerate((ma,mb)) for i,x in enumerate(w) if x==0]
    ids={v:j for j,v in enumerate(variables)}
    basis={}
    consistent=True
    raw=[]
    for u in range(1,d//2):
        mask=terms=wraps=0
        for s,w in enumerate((ma,mb)):
            for i in range(d):
                j=(i+u)%d
                if w[i]==w[j]==0:
                    terms+=1
                    wraps^=int(i+u>=d)
                    mask^=(1<<ids[s,i])^(1<<ids[s,j])
        R=target[u]
        if type(R) is not int or (terms-R)%2 or abs(R)>terms:
            consistent=False
            break
        rhs=(((terms-R)//2)&1)^wraps
        raw.append(dict(shift=u,mask=mask,rhs=rhs,terms=terms))
        while mask:
            pivot=mask.bit_length()-1
            if pivot not in basis:
                basis[pivot]=(mask,rhs)
                break
            other,bit=basis[pivot]
            mask^=other;rhs^=bit
        if not mask and rhs:
            consistent=False
            break
    return dict(variables=variables, equations=raw,
                basis=[(p,*basis[p]) for p in sorted(basis)], rank=len(basis),
                consistent=consistent,
                affine_assignments=(1<<(len(variables)-len(basis))) if consistent else 0)


def affine_basis(system: dict) -> Tuple[int, tuple]:
    """Return one solution and a complete nullspace basis for triangular rows."""
    if not system['consistent']:
        raise ValueError('inconsistent affine system')
    def complete(word,homogeneous):
        for pivot,mask,rhs in system['basis']:
            if (bin(word & (mask^(1<<pivot))).count('1')&1) ^ (0 if homogeneous else rhs):
                word^=1<<pivot
        return word
    pivots={p for p,_,_ in system['basis']}
    return complete(0,False),tuple(complete(1<<f,True)
        for f in range(len(system['variables'])) if f not in pivots)


def affine_bitset_basis(ma: Sequence[int], mb: Sequence[int], system: dict) -> Tuple[tuple,tuple]:
    """Map a particular solution and nullspace vectors to full binary words."""
    particular,kernel=affine_basis(system)
    d=len(ma)
    words=[0,0]
    for s,w in enumerate((ma,mb)):
        for i,value in enumerate(w):
            if value==-2: words[s]|=(1<<i)|(1<<(i+d))
            elif value==0: words[s]|=1<<(i+d)
    def changes(vector):
        masks=[0,0]
        for bit,(s,i) in enumerate(system['variables']):
            if vector>>bit&1: masks[s]^=(1<<i)|(1<<(i+d))
        return masks
    initial=changes(particular)
    words=[w^change for w,change in zip(words,initial)]
    deltas=[changes(v) for v in kernel]
    return tuple(words),tuple(tuple(delta) for delta in deltas)


def affine_binary_pairs(ma: Sequence[int], mb: Sequence[int], system: dict,
                        start_index: int = 0) -> Iterator[Tuple[int, int]]:
    """Yield the Gray-order suffix without replaying already checked assignments."""
    base,deltas=affine_bitset_basis(ma,mb,system)
    words=list(base)
    if type(start_index) is not int or not 0 <= start_index <= system['affine_assignments']:
        raise ValueError('invalid affine cursor')
    gray = start_index ^ (start_index >> 1)
    for bit,delta in enumerate(deltas):
        if gray >> bit & 1:
            words[0] ^= delta[0]
            words[1] ^= delta[1]
    for index in range(start_index,system['affine_assignments']):
        if index > start_index:
            changed=(index & -index).bit_length()-1
            words[0]^=deltas[changed][0];words[1]^=deltas[changed][1]
        yield words[0],words[1]


def solve_affine_lift(
    compressed_a: Sequence[int], compressed_b: Sequence[int], k: int, eta: int,
    timeout_ms: int, content_profile: Optional[TargetContentProfile] = None,
    cross_compressions: Optional[Sequence[Tuple[tuple, tuple]]] = None,
    max_assignments: int = 1048576, start_index: int = 0,
    max_steps: Optional[int] = None,
    backend: str = "auto",
) -> AffineLiftResult:
    """Check a bounded suffix of the COMPLETE necessary affine space.

    A saved cursor is trusted only as progress from this same immutable
    problem/order version; the durable caller checks that identity. Ordinary
    and parity content and optional cross disjunction are enforced, not
    inferred from a low score. Unsupported odd d or an excessive space is
    deferred to the existing Z3 backend, never pruned. max_steps is an optional
    deterministic work slice useful for resume tests. auto uses the identical
    integer Gray scan in C when available (releasing the GIL in bounded
    chunks); python is the portable reference/rollback backend. Order version
    is unchanged, so saved cursors can move between the two backends.
    """
    started = perf_counter()
    if backend not in ("auto","python","native"):
        raise ValueError("affine backend must be auto, python or native")
    if type(timeout_ms) is not int or timeout_ms <= 0:
        raise ValueError("positive integer time budget required")
    if type(max_assignments) is not int or max_assignments < 1:
        raise ValueError("positive integer affine cap required")
    if type(start_index) is not int or start_index < 0:
        raise ValueError("nonnegative affine cursor required")
    if max_steps is not None and (type(max_steps) is not int or max_steps < 1):
        raise ValueError("positive integer step cap required")
    ca,cb = _compressed_word(compressed_a),_compressed_word(compressed_b)
    if len(ca) != len(cb):
        raise ValueError("compressed length mismatch")
    d,L = len(ca),2*len(ca)
    desired = target_profile(L,k,eta)
    if content_profile is not None:
        _validate_content(content_profile,L,k,eta)
    if cross_compressions is not None:
        from .cross_compression import normalize_cross_pairs
        cross_compressions = normalize_cross_pairs(cross_compressions,L,k,eta)
    cross_set = None if cross_compressions is None else set(cross_compressions)
    common = dict(L=L,k=k,eta=eta,orientation_variables=ca.count(0)+cb.count(0),
                  correlation_constraints=d-1,content_constraints=4 if content_profile else 0,
                  z3_executed=False,cross_pattern_count=len(cross_compressions or ()))
    construction = 0.0
    rank = free = size = examined = 0
    next_index = start_index
    scan_backend = "python"

    def finish(status: str, reason: Optional[str] = None, **extra) -> AffineLiftResult:
        elapsed = perf_counter()-started
        return AffineLiftResult(
            status=status, reason=reason, **common, construction_time=construction,
            solve_time=max(0.0,elapsed-construction), rank=rank, free_bits=free,
            assignments=size,next_index=next_index,examined=examined,
            scan_backend=scan_backend,**extra)

    if tuple(x+y for x,y in zip(pacf(ca),pacf(cb))) != fold_profile(desired,2):
        return finish("UNSAT","fixed middle PACF mismatch")
    if d%2:
        return finish("UNKNOWN","odd half-length: use existing Z3",deferred=True)
    if content_profile is not None:
        actual = tuple(sum(2 if v==-2 else 1 if v==0 else 0 for v in w[p::2])
                       for w in (ca,cb) for p in (0,1))
        expected = (content_profile.a_even_ones,content_profile.a_odd_ones,
                    content_profile.b_even_ones,content_profile.b_odd_ones)
        if actual != expected:
            return finish("UNSAT","fixed middle cannot realize requested parity content")
    residual = tuple((desired[u]-desired[u+d])//4 for u in range(d))
    system = orientation_parity_system(ca,cb,residual)
    rank = system['rank']
    free = len(system['variables'])-rank
    size = system['affine_assignments']
    construction = perf_counter()-started
    if not system['consistent']:
        return finish("UNSAT","inconsistent necessary GF(2) equations")
    if start_index > size:
        raise ValueError("affine cursor beyond complete space")
    if size > max_assignments:
        return finish("UNKNOWN","affine space exceeds cap: use existing Z3",deferred=True)
    deadline = started+timeout_ms/1000
    mask = (1<<L)-1
    wanted = tuple(L-desired[u]//2 for u in range(1,d))

    def audit(a: int, b: int) -> Optional[AffineLiftResult]:
        A,B = (tuple((word>>i)&1 for i in range(L)) for word in (a,b))
        if cross_set is not None:
            rows = tuple(tuple(sum(1-2*v for v in w[r::4]) for r in range(4))
                         for w in (A,B))
            if rows not in cross_set:
                return None
        verification = verify_pqcp(A,B)
        if (not verification.is_valid or verification.profile != desired
                or compress_binary_sequence(A,2) != ca
                or compress_binary_sequence(B,2) != cb):
            raise RuntimeError("affine SAT failed independent target/compression audit")
        if content_profile is not None and pair_content(A,B) != expected:
            raise RuntimeError("affine SAT changed parity content")
        return finish("SAT",a=A,b=B,profile=verification.profile,verified=True)

    scanner = None
    if backend != "python":
        from .affine_native import NativeAffineScan, NativeUnavailable, optional_library, _library
        supported = L<=128 and free<=62
        if backend == "native" and not supported:
            raise NativeUnavailable("affine dimensions exceed native bounds")
        lib = (_library() if backend=="native" else optional_library()) if supported else None
        if lib is not None:
            base,deltas = affine_bitset_basis(ca,cb,system)
            scanner = NativeAffineScan(L,base,deltas,wanted,lib)
            scan_backend = "native"
        construction = perf_counter()-started
    if scanner is not None:
        # Bound each GIL-free call; time checks and interrupts return to Python
        # between chunks. A cross-rejected full PACF match continues at the
        # very next assignment, never restarting/skipping the current chunk.
        while next_index < size:
            if perf_counter() >= deadline or (max_steps is not None and examined>=max_steps):
                return finish("UNKNOWN","affine slice exhausted; exact cursor saved")
            end = min(size,next_index+16384)
            if max_steps is not None:
                end = min(end,next_index+max_steps-examined)
            cursor,pair = scanner.scan(next_index,end)
            examined += cursor-next_index
            next_index = cursor
            if pair is not None:
                verified = audit(*pair)
                if verified is not None:
                    return verified
        return finish("UNSAT","complete affine space rejected by exact full PACF")
    for index,(a,b) in enumerate(affine_binary_pairs(ca,cb,system,start_index),start_index):
        if (examined%256 == 0 and perf_counter() >= deadline) or (
                max_steps is not None and examined >= max_steps):
            return finish("UNKNOWN","affine slice exhausted; exact cursor saved")
        examined += 1
        next_index = index+1
        for u, mismatches in enumerate(wanted,1):
            ra = ((a>>u)|(a<<(L-u))) & mask
            rb = ((b>>u)|(b<<(L-u))) & mask
            if popcount(a^ra)+popcount(b^rb) != mismatches:
                break
        else:
            verified = audit(a,b)
            if verified is not None:
                return verified
    return finish("UNSAT","complete affine space rejected by exact full PACF")
