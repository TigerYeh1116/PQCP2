"""Complete cyclic norm PLUS symmetric cross-trace preimages over F2.

In each self-reciprocal field, solve u*u^q=N and w*u^q+w^q*u=h.
For nonzero w,h, put u=(h/w^q) z: z^q=z+1 and z(z+1)=N*Norm(w)/h^2.
This leaves at most TWO roots per component, hence at most EIGHT literal
L43 masks for generic partners. Zero partners use the complete norm fallback.
The trace anchors the phase; no rotation quotient is permitted.
"""
from functools import lru_cache
from itertools import product

from .cyclic_support import (support_ring,field_mul,field_pow,poly_divmod,poly_mul,
                             _norm_options)
from .sds_norm_completion import cyclic_binary_norm
from .skew_norm_completion import _skew_field_roots


def reverse_mask(mask,d):return sum(((mask>>i)&1)<<((-i)%d) for i in range(d))


def _cyclic_cross_trace_polynomial(mask,partner,d):
    modulus=(1<<d)|1
    return poly_divmod(poly_mul(mask,reverse_mask(partner,d))^
                       poly_mul(reverse_mask(mask,d),partner),modulus)[1]


def cyclic_cross_trace(mask,partner,d):
    """Exact symmetric cross-correlation parity, using literal bit rotations."""
    if (type(d) is not int or d<3 or not d%2
            or any(type(v) is not int or not 0<=v<1<<d for v in (mask,partner))):
        raise ValueError('odd length and literal binary support masks required')
    all_bits=(1<<d)-1;out=0
    for u in range(1,d//2+1):
        plus=((partner<<u)|(partner>>(d-u)))&all_bits
        minus=((partner<<(d-u))|(partner>>u))&all_bits
        if (bin(mask&plus).count('1')+bin(mask&minus).count('1'))%2:
            out|=(1<<u)|(1<<(d-u))
    return out


@lru_cache(maxsize=4096)
def _inverse(value,factor):
    if not value:raise ValueError('nonzero finite-field inverse required')
    return field_pow(value,(1<<(factor.bit_length()-1))-2,factor)


@lru_cache(maxsize=8)
def _frobenius_columns(factor):
    degree=factor.bit_length()-1;q=1<<(degree//2)
    return tuple(field_pow(1<<i,q,factor) for i in range(degree))


def _frobenius(value,factor):
    columns=_frobenius_columns(factor);out=0
    while value:
        bit=value&-value;out^=columns[bit.bit_length()-1];value^=bit
    return out


@lru_cache(maxsize=4096)
def _partner_data(w,factor):
    star=_frobenius(w,factor)
    return star,field_mul(w,star,factor)


@lru_cache(maxsize=8)
def _lift_columns(d,factor,idempotent):
    modulus=(1<<d)|1
    return tuple(poly_divmod(poly_mul(1<<i,idempotent),modulus)[1]
                 for i in range(factor.bit_length()-1))


def _lift(value,columns):
    out=0
    while value:
        bit=value&-value;out^=columns[bit.bit_length()-1];value^=bit
    return out


def norm_trace_preimages(desired,trace,partner,d=43,*,native=True):
    if type(native) is not bool:raise ValueError('native backend must be a boolean')
    ring=support_ring(d)
    if any(type(v) is not int or not 0<=v<1<<d for v in (desired,trace,partner)):
        raise ValueError('literal norm, trace and partner masks required')
    if reverse_mask(desired,d)!=desired or reverse_mask(trace,d)!=trace:
        raise ValueError('symmetric norm and cross-trace targets required')
    if bin(trace).count('1')%2:
        return dict(complete=True,masks=(),reason='symmetric cross-trace scalar must be zero')
    q=1<<(ring.degree//2);groups=[];fallback_components=0
    modulus=(1<<d)|1
    for factor,idempotent,generator in zip(ring.factors[1:],ring.idempotents[1:],ring.norm_generators):
        n=poly_divmod(desired,factor)[1];h=poly_divmod(trace,factor)[1];w=poly_divmod(partner,factor)[1]
        if not w:
            if h:return dict(complete=True,masks=(),reason='zero partner component cannot supply nonzero trace',factor=factor)
            fallback_components+=1
            groups.append(_norm_options(d,ring.degree,factor,idempotent,generator,n));continue
        w_star,norm_w=_partner_data(w,factor)
        if native:
            from .norm_trace_native import field_norm_trace_roots
            roots=field_norm_trace_roots(n,h,w,factor)
        elif not h:
            quotient=field_mul(n,_inverse(norm_w,factor),factor)
            v=field_pow(quotient,q//2,factor)
            roots=(field_mul(w,v,factor),)
        else:
            z_norm=field_mul(field_mul(n,norm_w,factor),_inverse(field_mul(h,h,factor),factor),factor)
            scale=field_mul(h,_inverse(w_star,factor),factor)
            roots=tuple(field_mul(scale,z,factor) for z in _skew_field_roots(factor).get(z_norm,()))
        if not roots:return dict(complete=True,masks=(),reason='field norm/trace fiber empty',factor=factor)
        # Native roots are field-audited inside C. The independently computed
        # Python literal cyclic audit below still checks every CRT output.
        if not native:
            for root in roots:
                root_star=_frobenius(root,factor)
                if (field_mul(root,root_star,factor)!=n
                        or field_mul(w,root_star,factor)^field_mul(w_star,root,factor)!=h):
                    raise RuntimeError('field root failed exact norm and trace audit')
        columns=_lift_columns(d,factor,idempotent)
        groups.append(tuple(_lift(v,columns) for v in roots))
    base=(bin(desired).count('1')%2)*ring.idempotents[0];masks=[]
    for parts in product(*groups):
        mask=base
        for part in parts:mask^=part
        if cyclic_binary_norm(mask,d)!=desired or cyclic_cross_trace(mask,partner,d)!=trace:
            raise RuntimeError('norm/trace CRT reconstruction failed literal cyclic audit')
        masks.append(mask)
    return dict(complete=True,masks=tuple(masks),norm_target=desired,trace_target=trace,
                partner=partner,fallback_components=fallback_components,
                rotation_quotient=False,
                scope='ALL literal cyclic F2 norm and cross-trace preimages. '
                      'Exact counts/PACFs still require integer verification; no global exclusion.')
