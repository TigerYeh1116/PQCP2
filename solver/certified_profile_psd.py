"""Integer upper bounds certifying impossible binary intersection profiles.

For a binary support U, Fourier(N_U)(j)=|Fourier(U)(j)|^2 >= 0.
Cosines are bounded ABOVE using rational Machin pi bounds and an even
Taylor truncation. No float, epsilon, sampled PSD or FFT roundoff can prune.
If the integer upper bound is strictly negative, the profile is impossible.
"""
from fractions import Fraction
from functools import lru_cache

SCALE=1<<40


def _atan_bounds(x,terms=20):
    partial=sum(((-1)**n*x**(2*n+1)/ (2*n+1) for n in range(terms)),Fraction(0))
    following=partial+(-1)**terms*x**(2*terms+1)/(2*terms+1)
    return min(partial,following),max(partial,following)


@lru_cache(maxsize=1)
def rational_pi_bounds():
    a,b=_atan_bounds(Fraction(1,5)),_atan_bounds(Fraction(1,239))
    lower,upper=16*a[0]-4*b[1],16*a[1]-4*b[0]
    # Machin: tan(4 atan(1/5)-atan(1/239)) = 1 in (0,pi/2).
    t=Fraction(1,5)
    for unused in range(2):t=2*t/(1-t*t)
    if (t-Fraction(1,239))/(1+t/Fraction(239))!=1:
        raise RuntimeError('Machin identity rational audit failed')
    if not Fraction(3)<lower<upper<Fraction(22,7):raise RuntimeError('pi bracket audit failed')
    return lower,upper


@lru_cache(maxsize=4)
def cosine_upper_grid(d):
    if d not in (3,5,11,43):raise ValueError('supported odd prime length required')
    pi_lower,pi_upper=rational_pi_bounds()
    if pi_upper*pi_upper>=12:raise RuntimeError('alternating cosine tail monotonicity unproved')
    bounds=[]
    for phase in range(d//2+1):
        angle=2*pi_lower*phase/d
        term=Fraction(1);partial=term
        # Even truncation at n=16 lies ABOVE cosine. The actual phase angle
        # is >= angle and < pi; cosine decreases there, so this also bounds
        # the true cosine at 2*pi*phase/d above.
        for n in range(1,17):
            term=-term*angle*angle/((2*n-1)*(2*n));partial+=term
        value=partial*SCALE
        bounds.append(-(-value.numerator//value.denominator))
    return tuple(bounds)


@lru_cache(maxsize=128)
def _frequency_coefficients(d,frequency):
    upper=cosine_upper_grid(d)
    return tuple(upper[min((frequency*u)%d,(-frequency*u)%d)] for u in range(1,d//2+1))


def sparse_profile_psd_obstructions(base,offsets):
    """Exactly the individual certificates for sparse affine profile families.

    Each actual N=base+eta*(delta_tau+delta_-tau) must be nonnegative.
    Linearity shares dot products, including their independent Python audit.
    A signed base is not itself asserted to be an upper spectral bound.
    """
    base=tuple(base);offsets=tuple(offsets);d=len(base)
    if (d not in (3,5,11,43) or any(type(v) is not int or not -d<=v<=2*d for v in base)
            or any(base[u]!=base[-u] for u in range(1,d))
            or any(type(t) is not int or not 1<=t<=d//2 or type(e) is not int or e not in (-1,1) for t,e in offsets)):
        raise ValueError('supported bounded symmetric base and sparse signed delta positions required')
    profiles=[]
    for tau,eta in offsets:
        values=list(base);values[tau]+=eta;values[-tau]+=eta;wanted=tuple(values)
        if any(not 0<=v<=d for v in wanted):raise ValueError('each actual profile must have coefficients in 0..d')
        profiles.append(wanted)
    from .joint_gram_pruning import native_sparse_profile_psd
    output=[];audited={}
    for start in range(0,len(offsets),64):
        found=native_sparse_profile_psd(base,offsets[start:start+64])
        for index,item in enumerate(found,start):
            if item is None:output.append(None);continue
            frequency,native_numerator=item;coefficients=_frequency_coefficients(d,frequency)
            if frequency not in audited:
                audited[frequency]=base[0]*SCALE+2*sum(base[u]*coefficients[u-1] for u in range(1,d//2+1))
            tau,eta=offsets[index];numerator=audited[frequency]+2*eta*coefficients[tau-1]
            if numerator!=native_numerator or numerator>=0:
                raise RuntimeError('affine native PSD certificate failed independent Python integer reconstruction')
            output.append(dict(frequency=frequency,upper_bound_numerator=numerator,denominator=SCALE,
                cosine_upper_numerators=coefficients,profile=profiles[index],
                proof='Nonnegative N coefficients times outward rational cosine bounds give PSD<=upper<0.'))
    return tuple(output)


def profile_psd_obstruction(wanted,*,native=True):
    """Return a literal negative upper-bound certificate or None.

    None means this necessary test did not exclude the profile, not feasible.
    Nonnegative intersection coefficients are required; range checks precede
    this call in exact norm completion.
    """
    wanted=tuple(wanted);d=len(wanted)
    if type(native) is not bool or d not in (3,5,11,43) or any(type(v) is not int or v<0 for v in wanted) or any(wanted[u]!=wanted[-u] for u in range(1,d)):
        raise ValueError('nonnegative symmetric integral intersection profile required')
    upper=cosine_upper_grid(d)
    if native and max(wanted)<=d:
        from .joint_gram_pruning import native_profile_psd
        found=native_profile_psd(wanted)
        if found is None:return None
        frequency,native_numerator=found
        coefficients=_frequency_coefficients(d,frequency)
        numerator=wanted[0]*SCALE+2*sum(wanted[u]*coefficients[u-1] for u in range(1,d//2+1))
        if numerator!=native_numerator or numerator>=0:raise RuntimeError('native PSD certificate failed independent Python integer dot product')
        return dict(frequency=frequency,upper_bound_numerator=numerator,denominator=SCALE,
                    cosine_upper_numerators=coefficients,profile=wanted,
                    proof='Nonnegative N coefficients times outward rational cosine bounds give PSD<=upper<0.')
    for frequency in range(1,d//2+1):
        coefficients=_frequency_coefficients(d,frequency)
        numerator=wanted[0]*SCALE+2*sum(wanted[u]*coefficients[u-1] for u in range(1,d//2+1))
        if numerator<0:
            return dict(frequency=frequency,upper_bound_numerator=numerator,denominator=SCALE,
                        cosine_upper_numerators=coefficients,profile=wanted,
                        proof='Nonnegative N coefficients times outward rational cosine bounds give PSD<=upper<0.')
    return None
