/* Outward rational Fourier intervals and a 2x2 spectral Gram obstruction.
   |Fourier(D)|^2 <= 4 Fourier(N_U) Fourier(N_W). No floating point.
   __int128 safely holds products of 43-point integer bounds at scale 2^40. */
#include <stdint.h>

/* All profiles are base + eta*(delta_tau + delta_-tau).  The signed
   base is only a common linear-expression origin, never a PSD bound by
   itself. Every actual profile is range-checked before the call. */
int pqcp_sparse_profile_psd(const int32_t *base, const int32_t *tau,
    const int32_t *eta, int count, const int64_t *cos_upper, int d,
    int32_t *frequencies, int64_t *numerators)
{
    if ((d!=3 && d!=5 && d!=11 && d!=43) || count<1 || count>64) return -1;
    for (int u=0;u<d;++u) if (base[u]<-d || base[u]>2*d) return -1;
    for (int t=0;t<count;++t) {
        if (tau[t]<1 || tau[t]>d/2 || (eta[t]!=1 && eta[t]!=-1)) return -1;
        for (int u=0;u<d;++u) {
            int value=base[u]+((u==tau[t] || u==d-tau[t]) ? eta[t] : 0);
            if (value<0 || value>d) return -1;
        }
        frequencies[t]=0;numerators[t]=0;
    }
    for (int f=1;f<=d/2;++f) {
        int64_t value=base[0]*(INT64_C(1)<<40);
        for (int u=1;u<=d/2;++u) {
            int phase=(f*u)%d;if (phase>d/2) phase=d-phase;
            value+=INT64_C(2)*base[u]*cos_upper[phase];
        }
        for (int t=0;t<count;++t) if (!frequencies[t]) {
            int phase=(f*tau[t])%d;if (phase>d/2) phase=d-phase;
            int64_t actual=value+INT64_C(2)*eta[t]*cos_upper[phase];
            if (actual<0) {frequencies[t]=f;numerators[t]=actual;}
        }
    }
    return 0;
}

int pqcp_profile_psd(const int32_t *profile, const int64_t *cos_upper, int d,
                     int32_t *frequency_out, int64_t *numerator_out)
{
    if (d!=3 && d!=5 && d!=11 && d!=43) return -1;
    for (int i=0;i<d;++i) if (profile[i]<0 || profile[i]>d) return -1;
    *frequency_out=0;*numerator_out=0;
    for (int frequency=1;frequency<=d/2;++frequency) {
        int64_t upper=profile[0]*(INT64_C(1)<<40);
        for (int u=1;u<=d/2;++u) {
            int phase=(frequency*u)%d;if (phase>d/2) phase=d-phase;
            upper+=INT64_C(2)*profile[u]*cos_upper[phase];
        }
        if (upper<0) {*frequency_out=frequency;*numerator_out=upper;return 0;}
    }
    return 0;
}

static void save_certificate(uint64_t *out, int code, int frequency, __int128 value)
{
    unsigned __int128 encoded=(unsigned __int128)value;
    out[0]=(uint64_t)code;out[1]=(uint64_t)frequency;
    out[2]=(uint64_t)encoded;out[3]=(uint64_t)(encoded>>64);
}

int pqcp_joint_gram_targets(const int32_t *nu, const int32_t *nw, const int32_t *base,
    const int64_t *cos_lower, const int64_t *cos_upper,
    const int32_t *tau, const int32_t *eta, int targets, int d,
    uint64_t *allowed, uint64_t *certificates)
{
    if ((d!=3 && d!=5 && d!=11 && d!=43) || targets<1 || targets>64) return -1;
    int half=d/2;int64_t scale=INT64_C(1)<<40;
    *allowed=targets==64 ? UINT64_MAX : ((UINT64_C(1)<<targets)-1);
    for (int t=0;t<targets;++t) {
        if (tau[t]<-1 || tau[t]>half || (eta[t]!=1 && eta[t]!=-1)) return -1;
        for (int i=0;i<4;++i) certificates[4*t+i]=0;
    }
    for (int u=0;u<d;++u)
        if (nu[u]<0 || nu[u]>d || nw[u]<0 || nw[u]>d || base[u]<-4*d || base[u]>4*d) return -1;
    for (int frequency=1;frequency<=half && *allowed;++frequency) {
        int64_t upper_u=nu[0]*scale, upper_w=nw[0]*scale;
        int64_t lower_d=base[0]*scale,upper_d=lower_d;
        for (int u=1;u<=half;++u) {
            int phase=(frequency*u)%d;if (phase>half) phase=d-phase;
            upper_u += INT64_C(2)*nu[u]*cos_upper[phase];
            upper_w += INT64_C(2)*nw[u]*cos_upper[phase];
            lower_d += INT64_C(2)*base[u]*(base[u]>=0 ? cos_lower[phase] : cos_upper[phase]);
            upper_d += INT64_C(2)*base[u]*(base[u]>=0 ? cos_upper[phase] : cos_lower[phase]);
        }
        if (upper_w<0 || lower_d>upper_d) return -2;
        if (upper_u<0) {
            for (int t=0;t<targets;++t) if ((*allowed>>t)&1U)
                save_certificate(certificates+4*t,1,frequency,upper_u);
            *allowed=0;break;
        }
        for (int t=0;t<targets;++t) if ((*allowed>>t)&1U) {
            int64_t lo=lower_d,hi=upper_d;
            if (tau[t]>=0) {
                int phase=(frequency*tau[t])%d;if (phase>half) phase=d-phase;
                if (eta[t]>0) { lo+=2*cos_lower[phase];hi+=2*cos_upper[phase]; }
                else { lo-=2*cos_upper[phase];hi-=2*cos_lower[phase]; }
            }
            int64_t minimum_abs=lo>0 ? lo : (hi<0 ? -hi : 0);
            __int128 upper=(__int128)4*upper_u*upper_w-(__int128)minimum_abs*minimum_abs;
            if (upper<0) {
                save_certificate(certificates+4*t,2,frequency,upper);
                *allowed &= ~(UINT64_C(1)<<t);
            }
        }
    }
    return 0;
}
