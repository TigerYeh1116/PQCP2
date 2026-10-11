/* In F_(p^2), star is Frobenius. A norm/trace quadratic has no conjugate
   pair when its discriminant is a NONZERO square in F_p. No float. */
#include <stdint.h>

static int32_t reduce(int64_t value,int32_t p)
{
    int64_t remainder=value%p;
    return (int32_t)(remainder<0 ? remainder+p : remainder);
}

static int finite_targets(const int32_t *nu,const int32_t *nw,const int32_t *base,
    const uint32_t *coefficients,const uint8_t *nonzero_squares,const int32_t *tau,
    const int32_t *eta,const int32_t *norm_tau,int targets,int d,int p,uint64_t *allowed,int32_t *proofs)
{
    if ((d!=3 && d!=5 && d!=11 && d!=43) || p<3 || p>65000 || (p+1)%d || targets<1 || targets>64) return -1;
    int width=d/2+1;*allowed=targets==64 ? UINT64_MAX : ((UINT64_C(1)<<targets)-1);
    for (int i=0;i<width;++i)
        if (nu[i]<-d || nu[i]>2*d || nw[i]<0 || nw[i]>d || base[i]<-4*d || base[i]>4*d) return -1;
    for (int t=0;t<targets;++t) {
        if (tau[t]<-1 || tau[t]>=width || (eta[t]!=1 && eta[t]!=-1)) return -1;
        if (norm_tau && (norm_tau[t]<-1 || norm_tau[t]>=width)) return -1;
        for (int i=0;i<3;++i) proofs[3*t+i]=0;
    }
    for (int row=0;row<d/2 && *allowed;++row) {
        int64_t n=0,w=0,h=0;
        for (int i=0;i<width;++i) {
            uint32_t c=coefficients[row*width+i];
            n+=(int64_t)nu[i]*c;w+=(int64_t)nw[i]*c;h+=(int64_t)base[i]*c;
        }
        int32_t nr=reduce(n,p),wr=reduce(w,p),hr=reduce(h,p);
        for (int t=0;t<targets;++t) if ((*allowed>>t)&1U) {
            int32_t target_h=hr,target_n=nr;
            if (tau[t]>=0) target_h=reduce((int64_t)target_h+(int64_t)eta[t]*coefficients[row*width+tau[t]],p);
            if (norm_tau && norm_tau[t]>=0)
                target_n=reduce((int64_t)target_n+(int64_t)eta[t]*coefficients[row*width+norm_tau[t]],p);
            int32_t delta=reduce((int64_t)target_h*target_h-(int64_t)4*target_n*wr,p),reason=0;
            if (!wr) { if (target_h) reason=1; }
            else if (delta && nonzero_squares[delta]) reason=2;
            if (reason) {
                proofs[3*t]=reason;proofs[3*t+1]=row;proofs[3*t+2]=delta;
                *allowed &= ~(UINT64_C(1)<<t);
            }
        }
    }
    return 0;
}

int pqcp_finite_trace_targets(const int32_t *nu,const int32_t *nw,const int32_t *base,
    const uint32_t *coefficients,const uint8_t *nonzero_squares,const int32_t *tau,
    const int32_t *eta,int targets,int d,int p,uint64_t *allowed,int32_t *proofs)
{
    return finite_targets(nu,nw,base,coefficients,nonzero_squares,tau,eta,0,targets,d,p,allowed,proofs);
}

int pqcp_finite_affine_targets(const int32_t *nu,const int32_t *nw,const int32_t *base,
    const uint32_t *coefficients,const uint8_t *nonzero_squares,const int32_t *tau,
    const int32_t *eta,const int32_t *norm_tau,int targets,int d,int p,uint64_t *allowed,int32_t *proofs)
{
    return finite_targets(nu,nw,base,coefficients,nonzero_squares,tau,eta,norm_tau,targets,d,p,allowed,proofs);
}
