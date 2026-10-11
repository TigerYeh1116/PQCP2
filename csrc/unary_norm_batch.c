/* Complete small affine sign fibers in batches. Each exclusion is audited
   by the independent original dense-variable planner/sign kernel. */
#include <stdint.h>
#include <string.h>

extern int pqcp_affine_sign_plan(const uint64_t *,int,const int *,const int *,int,int,
    uint64_t *,uint64_t *,uint64_t *,uint64_t *,int *,int *);
extern int pqcp_ternary_signs(const uint64_t *,const uint64_t *,const uint64_t *,int,int,
    const int *,const int *,uint64_t,uint64_t *,uint64_t *);

static uint64_t rotate(uint64_t mask,int shift,int d) {
    return ((mask<<shift)|(mask>>(d-shift)))&((UINT64_C(1)<<d)-1);
}

/* Literal position coordinates, rather than the original planner's compact
   variable ids. The extra cardinality parity equation is necessary. */
static int literal_plan(uint64_t support,int d,const int *rho,int negative,
    uint64_t *base,uint64_t *toggles,int *rank,int *free_count,int *reason) {
    uint64_t rows[43]={0};int rhs[43]={0};*rank=0;*reason=0;
    for (int u=1;u<=d/2+1;++u) {
        uint64_t row;int value;
        if (u==d/2+1) {row=support;value=negative&1;}
        else {
            uint64_t left=rotate(support,u,d),right=rotate(support,d-u,d);
            int intersection=__builtin_popcountll(support&left),delta=intersection-rho[u];
            if (rho[u]>intersection||rho[u]<-intersection||(delta&1)) {*reason=1;return 0;}
            row=support&(left^right);value=(delta/2)&1;
        }
        while (row) {
            int at=63-__builtin_clzll(row);
            if (!rows[at]) {rows[at]=row;rhs[at]=value;++*rank;break;}
            row^=rows[at];value^=rhs[at];
        }
        if (!row&&value) {*reason=2;return 0;}
    }
    uint64_t expressions[43]={0};*base=0;*free_count=0;
    for (int j=0;j<d;++j) {
        if (!(support>>j&1)) continue;
        uint64_t location=UINT64_C(1)<<j;
        if (!rows[j]) expressions[j]=UINT64_C(1)<<(*free_count)++;
        else {
            int constant=rhs[j];uint64_t rest=rows[j]^location;
            while (rest) {
                int at=__builtin_ctzll(rest);expressions[j]^=expressions[at];constant^=(*base>>at)&1;rest&=rest-1;
            }
            if (constant) *base|=location;
        }
    }
    uint64_t flexible=0;
    memset(toggles,0,43*sizeof(uint64_t));
    for (int j=0;j<d;++j) {
        uint64_t expression=expressions[j];
        if (expression) flexible|=UINT64_C(1)<<j;
        while (expression) {int at=__builtin_ctzll(expression);toggles[at]|=UINT64_C(1)<<j;expression&=expression-1;}
    }
    int minimum=__builtin_popcountll(*base&~flexible),maximum=minimum+__builtin_popcountll(flexible);
    if (negative<minimum||negative>maximum) {*reason=3;return 0;}
    return 1;
}

static int exact_norm(uint64_t support,uint64_t negative_mask,int d,const int *rho) {
    for (int u=1;u<=d/2;++u) {
        uint64_t edges=support&rotate(support,u,d);
        int cuts=__builtin_popcountll(edges&(negative_mask^rotate(negative_mask,u,d)));
        if (__builtin_popcountll(edges)-2*cuts!=rho[u]) return 0;
    }
    return 1;
}

int pqcp_verify_unary_affine_certificate(uint64_t support,int d,const int *rho,int negative,
    uint64_t base,const uint64_t *toggles,int q,uint64_t seen) {
    if (!rho||!toggles||d<3||d>43||!(d&1)||q<0||q>12||seen!=(UINT64_C(1)<<q)) return 0;
    uint64_t all=(UINT64_C(1)<<d)-1;
    int weight=__builtin_popcountll(support);
    if ((support&~all)||(base&~support)||rho[0]!=weight||negative<0||negative>weight) return 0;
    for (int u=1;u<d;++u) if (rho[u]<-d||rho[u]>d||rho[u]!=rho[d-u]) return 0;
    uint64_t basis[43]={0},directions[43]={0};int rank=0;
    /* Lower pivots independently check rank. The primary planner uses
       upper pivots and supplies a certificate, not a trusted dimension. */
    for (int u=1;u<=d/2+1;++u) {
        uint64_t row;int rhs;
        if (u==d/2+1) {row=support;rhs=negative&1;}
        else {
            uint64_t left=rotate(support,u,d),right=rotate(support,d-u,d);
            int edges=__builtin_popcountll(support&right),delta=edges-rho[u];
            if (rho[u]>edges||rho[u]<-edges||(delta&1)) return 0;
            row=support&(left^right);rhs=(delta/2)&1;
        }
        if ((__builtin_popcountll(row&base)&1)!=rhs) return 0;
        for (int j=0;j<q;++j) if (__builtin_popcountll(row&toggles[j])&1) return 0;
        uint64_t v=row;
        while (v) {
            int p=__builtin_ctzll(v);
            if (!basis[p]) {basis[p]=v;++rank;break;}
            v^=basis[p];
        }
    }
    if (q!=weight-rank) return 0;
    for (int j=0;j<q;++j) {
        uint64_t v=toggles[j];if (v&~support) return 0;
        while (v) {
            int p=__builtin_ctzll(v);
            if (!directions[p]) {directions[p]=v;break;}
            v^=directions[p];
        }
        if (!v) return 0;
    }
    return 1;
}

static int audit_exclusion(uint64_t support,int d,const int *rho,int negative,
    int reason,uint64_t literal_base,const uint64_t *literal_toggles,int literal_q,
    uint64_t literal_seen,int affine_audit,uint64_t *audit_assignments) {
    uint64_t supports[2]={support,0},base[2],free_toggles[94],patterns[94],checks[96];
    int counts[2]={negative,0},check_rhs[96],metadata[8];
    if (affine_audit==2&&reason>=4) {
        *audit_assignments=0;
        return pqcp_verify_unary_affine_certificate(support,d,rho,negative,literal_base,literal_toggles,literal_q,literal_seen);
    }
    int status=pqcp_affine_sign_plan(supports,d,rho,counts,-1,1,base,free_toggles,patterns,checks,check_rhs,metadata);
    *audit_assignments=0;
    if (status<0) return 0;
    if (reason<=3) return status==0;
    if (!status) return 1;
    int q=metadata[2];if (q>12) return 0;
    if (affine_audit) {
        /* Same-dimensional affine spaces are equal if bases differ by an
           original homogeneous vector and every literal direction lies in
           that space and is independent. The primary loop exhausted 2^q
           literal assignments; equality transfers its exact UNSAT result
           without enumerating the original space a second time. */
        if (q!=literal_q||literal_seen!=(UINT64_C(1)<<q)||base[1]) return 0;
        uint64_t original_basis[43]={0},literal_basis[43]={0};
        for (int j=0;j<q;++j) {
            if (free_toggles[2*j+1]||(free_toggles[2*j]&~support)) return 0;
            uint64_t v=free_toggles[2*j];
            while (v) {
                int p=63-__builtin_clzll(v);
                if (!original_basis[p]) {original_basis[p]=v;break;}
                v^=original_basis[p];
            }
            if (!v) return 0;
        }
        uint64_t delta=literal_base^base[0];
        while (delta) {
            int p=63-__builtin_clzll(delta);
            if (!original_basis[p]) return 0;
            delta^=original_basis[p];
        }
        for (int j=0;j<q;++j) {
            uint64_t v=literal_toggles[j];if (v&~support) return 0;
            uint64_t rest=v;
            while (rest) {
                int p=63-__builtin_clzll(rest);
                if (!original_basis[p]) return 0;
                rest^=original_basis[p];
            }
            while (v) {
                int p=63-__builtin_clzll(v);
                if (!literal_basis[p]) {literal_basis[p]=v;break;}
                v^=literal_basis[p];
            }
            if (!v) return 0;
        }
        return 1;
    }
    uint64_t output[2],stats[9]={0};
    status=pqcp_ternary_signs(supports,base,free_toggles,q,d,rho,counts,UINT64_C(1)<<q,output,stats);
    *audit_assignments=stats[0];
    return status==0;
}

/* Records: support, status|reason<<8|rank<<16|q<<24, negative witness,
   raw affine candidates inspected, independent-auditor candidates.
   Status 0 is complete UNSAT, 1 one SAT witness, 2 deferred to original
   complete solver. q_limit is scheduling only; no high-q fiber is cut. */
int pqcp_unary_norm_batch_certified(const uint64_t *supports,uint64_t count,int d,const int *rho,
    int negative,int q_limit,int affine_audit,uint64_t *records,uint64_t *retained,uint64_t *stats) {
    if (!supports||!rho||!records||!retained||!stats||d<3||d>43||!(d&1)
            ||negative<0||negative>rho[0]||rho[0]<0||rho[0]>d||q_limit<0||q_limit>12
            ||affine_audit<0||affine_audit>2) return -1;
    uint64_t all=(UINT64_C(1)<<d)-1;memset(stats,0,8*sizeof(uint64_t));
    for (uint64_t index=0;index<count;++index) {
        uint64_t support=supports[index],base=0,toggles[43],seen=0,word=0,audit_seen=0;
        if ((support&~all)||__builtin_popcountll(support)!=rho[0]) return -2;
        int rank=0,q=0,reason=0,status=2;
        int consistent=literal_plan(support,d,rho,negative,&base,toggles,&rank,&q,&reason);
        if (!consistent) status=0;
        else if (q<=q_limit) {
            uint64_t mask=base;int count_matches=0;status=0;
            for (uint64_t step=0;step<(UINT64_C(1)<<q);++step) {
                if (step) mask^=toggles[__builtin_ctzll(step)];
                ++seen;
                if (__builtin_popcountll(mask)!=negative) continue;
                count_matches=1;
                if (exact_norm(support,mask,d,rho)) {status=1;word=mask;break;}
            }
            if (!status) reason=count_matches?5:4;
        }
        if (!status&&!audit_exclusion(support,d,rho,negative,reason,base,toggles,q,seen,affine_audit,&audit_seen)) return -3;
        uint64_t *record=records+5*index;
        record[0]=support;record[1]=(uint64_t)status|((uint64_t)reason<<8)|((uint64_t)rank<<16)|((uint64_t)q<<24);
        record[2]=word;record[3]=seen;record[4]=audit_seen;
        ++stats[0];++stats[1+status];stats[5]+=seen;stats[6]+=audit_seen;
        if (status) retained[stats[4]++]=index;
    }
    return 0;
}

int pqcp_unary_norm_batch(const uint64_t *supports,uint64_t count,int d,const int *rho,
    int negative,int q_limit,uint64_t *records,uint64_t *retained,uint64_t *stats) {
    return pqcp_unary_norm_batch_certified(supports,count,d,rho,negative,q_limit,0,records,retained,stats);
}
