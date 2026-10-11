/* Canonicalize (integer row norm, support) under a common odd-unit
 * decimation and independent A-row rotation. All full k targets must be
 * searched before a whole-anchor UNSAT result is reused. */
#include <stdint.h>

static int gcd(int a,int b) {
    while (b) {int r=a%b;a=b;b=r;}
    return a;
}

static uint64_t rotate(uint64_t value,int shift,int d,uint64_t all) {
    if (!shift) return value;
    return ((value<<shift)|(value>>(d-shift)))&all;
}

int pqcp_coarse_anchor_key(uint64_t support,uint64_t minus,int d,int *profile,
    uint64_t *canonical_mask,int *action) {
    if (d<3 || d>47 || !(d&1)) return -1;
    uint64_t all=(UINT64_C(1)<<d)-1;
    if ((support&~all) || (minus&~support)) return -1;
    int norm[47];norm[0]=__builtin_popcountll(support);
    for (int shift=1;shift<d;++shift) {
        uint64_t active=support&rotate(support,shift,d,all);
        int mismatch=__builtin_popcountll((minus^rotate(minus,shift,d,all))&active);
        norm[shift]=__builtin_popcountll(active)-2*mismatch;
    }
    int initialized=0;
    uint64_t best=0;
    for (int unit=1;unit<d;++unit) if (gcd(unit,d)==1) {
        int comparison=initialized?0:-1;
        if (initialized) for (int shift=0;shift<d;++shift) {
            int value=norm[unit*shift%d];
            if (value!=profile[shift]) {comparison=value<profile[shift]?-1:1;break;}
        }
        if (comparison>0) continue;
        if (comparison<0) {
            for (int shift=0;shift<d;++shift) profile[shift]=norm[unit*shift%d];
            initialized=1;best=all;action[0]=0;
        }
        uint64_t transformed=0;
        for (int j=0;j<d;++j) transformed|=((support>>(unit*j%d))&1)<<j;
        for (int offset=0;offset<d;++offset) {
            if (transformed<best || action[0]==0) {
                best=transformed;action[0]=unit;action[1]=offset;
            }
            transformed=(transformed>>1)|((transformed&1)<<(d-1));
        }
    }
    *canonical_mask=best;
    return 0;
}

/* Unlike the norm/support key, this is a literal-word isometry and can
 * deduplicate entire L1 balls. Homometry alone cannot do that. */
int pqcp_coarse_word_key(uint64_t support,uint64_t minus,int d,uint64_t *key) {
    if (d<3 || d>47 || !(d&1)) return -1;
    uint64_t all=(UINT64_C(1)<<d)-1;
    if ((support&~all) || (minus&~support)) return -1;
    uint64_t best_support=all,best_minus=all;
    for (int unit=1;unit<d;++unit) if (gcd(unit,d)==1) {
        uint64_t active=0,negative=0;
        for (int j=0;j<d;++j) {
            active|=((support>>(unit*j%d))&1)<<j;
            negative|=((minus>>(unit*j%d))&1)<<j;
        }
        for (int offset=0;offset<d;++offset) {
            uint64_t flipped=active^negative,candidate=negative<flipped?negative:flipped;
            if (active<best_support || (active==best_support && candidate<best_minus)) {
                best_support=active;best_minus=candidate;
            }
            active=(active>>1)|((active&1)<<(d-1));
            negative=(negative>>1)|((negative&1)<<(d-1));
        }
    }
    key[0]=best_support;key[1]=best_minus;
    return 0;
}
