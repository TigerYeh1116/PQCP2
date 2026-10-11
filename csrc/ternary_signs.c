/* Exact Gray-code enumeration of ALL assignments to a GF(2) affine space.
 * Each word uses d<=47 bits; a 1 denotes a NEGATIVE active entry.
 * No randomization or floating point. 0=exhausted, 1=hit, 2=budget cap. */
#include <stdint.h>
#include <string.h>

static uint64_t rotate(uint64_t x,int u,int d,uint64_t all) {
    return ((x << u) | (x >> (d-u))) & all;
}
int pqcp_ternary_signs(const uint64_t *support,const uint64_t *initial,
    const uint64_t *toggles,int free_count,int d,const int *wanted,
    const int *negative,uint64_t budget,uint64_t *output,uint64_t *stats) {
    if(d<3 || d>47 || !(d&1) || free_count<0 || free_count>47 || !budget)return -1;
    uint64_t total=UINT64_C(1)<<free_count,limit=budget<total?budget:total;
    uint64_t all=(UINT64_C(1)<<d)-1,signs[2]={initial[0],initial[1]},active[2][23];
    int required[23];memset(stats,0,3*sizeof(uint64_t));
    for(int s=0;s<2;s++) {
        if(support[s] & ~all || signs[s] & ~support[s])return -1;
        for(int u=1;u<=d/2;u++)active[s][u-1]=support[s]&rotate(support[s],u,d,all);
    }
    for(int u=1;u<=d/2;u++) {
        int m=__builtin_popcountll(active[0][u-1])+__builtin_popcountll(active[1][u-1]);
        if(m<wanted[u] || m < -wanted[u] || (m-wanted[u])%2)return -1;
        required[u-1]=(m-wanted[u])/2;
    }
    for(uint64_t i=0;i<limit;i++) {
        stats[0]++;
        if(__builtin_popcountll(signs[0])==negative[0] && __builtin_popcountll(signs[1])==negative[1]) {
            stats[1]++;int okay=1;
            for(int u=1;u<=d/2;u++) {
                int n=0;for(int s=0;s<2;s++)
                    n+=__builtin_popcountll((signs[s]^rotate(signs[s],u,d,all))&active[s][u-1]);
                stats[2]++;
                if(n!=required[u-1]){okay=0;break;}
            }
            if(okay){output[0]=signs[0];output[1]=signs[1];return 1;}
        }
        if(i+1<limit) {
            int bit=__builtin_ctzll(i+1);
            signs[0]^=toggles[2*bit];signs[1]^=toggles[2*bit+1];
        }
    }
    return limit==total?0:2;
}
