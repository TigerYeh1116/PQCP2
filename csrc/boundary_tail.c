/* Exact mod-8 projection for arbitrary fixed binary prefixes and short tails.
 * For u>=tail, tail-tail edges are absent. T(u)/4 is affine in tail bits
 * modulo two. Enumerate EVERY solution of that necessary linear system,
 * including dependent-column nullspaces, then check ALL original sidelobes.
 * A miss excludes only this prefix pair and arbitrary appended tails. */
#include <stdint.h>
#include <string.h>

typedef __uint128_t word;

static word rotate(word a, int u, int length, word mask) {
    return ((a << u) | (a >> (length-u))) & mask;
}

static int population(word a) {
    return __builtin_popcountll((uint64_t)a)+__builtin_popcountll((uint64_t)(a >> 64));
}

static int profile(word a, word b, int u, int length, word mask) {
    return 2*length-2*(population(a ^ rotate(a,u,length,mask))+
                     population(b ^ rotate(b,u,length,mask)));
}

int pqcp_boundary_tails(const uint64_t *prefixes, int length, int tail,
    uint16_t *witnesses, uint8_t *lags, int8_t *signs, uint64_t *stats) {
    if (length<4 || length>94 || length%2 || tail<1 || tail>4 || tail>=length/2)
        return -1;
    const int prefix=length-tail, variables=2*tail, half=length/2;
    const word mask=(((word)1) << length)-1, pmask=(((word)1) << prefix)-1;
    word rows[2]={((word)prefixes[1] << 64)|prefixes[0],
                  ((word)prefixes[3] << 64)|prefixes[2]};
    if ((rows[0] & ~pmask) || (rows[1] & ~pmask)) return -1;
    memset(stats,0,8*sizeof(uint64_t));
    uint64_t columns[8]={0}, base=0;
    for (int u=tail;u<=half;++u) {
        int value=profile(rows[0],rows[1],u,length,mask);
        if (value%4) return -2;
        if ((value/4)%2) base |= UINT64_C(1) << (u-tail);
        for (int side=0;side<2;++side) for (int j=0;j<tail;++j) {
            int p=prefix+j;
            int left=(p+u)%length,right=(p-u+length)%length;
            if (left>=prefix || right>=prefix) return -2;
            int x=(int)((rows[side] >> left)&1),y=(int)((rows[side] >> right)&1);
            if (x==y) columns[side*tail+j] |= UINT64_C(1) << (u-tail);
        }
    }
    uint64_t pivots[48]={0};
    uint16_t labels[48]={0},nullspace[8];
    int rank=0,nullity=0;
    for (int j=0;j<variables;++j) {
        uint64_t value=columns[j];
        uint16_t label=(uint16_t)(1u << j);
        while (value) {
            int p=63-__builtin_clzll(value);
            if (!pivots[p]) { pivots[p]=value;labels[p]=label;++rank;break; }
            value ^= pivots[p];label ^= labels[p];
        }
        if (!value) nullspace[nullity++]=label;
    }
    stats[0]=(uint64_t)rank;stats[1]=(uint64_t)nullity;
    int count=0;
    /* Group zero represents k<tail, which share the same outer equations.
     * Nonzero groups cover k=tail..half-1. The self-reverse shift is zero. */
    for (int group=tail-1;group<half;++group) {
        const int small=group==tail-1;
        if (small && tail==1) continue;
        ++stats[2];
        uint64_t rhs=base ^ (small ? 0 : UINT64_C(1) << (group-tail));
        uint16_t particular=0;
        while (rhs) {
            int p=63-__builtin_clzll(rhs);
            if (!pivots[p]) break;
            rhs ^= pivots[p];particular ^= labels[p];
        }
        if (rhs) { ++stats[3];continue; }
        for (unsigned v=0;v<(1u << nullity);++v) {
            uint16_t assignment=particular;
            for (int j=0;j<nullity;++j) if ((v >> j)&1) assignment ^= nullspace[j];
            ++stats[4];
            word a=rows[0] | ((word)(assignment & ((1u << tail)-1)) << prefix);
            word b=rows[1] | ((word)(assignment >> tail) << prefix);
            int found=0,value=0,okay=1;
            for (int u=1;u<=half;++u) {
                int p=profile(a,b,u,length,mask);
                ++stats[5];
                if (!p) continue;
                if (u==half || found || (p!=4 && p!=-4)) { okay=0;break; }
                found=u;value=p;
            }
            if (okay && found) {
                if ((small && found>=tail) || (!small && found!=group) || count>=256)
                    return -2;
                witnesses[count]=assignment;lags[count]=(uint8_t)found;
                signs[count]=(int8_t)(value/4);++count;
            }
        }
    }
    stats[6]=(uint64_t)count;
    stats[7]=UINT64_C(1) << variables;
    return count;
}
