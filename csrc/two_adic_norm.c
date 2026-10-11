/* Exact cyclic arithmetic over Z/8 and lifted single-row norm roots.
   No float, root sampling, skipped nullspace vector or model cap. */
#include <stdint.h>
#include <string.h>

static uint64_t rotate(uint64_t x,int u,int d,uint64_t all) {
    return u ? ((x<<u)|(x>>(d-u)))&all : x;
}

int pqcp_project_mod8(const uint64_t *a,const uint64_t *e,int d,uint64_t *out) {
    if (d!=3 && d!=5 && d!=11 && d!=43) return -1;
    uint64_t all=(UINT64_C(1)<<d)-1,l=0,m=0,h=0;
    for (int k=0;k<3;++k) if ((a[k]|e[k])&~all) return -1;
    for (int u=0;u<d;++u) {
        uint64_t al=rotate(a[0],u,d,all),am=rotate(a[1],u,d,all),ah=rotate(a[2],u,d,all);
        if ((e[0]>>u)&1) {
            uint64_t carry=l&al,next=(m&am)|(m&carry)|(am&carry);
            l^=al;m^=am^carry;h^=ah^next;
        }
        if ((e[1]>>u)&1) {h^=am^(m&al);m^=al;}
        if ((e[2]>>u)&1) h^=al;
    }
    out[0]=l;out[1]=m;out[2]=h;return 0;
}

int pqcp_lifted_unary_signs(uint64_t support,uint64_t base,const uint64_t *toggles,
    const uint64_t *columns,int variables,int field_bits,uint64_t origin,
    const uint64_t *r0,int n0,const uint64_t *r1,int n1,const uint64_t *r2,int n2,
    int d,int negative,const int32_t *wanted,uint64_t budget,uint64_t *solution,uint64_t *stats) {
    if ((d!=3 && d!=5 && d!=11 && d!=43) || variables<0 || variables>d
        || field_bits<1 || field_bits>42 || n0<1 || n1<1 || n2<1
        || negative<0 || negative>__builtin_popcountll(support) || !budget) return -1;
    uint64_t all=(UINT64_C(1)<<d)-1,image[43]={0},literal[43]={0},kernel[43]={0};int nullity=0;
    if ((support|base)&~all || base&~support) return -1;
    for (int i=0;i<variables;++i) {
        uint64_t value=columns[i],mask=toggles[i];
        if (mask&~support || value>>field_bits) return -1;
        while (value) {
            int p=63-__builtin_clzll(value);
            if (!image[p]) {image[p]=value;literal[p]=mask;break;}
            value^=image[p];mask^=literal[p];
        }
        if (!value) {if (!mask) return -2;kernel[nullity++]=mask;}
    }
    memset(stats,0,6*sizeof(uint64_t));uint64_t total=UINT64_C(1)<<nullity;
    uint64_t active[21];int edges[21];
    for (int u=1;u<=d/2;++u) {
        active[u-1]=support&rotate(support,u,d,all);edges[u-1]=__builtin_popcountll(active[u-1]);
    }
    for (int i=0;i<n0;++i) for (int j=0;j<n1;++j) for (int k=0;k<n2;++k) {
        ++stats[0];uint64_t value=origin^r0[i]^r1[j]^r2[k],mask=base;
        if (value>>field_bits) return -1;
        while (value) {
            int p=63-__builtin_clzll(value);
            if (!image[p]) break;
            value^=image[p];mask^=literal[p];
        }
        if (value) continue;
        ++stats[1];
        for (uint64_t cursor=0;cursor<total;++cursor) {
            if (stats[2]>=budget) return 2;
            ++stats[2];
            if (__builtin_popcountll(mask)==negative) {
                ++stats[3];int good=1;
                for (int u=1;u<=d/2;++u) {
                    ++stats[4];int cut=__builtin_popcountll((mask^rotate(mask,u,d,all))&active[u-1]);
                    if (edges[u-1]-2*cut!=wanted[u]) {good=0;break;}
                }
                if (good) {*solution=mask;return 1;}
            }
            if (cursor+1<total) mask^=kernel[__builtin_ctzll(cursor+1)];
        }
    }
    stats[5]=UINT64_C(1)<<variables;return 0;
}
