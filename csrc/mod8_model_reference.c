/* Independent complete Gray enumeration of Boolean quadratic zero sets.
 * No prefix projection, GF2 tail elimination or GPU arithmetic is used. */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

int pqcp_pack_mod8_projection(uint32_t constant,const uint32_t *linear,const uint32_t *quadratic,
    int q,int tail,uint32_t *raw,uint32_t *projected) {
    if (q<0 || q>24 || tail<0 || tail>q || tail>8) return -1;
    uint32_t basis[32]={0};
    for (int i=q-tail;i<q;++i) for (int j=i+1;j<q;++j) {
        uint32_t value=quadratic[i*q+j];
        while (value) {
            int p=31-__builtin_clz(value);
            if (!basis[p]) {basis[p]=value;break;}
            value^=basis[p];
        }
    }
    memset(raw,0,601*sizeof(uint32_t));raw[0]=constant;
    for (int i=0;i<q;++i) {
        raw[1+i]=linear[i];
        for (int j=0;j<q;++j) raw[25+24*i+j]=quadratic[i*q+j];
    }
    for (int j=0;j<601;++j) {
        uint32_t value=raw[j];
        for (int p=31;p>=0;--p) if ((value>>p)&1) value^=basis[p];
        projected[j]=value;
    }
    return 0;
}

int pqcp_mod8_reference(uint32_t constant,const uint32_t *linear,const uint32_t *quadratic,
    int q,uint64_t capacity,uint32_t *output,uint64_t *count) {
    if (q<0 || q>24 || !capacity) return -1;
    int split=q/2;
    uint64_t ls=UINT64_C(1)<<split,hs=UINT64_C(1)<<(q-split);
    uint32_t *low=malloc((size_t)(q?q:1)*ls*sizeof(uint32_t));
    uint32_t *high=malloc((size_t)(q?q:1)*hs*sizeof(uint32_t));
    if (!low || !high) {free(low);free(high);return -2;}
    for (int i=0;i<q;++i) {
        low[i*ls]=high[i*hs]=0;
        for (uint64_t j=1;j<ls;++j)
            low[i*ls+j]=low[i*ls+(j&(j-1))]^quadratic[i*q+__builtin_ctzll(j)];
        for (uint64_t j=1;j<hs;++j)
            high[i*hs+j]=high[i*hs+(j&(j-1))]^quadratic[i*q+split+__builtin_ctzll(j)];
    }
    uint32_t value=constant,gray=0;*count=0;
    uint64_t end=UINT64_C(1)<<q;
    for (uint64_t i=0;i<end;++i) {
        if (!value) {if (*count<capacity) output[*count]=gray; ++*count;}
        if (i+1<end) {
            int bit=__builtin_ctzll(i+1);
            value^=linear[bit]^low[bit*ls+(gray&(ls-1))]^high[bit*hs+(gray>>split)];
            gray^=UINT32_C(1)<<bit;
        }
    }
    free(low);free(high);return 0;
}

/* Bit-sliced truth tables for ALL tail assignments. Each uint32 represents
 * 32 simultaneous Boolean assignments; zero intersections close a prefix.
 * The first 'equations' tables contain tail-quadratic terms only, followed
 * by the exact tail-variable truth tables. No linearization relaxation. */
int pqcp_pack_mod8_bit_tail(const uint32_t *raw,int q,int tail,int equations,uint32_t *table) {
    if (q<0 || q>24 || tail<0 || tail>q || tail>8 || equations<1 || equations>25) return -1;
    int prefix=q-tail,members=1<<tail,chunks=(members+31)/32;
    memset(table,0,(size_t)(equations+tail)*chunks*sizeof(uint32_t));
    for (int member=0;member<members;++member) {
        uint32_t syndrome=0,bit=UINT32_C(1)<<(member%32);
        for (int i=0;i<tail;++i) if ((member>>i)&1) {
            table[(equations+i)*chunks+member/32]|=bit;
            for (int j=i+1;j<tail;++j) if ((member>>j)&1)
                syndrome^=raw[25+24*(prefix+i)+prefix+j];
        }
        for (int equation=0;equation<equations;++equation) if ((syndrome>>equation)&1)
            table[equation*chunks+member/32]|=bit;
    }
    return 0;
}

int pqcp_pack_mod8_bit_coefficients(uint32_t constant,const uint32_t *linear,const uint32_t *quadratic,
    int q,int tail,int equations,uint32_t *raw,uint32_t *table) {
    if (q<0 || q>24 || tail<0 || tail>q || tail>8 || equations<1 || equations>25) return -1;
    uint32_t limit=UINT32_C(1)<<equations;
    if (constant>=limit) return -1;
    memset(raw,0,601*sizeof(uint32_t));raw[0]=constant;
    for (int i=0;i<q;++i) {
        if (linear[i]>=limit) return -1;
        raw[1+i]=linear[i];
        for (int j=0;j<q;++j) {
            if (quadratic[i*q+j]>=limit) return -1;
            raw[25+24*i+j]=quadratic[i*q+j];
        }
    }
    return pqcp_pack_mod8_bit_tail(raw,q,tail,equations,table);
}
