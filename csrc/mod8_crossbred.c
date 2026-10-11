/* Exact Boolean degree-three elimination, with original-row XOR proofs.
 * No Hilbert-series rank estimates or probabilistic exclusions are used. */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int constant, linear[24], pair[24][24], triple[24][24][24];
} Positions;

static int position(const Positions *p, uint32_t mask) {
    if (!mask) return p->constant;
    int i=__builtin_ctz(mask); mask&=mask-1;
    if (!mask) return p->linear[i];
    int j=__builtin_ctz(mask); mask&=mask-1;
    if (!mask) return p->pair[i][j];
    int k=__builtin_ctz(mask);
    return p->triple[i][j][k];
}

static void xor_term(uint64_t *row, const Positions *positions, uint32_t mask) {
    int column=position(positions,mask);
    row[column/64]^=UINT64_C(1)<<(column%64);
}

int pqcp_crossbred_tail3(uint32_t constant,const uint32_t *linear,const uint32_t *quadratic,
    int q,int tail,const int *axes,int axis_count,int capacity,
    uint32_t *monomials,uint64_t *outputs,uint64_t *proofs,int *meta) {
    if (q<0 || q>24 || tail<0 || tail>q || tail>8 || axis_count<0 || axis_count>q || capacity<1) return -1;
    uint32_t axis_mask=0;
    for (int j=0;j<axis_count;++j) {
        if (axes[j]<0 || axes[j]>=q || (axis_mask&(UINT32_C(1)<<axes[j]))) return -1;
        axis_mask|=UINT32_C(1)<<axes[j];
    }
    uint32_t combined=constant;
    for (int i=0;i<q;++i) {
        combined|=linear[i];
        for (int j=i+1;j<q;++j) combined|=quadratic[i*q+j];
    }
    int width=combined?32-__builtin_clz(combined):0;
    uint32_t original[2325]; int count=0;
    original[count++]=0;
    for (int i=0;i<q;++i) original[count++]=UINT32_C(1)<<i;
    for (int i=0;i<q;++i) for (int j=i+1;j<q;++j)
        original[count++]=(UINT32_C(1)<<i)|(UINT32_C(1)<<j);
    for (int i=0;i<q;++i) for (int j=i+1;j<q;++j) for (int k=j+1;k<q;++k)
        original[count++]=(UINT32_C(1)<<i)|(UINT32_C(1)<<j)|(UINT32_C(1)<<k);
    uint32_t tail_mask=((UINT32_C(1)<<q)-1)^((UINT32_C(1)<<(q-tail))-1);
    int good=0;
    for (int j=0;j<count;++j) if (__builtin_popcount(original[j]&tail_mask)<=1)
        monomials[good++]=original[j];
    int out=good;
    for (int j=0;j<count;++j) if (__builtin_popcount(original[j]&tail_mask)>=2)
        monomials[out++]=original[j];
    Positions positions;
    for (int j=0;j<count;++j) {
        uint32_t mask=monomials[j];
        if (!mask) {positions.constant=j;continue;}
        int a=__builtin_ctz(mask);mask&=mask-1;
        if (!mask) {positions.linear[a]=j;continue;}
        int b=__builtin_ctz(mask);mask&=mask-1;
        if (!mask) {positions.pair[a][b]=j;continue;}
        positions.triple[a][b][__builtin_ctz(mask)]=j;
    }
    int words=(count+63)/64, proof_words=(width*(q+1)+63)/64;
    if (!proof_words) proof_words=1;
    uint64_t *basis=calloc((size_t)count*words,sizeof(uint64_t));
    uint64_t *certificates=calloc((size_t)count*proof_words,sizeof(uint64_t));
    unsigned char *occupied=calloc((size_t)count,1);
    if (!basis || !certificates || !occupied) {free(basis);free(certificates);free(occupied);return -2;}
    for (int equation=0;equation<width;++equation) {
        for (int multiplier=-1;multiplier<axis_count;++multiplier) {
            int axis=multiplier<0?-1:axes[multiplier];
            uint32_t factor=axis<0?0:UINT32_C(1)<<axis;
            uint64_t row[37]={0}, proof[13]={0};
            if ((constant>>equation)&1) xor_term(row,&positions,factor);
            for (int i=0;i<q;++i) {
                if ((linear[i]>>equation)&1) xor_term(row,&positions,(UINT32_C(1)<<i)|factor);
                for (int j=i+1;j<q;++j) if ((quadratic[i*q+j]>>equation)&1)
                    xor_term(row,&positions,(UINT32_C(1)<<i)|(UINT32_C(1)<<j)|factor);
            }
            int number=equation*(q+1)+axis+1;
            proof[number/64]|=UINT64_C(1)<<(number%64);
            int word=words-1;
            while (word>=0) {
                if (!row[word]) {--word;continue;}
                int pivot=word*64+63-__builtin_clzll(row[word]);
                uint64_t *other=basis+(size_t)pivot*words;
                uint64_t *certificate=certificates+(size_t)pivot*proof_words;
                if (!occupied[pivot]) {
                    occupied[pivot]=1;
                    memcpy(other,row,(size_t)words*sizeof(uint64_t));
                    memcpy(certificate,proof,(size_t)proof_words*sizeof(uint64_t));
                    break;
                }
                for (int j=0;j<=word;++j) row[j]^=other[j];
                for (int j=0;j<proof_words;++j) proof[j]^=certificate[j];
            }
        }
    }
    int consequences=0, status=0;
    for (int pivot=0;pivot<good;++pivot) if (occupied[pivot]) {
        if (consequences>=capacity) {status=-3;break;}
        memcpy(outputs+(size_t)consequences*words,basis+(size_t)pivot*words,(size_t)words*sizeof(uint64_t));
        memcpy(proofs+(size_t)consequences*proof_words,certificates+(size_t)pivot*proof_words,(size_t)proof_words*sizeof(uint64_t));
        ++consequences;
    }
    meta[0]=consequences;meta[1]=count;meta[2]=good;meta[3]=words;
    meta[4]=proof_words;meta[5]=width*(axis_count+1);
    free(basis);free(certificates);free(occupied);
    return status;
}

/* Keep only consequences independent of the old quotient equations and
 * their prefix-variable multiples. This affects selection, not validity:
 * selected rows retain their original, unmodified XOR certificates. */
static int insert_row(uint64_t *row,uint64_t *basis,unsigned char *occupied,int words) {
    int word=words-1;
    while (word>=0) {
        if (!row[word]) {--word;continue;}
        int pivot=word*64+63-__builtin_clzll(row[word]);
        uint64_t *other=basis+(size_t)pivot*words;
        if (!occupied[pivot]) {
            occupied[pivot]=1;
            memcpy(other,row,(size_t)words*sizeof(uint64_t));
            return 1;
        }
        for (int j=0;j<=word;++j) row[j]^=other[j];
    }
    return 0;
}

/* A quadratic table in ordinary binary order. Recurrence removes the least
 * set bit; unlike Gray enumeration no layout conversion is needed on GPU. */
static void quadratic_table(int n,uint32_t constant,const uint32_t *linear,
    const uint32_t quadratic[24][24],uint32_t *table) {
    int size=1<<n;table[0]=constant;
    for (int bits=1;bits<size;++bits) {
        int i=__builtin_ctz((unsigned)bits),remaining=bits&(bits-1);
        uint32_t value=table[remaining]^linear[i];
        for (unsigned rest=(unsigned)remaining;rest;rest&=rest-1)
            value^=quadratic[i][__builtin_ctz(rest)];
        table[bits]=value;
    }
}

static void cubic_table(int n,uint32_t constant,const uint32_t *linear,
    const uint32_t quadratic[24][24],const uint32_t cubic[24][24][24],uint32_t *table) {
    int size=1<<n;table[0]=constant;
    for (int bits=1;bits<size;++bits) {
        int i=__builtin_ctz((unsigned)bits),remaining=bits&(bits-1);
        uint32_t value=table[remaining]^linear[i];
        for (unsigned rest=(unsigned)remaining;rest;rest&=rest-1) {
            int j=__builtin_ctz(rest);value^=quadratic[i][j];
            for (unsigned other=rest&(rest-1);other;other&=other-1)
                value^=cubic[i][j][__builtin_ctz(other)];
        }
        table[bits]=value;
    }
}

/* Direct GPU lookup packing of at most 32 additional certified equations.
 * Every Boolean cubic P(x,y) is split EXACTLY into pure-low, pure-high,
 * low-linear/high-quadratic and high-linear/low-quadratic terms.
 * Caller sizes table as (tail+1)*(L+H+low*H+high*L) uint32 words. */
int pqcp_crossbred_lookup(const uint32_t *monomials,const uint64_t *rows,
    int count,int columns,int words,int q,int tail,const uint32_t *projected,
    uint32_t *table,int *selected_indices,int *selected_count) {
    if (q<0 || q>24 || tail<0 || tail>q || tail>8 || count<0 || count>800
            || columns<1 || columns>2325 || words!=(columns+63)/64) return -1;
    int prefix=q-tail;uint32_t prefix_mask=(UINT32_C(1)<<prefix)-1;
    Positions positions;
    for (int j=0;j<columns;++j) {
        uint32_t mask=monomials[j];
        if (!mask) {positions.constant=j;continue;}
        int a=__builtin_ctz(mask);mask&=mask-1;
        if (!mask) {positions.linear[a]=j;continue;}
        int b=__builtin_ctz(mask);mask&=mask-1;
        if (!mask) {positions.pair[a][b]=j;continue;}
        positions.triple[a][b][__builtin_ctz(mask)]=j;
    }
    uint64_t *basis=calloc((size_t)columns*words,sizeof(uint64_t));
    unsigned char *occupied=calloc((size_t)columns,1);
    if (!basis || !occupied) {free(basis);free(occupied);return -2;}
    for (int equation=0;equation<32;++equation) for (int axis=-1;axis<prefix;++axis) {
        uint64_t row[37]={0};uint32_t factor=axis<0?0:UINT32_C(1)<<axis;
        if ((projected[0]>>equation)&1) xor_term(row,&positions,factor);
        for (int i=0;i<q;++i) {
            if ((projected[1+i]>>equation)&1) xor_term(row,&positions,(UINT32_C(1)<<i)|factor);
            for (int j=i+1;j<q;++j) if ((projected[25+24*i+j]>>equation)&1)
                xor_term(row,&positions,(UINT32_C(1)<<i)|(UINT32_C(1)<<j)|factor);
        }
        insert_row(row,basis,occupied,words);
    }
    int selected=0;
    for (int i=0;i<count && selected<32;++i) {
        uint64_t row[37];memcpy(row,rows+(size_t)i*words,(size_t)words*sizeof(uint64_t));
        if (insert_row(row,basis,occupied,words)) selected_indices[selected++]=i;
    }
    free(basis);free(occupied);*selected_count=selected;
    int low=prefix/2,high=prefix-low,L=1<<low,H=1<<high;
    int stride=L+H+low*H+high*L;
    memset(table,0,(size_t)(tail+1)*stride*sizeof(uint32_t));
    /* Per-block coefficients are small: at most 2325*32-bit words. */
    for (int block=0;block<=tail;++block) {
        uint32_t constant=0,linear[24]={0},quadratic[24][24]={{0}},cubic[24][24][24]={{{0}}};
        for (int equation=0;equation<selected;++equation) {
            const uint64_t *row=rows+(size_t)selected_indices[equation]*words;
            uint32_t field=UINT32_C(1)<<equation;
            for (int w=0;w<words;++w) for (uint64_t rest=row[w];rest;rest&=rest-1) {
                int column=w*64+__builtin_ctzll(rest);
                uint32_t term=monomials[column],suffix=term>>prefix;
                if (suffix!=(block?UINT32_C(1)<<(block-1):0)) continue;
                uint32_t mask=term&prefix_mask;
                if (!mask) {constant^=field;continue;}
                int i=__builtin_ctz(mask);mask&=mask-1;
                if (!mask) {linear[i]^=field;continue;}
                int j=__builtin_ctz(mask);mask&=mask-1;
                if (!mask) {quadratic[i][j]^=field;continue;}
                cubic[i][j][__builtin_ctz(mask)]^=field;
            }
        }
        uint32_t *destination=table+(size_t)block*stride;
        cubic_table(low,constant,linear,quadratic,cubic,destination);
        uint32_t hl[24]={0},hq[24][24]={{0}},hc[24][24][24]={{{0}}};
        for (int i=0;i<high;++i) {
            hl[i]=linear[low+i];
            for (int j=i+1;j<high;++j) {
                hq[i][j]=quadratic[low+i][low+j];
                for (int k=j+1;k<high;++k) hc[i][j][k]=cubic[low+i][low+j][low+k];
            }
        }
        cubic_table(high,0,hl,hq,hc,destination+L);
        for (int i=0;i<low;++i) {
            memset(hl,0,sizeof(hl));memset(hq,0,sizeof(hq));
            for (int j=0;j<high;++j) {
                hl[j]=quadratic[i][low+j];
                for (int k=j+1;k<high;++k) hq[j][k]=cubic[i][low+j][low+k];
            }
            quadratic_table(high,0,hl,hq,destination+L+H+i*H);
        }
        for (int k=0;k<high;++k) {
            memset(hl,0,sizeof(hl));memset(hq,0,sizeof(hq));
            for (int i=0;i<low;++i) for (int j=i+1;j<low;++j) hq[i][j]=cubic[i][j][low+k];
            quadratic_table(low,0,hl,hq,destination+L+H+low*H+k*L);
        }
    }
    return 0;
}
