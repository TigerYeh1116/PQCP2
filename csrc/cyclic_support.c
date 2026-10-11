/* Exact CRT support enumeration. Ranking changes scheduling, not validity. */
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#if defined(__aarch64__)
#include <arm_neon.h>

static uint32x4_t popcounts4(uint64x2_t left, uint64x2_t right)
{
    uint64x2_t a = vpaddlq_u32(vpaddlq_u16(vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u64(left)))));
    uint64x2_t b = vpaddlq_u32(vpaddlq_u16(vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u64(right)))));
    return vuzp1q_u32(vreinterpretq_u32_u64(a), vreinterpretq_u32_u64(b));
}
#endif

static int nearest(const uint64_t *a, int na,
                         const uint64_t *b, int nb,
                         const uint64_t *c, int nc,
                         uint64_t base, uint64_t reference, int weight, int minimum,
                         uint64_t *output, uint64_t capacity, uint64_t *stats)
{
    if (na < 1 || nb < 1 || nc < 1 || !capacity || weight < 0 || weight > 43 || minimum < 0 || minimum > 43)
        return -1;
    uint64_t valid = 0, ties = 0, stored = 0;
    int best = 64;
#if defined(__aarch64__)
    uint32x4_t counts = vdupq_n_u32(0), weight_vector = vdupq_n_u32((uint32_t)weight);
    uint32x4_t floor_vector = vdupq_n_u32((uint32_t)minimum), invalid = vdupq_n_u32(64);
    uint64x2_t reference_vector = vdupq_n_u64(reference);
#endif
    for (int i = 0; i < na; ++i) {
        for (int j = 0; j < nb; ++j) {
            uint64_t prefix = base ^ a[i] ^ b[j];
            int k = 0;
#if defined(__aarch64__)
            uint64x2_t prefix_vector = vdupq_n_u64(prefix);
            for (; k+4 <= nc; k += 4) {
                uint64x2_t left = veorq_u64(prefix_vector, vld1q_u64(c+k));
                uint64x2_t right = veorq_u64(prefix_vector, vld1q_u64(c+k+2));
                uint32x4_t weights = popcounts4(left, right);
                uint32x4_t matches = vceqq_u32(weights, weight_vector);
                counts = vsubq_u32(counts, matches); /* true is UINT32_MAX */
                uint32x4_t distances = popcounts4(veorq_u64(left, reference_vector),
                                                veorq_u64(right, reference_vector));
                uint32x4_t eligible = vbicq_u32(matches, vcltq_u32(distances, floor_vector));
                distances = vbslq_u32(eligible, distances, invalid);
                int batch_minimum = (int)vminvq_u32(distances);
                if (batch_minimum < 64 && batch_minimum <= best) {
                    if (batch_minimum < best) { best = batch_minimum; ties = 0; stored = 0; }
                    uint64_t masks[4]; uint32_t values[4];
                    vst1q_u64(masks, left); vst1q_u64(masks+2, right);
                    vst1q_u32(values, distances);
                    for (int lane = 0; lane < 4; ++lane) {
                        if ((int)values[lane] == best) {
                            ++ties;
                            if (stored < capacity) output[stored++] = masks[lane];
                        }
                    }
                }
            }
#endif
            for (; k < nc; ++k) {
                uint64_t mask = prefix ^ c[k];
                if (__builtin_popcountll(mask) != weight) continue;
                ++valid;
                int distance = __builtin_popcountll(mask ^ reference);
                if (distance < minimum) continue;
                if (distance < best) { best = distance; ties = 0; stored = 0; }
                if (distance == best) {
                    ++ties;
                    if (stored < capacity) output[stored++] = mask;
                }
            }
        }
    }
#if defined(__aarch64__)
    valid += vaddvq_u32(counts);
#endif
    stats[0] = valid; stats[1] = ties; stats[2] = stored;
    return ties ? best : -2;
}

int pqcp_nearest_supports(const uint64_t *a, int na, const uint64_t *b, int nb,
                         const uint64_t *c, int nc, uint64_t base, uint64_t reference,
                         int weight, uint64_t *output, uint64_t capacity, uint64_t *stats)
{
    return nearest(a, na, b, nb, c, nc, base, reference, weight, 0, output, capacity, stats);
}

int pqcp_support_layer(const uint64_t *a, int na, const uint64_t *b, int nb,
                      const uint64_t *c, int nc, uint64_t base, uint64_t reference,
                      int weight, int minimum, uint64_t *output, uint64_t capacity, uint64_t *stats)
{
    return nearest(a, na, b, nb, c, nc, base, reference, weight, minimum, output, capacity, stats);
}

/* Select the exact index-th literal mask of the required weight. No nearest
 * distance is imposed and no phase/preimage is merged with another one. */
int pqcp_weighted_support_at(const uint64_t *a, int na, const uint64_t *b, int nb,
    const uint64_t *c, int nc, uint64_t base, int weight, uint64_t index,
    uint64_t *output) {
    if (na < 1 || nb < 1 || nc < 1 || weight < 0 || weight > 43) return -1;
    uint64_t seen = 0;
    for (int i = 0; i < na; ++i)
        for (int j = 0; j < nb; ++j) {
            uint64_t prefix = base ^ a[i] ^ b[j];
            int k = 0;
#if defined(__aarch64__)
            uint64x2_t pv = vdupq_n_u64(prefix);
            uint32x4_t wv = vdupq_n_u32((uint32_t)weight);
            for (; k+4 <= nc; k += 4) {
                uint64x2_t left = veorq_u64(pv,vld1q_u64(c+k));
                uint64x2_t right = veorq_u64(pv,vld1q_u64(c+k+2));
                uint32x4_t matches = vceqq_u32(popcounts4(left,right),wv);
                uint64_t count = vaddvq_u32(vshrq_n_u32(matches,31));
                if (seen+count <= index) { seen+=count; continue; }
                uint64_t masks[4]; uint32_t valid[4];
                vst1q_u64(masks,left);vst1q_u64(masks+2,right);vst1q_u32(valid,matches);
                for (int lane = 0; lane < 4; ++lane)
                    if (valid[lane] && seen++ == index) { *output=masks[lane];return 1; }
            }
#endif
            for (; k < nc; ++k) {
                uint64_t mask = prefix ^ c[k];
                if (__builtin_popcountll(mask)==weight && seen++==index) {
                    *output=mask;return 1;
                }
            }
        }
    return 0;
}

static uint64_t block_rotate(uint64_t value,int shift,int d,uint64_t all) {
    if (!shift) return value;
    return ((value<<shift)|(value>>(d-shift)))&all;
}

/* Store the complete prescribed-weight fibre over the final phase index.
 * A cyclic phase translation only relabels this exact tensor of bits. */
int pqcp_weighted_support_grid(const uint64_t *a,int na,const uint64_t *b,int nb,
    const uint64_t *c,int nc,uint64_t base,int weight,
    uint64_t *grid,uint64_t capacity,uint64_t *count) {
    if (na<1 || na>129 || nb<1 || nb>129 || nc<1 || nc>129 || weight<0 || weight>43) return -1;
    int words=(nc+63)/64;uint64_t required=(uint64_t)na*(uint64_t)nb*(uint64_t)words;
    if (capacity<required) return -1;
    memset(grid,0,(size_t)required*sizeof(uint64_t));*count=0;
#if defined(__aarch64__)
    uint32x4_t wv=vdupq_n_u32((uint32_t)weight),counts=vdupq_n_u32(0);
    static const uint32_t positions[4]={1,2,4,8};uint32x4_t pvbits=vld1q_u32(positions);
#endif
    for (int i=0;i<na;++i) for (int j=0;j<nb;++j) {
        uint64_t prefix=base^a[i]^b[j];uint64_t *row=grid+((size_t)i*nb+j)*words;
        int k=0;
#if defined(__aarch64__)
        uint64x2_t pv=vdupq_n_u64(prefix);
        for (;k+4<=nc;k+=4) {
            uint64x2_t left=veorq_u64(pv,vld1q_u64(c+k)),right=veorq_u64(pv,vld1q_u64(c+k+2));
            uint32x4_t matches=vceqq_u32(popcounts4(left,right),wv);
            unsigned bits=vaddvq_u32(vandq_u32(matches,pvbits));
            counts=vsubq_u32(counts,matches);
            row[k/64]|=(uint64_t)bits<<(k%64);
        }
#endif
        for (;k<nc;++k) if (__builtin_popcountll(prefix^c[k])==weight) {
            row[k/64]|=UINT64_C(1)<<(k%64);++*count;
        }
    }
#if defined(__aarch64__)
    *count+=vaddvq_u32(counts);
#endif
    return 0;
}

/* Select the original literal lexicographic rank after phase translation.
 * Return the corresponding CANONICAL grid ordinal; no support is dropped. */
int pqcp_weighted_grid_at(const uint64_t *grid,int na,int nb,int nc,
    int shift_a,int shift_b,int shift_c,uint64_t rank,uint64_t *ordinal) {
    if (na<1 || na>129 || nb<1 || nb>129 || nc<1 || nc>129 || shift_a<0 || shift_a>=na ||
        shift_b<0 || shift_b>=nb || shift_c<0 || shift_c>=nc) return -1;
    int words=(nc+63)/64,ai=(na-shift_a)%na;
    for (int i=0;i<na;++i,ai=(ai+1==na?0:ai+1)) {
        int bj=(nb-shift_b)%nb;
        for (int j=0;j<nb;++j,bj=(bj+1==nb?0:bj+1)) {
            const uint64_t *row=grid+((size_t)ai*nb+bj)*words;
            uint64_t count=0;
            for (int t=0;t<words;++t) count+=(uint64_t)__builtin_popcountll(row[t]);
            if (rank>=count) {rank-=count;continue;}
            int ck=(nc-shift_c)%nc;
            for (int k=0;k<nc;++k,ck=(ck+1==nc?0:ck+1)) if (row[ck/64]>>(ck%64)&1) {
                if (!rank) {*ordinal=((uint64_t)ai*nb+bj)*nc+ck;return 1;}
                --rank;
            }
            return -1;
        }
    }
    return 0;
}

/* Every prescribed-weight preimage of an already audited norm family.
   No nearest-distance selection and no output truncation. */
int pqcp_all_weighted_norm_supports(const uint64_t *a,int na,const uint64_t *b,int nb,
    const uint64_t *c,int nc,uint64_t base,int d,int weight,
    uint64_t *output,uint64_t capacity,uint64_t *stats) {
    if (na<1||nb<1||nc<1||d<3||d>43||!(d&1)||weight<0||weight>d||!capacity) return -1;
    uint64_t all=(UINT64_C(1)<<d)-1;if (base&~all) return -1;
    stats[0]=stats[1]=0;
    for (int i=0;i<na;++i) for (int j=0;j<nb;++j) for (int k=0;k<nc;++k) {
        ++stats[0];uint64_t mask=base^a[i]^b[j]^c[k];if (mask&~all) return -1;
        if (__builtin_popcountll(mask)!=weight) continue;
        if (stats[1]>=capacity) return 2;
        output[stats[1]++]=mask;
    }
    return 0;
}

/* For signed rows on S and its complement, N(u)=I(u)-2 cut(u).
   Each minority-sign vertex has degree at most two, so cut<=2 minority.
   Both rows must satisfy this independently at EVERY cyclic shift. */
int pqcp_signed_complement_supports(const uint64_t *a,int na,const uint64_t *b,int nb,
    const uint64_t *c,int nc,uint64_t base,int d,int weight,const int *rho_c,const int *rho_g,
    int negative_c,int negative_g,uint64_t *output,uint64_t *proofs,uint64_t capacity,uint64_t *stats) {
    if (na<1||nb<1||nc<1||d<3||d>43||!(d&1)||weight<0||weight>d||!capacity
        ||negative_c<0||negative_c>weight||negative_g<0||negative_g>d-weight) return -1;
    uint64_t all=(UINT64_C(1)<<d)-1;if (base&~all) return -1;
    for (int t=0;t<4;++t) stats[t]=0;
    int minority[2]={negative_c<weight-negative_c?negative_c:weight-negative_c,
                    negative_g<d-weight-negative_g?negative_g:d-weight-negative_g};
    for (int i=0;i<na;++i) for (int j=0;j<nb;++j) for (int k=0;k<nc;++k) {
        ++stats[0];uint64_t mask=base^a[i]^b[j]^c[k];if (mask&~all) return -1;
        if (__builtin_popcountll(mask)!=weight) continue;
        ++stats[1];int code=-1;
        for (int u=1;u<=d/2 && code<0;++u) {
            int ic=__builtin_popcountll(mask&block_rotate(mask,u,d,all));
            int intersections[2]={ic,d-2*weight+ic};int wanted[2]={rho_c[u],rho_g?rho_g[u]:0};
            for (int side=0;side<(rho_g?2:1);++side) {
                int difference=intersections[side]-wanted[side],reason=-1;
                if (wanted[side]>intersections[side] || wanted[side]<-intersections[side]) reason=0;
                else if (difference&1) reason=2;
                else if (difference>4*minority[side]) reason=1;
                if (reason>=0) {code=(u<<8)|(side<<4)|reason;break;}
            }
        }
        if (code<0) {
            if (stats[2]>=capacity) return 2;
            output[stats[2]++]=mask;
        } else {
            if (stats[3]>=capacity) return 2;
            proofs[2*stats[3]]=mask;proofs[2*stats[3]+1]=(uint64_t)code;++stats[3];
        }
    }
    return 0;
}

/* Complete integer SDS completion inside an already certified CRT norm
 * family. No nearest-distance cap. Every output is a literal preimage.
 * Overflow is reported, never an exclusion of omitted exact models. */
int pqcp_complete_intersection_profile(const uint64_t *a,int na,const uint64_t *b,int nb,
    const uint64_t *c,int nc,uint64_t base,int d,int other_weight,
    const int *wanted,uint64_t *output,uint64_t capacity,uint64_t *stats) {
    if(na<1||nb<1||nc<1||d<3||d>43||!(d&1)||other_weight<0||other_weight>d
       ||!capacity||!wanted)return -1;
    uint64_t all=(UINT64_C(1)<<d)-1;
    if(base&~all)return -1;
    for(int i=0;i<5;++i)stats[i]=0;
    for(int i=0;i<na;++i)
        for(int j=0;j<nb;++j)
            for(int k=0;k<nc;++k) {
                ++stats[0];
                uint64_t mask=base^a[i]^b[j]^c[k];
                if(__builtin_popcountll(mask)!=other_weight)continue;
                ++stats[1];
                int valid=1;
                for(int u=1;u<=d/2;++u) {
                    ++stats[2];
                    if(__builtin_popcountll(mask&block_rotate(mask,u,d,all))!=wanted[u-1]){valid=0;break;}
                }
                if(!valid)continue;
                ++stats[3];
                if(stats[4]<capacity)output[stats[4]++]=mask;
            }
    return stats[3]>capacity?2:0;
}

int pqcp_complete_sds(const uint64_t *a,int na,const uint64_t *b,int nb,
    const uint64_t *c,int nc,uint64_t base,uint64_t fixed,int d,int other_weight,
    int lambda,uint64_t *output,uint64_t capacity,uint64_t *stats) {
    if(d<3||d>43||!(d&1)||lambda<0||lambda>2*d)return -1;
    uint64_t all=(UINT64_C(1)<<d)-1;
    if(fixed&~all)return -1;
    int wanted[21];
    for(int u=1;u<=d/2;++u)
        wanted[u-1]=lambda-__builtin_popcountll(fixed&block_rotate(fixed,u,d,all));
    return pqcp_complete_intersection_profile(a,na,b,nb,c,nc,base,d,other_weight,
                                              wanted,output,capacity,stats);
}

/* Ingredient quotient under common units and independent translations.
 * Every V completion and all perfect-O relative phases remain available.
 * This is not a quotient of fixed-center Hamming neighborhoods. */
uint64_t pqcp_sds_ingredient_key(uint64_t fixed,int d) {
    if(d!=3&&d!=5&&d!=11&&d!=43)return UINT64_MAX;
    uint64_t all=(UINT64_C(1)<<d)-1,best=UINT64_MAX;
    if(fixed&~all)return UINT64_MAX;
    for(int q=1;q<d;++q) {
        uint64_t changed=0;
        for(int i=0;i<d;++i)changed|=((fixed>>((q*i)%d))&1)<<i;
        for(int r=0;r<d;++r) {
            uint64_t value=block_rotate(changed,r,d,all);
            if(value<best)best=value;
        }
    }
    return best;
}

/* Literal support rotations/reflections ONLY: no complement or unit
   decimation of one B row under a fixed C_A norm problem. */
uint64_t pqcp_dihedral_support_key(uint64_t mask,int d) {
    if (d!=3&&d!=5&&d!=11&&d!=43) return UINT64_MAX;
    uint64_t all=(UINT64_C(1)<<d)-1;if (mask&~all) return UINT64_MAX;
    uint64_t reflected=mask&1,best=mask;
    for (int i=1;i<d;++i) reflected|=((mask>>i)&1)<<(d-i);
    for (int r=0;r<d;++r) {
        uint64_t a=block_rotate(mask,r,d,all),b=block_rotate(reflected,r,d,all);
        if (a<best) best=a;if (b<best) best=b;
    }
    return best;
}

int pqcp_dihedral_support_keys(const uint64_t *masks,uint64_t count,int d,uint64_t *keys) {
    if (!masks||!keys||(d!=3&&d!=5&&d!=11&&d!=43)||count>UINT64_C(2147483648)) return -1;
    for (uint64_t i=0;i<count;++i) {
        keys[i]=pqcp_dihedral_support_key(masks[i],d);
        if (keys[i]==UINT64_MAX) return -1;
    }
    return 0;
}

/* Independent literal audit by carryless polynomial convolution in
   F_2[x]/(x^d-1), rather than CRT construction or a sampled fingerprint.
   d<=43 implies product degree<=84, so 128 bits hold the full product. */
int pqcp_audit_literal_support_norms(const uint64_t *masks,uint64_t count,int d,
    uint64_t target,int weight,uint64_t *bad_index) {
    if (!masks||!bad_index||(d!=3&&d!=5&&d!=11&&d!=43)||weight<0||weight>d) return -1;
    uint64_t all=(UINT64_C(1)<<d)-1;if (target&~all) return -1;
    *bad_index=UINT64_MAX;
    for (uint64_t index=0;index<count;++index) {
        uint64_t mask=masks[index];
        if ((mask&~all)||__builtin_popcountll(mask)!=weight) {*bad_index=index;return 1;}
        uint64_t inverse=mask&1;
        for (int j=1;j<d;++j) inverse|=((mask>>j)&1)<<(d-j);
        __uint128_t product=0;uint64_t bits=mask;
        while (bits) {
            int bit=__builtin_ctzll(bits);product^=(__uint128_t)inverse<<bit;bits&=bits-1;
        }
        uint64_t norm=((uint64_t)product^(uint64_t)(product>>d))&all;
        if (norm!=target) {*bad_index=index;return 1;}
    }
    return 0;
}

/* Exact odd interleave PACFs for EVERY relative phase; retain all preimages. */
int pqcp_odd_cross_vectors(uint64_t even,uint64_t odd,int d,int *output) {
    if(d<3||d>43||!(d&1))return -1;
    uint64_t all=(UINT64_C(1)<<d)-1;
    if((even&~all)||(odd&~all))return -1;
    int cross[43],width=d/2+1;
    for(int u=0;u<d;++u)cross[u]=d-2*__builtin_popcountll(even^block_rotate(odd,u,d,all));
    for(int delta=0;delta<d;++delta)
        for(int r=0;r<width;++r)
            output[delta*width+r]=cross[(r+1-delta+d)%d]+cross[(2*d-r-delta)%d];
    return 0;
}

/* ALL in-block support masks, with exact parity-PACF signatures. Gray flips
 * update S_u by U_(p+u) XOR U_(p-u); no quadratic support term is omitted.
 * Outside coordinates stay fixed. All literal signature preimages survive. */
int pqcp_support_block_options(uint64_t reference,int d,const int *positions,int m,
    uint64_t allowed_weights,uint64_t *masks,uint32_t *signatures,unsigned char *weights,
    uint64_t *stats) {
    if (d<3 || d>47 || !(d&1) || m<1 || m>20 || m>d) return -1;
    uint64_t all=(UINT64_C(1)<<d)-1,selected=0;
    if ((reference&~all) || (allowed_weights>> (d+1))) return -1;
    for (int i=0;i<m;++i) {
        int p=positions[i];
        if (p<0 || p>=d || ((selected>>p)&1)) return -1;
        selected|=UINT64_C(1)<<p;
    }
    uint64_t current=reference&~selected,reflected=0;
    for (int i=0;i<d;++i)
        if ((current>>i)&1) reflected|=UINT64_C(1)<<((d-i)%d);
    uint32_t signature=0;
    for (int u=1;u<=d/2;++u)
        signature|=(uint32_t)(__builtin_popcountll(current&block_rotate(current,u,d,all))&1)<<(u-1);
    int weight=__builtin_popcountll(current);
    uint64_t total=UINT64_C(1)<<m,count=0,half=(UINT64_C(1)<<(d/2))-1;
    for (uint64_t i=0;i<total;++i) {
        if ((allowed_weights>>weight)&1) {
            masks[count]=current;signatures[count]=signature;weights[count]=(unsigned char)weight;++count;
        }
        if (i+1==total) break;
        int p=positions[__builtin_ctzll(i+1)];
        uint64_t forward=block_rotate(current,(d-p)%d,d,all)>>1;
        uint64_t backward=block_rotate(reflected,p,d,all)>>1;
        signature^=(uint32_t)((forward^backward)&half);
        weight+=1-2*(int)((current>>p)&1);
        current^=UINT64_C(1)<<p;
        reflected^=UINT64_C(1)<<((d-p)%d);
    }
    stats[0]=total;stats[1]=count;
    return 0;
}
