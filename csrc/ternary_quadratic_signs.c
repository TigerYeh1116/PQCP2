/* Exact mod-8 PACF / mod-4 cardinality filter on a GF(2) affine space.
 * The second bit of a sum of affine Boolean bits is quadratic.
 * Evaluate every free assignment; only zero syndromes need integer checks.
 * No probabilistic equations, symmetry assumptions, or approximate tests. */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static uint64_t rotate(uint64_t x, int u, int d, uint64_t all) {
    return ((x << u) | (x >> (d-u))) & all;
}

static uint32_t syndrome(const uint64_t *signs, uint64_t active[2][23],
    const int *required, const int *negative, int d, uint64_t all) {
    uint32_t result = 0;
    for (int u = 1; u <= d/2; ++u) {
        int count = 0;
        for (int s = 0; s < 2; ++s)
            count += __builtin_popcountll((signs[s] ^ rotate(signs[s], u, d, all)) & active[s][u-1]);
        result |= (uint32_t)(((count >> 1) ^ (required[u-1] >> 1)) & 1) << (u-1);
    }
    for (int s = 0; s < 2; ++s)
        result |= (uint32_t)(((__builtin_popcountll(signs[s]) >> 1) ^ (negative[s] >> 1)) & 1) << (d/2+s);
    return result;
}

static int make_quadratic_plan(const uint64_t *support, const uint64_t *initial,
    const uint64_t *toggles, int q, int d, const int *wanted, const int *negative,
    uint32_t *constant, uint32_t *linear, uint32_t *quadratic,int maximum_q) {
    if (d < 3 || d > 47 || !(d & 1) || q < 0 || q > maximum_q) return -1;
    uint64_t all = (UINT64_C(1) << d)-1, active[2][23];
    int required[23];
    for (int s = 0; s < 2; ++s) {
        if ((support[s] & ~all) || (initial[s] & ~support[s])) return -1;
        if (negative[s] < 0 || negative[s] > __builtin_popcountll(support[s])) return -1;
        for (int i = 0; i < q; ++i)
            if (toggles[2*i+s] & ~support[s]) return -1;
        for (int u = 1; u <= d/2; ++u)
            active[s][u-1] = support[s] & rotate(support[s], u, d, all);
    }
    for (int u = 1; u <= d/2; ++u) {
        int m = __builtin_popcountll(active[0][u-1])+__builtin_popcountll(active[1][u-1]);
        if (m < wanted[u] || m < -wanted[u] || (m-wanted[u])%2) return -1;
        required[u-1] = (m-wanted[u])/2;
    }
    *constant = syndrome(initial, active, required, negative, d, all);
    memset(quadratic, 0, (size_t)q*q*sizeof(uint32_t));
    for (int i = 0; i < q; ++i) {
        uint64_t signs[2] = {initial[0]^toggles[2*i], initial[1]^toggles[2*i+1]};
        linear[i] = syndrome(signs, active, required, negative, d, all)^*constant;
    }
    for (int i = 0; i < q; ++i)
        for (int j = i+1; j < q; ++j) {
            uint64_t signs[2] = {initial[0]^toggles[2*i]^toggles[2*j],
                                 initial[1]^toggles[2*i+1]^toggles[2*j+1]};
            uint32_t value = syndrome(signs, active, required, negative, d, all)^*constant^linear[i]^linear[j];
            quadratic[i*q+j] = quadratic[j*q+i] = value;
        }
    return 0;
}

int pqcp_quadratic_plan(const uint64_t *support,const uint64_t *initial,
    const uint64_t *toggles,int q,int d,const int *wanted,const int *negative,
    uint32_t *constant,uint32_t *linear,uint32_t *quadratic) {
    return make_quadratic_plan(support,initial,toggles,q,d,wanted,negative,constant,linear,quadratic,24);
}

/* Coefficients only: no 2^q enumeration is attempted for the extended
 * pattern/free parameterization. Public modular enumeration stays <=24. */
int pqcp_quadratic_extended_plan(const uint64_t *support,const uint64_t *initial,
    const uint64_t *toggles,int q,int d,const int *wanted,const int *negative,
    uint32_t *constant,uint32_t *linear,uint32_t *quadratic) {
    return make_quadratic_plan(support,initial,toggles,q,d,wanted,negative,constant,linear,quadratic,47);
}

/* F(p,f) is quadratic in BOTH pattern p and free f. Fixing p changes only
 * the constant and free-linear terms; free-quadratic coefficients remain
 * exactly the same. Packed output uses compact q_free* q_free stride. */
int pqcp_specialize_quadratic_plan(uint32_t constant,const uint32_t *linear,const uint32_t *quadratic,
    int patterns,int free_count,uint64_t pattern,uint32_t *output) {
    if (patterns<0 || patterns>47 || free_count<0 || free_count>24 || patterns+free_count>47 ||
        pattern>=(UINT64_C(1)<<patterns)) return -1;
    int total=patterns+free_count;memset(output,0,601*sizeof(uint32_t));output[0]=constant;
    for (int j=0;j<free_count;++j) {
        output[1+j]=linear[patterns+j];
        for (int k=0;k<free_count;++k) output[25+j*free_count+k]=quadratic[(patterns+j)*total+patterns+k];
    }
    for (uint64_t bits=pattern;bits;bits&=bits-1) {
        int i=__builtin_ctzll(bits);output[0]^=linear[i];
        for (uint64_t rest=bits&(bits-1);rest;rest&=rest-1)
            output[0]^=quadratic[i*total+__builtin_ctzll(rest)];
        for (int j=0;j<free_count;++j) output[1+j]^=quadratic[i*total+patterns+j];
    }
    return 0;
}

int pqcp_quadratic_signs(const uint64_t *support, const uint64_t *initial,
    const uint64_t *toggles, int q, int d, const int *wanted, const int *negative,
    uint64_t budget, uint64_t *output, uint64_t *stats) {
    uint32_t constant, linear[24], quadratic[24*24];
    if (!budget || pqcp_quadratic_plan(support, initial, toggles, q, d, wanted, negative,
                                      &constant, linear, quadratic)) return -1;
    memset(stats, 0, 4*sizeof(uint64_t));
    int split = q/2;
    size_t low_size = (size_t)1 << split, high_size = (size_t)1 << (q-split);
    uint32_t *low = malloc((size_t)(q ? q : 1)*low_size*sizeof(uint32_t));
    uint32_t *high = malloc((size_t)(q ? q : 1)*high_size*sizeof(uint32_t));
    if (!low || !high) { free(low); free(high); return -2; }
    for (int i = 0; i < q; ++i) {
        low[i*low_size] = high[i*high_size] = 0;
        for (size_t v = 1; v < low_size; ++v)
            low[i*low_size+v] = low[i*low_size+(v&(v-1))]^quadratic[i*q+__builtin_ctzll(v)];
        for (size_t v = 1; v < high_size; ++v)
            high[i*high_size+v] = high[i*high_size+(v&(v-1))]^quadratic[i*q+split+__builtin_ctzll(v)];
    }
    uint64_t total = UINT64_C(1) << q, limit = budget < total ? budget : total;
    uint64_t all = (UINT64_C(1) << d)-1, signs[2] = {initial[0], initial[1]}, active[2][23], gray = 0;
    int required[23], status = limit == total ? 0 : 2;
    uint32_t value = constant;
    for (int s = 0; s < 2; ++s)
        for (int u = 1; u <= d/2; ++u)
            active[s][u-1] = support[s] & rotate(support[s], u, d, all);
    for (int u = 1; u <= d/2; ++u)
        required[u-1] = (__builtin_popcountll(active[0][u-1])+__builtin_popcountll(active[1][u-1])-wanted[u])/2;
    for (uint64_t i = 0; i < limit; ++i) {
        ++stats[0];
        if (!value) {
            ++stats[1];
            if (__builtin_popcountll(signs[0]) == negative[0] && __builtin_popcountll(signs[1]) == negative[1]) {
                ++stats[2];
                int okay = 1;
                for (int u = 1; u <= d/2; ++u) {
                    int count = 0;
                    for (int s = 0; s < 2; ++s)
                        count += __builtin_popcountll((signs[s]^rotate(signs[s], u, d, all)) & active[s][u-1]);
                    ++stats[3];
                    if (count != required[u-1]) { okay = 0; break; }
                }
                if (okay) { output[0] = signs[0]; output[1] = signs[1]; status = 1; break; }
            }
        }
        if (i+1 < limit) {
            int bit = __builtin_ctzll(i+1);
            value ^= linear[bit]^low[bit*low_size+(gray&(low_size-1))]^high[bit*high_size+(gray >> split)];
            gray ^= UINT64_C(1) << bit;
            signs[0] ^= toggles[2*bit]; signs[1] ^= toggles[2*bit+1];
        }
    }
    free(low); free(high);
    return status;
}

static uint32_t quotient_reduce(uint32_t value, const uint32_t *basis) {
    for (int p = 24; p >= 0; --p)
        if ((value >> p) & 1) value ^= basis[p];
    return value;
}

/* Fix the prefix. Quotient by the span of ALL tail quadratic coefficient
 * vectors. The remaining tail problem is necessarily linear. Enumerate
 * every solution of that linear problem, and audit the original equations.
 * Columns may be dependent; nullspaces are enumerated, never discarded.
 * Stats: covered assignments, mod8 hits, count hits, exact checks,
 *        prefix branches, projected candidates, maximum tail nullity. */
static int quadratic_hybrid_core(const uint64_t *support, const uint64_t *initial,
    const uint64_t *toggles, int q, int d, const int *wanted, const int *negative,
    int tail, uint64_t budget, uint64_t *output, uint64_t *stats, int orbit_side) {
    if (tail < 1 || tail > 8 || q < 0 || q > 24 || !budget) return -1;
    memset(stats, 0, 7*sizeof(uint64_t));
    /* Keep the original partial-cap behavior and assignment ordering. */
    if (budget < (UINT64_C(1) << q))
        return pqcp_quadratic_signs(support, initial, toggles, q, d, wanted, negative, budget, output, stats);
    if (tail > q) tail = q;
    int prefix = q-tail;
    uint32_t constant, linear[24], quadratic[24*24], basis[25] = {0};
    if (pqcp_quadratic_plan(support, initial, toggles, q, d, wanted, negative,
                            &constant, linear, quadratic)) return -1;
    uint32_t syndrome_mask = UINT32_MAX;
    if (orbit_side >= 0 && ((__builtin_popcountll(support[orbit_side])-2*negative[orbit_side]) & 2))
        syndrome_mask ^= UINT32_C(1) << (d/2+orbit_side);
    constant &= syndrome_mask;
    for (int i = 0; i < q; ++i) linear[i] &= syndrome_mask;
    for (int i = 0; i < q*q; ++i) quadratic[i] &= syndrome_mask;
    /* Preserve the ORIGINAL necessary polynomials before taking the tail
     * quotient. High/low lanes carry original/projected values together.
     * Their XOR derivatives preserve the old Gray/nullspace order exactly. */
    uint32_t raw_constant=constant,raw_linear[24],raw_quadratic[24*24];
    memcpy(raw_linear,linear,(size_t)q*sizeof(uint32_t));
    memcpy(raw_quadratic,quadratic,(size_t)q*q*sizeof(uint32_t));
    uint64_t coverage = orbit_side >= 0 ? 2 : 1;
    for (int i = prefix; i < q; ++i)
        for (int j = i+1; j < q; ++j) {
            uint32_t value = quadratic[i*q+j];
            while (value) {
                int p = 31-__builtin_clz(value);
                if (!basis[p]) { basis[p] = value; break; }
                value ^= basis[p];
            }
        }
    constant = quotient_reduce(constant, basis);
    for (int i = 0; i < q; ++i) linear[i] = quotient_reduce(linear[i], basis);
    for (int i = 0; i < q*q; ++i) quadratic[i] = quotient_reduce(quadratic[i], basis);
    uint64_t packed_linear[24],packed_quadratic[24*24];
    for (int i=0;i<q;++i) packed_linear[i]=(uint64_t)linear[i]|((uint64_t)raw_linear[i]<<32);
    for (int i=0;i<q*q;++i) packed_quadratic[i]=(uint64_t)quadratic[i]|((uint64_t)raw_quadratic[i]<<32);
    int split = prefix/2;
    size_t low_size = (size_t)1 << split, high_size = (size_t)1 << (prefix-split);
    uint64_t *low = malloc((size_t)(prefix ? prefix : 1)*low_size*sizeof(uint64_t));
    uint64_t *high = malloc((size_t)(prefix ? prefix : 1)*high_size*sizeof(uint64_t));
    if (!low || !high) { free(low); free(high); return -2; }
    for (int i = 0; i < prefix; ++i) {
        low[i*low_size] = high[i*high_size] = 0;
        for (size_t v = 1; v < low_size; ++v)
            low[i*low_size+v] = low[i*low_size+(v&(v-1))]^packed_quadratic[i*q+__builtin_ctzll(v)];
        for (size_t v = 1; v < high_size; ++v)
            high[i*high_size+v] = high[i*high_size+(v&(v-1))]^packed_quadratic[i*q+split+__builtin_ctzll(v)];
    }
    uint64_t all = (UINT64_C(1) << d)-1, active[2][23];
    uint64_t branches = UINT64_C(1) << prefix, block = UINT64_C(1) << tail, gray = 0;
    int required[23], status = 0;
    uint64_t value=(uint64_t)constant|((uint64_t)raw_constant<<32),columns[8];
    for (int j = 0; j < tail; ++j) columns[j] = packed_linear[prefix+j];
    for (int s = 0; s < 2; ++s)
        for (int u = 1; u <= d/2; ++u)
            active[s][u-1] = support[s] & rotate(support[s], u, d, all);
    for (int u = 1; u <= d/2; ++u)
        required[u-1] = (__builtin_popcountll(active[0][u-1])+__builtin_popcountll(active[1][u-1])-wanted[u])/2;
    for (uint64_t i = 0; i < branches; ++i) {
        ++stats[4];
        uint32_t rows[25] = {0}, labels[25] = {0}, nullspace[8];
        int nullity = 0;
        for (int j = 0; j < tail; ++j) {
            uint32_t row = (uint32_t)columns[j], label = UINT32_C(1) << j;
            while (row) {
                int p = 31-__builtin_clz(row);
                if (!rows[p]) { rows[p] = row; labels[p] = label; break; }
                row ^= rows[p]; label ^= labels[p];
            }
            if (!row) nullspace[nullity++] = label;
        }
        if ((uint64_t)nullity > stats[6]) stats[6] = (uint64_t)nullity;
        uint32_t rhs = (uint32_t)value, solution = 0;
        while (rhs) {
            int p = 31-__builtin_clz(rhs);
            if (!rows[p]) break;
            rhs ^= rows[p]; solution ^= labels[p];
        }
        if (!rhs) {
            for (uint32_t v = 0; v < (UINT32_C(1) << nullity); ++v) {
                ++stats[5];
                uint32_t bits=solution,original=(uint32_t)(value>>32);
                while (bits) {
                    int j = __builtin_ctz(bits);
                    bits &= bits-1;
                    original^=(uint32_t)(columns[j]>>32);
                    for (uint32_t other=bits;other;other&=other-1)
                        original^=raw_quadratic[(prefix+j)*q+prefix+__builtin_ctz(other)];
                }
                #ifdef PQCP_HYBRID_AUDIT_RESIDUAL
                /* Diagnostic build only: audit EVERY projected candidate
                 * against the old direct integer syndrome, not just hits. */
                uint64_t audit_signs[2]={initial[0],initial[1]};
                for (uint64_t remaining=gray;remaining;remaining&=remaining-1) {
                    int j=__builtin_ctzll(remaining);
                    audit_signs[0]^=toggles[2*j];audit_signs[1]^=toggles[2*j+1];
                }
                for (bits=solution;bits;bits&=bits-1) {
                    int j=__builtin_ctz(bits);
                    audit_signs[0]^=toggles[2*(prefix+j)];audit_signs[1]^=toggles[2*(prefix+j)+1];
                }
                if (original!=(syndrome(audit_signs,active,required,negative,d,all)&syndrome_mask)) {
                    free(low);free(high);return -3;
                }
                #endif
                if (!original) {
                    /* Reconstruct signs only after the ORIGINAL mod8
                     * syndrome vanishes. Exact counts/PACF still follow. */
                    uint64_t signs[2]={initial[0],initial[1]};
                    for (uint64_t remaining=gray;remaining;remaining&=remaining-1) {
                        int j=__builtin_ctzll(remaining);
                        signs[0]^=toggles[2*j];signs[1]^=toggles[2*j+1];
                    }
                    for (bits=solution;bits;bits&=bits-1) {
                        int j=__builtin_ctz(bits);
                        signs[0]^=toggles[2*(prefix+j)];signs[1]^=toggles[2*(prefix+j)+1];
                    }
                    ++stats[1];
                    if (orbit_side >= 0 && __builtin_popcountll(signs[orbit_side]) != negative[orbit_side] &&
                        __builtin_popcountll(signs[orbit_side]) == __builtin_popcountll(support[orbit_side])-negative[orbit_side])
                        signs[orbit_side] ^= support[orbit_side];
                    if (__builtin_popcountll(signs[0]) == negative[0] && __builtin_popcountll(signs[1]) == negative[1]) {
                        ++stats[2];
                        int okay = 1;
                        for (int u = 1; u <= d/2; ++u) {
                            int count = 0;
                            for (int s = 0; s < 2; ++s)
                                count += __builtin_popcountll((signs[s]^rotate(signs[s], u, d, all)) & active[s][u-1]);
                            ++stats[3];
                            if (count != required[u-1]) { okay = 0; break; }
                        }
                        if (okay) { output[0] = signs[0]; output[1] = signs[1]; stats[0] += coverage; status = 1; goto finished; }
                    }
                }
                if (v+1 < (UINT32_C(1) << nullity)) solution ^= nullspace[__builtin_ctz(v+1)];
            }
        }
        stats[0] += coverage*block; /* Necessary equations account for every omitted tail/orientation. */
        if (i+1 < branches) {
            int bit = __builtin_ctzll(i+1);
            value ^= packed_linear[bit]^low[bit*low_size+(gray&(low_size-1))]^high[bit*high_size+(gray >> split)];
            gray ^= UINT64_C(1) << bit;
            for (int j = 0; j < tail; ++j) columns[j] ^= packed_quadratic[bit*q+prefix+j];
        }
    }
finished:
    free(low); free(high);
    return status;
}

int pqcp_quadratic_bit_tail_signs(const uint64_t *,const uint64_t *,const uint64_t *,int,int,
    const int *,const int *,int,uint64_t,uint64_t *,uint64_t *);

int pqcp_quadratic_hybrid_signs(const uint64_t *support, const uint64_t *initial,
    const uint64_t *toggles, int q, int d, const int *wanted, const int *negative,
    int tail, uint64_t budget, uint64_t *output, uint64_t *stats) {
    if (d==43 && q>=0 && q<=24 && tail>=1 && tail<=6 && budget>=(UINT64_C(1)<<q))
        return pqcp_quadratic_bit_tail_signs(support,initial,toggles,q,d,wanted,negative,tail,budget,output,stats);
    return quadratic_hybrid_core(support,initial,toggles,q,d,wanted,negative,tail,budget,output,stats,-1);
}

/* Exact truth-table intersection, at most 64 tails per prefix. Every bit
 * represents one literal tail assignment. Empty intersections certify a
 * whole prefix without a rank/genericity assumption. Full-budget L43
 * public hybrid fibres use this entry; partial-budget order is unchanged.
 * Stats[5] counts ORIGINAL mod8 survivors, not relaxed projected models. */
static int quadratic_bit_tail_core(const uint64_t *support,const uint64_t *initial,
    const uint64_t *toggles,int q,int d,const int *wanted,const int *negative,
    int tail,uint64_t budget,uint64_t *output,uint64_t *stats,
    uint32_t *models,uint64_t capacity,uint64_t *count,const uint32_t *provided) {
    if (q<0 || q>24 || tail<1 || tail>8 || !budget) return -1;
    if (models && (tail>6 || !capacity || budget<(UINT64_C(1)<<q))) return -1;
    if (tail>6 || budget<(UINT64_C(1)<<q))
        return pqcp_quadratic_hybrid_signs(support,initial,toggles,q,d,wanted,negative,tail,budget,output,stats);
    memset(stats,0,7*sizeof(uint64_t));
    if (tail>q) tail=q;
    int prefix=q-tail,members=1<<tail;
    uint32_t constant,linear[24],quadratic[24*24];
    if (provided) {
        constant=provided[0];memcpy(linear,provided+1,(size_t)q*sizeof(uint32_t));
        memcpy(quadratic,provided+25,(size_t)q*q*sizeof(uint32_t));
    } else if (pqcp_quadratic_plan(support,initial,toggles,q,d,wanted,negative,&constant,linear,quadratic)) return -1;
    uint32_t variable_fields=0;
    for (int i=0;i<q;++i) variable_fields|=linear[i];
    for (int i=0;i<q*q;++i) variable_fields|=quadratic[i];
    if (constant&~variable_fields) {stats[0]=UINT64_C(1)<<q;return 0;}
    uint64_t valid=members==64?UINT64_MAX:(UINT64_C(1)<<members)-1;
    uint64_t variable[6]={0},combinations[64]={0},pure[25]={0};
    static const uint64_t truth[6]={UINT64_C(0xaaaaaaaaaaaaaaaa),UINT64_C(0xcccccccccccccccc),
        UINT64_C(0xf0f0f0f0f0f0f0f0),UINT64_C(0xff00ff00ff00ff00),
        UINT64_C(0xffff0000ffff0000),UINT64_C(0xffffffff00000000)};
    for (int i=0;i<tail;++i) variable[i]=truth[i]&valid;
    for (int i=0;i<tail;++i) for (int j=i+1;j<tail;++j) {
        uint64_t product=variable[i]&variable[j];
        for (uint32_t fields=quadratic[(prefix+i)*q+prefix+j];fields;fields&=fields-1)
            pure[__builtin_ctz(fields)]^=product;
    }
    for (int i=1;i<members;++i) combinations[i]=combinations[i&(i-1)]^variable[__builtin_ctz(i)];
    /* One byte per equation carries its tail-linear coefficient mask.
     * Four XOR words update all equations after a prefix Gray step. */
    uint64_t state[4]={0},changes[24][4]={{0}};
    unsigned char *selectors=(unsigned char *)state;
    for (int u=0;u<d/2+2;++u) for (int j=0;j<tail;++j) {
        selectors[u]|=(unsigned char)(((linear[prefix+j]>>u)&1)<<j);
        for (int i=0;i<prefix;++i)
            ((unsigned char *)changes[i])[u]|=(unsigned char)(((quadratic[i*q+prefix+j]>>u)&1)<<j);
    }
    int split=prefix/2;size_t ls=(size_t)1<<split,hs=(size_t)1<<(prefix-split);
    uint32_t *low=malloc((size_t)(prefix?prefix:1)*ls*sizeof(uint32_t));
    uint32_t *high=malloc((size_t)(prefix?prefix:1)*hs*sizeof(uint32_t));
    if (!low || !high) {free(low);free(high);return -2;}
    for (int i=0;i<prefix;++i) {
        low[i*ls]=high[i*hs]=0;
        for (size_t v=1;v<ls;++v) low[i*ls+v]=low[i*ls+(v&(v-1))]^quadratic[i*q+__builtin_ctzll(v)];
        for (size_t v=1;v<hs;++v) high[i*hs+v]=high[i*hs+(v&(v-1))]^quadratic[i*q+split+__builtin_ctzll(v)];
    }
    uint64_t all=(UINT64_C(1)<<d)-1,active[2][23],gray=0,branches=UINT64_C(1)<<prefix;
    int required[23],status=0;uint32_t value=constant;
    for (int s=0;s<2;++s) for (int u=1;u<=d/2;++u) active[s][u-1]=support[s]&rotate(support[s],u,d,all);
    for (int u=1;u<=d/2;++u)
        required[u-1]=(__builtin_popcountll(active[0][u-1])+__builtin_popcountll(active[1][u-1])-wanted[u])/2;
    for (uint64_t i=0;i<branches;++i) {
        ++stats[4];uint64_t feasible=valid;
        for (int step=0;step<d/2+2 && feasible;++step) {
            int u=step<2?d/2+step:step-2;
            feasible&=~(pure[u]^combinations[selectors[u]]^(((value>>u)&1)?valid:0));
        }
        while (feasible) {
            unsigned t=(unsigned)__builtin_ctzll(feasible);feasible&=feasible-1;
            ++stats[1];++stats[5];
            if (models) {
                if (*count<capacity) models[*count]=(uint32_t)(gray|((uint64_t)t<<prefix));
                ++*count;continue;
            }
            uint64_t signs[2]={initial[0],initial[1]},bits=gray|((uint64_t)t<<prefix);
            while (bits) {int j=__builtin_ctzll(bits);bits&=bits-1;signs[0]^=toggles[2*j];signs[1]^=toggles[2*j+1];}
            if (__builtin_popcountll(signs[0])!=negative[0] || __builtin_popcountll(signs[1])!=negative[1]) continue;
            ++stats[2];int okay=1;
            for (int u=1;u<=d/2;++u) {
                int cuts=0;
                for (int s=0;s<2;++s) cuts+=__builtin_popcountll((signs[s]^rotate(signs[s],u,d,all))&active[s][u-1]);
                ++stats[3];if (cuts!=required[u-1]) {okay=0;break;}
            }
            if (okay) {output[0]=signs[0];output[1]=signs[1];stats[0]+=(uint64_t)t+1;status=1;goto bit_finished;}
        }
        stats[0]+=members;
        if (i+1<branches) {
            int bit=__builtin_ctzll(i+1);
            value^=linear[bit]^low[bit*ls+(gray&(ls-1))]^high[bit*hs+(gray>>split)];
            gray^=UINT64_C(1)<<bit;
            for (int lane=0;lane<4;++lane) state[lane]^=changes[bit][lane];
        }
    }
bit_finished:
    free(low);free(high);return status;
}

int pqcp_quadratic_bit_tail_signs(const uint64_t *support,const uint64_t *initial,
    const uint64_t *toggles,int q,int d,const int *wanted,const int *negative,
    int tail,uint64_t budget,uint64_t *output,uint64_t *stats) {
    return quadratic_bit_tail_core(support,initial,toggles,q,d,wanted,negative,
                                  tail,budget,output,stats,NULL,0,NULL,NULL);
}

/* Diagnostic export enumerates ALL original mod8 models through precisely
 * the production intersection loop. Count remains exact on overflow. */
int pqcp_quadratic_bit_tail_models(const uint64_t *support,const uint64_t *initial,
    const uint64_t *toggles,int q,int d,const int *wanted,const int *negative,
    int tail,uint64_t capacity,uint32_t *models,uint64_t *count) {
    if (q<0 || q>24 || !capacity) return -1;
    uint64_t output[2],stats[7];*count=0;
    return quadratic_bit_tail_core(support,initial,toggles,q,d,wanted,negative,
                                  tail,UINT64_C(1)<<q,output,stats,models,capacity,count,NULL);
}

/* Two affine cosets with the SAME free toggles, separate integer targets.
 * Write b1=b0+delta. The quadratic part in f is independent of b, so
 * F1(f)=F0(f)+A(f), with A affine. Build F0(f,t) at b0 using delta as
 * the final toggle, then specialize t=1 and adjust target RHS bits.
 * One quadratic Gray walk and one tail-linear state serve both labels.
 * Each label retains its own zero set, counts, witness and complete UNSAT.
 * Partial budgets are deliberately rejected by this experimental entry.
 */
static int paired_bit_tail_core(const uint64_t *support,const uint64_t *initial,
    const uint64_t *toggles,int q,int d,const int *wanted,const int *negative,
    int tail,uint64_t budget,uint64_t *output,int *statuses,uint64_t *stats,
    uint32_t *models,uint64_t capacity,uint64_t *counts) {
    if (q<0 || q>24 || d<3 || d>47 || !(d&1) || tail<1 || tail>6 ||
        budget<(UINT64_C(1)<<q) || (models && !capacity)) return -1;
    uint64_t all=(UINT64_C(1)<<d)-1,total=UINT64_C(1)<<q,active[2][23];
    int required[2][23];
    for (int s=0;s<2;++s) {
        if (support[s]&~all) return -1;
        for (int k=0;k<2;++k)
            if ((initial[2*k+s]&~support[s]) || negative[2*k+s]<0 ||
                negative[2*k+s]>__builtin_popcountll(support[s])) return -1;
        for (int u=1;u<=d/2;++u) active[s][u-1]=support[s]&rotate(support[s],u,d,all);
    }
    uint32_t rhs_difference=0;
    for (int u=1;u<=d/2;++u) {
        int terms=__builtin_popcountll(active[0][u-1])+__builtin_popcountll(active[1][u-1]);
        for (int k=0;k<2;++k) {
            int value=wanted[k*d+u];
            if (value>terms || value<-terms || (terms-value)%2) return -1;
            required[k][u-1]=(terms-value)/2;
        }
        rhs_difference|=(uint32_t)(((required[0][u-1]>>1)^(required[1][u-1]>>1))&1)<<(u-1);
    }
    for (int s=0;s<2;++s)
        rhs_difference|=(uint32_t)(((negative[s]>>1)^(negative[2+s]>>1))&1)<<(d/2+s);
    uint64_t extended_toggles[50];
    memcpy(extended_toggles,toggles,(size_t)2*q*sizeof(uint64_t));
    extended_toggles[2*q]=initial[0]^initial[2];
    extended_toggles[2*q+1]=initial[1]^initial[3];
    uint32_t constant,linear[25],quadratic[25*25];
    if (make_quadratic_plan(support,initial,extended_toggles,q+1,d,wanted,negative,
                            &constant,linear,quadratic,47)) return -1;
    uint32_t difference=linear[q]^rhs_difference,delta_linear[24];
    for (int j=0;j<q;++j) delta_linear[j]=quadratic[j*(q+1)+q];
    memset(stats,0,14*sizeof(uint64_t));statuses[0]=statuses[1]=0;
    if (counts) counts[0]=counts[1]=0;
    unsigned live=3;
    for (int k=0;k<2;++k) {
        uint32_t variable_fields=0;
        for (int j=0;j<q;++j) {
            variable_fields|=linear[j]^(k?delta_linear[j]:0);
            for (int z=0;z<q;++z) variable_fields|=quadratic[j*(q+1)+z];
        }
        if ((constant^(k?difference:0))&~variable_fields) {
            live&=~(1U<<k);stats[7*k]=total;
        }
    }
    if (!live) return 0;
    if (tail>q) tail=q;
    int prefix=q-tail,members=1<<tail;
    uint64_t valid=members==64?UINT64_MAX:(UINT64_C(1)<<members)-1;
    uint64_t variable[6],combinations[64]={0},pure[25]={0};
    static const uint64_t truth[6]={UINT64_C(0xaaaaaaaaaaaaaaaa),UINT64_C(0xcccccccccccccccc),
        UINT64_C(0xf0f0f0f0f0f0f0f0),UINT64_C(0xff00ff00ff00ff00),
        UINT64_C(0xffff0000ffff0000),UINT64_C(0xffffffff00000000)};
    for (int j=0;j<tail;++j) variable[j]=truth[j]&valid;
    for (int j=0;j<tail;++j) for (int z=j+1;z<tail;++z) {
        uint64_t product=variable[j]&variable[z];
        for (uint32_t fields=quadratic[(prefix+j)*(q+1)+prefix+z];fields;fields&=fields-1)
            pure[__builtin_ctz(fields)]^=product;
    }
    for (int j=1;j<members;++j) combinations[j]=combinations[j&(j-1)]^variable[__builtin_ctz(j)];
    uint64_t state[4]={0},changes[24][4]={{0}};
    unsigned char *selectors=(unsigned char *)state;
    unsigned char delta_selectors[25]={0};
    for (int u=0;u<d/2+2;++u) for (int j=0;j<tail;++j) {
        selectors[u]|=(unsigned char)(((linear[prefix+j]>>u)&1)<<j);
        delta_selectors[u]|=(unsigned char)(((delta_linear[prefix+j]>>u)&1)<<j);
        for (int i=0;i<prefix;++i)
            ((unsigned char *)changes[i])[u]|=(unsigned char)(((quadratic[i*(q+1)+prefix+j]>>u)&1)<<j);
    }
    int split=prefix/2;size_t ls=(size_t)1<<split,hs=(size_t)1<<(prefix-split);
    uint32_t *low=malloc((size_t)(prefix?prefix:1)*ls*sizeof(uint32_t));
    uint32_t *high=malloc((size_t)(prefix?prefix:1)*hs*sizeof(uint32_t));
    if (!low || !high) {free(low);free(high);return -2;}
    for (int i=0;i<prefix;++i) {
        low[i*ls]=high[i*hs]=0;
        for (size_t v=1;v<ls;++v) low[i*ls+v]=low[i*ls+(v&(v-1))]^quadratic[i*(q+1)+__builtin_ctzll(v)];
        for (size_t v=1;v<hs;++v) high[i*hs+v]=high[i*hs+(v&(v-1))]^quadratic[i*(q+1)+split+__builtin_ctzll(v)];
    }
    uint64_t gray=0,branches=UINT64_C(1)<<prefix;
    uint32_t value=constant,delta_value=difference;
    for (uint64_t i=0;i<branches && live;++i) {
        uint64_t feasible[2]={live&1?valid:0,live&2?valid:0};
        for (int k=0;k<2;++k) if (live&(1U<<k)) ++stats[7*k+4];
        for (int step=0;step<d/2+2 && (feasible[0] || feasible[1]);++step) {
            int u=step<2?d/2+step:step-2;
            uint64_t row=pure[u]^combinations[selectors[u]]^(((value>>u)&1)?valid:0);
            feasible[0]&=~row;
            feasible[1]&=~(row^combinations[delta_selectors[u]]^(((delta_value>>u)&1)?valid:0));
        }
        for (int k=0;k<2;++k) {
            uint64_t *st=stats+7*k;
            while (feasible[k]) {
                unsigned t=(unsigned)__builtin_ctzll(feasible[k]);feasible[k]&=feasible[k]-1;
                ++st[1];++st[5];
                uint64_t index=gray|((uint64_t)t<<prefix);
                #ifdef PQCP_PAIRED_AUDIT_SYNDROME
                uint64_t audit_signs[2]={initial[2*k],initial[2*k+1]};
                for (uint64_t bits=index;bits;bits&=bits-1) {
                    int j=__builtin_ctzll(bits);audit_signs[0]^=toggles[2*j];audit_signs[1]^=toggles[2*j+1];
                }
                if (syndrome(audit_signs,active,required[k],negative+2*k,d,all)) {
                    free(low);free(high);return -3;
                }
                #endif
                if (models) {
                    if (counts[k]<capacity) models[(uint64_t)k*capacity+counts[k]]=(uint32_t)index;
                    ++counts[k];continue;
                }
                uint64_t signs[2]={initial[2*k],initial[2*k+1]};
                for (uint64_t bits=index;bits;bits&=bits-1) {
                    int j=__builtin_ctzll(bits);signs[0]^=toggles[2*j];signs[1]^=toggles[2*j+1];
                }
                if (__builtin_popcountll(signs[0])!=negative[2*k] ||
                    __builtin_popcountll(signs[1])!=negative[2*k+1]) continue;
                ++st[2];int okay=1;
                for (int u=1;u<=d/2;++u) {
                    int cuts=0;
                    for (int s=0;s<2;++s) cuts+=__builtin_popcountll((signs[s]^rotate(signs[s],u,d,all))&active[s][u-1]);
                    ++st[3];if (cuts!=required[k][u-1]) {okay=0;break;}
                }
                if (okay) {
                    output[2*k]=signs[0];output[2*k+1]=signs[1];statuses[k]=1;
                    st[0]+=(uint64_t)t+1;live&=~(1U<<k);break;
                }
            }
            if (live&(1U<<k)) st[0]+=members;
        }
        if (i+1<branches && live) {
            int bit=__builtin_ctzll(i+1);
            value^=linear[bit]^low[bit*ls+(gray&(ls-1))]^high[bit*hs+(gray>>split)];
            delta_value^=delta_linear[bit];gray^=UINT64_C(1)<<bit;
            for (int lane=0;lane<4;++lane) state[lane]^=changes[bit][lane];
        }
    }
    free(low);free(high);return 0;
}

int pqcp_quadratic_paired_bit_tail_signs(const uint64_t *support,const uint64_t *initial,
    const uint64_t *toggles,int q,int d,const int *wanted,const int *negative,
    int tail,uint64_t budget,uint64_t *output,int *statuses,uint64_t *stats) {
    return paired_bit_tail_core(support,initial,toggles,q,d,wanted,negative,tail,budget,
                                output,statuses,stats,NULL,0,NULL);
}

/* Diagnostic: all modular models for each label, through the production
 * intersection loop. Overflow counts remain exact; no integer SAT claim. */
int pqcp_quadratic_paired_bit_tail_models(const uint64_t *support,const uint64_t *initial,
    const uint64_t *toggles,int q,int d,const int *wanted,const int *negative,
    int tail,uint64_t capacity,uint32_t *models,uint64_t *counts) {
    if (q<0 || q>24 || !capacity) return -1;
    uint64_t output[4],stats[14];int statuses[2];
    return paired_bit_tail_core(support,initial,toggles,q,d,wanted,negative,tail,UINT64_C(1)<<q,
                                output,statuses,stats,models,capacity,counts);
}

/* Exact fixed-weight prefix and <=64 free suffixes. Pattern toggles pin
 * every selected-row sign literally; free toggles never change that row.
 * Build F(p,f) once. Gosper transitions update its prefix value and mixed
 * tail selectors one changed bit at a time, including intermediate weights.
 * Only the final exact-weight p is visited. No per-pattern specialization.
 */
/* Columns of Q_FF impose homogeneous linear equations on combinations of
 * syndrome fields. Their nullspace contains precisely the combinations
 * whose free-quadratic terms vanish, independent of the weight pattern. */
static int affine_combination_masks(const uint32_t *quadratic,int total,int patterns,
    int free_count,int fields,uint32_t *masks) {
    uint32_t basis[25]={0};
    uint32_t field_mask=(UINT32_C(1)<<fields)-1;
    for (int i=0;i<free_count;++i) for (int j=i+1;j<free_count;++j) {
        uint32_t row=quadratic[(patterns+i)*total+patterns+j]&field_mask;
        while (row) {
            int pivot=31-__builtin_clz(row);
            if (!basis[pivot]) {basis[pivot]=row;break;}
            row^=basis[pivot];
        }
    }
    int count=0;
    for (int j=0;j<fields;++j) if (!basis[j]) {
        uint32_t mask=UINT32_C(1)<<j;
        for (int i=0;i<fields;++i)
            if (basis[i] && __builtin_parity(basis[i]&mask)) mask^=UINT32_C(1)<<i;
        masks[count++]=mask;
    }
    return count;
}

static uint32_t change_syndrome_coordinates(uint32_t value,const uint32_t *masks,int fields) {
    uint32_t result=value&~((UINT32_C(1)<<fields)-1);
    for (int i=0;i<fields;++i) result|=(uint32_t)__builtin_parity(value&masks[i])<<i;
    return result;
}

/* Extend the affine-combination basis to an invertible coordinate matrix.
 * Count fields keep their original coordinates; only PACF fields change. */
static void rebase_quadratic_fields(uint32_t *value,uint32_t *linear,uint32_t *quadratic,
    int total,int patterns,int free_count,int fields) {
    uint32_t masks[25],basis[25]={0};
    int count=affine_combination_masks(quadratic,total,patterns,free_count,fields,masks);
    for (int i=0;i<count;++i) {
        uint32_t row=masks[i];
        while (row) {
            int pivot=31-__builtin_clz(row);
            if (!basis[pivot]) {basis[pivot]=row;break;}
            row^=basis[pivot];
        }
    }
    for (int i=0;i<fields;++i) {
        uint32_t row=UINT32_C(1)<<i;
        while (row) {
            int pivot=31-__builtin_clz(row);
            if (!basis[pivot]) {basis[pivot]=row;masks[count++]=UINT32_C(1)<<i;break;}
            row^=basis[pivot];
        }
    }
    *value=change_syndrome_coordinates(*value,masks,fields);
    for (int i=0;i<total;++i) {
        linear[i]=change_syndrome_coordinates(linear[i],masks,fields);
        for (int j=i+1;j<total;++j) {
            uint32_t transformed=change_syndrome_coordinates(quadratic[i*total+j],masks,fields);
            quadratic[i*total+j]=quadratic[j*total+i]=transformed;
        }
    }
}

/* Exact Gaussian contradiction test of ALL affine equation combinations.
 * The remaining nonlinear equations are still evaluated by the old core. */
static int affine_pattern_contradiction(uint32_t value,const uint32_t *prefix_linear,
    const unsigned char *selectors,int prefix,const uint32_t *masks,int count) {
    uint32_t basis[14]={0};int targets[14]={0};
    for (int i=0;i<count;++i) {
        uint32_t mask=masks[i],row=0;int rhs=__builtin_parity(value&mask);
        for (int j=0;j<prefix;++j)
            row|=(uint32_t)__builtin_parity(prefix_linear[j]&mask)<<j;
        unsigned tail=0;
        for (uint32_t bits=mask;bits;bits&=bits-1) tail^=selectors[__builtin_ctz(bits)];
        row|=(uint32_t)tail<<prefix;
        while (row) {
            int pivot=31-__builtin_clz(row);
            if (!basis[pivot]) {basis[pivot]=row;targets[pivot]=rhs;break;}
            row^=basis[pivot];rhs^=targets[pivot];
        }
        if (!row && rhs) return 1;
    }
    return 0;
}

/* Gaussian elimination of PREFIX columns only. RHS entries encode affine
 * functions of all original tail bits. Residual rows restrict the tail;
 * triangular substitution leaves each pivot depending only on nonpivots
 * and tails. No nonlinear monomial is treated as an independent variable. */
static uint64_t affine_prefix_plan(uint32_t value,const uint32_t *prefix_linear,
    const unsigned char *selectors,int prefix,int tail,const uint32_t *masks,int count,
    const uint64_t *combinations,uint64_t valid,uint32_t *basis,unsigned *targets) {
    memset(basis,0,8*sizeof(uint32_t));memset(targets,0,8*sizeof(unsigned));
    uint64_t feasible=valid;
    unsigned tail_mask=(1U<<tail)-1;
    for (int i=0;i<count;++i) {
        uint32_t mask=masks[i],row=0;
        unsigned rhs=(unsigned)__builtin_parity(value&mask)<<tail;
        for (int j=0;j<prefix;++j) row|=(uint32_t)__builtin_parity(prefix_linear[j]&mask)<<j;
        for (uint32_t bits=mask;bits;bits&=bits-1) rhs^=selectors[__builtin_ctz(bits)];
        while (row) {
            int pivot=31-__builtin_clz(row);
            if (!basis[pivot]) {basis[pivot]=row;targets[pivot]=rhs;break;}
            row^=basis[pivot];rhs^=targets[pivot];
        }
        if (!row) feasible&=~(combinations[rhs&tail_mask]^((rhs>>tail)?valid:0));
        if (!feasible) return 0;
    }
    for (int i=0;i<prefix;++i) if (basis[i]) {
        for (int j=0;j<i;++j) if (basis[j] && (basis[i]>>j&1)) {
            basis[i]^=basis[j];targets[i]^=targets[j];
        }
    }
    return feasible;
}

static int weight_prefix_core(const uint64_t *support,const uint64_t *base,
    const uint64_t *pattern_toggles,int pattern_count,int selected,
    const uint64_t *check_masks,const int *check_rhs,int checks,
    const uint64_t *free_toggles,int free_count,int d,const int *wanted,const int *negative,
    uint64_t budget,uint64_t *output,uint64_t *stats,uint32_t *models,uint64_t capacity,uint64_t *count,
    int count_elimination,int affine_pruning,uint64_t *extra) {
    if (d<3 || d>47 || !(d&1) || selected<0 || selected>1 || pattern_count<0 || pattern_count>17 ||
        free_count<0 || free_count>14 || pattern_count+free_count>24 || checks<0 || checks>96 || !budget ||
        negative[selected]<0 || negative[selected]>pattern_count || (models && !capacity)) return -1;
    uint64_t upper=1;int minority=negative[selected];
    if (pattern_count-minority<minority) minority=pattern_count-minority;
    for (int j=1;j<=minority;++j) upper=upper*(uint64_t)(pattern_count-j+1)/(uint64_t)j;
    uint64_t free_size=UINT64_C(1)<<free_count;
    if (upper>budget/free_size) return -1; /* Never claim unvisited prefixes. */
    int tail=free_count<6?free_count:6,fprefix=free_count-tail;
    uint64_t members=UINT64_C(1)<<tail;
    uint64_t all=(UINT64_C(1)<<d)-1,active[2][23],pinned=0;
    int required[23],total=pattern_count+free_count,other=1-selected;
    if (base[selected] || __builtin_popcountll(support[selected])!=pattern_count) return -1;
    for (int s=0;s<2;++s) {
        if ((support[s]&~all) || (base[s]&~support[s]) || negative[s]<0 ||
            negative[s]>__builtin_popcountll(support[s])) return -1;
        for (int u=1;u<=d/2;++u) active[s][u-1]=support[s]&rotate(support[s],u,d,all);
    }
    for (int j=0;j<pattern_count;++j) {
        uint64_t bit=pattern_toggles[2*j+selected];
        if (!bit || (bit&(bit-1)) || (bit&pinned) || (bit&~support[selected])) return -1;
        pinned|=bit;
    }
    if (pinned!=support[selected]) return -1;
    for (int j=0;j<free_count;++j) if (free_toggles[2*j+selected]) return -1;
    for (int j=0;j<checks;++j)
        if ((check_masks[j]>>pattern_count) || check_rhs[j]<0 || check_rhs[j]>1) return -1;
    for (int u=1;u<=d/2;++u) {
        int terms=__builtin_popcountll(active[0][u-1])+__builtin_popcountll(active[1][u-1]);
        if (wanted[u]>terms || wanted[u]<-terms || (terms-wanted[u])%2) return -1;
        required[u-1]=(terms-wanted[u])/2;
    }
    uint64_t full_toggles[48];
    memcpy(full_toggles,pattern_toggles,(size_t)2*pattern_count*sizeof(uint64_t));
    memcpy(full_toggles+2*pattern_count,free_toggles,(size_t)2*free_count*sizeof(uint64_t));
    uint32_t value,linear[24],quadratic[24*24];
    if (make_quadratic_plan(support,base,full_toggles,total,d,wanted,negative,&value,linear,quadratic,47)) return -1;
    if (affine_pruning==2) rebase_quadratic_fields(&value,linear,quadratic,total,pattern_count,free_count,d/2);
    memset(stats,0,9*sizeof(uint64_t));if (count) *count=0;
    if (extra) memset(extra,0,(affine_pruning==3?5:affine_pruning?3:2)*sizeof(uint64_t));
    uint32_t affine_masks[25];int affine_count=0;
    if (affine_pruning==1 || affine_pruning==3) affine_count=affine_combination_masks(quadratic,total,pattern_count,free_count,d/2+2,affine_masks);
    uint32_t count_mask=UINT32_C(1)<<(d/2+other);
    int count_affine=count_elimination;
    for (int j=0;j<free_count && count_affine;++j) for (int z=j+1;z<free_count;++z)
        if (quadratic[(pattern_count+j)*total+pattern_count+z]&count_mask) count_affine=0;
    uint64_t valid=members==64?UINT64_MAX:(UINT64_C(1)<<members)-1;
    uint64_t variable[6],combinations[64]={0},pure[25]={0};
    static const uint64_t truth[6]={UINT64_C(0xaaaaaaaaaaaaaaaa),UINT64_C(0xcccccccccccccccc),
        UINT64_C(0xf0f0f0f0f0f0f0f0),UINT64_C(0xff00ff00ff00ff00),
        UINT64_C(0xffff0000ffff0000),UINT64_C(0xffffffff00000000)};
    for (int j=0;j<tail;++j) variable[j]=truth[j]&valid;
    for (int j=0;j<tail;++j) for (int z=j+1;z<tail;++z) {
        uint64_t product=variable[j]&variable[z];
        for (uint32_t fields=quadratic[(pattern_count+fprefix+j)*total+pattern_count+fprefix+z];fields;fields&=fields-1)
            pure[__builtin_ctz(fields)]^=product;
    }
    for (unsigned j=1;j<members;++j) combinations[j]=combinations[j&(j-1)]^variable[__builtin_ctz(j)];
    uint64_t state[4]={0},changes[17][4]={{0}};
    uint64_t free_changes[8][4]={{0}};
    uint32_t free_linear[8]={0},free_lookup[8][256]={{0}};
    for (int j=0;j<fprefix;++j) {
        free_linear[j]=linear[pattern_count+j];
        for (unsigned bits=1;bits<(1U<<fprefix);++bits)
            free_lookup[j][bits]=free_lookup[j][bits&(bits-1)]^
                quadratic[(pattern_count+j)*total+pattern_count+__builtin_ctz(bits)];
    }
    unsigned char *selectors=(unsigned char *)state;
    for (int u=0;u<d/2+2;++u) for (int j=0;j<tail;++j) {
        selectors[u]|=(unsigned char)(((linear[pattern_count+fprefix+j]>>u)&1)<<j);
        for (int i=0;i<pattern_count;++i)
            ((unsigned char *)changes[i])[u]|=(unsigned char)(((quadratic[i*total+pattern_count+fprefix+j]>>u)&1)<<j);
        for (int i=0;i<fprefix;++i)
            ((unsigned char *)free_changes[i])[u]|=(unsigned char)(((quadratic[(pattern_count+i)*total+pattern_count+fprefix+j]>>u)&1)<<j);
    }
    int split=pattern_count/2;size_t ls=(size_t)1<<split,hs=(size_t)1<<(pattern_count-split);
    uint32_t *low=malloc((size_t)(pattern_count?pattern_count:1)*ls*sizeof(uint32_t));
    uint32_t *high=malloc((size_t)(pattern_count?pattern_count:1)*hs*sizeof(uint32_t));
    if (!low || !high) {free(low);free(high);return -2;}
    for (int i=0;i<pattern_count;++i) {
        low[i*ls]=high[i*hs]=0;
        for (size_t v=1;v<ls;++v) low[i*ls+v]=low[i*ls+(v&(v-1))]^quadratic[i*total+__builtin_ctzll(v)];
        for (size_t v=1;v<hs;++v) high[i*hs+v]=high[i*hs+(v&(v-1))]^quadratic[i*total+split+__builtin_ctzll(v)];
    }
    uint64_t end=UINT64_C(1)<<pattern_count,pattern=(UINT64_C(1)<<negative[selected])-1,current=0;
    int status=0;
    do {
        for (uint64_t bits=current^pattern;bits;bits&=bits-1) {
            int j=__builtin_ctzll(bits);
            value^=linear[j]^low[j*ls+(current&(ls-1))]^high[j*hs+(current>>split)];
            current^=UINT64_C(1)<<j;
            for (int lane=0;lane<4;++lane) state[lane]^=changes[j][lane];
            for (int z=0;z<fprefix;++z) free_linear[z]^=quadratic[j*total+pattern_count+z];
            ++stats[7];
        }
        ++stats[0];int compatible=1;
        for (int j=0;j<checks;++j)
            if (__builtin_parityll(pattern&check_masks[j])!=check_rhs[j]) {compatible=0;break;}
        if (compatible) {
            ++stats[1];
            if (affine_pruning==1 && affine_pattern_contradiction(value,free_linear,selectors,fprefix,affine_masks,affine_count)) {
                stats[2]+=free_size;if (extra) ++extra[2];goto weight_prefix_next;
            }
            int pivot=-1,count_redundant=0,remaining[8],reduced=fprefix;
            uint32_t multi_basis[8],multi_delta[8];unsigned multi_targets[8];
            uint64_t multi_truth[8],affine_tail=valid;int multi_pivots[8],multi_count=0;
            if (affine_pruning==3) {
                affine_tail=affine_prefix_plan(value,free_linear,selectors,fprefix,tail,
                    affine_masks,affine_count,combinations,valid,multi_basis,multi_targets);
                if (!affine_tail) {stats[2]+=free_size;if (extra) ++extra[2];goto weight_prefix_next;}
                reduced=0;
                for (int j=0;j<fprefix;++j) {
                    if (multi_basis[j]) {
                        multi_pivots[multi_count++]=j;
                        multi_delta[j]=free_linear[j];
                        multi_truth[j]=combinations[multi_targets[j]&((1U<<tail)-1)]^
                            ((multi_targets[j]>>tail)?valid:0);
                    } else remaining[reduced++]=j;
                }
                stats[2]+=free_size-((UINT64_C(1)<<reduced)*members);
                if (extra && multi_count) {++extra[3];extra[4]+=(uint64_t)multi_count;}
            }
            if (count_affine && affine_pruning!=3) {
                for (int j=0;j<fprefix;++j) if (free_linear[j]&count_mask) {pivot=j;break;}
                if (pivot<0 && !selectors[d/2+other]) {
                    if (value&count_mask) {
                        stats[2]+=free_size;if (extra) ++extra[1];goto weight_prefix_next;
                    }
                    count_redundant=1;
                }
            }
            if (pivot>=0) {
                reduced=0;
                for (int j=0;j<fprefix;++j) if (j!=pivot) remaining[reduced++]=j;
                /* The excluded count half is accounted for by an exact
                 * affine equation. The other half is a bijective graph
                 * over remaining prefixes and ALL original tail bits. */
                stats[2]+=free_size/2;if (extra) ++extra[0];
            } else if (affine_pruning!=3) for (int j=0;j<fprefix;++j) remaining[j]=j;
            uint64_t local_state[4],gray=0;memcpy(local_state,state,sizeof(local_state));
            unsigned char *local_selectors=(unsigned char *)local_state;
            uint32_t local_value=value;
            uint32_t pivot_value=pivot<0?0:free_linear[pivot];
            unsigned char *pivot_selectors=pivot<0?NULL:(unsigned char *)free_changes[pivot];
            for (unsigned branch=0;branch<(1U<<reduced);++branch) {
            ++stats[5];uint64_t feasible=affine_tail;
            uint64_t pivot_truth=pivot<0?0:
                combinations[local_selectors[d/2+other]]^((local_value&count_mask)?valid:0);
            /* Exact selected-row weight already implies its count bit. */
            int skip_count=pivot>=0 || count_redundant;
            for (int step=0;step<d/2+!skip_count && feasible;++step) {
                int u=skip_count?step:step?step-1:d/2+other;
                uint64_t row=pure[u]^combinations[local_selectors[u]]^(((local_value>>u)&1)?valid:0);
                if (pivot>=0) row^=pivot_truth&
                    (combinations[pivot_selectors[u]]^(((pivot_value>>u)&1)?valid:0));
                if (affine_pruning==3) {
                    for (int i=0;i<multi_count;++i) {
                        int j=multi_pivots[i];
                        row^=multi_truth[j]&(combinations[((unsigned char *)free_changes[j])[u]]^
                            (((multi_delta[j]>>u)&1)?valid:0));
                        for (int k=i+1;k<multi_count;++k) {
                            int z=multi_pivots[k];
                            if (quadratic[(pattern_count+j)*total+pattern_count+z]>>u&1)
                                row^=multi_truth[j]&multi_truth[z];
                        }
                    }
                }
                feasible&=~row;
            }
            /* The symbolic state suffices for rejected prefixes. Build
             * physical signs only after the complete mod intersection. */
            uint64_t initial[2]={base[0],base[1]};
            if (feasible) for (uint64_t bits=pattern;bits;bits&=bits-1) {
                int j=__builtin_ctzll(bits);initial[0]^=pattern_toggles[2*j];initial[1]^=pattern_toggles[2*j+1];
            }
            if (feasible) for (uint64_t bits=gray;bits;bits&=bits-1) {
                int j=__builtin_ctzll(bits);initial[0]^=free_toggles[2*j];initial[1]^=free_toggles[2*j+1];
            }
            while (feasible) {
                unsigned f=(unsigned)__builtin_ctzll(feasible);feasible&=feasible-1;++stats[6];
                uint64_t original_prefix=gray;
                if (pivot>=0 && (pivot_truth>>f&1)) original_prefix|=UINT64_C(1)<<pivot;
                for (int i=0;i<multi_count;++i) {
                    int j=multi_pivots[i];
                    if (multi_truth[j]>>f&1) original_prefix|=UINT64_C(1)<<j;
                }
                #ifdef PQCP_WEIGHT_PREFIX_AUDIT_SYNDROME
                uint64_t audit_signs[2]={initial[0],initial[1]};
                for (uint64_t bits=original_prefix^gray;bits;bits&=bits-1) {
                    int j=__builtin_ctzll(bits);
                    audit_signs[0]^=free_toggles[2*j];audit_signs[1]^=free_toggles[2*j+1];
                }
                for (unsigned bits=f;bits;bits&=bits-1) {
                    int j=fprefix+__builtin_ctz(bits);audit_signs[0]^=free_toggles[2*j];audit_signs[1]^=free_toggles[2*j+1];
                }
                if (syndrome(audit_signs,active,required,negative,d,all)) {free(low);free(high);return -3;}
                #endif
                if (models) {
                    if (*count<capacity) models[*count]=(uint32_t)(pattern|((original_prefix|((uint64_t)f<<fprefix))<<pattern_count));
                    ++*count;continue;
                }
                uint64_t signs[2]={initial[0],initial[1]};
                for (uint64_t bits=original_prefix^gray;bits;bits&=bits-1) {
                    int j=__builtin_ctzll(bits);
                    signs[0]^=free_toggles[2*j];signs[1]^=free_toggles[2*j+1];
                }
                for (unsigned bits=f;bits;bits&=bits-1) {
                    int j=fprefix+__builtin_ctz(bits);signs[0]^=free_toggles[2*j];signs[1]^=free_toggles[2*j+1];
                }
                if (__builtin_popcountll(signs[other])!=negative[other]) continue;
                ++stats[3];int okay=1;
                for (int u=1;u<=d/2;++u) {
                    int cuts=0;
                    for (int s=0;s<2;++s) cuts+=__builtin_popcountll((signs[s]^rotate(signs[s],u,d,all))&active[s][u-1]);
                    ++stats[4];if (cuts!=required[u-1]) {okay=0;break;}
                }
                if (okay) {output[0]=signs[0];output[1]=signs[1];stats[2]+=(uint64_t)f+1;status=1;goto weight_prefix_finished;}
            }
            stats[2]+=members;
            if (branch+1<(1U<<reduced)) {
                int j=remaining[__builtin_ctz(branch+1)];
                local_value^=free_linear[j]^free_lookup[j][gray];gray^=UINT64_C(1)<<j;
                for (int lane=0;lane<4;++lane) local_state[lane]^=free_changes[j][lane];
                if (pivot>=0) pivot_value^=quadratic[(pattern_count+pivot)*total+pattern_count+j];
                for (int i=0;i<multi_count;++i) {
                    int z=multi_pivots[i];multi_delta[z]^=quadratic[(pattern_count+z)*total+pattern_count+j];
                    if (multi_basis[z]>>j&1) multi_truth[z]^=valid;
                }
            }
            }
        } else ++stats[8];
weight_prefix_next:
        if (!pattern) break;
        uint64_t bit=pattern&-pattern,raised=pattern+bit;
        pattern=raised|(((raised^pattern)>>2)/bit);
    } while (pattern<end);
weight_prefix_finished:
    free(low);free(high);return status;
}

int pqcp_weight_prefix_signs(const uint64_t *support,const uint64_t *base,
    const uint64_t *pattern_toggles,int pattern_count,int selected,
    const uint64_t *check_masks,const int *check_rhs,int checks,
    const uint64_t *free_toggles,int free_count,int d,const int *wanted,const int *negative,
    uint64_t budget,uint64_t *output,uint64_t *stats) {
    return weight_prefix_core(support,base,pattern_toggles,pattern_count,selected,check_masks,check_rhs,checks,
        free_toggles,free_count,d,wanted,negative,budget,output,stats,NULL,0,NULL,0,0,NULL);
}

/* Diagnostic models use p in the low pattern_count bits, then free f.
 * Capacity affects storage only. The modular survivor count stays exact. */
int pqcp_weight_prefix_models(const uint64_t *support,const uint64_t *base,
    const uint64_t *pattern_toggles,int pattern_count,int selected,
    const uint64_t *check_masks,const int *check_rhs,int checks,
    const uint64_t *free_toggles,int free_count,int d,const int *wanted,const int *negative,
    uint64_t capacity,uint32_t *models,uint64_t *count) {
    uint64_t output[2],stats[9];
    return weight_prefix_core(support,base,pattern_toggles,pattern_count,selected,check_masks,check_rhs,checks,
        free_toggles,free_count,d,wanted,negative,UINT64_MAX,output,stats,models,capacity,count,0,0,NULL);
}

int pqcp_weight_prefix_count_signs(const uint64_t *support,const uint64_t *base,
    const uint64_t *pattern_toggles,int pattern_count,int selected,
    const uint64_t *check_masks,const int *check_rhs,int checks,
    const uint64_t *free_toggles,int free_count,int d,const int *wanted,const int *negative,
    uint64_t budget,uint64_t *output,uint64_t *stats,uint64_t *extra) {
    return weight_prefix_core(support,base,pattern_toggles,pattern_count,selected,check_masks,check_rhs,checks,
        free_toggles,free_count,d,wanted,negative,budget,output,stats,NULL,0,NULL,1,0,extra);
}

int pqcp_weight_prefix_count_models(const uint64_t *support,const uint64_t *base,
    const uint64_t *pattern_toggles,int pattern_count,int selected,
    const uint64_t *check_masks,const int *check_rhs,int checks,
    const uint64_t *free_toggles,int free_count,int d,const int *wanted,const int *negative,
    uint64_t capacity,uint32_t *models,uint64_t *count,uint64_t *extra) {
    uint64_t output[2],stats[9];
    return weight_prefix_core(support,base,pattern_toggles,pattern_count,selected,check_masks,check_rhs,checks,
        free_toggles,free_count,d,wanted,negative,UINT64_MAX,output,stats,models,capacity,count,1,0,extra);
}

int pqcp_weight_prefix_affine_signs(const uint64_t *support,const uint64_t *base,
    const uint64_t *pattern_toggles,int pattern_count,int selected,
    const uint64_t *check_masks,const int *check_rhs,int checks,
    const uint64_t *free_toggles,int free_count,int d,const int *wanted,const int *negative,
    uint64_t budget,uint64_t *output,uint64_t *stats,uint64_t *extra) {
    return weight_prefix_core(support,base,pattern_toggles,pattern_count,selected,check_masks,check_rhs,checks,
        free_toggles,free_count,d,wanted,negative,budget,output,stats,NULL,0,NULL,1,1,extra);
}

int pqcp_weight_prefix_affine_models(const uint64_t *support,const uint64_t *base,
    const uint64_t *pattern_toggles,int pattern_count,int selected,
    const uint64_t *check_masks,const int *check_rhs,int checks,
    const uint64_t *free_toggles,int free_count,int d,const int *wanted,const int *negative,
    uint64_t capacity,uint32_t *models,uint64_t *count,uint64_t *extra) {
    uint64_t output[2],stats[9];
    return weight_prefix_core(support,base,pattern_toggles,pattern_count,selected,check_masks,check_rhs,checks,
        free_toggles,free_count,d,wanted,negative,UINT64_MAX,output,stats,models,capacity,count,1,1,extra);
}

int pqcp_weight_prefix_rebased_signs(const uint64_t *support,const uint64_t *base,
    const uint64_t *pattern_toggles,int pattern_count,int selected,
    const uint64_t *check_masks,const int *check_rhs,int checks,
    const uint64_t *free_toggles,int free_count,int d,const int *wanted,const int *negative,
    uint64_t budget,uint64_t *output,uint64_t *stats,uint64_t *extra) {
    return weight_prefix_core(support,base,pattern_toggles,pattern_count,selected,check_masks,check_rhs,checks,
        free_toggles,free_count,d,wanted,negative,budget,output,stats,NULL,0,NULL,1,2,extra);
}

int pqcp_weight_prefix_rebased_models(const uint64_t *support,const uint64_t *base,
    const uint64_t *pattern_toggles,int pattern_count,int selected,
    const uint64_t *check_masks,const int *check_rhs,int checks,
    const uint64_t *free_toggles,int free_count,int d,const int *wanted,const int *negative,
    uint64_t capacity,uint32_t *models,uint64_t *count,uint64_t *extra) {
    uint64_t output[2],stats[9];
    return weight_prefix_core(support,base,pattern_toggles,pattern_count,selected,check_masks,check_rhs,checks,
        free_toggles,free_count,d,wanted,negative,UINT64_MAX,output,stats,models,capacity,count,1,2,extra);
}

int pqcp_weight_prefix_multi_signs(const uint64_t *support,const uint64_t *base,
    const uint64_t *pattern_toggles,int pattern_count,int selected,
    const uint64_t *check_masks,const int *check_rhs,int checks,
    const uint64_t *free_toggles,int free_count,int d,const int *wanted,const int *negative,
    uint64_t budget,uint64_t *output,uint64_t *stats,uint64_t *extra) {
    return weight_prefix_core(support,base,pattern_toggles,pattern_count,selected,check_masks,check_rhs,checks,
        free_toggles,free_count,d,wanted,negative,budget,output,stats,NULL,0,NULL,1,3,extra);
}

int pqcp_weight_prefix_multi_models(const uint64_t *support,const uint64_t *base,
    const uint64_t *pattern_toggles,int pattern_count,int selected,
    const uint64_t *check_masks,const int *check_rhs,int checks,
    const uint64_t *free_toggles,int free_count,int d,const int *wanted,const int *negative,
    uint64_t capacity,uint32_t *models,uint64_t *count,uint64_t *extra) {
    uint64_t output[2],stats[9];
    return weight_prefix_core(support,base,pattern_toggles,pattern_count,selected,check_masks,check_rhs,checks,
        free_toggles,free_count,d,wanted,negative,UINT64_MAX,output,stats,models,capacity,count,1,3,extra);
}

/* The second count bit is e2 of the affine sign bits. Its polar coefficient
 * is parity(T_i)*parity(T_j) XOR parity(T_i AND T_j). If ALL coefficients
 * vanish, this necessary count equation is AFFINE and can be eliminated
 * exactly before norm-coefficient construction. No single-norm genericity
 * or invertibility assumption is required: every coefficient is checked.
 * Partial budgets keep the old enumeration. Stats cover ORIGINAL space. */
int pqcp_quadratic_count_hybrid_signs(const uint64_t *support,const uint64_t *initial,
    const uint64_t *toggles,int q,int d,const int *wanted,const int *negative,
    int tail,uint64_t budget,uint64_t *output,uint64_t *stats) {
    if (q<0 || q>24 || d<3 || d>47 || !(d&1) || tail<1 || tail>8 || !budget) return -1;
    if (budget<(UINT64_C(1)<<q))
        return pqcp_quadratic_hybrid_signs(support,initial,toggles,q,d,wanted,negative,tail,budget,output,stats);
    uint64_t all=(UINT64_C(1)<<d)-1,base[2]={initial[0],initial[1]},space[48];
    for (int side=0;side<2;++side) {
        if ((support[side]&~all) || (base[side]&~support[side]) || negative[side]<0 ||
            negative[side]>__builtin_popcountll(support[side])) return -1;
        for (int j=0;j<q;++j) if (toggles[2*j+side]&~support[side]) return -1;
    }
    memcpy(space,toggles,(size_t)2*q*sizeof(uint64_t));
    int dimension=q;
    for (int side=0;side<2;++side) {
        int affine=1;
        for (int i=0;i<dimension && affine;++i) for (int j=i+1;j<dimension;++j) {
            int polar=(__builtin_parityll(space[2*i+side])*__builtin_parityll(space[2*j+side]))^
                       __builtin_parityll(space[2*i+side]&space[2*j+side]);
            if (polar) {affine=0;break;}
        }
        if (!affine) continue;
        int at_zero=(__builtin_popcountll(base[side])>>1)&1;
        int rhs=at_zero^((negative[side]>>1)&1);uint32_t coefficients=0;
        for (int j=0;j<dimension;++j)
            if (((__builtin_popcountll(base[side]^space[2*j+side])>>1)&1)^at_zero)
                coefficients|=UINT32_C(1)<<j;
        if (!coefficients) {
            if (rhs) {memset(stats,0,7*sizeof(uint64_t));stats[0]=UINT64_C(1)<<q;return 0;}
            continue;
        }
        int pivot=__builtin_ctz(coefficients),used=0;
        uint64_t direction[2]={space[2*pivot],space[2*pivot+1]};
        if (rhs) {base[0]^=direction[0];base[1]^=direction[1];}
        for (int j=0;j<dimension;++j) if (j!=pivot) {
            int include=(coefficients>>j)&1;
            space[2*used]=space[2*j]^(include?direction[0]:0);
            space[2*used+1]=space[2*j+1]^(include?direction[1]:0);
            ++used;
        }
        dimension=used;
    }
    int original_tail=tail<q?tail:q;
    int reduced_tail=original_tail-(q-dimension);
    /* One uint64 represents all <=64 tails at equal intersection cost.
     * After an exact count hyperplane removes a dimension, retain the
     * requested bit-tail width and halve the number of prefixes instead.
     * The full Cartesian-product coverage proof is independent of split.
     * Keep the previous split for the projected Gaussian/orbit paths. */
    if (d==43 && original_tail<=6) reduced_tail=original_tail;
    if (reduced_tail<1) reduced_tail=1;
    int status=pqcp_quadratic_hybrid_signs(support,base,space,dimension,d,wanted,negative,
                                         reduced_tail,budget,output,stats);
    if (status>=0) stats[0]+=(UINT64_C(1)<<q)-(UINT64_C(1)<<dimension);
    return status;
}

static uint64_t compact_sign_pair(const uint64_t *pair, const uint64_t *support) {
    uint64_t packed = 0;
    int at = 0;
    for (int s = 0; s < 2; ++s) {
        uint64_t bits = support[s];
        while (bits) {
            uint64_t bit = bits & (UINT64_C(0)-bits);
            if (pair[s] & bit) packed |= UINT64_C(1) << at;
            ++at; bits ^= bit;
        }
    }
    return packed;
}

/* Quotient an even-weight row's whole-sign reversal. It lies in the raw
 * mod4/count-parity kernel. Preserve both orientations through the exact
 * count set {n,w-n}, then orient a witness back to the requested n.
 * A partial original budget keeps the old enumeration byte-for-byte. */
int pqcp_quadratic_orbit_hybrid_signs(const uint64_t *support, const uint64_t *initial,
    const uint64_t *toggles, int q, int d, const int *wanted, const int *negative,
    int tail, uint64_t budget, uint64_t *output, uint64_t *stats) {
    memset(stats,0,9*sizeof(uint64_t));
    if (q < 0 || q > 24 || d < 3 || d > 47 || !(d&1) || !budget) return -1;
    uint64_t all = (UINT64_C(1)<<d)-1;
    for (int s = 0; s < 2; ++s) {
        if ((support[s]&~all) || (initial[s]&~support[s]) || negative[s]<0 ||
            negative[s]>__builtin_popcountll(support[s])) return -1;
        for (int j = 0; j < q; ++j) if (toggles[2*j+s]&~support[s]) return -1;
    }
    if (!q || budget < (UINT64_C(1)<<q) ||
        __builtin_popcountll(support[0])+__builtin_popcountll(support[1]) > 63)
        return pqcp_quadratic_hybrid_signs(support,initial,toggles,q,d,wanted,negative,tail,budget,output,stats);
    uint64_t basis[64] = {0};
    uint32_t labels[64] = {0};
    for (int j = 0; j < q; ++j) {
        uint64_t row = compact_sign_pair(toggles+2*j,support);
        uint32_t label = UINT32_C(1)<<j;
        while (row) {
            int p = 63-__builtin_clzll(row);
            if (!basis[p]) { basis[p] = row; labels[p] = label; break; }
            row ^= basis[p]; label ^= labels[p];
        }
    }
    for (int side = 0; side < 2; ++side) {
        int weight = __builtin_popcountll(support[side]);
        if (!weight || (weight&1)) continue;
        uint64_t reversal[2] = {0,0}; reversal[side] = support[side];
        uint64_t row = compact_sign_pair(reversal,support);
        uint32_t representation = 0;
        while (row) {
            int p = 63-__builtin_clzll(row);
            if (!basis[p]) break;
            row ^= basis[p]; representation ^= labels[p];
        }
        if (row || !representation) continue; /* Arbitrary/conditioned subspaces may lack this direction. */
        int removed = __builtin_ctz(representation), used = 0;
        uint64_t reduced[48];
        for (int j = 0; j < q; ++j) if (j != removed) {
            reduced[2*used] = toggles[2*j]; reduced[2*used+1] = toggles[2*j+1]; ++used;
        }
        int original_tail = tail < q ? tail : q;
        int reduced_tail = original_tail > 1 ? original_tail-1 : 1;
        /* Keep the original prefix depth. If reversal was in the tail,
         * this preserves its quadratic span. Otherwise the retained tail
         * has a SUBSPACE of the old span, strengthening the necessary
         * linear projection without adding a prefix dimension. */
        if (reduced_tail < 1) reduced_tail = 1;
        int status = quadratic_hybrid_core(support,initial,reduced,q-1,d,wanted,negative,reduced_tail,budget,output,stats,side);
        stats[7] = (uint64_t)side+1; stats[8] = (uint64_t)(reduced_tail < q-1 ? reduced_tail : q-1);
        return status;
    }
    return pqcp_quadratic_hybrid_signs(support,initial,toggles,q,d,wanted,negative,tail,budget,output,stats);
}

extern int pqcp_weight_signs(const uint64_t *, const uint64_t *, const uint64_t *, int, int,
    const uint64_t *, const int *, int, const uint64_t *, int, int, const int *, const int *,
    uint64_t, uint64_t *, uint64_t *);
extern uint64_t pqcp_weight_pattern_count(int, int, const uint64_t *, const int *, int);

/* The same complete constant-weight partition, with exact quotient solving
 * inside every compatible fibre. A partial global budget keeps the original
 * kernel/ordering. Stats 0..4 retain the weight kernel meanings; 5..8 add
 * prefix branches, projected candidates, mod8 hits, maximum tail nullity. */
int pqcp_weight_quadratic_signs(const uint64_t *support, const uint64_t *base,
    const uint64_t *pattern_toggles, int pattern_count, int selected,
    const uint64_t *check_masks, const int *check_rhs, int checks,
    const uint64_t *free_toggles, int free_count, int d,
    const int *wanted, const int *negative, int tail, uint64_t budget,
    uint64_t *output, uint64_t *stats) {
    if (d < 3 || d > 47 || !(d & 1) || selected < 0 || selected > 1 ||
        pattern_count < 0 || pattern_count > 47 || free_count < 0 || free_count > 24 ||
        checks < 0 || checks > 96 || tail < 1 || tail > 8 || !budget ||
        negative[selected] < 0 || negative[selected] > pattern_count) return -1;
    memset(stats, 0, 9*sizeof(uint64_t));
    uint64_t patterns_upper = 1;
    int minority = negative[selected];
    if (pattern_count-minority < minority) minority = pattern_count-minority;
    for (int i = 1; i <= minority; ++i)
        patterns_upper = patterns_upper*(uint64_t)(pattern_count-i+1)/(uint64_t)i;
    if (patterns_upper > 4096)
        return pqcp_weight_signs(support, base, pattern_toggles, pattern_count, selected,
            check_masks, check_rhs, checks, free_toggles, free_count, d, wanted, negative, budget, output, stats);
    uint64_t combinations = pqcp_weight_pattern_count(pattern_count, negative[selected], check_masks, check_rhs, checks);
    uint64_t branch = UINT64_C(1) << free_count;
    if (combinations == UINT64_MAX) return -1;
    if (!combinations) return 0;
    if (combinations > budget/branch)
        return pqcp_weight_signs(support, base, pattern_toggles, pattern_count, selected,
            check_masks, check_rhs, checks, free_toggles, free_count, d, wanted, negative, budget, output, stats);
    uint64_t all = (UINT64_C(1) << d)-1, active[2][23];
    int required[23];
    for (int s = 0; s < 2; ++s) {
        if ((support[s] & ~all) || (base[s] & ~support[s]) || negative[s] < 0 ||
            negative[s] > __builtin_popcountll(support[s])) return -1;
        for (int u = 1; u <= d/2; ++u)
            active[s][u-1] = support[s] & rotate(support[s], u, d, all);
    }
    for (int i = 0; i < free_count; ++i)
        if (free_toggles[2*i+selected]) return -1;
    for (int u = 1; u <= d/2; ++u) {
        int m = __builtin_popcountll(active[0][u-1])+__builtin_popcountll(active[1][u-1]);
        if (m < wanted[u] || m < -wanted[u] || (m-wanted[u])%2) return -1;
        required[u-1] = (m-wanted[u])/2;
    }
    /* A large pattern bank shares ONE full quadratic parameterization.
     * Coefficients in f_i*f_j are invariant under the selected pattern.
     * Specialize the constant/mixed-linear terms exactly for every p.
     * Small banks retain affine count elimination and its reduced basis. */
    int total=pattern_count+free_count;
    int shared=tail<=6 && combinations>64 && total<=47;
    uint32_t shared_constant=0,shared_linear[47],shared_quadratic[47*47],prepared[601];
    if (shared) {
        uint64_t full_toggles[94];
        memcpy(full_toggles,pattern_toggles,(size_t)2*pattern_count*sizeof(uint64_t));
        memcpy(full_toggles+2*pattern_count,free_toggles,(size_t)2*free_count*sizeof(uint64_t));
        if (make_quadratic_plan(support,base,full_toggles,total,d,wanted,negative,
                               &shared_constant,shared_linear,shared_quadratic,47)) return -1;
    }
    uint64_t end = UINT64_C(1) << pattern_count;
    uint64_t pattern = (UINT64_C(1) << negative[selected])-1;
    do {
        ++stats[0];
        int compatible = 1;
        for (int j = 0; j < checks; ++j)
            if (__builtin_parityll(pattern & check_masks[j]) != check_rhs[j]) { compatible = 0; break; }
        if (compatible) {
            ++stats[1];
            uint64_t initial[2] = {base[0], base[1]}, bits = pattern;
            while (bits) {
                int bit = __builtin_ctzll(bits);
                initial[0] ^= pattern_toggles[2*bit]; initial[1] ^= pattern_toggles[2*bit+1];
                bits &= bits-1;
            }
            if (__builtin_popcountll(initial[selected]) != negative[selected]) return -1;
            int other = 1-selected, possible = 1, rare = negative[other];
            int occupied = __builtin_popcountll(support[other]);
            if (occupied-rare < rare) rare = occupied-rare;
            for (int u = 1; u <= d/2; ++u) {
                int fixed = __builtin_popcountll((initial[selected]^rotate(initial[selected], u, d, all)) & active[selected][u-1]);
                int maximum = __builtin_popcountll(active[other][u-1]), needed = required[u-1]-fixed;
                if (2*rare < maximum) maximum = 2*rare;
                if (needed < 0 || needed > maximum) { possible = 0; break; }
            }
            if (possible) {
                uint64_t part[7];
                int status;
                if (shared) {
                    if (pqcp_specialize_quadratic_plan(shared_constant,shared_linear,shared_quadratic,
                                                      pattern_count,free_count,pattern,prepared)) return -1;
                    status=quadratic_bit_tail_core(support,initial,free_toggles,free_count,d,wanted,negative,
                                                  tail,branch,output,part,NULL,0,NULL,prepared);
                } else status=pqcp_quadratic_count_hybrid_signs(support, initial, free_toggles, free_count, d,
                                                               wanted, negative, tail, branch, output, part);
                if (status < 0 || status == 2) return -1;
                stats[2] += part[0]; stats[3] += part[2]; stats[4] += part[3];
                stats[5] += part[4]; stats[6] += part[5]; stats[7] += part[1];
                if (part[6] > stats[8]) stats[8] = part[6];
                if (status == 1) return 1;
            }
        }
        if (!pattern) break;
        uint64_t low = pattern & -pattern, raised = pattern+low;
        pattern = raised | (((raised ^ pattern) >> 2)/low);
    } while (pattern < end);
    return 0;
}
