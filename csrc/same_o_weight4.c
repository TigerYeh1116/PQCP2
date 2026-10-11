/* Complete affine C enumeration, rational PSD and anchored norm/trace CRT.
   Every arithmetic decision is integral. This is a restricted family only. */
#include <stdint.h>
#include <stddef.h>

#define D 43
#define HALF 21
#define FIELD_SIZE 16384
#define ALL_BITS ((UINT64_C(1) << D)-1)

static uint64_t rotate(uint64_t mask, int u)
{
    return u ? ((mask << u) | (mask >> (D-u))) & ALL_BITS : mask;
}

static int count(uint64_t mask) { return __builtin_popcountll(mask); }

static int odd_match(uint64_t mask, uint64_t plus, uint64_t minus)
{
    for (int u = 0; u <= HALF; ++u) {
        uint64_t a = rotate(mask,u), b = rotate(mask,(D-u)%D);
        int cross = count(plus&a)+count(plus&b)-count(minus&a)-count(minus&b);
        if (cross != 2-(u==1)) return 0;
    }
    return 1;
}

int pqcp_complete_same_o_weight4(uint64_t base, const uint64_t *basis, int nullity,
    uint64_t plus, uint64_t minus, int negative_weight, const uint32_t *residue_columns,
    const uint64_t *root_lifts, uint64_t scalar_base, const int64_t *psd_coefficients,
    uint64_t *output, uint64_t capacity, uint64_t *stats,
    uint64_t *seed_output, uint64_t seed_capacity, uint64_t *seed_stats)
{
    if (nullity < 0 || nullity > 30 || !capacity || !seed_capacity ||
        (base|plus|minus|scalar_base) > ALL_BITS || plus&minus ||
        count(plus)-count(minus)!=2 || (count(plus|minus)!=2 && count(plus|minus)!=4) ||
        negative_weight!=17-count(plus|minus)/2 || base&(plus|minus)) return -1;
    for (int i=0;i<nullity;++i)
        if (basis[i] > ALL_BITS || basis[i]&(plus|minus)) return -1;
    for (int i=0;i<11;++i) stats[i]=0;
    seed_stats[0]=seed_stats[1]=0;
    uint64_t negative=base,total=UINT64_C(1)<<nullity;
    for (uint64_t step=0;step<total;++step) {
        if (step) negative ^= basis[__builtin_ctzll(step)];
        ++stats[0];
        if (count(negative)!=negative_weight) continue;
        ++stats[1];
        uint64_t ea=negative|minus,eb=negative|plus,norm=1;
        int wanted[HALF+1],invalid=0,sum=0;wanted[0]=21;
        for (int u=1;u<=HALF;++u) {
            int doubled=33-count(ea&rotate(ea,u))-count(eb&rotate(eb,u));
            if (doubled%2) return -2;
            wanted[u]=doubled/2;sum+=wanted[u];
            if (wanted[u]<0 || wanted[u]>20) invalid=1;
            if (wanted[u]%2) norm |= (UINT64_C(1)<<u)|(UINT64_C(1)<<(D-u));
        }
        if (sum!=210) return -3; /* N(0)+2*sum=N(0)^2 */
        if (invalid) { ++stats[2];continue; }
        for (int j=0;j<HALF;++j) {
            int64_t upper=INT64_C(21)*(INT64_C(1)<<40);
            for (int u=1;u<=HALF;++u)
                upper += INT64_C(2)*wanted[u]*psd_coefficients[j*HALF+u-1];
            if (upper<0) { invalid=1;break; }
        }
        if (invalid) { ++stats[3];continue; }
        uint64_t roots[3][2];
        for (int component=0;component<3;++component) {
            uint32_t residue=0;uint64_t bits=norm;
            while (bits) {
                int bit=__builtin_ctzll(bits);bits &= bits-1;
                residue ^= residue_columns[component*D+bit];
            }
            if (residue>=FIELD_SIZE) return -4;
            for (int r=0;r<2;++r)
                roots[component][r]=root_lifts[(component*2+r)*FIELD_SIZE+residue];
            if (roots[component][0]==UINT64_MAX) { invalid=1;break; }
            if (roots[component][1]==UINT64_MAX) return -5;
        }
        if (invalid) { ++stats[4];continue; }
        for (int i=0;i<2;++i) for (int j=0;j<2;++j) for (int k=0;k<2;++k) {
            uint64_t mask=scalar_base^roots[0][i]^roots[1][j]^roots[2][k];
            ++stats[5];
            if (count(mask)!=21) { ++stats[6];continue; }
            if (!odd_match(mask,plus,minus)) { ++stats[7];continue; }
            uint64_t deviation=0;
            for (int u=1;u<=HALF;++u) {
                int difference=count(mask&rotate(mask,u))-wanted[u];
                deviation+=(uint64_t)(difference*difference);
            }
            ++seed_stats[0];
            if (seed_stats[1]<seed_capacity) {
                uint64_t offset=3*seed_stats[1];
                seed_output[offset]=negative;seed_output[offset+1]=mask;
                seed_output[offset+2]=deviation;++seed_stats[1];
            }
            if (deviation) { ++stats[8];continue; }
            ++stats[9];
            if (stats[10]<capacity) {
                output[2*stats[10]]=negative;output[2*stats[10]+1]=mask;++stats[10];
            }
        }
    }
    return 0;
}
