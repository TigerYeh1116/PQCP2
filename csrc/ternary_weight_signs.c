/* Exhaustive constant-weight conditioning, then exact Gray-code signs.
 * Compatible projected patterns partition all cardinality-feasible signs.
 * 0=exhausted, 1=integer witness, 2=assignment budget, -1=invalid encoding. */
#include <stdint.h>
#include <string.h>

static uint64_t rotate(uint64_t x, int u, int d, uint64_t all) {
    return ((x << u) | (x >> (d-u))) & all;
}

uint64_t pqcp_weight_pattern_count(int variables, int negative, const uint64_t *checks,
                                  const int *rhs, int count) {
    if (variables < 0 || variables > 47 || negative < 0 || negative > variables || count < 0 || count > 96)
        return UINT64_MAX;
    int minority = negative;
    if (variables-minority < minority) minority = variables-minority;
    uint64_t combinations = 1;
    for (int i = 1; i <= minority; ++i)
        combinations = combinations*(uint64_t)(variables-i+1)/(uint64_t)i;
    if (combinations > 4096) return UINT64_MAX; /* Explicitly bounded preprocessing. */
    uint64_t end = UINT64_C(1) << variables, pattern = (UINT64_C(1) << negative)-1, compatible = 0;
    do {
        int okay = 1;
        for (int j = 0; j < count; ++j)
            if (__builtin_parityll(pattern & checks[j]) != rhs[j]) { okay = 0; break; }
        compatible += (uint64_t)okay;
        if (!pattern) break;
        uint64_t low = pattern & -pattern, raised = pattern+low;
        pattern = raised | (((raised ^ pattern) >> 2)/low);
    } while (pattern < end);
    return compatible;
}

int pqcp_weight_signs(const uint64_t *support, const uint64_t *base,
    const uint64_t *pattern_toggles, int pattern_count, int selected,
    const uint64_t *check_masks, const int *check_rhs, int checks,
    const uint64_t *free_toggles, int free_count, int d,
    const int *wanted, const int *negative, uint64_t budget,
    uint64_t *output, uint64_t *stats)
{
    if (d < 3 || d > 47 || !(d & 1) || selected < 0 || selected > 1 ||
        pattern_count < 0 || pattern_count > 47 || free_count < 0 || free_count > 47 ||
        negative[selected] < 0 || negative[selected] > pattern_count || !budget) return -1;
    uint64_t all = (UINT64_C(1) << d)-1;
    uint64_t active[2][23];
    int required[23];
    memset(stats, 0, 5*sizeof(uint64_t));
    for (int s = 0; s < 2; ++s) {
        if ((support[s] & ~all) || (base[s] & ~support[s])) return -1;
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
    uint64_t patterns_end = UINT64_C(1) << pattern_count;
    uint64_t pattern = (UINT64_C(1) << negative[selected])-1;
    uint64_t branch_size = UINT64_C(1) << free_count;
    do {
        ++stats[0];
        int compatible = 1;
        for (int j = 0; j < checks; ++j)
            if (__builtin_parityll(pattern & check_masks[j]) != check_rhs[j]) {
                compatible = 0; break;
            }
        if (compatible) {
            ++stats[1];
            uint64_t signs[2] = {base[0], base[1]}, bits = pattern;
            while (bits) {
                int bit = __builtin_ctzll(bits);
                signs[0] ^= pattern_toggles[2*bit];
                signs[1] ^= pattern_toggles[2*bit+1];
                bits &= bits-1;
            }
            if (__builtin_popcountll(signs[selected]) != negative[selected]) return -1;
            /* The selected row is constant on the entire affine fibre.
             * Subtract its exact cuts once; only the other row changes. */
            int other = 1-selected, needed[23], possible = 1;
            int minority = negative[other];
            int occupied = __builtin_popcountll(support[other]);
            if (occupied-minority < minority) minority = occupied-minority;
            for (int u = 1; u <= d/2; ++u) {
                int fixed_cuts = __builtin_popcountll((signs[selected] ^ rotate(signs[selected], u, d, all)) & active[selected][u-1]);
                int maximum = __builtin_popcountll(active[other][u-1]);
                if (2*minority < maximum) maximum = 2*minority;
                needed[u-1] = required[u-1]-fixed_cuts;
                if (needed[u-1] < 0 || needed[u-1] > maximum) { possible = 0; break; }
            }
            if (!possible) goto next_pattern;
            for (uint64_t i = 0; i < branch_size; ++i) {
                if (stats[2] == budget) return 2;
                ++stats[2];
                if (__builtin_popcountll(signs[1-selected]) == negative[1-selected]) {
                    ++stats[3];
                    int okay = 1;
                    for (int u = 1; u <= d/2; ++u) {
                        int n = __builtin_popcountll((signs[other] ^ rotate(signs[other], u, d, all)) & active[other][u-1]);
                        ++stats[4];
                        if (n != needed[u-1]) { okay = 0; break; }
                    }
                    if (okay) { output[0] = signs[0]; output[1] = signs[1]; return 1; }
                }
                if (i+1 < branch_size) {
                    int bit = __builtin_ctzll(i+1);
                    signs[0] ^= free_toggles[2*bit];
                    signs[1] ^= free_toggles[2*bit+1];
                }
            }
        }
next_pattern:
        if (!pattern) break; /* Unique zero-weight pattern. */
        /* Gosper: next subset of the same cardinality, no floating point. */
        uint64_t low = pattern & -pattern, raised = pattern+low;
        pattern = raised | (((raised ^ pattern) >> 2)/low);
    } while (pattern < patterns_end);
    return 0;
}
