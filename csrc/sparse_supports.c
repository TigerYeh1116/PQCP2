/* Exact binary bracelets and all independent periodic PACF parities. */
#include <stdint.h>

static uint64_t right_rotation(uint64_t mask, int shift, int d, uint64_t full) {
    return shift ? ((mask >> shift) | (mask << (d-shift))) & full : mask;
}

int pqcp_sparse_bracelets(int d, int weight, uint64_t capacity,
                         uint64_t *masks, uint32_t *signatures, uint64_t *stats) {
    if (d < 3 || d > 47 || !(d & 1) || weight < 1 || weight >= d || !capacity) return -1;
    uint64_t full = (UINT64_C(1) << d)-1, end = UINT64_C(1) << (d-1);
    uint64_t subset = (UINT64_C(1) << (weight-1))-1, candidates = 0, found = 0;
    do {
        uint64_t mask = (subset << 1) | 1, bits = mask, reverse = 0;
        int positions[47], count = 0, canonical = 1;
        ++candidates;
        while (bits) {
            int p = __builtin_ctzll(bits);
            positions[count++] = p;
            reverse |= UINT64_C(1) << ((d-p) % d);
            if (right_rotation(mask, p, d, full) < mask) { canonical = 0; break; }
            bits &= bits-1;
        }
        if (canonical) {
            bits = reverse;
            while (bits) {
                int p = __builtin_ctzll(bits);
                if (right_rotation(reverse, p, d, full) < mask) { canonical = 0; break; }
                bits &= bits-1;
            }
        }
        if (canonical) {
            uint32_t signature = 0;
            for (int i = 0; i < count; ++i) {
                for (int j = i+1; j < count; ++j) {
                    int distance = positions[j]-positions[i];
                    if (distance > d/2) distance = d-distance;
                    signature ^= UINT32_C(1) << (distance-1);
                }
            }
            if (found < capacity) { masks[found] = mask; signatures[found] = signature; }
            ++found;
        }
        if (!subset) break;
        uint64_t low = subset & -subset, raised = subset+low;
        subset = raised | (((raised ^ subset) >> 2)/low);
    } while (subset < end);
    stats[0] = candidates; stats[1] = found;
    return found > capacity ? 1 : 0;
}
