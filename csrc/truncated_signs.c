/* Low-energy truncated C-sign search (one reciprocal-paired factor pair, d = 23 or 47).
 *
 * pqcp_top_signs: every negative-sign subset of size nA on support SA; keep the M
 *   with the smallest sum_u R_A(u)^2 (u = 1..d/2), returning their PACF vectors.
 * pqcp_truncated_scan: the GF(2) family of C_B supports V with
 *   U U* + V V* = 1 + X^k + X^-k, one V per cyclic-rotation orbit (V_f = gamma^i,
 *   0 <= i < (2^m-1)/d, V_f* determined), weight wB; per V one GF(2) elimination of
 *   the single-row edge-cut system, then for each kept sign pattern of A the exact
 *   C_B completions with nB negatives and R_B = T - R_A. Every reported witness is an
 *   exact solution; patterns outside the kept M are not searched (no exclusion).
 * Masks: bit i = position i (d <= 47); a sign mask bit 1 = negative entry. */
#include <stdint.h>
#include <string.h>

static inline uint64_t rotl(uint64_t x, int u, int d, uint64_t all) {
    return ((x << u) | (x >> (d-u))) & all;
}
static inline uint64_t rotr(uint64_t x, int u, int d, uint64_t all) {
    return ((x >> u) | (x << (d-u))) & all;
}

int pqcp_top_signs(int d, uint64_t SA, int nA, int M, int *outR, uint64_t *outNeg, uint64_t *stats) {
    if (d < 3 || d > 47 || !(d & 1) || M < 1) return -1;
    int H = d/2, pos[47], w = 0;
    uint64_t all = (UINT64_C(1) << d) - 1, active[23];
    int I[24];
    for (int i = 0; i < d; i++) if (SA >> i & 1) pos[w++] = i;
    if (nA < 0 || nA > w) return -1;
    for (int u = 1; u <= H; u++) { active[u-1] = SA & rotl(SA, u, d, all); I[u] = __builtin_popcountll(active[u-1]); }
    int kept = 0, worst = 0;
    long long keyE[4096];
    if (M > 4096) return -1;
    uint64_t combo = nA ? ((UINT64_C(1) << nA) - 1) : 0, end = UINT64_C(1) << w, count = 0;
    while (1) {
        uint64_t x = 0;
        for (uint64_t c = combo; c; c &= c-1) x |= UINT64_C(1) << pos[__builtin_ctzll(c)];
        int r[23]; long long E = 0;
        for (int u = 1; u <= H; u++) {
            r[u-1] = I[u] - 2*__builtin_popcountll((x ^ rotl(x, u, d, all)) & active[u-1]);
            E += (long long)r[u-1]*r[u-1];
        }
        count++;
        int slot = -1;
        if (kept < M) slot = kept++;
        else if (E < keyE[worst]) slot = worst;   /* replace the worst kept pattern */
        if (slot >= 0) {
            keyE[slot] = E; outNeg[slot] = x; memcpy(outR + slot*H, r, H*sizeof(int));
            if (kept == M) { worst = 0; for (int j = 1; j < M; j++) if (keyE[j] > keyE[worst]) worst = j; }
        }
        if (!nA) break;
        uint64_t t = combo | (combo - 1);              /* Gosper: next nA-subset of w bits */
        uint64_t next = (t + 1) | (((~t & -~t) - 1) >> (__builtin_ctzll(combo) + 1));
        if (next >= end) break;
        combo = next;
    }
    stats[0] = count;
    return kept;
}

static uint32_t fmul(uint32_t a, uint32_t b, uint32_t f, int m) {
    uint64_t r = 0;
    for (int i = 0; i < m; i++) if (b >> i & 1) r ^= (uint64_t)a << i;
    for (int i = 2*m-2; i >= m; i--) if (r >> i & 1) r ^= (uint64_t)f << (i-m);
    return (uint32_t)r;
}
static uint64_t lin(const uint64_t *table, uint32_t v) {
    uint64_t out = 0;
    for (; v; v &= v-1) out ^= table[__builtin_ctz(v)];
    return out;
}

int pqcp_truncated_scan(int d, int wB, int nB, const int *T, int M, const int *RA,
                        int m, uint32_t f, uint32_t gamma, uint32_t gamma_inv, uint32_t rf, uint64_t reps,
                        uint64_t base, const uint64_t *LF, const uint64_t *LG,
                        uint64_t *outSB, uint32_t *outSig, uint64_t *outX, int maxOut, uint64_t *stats) {
    if (d < 3 || d > 47 || !(d & 1) || M < 1 || m < 2 || m > 31 || !rf) return -1;
    int H = d/2, found = 0;
    uint64_t all = (UINT64_C(1) << d) - 1;
    memset(stats, 0, 6*sizeof(uint64_t));
    if (M > 4096) return -1;
    uint64_t Q1[4096], Q0 = 0;             /* bit planes of R_A(u): bit1 per pattern, bit0 common */
    for (int s = 0; s < M; s++) {
        Q1[s] = 0;
        for (int u = 1; u <= H; u++) Q1[s] |= (uint64_t)((RA[s*H+u-1] >> 1) & 1) << (u-1);
    }
    for (int u = 1; u <= H; u++) Q0 |= (uint64_t)(RA[u-1] & 1) << (u-1);
    for (int s = 1; s < M; s++)
        for (int u = 1; u <= H; u++) if (((RA[s*H+u-1] ^ RA[u-1]) & 1)) return -2;   /* R_A parity is fixed by S_A */
    uint32_t v = 1, wv = rf;               /* V_f = gamma^i, rho(V_f*) = rf / V_f */
    for (uint64_t i = 0; i < reps; i++, v = fmul(v, gamma, f, m), wv = fmul(wv, gamma_inv, f, m)) {
        uint64_t SB = base ^ lin(LF, v) ^ lin(LG, wv);
        stats[0]++;
        if (__builtin_popcountll(SB) != wB) continue;
        stats[1]++;
        /* Single-row system on SB: rows u = 1..H (edge-cut parity), row H+1 = negative-count parity. */
        uint64_t act[23], row[24], comb[24], deps[24], contrib[24];
        int IB[24], piv[24], np = 0, nd = 0;
        for (int u = 1; u <= H; u++) {
            act[u-1] = SB & rotl(SB, u, d, all);
            IB[u] = __builtin_popcountll(act[u-1]);
        }
        for (int e = 0; e <= H; e++) {
            uint64_t r = e < H ? (act[e] ^ rotr(act[e], e+1, d, all)) : SB;
            uint64_t c = UINT64_C(1) << e;
            for (int p = 0; p < np; p++) if (r >> piv[p] & 1) { r ^= row[p]; c ^= comb[p]; }
            if (!r) { deps[nd++] = c; continue; }
            int col = __builtin_ctzll(r);
            for (int p = 0; p < np; p++) if (row[p] >> col & 1) { row[p] ^= r; comb[p] ^= c; }
            row[np] = r; comb[np] = c; piv[np] = col; np++;
        }
        uint64_t pivmask = 0;
        for (int p = 0; p < np; p++) pivmask |= UINT64_C(1) << piv[p];
        for (int e = 0; e <= H; e++) {        /* particular solution is linear in the rhs bits */
            contrib[e] = 0;
            for (int p = 0; p < np; p++) if (comb[p] >> e & 1) contrib[e] |= UINT64_C(1) << piv[p];
        }
        uint64_t freecols = SB & ~pivmask, kernel[47];
        int nk = 0;
        for (uint64_t fc = freecols; fc; fc &= fc-1) {
            int col = __builtin_ctzll(fc);
            uint64_t k = UINT64_C(1) << col;
            for (int p = 0; p < np; p++) if (row[p] >> col & 1) k |= UINT64_C(1) << piv[p];
            kernel[nk++] = k;
        }
        if (nk > 12) { stats[5]++; continue; }  /* never expected; would only be slower, so skip and count */
        /* rhs bit u-1 = (D(u) mod 2), D(u) = (IB(u) - T(u) + R_A(u))/2. With x = IB-T and y = R_A,
         * bit1(x+y) = bit1(x) ^ bit1(y) ^ (bit0(x) & bit0(y)); the GF(2) family forces bit0(x) == bit0(y). */
        uint64_t P1 = 0, P0 = 0;
        int base2[24];
        for (int u = 1; u <= H; u++) {
            base2[u] = IB[u] - T[u-1];
            P0 |= (uint64_t)(base2[u] & 1) << (u-1);
            P1 |= (uint64_t)((base2[u] >> 1) & 1) << (u-1);
        }
        if (P0 != Q0) continue;          /* parity mismatch: no pattern can complete (excluded by the family) */
        uint64_t fixed = P1 ^ (P0 & Q0) ^ ((uint64_t)(nB & 1) << H);
        uint64_t table[3][256];          /* x0 is linear in the rhs bits */
        for (int byte = 0; byte < 3; byte++) {
            table[byte][0] = 0;
            for (int v = 1; v < 256; v++) {
                int bit = byte*8 + __builtin_ctz(v);
                table[byte][v] = table[byte][v & (v-1)] ^ (bit <= H ? contrib[bit] : 0);
            }
        }
        for (int s = 0; s < M; s++) {
            uint64_t rhs = fixed ^ Q1[s];
            stats[2]++;
            int ok = 1;
            for (int q = 0; q < nd && ok; q++) if (__builtin_popcountll(deps[q] & rhs) & 1) ok = 0;
            if (!ok) continue;
            stats[3]++;
            uint64_t x0 = table[0][rhs & 255] ^ table[1][(rhs >> 8) & 255] ^ table[2][(rhs >> 16) & 255];
            const int *ra = RA + s*H;
            for (uint64_t z = 0; z < (UINT64_C(1) << nk); z++) {
                uint64_t x = x0;
                for (int t = 0; t < nk; t++) if (z >> t & 1) x ^= kernel[t];
                stats[4]++;
                if (__builtin_popcountll(x) != nB) continue;
                int exact = 1;
                for (int u = 1; u <= H && exact; u++) {
                    int twice = base2[u] + ra[u-1];
                    exact = twice >= 0 && 2*__builtin_popcountll((x ^ rotl(x, u, d, all)) & act[u-1]) == twice;
                }
                if (!exact) continue;
                if (found < maxOut) { outSB[found] = SB; outSig[found] = (uint32_t)s; outX[found] = x; }
                found++;
                goto next_support;   /* one witness per support orbit is enough for the G check */
            }
        }
        next_support:;
    }
    return found;
}
