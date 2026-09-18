/* Bounded fixed-margin rectangle search. No globals, files, or Z3 calls.
 * CRT cells are (i mod 4, i mod n), L=4n, n odd. A legal switch flips
 * four corners with checkerboard bits, preserving every row/column sum.
 * Correlation changes only on directed cyclic edges with ONE flipped end.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>

static int valid_length(int L) { return L >= 12 && L <= 4096 && L % 8 == 4; }

static uint64_t random_u64(uint64_t *state) {
    uint64_t z = (*state += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    return z ^ (z >> 31);
}

static int cell(int n, int r, int c) {
    /* n^{-1}=n mod 4 for odd n. */
    return c + n * (((r-c)*n % 4 + 4) % 4);
}

static int contains(const int p[4], int i) {
    return i == p[0] || i == p[1] || i == p[2] || i == p[3];
}

static int delta_at(const int *bits, int L, const int p[4], int u) {
    int change = 0;
    for (int j = 0; j < 4; ++j) {
        int x = p[j], forward = (x+u)%L, backward = (x-u+L)%L;
        int sign = 1-2*bits[x];
        if (!contains(p, forward)) change -= 2*sign*(1-2*bits[forward]);
        if (!contains(p, backward)) change -= 2*sign*(1-2*bits[backward]);
    }
    return change;
}

/* Diagnostic API also used to compare every delta with the Python baseline. */
int pqcp_rectangle_delta(int L, const int *bits, int r, int s, int c, int d, int *out) {
    if (!valid_length(L) || r < 0 || s < 0 || r >= 4 || s >= 4 || r == s ||
        c < 0 || d < 0 || c >= L/4 || d >= L/4 || c == d) return -1;
    for (int i = 0; i < L; ++i) if (bits[i] != 0 && bits[i] != 1) return -1;
    int n = L/4, p[4] = {cell(n,r,c),cell(n,r,d),cell(n,s,c),cell(n,s,d)};
    if (bits[p[0]] != bits[p[3]] || bits[p[1]] != bits[p[2]] || bits[p[0]] == bits[p[1]])
        return -2;
    out[0] = 0;
    for (int u = 1; u <= L/2; ++u) out[u] = out[L-u] = delta_at(bits,L,p,u);
    return 0;
}

static int64_t energy(const int *profile, int L, int k, int eta) {
    int64_t total = 0;
    for (int u = 1; u <= L/2; ++u) {
        int64_t residual = profile[u] - (u == k ? 4*eta : 0);
        total += residual*residual;
    }
    return total;
}

/* out bits contain the BEST state; returned stats are
 * moves, rectangle evaluations, accepted moves, initial energy, best energy,
 * no-legal-move flag. The full output profile belongs to the best state.
 * Raw integer E=sum_{1<=u<=L/2}(S[u]-T[u])^2, not the PQCP objective score.
 */
int pqcp_rectangle_search(int L, int k, int eta, int *a, int *b,
                          uint64_t seed, int moves, int samples,
                          int *out_profile, int64_t stats[6]) {
    if (!valid_length(L) || k <= 0 || k >= L/2 || (eta != 1 && eta != -1) ||
        moves < 0 || samples <= 0) return -1;
    for (int i = 0; i < L; ++i)
        if ((a[i] != 0 && a[i] != 1) || (b[i] != 0 && b[i] != 1)) return -1;
    /* 2L current bits, L profile, L best profile, 2L scratch deltas,
     * and 24*n column positions grouped by side/row pair/orientation. */
    int *memory = calloc((size_t)(12*L), sizeof(int));
    if (!memory) return -3;
    int *current = memory, *profile = current+2*L, *best_profile = profile+L;
    int *delta = best_profile+L, *selected_delta = delta+L, *columns = selected_delta+L;
    int n = L/4;
    memcpy(current,a,(size_t)L*sizeof(int));
    memcpy(current+L,b,(size_t)L*sizeof(int));
    memset(stats,0,6*sizeof(int64_t));
    for (int u = 0; u < L; ++u)
        for (int side = 0; side < 2; ++side)
            for (int i = 0; i < L; ++i)
                profile[u] += (1-2*current[side*L+i])*(1-2*current[side*L+(i+u)%L]);
    memcpy(best_profile,profile,(size_t)L*sizeof(int));
    int64_t now = energy(profile,L,k,eta), best = now;
    stats[3] = best;
    for (int move = 0; move < moves && best; ++move) {
        int counts[12][2] = {{0}}, rows[12][2], total = 0, group = 0;
        for (int side = 0; side < 2; ++side)
            for (int r = 0; r < 4; ++r)
                for (int s = r+1; s < 4; ++s, ++group) {
                    rows[group][0] = r; rows[group][1] = s;
                    for (int c = 0; c < n; ++c) {
                        int x = current[side*L+cell(n,r,c)], y = current[side*L+cell(n,s,c)];
                        if (x != y) columns[(2*group+x)*n+counts[group][x]++] = c;
                    }
                    total += counts[group][0]*counts[group][1];
                }
        if (!total) { stats[5] = 1; break; }
        int selected[4] = {0}, selected_side = 0;
        int64_t proposal = INT64_MAX;
        for (int sample = 0; sample < samples; ++sample) {
            int draw = (int)(random_u64(&seed) % (uint64_t)total), g = 0;
            while (draw >= counts[g][0]*counts[g][1]) {
                draw -= counts[g][0]*counts[g][1]; ++g;
            }
            int c = columns[2*g*n+draw/counts[g][1]];
            int d = columns[(2*g+1)*n+draw%counts[g][1]];
            int p[4] = {cell(n,rows[g][0],c),cell(n,rows[g][0],d),
                        cell(n,rows[g][1],c),cell(n,rows[g][1],d)};
            int side = g/6;
            int64_t candidate = now;
            for (int u = 1; u <= L/2; ++u) {
                delta[u] = delta_at(current+side*L,L,p,u);
                int64_t before = profile[u]-(u == k ? 4*eta : 0), after = before+delta[u];
                candidate += after*after-before*before;
            }
            ++stats[1];
            if (candidate < proposal) {
                proposal = candidate; selected_side = side;
                memcpy(selected,p,sizeof(selected));
                memcpy(selected_delta,delta,(size_t)L*sizeof(int));
            }
        }
        ++stats[0];
        /* Same raw-energy Metropolis scale and linear temperature family as
         * the existing C search. These are heuristics, NOT pruning rules. */
        double temperature = 6.0*(1.0-(double)move/moves)+0.15;
        double probability = proposal <= now ? 1.0 : exp(-(double)(proposal-now)/(64.0*temperature));
        double draw = (double)(random_u64(&seed) >> 11)*0x1.0p-53;
        if (proposal <= now || draw < probability) {
            for (int j = 0; j < 4; ++j) current[selected_side*L+selected[j]] ^= 1;
            for (int u = 1; u <= L/2; ++u) {
                profile[u] += selected_delta[u]; profile[L-u] = profile[u];
            }
            now = proposal; ++stats[2];
            if (now < best) {
                best = now;
                memcpy(a,current,(size_t)L*sizeof(int));
                memcpy(b,current+L,(size_t)L*sizeof(int));
                memcpy(best_profile,profile,(size_t)L*sizeof(int));
            }
        }
    }
    stats[4] = best;
    memcpy(out_profile,best_profile,(size_t)L*sizeof(int));
    free(memory);
    return 0;
}
