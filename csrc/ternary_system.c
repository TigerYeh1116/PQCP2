/* Exact integer/GF2 affine plan construction; no floating point or hashes. */
#include <stdint.h>
#include <string.h>

typedef struct { uint64_t row, label; int rhs; } Equation;
typedef struct {
    Equation basis[47], checks[96];
    int rank, count, consistent;
} System;

static void insert(System *s, Equation e) {
    while (e.row) {
        int p = 63-__builtin_clzll(e.row);
        if (!s->basis[p].row) { s->basis[p] = e; ++s->rank; return; }
        e.row ^= s->basis[p].row; e.label ^= s->basis[p].label; e.rhs ^= s->basis[p].rhs;
    }
    if (e.label) s->checks[s->count++] = e;
    else if (e.rhs) s->consistent = 0;
}

static int expressions(const System *system, int variables, uint64_t *gm, uint64_t *tm, int *constants) {
    memset(gm, 0, 47*sizeof(uint64_t)); memset(tm, 0, 47*sizeof(uint64_t));
    memset(constants, 0, 47*sizeof(int));
    int free_count = 0;
    for (int col = 0; col < variables; ++col) {
        Equation e = system->basis[col];
        if (!e.row) gm[col] = UINT64_C(1) << free_count++;
        else {
            tm[col] = e.label; constants[col] = e.rhs;
            uint64_t rest = e.row ^ (UINT64_C(1) << col);
            while (rest) {
                int bit = __builtin_ctzll(rest);
                gm[col] ^= gm[bit]; tm[col] ^= tm[bit]; constants[col] ^= constants[bit];
                rest &= rest-1;
            }
        }
    }
    return free_count;
}

int pqcp_affine_sign_plan(const uint64_t *supports, int d, const int *wanted,
    const int *negative, int selected, int tight_counts, uint64_t *base, uint64_t *free_toggles,
    uint64_t *pattern_toggles, uint64_t *check_masks, int *check_rhs, int *metadata)
{
    if (d < 3 || d > 47 || !(d & 1) || selected < -1 || selected > 1 || tight_counts < 0 || tight_counts > 1) return -1;
    uint64_t full = (UINT64_C(1) << d)-1;
    int ids[2][47], side[47], position[47], variables = 0, sizes[2];
    memset(ids, -1, sizeof(ids)); memset(metadata, 0, 8*sizeof(int));
    for (int s = 0; s < 2; ++s) {
        if (supports[s] & ~full) return -1;
        sizes[s] = __builtin_popcountll(supports[s]);
        if (negative[s] < 0 || negative[s] > sizes[s]) return -1;
        for (int i = 0; i < d; ++i) if ((supports[s] >> i) & 1) {
            if (variables == 47) return -1;
            ids[s][i] = variables; side[variables] = s; position[variables++] = i;
        }
    }
    metadata[0] = variables;
    metadata[2] = variables; /* Range contradiction before elimination. */
    Equation equations[25]; int count = 0;
    for (int u = 1; u <= d/2; ++u) {
        uint64_t row = 0; int terms = 0;
        for (int s = 0; s < 2; ++s) for (int i = 0; i < d; ++i) {
            int j = i+u; if (j >= d) j -= d;
            if (ids[s][i] >= 0 && ids[s][j] >= 0) {
                row ^= (UINT64_C(1) << ids[s][i]) ^ (UINT64_C(1) << ids[s][j]); ++terms;
            }
        }
        if (wanted[u] > terms || wanted[u] < -terms || (terms-wanted[u])%2) return 0;
        equations[count++] = (Equation){row, 0, ((terms-wanted[u])/2)&1};
    }
    for (int s = 0; s < 2; ++s) {
        uint64_t row = 0;
        for (int i = 0; i < d; ++i) if (ids[s][i] >= 0) row |= UINT64_C(1) << ids[s][i];
        equations[count++] = (Equation){row, 0, negative[s]&1};
    }
    System raw; memset(&raw, 0, sizeof(raw)); raw.consistent = 1;
    for (int i = 0; i < count; ++i) insert(&raw, equations[i]);
    metadata[1] = raw.rank; metadata[2] = variables-raw.rank; metadata[7] = raw.consistent;
    if (!raw.consistent) return 0;
    uint64_t gm[47], tm[47]; int constants[47];
    if (tight_counts) {
        expressions(&raw, variables, gm, tm, constants);
        int fixed[2] = {0}, flexible[2] = {0};
        for (int col = 0; col < variables; ++col)
            if (gm[col]) ++flexible[side[col]];
            else fixed[side[col]] += constants[col];
        for (int s = 0; s < 2; ++s)
            if (negative[s] < fixed[s] || negative[s] > fixed[s]+flexible[s]) {
                metadata[7] = 2; /* Linear space exists; its exact count cannot. */
                return 0;
            }
    }
    System augmented; System *plan = &raw;
    int patterns = 0;
    if (selected >= 0) {
        memset(&augmented, 0, sizeof(augmented)); augmented.consistent = 1;
        for (int i = 0; i < variables; ++i) if (side[i] == selected)
            insert(&augmented, (Equation){UINT64_C(1) << i, UINT64_C(1) << patterns++, 0});
        for (int i = 0; i < count; ++i) insert(&augmented, equations[i]);
        if (!augmented.consistent) return -1;
        plan = &augmented;
    }
    metadata[3] = plan->rank-raw.rank; metadata[4] = variables-plan->rank;
    metadata[5] = patterns; metadata[6] = plan->count;
    memset(base, 0, 2*sizeof(uint64_t));
    memset(free_toggles, 0, 94*sizeof(uint64_t));
    memset(pattern_toggles, 0, 94*sizeof(uint64_t));
    expressions(plan, variables, gm, tm, constants);
    for (int col = 0; col < variables; ++col) {
        uint64_t location = UINT64_C(1) << position[col];
        if (constants[col]) base[side[col]] |= location;
        uint64_t bits = gm[col];
        while (bits) { int bit = __builtin_ctzll(bits); free_toggles[2*bit+side[col]] |= location; bits &= bits-1; }
        bits = tm[col];
        while (bits) { int bit = __builtin_ctzll(bits); pattern_toggles[2*bit+side[col]] |= location; bits &= bits-1; }
    }
    for (int i = 0; i < plan->count; ++i) { check_masks[i] = plan->checks[i].label; check_rhs[i] = plan->checks[i].rhs; }
    return 1;
}
