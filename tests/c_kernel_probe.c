/* Expose the production static kernels to independent Python checks. */
#define main pqcp_embedded_main
#include "../csrc/pqcp_search.c"
#undef main

int probe_delta(const signed char *s, int L, int p, int q, int u)
{
    return swap_corr_delta(s, L, p, q, u);
}

int probe_quarter(int L, int k, int sign, int ae, int ao, int be, int bo)
{
    return quarter_content_feasible(L, k, sign, ae, ao, be, bo);
}

uint64_t probe_remainder(uint64_t value, int L)
{
    uint64_t reciprocal = UINT64_MAX / (uint64_t)L;
    uint64_t quotient = (uint64_t)(((__uint128_t)value * reciprocal) >> 64);
    uint64_t remainder = value - quotient * (uint64_t)L;
    return remainder >= (uint64_t)L ? remainder - (uint64_t)L : remainder;
}

long long probe_energy(const signed char *a, const signed char *b, int L,
                       int p, int q, int k, int sign, int reference)
{
    int ca[L], cb[L];
    compute_pacf(a, L, ca);
    compute_pacf(b, L, cb);
    long long current = energy(ca, cb, L, k, sign);
    return reference
        ? reference_energy_after_swap(current, a, ca, cb, L, p, q, k, sign)
        : energy_after_swap(current, a, ca, cb, L, p, q, k, sign);
}

int probe_pool_pairs(const signed char *s, int L, int *pairs)
{
    int positions[4 * L], slots[L];
    MovePool pool = {.positions = positions, .slot = slots};
    move_pool_init(&pool, s, L);
    for (uint64_t code = 0; code < pool.total; ++code)
        move_pool_decode(&pool, code, pairs + 2 * code, pairs + 2 * code + 1);
    return (int)pool.total;
}

int probe_pool_trace(signed char *s, int L, int steps, int *pairs)
{
    int positions[4 * L], slots[L];
    MovePool pool = {.positions = positions, .slot = slots};
    move_pool_init(&pool, s, L);
    uint64_t rng = 123;
    for (int step = 0; step < steps; ++step) {
        for (int bucket = 0; bucket < 4; ++bucket) {
            for (int slot = 0; slot < pool.count[bucket]; ++slot) {
                int pos = pool.positions[bucket * L + slot];
                if (2 * (pos & 1) + (s[pos] < 0) != bucket || pool.slot[pos] != slot)
                    return -1;
            }
        }
        int p, q;
        if (!choose_pool_pair(&pool, &rng, &p, &q)) return step;
        pairs[2 * step] = p; pairs[2 * step + 1] = q;
        move_pool_swap(&pool, s, p, q);
        s[p] = -s[p]; s[q] = -s[q];
    }
    return steps;
}
