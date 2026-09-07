#define _DARWIN_C_SOURCE
#define _POSIX_C_SOURCE 200809L

#include <math.h>
#include <limits.h>
#include <pthread.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

/*
 * Search for the Project-2 kind of binary (L,4)-PQCP:
 *   S(u) = rho(a;u) + rho(b;u)
 * is zero except at exactly two shifts k and L-k, where it is +4 or -4.
 *
 * Internally +1 is stored as 0 and -1 as 1 when a result is printed.
 */

typedef struct {
    int x, y;       /* sum(a)=2x, sum(b)=2y (L is even) */
    int side_sign;  /* desired nonzero PACS: 4 * side_sign */
} Profile;

typedef struct {
    int L;
    int profile_count;
    Profile *profiles;
    int k_count;
    int *k_values;
    unsigned thread_id;
} WorkerArgs;
typedef struct {
    signed char *a, *b;
    int k, side_sign;
} PairSeed;
typedef struct {
    signed char *values;
    int half_sum;
    int half_alt;
} FkmSeed;

static atomic_int g_found = 0; /* number of distinct valid results found */
static atomic_ullong g_restarts = 0;
static atomic_ullong g_moves = 0;
static atomic_ullong g_swap_evaluations = 0;
static atomic_ullong g_valid_results = 0;
static atomic_ullong g_fkm_applied = 0;
static atomic_ullong g_fkm_misses = 0;
static volatile sig_atomic_t g_stop = 0;
static signed char *g_answer_a;
static signed char *g_answer_b;
static int g_answer_k;
static int g_answer_side_sign;
static pthread_mutex_t g_answer_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t g_output_lock = PTHREAD_MUTEX_INITIALIZER;
static int g_candidate_count = 2;
static unsigned long long g_restart_limit = 0; /* optional finite test budget */
static uint64_t g_length_reciprocal = 0;
static int g_quench = 0; /* opt-in measurement; production default unchanged */
static int g_move_pool = 0; /* opt-in direct legal-swap sampler */
static int g_quarter_pruning = 0; /* opt-in exact content feasibility test */
static FkmSeed *g_fkm_seeds = NULL;
static size_t g_fkm_seed_count = 0;
static size_t g_fkm_seed_cursor = 0;
static pthread_mutex_t g_seed_lock = PTHREAD_MUTEX_INITIALIZER;
static PairSeed *g_pair_seeds = NULL;
static size_t g_pair_seed_count = 0;

static void print_bits(FILE *f, const signed char *s, int L);

static int claim_restart(unsigned long long *restart)
{
    if (!g_restart_limit) {
        *restart = atomic_fetch_add(&g_restarts, 1);
        return 1;
    }
    *restart = atomic_load(&g_restarts);
    while (*restart < g_restart_limit) {
        if (atomic_compare_exchange_weak(&g_restarts, restart, *restart + 1))
            return 1;
    }
    return 0;
}

static void load_pair_seeds(const char *filename, int L)
{
    FILE *f = fopen(filename, "r");
    if (!f) return;
    char *line = NULL; size_t cap = 0;
    int k = 1, side = 1; char *a_line = NULL;
    while (getline(&line, &cap, f) != -1) {
        size_t n = strcspn(line, "\r\n"); line[n] = '\0';
        if (strncmp(line, "nonzero shifts=", 15) == 0)
            sscanf(line + 15, "%d", &k);
        else if (strncmp(line, "nonzero PACS=", 13) == 0)
            sscanf(line + 13, "%d", &side), side = side < 0 ? -1 : 1;
        else if (strncmp(line, "a=", 2) == 0) {
            free(a_line); a_line = strdup(line + 2);
        } else if (strncmp(line, "b=", 2) == 0 && a_line && (int)strlen(a_line) == L && (int)strlen(line + 2) == L) {
            PairSeed *tmp = realloc(g_pair_seeds, (g_pair_seed_count + 1) * sizeof(*tmp));
            if (!tmp) exit(EXIT_FAILURE);
            g_pair_seeds = tmp;
            PairSeed *p = &g_pair_seeds[g_pair_seed_count++];
            p->a = malloc((size_t)L); p->b = malloc((size_t)L);
            p->k = k; p->side_sign = side;
            int valid = 1;
            for (int i = 0; i < L; ++i) {
                if ((a_line[i] != '0' && a_line[i] != '1') ||
                    (line[2+i] != '0' && line[2+i] != '1')) { valid = 0; break; }
                p->a[i] = a_line[i] == '0' ? 1 : -1;
                p->b[i] = line[2+i] == '0' ? 1 : -1;
            }
            if (!valid) { free(p->a); free(p->b); --g_pair_seed_count; }
            free(a_line); a_line = NULL;
        }
    }
    free(a_line); free(line); fclose(f);
    if (g_pair_seed_count)
        printf("pair seed queue enabled: %s (%zu pairs)\n", filename, g_pair_seed_count);
}

static void load_fkm_seeds(const char *filename, int L)
{
    FILE *f = fopen(filename, "r");
    if (!f) {
        perror(filename);
        exit(EXIT_FAILURE);
    }
    char *line = NULL;
    size_t cap = 0;
    while (getline(&line, &cap, f) != -1) {
        size_t n = strcspn(line, "\r\n"); line[n] = '\0';
        if ((int)n != L) continue;
        signed char *values = malloc((size_t)L);
        if (!values) exit(EXIT_FAILURE);
        int sum = 0, alt = 0, valid = 1;
        for (int i = 0; i < L; ++i) {
            if (line[i] != '0' && line[i] != '1') { valid = 0; break; }
            values[i] = line[i] == '0' ? 1 : -1;
            sum += values[i];
            alt += (i & 1) ? -values[i] : values[i];
        }
        if (!valid || (sum & 1) || (alt & 1)) {
            free(values);
            continue;
        }
        FkmSeed *resized = realloc(
            g_fkm_seeds, (g_fkm_seed_count + 1) * sizeof(*resized));
        if (!resized) exit(EXIT_FAILURE);
        g_fkm_seeds = resized;
        g_fkm_seeds[g_fkm_seed_count++] = (FkmSeed){
            values, sum / 2, alt / 2
        };
    }
    free(line); fclose(f);
    if (g_fkm_seed_count == 0) {
        fprintf(stderr, "compressed FKM seed bank is empty: %s\n", filename);
        exit(EXIT_FAILURE);
    }
    printf("compressed FKM seed bank enabled: %s (%zu seeds)\n",
           filename, g_fkm_seed_count);
}

static int next_fkm_seed(signed char *s, int L, int wanted_half,
                         int wanted_alt)
{
    if (!g_fkm_seeds || g_fkm_seed_count == 0) return 0;
    int ok = 0;
    pthread_mutex_lock(&g_seed_lock);
    for (size_t offset = 0; offset < g_fkm_seed_count; ++offset) {
        size_t index = (g_fkm_seed_cursor + offset) % g_fkm_seed_count;
        FkmSeed *candidate = &g_fkm_seeds[index];
        if (candidate->half_sum == wanted_half &&
            (wanted_alt == INT_MAX || candidate->half_alt == wanted_alt)) {
            memcpy(s, candidate->values, (size_t)L);
            g_fkm_seed_cursor = (index + 1) % g_fkm_seed_count;
            ok = 1;
            break;
        }
    }
    pthread_mutex_unlock(&g_seed_lock);
    return ok;
}

static void on_sigint(int sig)
{
    (void)sig;
    g_stop = 1;
}

static uint64_t rng_next(uint64_t *state)
{
    uint64_t x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    return x * UINT64_C(2685821657736338717);
}

static double rng_unit(uint64_t *state)
{
    return (rng_next(state) >> 11) * (1.0 / 9007199254740992.0);
}

static int gcd_int(int a, int b)
{
    while (b != 0) {
        int t = a % b;
        a = b;
        b = t;
    }
    return a;
}

static int random_fixed_sum_and_alternating_sum(signed char *s, int L,
                                                 int half_sum,
                                                 int half_alt_sum,
                                                 uint64_t *rng)
{
    int n = L / 2;
    int even_sum = half_sum + half_alt_sum;
    int odd_sum = half_sum - half_alt_sum;
    if (abs(even_sum) > n || abs(odd_sum) > n ||
        ((n - even_sum) & 1) || ((n - odd_sum) & 1))
        return 0;
    int even_minus = (n - even_sum) / 2;
    int odd_minus = (n - odd_sum) / 2;
    for (int j = 0; j < n; ++j) {
        s[2 * j] = j < even_minus ? -1 : 1;
        s[2 * j + 1] = j < odd_minus ? -1 : 1;
    }
    for (int parity = 0; parity < 2; ++parity) {
        for (int j = n - 1; j > 0; --j) {
            int z = (int)(rng_next(rng) % (uint64_t)(j + 1));
            signed char t = s[2 * j + parity];
            s[2 * j + parity] = s[2 * z + parity];
            s[2 * z + parity] = t;
        }
    }
    return 1;
}

/* Change a PG(68) seed to the required DC and Nyquist marginals with the
 * fewest possible bit edits.  A first phase changes the ordinary sum; a
 * cross-parity swap then changes the alternating sum by exactly 4 while
 * preserving the ordinary sum. */
static int repair_marginals(signed char *s, int L, int target_half,
                            int target_alt, uint64_t *rng)
{
    int current_sum = 0, current_alt = 0;
    for (int i = 0; i < L; ++i) {
        current_sum += s[i];
        current_alt += (i & 1) ? -s[i] : s[i];
    }
    int wanted_sum = 2 * target_half;
    int direction = wanted_sum > current_sum ? 1 : -1;
    int flips = abs(wanted_sum - current_sum) / 2;
    int best_even = -1, best_distance = INT_MAX;
    for (int even_flips = 0; even_flips <= flips; ++even_flips) {
        int odd_flips = flips - even_flips;
        int delta_alt = direction * 2 * (even_flips - odd_flips);
        int remaining = target_alt - (current_alt + delta_alt);
        if ((remaining & 3) != 0) continue;
        int available_even = 0, available_odd = 0;
        int wanted_value = direction > 0 ? -1 : 1;
        for (int i = 0; i < L; ++i)
            if (s[i] == wanted_value) {
                if (i & 1) ++available_odd;
                else ++available_even;
            }
        if (even_flips > available_even || odd_flips > available_odd) continue;
        if (abs(remaining) < best_distance) {
            best_distance = abs(remaining);
            best_even = even_flips;
        }
    }
    if (best_even < 0) return 0;
    int odd_flips = flips - best_even;
    int wanted_value = direction > 0 ? -1 : 1;
    for (int parity = 0; parity < 2; ++parity) {
        int count = parity == 0 ? best_even : odd_flips;
        for (int z = 0; z < count; ++z) {
            int i, tries = 0;
            do {
                i = (int)(rng_next(rng) % (uint64_t)L);
                ++tries;
            } while ((i & 1) != parity || s[i] != wanted_value);
            (void)tries;
            s[i] = (signed char)-s[i];
        }
    }
    current_alt = 0;
    for (int i = 0; i < L; ++i)
        current_alt += (i & 1) ? -s[i] : s[i];
    int difference = target_alt - current_alt;
    if (difference & 3) return 0;
    while (difference != 0) {
        int need_increase = difference > 0;
        int even_value = need_increase ? -1 : 1;
        int odd_value = need_increase ? 1 : -1;
        int even = -1, odd = -1;
        for (int i = 0; i < L; i += 2)
            if (s[i] == even_value) { even = i; break; }
        for (int i = 1; i < L; i += 2)
            if (s[i] == odd_value) { odd = i; break; }
        if (even < 0 || odd < 0) return 0;
        s[even] = (signed char)-s[even];
        s[odd] = (signed char)-s[odd];
        difference += need_increase ? -4 : 4;
    }
    return 1;
}

/* A verified PG(68) from the SDS literature.  It has zero out-of-phase
 * PACS and rowsums 6 and 10, making it a useful nearby starting point for
 * the required rowsums (8,8) or (0,12).  Equivalence transformations create
 * many distinct basins before the minimal marginal adjustment. */
static int seeded_pg68_pair(signed char *a, signed char *b, int half_a,
                            int alt_a, int half_b, int alt_b, uint64_t *rng)
{
    static const char seed_a[] =
        "11111111011010110000110001001001011000001110011000101010110010000000";
    static const char seed_b[] =
        "10111011001101001010110101011100000110101000110000110000001100100000";
    signed char x[68], y[68];
    for (int i = 0; i < 68; ++i) {
        x[i] = seed_a[i] == '1' ? -1 : 1;
        y[i] = seed_b[i] == '1' ? -1 : 1;
    }
    int shift_x = (int)(rng_next(rng) % 68), shift_y = (int)(rng_next(rng) % 68);
    int decimation;
    do { decimation = 1 + (int)(rng_next(rng) % 67); }
    while (gcd_int(decimation, 68) != 1);
    int negate_x = (int)(rng_next(rng) & 1), negate_y = (int)(rng_next(rng) & 1);
    int swap = (int)(rng_next(rng) & 1);
    for (int i = 0; i < 68; ++i) {
        signed char vx = x[(decimation * i + shift_x) % 68];
        signed char vy = y[(decimation * i + shift_y) % 68];
        if (negate_x) vx = (signed char)-vx;
        if (negate_y) vy = (signed char)-vy;
        a[i] = swap ? vy : vx;
        b[i] = swap ? vx : vy;
    }
    return repair_marginals(a, 68, half_a, alt_a, rng) &&
           repair_marginals(b, 68, half_b, alt_b, rng);
}

/* At frequency L/4 (4 divides L), the target spectrum is
 * 2L + 8*sign*cos(pi*k/2). For one sequence with w ones in a parity class,
 * split over its two mod-4 classes, the Fourier component divided by 2 is
 * d=w-2*c, c in [max(0,w-L/4), min(w,L/4)]. Its attainable squared values
 * are d^2 with 0<=d<=min(w,L/2-w), d==w (mod 2).
 * The four components (A-even,A-odd,B-even,B-odd) are independent given
 * these contents. Test their sum of squares against L/2+2*sign*cos(pi*k/2).
 * A rejection rules out EVERY sequence pair with these fixed parity counts,
 * not just the current assignment; same-parity swaps cannot escape it. */
static int quarter_content_feasible(int L, int k, int side_sign,
                                     int ae, int ao, int be, int bo)
{
    if (L % 4) return 1;
    int cosine = (k & 1) ? 0 : ((k % 4 == 0) ? 1 : -1);
    int target = L / 2 + 2 * side_sign * cosine;
    if (target < 0) return 0;
    int weights[4] = {ae, ao, be, bo}, bounds[4];
    for (int z = 0; z < 4; ++z) {
        if (weights[z] < 0 || weights[z] > L / 2) return 0;
        bounds[z] = weights[z] < L / 2 - weights[z]
                    ? weights[z] : L / 2 - weights[z];
    }
    unsigned char reachable[target + 1];
    memset(reachable, 0, sizeof(reachable));
    for (int x = weights[0] & 1; x <= bounds[0] && x*x <= target; x += 2)
        for (int y = weights[1] & 1; y <= bounds[1] && x*x+y*y <= target; y += 2)
            reachable[x*x + y*y] = 1;
    for (int x = weights[2] & 1; x <= bounds[2] && x*x <= target; x += 2)
        for (int y = weights[3] & 1; y <= bounds[3] && x*x+y*y <= target; y += 2)
            if (reachable[target - x*x - y*y]) return 1;
    return 0;
}

/* Enforce both the DC and Nyquist spectral identities.  Besides
 * x^2+y^2=L/2+2*sign, an exact solution must satisfy
 * r^2+s^2=L/2+2*sign*(-1)^k, where 2r and 2s are alternating sums. */
static int random_spectral_pair(signed char *a, signed char *b, int L,
                                Profile p, int k, uint64_t *rng)
{
    int n = L / 2 + 2 * p.side_sign * ((k & 1) ? -1 : 1);
    if (n < 0) return 0;
    typedef struct { int r, s; } AltProfile;
    AltProfile choices[128];
    int count = 0;
    for (int r0 = 0; r0 * r0 <= n; ++r0) {
        int s2 = n - r0 * r0;
        int s0 = (int)(sqrt((double)s2) + 0.5);
        if (s0 * s0 != s2) continue;
        int base[2][2] = {{r0, s0}, {s0, r0}};
        int assignments = r0 == s0 ? 1 : 2;
        for (int z = 0; z < assignments; ++z) {
            for (int sr = -1; sr <= 1; sr += 2) {
                for (int ss = -1; ss <= 1; ss += 2) {
                    int r = sr * base[z][0];
                    int s = ss * base[z][1];
                    int half = L / 2;
                    int ae = p.x + r, ao = p.x - r;
                    int be = p.y + s, bo = p.y - s;
                    if (abs(ae) <= half && abs(ao) <= half &&
                        abs(be) <= half && abs(bo) <= half &&
                        !((half - ae) & 1) && !((half - ao) & 1) &&
                        !((half - be) & 1) && !((half - bo) & 1) &&
                        count < (int)(sizeof(choices) / sizeof(choices[0])) &&
                        (!g_quarter_pruning || quarter_content_feasible(
                            L, k, p.side_sign, (half-ae)/2, (half-ao)/2,
                            (half-be)/2, (half-bo)/2)))
                        choices[count++] = (AltProfile){r, s};
                }
            }
        }
    }
    if (count == 0) return 0;
    AltProfile q = choices[rng_next(rng) % (uint64_t)count];
    if (L == 68 && (rng_next(rng) & 1))
        return seeded_pg68_pair(a, b, p.x, q.r, p.y, q.s, rng);
    return random_fixed_sum_and_alternating_sum(a, L, p.x, q.r, rng) &&
           random_fixed_sum_and_alternating_sum(b, L, p.y, q.s, rng);
}

static void compute_pacf(const signed char *s, int L, int *corr)
{
    corr[0] = L;
    for (int u = 1; u <= L / 2; ++u) {
        int total = 0;
        for (int i = 0; i < L; ++i)
            total += s[i] * s[(i + u) % L];
        corr[u] = total;
        corr[L - u] = total;
    }
}

/* PACF(u)=PACF(L-u), so the search only maintains 1 <= u <= L/2.
 * Compute the PACF change caused by flipping the opposite entries p and q.
 * A directed product changes sign iff exactly one endpoint is flipped. */
static inline int reference_swap_corr_delta(const signed char *s, int L, int p, int q, int u)
{
    int idx[4] = {p, q, (p - u + L) % L, (q - u + L) % L};
    int delta = 0;
    for (int z = 0; z < 4; ++z) {
        int unique = 1;
        for (int j = 0; j < z; ++j)
            if (idx[j] == idx[z]) unique = 0;
        if (!unique) continue;
        int right = (idx[z] + u) % L;
        int left_flipped = idx[z] == p || idx[z] == q;
        int right_flipped = right == p || right == q;
        if (left_flipped != right_flipped)
            delta -= 2 * s[idx[z]] * s[right];
    }
    return delta;
}

/* Exact two-flip identity (u != 0): add the two individual flip changes,
 * then restore the edge p--q if it was counted twice.  At u=L/2 there are
 * two directed edges, hence both correction indicators must be retained.
 * Conditional wrapping replaces division; all indices lie in (-L,2L).
 * PQCP_REFERENCE_KERNEL retains the original evaluator for rollback and
 * paired trajectory benchmarks.  No RNG or acceptance rule changes. */
static inline int swap_corr_delta(const signed char *s, int L, int p, int q, int u)
{
#ifdef PQCP_REFERENCE_KERNEL
    return reference_swap_corr_delta(s, L, p, q, u);
#else
    int pp = p + u, pm = p - u, qp = q + u, qm = q - u;
    if (pp >= L) pp -= L;
    if (pm < 0) pm += L;
    if (qp >= L) qp -= L;
    if (qm < 0) qm += L;
    return -2 * s[p] * (s[pp] + s[pm])
           -2 * s[q] * (s[qp] + s[qm])
           +4 * s[p] * s[q] * ((pp == q) + (qp == p));
#endif
}

/* Flip two opposite entries (equivalent to swapping a 0 and a 1) and update
 * the independent half of the periodic autocorrelation in O(L) time. */
static void swap_and_update(signed char *s, int L, int p, int q, int *corr,
                            int *deltas)
{
    for (int u = 1; u <= L / 2; ++u)
        deltas[u] = swap_corr_delta(s, L, p, q, u);
    s[p] = (signed char)-s[p];
    s[q] = (signed char)-s[q];
    for (int u = 1; u <= L / 2; ++u)
        corr[u] += deltas[u];
}

static int folded_shift(int u, int L)
{
    u %= L;
    return u <= L - u ? u : L - u;
}

static int target_pacs(int u, int L, int k, int side_sign)
{
    u %= L;
    if (u == 0) return 2 * L;
    return (u == k || u == L - k) ? 4 * side_sign : 0;
}

/* Error after m-compression.  The compressed PACS at r is the alias sum
 * sum_q S(r+q*d), d=L/m.  This identity remains valid for our two-sidelobe
 * PQCP target even though the usual theorem is stated for Golay pairs. */
static long long compressed_energy(const int *ca, const int *cb, int L,
                                   int k, int side_sign, int m)
{
    int d = L / m;
    long long result = 0;
    for (int r = 0; r < d; ++r) {
        long long actual = 0, target = 0;
        for (int q = 0; q < m; ++q) {
            int u = r + q * d;
            int f = folded_shift(u, L);
            actual += ca[f] + cb[f];
            target += target_pacs(u, L, k, side_sign);
        }
        long long error = actual - target;
        result += error * error;
    }
    return result;
}

static long long energy(const int *ca, const int *cb, int L, int k,
                        int side_sign)
{
    long long e = 0;
    for (int u = 1; u <= L / 2; ++u) {
        int target = u == k ? 4 * side_sign : 0;
        long long d = (long long)ca[u] + cb[u] - target;
        e += d * d;
    }
    /* Coarse levels smooth the local-search landscape.  Factor 2 applies to
     * every Project-2 length; factor 4 adds the 44->22->11 and 68->34->17
     * multi-level paths. */
    e += 2 * compressed_energy(ca, cb, L, k, side_sign, 2);
    if (L % 4 == 0)
        e += 4 * compressed_energy(ca, cb, L, k, side_sign, 4);
    return e;
}

/* Evaluate a swap without modifying the sequence.  This lets each move look
 * at several alternatives and apply only the best one, avoiding repeated
 * apply/revert work for poor random proposals. */
static long long reference_energy_after_swap(long long current, const signed char *s,
                                   const int *own_corr,
                                   const int *other_corr, int L, int p, int q,
                                   int k, int side_sign)
{
    int delta[L / 2 + 1];
    for (int u = 1; u <= L / 2; ++u)
        delta[u] = reference_swap_corr_delta(s, L, p, q, u);
    long long result = current;
    for (int u = 1; u <= L / 2; ++u) {
        int target = u == k ? 4 * side_sign : 0;
        long long before = (long long)own_corr[u] + other_corr[u] - target;
        long long after = before + delta[u];
        result += after * after - before * before;
    }
    for (int m = 2; m <= 4; m *= 2) {
        if (m == 4 && L % 4 != 0) break;
        int d = L / m;
        long long old_e = 0, new_e = 0;
        for (int r = 0; r < d; ++r) {
            long long old_actual = 0, target = 0, change = 0;
            for (int z = 0; z < m; ++z) {
                int u = r + z * d;
                int f = folded_shift(u, L);
                old_actual += own_corr[f] + other_corr[f];
                target += target_pacs(u, L, k, side_sign);
                if (f != 0) change += delta[f];
            }
            long long old_err = old_actual - target;
            long long new_err = old_err + change;
            old_e += old_err * old_err;
            new_e += new_err * new_err;
        }
        result += (m == 2 ? 2 : 4) * (new_e - old_e);
    }
    return result;
}

/* e[u]=e[L-u], e[0]=0 for binary pairs.  With h=L/2, the factor-two
 * residual is v[r]=e[r]+e[h-r].  If 4 divides L, factor four is a second
 * fold: w[r]=v[r]+v[L/4-r].  Compute exactly the same weighted energy as
 * energy(), without repeated modulo/target evaluation per alias term. */
static long long energy_after_swap_collect(long long current, const signed char *s,
                                   const int *own_corr, const int *other_corr,
                                   int L, int p, int q, int k, int side_sign,
                                   int *deltas)
{
#ifdef PQCP_REFERENCE_KERNEL
    if (deltas)
        for (int u = 1; u <= L / 2; ++u)
            deltas[u] = reference_swap_corr_delta(s, L, p, q, u);
    return reference_energy_after_swap(current, s, own_corr, other_corr,
                                       L, p, q, k, side_sign);
#else
    (void)current;
    const int h = L / 2;
    int residual[h + 1], folded[h + 1];
    signed char periodic[3 * L];
    memcpy(periodic, s, (size_t)L);
    memcpy(periodic + L, s, (size_t)L);
    memcpy(periodic + 2 * L, s, (size_t)L);
    const signed char *sp = periodic + L + p;
    const signed char *sq = periodic + L + q;
    int separation = abs(p - q);
    residual[0] = 0;
    long long total = 0;
    for (int u = 1; u <= h; ++u) {
        int change = -2 * s[p] * (sp[u] + sp[-u])
                     -2 * s[q] * (sq[u] + sq[-u])
                     +4 * s[p] * s[q] * ((u == separation) + (u == L - separation));
        if (deltas) deltas[u] = change;
        int e = own_corr[u] + other_corr[u]
                - (u == k ? 4 * side_sign : 0)
                + change;
        residual[u] = e;
        total += (long long)e * e;
    }
    for (int r = 0; r < h; ++r) {
        int e = residual[r] + residual[h - r];
        folded[r] = e;
        total += 2LL * e * e;
    }
    if (L % 4 == 0) {
        const int d = L / 4;
        for (int r = 0; r < d; ++r) {
            int e = folded[r] + folded[d - r];
            total += 4LL * e * e;
        }
    }
    return total;
#endif
}

static long long energy_after_swap(long long current, const signed char *s,
                                   const int *own_corr, const int *other_corr,
                                   int L, int p, int q, int k, int side_sign)
{
    return energy_after_swap_collect(current, s, own_corr, other_corr,
                                     L, p, q, k, side_sign, NULL);
}

static int choose_opposite_pair(const signed char *s, int L, uint64_t *rng,
                                int *p, int *q)
{
    for (int tries = 0; tries < 64; ++tries) {
#ifdef PQCP_REFERENCE_KERNEL
        int i = (int)(rng_next(rng) % (uint64_t)L);
        int j = (int)(rng_next(rng) % (uint64_t)L);
#else
        /* floor((2^64-1)/L) gives a quotient no more than one too small;
         * one correction therefore computes exactly n % L. Retain every
         * RNG draw and rejection, so the proposal distribution AND the
         * seed-specific trajectory are identical to the original code. */
        uint64_t ni = rng_next(rng), nj = rng_next(rng);
        uint64_t qi = (uint64_t)(((__uint128_t)ni * g_length_reciprocal) >> 64);
        uint64_t qj = (uint64_t)(((__uint128_t)nj * g_length_reciprocal) >> 64);
        uint64_t ri = ni - qi * (uint64_t)L, rj = nj - qj * (uint64_t)L;
        int i = (int)(ri >= (uint64_t)L ? ri - (uint64_t)L : ri);
        int j = (int)(rj >= (uint64_t)L ? rj - (uint64_t)L : rj);
#endif
        /* Same parity preserves both the ordinary sum and the alternating
         * sum fixed by the DC/Nyquist necessary conditions. */
        if (i != j && ((i ^ j) & 1) == 0 && s[i] != s[j]) {
            *p = i;
            *q = j;
            return 1;
        }
    }
    return 0;
}

/* Partition positions by (parity, sign). Each Cartesian product of the
 * positive and negative lists is exactly one parity-preserving swap.
 * The two products are concatenated, NOT selected with equal parity weight:
 * that would bias the proposal whenever their sizes differ. */
typedef struct {
    int L, count[4];
    int *positions, *slot;
    uint64_t even_pairs, total;
    uint64_t total_reciprocal, rejection_threshold, minus_reciprocal[2];
} MovePool;

static void move_pool_init(MovePool *pool, const signed char *s, int L)
{
    pool->L = L;
    memset(pool->count, 0, sizeof(pool->count));
    for (int i = 0; i < L; ++i) {
        int bucket = 2 * (i & 1) + (s[i] < 0);
        int slot = pool->count[bucket]++;
        pool->positions[bucket * L + slot] = i;
        pool->slot[i] = slot;
    }
    pool->even_pairs = (uint64_t)pool->count[0] * pool->count[1];
    pool->total = pool->even_pairs + (uint64_t)pool->count[2] * pool->count[3];
    pool->total_reciprocal = pool->total ? UINT64_MAX / pool->total : 0;
    pool->rejection_threshold = pool->total ? -pool->total % pool->total : 0;
    for (int parity = 0; parity < 2; ++parity) {
        int count = pool->count[2 * parity + 1];
        pool->minus_reciprocal[parity] = count ? UINT64_MAX / (uint64_t)count : 0;
    }
}

static void move_pool_decode(const MovePool *pool, uint64_t code, int *p, int *q)
{
    int bucket = 0;
    if (code >= pool->even_pairs) {
        bucket = 2;
        code -= pool->even_pairs;
    }
    int minus_count = pool->count[bucket + 1];
#ifdef PQCP_REFERENCE_KERNEL
    *p = pool->positions[bucket * pool->L + code / (uint64_t)minus_count];
    *q = pool->positions[(bucket + 1) * pool->L + code % (uint64_t)minus_count];
#else
    uint64_t quotient = (uint64_t)(((__uint128_t)code * pool->minus_reciprocal[bucket / 2]) >> 64);
    uint64_t remainder = code - quotient * (uint64_t)minus_count;
    if (remainder >= (uint64_t)minus_count) {
        remainder -= (uint64_t)minus_count;
        ++quotient;
    }
    *p = pool->positions[bucket * pool->L + quotient];
    *q = pool->positions[(bucket + 1) * pool->L + remainder];
#endif
}

static int choose_pool_pair(const MovePool *pool, uint64_t *rng, int *p, int *q)
{
    if (!pool->total) return 0;
    /* Rejection only removes the uint64 modulo bias, rather than repeatedly
     * drawing illegal bit pairs. This deliberately consumes a different RNG
     * stream from the old sampler; use paired statistical comparisons. */
    uint64_t value;
    do { value = rng_next(rng); } while (value < pool->rejection_threshold);
#ifdef PQCP_REFERENCE_KERNEL
    move_pool_decode(pool, value % pool->total, p, q);
#else
    uint64_t quotient = (uint64_t)(((__uint128_t)value * pool->total_reciprocal) >> 64);
    uint64_t remainder = value - quotient * pool->total;
    if (remainder >= pool->total) remainder -= pool->total;
    move_pool_decode(pool, remainder, p, q);
#endif
    return 1;
}

/* Must run before signs change. Each bucket loses one position and gains
 * the other; its count stays fixed. The inverse slots allow O(1) updates. */
static void move_pool_swap(MovePool *pool, const signed char *s, int p, int q)
{
    int bp = 2 * (p & 1) + (s[p] < 0), bq = 2 * (q & 1) + (s[q] < 0);
    int ip = pool->slot[p], iq = pool->slot[q];
    pool->positions[bp * pool->L + ip] = q;
    pool->positions[bq * pool->L + iq] = p;
    pool->slot[p] = iq;
    pool->slot[q] = ip;
}

static int choose_first_opposite_pair(const signed char *s, int L,
                                      int *p, int *q)
{
    for (int i = 0; i < L; ++i)
        for (int j = i + 1; j < L; ++j)
            if (((i ^ j) & 1) == 0 && s[i] != s[j]) {
                *p = i; *q = j; return 1;
            }
    return 0;
}

static int choose_ordered_opposite_pair(const signed char *s, int L,
                                        unsigned long long *cursor,
                                        int *p, int *q)
{
    unsigned long long total = (unsigned long long)L * (unsigned long long)L;
    unsigned long long start = (*cursor)++ % total;
    for (unsigned long long z = 0; z < total; ++z) {
        unsigned long long code = (start + z) % total;
        int i = (int)(code / (unsigned long long)L);
        int j = (int)(code % (unsigned long long)L);
        if (i != j && ((i ^ j) & 1) == 0 && s[i] != s[j]) {
            *p = i; *q = j; return 1;
        }
    }
    return 0;
}

static int choose_guided_pair(const signed char *s, const int *own_corr,
                              const int *other_corr, int L, int k,
                              int side_sign, int *p, int *q)
{
    int u_best = 1, err_best = -1;
    for (int u = 1; u <= L / 2; ++u) {
        int target = (u == k) ? 4 * side_sign : 0;
        int e = abs(own_corr[u] + other_corr[u] - target);
        if (e > err_best) { err_best = e; u_best = u; }
    }
    /* Preserve DC/Nyquist by using same-parity positions, while choosing a
     * separation close to the largest PACF error. */
    for (int d = 0; d <= L / 2; ++d) {
        int u = (u_best + d) % L;
        for (int i = 0; i < L; ++i) {
            int j = (i + u) % L;
            if (i != j && ((i ^ j) & 1) == 0 && s[i] != s[j]) {
                *p = i; *q = j; return 1;
            }
        }
    }
    return 0;
}

static int verify_solution(const signed char *a, const signed char *b, int L,
                           int k, int side_sign)
{
    int *ca = malloc((size_t)L * sizeof(*ca));
    int *cb = malloc((size_t)L * sizeof(*cb));
    if (!ca || !cb) exit(EXIT_FAILURE);
    compute_pacf(a, L, ca);
    compute_pacf(b, L, cb);
    int ok = energy(ca, cb, L, k, side_sign) == 0;
    free(ca);
    free(cb);
    return ok;
}

static void print_candidate(const signed char *a, const signed char *b, int L,
                            long long e, int k, int sign, unsigned long long n)
{
    pthread_mutex_lock(&g_output_lock);
    printf("candidate restart=%llu k=%d sign=%d energy=%lld\n", n, k, sign, e);
    printf("a="); print_bits(stdout, a, L);
    printf("b="); print_bits(stdout, b, L);
    fflush(stdout);
    pthread_mutex_unlock(&g_output_lock);
}

static void print_machine_pair(const char *kind, const signed char *a,
                               const signed char *b, int L, int k,
                               int side_sign, long long energy)
{
    /* Cheap exact metadata lets the parent reject already-ineligible elites
     * before doing Python O(L^2) work. Successful handoffs are independently
     * recomputed by Python; this never changes the C search decisions. */
    long long score = -1;
    if (strcmp(kind, "PQCP_ELITE") == 0) {
        int ca[L], cb[L], count = 0;
        compute_pacf(a, L, ca);
        compute_pacf(b, L, cb);
        score = 0;
        for (int u = 1; u <= L / 2; ++u) {
            int magnitude = abs(ca[u] + cb[u]);
            int distance = abs(magnitude - 4);
            if (magnitude < distance) distance = magnitude;
            int multiplicity = u == L - u ? 1 : 2;
            score += (long long)multiplicity * distance;
            if (magnitude) count += multiplicity;
        }
        score += abs(count - 2);
    }
    pthread_mutex_lock(&g_output_lock);
    printf("%s L=%d k=%d sign=%d energy=%lld",
           kind, L, k, side_sign, energy);
    if (score >= 0) printf(" score=%lld", score);
    printf(" a=");
    for (int i = 0; i < L; ++i) fputc(a[i] == 1 ? '0' : '1', stdout);
    printf(" b=");
    for (int i = 0; i < L; ++i) fputc(b[i] == 1 ? '0' : '1', stdout);
    fputc('\n', stdout);
    fflush(stdout);
    pthread_mutex_unlock(&g_output_lock);
}

static void publish_answer(const signed char *a, const signed char *b, int L,
                           int k, int side_sign)
{
    pthread_mutex_lock(&g_answer_lock);
    if (verify_solution(a, b, L, k, side_sign)) {
        memcpy(g_answer_a, a, (size_t)L);
        memcpy(g_answer_b, b, (size_t)L);
        g_answer_k = k;
        g_answer_side_sign = side_sign;
        atomic_fetch_add(&g_found, 1);
        atomic_fetch_add(&g_valid_results, 1);
        /* The Python parent treats this line as an untrusted candidate event.
         * It recomputes the full profile with correlation.py, calls the
         * independent verifier, performs A/B-swap deduplication, and is the
         * only process allowed to update the official L.txt. */
        print_machine_pair("PQCP_CANDIDATE", a, b, L, k, side_sign, 0);
    }
    pthread_mutex_unlock(&g_answer_lock);
}

/* Optional end-of-restart steepest descent from the saved local best.
 * Every proposed move uses the existing opposite-sign/same-parity swap.
 * Only strictly smaller EXACT multiscale energy is accepted, at most eight
 * moves. No random numbers are consumed and no objective is changed. */
static long long quench_best(signed char *a, signed char *b, int L,
                            int k, int side_sign, long long best)
{
    if (best <= 0 || best > 256) return best;
    int ca[L], cb[L], delta[L], selected[L];
    compute_pacf(a, L, ca);
    compute_pacf(b, L, cb);
    unsigned long long evaluations = 0, moves = 0;
    for (int step = 0; step < 8 && !g_stop; ++step) {
        long long next = best;
        int picked = -1, p = 0, q = 0;
        for (int which = 0; which < 2; ++which) {
            signed char *s = which ? b : a;
            int *own = which ? cb : ca, *other = which ? ca : cb;
            for (int i = 0; i < L; ++i) {
                for (int j = i + 2; j < L; j += 2) {
                    if (s[i] == s[j]) continue;
                    long long candidate = energy_after_swap_collect(
                        best, s, own, other, L, i, j, k, side_sign, delta);
                    ++evaluations;
                    if (candidate < next) {
                        next = candidate; picked = which; p = i; q = j;
                        memcpy(selected + 1, delta + 1, (size_t)(L / 2) * sizeof(int));
                    }
                }
            }
        }
        if (picked < 0) break;
        signed char *s = picked ? b : a;
        int *corr = picked ? cb : ca;
        /* Keep the reference build usable for this experimental option. */
#ifdef PQCP_REFERENCE_KERNEL
        swap_and_update(s, L, p, q, corr, delta);
#else
        s[p] = -s[p]; s[q] = -s[q];
        for (int u = 1; u <= L / 2; ++u) corr[u] += selected[u];
#endif
        ++moves;
        best = next;
        if (best == 0) {
            publish_answer(a, b, L, k, side_sign);
            break;
        }
    }
    atomic_fetch_add(&g_moves, moves);
    atomic_fetch_add(&g_swap_evaluations, evaluations);
    return best;
}

static void *search_worker(void *opaque)
{
    WorkerArgs *w = opaque;
    const int L = w->L;
    signed char *a = malloc((size_t)L);
    signed char *b = malloc((size_t)L);
    int *ca = malloc((size_t)L * sizeof(*ca));
    int *cb = malloc((size_t)L * sizeof(*cb));
    int *scratch = malloc((size_t)L * sizeof(*scratch));
    int *best_deltas = malloc((size_t)L * sizeof(*best_deltas));
    signed char *best_a = malloc((size_t)L);
    signed char *best_b = malloc((size_t)L);
    MovePool pool_a = {0}, pool_b = {0};
    if (g_move_pool) {
        pool_a.positions = malloc(4 * (size_t)L * sizeof(int));
        pool_b.positions = malloc(4 * (size_t)L * sizeof(int));
        pool_a.slot = malloc((size_t)L * sizeof(int));
        pool_b.slot = malloc((size_t)L * sizeof(int));
        if (!pool_a.positions || !pool_b.positions || !pool_a.slot || !pool_b.slot)
            exit(EXIT_FAILURE);
    }
    if (!a || !b || !ca || !cb || !scratch || !best_deltas || !best_a || !best_b)
        exit(EXIT_FAILURE);

    /* Reproducible streams make search changes measurable.  Set
     * PQCP_SEED to try a different deterministic run. */
    uint64_t master_seed = UINT64_C(0x243f6a8885a308d3);
    const char *seed_text = getenv("PQCP_SEED");
    if (seed_text && *seed_text)
        master_seed = strtoull(seed_text, NULL, 0);
    uint64_t rng = master_seed ^
                   ((uint64_t)w->thread_id + 1) * UINT64_C(0x9e3779b97f4a7c15);
    const int deterministic_hill = getenv("PQCP_DETERMINISTIC") != NULL;
    const int deterministic_kick = getenv("PQCP_DETERMINISTIC_KICK") != NULL;
    const int ordered_moves = getenv("PQCP_ORDERED_MOVES") != NULL;
    const int guided_moves = getenv("PQCP_GUIDED_MOVES") != NULL;
    unsigned long long pair_cursor = 0;
    unsigned long long pending_moves = 0, pending_evaluations = 0;

    while (!g_stop) {
        unsigned long long restart;
        if (!claim_restart(&restart)) break;
        /* Low-overhead progress reporting: avoid printing every candidate,
         * but make the current search count visible during long runs. */
        if ((restart % 100ULL) == 0) {
            pthread_mutex_lock(&g_output_lock);
            printf("搜尋進度：已測試 %llu 組序列對\n",
                   (unsigned long long)restart);
            printf("搜尋統計：restart=%llu, moves=%llu, swap evaluations=%llu, valid results=%llu\n",
                   (unsigned long long)atomic_load(&g_restarts),
                   (unsigned long long)atomic_load(&g_moves),
                   (unsigned long long)atomic_load(&g_swap_evaluations),
                   (unsigned long long)atomic_load(&g_valid_results));
            fflush(stdout);
            pthread_mutex_unlock(&g_output_lock);
        }
        Profile p = w->profiles[(restart + w->thread_id) % w->profile_count];
        int k = w->k_values[(restart / (unsigned)w->profile_count +
                             w->thread_id) % w->k_count];

        int from_pair_seed = 0;
        if (g_pair_seed_count) {
            PairSeed *ps = &g_pair_seeds[restart % g_pair_seed_count];
            memcpy(a, ps->a, (size_t)L); memcpy(b, ps->b, (size_t)L);
            k = ps->k; p.side_sign = ps->side_sign;
            compute_pacf(a, L, ca); compute_pacf(b, L, cb);
            /* Systematic multi-radius neighborhood: cycle through one,
             * two, and three legal swaps.  The deterministic RNG stream is
             * restart-specific, so the same seed never repeats one neighbor
             * forever while remaining reproducible. */
            int perturbations = 1 + (int)(restart % 3ULL);
            for (int z = 0; z < perturbations; ++z) {
                int i, j;
                signed char *s = (z & 1) ? b : a;
                /* Deterministic per-restart pair selection prevents every
                 * known solution from revisiting the same first neighbor. */
                if (choose_opposite_pair(s, L, &rng, &i, &j)) {
                    int *cc = (s == a) ? ca : cb;
                    int scratch[L / 2 + 1];
                    swap_and_update(s, L, i, j, cc, scratch);
                }
            }
            from_pair_seed = 1;
        } else if (!random_spectral_pair(a, b, L, p, k, &rng)) {
            continue;
        }
        /* Optional FKM seed stream.  The seed must preserve the selected
         * ordinary row sum; otherwise the original random pair is retained.
         * Keeping this replacement after spectral initialization makes the
         * feature a safe acceleration layer with a deterministic fallback. */
        /* FKM seed injection is opt-in for benchmarking; the default keeps
         * the fast original random spectral initialization. */
        if (!from_pair_seed && g_fkm_seeds) {
            /* Swaps preserve both DC and Nyquist marginals.  Only accept an
             * FKM seed matching the current spectral pair; otherwise it can
             * make the target unreachable for this restart. */
            int alt = 0;
            for (int i = 0; i < L; ++i)
                alt += (i & 1) ? -a[i] : a[i];
            if (!next_fkm_seed(a, L, p.x, alt / 2)) {
                atomic_fetch_add(&g_fkm_misses, 1);
                continue;
            }
            atomic_fetch_add(&g_fkm_applied, 1);
        }
        compute_pacf(a, L, ca);
        compute_pacf(b, L, cb);
        long long e = energy(ca, cb, L, k, p.side_sign);
        if (g_move_pool) {
            move_pool_init(&pool_a, a, L);
            move_pool_init(&pool_b, b, L);
        }
        long long best = e;
        memcpy(best_a, a, (size_t)L);
        memcpy(best_b, b, (size_t)L);
        int stagnant = 0;
        const int max_moves = 20000 + 1000 * L;

        if (getenv("PQCP_VERBOSE_CANDIDATES"))
            print_candidate(a, b, L, e, k, p.side_sign, restart);
        for (int move = 0; move < max_moves && !g_stop; ++move) {
#ifdef PQCP_REFERENCE_KERNEL
            atomic_fetch_add(&g_moves, 1);
#else
            /* Observation counters do not participate in search decisions.
             * Batch per-thread updates to avoid making every evaluation
             * contend for the same cache line. Final counts remain exact. */
            if (++pending_moves >= 1024) {
                atomic_fetch_add(&g_moves, pending_moves);
                atomic_fetch_add(&g_swap_evaluations, pending_evaluations);
                pending_moves = pending_evaluations = 0;
            }
#endif
            if (e == 0) {
                publish_answer(a, b, L, k, p.side_sign);
                if (!g_pair_seed_count) break;
                /* A known-pair neighborhood can repeatedly return to the
                 * same valid state.  Continue this restart after a fixed
                 * kick so it can search beyond that basin. */
                int ki, kj;
                signed char *ks = (move & 1) ? a : b;
                int *kc = (ks == a) ? ca : cb;
                if (choose_opposite_pair(ks, L, &rng, &ki, &kj)) {
                    if (g_move_pool)
                        move_pool_swap(ks == a ? &pool_a : &pool_b, ks, ki, kj);
                    swap_and_update(ks, L, ki, kj, kc, scratch);
                }
                e = energy(ca, cb, L, k, p.side_sign);
                stagnant = 0;
                continue;
            }

            /* Sample several legal swaps and take the most promising one.
             * Four candidates gave a good speed/convergence balance for the
             * Project-2 lengths (44..94). */
            int best_which = -1, best_i = 0, best_j = 0;
            long long ne = LLONG_MAX;
            int candidate_count = g_candidate_count;
            for (int candidate = 0; candidate < candidate_count; ++candidate) {
                int which = (int)(rng_next(&rng) & 1);
                signed char *cs = which ? a : b;
                int *cc = which ? ca : cb;
                int *oc = which ? cb : ca;
                int ci, cj;
                int picked = guided_moves
                    ? choose_guided_pair(cs, cc, oc, L, k, p.side_sign, &ci, &cj)
                    : ordered_moves
                    ? choose_ordered_opposite_pair(cs, L, &pair_cursor, &ci, &cj)
                    : g_move_pool
                    ? choose_pool_pair(which ? &pool_a : &pool_b, &rng, &ci, &cj)
                    : choose_opposite_pair(cs, L, &rng, &ci, &cj);
                if (!picked) continue;
#ifdef PQCP_REFERENCE_KERNEL
                long long candidate_energy = energy_after_swap(
                    e, cs, cc, oc, L, ci, cj, k, p.side_sign);
                atomic_fetch_add(&g_swap_evaluations, 1);
#else
                long long candidate_energy = energy_after_swap_collect(
                    e, cs, cc, oc, L, ci, cj, k, p.side_sign, scratch);
                ++pending_evaluations;
#endif
                if (candidate_energy < ne) {
                    ne = candidate_energy;
                    best_which = which;
                    best_i = ci;
                    best_j = cj;
#ifndef PQCP_REFERENCE_KERNEL
                    memcpy(best_deltas + 1, scratch + 1,
                           (size_t)(L / 2) * sizeof(*best_deltas));
#endif
                }
            }
            if (best_which < 0) break;
            signed char *s = best_which ? a : b;
            int *c = best_which ? ca : cb;

            double temperature = 6.0 * (1.0 - (double)move / max_moves) + 0.15;
            int accept = ne <= e;
            if (!accept && !deterministic_hill) {
                /* Compression penalties make energy steps larger than in the
                 * single-level objective, hence the wider scale here. */
                double scaled = (double)(ne - e) / (64.0 * temperature);
                accept = scaled < 40.0 && rng_unit(&rng) < exp(-scaled);
            }

            if (accept) {
                if (g_move_pool)
                    move_pool_swap(best_which ? &pool_a : &pool_b, s, best_i, best_j);
#ifdef PQCP_REFERENCE_KERNEL
                swap_and_update(s, L, best_i, best_j, c, scratch);
#else
                s[best_i] = (signed char)-s[best_i];
                s[best_j] = (signed char)-s[best_j];
                for (int u = 1; u <= L / 2; ++u)
                    c[u] += best_deltas[u];
#endif
                e = ne;
                if (e < best) {
                    best = e;
                    memcpy(best_a, a, (size_t)L);
                    memcpy(best_b, b, (size_t)L);
                    stagnant = 0;
                } else {
                    ++stagnant;
                }
            } else {
                ++stagnant;
            }

            /* A small random kick helps escape flat local minima. */
            if (stagnant > 1000) {
                int i, j;
                for (int z = 0; z < 3; ++z) {
                    signed char *ks = (rng_next(&rng) & 1) ? a : b;
                    int *kc = (ks == a) ? ca : cb;
                    int picked = deterministic_kick
                        ? choose_first_opposite_pair(ks, L, &i, &j)
                        : choose_opposite_pair(ks, L, &rng, &i, &j);
                    if (picked) {
                        if (g_move_pool)
                            move_pool_swap(ks == a ? &pool_a : &pool_b, ks, i, j);
                        swap_and_update(ks, L, i, j, kc, scratch);
                    }
                }
                e = energy(ca, cb, L, k, p.side_sign);
                stagnant = 0;
            }
        }
#ifndef PQCP_REFERENCE_KERNEL
        atomic_fetch_add(&g_moves, pending_moves);
        atomic_fetch_add(&g_swap_evaluations, pending_evaluations);
        pending_moves = pending_evaluations = 0;
#endif
        if (g_quench)
            best = quench_best(best_a, best_b, L, k, p.side_sign, best);
        if (getenv("PQCP_EMIT_ELITES"))
            print_machine_pair("PQCP_ELITE", best_a, best_b, L, k,
                               p.side_sign, best);
    }

    free(a); free(b); free(ca); free(cb); free(scratch);
    free(best_deltas);
    free(best_a); free(best_b);
    free(pool_a.positions); free(pool_b.positions);
    free(pool_a.slot); free(pool_b.slot);
    return NULL;
}

static int build_profiles(int L, Profile **out)
{
    int capacity = 16;
    int count = 0;
    Profile *p = malloc((size_t)capacity * sizeof(*p));
    if (!p) exit(EXIT_FAILURE);

    for (int sign = -1; sign <= 1; sign += 2) {
        int n = L / 2 + 2 * sign;
        if (n < 0) continue;
        for (int x = 0; x * x <= n; ++x) {
            int y2 = n - x * x;
            int y = (int)(sqrt((double)y2) + 0.5);
            if (y * y != y2 || x > y) continue;
            if (count == capacity) {
                capacity *= 2;
                p = realloc(p, (size_t)capacity * sizeof(*p));
                if (!p) exit(EXIT_FAILURE);
            }
            p[count++] = (Profile){x, y, sign};
            if (x != y) {
                if (count == capacity) {
                    capacity *= 2;
                    p = realloc(p, (size_t)capacity * sizeof(*p));
                    if (!p) exit(EXIT_FAILURE);
                }
                p[count++] = (Profile){y, x, sign};
            }
        }
    }
    *out = p;
    return count;
}

/* Decimation by a unit modulo L maps k to every shift having the same gcd
 * with L, so one representative per gcd class is enough. */
static int build_k_representatives(int L, int **out)
{
    int *values = malloc((size_t)(L / 2) * sizeof(*values));
    unsigned char *seen = calloc((size_t)L + 1, 1);
    if (!values || !seen) exit(EXIT_FAILURE);
    int count = 0;
    for (int k = 1; k < L / 2; ++k) {
        int g = gcd_int(k, L);
        if (!seen[g]) {
            seen[g] = 1;
            values[count++] = k;
        }
    }
    free(seen);
    *out = values;
    return count;
}

static void print_bits(FILE *f, const signed char *s, int L)
{
    for (int i = 0; i < L; ++i)
        fputc(s[i] == 1 ? '0' : '1', f);
    fputc('\n', f);
}

int main(int argc, char **argv)
{
    int L = 0;
    if (argc >= 2) {
        L = atoi(argv[1]);
    } else {
        printf("請輸入長度 L: ");
        fflush(stdout);
        if (scanf("%d", &L) != 1) {
            fprintf(stderr, "輸入錯誤。\n");
            return EXIT_FAILURE;
        }
    }

    if (L < 4 || (L & 1)) {
        fprintf(stderr, "本工具針對 Project 2 第三題的偶數長度 L >= 4。\n");
        return EXIT_FAILURE;
    }
    g_length_reciprocal = UINT64_MAX / (uint64_t)L;
    g_quench = getenv("PQCP_QUENCH") != NULL;
    g_move_pool = getenv("PQCP_MOVE_POOL") != NULL;
    g_quarter_pruning = getenv("PQCP_QUARTER_PRUNING") != NULL;

    Profile *profiles = NULL;
    int profile_count = build_profiles(L, &profiles);
    if (profile_count == 0) {
        printf("不存在符合題目條件的 (%d,4)-PQCP。\n", L);
        printf("理由：L/2-2 與 L/2+2 都無法表示為兩個整數平方和。\n");
        free(profiles);
        return EXIT_SUCCESS;
    }

    int *k_values = NULL;
    int k_count = build_k_representatives(L, &k_values);
    if (k_count == 0) {
        printf("不存在兩個不同的非零位移位置。\n");
        free(profiles); free(k_values);
        return EXIT_SUCCESS;
    }

    long cpu_count = sysconf(_SC_NPROCESSORS_ONLN);
    int thread_count = cpu_count > 0 ? (int)cpu_count : 1;
    const char *env_threads = getenv("PQCP_THREADS");
    if (env_threads && atoi(env_threads) > 0) thread_count = atoi(env_threads);
    if (thread_count > 64) thread_count = 64;
    const char *env_candidates = getenv("PQCP_CANDIDATES");
    if (env_candidates && atoi(env_candidates) > 0)
        g_candidate_count = atoi(env_candidates);
    if (g_candidate_count > 16) g_candidate_count = 16;
    const char *restart_limit = getenv("PQCP_MAX_RESTARTS");
    if (restart_limit && *restart_limit)
        g_restart_limit = strtoull(restart_limit, NULL, 10);
    const char *seed_file = getenv("PQCP_FKM_SEEDS");
    char automatic_seed_file[64];
    if (!seed_file || !*seed_file) {
        snprintf(automatic_seed_file, sizeof(automatic_seed_file),
                 "%d_fkm_seeds.txt", L);
        seed_file = automatic_seed_file;
    }
    load_fkm_seeds(seed_file, L);
    const char *pair_seed_file = getenv("PQCP_PAIR_SEEDS");
    if (pair_seed_file && *pair_seed_file)
        load_pair_seeds(pair_seed_file, L);

    g_answer_a = malloc((size_t)L);
    g_answer_b = malloc((size_t)L);
    pthread_t *threads = malloc((size_t)thread_count * sizeof(*threads));
    WorkerArgs *args = malloc((size_t)thread_count * sizeof(*args));
    if (!g_answer_a || !g_answer_b || !threads || !args) exit(EXIT_FAILURE);

    signal(SIGINT, on_sigint);
    printf("必要條件通過；使用 %d 個執行緒開始搜尋 L=%d。\n",
           thread_count, L);
    printf("壓縮引導：factor 2（%d -> %d）", L, L / 2);
    if (L % 4 == 0)
        printf("，factor 4 多層路徑（%d -> %d -> %d）",
               L, L / 2, L / 4);
    printf("。\n");
    printf("尚未找到不代表不存在；可按 Ctrl-C 停止。\n");

    for (int i = 0; i < thread_count; ++i) {
        args[i] = (WorkerArgs){L, profile_count, profiles, k_count, k_values,
                               (unsigned)i};
        if (pthread_create(&threads[i], NULL, search_worker, &args[i]) != 0) {
            perror("pthread_create");
            g_stop = 1;
            thread_count = i;
            break;
        }
    }
    for (int i = 0; i < thread_count; ++i)
        pthread_join(threads[i], NULL);

    if (atomic_load(&g_found)) {
        printf("找到 (%d,4)-PQCP！非零位移為 %d 與 %d，數值為 %d。\n",
               L, g_answer_k, L - g_answer_k, 4 * g_answer_side_sign);
        printf("a = "); print_bits(stdout, g_answer_a, L);
        printf("b = "); print_bits(stdout, g_answer_b, L);
        printf("候選已交由 Python 獨立 verifier 處理。\n");
    } else {
        printf("\n搜尋已停止，共執行 %llu 次 restart；目前無法判定是否存在。\n",
               (unsigned long long)atomic_load(&g_restarts));
    }
    printf("搜尋統計：restart=%llu, moves=%llu, swap evaluations=%llu, valid results=%llu\n",
           (unsigned long long)atomic_load(&g_restarts),
           (unsigned long long)atomic_load(&g_moves),
           (unsigned long long)atomic_load(&g_swap_evaluations),
           (unsigned long long)atomic_load(&g_valid_results));
    printf("FKM seed statistics: applied=%llu, misses=%llu\n",
           (unsigned long long)atomic_load(&g_fkm_applied),
           (unsigned long long)atomic_load(&g_fkm_misses));
    fflush(stdout);

    free(profiles); free(k_values); free(g_answer_a); free(g_answer_b);
    for (size_t i = 0; i < g_fkm_seed_count; ++i)
        free(g_fkm_seeds[i].values);
    free(g_fkm_seeds);
    for (size_t i = 0; i < g_pair_seed_count; ++i) {
        free(g_pair_seeds[i].a); free(g_pair_seeds[i].b);
    }
    free(g_pair_seeds);
    free(threads); free(args);
    return EXIT_SUCCESS;
}
