/* Exact Gray-order affine PACF scan. No RNG, floating point or pruning.
 * Four uint64 words encode (A low, A high, B low, B high), up to L=128.
 * Return the FIRST full-correlation match in [start,end), or exhaustion.
 * Python still enforces cross constraints and independently verifies SAT.
 */
#include <stdint.h>

typedef unsigned __int128 word;

static word unpack(const uint64_t *p) {
    return (word)p[0] | ((word)p[1] << 64);
}

static unsigned pop(word x) {
    return __builtin_popcountll((uint64_t)x)
         + __builtin_popcountll((uint64_t)(x >> 64));
}

int pqcp_affine_scan(int L, int free_bits, const uint64_t *base,
                    const uint64_t *deltas, const int *wanted,
                    uint64_t start, uint64_t end, uint64_t *next,
                    uint64_t *found) {
    if (L < 4 || L > 128 || L % 2 || free_bits < 0 || free_bits > 62
            || !base || !deltas || !wanted || !next || !found
            || start > end || end > (UINT64_C(1) << free_bits)) return -1;
    const word mask = L == 128 ? ~(word)0 : (((word)1 << L) - 1);
    word a = unpack(base), b = unpack(base+2);
    uint64_t gray = start ^ (start >> 1);
    for (int j=0; j<free_bits; ++j) {
        if ((gray >> j) & 1) {
            a ^= unpack(deltas+4*j);
            b ^= unpack(deltas+4*j+2);
        }
    }
    for (uint64_t index=start; index<end; ++index) {
        if (index != start) {
            unsigned j = __builtin_ctzll(index); /* index is nonzero */
            a ^= unpack(deltas+4*j);
            b ^= unpack(deltas+4*j+2);
        }
        int match=1;
        for (int u=1; u<L/2; ++u) {
            word ra = ((a >> u) | (a << (L-u))) & mask;
            word rb = ((b >> u) | (b << (L-u))) & mask;
            if ((int)(pop(a ^ ra) + pop(b ^ rb)) != wanted[u-1]) {
                match=0;
                break;
            }
        }
        if (match) {
            *next=index+1;
            found[0]=(uint64_t)a; found[1]=(uint64_t)(a >> 64);
            found[2]=(uint64_t)b; found[3]=(uint64_t)(b >> 64);
            return 1;
        }
    }
    *next=end;
    return 0;
}
