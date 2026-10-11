/* Exact GF(2^m) norm/trace completion, m <= 14. No floating point. */
#include <stdint.h>

static uint32_t multiply(uint32_t a, uint32_t b, uint32_t factor, int degree)
{
    uint32_t out = 0;
    while (b) {
        if (b & 1U) out ^= a;
        b >>= 1;
        a <<= 1;
        if (a & (1U << degree)) a ^= factor;
    }
    return out;
}

static uint32_t power(uint32_t a, uint32_t exponent, uint32_t factor, int degree)
{
    uint32_t out = 1;
    while (exponent) {
        if (exponent & 1U) out = multiply(out, a, factor, degree);
        exponent >>= 1;
        a = multiply(a, a, factor, degree);
    }
    return out;
}

static uint32_t star(uint32_t a, const uint32_t *columns)
{
    uint32_t out = 0;
    for (int i = 0; a; ++i, a >>= 1)
        if (a & 1U) out ^= columns[i];
    return out;
}

/* roots[n] is one z with z^q=z+1, z(z+1)=n, or UINT32_MAX.
   Its only other preimage is z+1. Zero partners use Python's complete
   norm fallback; -1 is an API error, never an UNSAT certificate. */
int pqcp_field_norm_trace(uint32_t n, uint32_t h, uint32_t w,
                         uint32_t factor, int degree,
                         const uint32_t *columns, const uint32_t *roots,
                         uint32_t *output)
{
    if (degree < 2 || degree > 14 || degree % 2 ||
        factor < (1U << degree) || factor >= (2U << degree) ||
        n >= (1U << degree) || h >= (1U << degree) ||
        !w || w >= (1U << degree)) return -1;
    uint32_t ws = star(w, columns), nw = multiply(w, ws, factor, degree);
    uint32_t exponent = (1U << degree)-2, q = 1U << (degree/2);
    int count;
    if (!h) {
        uint32_t ratio = multiply(n, power(nw, exponent, factor, degree), factor, degree);
        output[0] = multiply(w, power(ratio, q/2, factor, degree), factor, degree);
        count = 1;
    } else {
        uint32_t z_norm = multiply(multiply(n, nw, factor, degree),
            power(multiply(h, h, factor, degree), exponent, factor, degree), factor, degree);
        uint32_t z = roots[z_norm];
        if (z == UINT32_MAX) return 0;
        uint32_t scale = multiply(h, power(ws, exponent, factor, degree), factor, degree);
        output[0] = multiply(scale, z, factor, degree);
        output[1] = multiply(scale, z ^ 1U, factor, degree);
        count = 2;
    }
    for (int i = 0; i < count; ++i) {
        uint32_t us = star(output[i], columns);
        if (multiply(output[i], us, factor, degree) != n ||
            (multiply(w, us, factor, degree) ^ multiply(ws, output[i], factor, degree)) != h)
            return -2;
    }
    return count;
}
