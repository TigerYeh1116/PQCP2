/* Exhaust all single-row four-bit changes preserving both parity contents.
 * Combined with the Python two-row (2+2) join, this covers the full radius-4
 * fixed-content Hamming ball. Integer correlation failures prune only the
 * current literal candidate. No floating point or probabilistic hashes. */
#include <stdint.h>
#include <string.h>

typedef unsigned __int128 word;

static int weight(word x) {
    return __builtin_popcountll((uint64_t)x)+__builtin_popcountll((uint64_t)(x>>64));
}

int pqcp_binary_profile128(uint64_t lo, uint64_t hi, int length, int *profile) {
    if (length < 4 || length > 94 || (length&1)) return -1;
    word candidate = ((word)hi<<64)|lo;
    if (candidate>>length) return -1;
    word all = (((word)1)<<length)-1;
    profile[0] = length;
    for (int u = 1; u < length; ++u) {
        word rotated = ((candidate<<u)|(candidate>>(length-u)))&all;
        profile[u] = length-2*weight(candidate^rotated);
    }
    return 0;
}

static int check(word candidate, int length, const int *other, uint64_t positive,
    uint64_t negative, uint64_t *output, uint64_t *stats) {
    word all = (((word)1)<<length)-1;
    int found = 0, k = 0, eta = 0;
    ++stats[0];
    for (int u = 1; u <= length/2; ++u) {
        word rotated = ((candidate<<u)|(candidate>>(length-u)))&all;
        int value = other[u]+length-2*weight(candidate^rotated);
        ++stats[1];
        if (!value) continue;
        if (found || u == length/2 || (value != 4 && value != -4)) return 0;
        if (!(((value == 4 ? positive : negative)>>u)&1)) return 0;
        found = 1; k = u; eta = value/4;
    }
    if (!found) return 0;
    output[0] = (uint64_t)candidate;
    output[1] = (uint64_t)(candidate>>64);
    output[2] = (uint64_t)k;
    output[3] = eta == 1 ? 1 : 0;
    return 1;
}

int pqcp_row_four_flips(uint64_t lo, uint64_t hi, int length, const int *other,
    uint64_t positive, uint64_t negative, uint64_t *output, uint64_t *stats) {
    if (length < 4 || length > 94 || (length&1)) return -1;
    word original = ((word)hi<<64)|lo;
    if (original>>length) return -1;
    int indices[2][2][47], counts[2][2] = {{0,0},{0,0}};
    memset(stats, 0, 2*sizeof(uint64_t));
    for (int i = 0; i < length; ++i) {
        int parity = i&1, bit = (int)((original>>i)&1);
        indices[parity][bit][counts[parity][bit]++] = i;
    }
    for (int parity = 0; parity < 2; ++parity)
        for (int a = 0; a < counts[parity][0]; ++a)
            for (int b = a+1; b < counts[parity][0]; ++b)
                for (int c = 0; c < counts[parity][1]; ++c)
                    for (int d = c+1; d < counts[parity][1]; ++d) {
                        word changed = original;
                        changed ^= (word)1<<indices[parity][0][a];
                        changed ^= (word)1<<indices[parity][0][b];
                        changed ^= (word)1<<indices[parity][1][c];
                        changed ^= (word)1<<indices[parity][1][d];
                        if (check(changed,length,other,positive,negative,output,stats)) return 1;
                    }
    for (int a = 0; a < counts[0][0]; ++a)
        for (int b = 0; b < counts[0][1]; ++b)
            for (int c = 0; c < counts[1][0]; ++c)
                for (int d = 0; d < counts[1][1]; ++d) {
                    word changed = original;
                    changed ^= (word)1<<indices[0][0][a];
                    changed ^= (word)1<<indices[0][1][b];
                    changed ^= (word)1<<indices[1][0][c];
                    changed ^= (word)1<<indices[1][1][d];
                    if (check(changed,length,other,positive,negative,output,stats)) return 1;
                }
    return 0;
}

struct batch_context {
    uint64_t skip, visited;
    int length, capacity, collected;
    const int *base;
    uint64_t *masks;
    unsigned char *vectors;
};

static int batch_visit(word candidate, struct batch_context *ctx) {
    if (ctx->visited++ < ctx->skip) return 0;
    word all = (((word)1)<<ctx->length)-1;
    int width = ctx->length/2;
    for (int u = 1; u <= width; ++u) {
        word rotated = ((candidate<<u)|(candidate>>(ctx->length-u)))&all;
        int delta = ctx->length-2*weight(candidate^rotated)-ctx->base[u];
        if (delta%4 || delta/4 < -128 || delta/4 > 127) return -2;
        ctx->vectors[ctx->collected*width+u-1] = (unsigned char)(128+delta/4);
    }
    ctx->masks[2*ctx->collected] = (uint64_t)candidate;
    ctx->masks[2*ctx->collected+1] = (uint64_t)(candidate>>64);
    ++ctx->collected;
    return ctx->collected == ctx->capacity;
}

/* Stream exact 43-byte signed-delta vectors for ALL literal row-four flips.
 * Skip is an enumeration cursor, not a cutoff or a symmetry quotient. */
int pqcp_row_four_vectors(uint64_t lo, uint64_t hi, int length, const int *base,
    uint64_t skip, int capacity, uint64_t *masks, unsigned char *vectors,
    uint64_t *stats) {
    if (length < 4 || length > 94 || (length&1) || capacity < 1) return -1;
    word original = ((word)hi<<64)|lo;
    if (original>>length) return -1;
    int indices[2][2][47], counts[2][2] = {{0,0},{0,0}};
    struct batch_context ctx = {skip,0,length,capacity,0,base,masks,vectors};
    for (int i = 0; i < length; ++i) {
        int parity = i&1, bit = (int)((original>>i)&1);
        indices[parity][bit][counts[parity][bit]++] = i;
    }
    int status = 0;
    for (int parity = 0; parity < 2; ++parity)
        for (int a = 0; a < counts[parity][0]; ++a)
            for (int b = a+1; b < counts[parity][0]; ++b)
                for (int c = 0; c < counts[parity][1]; ++c)
                    for (int d = c+1; d < counts[parity][1]; ++d) {
                        word changed = original;
                        changed ^= (word)1<<indices[parity][0][a];
                        changed ^= (word)1<<indices[parity][0][b];
                        changed ^= (word)1<<indices[parity][1][c];
                        changed ^= (word)1<<indices[parity][1][d];
                        status = batch_visit(changed,&ctx);
                        if (status) goto done;
                    }
    for (int a = 0; a < counts[0][0]; ++a)
        for (int b = 0; b < counts[0][1]; ++b)
            for (int c = 0; c < counts[1][0]; ++c)
                for (int d = 0; d < counts[1][1]; ++d) {
                    word changed = original;
                    changed ^= (word)1<<indices[0][0][a];
                    changed ^= (word)1<<indices[0][1][b];
                    changed ^= (word)1<<indices[1][0][c];
                    changed ^= (word)1<<indices[1][1][d];
                    status = batch_visit(changed,&ctx);
                    if (status) goto done;
                }
done:
    stats[0] = (uint64_t)ctx.collected;
    stats[1] = ctx.visited;
    return status;
}
