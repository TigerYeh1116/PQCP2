/* Necessary p^2 divisibility of a single integer cyclic autocorrelation.
   p == -1 (mod d); each row is a real-cyclotomic Hensel-lifted root. */
#include <stdint.h>

int pqcp_prime_norm_obstruction(const int32_t *profile, int d,
    const uint32_t *primes, const uint32_t *coefficients, int rows,
    int32_t *row_out, uint64_t *residue_out)
{
    if ((d!=3 && d!=5 && d!=11 && d!=43) || rows<0) return -1;
    int width=d/2+1;*row_out=-1;*residue_out=0;
    for (int i=0;i<width;++i) if (profile[i]<-d || profile[i]>d) return -1;
    for (int row=0;row<rows;++row) {
        uint32_t p=primes[row];
        if (p>65000 || p<3 || (p+1)%d) return -1;
        int64_t modulus=(int64_t)p*p,value=0;
        for (int i=0;i<width;++i) value+=(int64_t)profile[i]*coefficients[row*width+i];
        int64_t residue=value%modulus;if (residue<0) residue+=modulus;
        if (residue && residue%p==0) {
            *row_out=row;*residue_out=(uint64_t)residue;return 0;
        }
    }
    return 0;
}
