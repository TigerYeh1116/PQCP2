/* Exhaustive <=2 swaps over two ternary words: exactly their fixed-content
 * Hamming ball of radius 4.  No floating point and no heuristic pruning. */
#include <stdint.h>
#include <string.h>

static int delta(const int *w, int d, int p, int q, int u) {
    int t = w[q] - w[p];
    int linear = t * (w[(p+u)%d] + w[(p-u+d)%d]
                     - w[(q+u)%d] - w[(q-u+d)%d]);
    int edges = ((p+u)%d == q) + ((q+u)%d == p);
    return linear - t*t*edges; /* u>0 */
}

static void swap(int *w, int p, int q) {
    int t=w[p]; w[p]=w[q]; w[q]=t;
}

/* 1=found, 0=complete local UNSAT, -1=budget exhausted, -2=invalid input. */
int pqcp_ternary_repair(const int *input, const int *wanted, int d,
                        uint64_t limit, int *output, uint64_t *evaluations) {
    if (d<3 || d>47 || !(d&1)) return -2;
    int current[94], residual[24], first_delta[24];
    memcpy(current,input,2*d*sizeof(int));
    *evaluations=0;
    int zero=1;
    for(int u=1;u<=d/2;u++) {
        int value=0;
        for(int s=0;s<2;s++) for(int i=0;i<d;i++)
            value+=current[s*d+i]*current[s*d+(i+u)%d];
        residual[u]=value-wanted[u];
        if(residual[u]) zero=0;
    }
    if(zero) {memcpy(output,current,2*d*sizeof(int)); return 1;}
    for(int s=0;s<2;s++) for(int p=0;p<d;p++) for(int q=p+1;q<d;q++) {
        int *w=current+s*d;
        if(w[p]==w[q]) continue;
        if(limit && *evaluations>=limit) return -1;
        ++*evaluations;
        zero=1;
        for(int u=1;u<=d/2;u++) {
            first_delta[u]=delta(w,d,p,q,u);
            if(residual[u]+first_delta[u]) zero=0;
        }
        swap(w,p,q);
        if(zero) {memcpy(output,current,2*d*sizeof(int)); return 1;}
        for(int t=0;t<2;t++) for(int r=0;r<d;r++) for(int v=r+1;v<d;v++) {
            int *other=current+t*d;
            if(other[r]==other[v]) continue;
            if(limit && *evaluations>=limit) return -1;
            ++*evaluations;
            zero=1;
            for(int u=1;u<=d/2;u++) {
                if(residual[u]+first_delta[u]+delta(other,d,r,v,u)) {zero=0;break;}
            }
            if(zero) {
                swap(other,r,v);
                memcpy(output,current,2*d*sizeof(int));
                return 1;
            }
        }
        swap(w,p,q);
    }
    return 0;
}
