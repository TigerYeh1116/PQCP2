/* Exact integer energy tie selection inside minimum-Hamming pair-type moves. */
#include <stdint.h>
#include <string.h>

static int even_endpoint(int i, int d) { return i%2 ? i+d : i; }
static int pair_type(const int *row, int i, int d) {
    int e=even_endpoint(i,d), o=e==i ? i+d : i;
    return 2*row[e]+row[o];
}
static void move_delta(const int *row, int L, int p, int q, int *delta) {
    int xp=1-2*row[p], xq=1-2*row[q];
    delta[0]=0;
    for (int u=1;u<L;u++) {
        delta[u]=-2*xp*((1-2*row[(p+u)%L])+(1-2*row[(p+L-u)%L]))
                 -2*xq*((1-2*row[(q+u)%L])+(1-2*row[(q+L-u)%L]));
        if ((p+u)%L==q) delta[u]+=4*xp*xq;
        if ((q+u)%L==p) delta[u]+=4*xp*xq;
    }
}
static int64_t energy(const int *residual, const int *delta, int L, int compression) {
    int d=L/2;
    int64_t value=0;
    for (int u=1;u<=d;u++) {
        int64_t x=residual[u]+(delta ? delta[u] : 0);
        value+=x*x;
    }
    for (int u=0;compression && u<d;u++) {
        int64_t x=residual[u]+residual[u+d]+(delta ? delta[u]+delta[u+d] : 0);
        value+=2*x*x;
    }
    return value;
}

int pqcp_half_shift_project(const int *input, int L, int k, int eta,
                           int negative_a, int negative_b, int reverse_order,
                           int compression,
                           int *output, int64_t *final_energy) {
    if (L<6 || L>94 || L%4!=2 || k<1 || k>=L/2 || (eta!=1 && eta!=-1)
        || reverse_order<0 || reverse_order>1 || compression<0 || compression>1) return -1;
    for (int i=0;i<2*L;i++) if (input[i]!=0 && input[i]!=1) return -1;
    int d=L/2, residual[94], delta[94], best_delta[94];
    int desired[2]={negative_a,negative_b};
    memcpy(output,input,(unsigned)(2*L)*sizeof(int));
    for (int side=0;side<2;side++) {
        int e=0,o=0;
        for (int i=0;i<L;i++) { if (i%2) o+=input[side*L+i]; else e+=input[side*L+i]; }
        if (desired[side]<0 || desired[side]>e || desired[side]>o || desired[side]<e+o-d) return -1;
    }
    for (int u=0;u<L;u++) {
        residual[u]=0;
        for (int side=0;side<2;side++) for (int i=0;i<L;i++)
            residual[u]+=(1-2*input[side*L+i])*(1-2*input[side*L+(i+u)%L]);
        residual[u]-=u==0 ? 2*L : (u==k || u==L-k) ? 4*eta : 0;
    }
    for (int phase=0;phase<2;phase++) {
        int side=reverse_order ? 1-phase : phase, *row=output+side*L;
        int current=0;
        for (int i=0;i<d;i++) current+=pair_type(row,i,d)==3;
        while (current!=desired[side]) {
            int increase=current<desired[side], best_p=-1,best_q=-1;
            int64_t best_energy=INT64_MAX;
            for (int i=0;i<d;i++) if (pair_type(row,i,d)==(increase ? 2 : 3))
                for (int j=0;j<d;j++) if (pair_type(row,j,d)==(increase ? 1 : 0))
                    for (int parity=0;parity<2;parity++) {
                        int ei=even_endpoint(i,d), ej=even_endpoint(j,d);
                        int oi=ei==i ? i+d : i, oj=ej==j ? j+d : j;
                        int p=parity==0 ? ei : increase ? oj : oi;
                        int q=parity==0 ? ej : increase ? oi : oj;
                        if (row[p]!=1 || row[q]!=0) return -2;
                        move_delta(row,L,p,q,delta);
                        int64_t candidate=energy(residual,delta,L,compression);
                        if (candidate<best_energy) {
                            best_energy=candidate; best_p=p; best_q=q;
                            memcpy(best_delta,delta,(unsigned)L*sizeof(int));
                        }
                    }
            if (best_p<0) return -2;
            row[best_p]=0; row[best_q]=1;
            for (int u=0;u<L;u++) residual[u]+=best_delta[u];
            current+=increase ? 1 : -1;
        }
    }
    *final_energy=energy(residual,0,L,compression);
    return 0;
}
