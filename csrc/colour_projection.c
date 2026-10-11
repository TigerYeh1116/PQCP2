/* Assignment projection, with dual variables for exact independent checking. */
#include <float.h>
#include <math.h>
#include <stddef.h>

int pqcp_colour_assignment(const double *c, const double *g, int n, const int *counts,
                          int *assignment, double *row_dual, double *slot_dual) {
    if (n<1 || n>47) return -1;
    int labels[48], p[48]={0}, way[48]={0}, total=0;
    double u[48]={0}, v[48]={0};
    for (int color=0;color<4;color++) {
        if (counts[color]<0 || counts[color]>n) return -1;
        for (int j=0;j<counts[color];j++) { if (total>=n) return -1; labels[++total]=color; }
    }
    if (total!=n) return -1;
    for (int i=0;i<n;i++) if (!isfinite(c[i]) || !isfinite(g[i])) return -1;
    for (int i=1;i<=n;i++) {
        p[0]=i;
        int j0=0, used[48]={0};
        double minv[48];
        for (int j=0;j<=n;j++) minv[j]=DBL_MAX;
        do {
            used[j0]=1;
            int i0=p[j0], j1=0;
            double delta=DBL_MAX;
            for (int j=1;j<=n;j++) if (!used[j]) {
                int color=labels[j];
                double score=color==0 ? c[i0-1] : color==1 ? -c[i0-1] : color==2 ? g[i0-1] : -g[i0-1];
                double current=-score-u[i0]-v[j];
                if (current<minv[j]) { minv[j]=current; way[j]=j0; }
                if (minv[j]<delta) { delta=minv[j]; j1=j; }
            }
            if (j1==0 || !isfinite(delta)) return -2;
            for (int j=0;j<=n;j++) {
                if (used[j]) { u[p[j]]+=delta; v[j]-=delta; }
                else minv[j]-=delta;
            }
            j0=j1;
        } while (p[j0]!=0);
        do { int j1=way[j0]; p[j0]=p[j1]; j0=j1; } while (j0!=0);
    }
    for (int j=1;j<=n;j++) assignment[p[j]-1]=j-1;
    for (int i=0;i<n;i++) { row_dual[i]=u[i+1]; slot_dual[i]=v[i+1]; }
    return 0;
}
