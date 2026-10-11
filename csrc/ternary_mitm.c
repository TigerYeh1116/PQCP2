/* Exact N2(a) x N2(b) PACF matching. All equal signatures are visited.
 * Output cap stops only on an additional distinct pair; never reports UNSAT.
 * No globals: separate calls may run concurrently. */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct { int16_t v[23]; uint8_t p,q,r,s; } Entry;
typedef struct {
    int d,h,cap,count,side; size_t n,capacity; uint64_t probes;
    const int *input,*wanted; int *out; Entry *table;
} Context;

static void swap(int *w,int p,int q) {int t=w[p];w[p]=w[q];w[q]=t;}
static void delta(const int *w,int d,int p,int q,int *v) {
    int t=w[q]-w[p];
    for(int u=1;u<=d/2;u++)
        v[u-1]=t*(w[(p+u)%d]+w[(p-u+d)%d]-w[(q+u)%d]-w[(q-u+d)%d])
          -t*t*(((p+u)%d==q)+((q+u)%d==p));
}
static int compare(const void *aa,const void *bb) {
    const Entry *a=aa,*b=bb;
    for(int u=0;u<23;u++) {if(a->v[u]<b->v[u])return -1;if(a->v[u]>b->v[u])return 1;}
    return 0;
}
static int visit(Context *c,const int *w,const int *profile,int p,int q,int r,int s) {
    Entry e; memset(&e,0,sizeof(e));
    for(int u=0;u<c->h;u++)e.v[u]=(int16_t)(c->side?c->wanted[u+1]-profile[u]:profile[u]);
    if(!c->side) {
        if(c->n>=c->capacity)return -2;
        e.p=(uint8_t)p;e.q=(uint8_t)q;e.r=(uint8_t)r;e.s=(uint8_t)s;
        c->table[c->n++]=e; return 0;
    }
    c->probes++;
    size_t lo=0,hi=c->n;
    while(lo<hi){size_t mid=lo+(hi-lo)/2;if(compare(&c->table[mid],&e)<0)lo=mid+1;else hi=mid;}
    for(size_t i=lo;i<c->n && compare(&c->table[i],&e)==0;i++) {
        int pair[94]; memcpy(pair,c->input,c->d*sizeof(int));
        const Entry *a=&c->table[i];
        swap(pair,a->p,a->q);swap(pair,a->r,a->s);
        memcpy(pair+c->d,w,c->d*sizeof(int));
        int duplicate=0;
        for(int j=0;j<c->count;j++)if(!memcmp(pair,c->out+j*2*c->d,2*c->d*sizeof(int))){duplicate=1;break;}
        if(duplicate)continue;
        if(c->count==c->cap)return 1; /* incomplete enumeration */
        memcpy(c->out+c->count*2*c->d,pair,2*c->d*sizeof(int));c->count++;
    }
    return 0;
}
static int generate(Context *c,const int *input) {
    int w[47],base[23],first[23],second[23],change[23];
    memcpy(w,input,c->d*sizeof(int));
    for(int u=1;u<=c->h;u++){base[u-1]=0;for(int i=0;i<c->d;i++)base[u-1]+=w[i]*w[(i+u)%c->d];}
    int status=visit(c,w,base,0,0,0,0);if(status)return status;
    for(int p=0;p<c->d;p++)for(int q=p+1;q<c->d;q++) {
        if(w[p]==w[q])continue;
        delta(w,c->d,p,q,change);
        for(int u=0;u<c->h;u++)first[u]=base[u]+change[u];
        swap(w,p,q);status=visit(c,w,first,p,q,0,0);if(status)return status;
        for(int r=0;r<c->d;r++)for(int s=r+1;s<c->d;s++) {
            if(w[r]==w[s])continue;
            delta(w,c->d,r,s,change);
            for(int u=0;u<c->h;u++)second[u]=first[u]+change[u];
            swap(w,r,s);status=visit(c,w,second,p,q,r,s);if(status)return status;swap(w,r,s);
        }
        swap(w,p,q);
    }
    return 0;
}
/* 0=complete (possibly SAT), 1=output capped, negative=error. */
int pqcp_ternary_mitm(const int *input,const int *wanted,int d,int cap,int *out,
                      int *count,uint64_t *entries,uint64_t *probes) {
    if(d<3 || d>47 || !(d&1) || cap<1)return -1;
    size_t m=0;for(int p=0;p<d;p++)for(int q=p+1;q<d;q++)if(input[p]!=input[q])m++;
    Context c={0};c.d=d;c.h=d/2;c.cap=cap;c.capacity=1+m+m*m;
    c.input=input;c.wanted=wanted;c.out=out;c.table=malloc(c.capacity*sizeof(Entry));
    if(!c.table)return -3;
    int status=generate(&c,input);
    if(!status){qsort(c.table,c.n,sizeof(Entry),compare);c.side=1;status=generate(&c,input+d);}
    *count=c.count;*entries=c.n;*probes=c.probes;free(c.table);return status;
}
