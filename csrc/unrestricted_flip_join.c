/* Complete <=5 flips PER ROW, unrestricted parity contents and positions.
 * Delta PACF / 4 modulo 2 is a quadratic Boolean function. Exact 64-bit
 * signatures only reject necessary-condition failures; collisions retain
 * every literal preimage and undergo full integer PACF verification.
 * No probabilistic filter, approximate arithmetic or fixed-content pruning.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef unsigned __int128 word;
struct entry { uint64_t signature,lo,hi; };
struct context {
    int length,radius,side,group_content;
    word original,all,even;
    uint64_t singles[94],base_signature;
    struct entry *index;
    size_t used;
    uint64_t *stats,*output;
    int found;
};

static int weight(word x) {
    return __builtin_popcountll((uint64_t)x)+__builtin_popcountll((uint64_t)(x>>64));
}
static int profile(word x,int n,int u,word all) {
    return n-2*weight(x^(((x<<u)|(x>>(n-u)))&all));
}
static int compare_entry(const void *a,const void *b) {
    uint64_t x=((const struct entry *)a)->signature,y=((const struct entry *)b)->signature;
    return (x>y)-(x<y);
}
static int content_code(int value) {
    switch(abs(value)) {case 6:return 0;case 8:return 1;case 10:return 2;case 12:return 3;}
    return -1;
}
static int verify(struct context *ctx,word a,word b,int k) {
    int n=ctx->length;
    int sa=n-2*weight(a),sb=n-2*weight(b);
    int dc=sa*sa+sb*sb;
    if(dc!=2*n+8&&dc!=2*n-8)return 0;
    int eta=(dc-2*n)/8;
    sa=2*(weight(a&~ctx->even)-weight(a&ctx->even));
    sb=2*(weight(b&~ctx->even)-weight(b&ctx->even));
    if(sa*sa+sb*sb!=2*n+8*eta*((k&1)?-1:1))return 0;
    for(int u=1;u<=n/2;++u) {
        ++ctx->stats[6];
        if(profile(a,n,u,ctx->all)+profile(b,n,u,ctx->all)!=(u==k?4*eta:0))return 0;
    }
    ctx->output[0]=(uint64_t)a;ctx->output[1]=(uint64_t)(a>>64);
    ctx->output[2]=(uint64_t)b;ctx->output[3]=(uint64_t)(b>>64);
    ctx->output[4]=(uint64_t)k;ctx->output[5]=(eta==1);
    return 1;
}
static void visit(struct context *ctx,word changed,uint64_t signature,int dc,int alt) {
    ++ctx->stats[ctx->side];
    int dc_code=0,alt_code=0;
    if(ctx->length==86) {
        dc_code=content_code(dc);alt_code=content_code(alt);
        if(dc_code<0||alt_code<0)return;
    }
    if(!ctx->side) {
        if(ctx->group_content)signature|=(uint64_t)dc_code<<43|(uint64_t)alt_code<<45;
        ctx->index[ctx->used++]=(struct entry){signature,(uint64_t)changed,(uint64_t)(changed>>64)};
        ++ctx->stats[2];return;
    }
    ++ctx->stats[3];
    int first=1,step=1;
    uint64_t required_content=0;
    if(ctx->group_content) {
        /* DC determines eta, ALT determines eta*(-1)^k. Complementary
         * absolute contents are 6<->12 and 8<->10, i.e. code -> 3-code.
         * Opposite square-sum signs require odd k, equal signs even k. */
        int dc_plus=(dc_code==0||dc_code==3),alt_plus=(alt_code==0||alt_code==3);
        first=dc_plus==alt_plus?2:1;step=2;
        required_content=(uint64_t)(3-dc_code)<<43|(uint64_t)(3-alt_code)<<45;
    }
    for(int k=first;k<ctx->length/2&&!ctx->found;k+=step) {
        uint64_t wanted=ctx->base_signature^signature^((uint64_t)1<<(k-1));
        wanted|=required_content;
        ++ctx->stats[4];
        size_t lo=0,hi=ctx->used;
        while(lo<hi) {
            size_t mid=lo+(hi-lo)/2;
            if(ctx->index[mid].signature<wanted)lo=mid+1;else hi=mid;
        }
        for(size_t i=lo;i<ctx->used&&ctx->index[i].signature==wanted;++i) {
            ++ctx->stats[5];
            word a=((word)ctx->index[i].hi<<64)|ctx->index[i].lo;
            if(verify(ctx,a,changed,k)){ctx->found=1;return;}
        }
    }
}
static void enumerate(struct context *ctx,int start,int depth,int *chosen,word changed,uint64_t signature,int dc,int alt) {
    visit(ctx,changed,signature,dc,alt);
    if(depth==ctx->radius||ctx->found)return;
    for(int i=start;i<ctx->length&&!ctx->found;++i) {
        uint64_t next=signature^ctx->singles[i];
        for(int j=0;j<depth;++j) {
            int distance=i-chosen[j];
            if(distance>ctx->length/2)distance=ctx->length-distance;
            /* Midpoint has two oriented edges, so its quadratic term is even. */
            if(distance!=ctx->length/2)next^=(uint64_t)1<<(distance-1);
        }
        chosen[depth]=i;
        int delta=((ctx->original>>i)&1)?2:-2;
        enumerate(ctx,i+1,depth+1,chosen,changed^((word)1<<i),next,dc+delta,alt+((i&1)?-delta:delta));
    }
}
static void prepare_singles(struct context *ctx) {
    for(int i=0;i<ctx->length;++i) {
        uint64_t signature=0;
        for(int u=1;u<=ctx->length/2;++u) {
            int bit=1^(int)((ctx->original>>((i+u)%ctx->length))&1)
                     ^(int)((ctx->original>>((i+ctx->length-u)%ctx->length))&1);
            signature|=(uint64_t)bit<<(u-1);
        }
        ctx->singles[i]=signature;
    }
}

int pqcp_unrestricted_flip_join(uint64_t alo,uint64_t ahi,uint64_t blo,uint64_t bhi,
    int length,int radius,int group_content,uint64_t *output,uint64_t *stats) {
    if(length<4||length>94||(length&1)||radius<0||radius>5||(group_content!=0&&group_content!=1))return -1;
    word a=((word)ahi<<64)|alo,b=((word)bhi<<64)|blo;
    if((a>>length)||(b>>length))return -1;
    size_t count=1,term=1;
    for(int i=1;i<=radius;++i){term=term*(size_t)(length-i+1)/(size_t)i;count+=term;}
    struct entry *index=malloc(count*sizeof(*index));
    if(!index)return -2;
    memset(stats,0,7*sizeof(*stats));
    struct context ctx={.length=length,.radius=radius,.original=a,
        .group_content=group_content&&length==86,
        .all=(((word)1)<<length)-1,.index=index,.stats=stats,.output=output};
    for(int i=0;i<length;i+=2)ctx.even|=(word)1<<i;
    for(int u=1;u<=length/2;++u) {
        int t=profile(a,length,u,ctx.all)+profile(b,length,u,ctx.all);
        if(t%4){free(index);return -3;}
        ctx.base_signature|=(uint64_t)((t/4)&1)<<(u-1);
    }
    int chosen[5];
    int dc=length-2*weight(a),alt=2*(weight(a&~ctx.even)-weight(a&ctx.even));
    prepare_singles(&ctx);enumerate(&ctx,0,0,chosen,a,0,dc,alt);
    if(stats[0]!=count){free(index);return -4;}
    qsort(index,ctx.used,sizeof(*index),compare_entry);
    ctx.original=b;ctx.side=1;
    dc=length-2*weight(b);alt=2*(weight(b&~ctx.even)-weight(b&ctx.even));
    prepare_singles(&ctx);enumerate(&ctx,0,0,chosen,b,0,dc,alt);
    if(!ctx.found&&stats[1]!=count){free(index);return -4;}
    free(index);
    return ctx.found;
}
