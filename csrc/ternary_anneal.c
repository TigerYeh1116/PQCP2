/* Integer ternary navigation. Float exp is used only for a heuristic
 * acceptance probability, never for PACF, energy, feasibility or exclusions. */
#include <math.h>
#include <stdint.h>
#include <string.h>

static uint64_t random64(uint64_t *s) {
    *s^=*s>>12;*s^=*s<<25;*s^=*s>>27;return *s*UINT64_C(2685821657736338717);
}
static int pick(uint64_t *s,int n) {return (int)(random64(s)%(uint64_t)n);}
static void full(const int *w,int d,const int *wanted,int *r) {
    for(int u=1;u<=d/2;u++) {
        int v=0;for(int s=0;s<2;s++)for(int i=0;i<d;i++)v+=w[s*d+i]*w[s*d+(i+u)%d];
        r[u-1]=v-wanted[u];
    }
}
static int energy(const int *r,int h) {int v=0;for(int u=0;u<h;u++)v+=r[u]*r[u];return v;}
static int audit(const int *w,int d,const int *wanted,const int *r,const int *bounds,const int *sums) {
    int check[23],norm=0;full(w,d,wanted,check);
    for(int u=0;u<d/2;u++)if(check[u]!=r[u])return 0;
    for(int s=0;s<2;s++) {
        int sum=0,n=0;for(int i=0;i<d;i++){int v=w[s*d+i];if(v < -1 || v > 1)return 0;sum+=v;n+=v==-1;norm+=v*v;}
        if(sum!=sums[s] || n<bounds[2*s] || n>bounds[2*s+1])return 0;
    }
    return norm==wanted[0];
}
/* stats: proposals,accepted,transfer_proposals,transfer_accepted,audits.
 * Returns 1=compressed hit, 0=bounded miss, -1=bad input, -2=audit failure. */
int pqcp_ternary_anneal(const int *input,const int *wanted,const int *bounds,int d,
    uint64_t seed,uint64_t moves,int period,int audit_interval,int *output,int *best_energy,uint64_t *stats) {
    if(d<3 || d>47 || !(d&1) || period<1 || audit_interval<1)return -1;
    int w[94],residual[23],trial[23],counts[2][3],sums[2]={0};
    int plus[47][24],minus[47][24];
    memcpy(w,input,2*d*sizeof(int));memcpy(output,input,2*d*sizeof(int));memset(stats,0,5*sizeof(uint64_t));
    for(int i=0;i<d;i++)for(int u=0;u<=d/2;u++){plus[i][u]=(i+u)%d;minus[i][u]=(i-u+d)%d;}
    for(int s=0;s<2;s++)for(int i=0;i<d;i++)sums[s]+=w[s*d+i];
    full(w,d,wanted,residual);int current=energy(residual,d/2);*best_energy=current;
    if(!audit(w,d,wanted,residual,bounds,sums))return -1;
    if(current==0)return 1;
    uint64_t rng=seed?seed:UINT64_C(0x9e3779b97f4a7c15);
    for(uint64_t step=0;step<moves;step++) {
        int positions[4],values[4],changes=0;
        int slots[2][3][47];memset(counts,0,sizeof(counts));
        for(int s=0;s<2;s++)for(int i=0;i<d;i++){int v=w[s*d+i]+1;slots[s][v][counts[s][v]++]=i;}
        int create[2],directions=0;
        for(int s=0;s<2;s++) {
            int t=1-s;
            if(counts[s][1]>=2 && counts[t][0]>0 && counts[t][2]>0 &&
               counts[s][0]+1<=bounds[2*s+1] && counts[t][0]-1>=bounds[2*t])create[directions++]=s;
        }
        int transfer=directions && (pick(&rng,4)==0);
        if(transfer) {
            int s=create[pick(&rng,directions)],t=1-s;
            int zi=pick(&rng,counts[s][1]),zj=pick(&rng,counts[s][1]-1);if(zj>=zi)zj++;
            int sign=pick(&rng,2)?1:-1;
            positions[0]=s*d+slots[s][1][zi];positions[1]=s*d+slots[s][1][zj];
            positions[2]=t*d+slots[t][0][pick(&rng,counts[t][0])];
            positions[3]=t*d+slots[t][2][pick(&rng,counts[t][2])];
            values[0]=sign;values[1]=-sign;values[2]=values[3]=0;changes=4;stats[2]++;
        } else {
            int s=pick(&rng,2),valid[3],nvalid=0;
            for(int v=0;v<3;v++)if(counts[s][v])valid[nvalid++]=v;
            if(nvalid<2){s=1-s;nvalid=0;for(int v=0;v<3;v++)if(counts[s][v])valid[nvalid++]=v;}
            if(nvalid<2)break;
            int vi=pick(&rng,nvalid),vj=pick(&rng,nvalid-1);if(vj>=vi)vj++;
            int a=valid[vi],b=valid[vj];
            positions[0]=s*d+slots[s][a][pick(&rng,counts[s][a])];
            positions[1]=s*d+slots[s][b][pick(&rng,counts[s][b])];
            values[0]=b-1;values[1]=a-1;changes=2;
        }
        int old[4];for(int j=0;j<changes;j++)old[j]=w[positions[j]];
        for(int u=1;u<=d/2;u++) {
            int value=residual[u-1];
            for(int j=0;j<changes;j++) {
                int s=positions[j]/d,p=positions[j]%d,alpha=values[j]-old[j];
                value+=alpha*(w[s*d+plus[p][u]]+w[s*d+minus[p][u]]);
                for(int k=0;k<j;k++)if(positions[k]/d==s) {
                    int q=positions[k]%d,beta=values[k]-old[k];
                    value+=alpha*beta*((plus[p][u]==q)+(plus[q][u]==p));
                }
            }
            trial[u-1]=value;
        }
        stats[0]++;
        if(step%(uint64_t)audit_interval==0) {
            for(int j=0;j<changes;j++)w[positions[j]]=values[j];
            int okay=audit(w,d,wanted,trial,bounds,sums);stats[4]++;
            for(int j=0;j<changes;j++)w[positions[j]]=old[j];
            if(!okay)return -2;
        }
        int next=energy(trial,d/2);
        double temperature=4.0+(0.15-4.0)*(double)(step%(uint64_t)period)/(double)period;
        double uniform=(double)(random64(&rng)>>11)*0x1.0p-53;
        if(next<=current || uniform<exp(-(double)(next-current)/temperature)) {
            for(int j=0;j<changes;j++)w[positions[j]]=values[j];
            memcpy(residual,trial,(d/2)*sizeof(int));current=next;stats[1]++;if(transfer)stats[3]++;
            if(current<*best_energy){*best_energy=current;memcpy(output,w,2*d*sizeof(int));}
            if(current==0)return 1;
        }
    }
    return 0;
}
