#include <metal_stdlib>
using namespace metal;

/* All arithmetic determining a candidate energy is signed int32.
 * d<=47, m<=8: every intermediate is below 5,193,216 in magnitude.
 * No floating matrix multiplication, epsilon or rounded pruning. */
kernel void pqcp_block_deltas(device const int* labels,
    device const int* old_values,device const int* linear,
    device const int* positions,device const int* shape,
    device int* output,uint index [[thread_position_in_grid]]) {
    const int n=shape[0],m=shape[1],h=shape[2],d=shape[3];
    if (index>=uint(n*2*h)) return;
    const int row=int(index)/(2*h),component=(int(index)/h)%2,u=int(index)%h+1;
    int changes[8],value=0;
    for (int i=0;i<m;++i) {
        int color=labels[row*m+i];
        int next=color==2*component ? 1 : color==2*component+1 ? -1 : 0;
        changes[i]=next-old_values[component*m+i];
        value+=changes[i]*linear[(component*m+i)*h+u-1];
    }
    for (int i=0;i<m;++i) for (int j=i+1;j<m;++j) {
        int gap=positions[j]-positions[i];
        if (gap<0) gap+=d;
        if (min(gap,d-gap)==u) value+=changes[i]*changes[j];
    }
    output[index]=value;
}

/* Direct sum-of-squares, with 16x16 shared tiles; every output pair is
 * computed once. Padded threads still participate in ALL barriers. */
kernel void pqcp_block_scores(device const int* a,device const int* b,
    device const int* residual,device const int* shape,device int* output,
    uint2 local [[thread_position_in_threadgroup]],uint2 group [[threadgroup_position_in_grid]]) {
    const int na=shape[0],nb=shape[1],width=shape[2];
    const int tx=int(local.x),ty=int(local.y);
    const int row=int(group.y)*16+ty,col=int(group.x)*16+tx;
    threadgroup int left[16][16],right[16][16];
    int energy=0;
    for (int start=0;start<width;start+=16) {
        const int ka=start+tx,kb=start+ty;
        left[ty][tx]=row<na && ka<width ? a[row*width+ka]+residual[ka] : 0;
        right[ty][tx]=col<nb && kb<width ? b[col*width+kb] : 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int k=0;k<16;++k) {
            int value=left[ty][k]+right[k][tx];
            energy+=value*value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row<na && col<nb) output[row*nb+col]=energy;
}
