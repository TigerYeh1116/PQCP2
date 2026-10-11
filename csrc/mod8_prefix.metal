#include <metal_stdlib>
using namespace metal;

// Fixed row strides: constant, 24 linear coefficients, 24*24 quadratic.
kernel void pqcp_mod8_prefix(
    device const int *raw [[buffer(0)]],
    device const int *projected [[buffer(1)]],
    device const int *metadata [[buffer(2)]],
    device int *models [[buffer(3)]],
    device atomic_uint *counts [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]) {
    uint system=tid.y, q=uint(metadata[4*system]), tail=uint(metadata[4*system+1]);
    uint capacity=uint(metadata[4*system+2]), prefix=q-tail;
    if (tid.x >= (1u<<prefix)) return;
    uint offset=601u*system, bits=tid.x;
    uint value=uint(projected[offset]), actual=uint(raw[offset]);
    for (uint i=0;i<prefix;++i) if ((bits>>i)&1u) {
        value ^= uint(projected[offset+1+i]); actual ^= uint(raw[offset+1+i]);
        for (uint j=i+1;j<prefix;++j) if ((bits>>j)&1u) {
            value ^= uint(projected[offset+25+24*i+j]);
            actual ^= uint(raw[offset+25+24*i+j]);
        }
    }
    uint rows[25]={0},labels[25]={0},nullspace[8],raw_columns[8];
    uint nullity=0;
    for (uint j=0;j<tail;++j) {
        uint column=uint(projected[offset+1+prefix+j]);
        uint original=uint(raw[offset+1+prefix+j]);
        for (uint i=0;i<prefix;++i) if ((bits>>i)&1u) {
            column ^= uint(projected[offset+25+24*i+prefix+j]);
            original ^= uint(raw[offset+25+24*i+prefix+j]);
        }
        raw_columns[j]=original;
        uint label=1u<<j;
        while (column) {
            uint pivot=31u-clz(column);
            if (!rows[pivot]) { rows[pivot]=column;labels[pivot]=label;break; }
            column ^= rows[pivot];label ^= labels[pivot];
        }
        if (!column) nullspace[nullity++]=label;
    }
    uint rhs=value,solution=0;
    while (rhs) {
        uint pivot=31u-clz(rhs);
        if (!rows[pivot]) return;
        rhs ^= rows[pivot];solution ^= labels[pivot];
    }
    for (uint member=0;member<(1u<<nullity);++member) {
        uint candidate=solution;
        for (uint j=0;j<nullity;++j) if ((member>>j)&1u) candidate ^= nullspace[j];
        uint syndrome=actual;
        for (uint i=0;i<tail;++i) if ((candidate>>i)&1u) {
            syndrome ^= raw_columns[i];
            for (uint j=i+1;j<tail;++j) if ((candidate>>j)&1u)
                syndrome ^= uint(raw[offset+25+24*(prefix+i)+prefix+j]);
        }
        if (!syndrome) {
            uint index=atomic_fetch_add_explicit(counts+system,1u,memory_order_relaxed);
            if (index<capacity) models[uint(metadata[4*system+3])+index]=int(bits|(candidate<<prefix));
        }
    }
}

// Exact bit-sliced tail search. Intersect the satisfying truth masks after
// EACH original Boolean constraint; an empty intersection excludes every
// tail member in that prefix. All retained original MQ zeros are emitted.
kernel void pqcp_mod8_bit_tail_prefix(
    device const uint *packet [[buffer(0)]],
    device int *models [[buffer(1)]],
    device atomic_uint *counts [[buffer(2)]],
    uint2 tid [[thread_position_in_grid]]) {
    // One host upload: 7 metadata words, 601 original coefficients,
    // up to (25+8)*8=264 tail-truth words. Total fixed stride 872.
    uint system=tid.y,header=872u*system,q=packet[header],tail=packet[header+1];
    uint prefix=q-tail,bits=tid.x,capacity=packet[header+2];
    if (bits>=(1u<<prefix)) return;
    uint offset=header+7u,value=packet[offset];
    for (uint i=0;i<prefix;++i) if ((bits>>i)&1u) {
        value^=packet[offset+1+i];
        for (uint j=i+1;j<prefix;++j) if ((bits>>j)&1u)
            value^=packet[offset+25+24*i+j];
    }
    uint columns[8];
    for (uint j=0;j<tail;++j) {
        uint column=packet[offset+1+prefix+j];
        for (uint i=0;i<prefix;++i) if ((bits>>i)&1u)
            column^=packet[offset+25+24*i+prefix+j];
        columns[j]=column;
    }
    uint begin=offset+601u,equations=packet[header+4];
    uint chunks=packet[header+5],feasible[8],members=1u<<tail;
    for (uint chunk=0;chunk<chunks;++chunk)
        feasible[chunk]=members>=32u?0xffffffffu:(1u<<members)-1u;
    // Count constraints first, then all independent PACF shifts. Ordering
    // is only scheduling: every original equation is eventually checked.
    for (uint step=0;step<equations;++step) {
        uint count_fields=min(2u,equations);
        uint equation=step<count_fields?equations-count_fields+step:step-count_fields;
        uint present=0;
        for (uint chunk=0;chunk<chunks;++chunk) {
            uint polynomial=packet[begin+equation*chunks+chunk];
            if ((value>>equation)&1u) polynomial^=0xffffffffu;
            for (uint j=0;j<tail;++j) if ((columns[j]>>equation)&1u)
                polynomial^=packet[begin+(equations+j)*chunks+chunk];
            feasible[chunk]&=~polynomial;present|=feasible[chunk];
        }
        if (!present) return;
    }
    for (uint chunk=0;chunk<chunks;++chunk) for (uint rest=feasible[chunk];rest;rest&=rest-1u) {
        uint member=chunk*32u+ctz(rest);
        uint index=atomic_fetch_add_explicit(counts+system,1u,memory_order_relaxed);
        if (index<capacity) models[packet[header+3]+index]=int(bits|(member<<prefix));
    }
}

// Common L43 raw affine size: 21 variables, six-tail/15-prefix split.
// uint2 holds ALL 64 tail members in registers. Other shapes use the
// generic kernel above, preserving the exact same modular zero set.
kernel void pqcp_mod8_bit_tail_q21_t6(
    device const uint *packet [[buffer(0)]],
    device int *models [[buffer(1)]],
    device atomic_uint *counts [[buffer(2)]],
    uint2 tid [[thread_position_in_grid]]) {
    if (tid.x>=32768u) return;
    uint system=tid.y,header=872u*system,offset=header+7u,bits=tid.x;
    uint value=packet[offset];
    for (uint i=0;i<15u;++i) if ((bits>>i)&1u) {
        value^=packet[offset+1u+i];
        for (uint j=i+1u;j<15u;++j) if ((bits>>j)&1u)
            value^=packet[offset+25u+24u*i+j];
    }
    uint columns[6];
    for (uint j=0;j<6u;++j) {
        uint column=packet[offset+16u+j];
        for (uint i=0;i<15u;++i) if ((bits>>i)&1u)
            column^=packet[offset+25u+24u*i+15u+j];
        columns[j]=column;
    }
    device const uint2 *truth=(device const uint2 *)(packet+offset+601u);
    uint2 feasible=uint2(0xffffffffu);
    for (uint step=0;step<23u;++step) {
        uint equation=step<2u?21u+step:step-2u;
        uint2 polynomial=truth[equation];
        if ((value>>equation)&1u) polynomial^=uint2(0xffffffffu);
        #pragma clang loop unroll(full)
        for (uint j=0;j<6u;++j) if ((columns[j]>>equation)&1u)
            polynomial^=truth[23u+j];
        feasible&=~polynomial;
        if (!(feasible.x|feasible.y)) return;
    }
    for (uint chunk=0;chunk<2u;++chunk) {
        uint rest=chunk?feasible.y:feasible.x;
        while (rest) {
            uint member=chunk*32u+ctz(rest);rest&=rest-1u;
            uint index=atomic_fetch_add_explicit(counts+system,1u,memory_order_relaxed);
            if (index<packet[header+2]) models[packet[header+3]+index]=int(bits|(member<<15u));
        }
    }
}

// Same complete raw MQ enumeration, preceded by certified degree-three
// consequences. Selected conditions are necessary; omitted conditions and
// failed/overflowed dispatches never become exclusions.
kernel void pqcp_mod8_crossbred_prefix(
    device const int *raw [[buffer(0)]],
    device const int *projected [[buffer(1)]],
    device const int *metadata [[buffer(2)]],
    device int *models [[buffer(3)]],
    device atomic_uint *counts [[buffer(4)]],
    device const uint *lookup [[buffer(5)]],
    device const int *lookup_meta [[buffer(6)]],
    uint2 tid [[thread_position_in_grid]]) {
    uint system=tid.y,q=uint(metadata[4*system]),tail=uint(metadata[4*system+1]);
    uint capacity=uint(metadata[4*system+2]),prefix=q-tail;
    if (tid.x>=(1u<<prefix)) return;
    uint rows[32]={0};
    if (lookup_meta[6*system+5]) {
        uint offset=uint(lookup_meta[6*system]),low=uint(lookup_meta[6*system+1]);
        uint L=uint(lookup_meta[6*system+2]),H=uint(lookup_meta[6*system+3]);
        uint stride=uint(lookup_meta[6*system+4]),lo=tid.x&(L-1u),hi=tid.x>>low;
        uint values[9];
        for (uint block=0;block<=tail;++block) {
            uint begin=offset+block*stride;
            uint value=lookup[begin+lo]^lookup[begin+L+hi];
            for (uint rest=lo;rest;rest&=rest-1u) {
                uint i=ctz(rest);value^=lookup[begin+L+H+i*H+hi];
            }
            for (uint rest=hi;rest;rest&=rest-1u) {
                uint i=ctz(rest);value^=lookup[begin+L+H+low*H+i*L+lo];
            }
            values[block]=value;
        }
        for (uint j=0;j<tail;++j) {
            uint column=values[1+j];
            while (column) {
                uint pivot=31u-clz(column);
                if (!rows[pivot]) {rows[pivot]=column;break;}
                column^=rows[pivot];
            }
        }
        uint rhs=values[0];
        while (rhs) {
            uint pivot=31u-clz(rhs);
            if (!rows[pivot]) return;
            rhs^=rows[pivot];
        }
        for (uint j=0;j<32;++j) rows[j]=0;
    }
    uint offset=601u*system,bits=tid.x;
    uint value=uint(projected[offset]),actual=uint(raw[offset]);
    for (uint i=0;i<prefix;++i) if ((bits>>i)&1u) {
        value^=uint(projected[offset+1+i]);actual^=uint(raw[offset+1+i]);
        for (uint j=i+1;j<prefix;++j) if ((bits>>j)&1u) {
            value^=uint(projected[offset+25+24*i+j]);
            actual^=uint(raw[offset+25+24*i+j]);
        }
    }
    uint labels[32]={0},nullspace[8],raw_columns[8],nullity=0;
    for (uint j=0;j<tail;++j) {
        uint column=uint(projected[offset+1+prefix+j]);
        uint original=uint(raw[offset+1+prefix+j]);
        for (uint i=0;i<prefix;++i) if ((bits>>i)&1u) {
            column^=uint(projected[offset+25+24*i+prefix+j]);
            original^=uint(raw[offset+25+24*i+prefix+j]);
        }
        raw_columns[j]=original;
        uint label=1u<<j;
        while (column) {
            uint pivot=31u-clz(column);
            if (!rows[pivot]) {rows[pivot]=column;labels[pivot]=label;break;}
            column^=rows[pivot];label^=labels[pivot];
        }
        if (!column) nullspace[nullity++]=label;
    }
    uint rhs=value,solution=0;
    while (rhs) {
        uint pivot=31u-clz(rhs);
        if (!rows[pivot]) return;
        rhs^=rows[pivot];solution^=labels[pivot];
    }
    for (uint member=0;member<(1u<<nullity);++member) {
        uint candidate=solution;
        for (uint j=0;j<nullity;++j) if ((member>>j)&1u) candidate^=nullspace[j];
        uint syndrome=actual;
        for (uint i=0;i<tail;++i) if ((candidate>>i)&1u) {
            syndrome^=raw_columns[i];
            for (uint j=i+1;j<tail;++j) if ((candidate>>j)&1u)
                syndrome^=uint(raw[offset+25+24*(prefix+i)+prefix+j]);
        }
        if (!syndrome) {
            uint index=atomic_fetch_add_explicit(counts+system,1u,memory_order_relaxed);
            if (index<capacity) models[uint(metadata[4*system+3])+index]=int(bits|(candidate<<prefix));
        }
    }
}
