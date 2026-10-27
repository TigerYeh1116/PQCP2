/* Metal port of the per-support part of pqcp_truncated_scan (csrc/truncated_signs.c).
 * One thread per C_B support (already weight-filtered on the CPU): GF(2) elimination of
 * the single-row edge-cut system, then the kept C_A sign patterns of its C_A support in
 * order; the first exact completion is reported as (pattern index, C_B negative mask).
 * The host compacts the hits to (thread index, pattern index, mask) and returns their
 * count (stats[0] = wide kernels skipped), negative on failure; callers re-verify hits. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdint.h>
#include <string.h>

typedef struct { uint32_t nsb, d, H, Mmax; } TruncParams;

static NSString *const kTruncSource = @
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"struct TruncParams { uint nsb; uint d; uint H; uint Mmax; };\n"
"inline ulong rotl(ulong x, uint u, uint d, ulong all) { return ((x << u) | (x >> (d-u))) & all; }\n"
"inline ulong rotr(ulong x, uint u, uint d, ulong all) { return ((x >> u) | (x << (d-u))) & all; }\n"
"kernel void trunc_scan(device const ulong *sb [[buffer(0)]], device const uint *sai [[buffer(1)]],\n"
"                       device const int *T [[buffer(2)]], device const int *nBv [[buffer(3)]],\n"
"                       device const uint *Mv [[buffer(4)]], device const uint *Q1 [[buffer(5)]],\n"
"                       device const ulong *Q0v [[buffer(6)]], device const char *RA [[buffer(7)]],\n"
"                       device uint *hit_s [[buffer(8)]], device ulong *hit_x [[buffer(9)]],\n"
"                       constant TruncParams &P [[buffer(10)]], uint gid [[thread_position_in_grid]]) {\n"
"  if (gid >= P.nsb) return;\n"
"  hit_s[gid] = 0xffffffffu;\n"
"  uint d = P.d, H = P.H, j = sai[gid];\n"
"  ulong all = (1ul << d) - 1ul, SB = sb[gid], Q0 = Q0v[j];\n"
"  int nB = nBv[j]; uint M = Mv[j];\n"
"  ulong act[23]; int base2[23]; ulong P0 = 0, P1 = 0;\n"
"  for (uint u = 1; u <= H; u++) {\n"
"    ulong a = SB & rotl(SB, u, d, all); act[u-1] = a;\n"
"    int b2 = (int)popcount(a) - T[j*H + u-1]; base2[u-1] = b2;\n"
"    P0 |= (ulong)(b2 & 1) << (u-1); P1 |= (ulong)((b2 >> 1) & 1) << (u-1);\n"
"  }\n"
"  if (P0 != Q0) return;\n"
"  ulong row[24]; uint comb[24], piv[24], deps[24], np = 0, nd = 0;\n"
"  for (uint e = 0; e <= H; e++) {\n"
"    ulong r = e < H ? (act[e] ^ rotr(act[e], e+1, d, all)) : SB;\n"
"    uint c = 1u << e;\n"
"    for (uint p = 0; p < np; p++) if ((r >> piv[p]) & 1ul) { r ^= row[p]; c ^= comb[p]; }\n"
"    if (r == 0ul) { deps[nd++] = c; continue; }\n"
"    uint col = ctz(r);\n"
"    for (uint p = 0; p < np; p++) if ((row[p] >> col) & 1ul) { row[p] ^= r; comb[p] ^= c; }\n"
"    row[np] = r; comb[np] = c; piv[np] = col; np++;\n"
"  }\n"
"  ulong pivmask = 0, contrib[24];\n"
"  for (uint p = 0; p < np; p++) pivmask |= 1ul << piv[p];\n"
"  for (uint e = 0; e <= H; e++) { contrib[e] = 0; for (uint p = 0; p < np; p++) if ((comb[p] >> e) & 1u) contrib[e] |= 1ul << piv[p]; }\n"
"  ulong kern[12]; uint nk = 0; ulong fc = SB & ~pivmask;\n"
"  while (fc) {\n"
"    uint col = ctz(fc); fc &= fc - 1ul;\n"
"    if (nk >= 12u) { hit_s[gid] = 0xfffffffeu; return; }\n"
"    ulong k = 1ul << col;\n"
"    for (uint p = 0; p < np; p++) if ((row[p] >> col) & 1ul) k |= 1ul << piv[p];\n"
"    kern[nk++] = k;\n"
"  }\n"
"  uint fixed = (uint)(P1 ^ (P0 & Q0)) ^ ((uint)(nB & 1) << H);\n"
"  device const uint *q1 = Q1 + (ulong)j * P.Mmax;\n"
"  device const char *ra = RA + (ulong)j * P.Mmax * H;\n"
"  for (uint s = 0; s < M; s++) {\n"
"    uint rhs = fixed ^ q1[s];\n"
"    bool ok = true;\n"
"    for (uint q = 0; q < nd; q++) if (popcount(deps[q] & rhs) & 1u) { ok = false; break; }\n"
"    if (!ok) continue;\n"
"    ulong x0 = 0; uint b = rhs;\n"
"    while (b) { x0 ^= contrib[ctz(b)]; b &= b - 1u; }\n"
"    ulong x = x0;\n"
"    for (uint z = 1; ; z++) {\n"   /* Gray code over the kernel */
"      if ((int)popcount(x) == nB) {\n"
"        bool ex = true;\n"
"        for (uint u = 1; u <= H && ex; u++) {\n"
"          int tw = base2[u-1] + (int)ra[s*H + u-1];\n"
"          ex = tw >= 0 && 2 * (int)popcount((x ^ rotl(x, u, d, all)) & act[u-1]) == tw;\n"
"        }\n"
"        if (ex) { hit_s[gid] = s; hit_x[gid] = x; return; }\n"
"      }\n"
"      if (z == (1u << nk)) break;\n"
"      x ^= kern[ctz(z)];\n"
"    }\n"
"  }\n"
"}\n";

typedef struct { void *device, *queue, *pipeline; } TruncGpu;

void *pqcp_trunc_gpu_create(char *error, int error_length) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) { snprintf(error, error_length, "no Metal device"); return NULL; }
        NSError *failure = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:kTruncSource options:nil error:&failure];
        if (!library) { snprintf(error, error_length, "%s", failure.localizedDescription.UTF8String); return NULL; }
        id<MTLComputePipelineState> pipeline =
            [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"trunc_scan"] error:&failure];
        if (!pipeline) { snprintf(error, error_length, "%s", failure.localizedDescription.UTF8String); return NULL; }
        TruncGpu *gpu = calloc(1, sizeof(TruncGpu));
        gpu->device = (__bridge_retained void *)device;
        gpu->queue = (__bridge_retained void *)[device newCommandQueue];
        gpu->pipeline = (__bridge_retained void *)pipeline;
        return gpu;
    }
}

int pqcp_trunc_gpu_scan(void *handle, uint32_t nsb, uint32_t nsa, uint32_t d, uint32_t Mmax,
                        const uint64_t *sb, const uint32_t *sai, const int32_t *T, const int32_t *nB,
                        const uint32_t *M, const uint32_t *Q1, const uint64_t *Q0, const int8_t *RA,
                        uint32_t *out_index, uint32_t *out_s, uint64_t *out_x, int max_out, uint64_t *stats) {
    if (!handle || !nsb || !nsa || d < 3 || d > 47 || !(d & 1) || !Mmax) return -1;
    @autoreleasepool {
        TruncGpu *gpu = handle;
        id<MTLDevice> device = (__bridge id<MTLDevice>)gpu->device;
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)gpu->queue;
        id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)gpu->pipeline;
        uint32_t H = d/2;
        TruncParams params = {nsb, d, H, Mmax};
        MTLResourceOptions shared = MTLResourceStorageModeShared;
        id<MTLBuffer> buffers[10] = {
            [device newBufferWithBytes:sb length:nsb*sizeof(uint64_t) options:shared],
            [device newBufferWithBytes:sai length:nsb*sizeof(uint32_t) options:shared],
            [device newBufferWithBytes:T length:nsa*H*sizeof(int32_t) options:shared],
            [device newBufferWithBytes:nB length:nsa*sizeof(int32_t) options:shared],
            [device newBufferWithBytes:M length:nsa*sizeof(uint32_t) options:shared],
            [device newBufferWithBytes:Q1 length:(size_t)nsa*Mmax*sizeof(uint32_t) options:shared],
            [device newBufferWithBytes:Q0 length:nsa*sizeof(uint64_t) options:shared],
            [device newBufferWithBytes:RA length:(size_t)nsa*Mmax*H options:shared],
            [device newBufferWithLength:nsb*sizeof(uint32_t) options:shared],
            [device newBufferWithLength:nsb*sizeof(uint64_t) options:shared]};
        for (int i = 0; i < 10; i++) if (!buffers[i]) return -2;
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        for (int i = 0; i < 10; i++) [encoder setBuffer:buffers[i] offset:0 atIndex:i];
        [encoder setBytes:&params length:sizeof(params) atIndex:10];
        NSUInteger width = MIN(pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)64);
        [encoder dispatchThreads:MTLSizeMake(nsb, 1, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) return -3;
        const uint32_t *hit_s = buffers[8].contents;
        const uint64_t *hit_x = buffers[9].contents;
        int found = 0;
        stats[0] = 0;
        for (uint32_t i = 0; i < nsb; i++) {
            if (hit_s[i] == 0xfffffffeu) stats[0]++;
            if (hit_s[i] >= 0xfffffffeu) continue;
            if (found < max_out) { out_index[found] = i; out_s[found] = hit_s[i]; out_x[found] = hit_x[i]; }
            found++;
        }
        return found;
    }
}

void pqcp_trunc_gpu_destroy(void *handle) {
    if (!handle) return;
    TruncGpu *gpu = handle;
    CFRelease(gpu->pipeline); CFRelease(gpu->queue); CFRelease(gpu->device);
    free(gpu);
}
