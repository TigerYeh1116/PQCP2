/* Metal port of csrc/ternary_signs.c for many support pairs at once.
 * Each pair owns 2^cb threads; thread c walks Gray-code indices
 * [c*2^(F-cb), (c+1)*2^(F-cb)) of its GF(2) affine sign space, checks the
 * exact negative counts, then every shift's exact edge-cut count. A hit sets
 * hits[pair] and one witness; the caller re-verifies hits on the CPU.
 * Exhausting a pair without a hit is the same complete enumeration as the
 * CPU kernel. Returns 0 on success, negative on host/Metal failure. */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdint.h>
#include <string.h>

typedef struct { uint32_t npairs, d, H, maxF, cb; } PqcpParams;

static NSString *const kSource = @
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"struct PqcpParams { uint npairs; uint d; uint H; uint maxF; uint cb; };\n"
"inline ulong rot(ulong x, uint u, uint d, ulong all) { return ((x << u) | (x >> (d-u))) & all; }\n"
"kernel void pqcp_signs(device const ulong *support [[buffer(0)]],\n"
"                       device const ulong *initial [[buffer(1)]],\n"
"                       device const ulong *toggles [[buffer(2)]],\n"
"                       device const uint *freec [[buffer(3)]],\n"
"                       device const int *required [[buffer(4)]],\n"
"                       device const int *negative [[buffer(5)]],\n"
"                       device atomic_uint *hits [[buffer(6)]],\n"
"                       device ulong *witness [[buffer(7)]],\n"
"                       constant PqcpParams &P [[buffer(8)]],\n"
"                       uint gid [[thread_position_in_grid]],\n"
"                       uint lid [[thread_position_in_threadgroup]]) {\n"
"  threadgroup uint T[4*32];\n"
"  uint pair = gid >> P.cb, chunk = gid & ((1u << P.cb) - 1u);\n"
"  bool live = pair < P.npairs;\n"
"  uint F = live ? freec[pair] : 0u;\n"
"  if (live && lid < F) {\n"
"    ulong t0 = toggles[((ulong)pair * P.maxF + lid) * 2ul], t1 = toggles[((ulong)pair * P.maxF + lid) * 2ul + 1ul];\n"
"    T[4*lid] = (uint)t0; T[4*lid+1] = (uint)(t0 >> 32); T[4*lid+2] = (uint)t1; T[4*lid+3] = (uint)(t1 >> 32);\n"
"  }\n"
"  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
"  if (!live) return;\n"
"  uint cbits = min(P.cb, F);\n"
"  if (chunk >= (1u << cbits)) return;\n"
"  uint inner = F - cbits, K = 1u << inner, start = chunk << inner, g = start ^ (start >> 1);\n"
"  ulong i0 = initial[2*pair], i1 = initial[2*pair+1];\n"
"  uint a0 = (uint)i0, a1 = (uint)(i0 >> 32), b0 = (uint)i1, b1 = (uint)(i1 >> 32);\n"
"  for (uint b = 0; b < F; b++) if ((g >> b) & 1u) { a0 ^= T[4*b]; a1 ^= T[4*b+1]; b0 ^= T[4*b+2]; b1 ^= T[4*b+3]; }\n"
"  int n0 = negative[2*pair], n1 = negative[2*pair+1];\n"
"  for (uint i = 0; i < K; i++) {\n"
"    if ((int)(popcount(a0) + popcount(a1)) == n0 && (int)(popcount(b0) + popcount(b1)) == n1) {\n"
"      ulong all = (1ul << P.d) - 1ul, S0 = support[2*pair], S1 = support[2*pair+1];\n"
"      ulong s0 = ((ulong)a1 << 32) | a0, s1 = ((ulong)b1 << 32) | b0;\n"
"      device const int *req = required + pair * P.H;\n"
"      bool ok = true;\n"
"      for (uint u = 1; u <= P.H && ok; u++) {\n"
"        int n = (int)popcount((s0 ^ rot(s0, u, P.d, all)) & S0 & rot(S0, u, P.d, all))\n"
"              + (int)popcount((s1 ^ rot(s1, u, P.d, all)) & S1 & rot(S1, u, P.d, all));\n"
"        ok = (n == req[u-1]);\n"
"      }\n"
"      if (ok) {\n"
"        if (atomic_exchange_explicit(&hits[pair], 1u, memory_order_relaxed) == 0u) { witness[2*pair] = s0; witness[2*pair+1] = s1; }\n"
"        return;\n"
"      }\n"
"    }\n"
"    if (i + 1u < K) { uint bit = ctz(start + i + 1u); a0 ^= T[4*bit]; a1 ^= T[4*bit+1]; b0 ^= T[4*bit+2]; b1 ^= T[4*bit+3]; }\n"
"  }\n"
"}\n";

typedef struct {
    void *device, *queue, *pipeline;
} PqcpGpu;

void *pqcp_gpu_create(char *error, int error_length) {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) { snprintf(error, error_length, "no Metal device"); return NULL; }
        NSError *failure = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:kSource options:nil error:&failure];
        if (!library) { snprintf(error, error_length, "%s", failure.localizedDescription.UTF8String); return NULL; }
        id<MTLFunction> function = [library newFunctionWithName:@"pqcp_signs"];
        id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&failure];
        if (!pipeline) { snprintf(error, error_length, "%s", failure.localizedDescription.UTF8String); return NULL; }
        PqcpGpu *gpu = calloc(1, sizeof(PqcpGpu));
        gpu->device = (__bridge_retained void *)device;
        gpu->queue = (__bridge_retained void *)[device newCommandQueue];
        gpu->pipeline = (__bridge_retained void *)pipeline;
        return gpu;
    }
}

int pqcp_gpu_check(void *handle, uint32_t npairs, uint32_t d, uint32_t maxF, uint32_t cb,
                   const uint64_t *support, const uint64_t *initial, const uint64_t *toggles,
                   const uint32_t *freec, const int32_t *required, const int32_t *negative,
                   uint32_t *hits, uint64_t *witness) {
    if (!handle || !npairs || d < 3 || d > 47 || !(d & 1) || maxF > 31 || cb < 8 || cb > 20) return -1;
    @autoreleasepool {
        PqcpGpu *gpu = handle;
        id<MTLDevice> device = (__bridge id<MTLDevice>)gpu->device;
        id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)gpu->queue;
        id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)gpu->pipeline;
        uint32_t H = d/2;
        PqcpParams params = {npairs, d, H, maxF, cb};
        size_t tog = (size_t)npairs*(maxF ? maxF : 1)*2*sizeof(uint64_t);
        id<MTLBuffer> b0 = [device newBufferWithBytes:support length:npairs*2*sizeof(uint64_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> b1 = [device newBufferWithBytes:initial length:npairs*2*sizeof(uint64_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> b2 = maxF ? [device newBufferWithBytes:toggles length:tog options:MTLResourceStorageModeShared]
                                : [device newBufferWithLength:tog options:MTLResourceStorageModeShared];
        id<MTLBuffer> b3 = [device newBufferWithBytes:freec length:npairs*sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> b4 = [device newBufferWithBytes:required length:npairs*H*sizeof(int32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> b5 = [device newBufferWithBytes:negative length:npairs*2*sizeof(int32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> b6 = [device newBufferWithLength:npairs*sizeof(uint32_t) options:MTLResourceStorageModeShared];
        id<MTLBuffer> b7 = [device newBufferWithLength:npairs*2*sizeof(uint64_t) options:MTLResourceStorageModeShared];
        if (!b0 || !b1 || !b2 || !b3 || !b4 || !b5 || !b6 || !b7) return -2;
        memset(b6.contents, 0, npairs*sizeof(uint32_t));
        id<MTLCommandBuffer> command = [queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        id<MTLBuffer> buffers[8] = {b0, b1, b2, b3, b4, b5, b6, b7};
        for (int i = 0; i < 8; i++) [encoder setBuffer:buffers[i] offset:0 atIndex:i];
        [encoder setBytes:&params length:sizeof(params) atIndex:8];
        NSUInteger width = MIN(pipeline.maxTotalThreadsPerThreadgroup, (NSUInteger)256);
        [encoder dispatchThreads:MTLSizeMake((NSUInteger)npairs << cb, 1, 1) threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) return -3;
        memcpy(hits, b6.contents, npairs*sizeof(uint32_t));
        memcpy(witness, b7.contents, npairs*2*sizeof(uint64_t));
        return 0;
    }
}

void pqcp_gpu_destroy(void *handle) {
    if (!handle) return;
    PqcpGpu *gpu = handle;
    CFRelease(gpu->pipeline); CFRelease(gpu->queue); CFRelease(gpu->device);
    free(gpu);
}
