/* Public Foundation API: bound autoreleased host objects on this thread.
 * Tensor buffers and search state keep their normal retained owners.
 * No allocator purge, stream synchronization or private PyTorch API. */
#import <Foundation/Foundation.h>

void *pqcp_autorelease_create(void) {
    return [[NSAutoreleasePool alloc] init];
}
void pqcp_autorelease_drain(void *pool) {
    [(NSAutoreleasePool *)pool drain];
}
