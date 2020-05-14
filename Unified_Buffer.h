#ifndef __UNIFIED_BUFFER
#define __UNIFIED_BUFFER

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename T>
class UnifiedBuffer
{
public:
    UnifiedBuffer();
    UnifiedBuffer(int nelements);
    T* readTarget();
    T* writeTarget();
    void swap();
    int byteCount();
    ~UnifiedBuffer();
private:
    int nbytes;
    T* A;
    T* B;
};



#endif // !__UNIFIED_BUFFER
