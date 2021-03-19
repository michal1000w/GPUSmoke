#include "double_buffer.h"
#include <utility>


template <typename T>
DoubleBuffer<T>::DoubleBuffer(int devices)
{
    A = 0;
    B = 0;
    nbytes = 0;
    devicesCount = devices;
}

template <typename T>
DoubleBuffer<T>::DoubleBuffer(int nelements, int devicesCount)
{
    this->devicesCount = devicesCount;
    nbytes = sizeof(T)*nelements;
    //cudaMalloc( (void**)&A, nbytes );
    //cudaMalloc( (void**)&B, nbytes );
    A = multiGPU_malloc<T>(this->devicesCount, nelements);
    B = multiGPU_malloc<T>(this->devicesCount, nelements);
    
    printf("Allocated %.2f MB on GPU\n", 2*nbytes/(1024.f*1024.f));
}

template <typename T>
std::vector<T*>* DoubleBuffer<T>::readTarget()
{
    return &A;
}
 


template<typename T>
void DoubleBuffer<T>::setDim(int3 dim) {
    this->dim = dim;
}

template <typename T>
std::vector<T*>* DoubleBuffer<T>::writeTarget()
{
    return &B;
}

template <typename T>
void DoubleBuffer<T>::swap()
{
    std::swap(A,B);
}

template <typename T>
int DoubleBuffer<T>::byteCount()
{
    return nbytes;
}

template <typename T>
DoubleBuffer<T>::~DoubleBuffer()
{
    //cudaFree(A);
    //cudaFree(B);
    multiGPU_free(this->devicesCount, A);
    multiGPU_free(this->devicesCount, B);
}
