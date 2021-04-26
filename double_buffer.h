#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "GRID3D.h"



template <typename T>
class DoubleBuffer
{
public:
    DoubleBuffer(int devices);
    DoubleBuffer(int nelements, int devicesCount, int deviceIndex);
    std::vector<T*>* readTarget();
    std::vector<T*>* writeTarget();
    T* readTargett(unsigned int device = 0) { return A[device]; }
    T* writeTargett(unsigned int device = 0) { return B[device]; }
    inline GRID3D* readToGrid(int device = 0, bool debug = false) {
        if (debug)
            std::cout << "Reading Grid: \n";
        clock_t startTime = clock();
        GRID3D* output = new GRID3D(devicesCount);
        output->deviceCount = devicesCount;
        if (debug) std::cout << "GRID3D->";
        output->load_from_device(dim, readTargett(device),debug);
        //output->set_pointer(new GRID3D(dim, readTarget()));
        if (debug)
            std::cout << (clock() - startTime);
        return output;
    }
    
    inline GRID3D* readToGrid3D(int device = 0,bool debug = false) {
        if (debug) {
            std::cout << " Reading Grid: \n";
            std::cout << dim.x << "x" << dim.y << "x" << dim.z << std::endl;
        }
        clock_t startTime = clock();
        GRID3D* output = new GRID3D(dim,devicesCount);
        output->deviceCount = devicesCount;
        if (debug) std::cout << "Copying...\n" << std::endl;
        output->load_from_device3D(dim, readTargett(device),device);
        //std::cout << "Done";
        //output->set_pointer(new GRID3D(dim, readTarget()));
        if (debug)
            std::cout << (clock() - startTime);
        return output;
    }
    
    void swap();
    int byteCount();
    void setDim(int3 dim);
    ~DoubleBuffer();
    void freeWriteTarget() {
        //cudaFree(B);
        multiGPU_free(devicesCount, &B);
    }
    void zeros(int deviceIndex) {
        multiGPU_clear(&A, nbytes, devicesCount, deviceIndex);
        multiGPU_clear(&B, nbytes, devicesCount, deviceIndex);
    }
    T* temporary;


private:
    std::vector<T*> B;
    std::vector<T*> A;
    int nbytes;
    int devicesCount;
    int3 dim;
};
