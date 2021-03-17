#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "GRID3D.h"



template <typename T>
class DoubleBuffer
{
public:
    DoubleBuffer();
    DoubleBuffer(int nelements);
    T* readTarget();
    T* writeTarget();
    inline GRID3D* readToGrid(bool debug = false) {
        if (debug)
            std::cout << "Reading Grid: ";
        clock_t startTime = clock();
        GRID3D* output = new GRID3D();
        if (debug) std::cout << "GRID3D->";
        output->load_from_device(dim, readTarget(),debug);
        //output->set_pointer(new GRID3D(dim, readTarget()));
        if (debug)
            std::cout << (clock() - startTime);
        return output;
    }
    
    inline GRID3D* readToGrid3D(bool debug = false) {
        if (debug) {
            std::cout << " Reading Grid:";
            std::cout << dim.x << "x" << dim.y << "x" << dim.z << std::endl;
        }
        clock_t startTime = clock();
        GRID3D* output = new GRID3D(dim);
        if (debug) std::cout << "Copying..." << std::endl;
        output->load_from_device3D(dim, readTarget());
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
        cudaFree(B);
    }
private:
    int nbytes;
    T* A;
    T* B;
    int3 dim;
};
