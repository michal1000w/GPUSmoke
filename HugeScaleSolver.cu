#include <stdio.h>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cuda_fp16.h>
#include "cutil_math.h"



#include "Fluid_Kernels.cuh"
#include "Unified_Buffer.cpp"

//#define EXPERIMENTAL
//#include <driver_functions.h>
//#include <driver_types.h>



struct fluid_state_huge {

    float3 impulseLoc;
    float impulseTemp;
    float impulseDensity;
    float impulseRadius;
    float f_weight;
    float cell_size;
    float time_step;
    int3 dim;
    int64_t nelems;
    int step;
    UnifiedBuffer<float3>* velocity;
    UnifiedBuffer<float>* density;
    UnifiedBuffer<float>* temperature;
    UnifiedBuffer<float>* pressure;
    float* diverge;

    fluid_state_huge(int3 dims) {
        step = 0;
        dim = dims;
        nelems = dims.x * dims.y * dims.z;
        velocity = new UnifiedBuffer<float3>((int)nelems);
        density = new UnifiedBuffer<float>((int)nelems);
        temperature = new UnifiedBuffer<float>((int)nelems);
        pressure = new UnifiedBuffer<float>((int)nelems);
        

        cudaMalloc((void**)&diverge, sizeof(float) * nelems);
        //cudaDeviceSynchronize();
    }

    ~fluid_state_huge() {
        delete velocity;
        delete density;
        delete temperature;
        delete pressure;
        cudaFree(diverge);
    }
};
