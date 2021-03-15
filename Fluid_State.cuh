#include "Object.h"

// Container for simulation state
struct fluid_state {

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
    DoubleBuffer<float3>* velocity;
    DoubleBuffer<float>* density;
    DoubleBuffer<float>* temperature;
    DoubleBuffer<float>* flame;
    DoubleBuffer<float>* pressure;
    float* diverge;

    

    fluid_state(int3 dims) {
        step = 0;
        dim = dims;
        nelems = dims.x * dims.y * dims.z;
        velocity = new DoubleBuffer<float3>((int)nelems);
        velocity->setDim(dims);
        density = new DoubleBuffer<float>((int)nelems);
        density->setDim(dims);
        temperature = new DoubleBuffer<float>((int)nelems);
        temperature->setDim(dims);
        pressure = new DoubleBuffer<float>((int)nelems);
        pressure->setDim(dims);
        flame = new DoubleBuffer<float>((int)nelems);
        flame->setDim(dims);
        cudaMalloc((void**)&diverge, sizeof(float) * nelems);
    }

    fluid_state() {

    }

    ~fluid_state() {
        delete velocity;
        delete density;
        delete temperature;
        delete pressure;
        delete flame;
        cudaFree(diverge);
    }
};