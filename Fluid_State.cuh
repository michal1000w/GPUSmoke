#include "Object.h"


template <typename T>
__global__ void combine(T* left, T* right, int3 vd) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int z = blockDim.z * blockIdx.z + threadIdx.z;

    if (x >= vd.x || y >= vd.y || z >= vd.z) return;

    left[z * vd.y * vd.x + y * vd.x + x] += right[z * vd.y * vd.x + y * vd.x + x];
}



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
    DoubleBuffer<float>* noise;
    std::vector<float*> diverge;
    int devicesCount = 1;

    

    fluid_state(int3 dims, int devicesCount, int deviceIndex) {
        this->devicesCount = devicesCount;
        step = 0;
        dim = dims;
        nelems = dims.x * dims.y * dims.z;
        velocity = new DoubleBuffer<float3>((int)nelems, devicesCount, deviceIndex);
        velocity->setDim(dims);
        density = new DoubleBuffer<float>((int)nelems, devicesCount, deviceIndex);
        density->setDim(dims);
        temperature = new DoubleBuffer<float>((int)nelems, devicesCount, deviceIndex);
        temperature->setDim(dims);
        pressure = new DoubleBuffer<float>((int)nelems, devicesCount, deviceIndex);
        pressure->setDim(dims);
        flame = new DoubleBuffer<float>((int)nelems, devicesCount, deviceIndex);
        flame->setDim(dims);

        //cudaMalloc((void**)&diverge, sizeof(float) * nelems);
        diverge = multiGPU_malloc<float>(devicesCount,deviceIndex, nelems);




        int noiseDim = 64;
        int3 noiseDims = make_int3(noiseDim, noiseDim, noiseDim);
        int nnelems = noiseDim * noiseDim * noiseDim;

        noise = new DoubleBuffer<float>((int)nnelems, devicesCount, deviceIndex);
        noise->setDim(noiseDims);
    }

    fluid_state() {

    }

    ~fluid_state() {
        delete velocity;
        delete density;
        delete temperature;
        delete pressure;
        delete flame;
        delete noise;
        //cudaFree(diverge);
        multiGPU_free(devicesCount, diverge);
    }

    void sync_devices(int devicesCount = 1) {
        //todo
        
    }
};