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
    DoubleBuffer<float>* noise;
    std::vector<float*> diverge;
    int devicesCount = 1;

    

    fluid_state(int3 dims, int devicesCount) {
        this->devicesCount = devicesCount;
        step = 0;
        dim = dims;
        nelems = dims.x * dims.y * dims.z;
        velocity = new DoubleBuffer<float3>((int)nelems, devicesCount);
        velocity->setDim(dims);
        density = new DoubleBuffer<float>((int)nelems, devicesCount);
        density->setDim(dims);
        temperature = new DoubleBuffer<float>((int)nelems, devicesCount);
        temperature->setDim(dims);
        pressure = new DoubleBuffer<float>((int)nelems, devicesCount);
        pressure->setDim(dims);
        flame = new DoubleBuffer<float>((int)nelems, devicesCount);
        flame->setDim(dims);

        //cudaMalloc((void**)&diverge, sizeof(float) * nelems);
        diverge = multiGPU_malloc<float>(devicesCount, nelems);




        int noiseDim = 64;
        int3 noiseDims = make_int3(noiseDim, noiseDim, noiseDim);
        int nnelems = noiseDim * noiseDim * noiseDim;

        noise = new DoubleBuffer<float>((int)nnelems, devicesCount);
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

    void sync_devices() {
        /*
        for (int i = 0; i < devicesCount; i++) {
            cudaSetDevice(i);
            cudaThreadSynchronize();
            cudaDeviceSynchronize();
        }
        */
        checkCudaErrors(cudaMemcpyAsync(density->readTargett(1), density->readTargett(0), density->byteCount(), cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaMemcpyAsync(flame->readTargett(1), flame->readTargett(0), flame->byteCount(), cudaMemcpyDeviceToDevice));
    }
};