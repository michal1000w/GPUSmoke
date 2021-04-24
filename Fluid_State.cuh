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
    //std::vector<float*> diverge;
    DoubleBuffer<float>* diverge;
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

        //diverge = multiGPU_malloc<float>(devicesCount,deviceIndex, nelems);
        diverge = new DoubleBuffer<float>((int)nelems, devicesCount, deviceIndex);
        diverge->setDim(dims);



        int noiseDim = 64;
        int3 noiseDims = make_int3(noiseDim, noiseDim, noiseDim);
        int nnelems = noiseDim * noiseDim * noiseDim;

        noise = new DoubleBuffer<float>((int)nnelems, devicesCount, deviceIndex);
        noise->setDim(noiseDims);
    }

    void zeros(int devicesCount, int deviceIndex) {
        std::cout << "c->";
        density->zeros(deviceIndex);
        std::cout << "d;";
        temperature->zeros(deviceIndex);
        std::cout << "t;";
        pressure->zeros(deviceIndex);
        std::cout << "p;";
        flame->zeros(deviceIndex);
        std::cout << "f;";
        diverge->zeros(deviceIndex);
        std::cout << "d;";
        noise->zeros(deviceIndex);
        std::cout << "n;";
        velocity->zeros(deviceIndex);
        std::cout << "v;";
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
        //multiGPU_free(devicesCount, diverge);
        delete diverge;
    }

    void sync_devices(int devicesCount = 1) {
        //todo
        
    }
};