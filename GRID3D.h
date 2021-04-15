#pragma once
//VDB







//#define HOSTALLOC
//#define NEW_HOST_ALLOC






#include "cutil_math.h"


#define _USE_MATH_DEFINES
#include <nanovdb/NanoVDB.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/GridTransformer.h>
//#include <openvdb/tools/Interpolation.h>
#include <openvdb/tools/LevelSetSphere.h>
#include <openvdb/tools/SignedFloodFill.h>

#include <cmath>
#include <cuda_runtime.h>

#include <tbb/parallel_for.h>
#include <tbb/atomic.h>

#define SIZEOF_FLOAT3 (sizeof(float) * 3)
//#define SIZEOF_FLOAT3 (sizeof(float3))

#define SUPER_NULL -122.1123123
#define VELOCITY_NOISE

/*
#define CHECK cudaError_t error = cudaGetLastError();\
 if (error != cudaSuccess){\
 printf("CUDA Error: %s\n", cudaGetErrorString(error));\
}

#define cudaCheckError() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
 }                                                                 \
}
*/
#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)

inline std::vector<int> enumerate(int start_value, int max_values) {
    std::vector<int> output;
    output.push_back(start_value);
    int j = 0;
    for (int i = 0; i < max_values; i++) {
        if (i == start_value) j++;
        output.push_back(i + j);
    }
    return output;
}

inline void __checkCudaErrors(cudaError err, const char* file, const int line)
{
    if (cudaSuccess != err)
    {
        printf("\n\n%s  line:(%i) : CUDA Runtime API error %d: %s.\n", file, line, (int)err, cudaGetErrorString(err));
        exit(-1);
    }
}

inline void multiGPU_copyHTD(int devices_count, std::vector<float*>* dst, float* src, int size, int deviceIndex) {
    std::vector<std::thread> threads;

    std::cout << "CPPPPPP(" << dst->size() << ")";

    auto lista = enumerate(deviceIndex, devices_count);

    for (unsigned int device_id = 0; device_id < devices_count; device_id++)
    {
        threads.push_back(std::thread([&, device_id]() {
            checkCudaErrors(cudaSetDevice(lista[device_id]));


            std::cout << device_id;
            checkCudaErrors(cudaMemcpyAsync(dst->at(device_id), src, sizeof(float) * size, cudaMemcpyHostToDevice));
            //checkCudaErrors(cudaMemcpy(&dst[device_id], &src, sizeof(float) * size, cudaMemcpyHostToDevice));

            std::cout << "Copy";
            cudaDeviceSynchronize();
            }));
    }

    for (auto& thread : threads)
        thread.join();
}

inline void multiGPU_copy(int devices_count, std::vector<float3*>& dst, float3* src, int size, cudaMemcpyKind type, int deviceIndex) {
    std::vector<std::thread> threads;
    auto lista = enumerate(deviceIndex, devices_count);

    for (unsigned int device_id = 0; device_id < devices_count; device_id++)
    {
        threads.push_back(std::thread([&, device_id]() {
            cudaSetDevice(lista[device_id]);

            checkCudaErrors(cudaMemcpyAsync(dst[device_id], src, SIZEOF_FLOAT3 * size, type));

            cudaDeviceSynchronize();
            }));
    }

    for (auto& thread : threads)
        thread.join();
}

inline void multiGPU_copy(int devices_count, float* dst, std::vector<float*>& src, int size, cudaMemcpyKind type, int deviceIndex) {
    std::vector<std::thread> threads;
    auto lista = enumerate(deviceIndex, devices_count);

    for (unsigned int device_id = 0; device_id < devices_count; device_id++)
    {
        threads.push_back(std::thread([&, device_id]() {
            cudaSetDevice(lista[device_id]);

            checkCudaErrors(cudaMemcpyAsync(dst, src[device_id], sizeof(float) * size, type));

            cudaDeviceSynchronize();
            }));
    }

    for (auto& thread : threads)
        thread.join();
}

inline void multiGPU_copy(int devices_count, float3* dst, std::vector<float3*>& src, int size, cudaMemcpyKind type, int deviceIndex) {
    std::vector<std::thread> threads;
    auto lista = enumerate(deviceIndex, devices_count);

    for (unsigned int device_id = 0; device_id < devices_count; device_id++)
    {
        threads.push_back(std::thread([&, device_id]() {
            cudaSetDevice(lista[device_id]);

            checkCudaErrors(cudaMemcpyAsync(dst, src[device_id], SIZEOF_FLOAT3 * size, type));

            cudaDeviceSynchronize();
            }));
    }

    for (auto& thread : threads)
        thread.join();
}

inline void multiGPU_copy(int devices_count, float* dst, float* src, int size, cudaMemcpyKind type, int deviceIndex) {
    std::vector<std::thread> threads;
    auto lista = enumerate(deviceIndex, devices_count);

    for (unsigned int device_id = 0; device_id < devices_count; device_id++)
    {
        threads.push_back(std::thread([&, device_id]() {
            cudaSetDevice(lista[device_id]);

            checkCudaErrors(cudaMemcpyAsync(dst, src, sizeof(float) * size, type));

            cudaDeviceSynchronize();
            }));
    }

    for (auto& thread : threads)
        thread.join();
}

inline void multiGPU_copy(int devices_count, float3* dst, float3* src, int size, cudaMemcpyKind type, int deviceIndex) {
    std::vector<std::thread> threads;
    auto lista = enumerate(deviceIndex, devices_count);

    for (unsigned int device_id = 0; device_id < devices_count; device_id++)
    {
        threads.push_back(std::thread([&, device_id]() {
            cudaSetDevice(lista[device_id]);

            checkCudaErrors(cudaMemcpyAsync(dst, src, SIZEOF_FLOAT3 * size, type));

            cudaDeviceSynchronize();
            }));
    }

    for (auto& thread : threads)
        thread.join();
}

inline void multiGPU_copy(int devices_count, std::vector<float*>* dst, std::vector<float*>* src, int size, cudaMemcpyKind type, int deviceIndex) { //dobre
    std::vector<std::thread> threads;
    auto lista = enumerate(deviceIndex, devices_count);
    for (unsigned int device_id = 0; device_id < devices_count; device_id++)
    {
        threads.push_back(std::thread([&, device_id]() {
            cudaSetDevice(lista[device_id]);

            checkCudaErrors(cudaMemcpyAsync(dst->at(device_id), src->at(device_id), sizeof(float) * size, type));

            cudaDeviceSynchronize();
            }));
    }

    for (auto& thread : threads)
        thread.join();
}

inline void multiGPU_copyn(int devices_count, std::vector<float*>* dst, std::vector<float*>* src, int size, cudaMemcpyKind type, int deviceIndex, std::string message = "") { //dobre
    std::vector<std::thread> threads;
    auto lista = enumerate(deviceIndex, devices_count);
    for (unsigned int device_id = 0; device_id < devices_count; device_id++)
    {
        threads.push_back(std::thread([&, device_id]() {
            cudaSetDevice(lista[device_id]);
            std::cout << "Oł shieeeeeeeeeeeeeet";

            std::cout << "COpy -> " << device_id << "  at  " << message << std::endl;
            checkCudaErrors(cudaMemcpyAsync(dst->at(device_id), src->at(device_id), sizeof(float) * size, type));

            cudaDeviceSynchronize();
            }));
    }

    for (auto& thread : threads)
        thread.join();
}

inline void multiGPU_copy(int devices_count, std::vector<float3*>& dst, std::vector<float3*>& src, int size, cudaMemcpyKind type, int deviceIndex) {
    std::vector<std::thread> threads;
    auto lista = enumerate(deviceIndex, devices_count);

    for (unsigned int device_id = 0; device_id < devices_count; device_id++)
    {
        threads.push_back(std::thread([&, device_id]() {
            cudaSetDevice(lista[device_id]);

            checkCudaErrors(cudaMemcpyAsync(dst[device_id], src[device_id], SIZEOF_FLOAT3 * size, type));

            cudaDeviceSynchronize();
            }));
    }

    for (auto& thread : threads)
        thread.join();
}


template <typename T>
inline std::vector<T*> multiGPU_malloc(int devices_count, int deviceIndex, long long size) {
    std::vector<std::thread> threads;
    auto lista = enumerate(deviceIndex, devices_count);

    std::vector<T*> _dst;
    for (unsigned int device_id = 0; device_id < devices_count; device_id++)
        _dst.push_back(new T);

    std::mutex m1;
    for (unsigned int device_id = 0; device_id < devices_count; device_id++)
    {
        threads.push_back(std::thread([&, device_id]() {
            cudaSetDevice(lista[device_id]);

            std::cout << device_id;

#ifndef HOSTALLOC
            checkCudaErrors(cudaMalloc((void**)&_dst[device_id], sizeof(T) * size));
#else
            checkCudaErrors(cudaHostAlloc(&_dst[device_id], size * sizeof(T), cudaHostAllocMapped));
#endif

            cudaDeviceSynchronize();
            std::cout << device_id;
         }));
    }

    
    for (auto& thread : threads)
        thread.join();
    std::cout << "Done";

    //allow_p2p_sharing(devices_count);

    return _dst;
}



inline std::vector<float3*> multiGPU_malloc3(int devices_count, int deviceIndex, long long size) {
    std::vector<std::thread> threads;
    auto lista = enumerate(deviceIndex, devices_count);

    std::vector<float3*> _dst;
    for (unsigned int device_id = 0; device_id < devices_count; device_id++)
        _dst.push_back(new float3);


    std::mutex m1;
    for (unsigned int device_id = 0; device_id < devices_count; device_id++)
    {
        threads.push_back(std::thread([&, device_id]() {
            cudaSetDevice(lista[device_id]);
            std::cout << device_id;

#ifndef HOSTALLOC
            checkCudaErrors(cudaMalloc((void**)&_dst[device_id], SIZEOF_FLOAT3 * size));
#else
            checkCudaErrors(cudaHostAlloc(&_dst[device_id], size * SIZEOF_FLOAT3, cudaHostAllocMapped));
#endif

            cudaDeviceSynchronize();
            }));
    }

    for (auto& thread : threads)
        thread.join();
    std::cout << "Done";

    //allow_p2p_sharing(devices_count);

    return _dst;
}




inline void multiGPU_free(int devices_count, std::vector<float*>& dst) {
    for (auto device : dst)
    {
        checkCudaErrors(cudaFree(device));
    }
    dst.clear();
}

inline void multiGPU_free(int devices_count, std::vector<float3*>& dst) {
    for (auto device : dst)
    {
        checkCudaErrors(cudaFree(device));
    }
    dst.clear();
}




class GRID3D {


    void deletep(float&) {}
    void deletep(float*& ptr) {
        delete[] ptr;
        ptr = nullptr;
    }
    void deletec(float3&) {}
    void deletec(float3*& ptr) {
        delete[] ptr;
        ptr = nullptr;
    }
    void initNoiseGrid(int deviceIndex, bool vell = true) {
        grid_noise = new float[1];
        grid_noise[0] = SUPER_NULL;

        if (vell) {
            grid_vel = new float3[size()];
            grid_vel[0] = make_float3(SUPER_NULL,SUPER_NULL,SUPER_NULL);
        }
        if (!cuda_velocity_initialized) {
            cuda_velocity_initialized = true;
            grid_vel_gpu = multiGPU_malloc3(deviceCount, deviceIndex, size());
        }

        
    }
    bool cuda_velocity_initialized = false;
public:
    GRID3D(int devicesCount=1, int deviceIndex=0) {
        this->deviceCount = devicesCount;
        cudaSetDevice(deviceIndex);
        resolution.x = resolution.y = resolution.z = 1;
        grid = new float[1];
        grid[0] = 0.0;
        grid_temp = new float[1];
        grid_temp[0] = 0.0;
        initNoiseGrid(deviceIndex);
        //grid_noise = new float[1];
        //grid_noise[0] = 0.0;
    }
    GRID3D(int x, int y, int z, int deviceCount, int deviceIndex) {
        this->deviceCount = deviceCount;
        resolution.x = x;
        resolution.y = y;
        resolution.z = z;

        //grid_noise = new float[1];
        //grid_noise[0] = 0.0;
#ifdef NEW_HOST_ALLOC
        checkCudaErrors(cudaHostAlloc(&grid, size() * sizeof(float), cudaHostAllocMapped));
        checkCudaErrors(cudaHostAlloc(&grid_temp, size() * sizeof(float), cudaHostAllocMapped));
#else
        grid = new float[(long long)x * (long long)y * (long long)z];
        grid_temp = new float[(long long)x * (long long)y * (long long)z];
#endif
        for (long long i = 0; i < size(); i++) {
            grid[i] = 0.0;
            grid_temp[i] = 0.0;
        }
        initNoiseGrid(deviceIndex);
    }
    GRID3D(int3 dim, int deviceCount, int deviceIndex) {
        this->deviceCount = deviceCount;
        resolution.x = dim.x;
        resolution.y = dim.y;
        resolution.z = dim.z;

        //grid_noise = new float[1];
        //grid_noise[0] = 0.0;
#ifdef NEW_HOST_ALLOC
        checkCudaErrors(cudaHostAlloc(&grid, size() * sizeof(float), cudaHostAllocMapped));
        checkCudaErrors(cudaHostAlloc(&grid_temp, size() * sizeof(float), cudaHostAllocMapped));
#else
        grid = new float[size()];
        grid_temp = new float[size()];
#endif
        for (long long i = 0; i < size(); i++) {
            grid[i] = 0.0;
            grid_temp[i] = 0.0;
        }
        initNoiseGrid(deviceIndex);
    }
    GRID3D(int elem, float* grid, int deviceCount, int deviceIndex) {
        this->deviceCount = deviceCount;

#ifdef NEW_HOST_ALLOC
        checkCudaErrors(cudaHostAlloc(&grid, elem * sizeof(float), cudaHostAllocMapped));
        checkCudaErrors(cudaHostAlloc(&grid_temp, elem * sizeof(float), cudaHostAllocMapped));
#else
        grid = new float[elem];
        grid_temp = new float[elem];
#endif
        for (int i = 0; i < elem; i++) {
            this->grid[i] = grid[i];
            this->grid_temp[i] = grid[i];
        }
        //grid_noise = new float[1];
        //grid_noise[0] = 0.0;
        initNoiseGrid(deviceIndex);
    }
    GRID3D(int3 dim, float* grid_src, int deviceCount, int deviceIndex) {
        this->deviceCount = deviceCount;
        this->resolution = dim;
        //grid = new float[(long long)dim.x * (long long)dim.y * (long long)dim.z];
#ifdef NEW_HOST_ALLOC
        checkCudaErrors(cudaHostAlloc(&grid, size() * sizeof(float), cudaHostAllocMapped));
        checkCudaErrors(cudaHostAlloc(&grid_temp, size() * sizeof(float), cudaHostAllocMapped));
#else
        grid = new float[size()];
        grid_temp = new float[size()];
#endif
        //cudaMemcpyAsync(grid, grid_src, sizeof(float) * size(), cudaMemcpyDeviceToHost,0);
        multiGPU_copy(deviceCount, grid, grid_src, size(), cudaMemcpyDeviceToHost, deviceIndex);
        grid_temp = new float[1];
        initNoiseGrid(deviceIndex);
    }

    void load_from_device(int3 dim, float* grid_src, int deviceIndex,bool debug = false) {
        freeOnlyGrid(); //free
        this->resolution = dim;
#ifdef NEW_HOST_ALLOC
        checkCudaErrors(cudaHostAlloc(&grid, size() * sizeof(float), cudaHostAllocMapped));
        checkCudaErrors(cudaHostAlloc(&grid_temp, size() * sizeof(float), cudaHostAllocMapped));
#else
        grid = new float[size()];
        grid_temp = new float[size()];
#endif
        //checkCudaErrors(cudaMemcpyAsync(grid, grid_src, sizeof(float) * size(), cudaMemcpyDeviceToHost,0));
        multiGPU_copy(deviceCount, grid, grid_src, size(), cudaMemcpyDeviceToHost, deviceIndex);



        //cudaCheckError();
    }

#ifdef VELOCITY_NOISE
    void load_from_device3D(int3 dim, float3* grid_src, int deviceIndex) {

        if (this->resolution.x == dim.x && this->resolution.y == dim.y && this->resolution.z == dim.z) {
            //std::cout << "Copying";
            //checkCudaErrors(cudaMemcpyAsync(grid_vel, grid_src, SIZEOF_FLOAT3 * size(), cudaMemcpyDeviceToHost,0));
            multiGPU_copy(deviceCount, grid_vel, grid_src, size(), cudaMemcpyDeviceToHost, deviceIndex);
        }
        else {
            //std::cout << "Free data";
            this->free_velocity();
            this->resolution = dim;
            //std::cout << "Allocating memory";
            grid_vel = new float3[size()];
            //std::cout << "Copying";
            //checkCudaErrors(cudaMemcpyAsync(grid_vel, grid_src, SIZEOF_FLOAT3 * size(), cudaMemcpyDeviceToHost,0));
            multiGPU_copy(deviceCount, grid_vel, grid_src, size(), cudaMemcpyDeviceToHost, deviceIndex);
        }
        //cudaCheckError();
        //std::cout << "Copied from device" << std::endl;
    }
#endif

    GRID3D(int x, int y, int z, float* vdb, int deviceCount, int deviceIndex) {
        resolution.x = x;
        resolution.y = y;
        resolution.z = z;

#ifdef NEW_HOST_ALLOC
        checkCudaErrors(cudaHostAlloc(&grid, size() * sizeof(float), cudaHostAllocMapped));
        checkCudaErrors(cudaHostAlloc(&grid_temp, size() * sizeof(float), cudaHostAllocMapped));
#else
        grid = new float[size()];
        grid_temp = new float[size()];
#endif
        for (long long i = 0; i < size(); i++) {
            grid[i] = vdb[i];
            grid_temp[i] = vdb[i];
        }
        //grid_noise = new float[1];
        //grid_noise[0] = 0.0;
        initNoiseGrid(deviceIndex);
    }
    float operator()(int x, int y, int z) {
        float output = 0.0;
        long long iter = z * resolution.y * resolution.x + y * resolution.x + x;
        if (iter <= size())
            output = grid[iter];
        else {
            std::cout << "GRID READ ERROR:\n";
            std::cout << "Max ID:   " << size() << "\nGiven ID: " << iter << "\n";
        }
        return output;
    }

    float operator()(openvdb::Coord ijk) {
        float output = 0.0;
        long long iter = ijk[2] * resolution.y * resolution.x + ijk[1] * resolution.x + ijk[0];
        if (iter <= size())
            output = grid[iter];
        else {
            std::cout << "GRID READ ERROR:\n";
            std::cout << "Max ID:   " << size() << "\nGiven ID: " << iter << "\n";
        }
        return output;
    }

    float get(openvdb::Coord ijk) {
        float output = 0.0;
        long long iter = ijk[2] * resolution.y * resolution.x + ijk[1] * resolution.x + ijk[0];
        if (iter <= size())
            output = grid[iter];
        else {
            std::cout << "GRID READ ERROR:\n";
            std::cout << "Max ID:   " << size() << "\nGiven ID: " << iter << "\n";
        }
        return output;
    }

    float get(nanovdb::Coord ijk) {
        float output = 0.0;
        long long iter = ijk.z() * resolution.y * resolution.x + ijk.y() * resolution.x + ijk.x();
        if (iter <= size())
            output = grid[iter];
        else {
            std::cout << "GRID READ ERROR:\n";
            std::cout << "Max ID:   " << size() << "\nGiven ID: " << iter << "\n";
        }
        return output;
    }

    GRID3D operator=(const GRID3D& rhs) {
        free();
        resolution.x = rhs.resolution.x;
        resolution.y = rhs.resolution.y;
        resolution.z = rhs.resolution.z;

#ifdef NEW_HOST_ALLOC
        checkCudaErrors(cudaHostAlloc(&grid, size() * sizeof(float), cudaHostAllocMapped));
        checkCudaErrors(cudaHostAlloc(&grid_temp, size() * sizeof(float), cudaHostAllocMapped));
#else
        grid = new float[size()];
        grid_temp = new float[size()];
#endif
        for (long long i = 0; i < size(); i++) {
            grid[i] = rhs.grid[i];
            grid_temp[i] = rhs.grid_temp[i];
        }
        //grid_noise = new float[1];
        //grid_noise[0] = 0.0;
        //initNoiseGrid();
        return *this;
    }
    GRID3D operator=(const GRID3D* rhs) {
        free();
        resolution.x = rhs->resolution.x;
        resolution.y = rhs->resolution.y;
        resolution.z = rhs->resolution.z;

        grid = rhs->grid;
        grid_temp = rhs->grid_temp;
        
        //grid_noise = new float[1];
        //grid_noise[0] = 0.0;
        //initNoiseGrid();
        return *this;
    }

    void set_pointer(const GRID3D* rhs) {
        free();
        resolution.x = rhs->resolution.x;
        resolution.y = rhs->resolution.y;
        resolution.z = rhs->resolution.z;

        grid = rhs->grid;
        grid_temp = rhs->grid_temp;
        //grid_noise = new float[1];
        //initNoiseGrid();
    }

    GRID3D load(const GRID3D* rhs) {
        free();
        resolution.x = rhs->resolution.x;
        resolution.y = rhs->resolution.y;
        resolution.z = rhs->resolution.z;

#ifdef NEW_HOST_ALLOC
        checkCudaErrors(cudaHostAlloc(&grid, size() * sizeof(float), cudaHostAllocMapped));
        checkCudaErrors(cudaHostAlloc(&grid_temp, size() * sizeof(float), cudaHostAllocMapped));
#else
        grid = new float[size()];
        grid_temp = new float[size()];
#endif
        for (long long i = 0; i < size(); i++) {
            grid[i] = rhs->grid[i];
            grid_temp[i] = rhs->grid_temp[i];
        }
        //grid_noise = new float[1];
        //grid_noise[0] = 0.0;
        //initNoiseGrid();
        return *this;
    }

    void combine_with_temp_grid(const GRID3D* rhs) {
        deletep(grid_temp);
        grid_temp = rhs->grid;
    }

    void normalizeData() {
        for (int i = 0; i < size(); i++) {
            if (grid[i] < 0.01)
                grid[i] = 0.0f;
            grid[i] = fmin(grid[i], 1.0);
            grid[i] = fmax(grid[i], 0.0);
        }
    }

    void addNoise() {
        float ratio = 10000.0 / 300.0;
        ratio = ratio * 0.5;
        for (int i = 0; i < size(); i++) {
            if (grid_temp[i] > 0.0) {
                float random = (float)(rand() % 10000) / 300.0;
                random -= ratio;

                grid_temp[i] += random;
                grid_temp[i] = max(0.5f, grid_temp[i]);
            }
        }
    }

    void copyToDevice(int deviceIndex, bool addNoise = true) {
        if (addNoise)
            this->addNoise();
        //checkCudaErrors(cudaMalloc((void**)&vdb_temp, sizeof(float) * size()));
        std::cout << "Malloc";
        vdb_temp = multiGPU_malloc<float>(deviceCount, deviceIndex, size());
        //checkCudaErrors(cudaMemcpyAsync(vdb_temp, grid_temp, sizeof(float) * size(), cudaMemcpyHostToDevice,0));
        std::cout << "Copy";
        multiGPU_copyHTD(deviceCount, &vdb_temp, grid_temp, size(), deviceIndex);

        normalizeData();
        //copy to device
        //checkCudaErrors(cudaMalloc((void**)&vdb, sizeof(float) * size()));
        vdb = multiGPU_malloc<float>(deviceCount, deviceIndex, size());
        //checkCudaErrors(cudaMemcpyAsync(vdb, grid, sizeof(float) * size(), cudaMemcpyHostToDevice,0));
        multiGPU_copyHTD(deviceCount, &vdb, grid, size(), deviceIndex);
        //cudaCheckError();
    }
#ifdef VELOCITY_NOISE
    void copyToDeviceVel(int deviceIndex) {
        //checkCudaErrors(cudaMalloc((void**)&grid_vel_gpu, SIZEOF_FLOAT3 * size()));
        grid_vel_gpu = multiGPU_malloc3(deviceCount, deviceIndex, size());
        //checkCudaErrors(cudaMemcpyAsync(grid_vel_gpu, grid_vel, SIZEOF_FLOAT3 * size(), cudaMemcpyHostToDevice,0));
        multiGPU_copy(deviceCount, grid_vel_gpu, grid_vel, size(), cudaMemcpyHostToDevice, deviceIndex);
        //cudaCheckError();
    }
#endif
    void copyToDeviceNoise(int NTS, int deviceIndex, int deviceCount = 1) {
        //checkCudaErrors(cudaMalloc((void**)&vdb_noise, sizeof(float) * NTS * NTS * NTS));
        std::cout << "Malloc << " << NTS;
        vdb_noise = multiGPU_malloc<float>(deviceCount, deviceIndex, NTS*NTS*NTS);

        //this->generateTile(NTS);

        //checkCudaErrors(cudaMemcpyAsync(vdb_noise, grid_noise, sizeof(float) * NTS * NTS * NTS, cudaMemcpyHostToDevice,0));
        std::cout << "Copy(" << deviceCount << ")";
        multiGPU_copyHTD(deviceCount, &vdb_noise, grid_noise, NTS*NTS*NTS, deviceIndex);
        std::cout << "Endl\n";
        //cudaCheckError();
    }


    void set(int x, int y, int z, float value) {
        grid[z * resolution.y * resolution.x + y * resolution.x + x] = value;
    }
    void set_temp(int x, int y, int z, float value) {
        grid_temp[z * resolution.y * resolution.x + y * resolution.x + x] = value;
    }
    int3 get_resolution() {
        return resolution;
    }
    int get_max_resolution() {
        return max(max(resolution.x, resolution.y), resolution.z);
    }
    void freeOnlyGrid() {
        deletep(grid);
    }
    void free() {
        //std::cout << "Free grid memory" << std::endl;
        deletep(grid);
        deletep(grid_temp);
        //deletep(grid_noise);
    }

#ifdef VELOCITY_NOISE
    void free_velocity() {
        deletec(grid_vel);
    }
#endif

    void free_noise() {
        deletep(grid_noise);
        grid_noise = new float[0];
        grid_noise[0] = SUPER_NULL;
    }
    void freeCuda() {
        multiGPU_free(deviceCount, vdb);
        multiGPU_free(deviceCount, vdb_temp);
    }
    void freeCuda1() {
        multiGPU_free(deviceCount, vdb);
    }
    void freeNoise() {
        multiGPU_free(deviceCount, vdb_noise);
    }
#ifdef VELOCITY_NOISE
    void freeCudaVel() {
        multiGPU_free(deviceCount, grid_vel_gpu);
        this->cuda_velocity_initialized = false;
    }
#endif














    ~GRID3D() {
        //free();
    }
    float* get_grid() const {
        return this->grid;
    }
    float* get_grid_temp() const {
        return this->grid_temp;
    }

    std::vector<float*>* get_grid_device() {
        return &this->vdb;
    }
    std::vector<float*>* get_grid_device_temp() {
        return &this->vdb_temp;
    }

    std::vector<float*>* get_grid_device_noise() {
        return &this->vdb_noise;
    }

#ifdef VELOCITY_NOISE
    std::vector<float3*>* get_grid_device_vel() {
        return &this->grid_vel_gpu;
    }
#endif

    void UpScale(int power, int SEED = 2, int frame = 0, float offset = 0.5, float scale = 0.1, int noise_scale = 128,
        int apply_method = 0, float intensity = 1, float time_anim = 0.1) {
        int noise_tile_size = power * min(min(resolution.x, resolution.y) //max max
            , resolution.z);

        noise_tile_size = noise_scale;

        srand(SEED);
        if (this->grid_noise[0] < -100)
            generateTile(noise_tile_size);

        if (apply_method == 0)
            applyNoise(intensity, noise_tile_size, offset, scale, frame, time_anim);
        else if (apply_method == 1)
            applyNoise2(intensity, noise_tile_size, offset, scale, frame, time_anim);
        else if (apply_method == 2)
            applyCurl(intensity, noise_tile_size, offset, scale, frame,time_anim);

    }

    void LoadNoise(GRID3D* rhs) {
        this->grid_noise = rhs->grid_noise;
    }

#ifdef VELOCITY_NOISE
    void LoadVelocity(GRID3D* rhs) {
        this->grid_vel = rhs->grid_vel;
    }
#endif

    bool is_noise_grid() {
        if (this->grid_noise[0] < -100)
            return false;
        else
            return true;
    }

    float get_noise_status() {
        return this->grid_noise[0];
    }

    inline float evaluate(float3 pos, int tile, int3 resolution, int NTS = 0, float offset = 0.5, float scale = 0.1,
        float time_anim = 0.1)
    {

        pos.x *= resolution.x;
        pos.y *= resolution.y;
        pos.z *= resolution.z;
        pos.x += 1; pos.y += 1; pos.z += 1;

        // time anim
        pos.x += time_anim; pos.y += time_anim; pos.z += time_anim;

        pos.x *= scale;
        pos.y *= scale;
        pos.z *= scale;


        const int n3 = NTS * NTS * NTS;
        float v = WNoiseDx(pos, &this->grid_noise[int(tile * NTS) % n3], NTS);
        //float v = WNoise(pos, &this->grid_noise[int(tile * n3 * 0.01) % n3], NOISE_TILE_SIZE);

        v += offset;//offset //0.5
        //v *= scale;//scale //0.1
        return v;
    }

    void applyNoise(float intensity = 0.2f, int NTS = 0, float offset = 0.5, float scale = 0.1, int frame = 0,
        float time_anim = 0.1) {
        if (NTS == 0)
            NTS = min(min(resolution.x, resolution.y), resolution.z);
        int NTS2 = NTS * NTS;
        int NTS3 = NTS2 * NTS;
        //std::cout << "Applying noise" << std::endl;


        //offset *= 1.4; //1.2

        float tempp = 0.0;
        int THREADS = 16;
        int sizee = ceil((double)resolution.x / (double)THREADS);
        tbb::parallel_for(0, THREADS, [&](int i) {
            int end = (i * sizee) + (sizee);
            if (end > resolution.x) {
                end = resolution.x;
            }
            for (int x = i * sizee; x < end; x++)



                for (int y = 0; y < resolution.y; y++)
                    for (int z = 0; z < resolution.z; z++) {
                        float* position = &this->grid[z * resolution.x * resolution.y +
                            y * resolution.x + x];

                        //if (*position >= 0.01) {
                        if (*position != 0.0) {
                            *position += evaluate(make_float3(x, y, z), frame % 512, resolution, NTS, offset, scale, time_anim) * intensity * fmax(0.05, fmin((*position), 1.0));
                        }

                    }
            });
    }

    void applyNoise2(float intensity = 0.2f, int NTS = 0, float offset = 0.5, float scale = 0.1, int frame = 0,
        float time_anim = 0.1) {
        if (NTS == 0)
            NTS = min(min(resolution.x, resolution.y), resolution.z);
        int NTS2 = NTS * NTS;
        int NTS3 = NTS2 * NTS;
        //std::cout << "Applying noise" << std::endl;

        float tempp = 0.0;
        int THREADS = 16;
        int sizee = ceil((double)resolution.x / (double)THREADS);

        //weights
        //float* weights = new float[size()];
        //computeCoeff(grid_noise,NTS);
        //std::cout << "Computed" << std::endl;

        tbb::parallel_for(0, THREADS, [&](int i) {
            int end = (i * sizee) + (sizee);
            if (end > resolution.x) {
                end = resolution.x;
            }
            for (int x = i * sizee; x < end; x++)



                for (int y = 0; y < resolution.y; y++)
                    for (int z = 0; z < resolution.z; z++) {
                        float* position = &this->grid[z * resolution.x * resolution.y +
                            y * resolution.x + x];

                        //if (*position >= 0.01) {
                        if (*position >= 0.01) {
                            //*position += evaluate(make_float3(x, y, z), frame % 512, resolution, NTS, offset, scale, time_anim) * intensity * max(0.01,min((*position), 1.0));
                            *position += evaluate(make_float3(x, y, z), frame % 512, resolution, NTS, offset, scale, time_anim) * intensity * fmax(0.01, fmin((*position), 1.0));
                        }

                    }
            });
        //delete[] weights;
    }
#ifdef VELOCITY_NOISE
    void applyCurl(float intensity = 0.2f, int NTS = 0, float offset = 0.5, float scale = 0.1, int frame = 0,
        float time_anim = 0.1) {

        //std::cout << resolution.x << "x" << resolution.y << "x" << resolution.z << std::endl;

        if (NTS == 0)
            NTS = min(min(resolution.x, resolution.y), resolution.z);
        //std::cout << "Applying noise" << std::endl;

        int THREADS = 32;
        int sizee = ceil((double)resolution.x / (double)THREADS);

        
        tbb::parallel_for(0, THREADS, [&](int i) {
            int end = (i * sizee) + (sizee);
            if (end > resolution.x) {
                end = resolution.x;
            }
            for (int x = i * sizee; x < end; x++)
        /*
        for (int x = 0; x < resolution.x; x++){
            */
                for (int y = 0; y < resolution.y; y++)
                    for (int z = 0; z < resolution.z; z++) {

                        this->grid_vel[z * resolution.x * resolution.y + y * resolution.x + x]
                                +=
                            evaluateCurl(make_float3(x, y, z), resolution, NTS, offset, scale, time_anim, frame % 128)
                                * 
                            (intensity);

                    }
            }
        );
        //std::cout << "Done";
    }
#else
    void applyCurl(float intensity = 0.2f, int NTS = 0, float offset = 0.5, float scale = 0.1, int frame = 0,
        float time_anim = 0.1) {}
#endif



    void computeCoeff(float* input, int DIMS, float* tempIn1 = nullptr, float* tempIn2 = nullptr) {
        // generate tile
        
        const int sx = resolution.x;
        const int sy = resolution.y;
        const int sz = resolution.z;
        
        /*
        const int sx = DIMS;
        const int sy = DIMS;
        const int sz = DIMS;
        */

        const int n3 = sx * sy * sz;
        // just for compatibility with wavelet turb code
        //float* temp13 = &tempIn1[0];
        //float* temp23 = &tempIn2[0];
        float* noise3 = input;
        float* temp13 = new float[size()];
        float* temp23 = new float[size()];
        //initialize
        for (int i = 0; i < n3; i++) {
            temp13[i] = temp23[i] = 0.0f;
        }

        // Steps 2 and 3. Downsample and upsample the tile
        for (int iz = 0; iz < sz; iz++)
            for (int iy = 0; iy < sy; iy++)
            {
                const int i = iz * sx * sy + iy * sx;
                downsample_neumann(&noise3[i], &temp13[i], sx, 1);
                upsample_neumann(&temp13[i], &temp23[i], sx, 1);
            }
        for (int iz = 0; iz < sz; iz++)
            for (int ix = 0; ix < sx; ix++)
            {
                const int i = iz * sx * sy + ix;
                downsample_neumann(&temp23[i], &temp13[i], sy, sx);
                upsample_neumann(&temp13[i], &temp23[i], sy, sx);
            }
        if (true) {
            for (int iy = 0; iy < sy; iy++)
                for (int ix = 0; ix < sx; ix++)
                {
                    const int i = iy * sx + ix;
                    downsample_neumann(&temp23[i], &temp13[i], sz, sy * sx);
                    upsample_neumann(&temp13[i], &temp23[i], sz, sy * sx);
                }
        }

        // Step 4. Subtract out the coarse-scale contribution
        for (int i = 0; i < n3; i++) {
            float residual = noise3[i] - temp23[i];
            temp13[i] = sqrtf(fabs(residual));
        }
        // copy back, and compute actual weight for wavelet turbulence...
        float smoothingFactor = 1. / 6.;
        if (true) smoothingFactor = 1. / 4.;
        for (int i = 1; i < sx - 1; i++)
            for (int j = 1; j < sy - 1; j++)
                for (int k = 1; k < sz - 1; k++) {
                    // apply some brute force smoothing
                    float res = temp13[k * sx * sy + j * sx + i - 1] + temp13[k * sx * sy + j * sx + i + 1];
                    res += temp13[k * sx * sy + j * sx + i - sx] + temp13[k * sx * sy + j * sx + i + sx];
                    if (true) res += temp13[k * sx * sy + j * sx + i - sx * sy] + temp13[k * sx * sy + j * sx + i + sx * sy];
                    input[k * sy * sx + j * sx + i] = res * smoothingFactor;
                }
        
    }

    void generateTile(int NOISE_TILE_SIZE) {
        const int n = NOISE_TILE_SIZE;
        const int n3 = n * n * n, n3d = n3 * 3;

        float* noise3 = new float[n3d];

        std::cout << "Generating 3x " << n << "^3 noise tile" << std::endl;
        float* temp13 = new float[n3d];
        float* temp23 = new float[n3d];

        //initialize
        for (int i = 0; i < n3d; i++) {
            temp13[i] = temp23[i] = noise3[i] = 0.0f;
        }

        //STEP 1 - fill the tile with random values from -1 to 1;
        float random = 0.0f;
        for (int i = 0; i < n3d; i++) {
            random = ((float(rand() % 1000) * 2.0) / 1000.0) - 1.0f;
            noise3[i] = random;
        }

        //STEP 2&3 - downsample and upsample the tile
        //for (int tile = 0; tile < 3; tile++) {
        tbb::parallel_for(0, 3, [&](int tile) {
            for (int iy = 0; iy < n; iy++)
                for (int iz = 0; iz < n; iz++) {
                    const int i = iy * n + iz * n * n + tile * n3;
                    downsample(&noise3[i], &temp13[i], n, 1);
                    upsample(&temp13[i], &temp23[i], n, 1);
                }
            for (int ix = 0; ix < n; ix++)
                for (int iz = 0; iz < n; iz++) {
                    const int i = ix + iz * n * n + tile * n3;
                    downsample(&temp23[i], &temp13[i], n, n);
                    upsample(&temp13[i], &temp23[i], n, n);
                }
            for (int ix = 0; ix < n; ix++)
                for (int iy = 0; iy < n; iy++) {
                    const int i = ix + iy * n + tile * n3;
                    downsample(&temp23[i], &temp13[i], n, n * n);
                    upsample(&temp13[i], &temp23[i], n, n * n);
                }
            });

        //STEP 4 - subtract out the coarse-scale contribution
        for (int i = 0; i < n3d; i++) {
            noise3[i] -= temp23[i];
        }

        //STEP 5 - avoid even/odd variance
        int offset = n / 2;
        if (offset % 2 == 0)
            offset++;

        int icnt = 0;
        //for (int tile = 0; tile < 3; tile++)
        tbb::parallel_for(0, 3, [&](int tile) {
            for (int ix = 0; ix < n; ix++)
                for (int iy = 0; iy < n; iy++)
                    for (int iz = 0; iz < n; iz++) {
                        temp13[icnt] = noise3[Mod(ix + offset, n) + Mod(iy + offset, n) * n +
                            Mod(iz + offset, n) * n * n + tile * n3];
                        icnt++;
                    }
            });
        for (int i = 0; i < n3d; i++) {
            noise3[i] += temp13[i];
        }

        delete[] this->grid_noise;
        this->grid_noise = noise3;

        delete[] temp13;
        delete[] temp23;
    }

    //#define modFast128(x)  ((x) & 127)

#define ADD_WEIGHTED(x, y, z) \
  weight = 1.0f; \
  xC = Mod(midX + (x),NOISE_TILE_SIZE); \
  weight *= w[0][(x) + 1]; \
  yC = Mod(midY + (y),NOISE_TILE_SIZE); \
  weight *= w[1][(y) + 1]; \
  zC = Mod(midZ + (z),NOISE_TILE_SIZE); \
  weight *= w[2][(z) + 1]; \
  result += weight * data[(zC * NOISE_TILE_SIZE + yC) * NOISE_TILE_SIZE + xC];

    float WNoise(float3& p, float* data, int max_dim = 128) {
        float w[3][3], t, result = 0;
        const int NOISE_TILE_SIZE = max_dim;

        // Evaluate quadratic B-spline basis functions
        int midX = (int)ceilf(p.x - 0.5f);
        t = midX - (p.x - 0.5f);
        w[0][0] = t * t * 0.5f;
        w[0][2] = (1.f - t) * (1.f - t) * 0.5f;
        w[0][1] = 1.f - w[0][0] - w[0][2];

        int midY = (int)ceilf(p.y - 0.5f);
        t = midY - (p.y - 0.5f);
        w[1][0] = t * t * 0.5f;
        w[1][2] = (1.f - t) * (1.f - t) * 0.5f;
        w[1][1] = 1.f - w[1][0] - w[1][2];

        int midZ = (int)ceilf(p.z - 0.5f);
        t = midZ - (p.z - 0.5f);
        w[2][0] = t * t * 0.5f;
        w[2][2] = (1.f - t) * (1.f - t) * 0.5f;
        w[2][1] = 1.f - w[2][0] - w[2][2];

        // Evaluate noise by weighting noise coefficients by basis function values
        int xC, yC, zC;
        float weight = 1;

        ADD_WEIGHTED(-1, -1, -1);
        ADD_WEIGHTED(0, -1, -1);
        ADD_WEIGHTED(1, -1, -1);
        ADD_WEIGHTED(-1, 0, -1);
        ADD_WEIGHTED(0, 0, -1);
        ADD_WEIGHTED(1, 0, -1);
        ADD_WEIGHTED(-1, 1, -1);
        ADD_WEIGHTED(0, 1, -1);
        ADD_WEIGHTED(1, 1, -1);

        ADD_WEIGHTED(-1, -1, 0);
        ADD_WEIGHTED(0, -1, 0);
        ADD_WEIGHTED(1, -1, 0);
        ADD_WEIGHTED(-1, 0, 0);
        ADD_WEIGHTED(0, 0, 0);
        ADD_WEIGHTED(1, 0, 0);
        ADD_WEIGHTED(-1, 1, 0);
        ADD_WEIGHTED(0, 1, 0);
        ADD_WEIGHTED(1, 1, 0);

        ADD_WEIGHTED(-1, -1, 1);
        ADD_WEIGHTED(0, -1, 1);
        ADD_WEIGHTED(1, -1, 1);
        ADD_WEIGHTED(-1, 0, 1);
        ADD_WEIGHTED(0, 0, 1);
        ADD_WEIGHTED(1, 0, 1);
        ADD_WEIGHTED(-1, 1, 1);
        ADD_WEIGHTED(0, 1, 1);
        ADD_WEIGHTED(1, 1, 1);

        return result;
    }


    float WNoiseDx(float3& p, float* data, int max_dim = 128) {
        float w[3][3], t, result = 0;
        const int NOISE_TILE_SIZE = max_dim;



        // Evaluate quadratic B-spline basis functions
        int midX = (int)ceil(p.x - 0.5f);
        t = midX - (p.x - 0.5f);
        w[0][0] = -t;
        w[0][2] = (1.f - t);
        w[0][1] = 2.0f * t - 1.0f;

        int midY = (int)ceil(p.y - 0.5f);
        t = midY - (p.y - 0.5f);
        w[1][0] = t * t * 0.5f;
        w[1][2] = (1.f - t) * (1.f - t) * 0.5f;
        w[1][1] = 1.f - w[1][0] - w[1][2];

        int midZ = (int)ceil(p.z - 0.5f);
        t = midZ - (p.z - 0.5f);
        w[2][0] = t * t * 0.5f;
        w[2][2] = (1.f - t) * (1.f - t) * 0.5f;
        w[2][1] = 1.f - w[2][0] - w[2][2];

        // Evaluate noise by weighting noise coefficients by basis function values
        int xC, yC, zC;
        float weight = 1;

        ADD_WEIGHTED(-1, -1, -1); ADD_WEIGHTED(0, -1, -1); ADD_WEIGHTED(1, -1, -1);
        ADD_WEIGHTED(-1, 0, -1); ADD_WEIGHTED(0, 0, -1); ADD_WEIGHTED(1, 0, -1);
        ADD_WEIGHTED(-1, 1, -1); ADD_WEIGHTED(0, 1, -1); ADD_WEIGHTED(1, 1, -1);

        ADD_WEIGHTED(-1, -1, 0);  ADD_WEIGHTED(0, -1, 0);  ADD_WEIGHTED(1, -1, 0);
        ADD_WEIGHTED(-1, 0, 0);  ADD_WEIGHTED(0, 0, 0);  ADD_WEIGHTED(1, 0, 0);
        ADD_WEIGHTED(-1, 1, 0);  ADD_WEIGHTED(0, 1, 0);  ADD_WEIGHTED(1, 1, 0);

        ADD_WEIGHTED(-1, -1, 1);  ADD_WEIGHTED(0, -1, 1);  ADD_WEIGHTED(1, -1, 1);
        ADD_WEIGHTED(-1, 0, 1);  ADD_WEIGHTED(0, 0, 1);  ADD_WEIGHTED(1, 0, 1);
        ADD_WEIGHTED(-1, 1, 1);  ADD_WEIGHTED(0, 1, 1);  ADD_WEIGHTED(1, 1, 1);

        return result;

    }

#define ADD_WEIGHTEDX(x,y,z)\
  weight = dw[0][(x) + 1] * w[1][(y) + 1] * w[2][(z) + 1];\
  result += weight * neighbors[x + 1][y + 1][z + 1];

#define ADD_WEIGHTEDY(x,y,z)\
  weight = w[0][(x) + 1] * dw[1][(y) + 1] * w[2][(z) + 1];\
  result += weight * neighbors[x + 1][y + 1][z + 1];

#define ADD_WEIGHTEDZ(x,y,z)\
  weight = w[0][(x) + 1] * w[1][(y) + 1] * dw[2][(z) + 1];\
  result += weight * neighbors[x + 1][y + 1][z + 1];


    static inline float3 WNoiseVec(float3& p, float* data, int max_dim = 128) {

        float3 final = make_float3(0, 0, 0);
        const int NOISE_TILE_SIZE = max_dim;
        float w[3][3];
        float dw[3][3];
        float result = 0;
        int xC, yC, zC;
        float weight;

        int midX = (int)ceil(p.x - 0.5f);
        int midY = (int)ceil(p.y - 0.5f);
        int midZ = (int)ceil(p.z - 0.5f);

        float t0 = midX - (p.x - 0.5f);
        float t1 = midY - (p.y - 0.5f);
        float t2 = midZ - (p.z - 0.5f);

        // precache all the neighbors for fast access
        float neighbors[3][3][3];
        for (int z = -1; z <= 1; z++)
            for (int y = -1; y <= 1; y++)
                for (int x = -1; x <= 1; x++)
                {
                    xC = Mod(midX + (x), NOISE_TILE_SIZE);
                    yC = Mod(midY + (y), NOISE_TILE_SIZE);
                    zC = Mod(midZ + (z), NOISE_TILE_SIZE);
                    neighbors[x + 1][y + 1][z + 1] = data[zC * NOISE_TILE_SIZE * NOISE_TILE_SIZE + yC * NOISE_TILE_SIZE + xC];
                }

        ///////////////////////////////////////////////////////////////////////////////////////
        // evaluate splines
        ///////////////////////////////////////////////////////////////////////////////////////
        dw[0][0] = -t0;
        dw[0][2] = (1.f - t0);
        dw[0][1] = 2.0f * t0 - 1.0f;

        dw[1][0] = -t1;
        dw[1][2] = (1.0f - t1);
        dw[1][1] = 2.0f * t1 - 1.0f;

        dw[2][0] = -t2;
        dw[2][2] = (1.0f - t2);
        dw[2][1] = 2.0f * t2 - 1.0f;

        w[0][0] = t0 * t0 * 0.5f;
        w[0][2] = (1.f - t0) * (1.f - t0) * 0.5f;
        w[0][1] = 1.f - w[0][0] - w[0][2];

        w[1][0] = t1 * t1 * 0.5f;
        w[1][2] = (1.f - t1) * (1.f - t1) * 0.5f;
        w[1][1] = 1.f - w[1][0] - w[1][2];

        w[2][0] = t2 * t2 * 0.5f;
        w[2][2] = (1.f - t2) * (1.f - t2) * 0.5f;
        w[2][1] = 1.f - w[2][0] - w[2][2];

        ///////////////////////////////////////////////////////////////////////////////////////
        // x derivative
        ///////////////////////////////////////////////////////////////////////////////////////
        result = 0.0f;
        ADD_WEIGHTEDX(-1, -1, -1); ADD_WEIGHTEDX(0, -1, -1); ADD_WEIGHTEDX(1, -1, -1);
        ADD_WEIGHTEDX(-1, 0, -1); ADD_WEIGHTEDX(0, 0, -1); ADD_WEIGHTEDX(1, 0, -1);
        ADD_WEIGHTEDX(-1, 1, -1); ADD_WEIGHTEDX(0, 1, -1); ADD_WEIGHTEDX(1, 1, -1);

        ADD_WEIGHTEDX(-1, -1, 0);  ADD_WEIGHTEDX(0, -1, 0);  ADD_WEIGHTEDX(1, -1, 0);
        ADD_WEIGHTEDX(-1, 0, 0);  ADD_WEIGHTEDX(0, 0, 0);  ADD_WEIGHTEDX(1, 0, 0);
        ADD_WEIGHTEDX(-1, 1, 0);  ADD_WEIGHTEDX(0, 1, 0);  ADD_WEIGHTEDX(1, 1, 0);

        ADD_WEIGHTEDX(-1, -1, 1);  ADD_WEIGHTEDX(0, -1, 1);  ADD_WEIGHTEDX(1, -1, 1);
        ADD_WEIGHTEDX(-1, 0, 1);  ADD_WEIGHTEDX(0, 0, 1);  ADD_WEIGHTEDX(1, 0, 1);
        ADD_WEIGHTEDX(-1, 1, 1);  ADD_WEIGHTEDX(0, 1, 1);  ADD_WEIGHTEDX(1, 1, 1);
        final.x = result;

        ///////////////////////////////////////////////////////////////////////////////////////
        // y derivative
        ///////////////////////////////////////////////////////////////////////////////////////
        result = 0.0f;
        ADD_WEIGHTEDY(-1, -1, -1); ADD_WEIGHTEDY(0, -1, -1); ADD_WEIGHTEDY(1, -1, -1);
        ADD_WEIGHTEDY(-1, 0, -1); ADD_WEIGHTEDY(0, 0, -1); ADD_WEIGHTEDY(1, 0, -1);
        ADD_WEIGHTEDY(-1, 1, -1); ADD_WEIGHTEDY(0, 1, -1); ADD_WEIGHTEDY(1, 1, -1);

        ADD_WEIGHTEDY(-1, -1, 0);  ADD_WEIGHTEDY(0, -1, 0);  ADD_WEIGHTEDY(1, -1, 0);
        ADD_WEIGHTEDY(-1, 0, 0);  ADD_WEIGHTEDY(0, 0, 0);  ADD_WEIGHTEDY(1, 0, 0);
        ADD_WEIGHTEDY(-1, 1, 0);  ADD_WEIGHTEDY(0, 1, 0);  ADD_WEIGHTEDY(1, 1, 0);

        ADD_WEIGHTEDY(-1, -1, 1);  ADD_WEIGHTEDY(0, -1, 1);  ADD_WEIGHTEDY(1, -1, 1);
        ADD_WEIGHTEDY(-1, 0, 1);  ADD_WEIGHTEDY(0, 0, 1);  ADD_WEIGHTEDY(1, 0, 1);
        ADD_WEIGHTEDY(-1, 1, 1);  ADD_WEIGHTEDY(0, 1, 1);  ADD_WEIGHTEDY(1, 1, 1);
        final.y = result;

        ///////////////////////////////////////////////////////////////////////////////////////
        // z derivative
        ///////////////////////////////////////////////////////////////////////////////////////
        result = 0.0f;
        ADD_WEIGHTEDZ(-1, -1, -1); ADD_WEIGHTEDZ(0, -1, -1); ADD_WEIGHTEDZ(1, -1, -1);
        ADD_WEIGHTEDZ(-1, 0, -1); ADD_WEIGHTEDZ(0, 0, -1); ADD_WEIGHTEDZ(1, 0, -1);
        ADD_WEIGHTEDZ(-1, 1, -1); ADD_WEIGHTEDZ(0, 1, -1); ADD_WEIGHTEDZ(1, 1, -1);

        ADD_WEIGHTEDZ(-1, -1, 0);  ADD_WEIGHTEDZ(0, -1, 0);  ADD_WEIGHTEDZ(1, -1, 0);
        ADD_WEIGHTEDZ(-1, 0, 0);  ADD_WEIGHTEDZ(0, 0, 0);  ADD_WEIGHTEDZ(1, 0, 0);
        ADD_WEIGHTEDZ(-1, 1, 0);  ADD_WEIGHTEDZ(0, 1, 0);  ADD_WEIGHTEDZ(1, 1, 0);

        ADD_WEIGHTEDZ(-1, -1, 1);  ADD_WEIGHTEDZ(0, -1, 1);  ADD_WEIGHTEDZ(1, -1, 1);
        ADD_WEIGHTEDZ(-1, 0, 1);  ADD_WEIGHTEDZ(0, 0, 1);  ADD_WEIGHTEDZ(1, 0, 1);
        ADD_WEIGHTEDZ(-1, 1, 1);  ADD_WEIGHTEDZ(0, 1, 1);  ADD_WEIGHTEDZ(1, 1, 1);
        final.z = result;

        //std::cout << "FINAL at = " << final.x <<";"<< final.y << ";" << final.z << std::endl; // DEBUG
        return final;
    }

#undef ADD_WEIGHTEDX
#undef ADD_WEIGHTEDY
#undef ADD_WEIGHTEDZ

    inline float3 evaluateVec(float3 pos, int3 resolution, int NTS = 0, float offset = 0.5, float scale = 0.1,
        float time_anim = 0.1, int tile = 0) const {
        //std::cout << "\nTILE: ";
        //std::cout << tile << ";";
        pos.x *= resolution.x;
        pos.y *= resolution.y;
        pos.z *= resolution.z;
        pos.x += 1; pos.y += 1; pos.z += 1;

        // time anim
        pos.x += time_anim; pos.y += time_anim; pos.z += time_anim;

        pos.x *= scale;
        pos.y *= scale;
        pos.z *= scale;

        const int n3 = NTS * NTS * NTS;
        float3 v = WNoiseVec(pos, &this->grid_noise[int(tile * NTS) % n3], NTS);

        v.x += offset; v.y += offset; v.z += offset;
        //v *= scale;//scale //0.1
        return v;
    }

    inline float3 evaluateCurl(float3 pos, int3 resolution, int NTS = 0, float offset = 0.5, float scale = 0.1,
        float time_anim = 0.1, int frame = 0) const {

        offset *= 25;

        float3 d0 = evaluateVec(pos, resolution, NTS, offset, scale, time_anim, 0);
        float3 d1 = evaluateVec(pos, resolution, NTS, offset, scale, time_anim, 1);
        float3 d2 = evaluateVec(pos, resolution, NTS, offset, scale, time_anim, 2);

        //std::cout << d0.y - d1.z << ";" << d2.z - d0.x << ";" << d1.x - d2.y << std::endl;

        return make_float3(d0.y - d1.z, d2.z - d0.x, d1.x - d2.y);
    }


















    long long size() {
        return (long long)resolution.x * (long long)resolution.y * (long long)resolution.z;
    }
    int deviceCount = 1;
private:
    float* grid;
    float* grid_temp;
    float* grid_noise;
    float3* grid_vel;

    int3 resolution;
    

    std::vector<float*> vdb;
    std::vector<float*> vdb_temp;
    std::vector<float*> vdb_noise;
    std::vector<float3*> grid_vel_gpu;
    

    static inline int Mod(int x, int n) { int m = x % n; return (m < 0) ? m + n : m; }

    float _aCoeffs[32] = {
        0.000334,  -0.001528, 0.000410,  0.003545,  -0.000938, -0.008233, 0.002172,  0.019120,
        -0.005040, -0.044412, 0.011655,  0.103311,  -0.025936, -0.243780, 0.033979,  0.655340,
        0.655340,  0.033979,  -0.243780, -0.025936, 0.103311,  0.011655,  -0.044412, -0.005040,
        0.019120,  0.002172,  -0.008233, -0.000938, 0.003546,  0.000410,  -0.001528, 0.000334
    };

    void downsample(float* from, float* to, int n, int stride) {
        const float* a = &_aCoeffs[16];
        for (int i = 0; i < n / 2; i++) {
            to[i * stride] = 0;
            for (int k = 2 * i - 16; k < 2 * i + 16; k++) {
                to[i * stride] += a[k - 2 * i] * from[Mod(k, n) * stride];
            }
        }
    }

    float _pCoeffs[4] = { 0.25,0.75,0.75,0.25 };

    void upsample(float* from, float* to, int n, int stride) {
        const float* pp = &_pCoeffs[1];

        for (int i = 0; i < n; i++) {
            to[i * stride] = 0;
            for (int k = i / 2 - 1; k < i / 2 + 3; k++) {
                to[i * stride] += 0.5 * pp[k - i / 2] * from[Mod(k, n / 2) * stride];
            }
        }
    }




    void downsample_neumann(float* from, float* to, int n, int stride) {
        static const float* const aCoCenter = &_aCoeffs[16];
        for (int i = 0; i < n / 2; i++) {
            to[i * stride] = 0;
            for (int k = 2 * i - 16; k < 2 * i + 16; k++) {
                // handle boundary
                float fromval;
                if (k < 0) {
                    fromval = from[0];
                }
                else if (k > n - 1) {
                    fromval = from[(n - 1) * stride];
                }
                else {
                    fromval = from[k * stride];
                }
                to[i * stride] += aCoCenter[k - 2 * i] * fromval;
            }
        }
    }

    void upsample_neumann(float* from, float* to, int n, int stride) {
        static const float* const pp = &_pCoeffs[1];
        for (int i = 0; i < n; i++) {
            to[i * stride] = 0;
            for (int k = i / 2 - 1; k < i / 2 + 3; k++) {
                float fromval;
                if (k > n / 2 - 1) {
                    fromval = from[(n / 2 - 1) * stride];
                }
                else if (k < 0) {
                    fromval = from[0];
                }
                else {
                    fromval = from[k * stride];
                }
                to[i * stride] += 0.5 * pp[k - i / 2] * fromval;
            }
        }
    }

};