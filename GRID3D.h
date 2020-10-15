#pragma once
//VDB

#define _USE_MATH_DEFINES
#include <cmath>
#include <openvdb/openvdb.h>
#include <openvdb/tools/LevelSetSphere.h>
#include <openvdb/tools/SignedFloodFill.h>


class GRID3D {
    void deletep(float&) {}
    template <typename Y>
    void deletep(float*& ptr) {
        delete ptr;
        ptr = nullptr;
    }
public:
    GRID3D() {
        resolution.x = resolution.y = resolution.z = 1;
        grid = new float[1];
        grid[0] = 0.0;
        grid_temp = new float[1];
        grid_temp[0] = 0.0;
    }
    GRID3D(int x, int y, int z) {
        resolution.x = x;
        resolution.y = y;
        resolution.z = z;

        grid = new float[(long long)x * (long long)y * (long long)z];
        grid_temp = new float[(long long)x * (long long)y * (long long)z];
        for (long long i = 0; i < size(); i++) {
            grid[i] = 0.0;
            grid_temp[i] = 0.0;
        }
    }
    GRID3D(int elem, float* grid) {
        grid = new float[elem];
        grid_temp = new float[elem];
        for (int i = 0; i < elem; i++) {
            this->grid[i] = grid[i];
            this->grid_temp[i] = grid[i];
        }
    }
    GRID3D(int3 dim, float* grid_src) {
        this->resolution = dim;
        grid = new float[(long long)dim.x * (long long)dim.y * (long long)dim.z];
        //grid_temp = new float[(long long)dim.x * (long long)dim.y * (long long)dim.z];
        cudaMemcpy(grid, grid_src, sizeof(float) * size(), cudaMemcpyDeviceToHost);
        //cudaMemcpy(grid_temp, grid_src_temp, sizeof(float) * size(), cudaMemcpyDeviceToHost);
    }

    GRID3D(int x, int y, int z, float* vdb) {
        resolution.x = x;
        resolution.y = y;
        resolution.z = z;

        grid = new float[(long long)x * (long long)y * (long long)z];
        grid_temp = new float[(long long)x * (long long)y * (long long)z];
        for (long long i = 0; i < size(); i++) {
            grid[i] = vdb[i];
            grid_temp[i] = vdb[i];
        }
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

    GRID3D operator=(const GRID3D& rhs) {
        resolution.x = rhs.resolution.x;
        resolution.y = rhs.resolution.y;
        resolution.z = rhs.resolution.z;

        grid = new float[(long long)rhs.resolution.x * (long long)rhs.resolution.y * (long long)rhs.resolution.z];
        grid_temp = new float[(long long)rhs.resolution.x * (long long)rhs.resolution.y * (long long)rhs.resolution.z];
        for (long long i = 0; i < size(); i++) {
            grid[i] = rhs.grid[i];
            grid_temp[i] = rhs.grid_temp[i];
        }
        return *this;
    }

    void normalizeData() {
        for (int i = 0; i < size(); i++) {
            if (grid[i] < 0.01)
                grid[i] = 0.0f;
            grid[i] = min(grid[i], 1.0);
            grid[i] = max(grid[i], 0.0);
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
            }
        }
    }

    void copyToDevice() {
        addNoise();
        cudaMalloc((void**)&vdb_temp, sizeof(float) * size());
        cudaMemcpy(vdb_temp, grid_temp, sizeof(float) * size(), cudaMemcpyHostToDevice);

        normalizeData();
        //copy to device
        cudaMalloc((void**)&vdb, sizeof(float) * size());
        cudaMemcpy(vdb, grid, sizeof(float) * size(), cudaMemcpyHostToDevice);
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
    void free() {
        //std::cout << "Free grid memory" << std::endl;
        deletep(*grid);

    }
    void freeCuda() {
        cudaFree(vdb);
        cudaFree(vdb_temp);
    }
    ~GRID3D() {
        free();
    }
    float* get_grid() {
        return this->grid;
    }
    float* get_grid_temp() {
        return this->grid_temp;
    }

    float* get_grid_device() {
        return this->vdb;
    }
    float* get_grid_device_temp() {
        return this->vdb_temp;
    }
private:
    long long size() {
        return (long long)resolution.x * (long long)resolution.y * (long long)resolution.z;
    }
    float* grid;
    float* grid_temp;
    int3 resolution;
    float* vdb;
    float* vdb_temp;
};