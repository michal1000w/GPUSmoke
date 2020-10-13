#ifndef __OBJECT__
#define __OBJECT__

#include <string>
#include <iostream>
#include "cutil_math.h"
#include <cuda_runtime.h>
#include "double_buffer.cpp"

#define EMMITER 1
#define SMOKE 2
#define VDBOBJECT 3
#define MS_TO_SEC 0.001


//VDB
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
    }
    GRID3D(int x, int y, int z) {
        resolution.x = x;
        resolution.y = y;
        resolution.z = z;

        grid = new float[(long long)x * (long long)y * (long long)z];
        for (long long i = 0; i < size(); i++)
            grid[i] = 0.0;
    }
    GRID3D(int x, int y, int z, float* vdb) {
        resolution.x = x;
        resolution.y = y;
        resolution.z = z;

        grid = new float[(long long)x * (long long)y * (long long)z];
        for (long long i = 0; i < size(); i++)
            grid[i] = 0.0;
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

    GRID3D operator=(const GRID3D& rhs) {
        resolution.x = rhs.resolution.x;
        resolution.y = rhs.resolution.y;
        resolution.z = rhs.resolution.z;

        grid = new float[(long long)rhs.resolution.x * (long long)rhs.resolution.y * (long long)rhs.resolution.z];
        for (long long i = 0; i < size(); i++)
            grid[i] = rhs.grid[i];
        return *this;
    }

    void normalizeData() {
        for (int i = 0; i < size(); i++) {
            grid[i] = min(grid[i], 1.0);
            grid[i] = max(grid[i], 0.0);
        }       
    }
    void copyToDevice() {
        cudaMalloc((void**)&vdb_temp, sizeof(float) * size());
        cudaMemcpy(vdb_temp, grid, sizeof(float) * size(), cudaMemcpyHostToDevice);

        normalizeData();
        //copy to device
        cudaMalloc((void**)&vdb, sizeof(float) * size());
        cudaMemcpy(vdb, grid, sizeof(float) * size(), cudaMemcpyHostToDevice);
    }

    void set(int x, int y, int z, float value) {
        grid[z * resolution.y * resolution.x + y * resolution.x + x] = value;
    }
    int3 get_resolution() {
        return resolution;
    }
    void free() {
        //std::cout << "Free grid memory" << std::endl;
        deletep(*grid);

    }
    ~GRID3D() {
        free();
    }
    float* get_grid() {
        return this->grid;
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
    int3 resolution;
    float* vdb;
    float* vdb_temp;
};








class OBJECT {
public:
	//Contructors
	OBJECT(std::string type = "SMOKE", float size = 1.0f, float initial_velocity = 0.0f, float velocity_frequence = 0.0f, float3 location = make_float3(0.0,0.0,0.0));
	OBJECT(std::string type = "SMOKE", float size = 1.0f, float initial_velocity = 0.0f, float velocity_frequence = 0.0f,float Temp = 5.0f, float Density = 0.9f, float3 location = make_float3(0.0, 0.0, 0.0));
	//SETTERS-GETTERS
	std::string get_type();
	void set_type(std::string type = "emmiter");
	float get_size();
	void set_size(float size = 1.0f);
	float get_initial_velocity();
	void set_initial_velocity(float velocity = 1.0f);
	float get_velocity_frequence();
	void set_velocity_frequence(float frequence = 0.9f);
	float get_impulseTemp();
	void set_impulseTemp(float temp = 5.0f);
	float get_impulseDensity();
	void set_impulseDensity(float density = 0.6f);
	float3 get_location();
	void set_location(float x = 0, float y = 0, float z = 0);
	void set_location(float3 location);
    void load_density_grid(GRID3D obj);
    GRID3D get_density_grid();
	

private:
	int type;
	float size;
	float initial_velocity;
	float velocity_frequence;
	float impulseTemp;
	float impulseDensity;
	float3 location;
    GRID3D vdb_object;
};


#endif // !OBJECT

