#pragma once
//VDB

#define _USE_MATH_DEFINES
#include <nanovdb/NanoVDB.h>
#include <cmath>
#include <openvdb/openvdb.h>
#include <openvdb/tools/LevelSetSphere.h>
#include <openvdb/tools/SignedFloodFill.h>
#include <cuda_runtime.h>

#include <tbb/parallel_for.h>
#include <tbb/atomic.h>

class GRID3D {
    void deletep(float&) {}
    void deletep(float*& ptr) {
        delete[] ptr;
        ptr = nullptr;
    }
    void initNoiseGrid() {
        grid_noise = new float[1];
        grid_noise[0] = -122.1123123;
    }
public:
    GRID3D() {
        resolution.x = resolution.y = resolution.z = 1;
        grid = new float[1];
        grid[0] = 0.0;
        grid_temp = new float[1];
        grid_temp[0] = 0.0;
        initNoiseGrid();
        //grid_noise = new float[1];
        //grid_noise[0] = 0.0;
    }
    GRID3D(int x, int y, int z) {
        resolution.x = x;
        resolution.y = y;
        resolution.z = z;

        initNoiseGrid();
        //grid_noise = new float[1];
        //grid_noise[0] = 0.0;
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
        //grid_noise = new float[1];
        //grid_noise[0] = 0.0;
        initNoiseGrid();
    }
    GRID3D(int3 dim, float* grid_src) {
        this->resolution = dim;
        grid = new float[(long long)dim.x * (long long)dim.y * (long long)dim.z];
        //grid_temp = new float[(long long)dim.x * (long long)dim.y * (long long)dim.z];
        cudaMemcpy(grid, grid_src, sizeof(float) * size(), cudaMemcpyDeviceToHost);
        grid_temp = new float[1];
        //cudaMemcpy(grid_temp, grid_src_temp, sizeof(float) * size(), cudaMemcpyDeviceToHost);
        //grid_noise = new float[1];
        //grid_noise[0] = 0.0;
        initNoiseGrid();
    }

    void load_from_device(int3 dim, float* grid_src) {
        free();
        this->resolution = dim;
        grid = new float[(long long)dim.x * (long long)dim.y * (long long)dim.z];
        cudaMemcpy(grid, grid_src, sizeof(float) * size(), cudaMemcpyDeviceToHost);
        //grid_noise = new float[1];
        //grid_noise[0] = 0.0;
        initNoiseGrid();
        grid_temp = new float[1];
        grid_temp[0] = 0.0;
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
        //grid_noise = new float[1];
        //grid_noise[0] = 0.0;
        initNoiseGrid();
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

        grid = new float[(long long)rhs.resolution.x * (long long)rhs.resolution.y * (long long)rhs.resolution.z];
        grid_temp = new float[(long long)rhs.resolution.x * (long long)rhs.resolution.y * (long long)rhs.resolution.z];
        for (long long i = 0; i < size(); i++) {
            grid[i] = rhs.grid[i];
            grid_temp[i] = rhs.grid_temp[i];
        }
        //grid_noise = new float[1];
        //grid_noise[0] = 0.0;
        initNoiseGrid();
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
        initNoiseGrid();
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
        initNoiseGrid();
    }

    GRID3D load(const GRID3D* rhs) {
        free();
        resolution.x = rhs->resolution.x;
        resolution.y = rhs->resolution.y;
        resolution.z = rhs->resolution.z;

        grid = new float[(long long)rhs->resolution.x * (long long)rhs->resolution.y * (long long)rhs->resolution.z];
        grid_temp = new float[(long long)rhs->resolution.x * (long long)rhs->resolution.y * (long long)rhs->resolution.z];
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
        grid_temp = rhs->grid;
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
                grid_temp[i] = max(0.5f, grid_temp[i]);
            }
        }
    }

    void copyToDevice(bool addNoise = true) {
        if (addNoise)
            this->addNoise();
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
    void freeOnlyGrid() {
        deletep(grid);
    }
    void free() {
        //std::cout << "Free grid memory" << std::endl;
        deletep(grid);
        deletep(grid_temp);
        //deletep(grid_noise);
    }
    void free_noise() {
        deletep(grid_noise);
    }
    void freeCuda() {
        cudaFree(vdb);
        cudaFree(vdb_temp);
    }
    ~GRID3D() {
        //free();
    }
    float* get_grid() const {
        return this->grid;
    }
    float* get_grid_temp() const {
        return this->grid_temp;
    }

    float* get_grid_device() {
        return this->vdb;
    }
    float* get_grid_device_temp() {
        return this->vdb_temp;
    }
    void UpScale(int power, int SEED = 2, int frame = 0, float offset = 0.5, float scale = 0.1, int noise_scale = 128) {
        int noise_tile_size = power * min(min(resolution.x, resolution.y) //max max
            , resolution.z);

        noise_tile_size = noise_scale;

        srand(SEED);
        if (this->grid_noise[0] < -1)
            generateTile(noise_tile_size);


        applyNoise(1, noise_tile_size, offset, scale, frame);
    }

    void LoadNoise(GRID3D* rhs) {
        this->grid_noise = rhs->grid_noise;
    }

    inline float evaluate(float3 pos, int tile, int3 resolution, int NTS = 0, float offset = 0.5, float scale = 0.1)
    {
        int NOISE_TILE_SIZE = min(min(resolution.x,resolution.y),resolution.z);
        if (NTS != 0)
            NOISE_TILE_SIZE = NTS;

        pos.x *= resolution.x;
        pos.y *= resolution.y;
        pos.z *= resolution.z;
        pos.x += 1; pos.y += 1; pos.z += 1;

        // time anim
        pos.x += 0.1; pos.y += 0.1; pos.z += 0.1;

        pos.x *= 1;
        pos.y *= 1;
        pos.z *= 1;

        const int n3 = NOISE_TILE_SIZE * NOISE_TILE_SIZE * NOISE_TILE_SIZE;
        float v = WNoise(pos, &this->grid_noise[int(tile * n3 * 0.01)%n3], NOISE_TILE_SIZE); //0.01

        v += offset;//offset //0.5
        v *= scale;//scale //0.1
        return v;
    }

    void applyNoise(float intensity = 0.2f, int NTS = 0, float offset = 0.5, float scale = 0.1, int frame = 0) {
        if (NTS == 0)
            NTS = min(min(resolution.x, resolution.y), resolution.z);
        int NTS2 = NTS * NTS;
        int NTS3 = NTS2 * NTS;
        //std::cout << "Applying noise" << std::endl;


        int THREADS = 32;
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

                        if (*position >= 0.01) {
                            //*position += this->grid_noise[(z * (resolution.x * resolution.y)%NTS2) +
                            //(y * (resolution.x % NTS)) + (x % NTS)] * intensity;
                            *position += evaluate(make_float3(x, y, z), frame%512, resolution, NTS, offset, scale);
                        }

                    }
        });
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

    long long size() {
        return (long long)resolution.x * (long long)resolution.y * (long long)resolution.z;
    }
private:
    float* grid;
    float* grid_temp;
    float* grid_noise;
    int3 resolution;
    float* vdb;
    float* vdb_temp;


    int Mod(int x, int n) { int m = x % n; return (m < 0) ? m + n : m; }
    
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
                to[i * stride] += a[k - 2 * i] * from[Mod(k,n) * stride];
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

};