#include <stdio.h>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
//#include <cuda_fp16.h>
#include "cutil_math.h"

#include "Fluid_State.cuh"
//#include "Fluid_State_Huge.cuh"


#define PADDING 1



// GPU helper functions
inline __device__ int3 operator*(const dim3 a, const uint3 b) {
    return make_int3(a.x * b.x, a.y * b.y, a.z * b.z);
}

inline __device__ int3 operator+(dim3 a, int3 b) {
    return make_int3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __device__ int get_voxel(int x, int y, int z, int3 d)
{
    return z * d.y * d.x + y * d.x + x;
}

inline __device__ float get_voxel_density(int x, int y, int z,int3 d, float* vdb)
{
    return vdb[z * d.y * d.x + y * d.x + x];
}

template <typename T> inline __device__ T zero() { return 0.0; }

template <> inline __device__ float  zero<float>() { return 0.0f; }
template <> inline __device__ float3 zero<float3>() { return make_float3(0.0f); }

template <typename T>
inline __device__ T get_cell(int3 c, int3 d, T* vol) {
    if (c.x < 0 || c.y < 0 || c.z < 0 ||
        c.x >= d.x || c.y >= d.y || c.z >= d.z) {
        return zero<T>();
    }
    else {
        return vol[get_voxel(c.x, c.y, c.z, d)];
    }
}

inline __host__ __device__ float3 floorr(const float3 v)
{
    return make_float3(floorf(v.x), floorf(v.y), floorf(v.z));
}

template <typename T>
inline __device__ T get_cellF(float3 p, int3 d, T* vol) {

    // bilinear interpolation
    float3 l = floorr(p);
    int3 rp = make_int3(l);
    float3 dif = p - l;
    T sum = zero<T>();

#pragma unroll
    for (int a = 0; a <= 1; a++)
    {
#pragma unroll
        for (int b = 0; b <= 1; b++)
        {
#pragma unroll
            for (int c = 0; c <= 1; c++)
            {
                sum += abs(float(1 - a) - dif.x) *
                    abs(float(1 - b) - dif.y) *
                    abs(float(1 - c) - dif.z) *
                    get_cell(make_int3(rp.x + a, rp.y + b, rp.z + c), d, vol);
            }
        }
    }

    return sum;
}

// Convert single index into 3D coordinates
inline __device__ int3 mod_coords(int i, int d) {
    return make_int3(i % d, (i / d) % d, (i / (d * d)));
}

template <typename T>
inline __device__ T read_shared(T* mem, dim3 c,
    int3 blk_dim, int pad, int x, int y, int z)
{
    return mem[get_voxel(c.x + pad + x, c.y + pad + y, c.z + pad + z, blk_dim)];
}

template <typename T>
__device__ void load_shared(dim3 blkDim, dim3 blkIdx,
    dim3 thrIdx, int3 vd, int sdim, T* shared, T* src)
{
    int t_idx = thrIdx.z * blkDim.y * blkDim.x
        + thrIdx.y * blkDim.x + thrIdx.x;
    // Load sdim*sdim*sdim cube of memory into shared array 
    const int cutoff = (sdim * sdim * sdim) / 2;
    if (t_idx < cutoff) {
        int3 sp = mod_coords(t_idx, sdim);
        sp = sp + blkDim * blkIdx - 1;
        shared[t_idx] = get_cell(sp, vd, src);
        sp = mod_coords(t_idx + cutoff, sdim);
        sp = sp + blkDim * blkIdx - 1;
        shared[t_idx + cutoff] = get_cell(sp, vd, src);
    }
}


// Simulation compute kernels
template <typename T>
__global__ void pressure_solve(T* div, T* p_src, T* p_dst,
    int3 vd, float amount)
{
    __shared__ T loc[1024];
    const int padding = PADDING; // How far to load past end of cube
    const int sdim = blockDim.x + 2 * padding; // 10 with blockdim 8
    const int3 s_dims = make_int3(sdim, sdim, sdim);
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int z = blockDim.z * blockIdx.z + threadIdx.z;

    load_shared(
        blockDim, blockIdx, threadIdx, vd, sdim, loc, p_src);
    __syncthreads();

    if (x >= vd.x || y >= vd.y || z >= vd.z) return;

    T d = div[get_voxel(x, y, z, vd)];

    T p_sum =
        read_shared(loc, threadIdx, s_dims, padding, -1, 0, 0);
    p_sum += read_shared(loc, threadIdx, s_dims, padding, 1, 0, 0);
    p_sum += read_shared(loc, threadIdx, s_dims, padding, 0, -1, 0);
    p_sum += read_shared(loc, threadIdx, s_dims, padding, 0, 1, 0);
    p_sum += read_shared(loc, threadIdx, s_dims, padding, 0, 0, -1);
    p_sum += read_shared(loc, threadIdx, s_dims, padding, 0, 0, 1);
    //avg /= 6.0;
    //avg -= o;

    p_dst[get_voxel(x, y, z, vd)] = (p_sum + amount * d) * 0.166667;//o + avg*amount;
}



template <typename V, typename T>
__global__ void divergence(V* velocity, T* div, int3 vd, float half_cell)
{
    __shared__ V loc[1024];
    const int padding = PADDING; // How far to load past end of cube
    const int sdim = blockDim.x + 2 * padding; // 10 with blockdim 8
    const int3 s_dims = make_int3(sdim, sdim, sdim);
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int z = blockDim.z * blockIdx.z + threadIdx.z;

    load_shared(
        blockDim, blockIdx, threadIdx, vd, sdim, loc, velocity);
    __syncthreads();

    if (x >= vd.x || y >= vd.y || z >= vd.z) return;

    T d =
        read_shared(loc, threadIdx, s_dims, padding, 1, 0, 0).x;
    d -= read_shared(loc, threadIdx, s_dims, padding, -1, 0, 0).x;
    d += read_shared(loc, threadIdx, s_dims, padding, 0, 1, 0).y;
    d -= read_shared(loc, threadIdx, s_dims, padding, 0, -1, 0).y;
    d += read_shared(loc, threadIdx, s_dims, padding, 0, 0, 1).z;
    d -= read_shared(loc, threadIdx, s_dims, padding, 0, 0, -1).z;
    d *= half_cell;

    div[get_voxel(x, y, z, vd)] = d;
}

template <typename V, typename T>
__global__ void subtract_pressure(V* v_src, V* v_dest, T* pressure,
    int3 vd, float grad_scale)
{
    __shared__ T loc[1024];
    const int padding = PADDING; // How far to load past end of cube
    const int sdim = blockDim.x + 2 * padding; // 10 with blockdim 8
    const int3 s_dims = make_int3(sdim, sdim, sdim);
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int z = blockDim.z * blockIdx.z + threadIdx.z;

    load_shared(
        blockDim, blockIdx, threadIdx, vd, sdim, loc, pressure);
    __syncthreads();

    if (x >= vd.x || y >= vd.y || z >= vd.z) return;

    V old_v = get_cell(make_int3(x, y, z), vd, v_src);

    V grad;
    grad.x =
        read_shared(loc, threadIdx, s_dims, padding, 1, 0, 0) -
        read_shared(loc, threadIdx, s_dims, padding, -1, 0, 0);
    grad.y =
        read_shared(loc, threadIdx, s_dims, padding, 0, 1, 0) -
        read_shared(loc, threadIdx, s_dims, padding, 0, -1, 0);
    grad.z =
        read_shared(loc, threadIdx, s_dims, padding, 0, 0, 1) -
        read_shared(loc, threadIdx, s_dims, padding, 0, 0, -1);

    v_dest[get_voxel(x, y, z, vd)] = old_v - grad * grad_scale;
}

template <typename V, typename T>
__global__ void advection(V* velocity, T* source, T* dest, int3 vd,
    float time_step, float dissipation)
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int z = blockDim.z * blockIdx.z + threadIdx.z;

    if (x >= vd.x || y >= vd.y || z >= vd.z) return;

    V vel = velocity[get_voxel(x, y, z, vd)];

    float3 np = make_float3(float(x), float(y), float(z)) - time_step * vel;

    dest[get_voxel(x, y, z, vd)] = dissipation * get_cellF(np, vd, source);
}

template <typename T>
__global__ void impulse(T* target, float3 c,
    float radius, T val, int3 vd)
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int z = blockDim.z * blockIdx.z + threadIdx.z;

    if (x >= vd.x || y >= vd.y || z >= vd.z) return;

    float3 p = make_float3(float(x), float(y), float(z));

    float dist = length(p - c);

    if (dist < radius) {
        target[get_voxel(x, y, z, vd)] = val;
    }
}

template <typename T>
__global__ void impulse_vdb(T* target, float3 c, T val, int3 vd, float* vdb, float temp = 1.0)
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int z = blockDim.z * blockIdx.z + threadIdx.z;

    if (x >= vd.x || y >= vd.y || z >= vd.z) return;

    float adding = temp * get_voxel_density(x, y, z, vd, vdb);
    float sum = adding + target[get_voxel(x, y, z, vd)];
    if (target[get_voxel(x, y, z, vd)] < adding*0.7)
        target[get_voxel(x, y, z, vd)] = sum; 
}

template <typename T>
__global__ void impulse_vdb_single(T* target, float3 c, T val, int3 vd, float* vdb, float temp = 1.0)
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int z = blockDim.z * blockIdx.z + threadIdx.z;

    if (x >= vd.x || y >= vd.y || z >= vd.z) return;

    target[get_voxel(x, y, z, vd)] += temp * get_voxel_density(x, y, z, vd, vdb);
}

template <typename T>
__global__ void soft_impulse(T* target, float3 c,
    float radius, T val, float speed, int3 vd)
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int z = blockDim.z * blockIdx.z + threadIdx.z;

    if (x >= vd.x || y >= vd.y || z >= vd.z) return;

    float3 p = make_float3(float(x), float(y), float(z));

    float dist = length(p - c);

    T cur = target[get_voxel(x, y, z, vd)];

    if (dist < radius && cur < val) {
        target[get_voxel(x, y, z, vd)] = cur + speed * val;
    }
}

template <typename T>
__global__ void force_field_power(T* target, float3 c,
    float radius, float force, int3 vd)
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int z = blockDim.z * blockIdx.z + threadIdx.z;

    if (x >= vd.x || y >= vd.y || z >= vd.z) return;

    float3 p = make_float3(float(x), float(y), float(z));

    float dist = length(p - c);

    T cur = target[get_voxel(x, y, z, vd)];

    if (dist < radius) {
        target[get_voxel(x, y, z, vd)] = cur + force * (1.0f/(dist*dist));
    }
}

template <typename T>
__global__ void force_field_force(T* target, float3 c,
    float radius, float force, int3 vd)
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int z = blockDim.z * blockIdx.z + threadIdx.z;

    if (x >= vd.x || y >= vd.y || z >= vd.z) return;

    float3 p = make_float3(float(x), float(y), float(z));

    float dist = length(p - c);

    T cur = target[get_voxel(x, y, z, vd)];

    if (dist < radius) {
        float power = force * (1.0f / (dist * dist));
        float3 vector = make_float3(c.x - p.x, c.y - p.y, c.z - p.z);

        target[get_voxel(x, y, z, vd)] = cur + vector * power;
    }
}

template <typename T>
__global__ void collision_sphere(T* target, float3 c,
    float radius, int3 vd)
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int z = blockDim.z * blockIdx.z + threadIdx.z;

    if (x >= vd.x || y >= vd.y || z >= vd.z) return;

    float3 p = make_float3(float(x), float(y), float(z));

    float dist = length(p - c);

    T cur = target[get_voxel(x, y, z, vd)];

    if (dist >= radius * 1.0 && dist <= radius * 1.2) {
        float3 vector = make_float3(c.x - p.x, c.y - p.y, c.z - p.z);

        target[get_voxel(x, y, z, vd)] = cur + vector * -1.0f;
    }
}

template <typename V, typename T, typename Z>
__global__ void collision_sphere2(V* v_src, T* temperature, Z* density,
    int3 vd, float3 c, float radius, float ambient_temp)
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int z = blockDim.z * blockIdx.z + threadIdx.z;

    if (x >= vd.x || y >= vd.y || z >= vd.z) return;

    float3 p = make_float3(float(x), float(y), float(z));

    float dist = length(p - c);


    if (dist <= radius) {
        V vel = v_src[get_voxel(x, y, z, vd)];
        T temp = temperature[get_voxel(x, y, z, vd)];
        float3 vector = make_float3(c.x - p.x, c.y - p.y, c.z - p.z);

        float direction = vel.y;

        //v_src[get_voxel(x, y, z, vd)] = (-1.0*(vel*grad_scale)*vel) + (viscosity*grad_scale*grad_scale) - ((1.0/press) * grad_scale*press);

        density[get_voxel(x, y, z, vd)] *= 0.95; //zanikanie density
        v_src[get_voxel(x, y, z, vd)] = (vel + vector * -1.0f)*0.1; //zanikanie momentu
        if (temp <= 0.2 && direction >= 0)
            temp += 0.1;
        else if (temp >= -0.2 && direction < 0)
            temp -= 0.1;
        v_src[get_voxel(x, y, z, vd)].y = v_src[get_voxel(x, y, z, vd)].y + (temp * 2.0 * (1.0 / (dist*dist)));
    }
}





















template <typename T>
__global__ void force_field_wind(T* target, float3 c,
    float radius, float force, float3 direction, int3 vd)
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int z = blockDim.z * blockIdx.z + threadIdx.z;

    if (x >= vd.x || y >= vd.y || z >= vd.z) return;

    float3 p = make_float3(float(x), float(y), float(z));

    float dist = length(p - c);

    T cur = target[get_voxel(x, y, z, vd)];

    if (dist < radius) {
        float power = force * (1.0f / (dist * dist));
        

        target[get_voxel(x, y, z, vd)] = cur + direction * power;
    }
}

template <typename T>
__global__ void force_field_turbulence(T* target, float3 c,
    float radius, float force, float freq, int3 vd, int frame)
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int z = blockDim.z * blockIdx.z + threadIdx.z;

    if (x >= vd.x || y >= vd.y || z >= vd.z) return;

    float3 p = make_float3(float(x), float(y), float(z));

    float dist = length(p - c);

    T cur = target[get_voxel(x, y, z, vd)];

    if (dist < radius) {
        float power = force * (1.0f / (dist * dist));
        float random = float((frame * (x + y - z)) % 1000) / 1000.0;
        float v = (sin(freq * p.x + random) + sin(freq * p.z + random) - 0.1f);
        //v = v * v * v * v * v;
        //v = v * v * v;

        target[get_voxel(x, y, z, vd)] = cur + v * power;
    }
}

template <typename T>
__global__ void wavey_impulse_temperature(T* target, float3 c,
    float3 size, T base, float amp, float freq, int3 vd, int frame)
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int z = blockDim.z * blockIdx.z + threadIdx.z;

    if (x >= vd.x || y >= vd.y || z >= vd.z) return;

    float3 p = make_float3(float(x), float(y), float(z));

    //float dist = length(p-c);
    float3 minC = c - size;
    float3 maxC = c + size;

    //T cur = target[ get_voxel(x,y,z, vd) ];

    if (p.x > minC.x && p.y > minC.y && p.z > minC.z &&
        p.x < maxC.x && p.y < maxC.y && p.z < maxC.z) {
        float random = float((frame * (x + y - z)) % 1000) / 1000.0;
        float v = 0.5 * (sin(freq * p.x + random) + sin(freq * p.z + random) + 0.0);
        v = v * v * v * v * v;
        if (base + amp * v > 0)
            target[get_voxel(x, y, z, vd)] = base + amp * v;
        else
            target[get_voxel(x, y, z, vd)] = base;
    }
}

template <typename T, typename V>
__global__ void wavey_impulse_temperature_new(T* target,V* velocity, float3 c,
    float3 size, T base, float amp, float freq, int3 vd, int frame)
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int z = blockDim.z * blockIdx.z + threadIdx.z;

    if (x >= vd.x || y >= vd.y || z >= vd.z) return;

    float3 p = make_float3(float(x), float(y), float(z));

    //float dist = length(p-c);
    float3 minC = c - size;
    float3 maxC = c + size;

    if (p.x > minC.x && p.y > minC.y && p.z > minC.z &&
        p.x < maxC.x && p.y < maxC.y && p.z < maxC.z) {
        float random = float((frame * frame) % 100) / 100.0;
        float v = 0.5 * (sin(freq * freq + random) + sin(freq * freq + random) + 0.0);
        v = 0.05;
        if (base >= 0)
            if (base + amp * v > 0)
                target[get_voxel(x, y, z, vd)] = base + amp * v;
            else
                target[get_voxel(x, y, z, vd)] = base;
        else {
            target[get_voxel(x, y, z, vd)] = base - amp * v;
            velocity[get_voxel(x, y, z, vd)].y = base;
        }
    }
}

template <typename T>
__global__ void wavey_impulse_temperature_new_old(T* target, float3 c,
    float3 size, T base, float amp, float freq, int3 vd, int frame)
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int z = blockDim.z * blockIdx.z + threadIdx.z;

    if (x >= vd.x || y >= vd.y || z >= vd.z) return;

    float3 p = make_float3(float(x), float(y), float(z));

    //float dist = length(p-c);
    float3 minC = c - size;
    float3 maxC = c + size;

    if (p.x > minC.x && p.y > minC.y && p.z > minC.z &&
        p.x < maxC.x && p.y < maxC.y && p.z < maxC.z) {
        float random = float((frame * frame) % 100) / 100.0;
        float v = 0.5 * (sin(freq * freq + random) + sin(freq * freq + random) + 0.0);
        v = 0.05;
        if (base >= 0)
            if (base + amp * v > 0)
                target[get_voxel(x, y, z, vd)] = base + amp * v;
            else
                target[get_voxel(x, y, z, vd)] = base;
        else {
            if (base - amp * v < 0)
                target[get_voxel(x, y, z, vd)] = base - amp * v;
            else
                target[get_voxel(x, y, z, vd)] = base;
        }
    }
}

template <typename T>
__global__ void wavey_impulse_density_new(T* target, float3 c,
    float3 size, T base, float amp, float freq, int3 vd, int frame)
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int z = blockDim.z * blockIdx.z + threadIdx.z;

    if (x >= vd.x || y >= vd.y || z >= vd.z) return;

    float3 p = make_float3(float(x), float(y), float(z));

    //float dist = length(p-c);
    float3 minC = c - size;
    float3 maxC = c + size;

    //T cur = target[ get_voxel(x,y,z, vd) ];

    if (p.x > minC.x && p.y > minC.y && p.z > minC.z &&
        p.x < maxC.x && p.y < maxC.y && p.z < maxC.z) {
        float random = float(frame*frame % 100) / 100.0;
        float v = 0.5 * (sin(freq*freq  + random) + sin(freq*freq + random) + 0.0);
        v = v * v * v * v * v;
        amp = 0.5;
        if (base + amp * v <= 1.0)
            target[get_voxel(x, y, z, vd)] = base + amp * v * 0.1;
    }
}

template <typename T>
__global__ void wavey_impulse_density(T* target, float3 c,
    float3 size, T base, float amp, float freq, int3 vd,int frame)
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int z = blockDim.z * blockIdx.z + threadIdx.z;

    if (x >= vd.x || y >= vd.y || z >= vd.z) return;

    float3 p = make_float3(float(x), float(y), float(z));

    //float dist = length(p-c);
    float3 minC = c - size;
    float3 maxC = c + size;

    //T cur = target[ get_voxel(x,y,z, vd) ];

    if (p.x > minC.x && p.y > minC.y && p.z > minC.z &&
        p.x < maxC.x && p.y < maxC.y && p.z < maxC.z) {
        float random = float((frame * (x + y - z)) % 1000) / 1000.0;
        float v = 0.5 * (sin(freq * p.x + random) + sin(freq * p.z + random) + 0.0);
        v = v * v * v * v * v;
        amp = 0.5;
        if (base + amp * v <= 1.0)
            target[get_voxel(x, y, z, vd)] = base + amp * v;
    }
}


template <typename T>
__global__ void wavey_impulse(T* target, float3 c,
    float3 size, T base, float amp, float freq, int3 vd, bool temp = false)
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int z = blockDim.z * blockIdx.z + threadIdx.z;

    if (x >= vd.x || y >= vd.y || z >= vd.z) return;

    float3 p = make_float3(float(x), float(y), float(z));

    //float dist = length(p-c);
    float3 minC = c - size;
    float3 maxC = c + size;

    //T cur = target[ get_voxel(x,y,z, vd) ];

    if (p.x > minC.x&& p.y > minC.y&& p.z > minC.z&&
        p.x < maxC.x && p.y < maxC.y && p.z < maxC.z) {
        float v = 0.5 * (sin(freq * p.x) + sin(freq * p.z) + 0.0);
        v = v * v * v * v * v;
        target[get_voxel(x, y, z, vd)] = base + amp * v;
        if (temp && target[get_voxel(x, y, z, vd)] <= 1)
            target[get_voxel(x, y, z, vd)] = 1;
    }
}



template <typename V, typename T>
__global__ void buoyancy(V* v_src, T* t_src, T* d_src, V* v_dest, 
    float amb_temp, float time_step, float buoy, float weight, int3 vd)
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int z = blockDim.z * blockIdx.z + threadIdx.z;

    if (x >= vd.x || y >= vd.y || z >= vd.z) return;

    T temp = t_src[get_voxel(x, y, z, vd)];
    V vel = v_src[get_voxel(x, y, z, vd)];

    if (temp > amb_temp)
    {
        T dense = d_src[get_voxel(x, y, z, vd)];
        vel.y += (time_step * (temp - amb_temp) * buoy - dense * weight);
    }

    v_dest[get_voxel(x, y, z, vd)] = vel;
}

















static inline __host__ __device__ int Mod(int x, int n) { int m = x % n; return (m < 0) ? m + n : m; }

inline __host__ __device__ float minf(const float a, const float b)
{
    return (a < b) ? a : b;
}
inline __host__ __device__ float maxf(const float a, const float b)
{
    return (a > b) ? a : b;
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


//Wavelet Noise Kernels
inline __device__ float getWNoiseDx(float3& p, float* data, int max_dim) {
    float w[3][3], t, result = 0;
    const int NOISE_TILE_SIZE = max_dim;



    // Evaluate quadratic B-spline basis functions
    int midX = (int)ceilf(p.x - 0.5f);
    t = midX - (p.x - 0.5f);
    w[0][0] = -t;
    w[0][2] = (1.f - t);
    w[0][1] = 2.0f * t - 1.0f;

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

#undef ADD_WEIGHTED

inline __device__ float eevaluateNoise(float3 pos, int3 resolution, float* data, int NTS = 64, float offset = 0.5, float scale = 0.1,
    float time_anim = 0.1, int tile = 0) {
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
    float v = getWNoiseDx(pos, &data[int(tile * NTS) % n3], NTS);
    //float v = WNoise(pos, &this->grid_noise[int(tile * n3 * 0.01) % n3], NOISE_TILE_SIZE);

    v += offset;//offset //0.5
    //v *= scale;//scale //0.1
    return v;
}



template <typename V, typename T>
__global__ void applyNoiseDT(T* t_src, T* d_src, T* t_dest, T* d_dest, V* noise,
    int3 vd, float intensity, float offset, float scale, int frame = 1, float time_anim = 0.5f,
    float density_cutoff = 0.01)
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int z = blockDim.z * blockIdx.z + threadIdx.z;

    if (x >= vd.x || y >= vd.y || z >= vd.z) return;

    float temp = t_src[get_voxel(x, y, z, vd)];
    float dens = d_src[get_voxel(x, y, z, vd)];
    
        


    if (temp != 0) {
        t_dest[get_voxel(x, y, z, vd)] = t_src[get_voxel(x, y, z, vd)] + (eevaluateNoise(make_float3(x, y, z), vd, noise,
            64, offset * 1.6f, scale, 0.1, frame % 128) * intensity * maxf(0.05, minf((t_src[get_voxel(x, y, z, vd)]), 1.0)));
    }
    else {
        t_dest[get_voxel(x, y, z, vd)] = 0.0f;
    }

        
    if (dens >= density_cutoff) {
        d_dest[get_voxel(x, y, z, vd)] = d_src[get_voxel(x, y, z, vd)] + (eevaluateNoise(make_float3(x, y, z), vd, noise,
            64, offset, scale, time_anim * float(frame%256)/*Time Anim*/, 1/* frame % 128*/) * intensity * maxf(0.01, minf((d_src[get_voxel(x, y, z, vd)]), 1.0)));
    }
    else {
        d_dest[get_voxel(x, y, z, vd)] = 0.0f;
    }
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


inline __device__ float3 WNoiseVecGPU(float3& p, float* data, int max_dim) {
    float3 final = make_float3(0, 0, 0);
    const int NOISE_TILE_SIZE = max_dim;
    float w[3][3];
    float dw[3][3];
    float result = 0;
    int xC, yC, zC;
    float weight;

    int midX = (int)ceilf(p.x - 0.5f);
    int midY = (int)ceilf(p.y - 0.5f);
    int midZ = (int)ceilf(p.z - 0.5f);

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


inline __device__ float3 evaluateVectorGPU(float3 pos, int3 resolution, float* data, int NTS = 64, float offset = 0.5, float scale = 0.1,
    float time_anim = 0.1, int tile = 0) {
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
    float3 v = WNoiseVecGPU(pos, &data[int(tile * NTS) % n3], NTS);

    v.x += offset; v.y += offset; v.z += offset;
    //v *= scale;//scale //0.1
    return v;
}

inline __device__ float3 evaluateCurlGPU(float3 pos, int3 resolution, float* data, int NTS = 64, float offset = 0.5, float scale = 0.1,
    float time_anim = 0.1, int tile = 0) {

    offset *= 25;

    float3 d0 = evaluateVectorGPU(pos, resolution, data, NTS, offset, scale, time_anim, 0);
    float3 d1 = evaluateVectorGPU(pos, resolution, data, NTS, offset, scale, time_anim, 1);
    float3 d2 = evaluateVectorGPU(pos, resolution, data, NTS, offset, scale, time_anim, 2);


    return make_float3(d0.y - d1.z, d2.z - d0.x, d1.x - d2.y);
}






template <typename V, typename T>
__global__ void applyNoiseV(T* v_src, T* v_dest, V* noise,
    int3 vd, float intensity, float offset, float scale, int frame = 1, float time_anim = 0.5f)
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int z = blockDim.z * blockIdx.z + threadIdx.z;

    if (x >= vd.x || y >= vd.y || z >= vd.z) return;


    v_dest[get_voxel(x, y, z, vd)] = v_src[get_voxel(x, y, z, vd)] + (evaluateCurlGPU(make_float3(x, y, z), vd, noise,
            64, offset * 10.0, scale * 2.0, time_anim * float(frame%256)/*Time Anim*/, 0/*frame % 128*/) * intensity * 1.5f);
}