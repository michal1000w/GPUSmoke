﻿#ifndef __RENDERER
#define __RENDERER

#include "Libraries.h"

__device__ float2 rotate(float2 p, float a)
{
    return make_float2(p.x * cos(a) - p.y * sin(a),
        p.y * cos(a) + p.x * sin(a));
}

__device__ float mix(float a, float b, float inbetween = 0.5) {
    a *= 1.0f - inbetween;
    b *= inbetween;
    return a + b;
}


__global__ void render_pixel(uint8_t* image, float* volume,
    float* temper, int3 img_dims, int3 vol_dims, float step_size,
    float3 light_dir, float3 cam_pos, float rotation, int steps,
    float Fire_Max_temp = 5.0f, bool Smoke_And_Fire = false, float density_influence = 0.2, float fire_multiply = 0.5f)
{
    step_size *= 512.0 / float(steps); //beta

    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= img_dims.x || y >= img_dims.y) return;

    int3 vd = make_int3(vol_dims.x, vol_dims.y, vol_dims.z);
    // Create Normalized UV image coordinates
    float uvx = float(x) / float(img_dims.x) - 0.5;
    float uvy = -float(y) / float(img_dims.y) + 0.5;
    uvx *= float(img_dims.x) / float(img_dims.y);

    float3 v_center = make_float3(
        0.5 * float(vol_dims.x),
        0.5 * float(vol_dims.y),
        0.5 * float(vol_dims.z));

    // Set up ray originating from camera
    float3 ray_pos = cam_pos - v_center;
    float2 pos_rot = rotate(make_float2(ray_pos.x, ray_pos.z), rotation);
    ray_pos.x = pos_rot.x;
    ray_pos.z = pos_rot.y;
    ray_pos += v_center;
    float3 ray_dir = normalize(make_float3(uvx, uvy, 0.5));
    float2 dir_rot = rotate(make_float2(ray_dir.x, ray_dir.z), rotation);
    ray_dir.x = dir_rot.x;
    ray_dir.z = dir_rot.y;
    const float3 dir_to_light = normalize(light_dir);
    const float occ_thresh = 0.001;
    float d_accum = 1.0;//1.0
    float light_accum = 0.0;//0.02   background color
    float temp_accum = 1;//0.0

    float MAX_DENSITY = 1.0f;

    float d_accum2 = 1.0;//1.0
    float light_accum2 = 0.0;//0.0   background color

    int empty_steps = steps/2;
    float empty_step_size = 1.0f * (256.0 / float(empty_steps));


    


    bool _SMOKE = false;
    bool _SMOKE_AND_FIRE = true;
    //RENDER SMOKE
    if (!Smoke_And_Fire || true) {
        //Trace through empty space
#pragma unroll
        for (int step = 0; step < empty_steps;) {
            // At each step, cast occlusion ray towards light source
            float c_density = get_cellF2(ray_pos, vd, volume) * density_influence;
            float temp = get_cellF2(ray_pos, vd, temper); //dla ognia
            ray_pos += ray_dir * empty_step_size * 3.0f;
            // Don't bother with occlusion ray if theres nothing there
            if (c_density >= occ_thresh || temp >= occ_thresh) break;
            step++;
            if (step == empty_steps) goto NOTHING;
        }

        ray_pos -= ray_dir * (empty_step_size * 3.0f);

        steps /= 2;

        // Trace ray through volume
#pragma unroll
        for (int step = 0; step < steps; step++) {
            // At each step, cast occlusion ray towards light source
            float c_density = get_cellF2(ray_pos, vd, volume) * density_influence;
            if (c_density > 1.0) c_density = MAX_DENSITY; //bo �le si� renderuje beta
            float3 occ_pos = ray_pos;
            ray_pos += ray_dir * step_size;
            // Don't bother with occlusion ray if theres nothing there
            if (c_density < occ_thresh) continue;
            float transparency = 1.0;
#pragma unroll
            for (int occ = 0; occ < steps/2; occ++) {
                transparency *= fmax(1.0 - get_cellF2(occ_pos, vd, volume), 0.0);
                if (transparency > 1.0) transparency = 1.0; //beta
                if (transparency < occ_thresh) break;
                occ_pos += dir_to_light * step_size;
            }

            d_accum *= fmax(1.0 - c_density, 0.0);
            light_accum += d_accum * c_density * transparency;
            if (d_accum < occ_thresh) break;
        }

        ray_pos -= ray_dir * step_size * steps;

        ///fire
        float R, G, B;
        R = G = B = 0.0f;

        if (Smoke_And_Fire) {
            float downresing = 2.0f;
            float transparency = 1.0f;
#pragma unroll
            for (int step = 0; step < steps/downresing; step++) {
                // At each step, cast occlusion ray towards light source
                float temp = get_cellF2(ray_pos, vd, temper); //dla ognia
                float3 occ_pos = ray_pos;
                ray_pos += ray_dir * step_size * downresing;
                // Don't bother with occlusion ray if theres nothing there
                if (temp < occ_thresh) continue;

                transparency -= float(step) / float(steps * downresing);
                d_accum2 *= (fmax(1 - temp, 0.0f) / Fire_Max_temp);// *fmax(transparency, 0.0f);
                R += d_accum2 * mix(1.0f,1.0f,transparency);
                G += d_accum2 * mix(1.0f,0.45f,transparency);
                B += d_accum2 * mix(1.0f,0.2f,transparency);
            }
        }




        /*
        float light_accum_R = (light_accum)+fmax(0.0f, (light_accum2 * 0.7f) * (light_accum + 0.5f));
        float light_accum_G = (light_accum)+fmax(0.0f, (light_accum2 * 0.25f) * (light_accum + 0.5f));
        float light_accum_B = (light_accum)+fmax(0.0f, (light_accum2 * 0.1f) * (light_accum + 0.5f));
        */
        float light_accum_R = fmax(light_accum,0.0f)+fmax(0.0f, R * mix(light_accum + 0.1f, 1.0f, fire_multiply));
        float light_accum_G = fmax(light_accum,0.0f)+fmax(0.0f, G * mix(light_accum + 0.1f, 1.0f, fire_multiply));
        float light_accum_B = fmax(light_accum,0.0f)+fmax(0.0f, B * mix(light_accum + 0.1f, 1.0f, fire_multiply));
        ///////
        

        const int pixel = 3 * (y * img_dims.x + x);
        image[pixel + 0] = (uint8_t)(fmin(255.0 * light_accum_R, 255.0));
        image[pixel + 1] = (uint8_t)(fmin(255.0 * light_accum_G, 255.0));
        image[pixel + 2] = (uint8_t)(fmin(255.0 * light_accum_B, 255.0));
    }

    return;

NOTHING:
    const int pixel = 3 * (y * img_dims.x + x);
    image[pixel + 0] = (uint8_t)(0);
    image[pixel + 1] = (uint8_t)(0);
    image[pixel + 2] = (uint8_t)(0);
    return;
}














// GPU volumetric raymarcher
__global__ void render_pixel_old(uint8_t* image, float* volume,
    float* temper, int3 img_dims, int3 vol_dims, float step_size,
    float3 light_dir, float3 cam_pos, float rotation, int steps,
    float Fire_Max_temp = 5.0f, bool Smoke_And_Fire = false)
{

    step_size *= 512.0 / float(steps); //beta

    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= img_dims.x || y >= img_dims.y) return;

    int3 vd = make_int3(vol_dims.x, vol_dims.y, vol_dims.z);
    // Create Normalized UV image coordinates
    float uvx = float(x) / float(img_dims.x) - 0.5;
    float uvy = -float(y) / float(img_dims.y) + 0.5;
    uvx *= float(img_dims.x) / float(img_dims.y);

    float3 v_center = make_float3(
        0.5 * float(vol_dims.x),
        0.5 * float(vol_dims.y),
        0.5 * float(vol_dims.z));

    // Set up ray originating from camera
    float3 ray_pos = cam_pos - v_center;
    float2 pos_rot = rotate(make_float2(ray_pos.x, ray_pos.z), rotation);
    ray_pos.x = pos_rot.x;
    ray_pos.z = pos_rot.y;
    ray_pos += v_center;
    float3 ray_dir = normalize(make_float3(uvx, uvy, 0.5));
    float2 dir_rot = rotate(make_float2(ray_dir.x, ray_dir.z), rotation);
    ray_dir.x = dir_rot.x;
    ray_dir.z = dir_rot.y;
    const float3 dir_to_light = normalize(light_dir);
    const float occ_thresh = 0.001;
    float d_accum = 1.0;//1.0
    float light_accum = 0.025;//0.0   background color
    float temp_accum = 1;//0.0

    float MAX_DENSITY = 1.0f;




    bool _SMOKE = false;
    bool _SMOKE_AND_FIRE = true;
    //RENDER SMOKE
    if (!Smoke_And_Fire) {
        // Trace ray through volume
        for (int step = 0; step < steps; step++) {
            // At each step, cast occlusion ray towards light source
            float c_density = get_cellF(ray_pos, vd, volume);
            if (c_density > 1.0) c_density = MAX_DENSITY; //bo �le si� renderuje beta
            float3 occ_pos = ray_pos;
            ray_pos += ray_dir * step_size;
            // Don't bother with occlusion ray if theres nothing there
            if (c_density < occ_thresh) continue;
            float transparency = 1.0;
            for (int occ = 0; occ < steps; occ++) {
                transparency *= fmax(1.0 - get_cellF(occ_pos, vd, volume), 0.0);
                if (transparency > 1.0) transparency = 1.0; //beta
                if (transparency < occ_thresh) break;
                occ_pos += dir_to_light * step_size;
            }
            d_accum *= fmax(1.0 - c_density, 0.0);
            light_accum += d_accum * c_density * transparency;
            if (d_accum < occ_thresh) break;
        }

        const int pixel = 3 * (y * img_dims.x + x);
        image[pixel + 0] = (uint8_t)(fmin(255.0 * light_accum, 255.0));
        image[pixel + 1] = (uint8_t)(fmin(255.0 * light_accum, 255.0));
        image[pixel + 2] = (uint8_t)(fmin(255.0 * light_accum, 255.0));
    }
    else if (_SMOKE_AND_FIRE) {
        float light_accum_R = light_accum;
        float light_accum_G = light_accum;
        float light_accum_B = light_accum;
        float light_accum2 = light_accum;
        //Render SMOKE
        float d_accum2 = d_accum;
        for (int step = 0; step < steps; step++) {
            // At each step, cast occlusion ray towards light source
            float c_density = get_cellF(ray_pos, vd, volume);
            if (c_density > 1.0) c_density = MAX_DENSITY; //bo �le si� renderuje beta
            float3 occ_pos = ray_pos;
            ray_pos += ray_dir * step_size;
            // Don't bother with occlusion ray if theres nothing there
            if (c_density < occ_thresh) continue;
            float transparency = 1.0;
            for (int occ = 0; occ < steps; occ++) {
                transparency *= fmax(1.0 - get_cellF(occ_pos, vd, volume), 0.0);
                if (transparency > 1.0) transparency = 1.0; //beta
                if (transparency < occ_thresh) break;
                occ_pos += dir_to_light * step_size;
            }
            d_accum2 *= fmax(1.0 - c_density, 0.0);
            light_accum2 += d_accum2 * c_density * transparency;
            if (d_accum2 < occ_thresh) break;
        }

        //reset variables
        ray_pos = cam_pos - v_center;
        float2 pos_rot = rotate(make_float2(ray_pos.x, ray_pos.z), rotation);
        ray_pos.x = pos_rot.x;
        ray_pos.z = pos_rot.y;
        ray_pos += v_center;
        

        //FIRE
        float MAX_Temperature = Fire_Max_temp;
        for (int step = 0; step < steps; step++) {
            float temp = get_cellF(ray_pos, vd, temper);
            ray_pos += ray_dir * step_size;
            if (temp < occ_thresh) continue;


            d_accum *= fmax(temp / MAX_Temperature, 0.0f);
            light_accum += d_accum;
            if (d_accum < occ_thresh) break;
        }



        light_accum_R = (light_accum2) + fmax(0.0f, (light_accum * 0.7f)*(light_accum2+0.5f));
        light_accum_G = (light_accum2) + fmax(0.0f, (light_accum * 0.25f)*(light_accum2+0.5f));
        light_accum_B = (light_accum2) + fmax(0.0f,(light_accum * 0.1f)*(light_accum2+0.5f));
        const int pixel = 3 * (y * img_dims.x + x);
        image[pixel + 0] = (uint8_t)(fmin(255.0f * light_accum_R, 255.0f));
        image[pixel + 1] = (uint8_t)(fmin(255.0f * light_accum_G, 255.0f));
        image[pixel + 2] = (uint8_t)(fmin(255.0f * light_accum_B, 255.0f));
    }
    else { //render heat
        float MAX_Temperature = 5.0f;
        // Trace ray through volume
        for (int step = 0; step < steps; step++) {
            float temp = get_cellF(ray_pos, vd, temper);
            ray_pos += ray_dir * step_size;
            if (temp < occ_thresh) continue;
            

            d_accum *= fmax(temp /MAX_Temperature, 0.0f);
            light_accum += d_accum;
            if (d_accum < occ_thresh) break;
        }

        const int pixel = 3 * (y * img_dims.x + x);
        image[pixel + 0] = (uint8_t)(fmin(255.0f * light_accum * 0.7f, 255.0f));
        image[pixel + 1] = (uint8_t)(fmin(255.0f * light_accum * 0.25f, 255.0f));
        image[pixel + 2] = (uint8_t)(fmin(255.0f * light_accum * 0.1f, 255.0f));
    }
}

void render_fluid(uint8_t* render_target, int3 img_dims,
    float* d_volume, float* temper, int3 vol_dims,
    float step_size, float3 light_dir, float3 cam_pos, float rotation, int STEPS, float Fire_Max_Temp = 5.0f, 
    bool Smoke_and_fire = false, float density_influence = 0.2, float fire_multiply = 0.5f) {

    float measured_time = 0.0f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    /*
    dim3 block(32, 32);//16
    dim3 grid((img_dims.x + 32 - 1) / 32, (img_dims.y + 32 - 1) / 32);

    dim3 block(16, 16);//23
    dim3 grid((img_dims.x + 16 - 1) / 16, (img_dims.y + 16 - 1) / 16);
    */

    dim3 block(8, 8);//30
    dim3 grid((img_dims.x + 8 - 1) / 8, (img_dims.y + 8 - 1) / 8);

    cudaEventRecord(start, 0);

    // Allocate device memory for image
    int img_bytes = 3 * sizeof(uint8_t) * img_dims.x * img_dims.y;
    uint8_t* device_img;
    cudaMalloc((void**)&device_img, img_bytes);
    if (0 == device_img)
    {
        printf("couldn't allocate GPU memory\n");
        return;
    }

    render_pixel << <grid, block >> > (
        device_img, d_volume, temper, img_dims, vol_dims,
        step_size, light_dir, cam_pos, rotation, STEPS, Fire_Max_Temp, Smoke_and_fire,
        density_influence, fire_multiply);

    // Read image back
    cudaMemcpy(render_target, device_img, img_bytes, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaThreadSynchronize();
    cudaEventElapsedTime(&measured_time, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::cout << "Render Time: " << measured_time << "  --  ";
    cudaFree(device_img);
}
#endif




















#ifndef NEW_RENDERER
#define NEW_RENDERER



#define _USE_MATH_DEFINES
#include <cmath>
#include <chrono>

#include <nanovdb/util/IO.h>
#include <nanovdb/util/CudaDeviceBuffer.h>
#include <nanovdb/util/Ray.h>

#include "RendererCommon.h"

#include <algorithm>
#include <iostream>
#include <nanovdb/util/GridBuilder.h>

using BufferT = nanovdb::CudaDeviceBuffer;


void runNanoVDB2(nanovdb::GridHandle<BufferT>& handle, int numIterations, int width, int height, BufferT& imageBuffer)
{
    using GridT = nanovdb::FloatGrid;
    using CoordT = nanovdb::Coord;
    using RealT = float;
    using Vec3T = nanovdb::Vec3<RealT>;
    using RayT = nanovdb::Ray<RealT>;

    auto* h_grid = handle.grid<float>();
    if (!h_grid)
        throw std::runtime_error("GridHandle does not contain a valid host grid");

    float* h_outImage = reinterpret_cast<float*>(imageBuffer.data());

    float              wBBoxDimZ = (float)h_grid->worldBBox().dim()[2] * 2;
    Vec3T              wBBoxCenter = Vec3T(h_grid->worldBBox().min() + h_grid->worldBBox().dim() * 0.5f);
    nanovdb::CoordBBox treeIndexBbox = h_grid->tree().bbox();
    std::cout << "Bounds: "
        << "[" << treeIndexBbox.min()[0] << "," << treeIndexBbox.min()[1] << "," << treeIndexBbox.min()[2] << "] -> ["
        << treeIndexBbox.max()[0] << "," << treeIndexBbox.max()[1] << "," << treeIndexBbox.max()[2] << "]" << std::endl;

    RayGenOp<Vec3T> rayGenOp(wBBoxDimZ, wBBoxCenter);
    CompositeOp     compositeOp;

    auto renderOp = [width, height, rayGenOp, compositeOp, treeIndexBbox] __hostdev__(int start, int end, float* image, const GridT * grid) {
        // get an accessor.
        auto acc = grid->tree().getAccessor();

        for (int i = start; i < end; ++i) {
            Vec3T rayEye;
            Vec3T rayDir;
            rayGenOp(i, width, height, rayEye, rayDir);
            // generate ray.
            RayT wRay(rayEye, rayDir);
            // transform the ray to the grid's index-space.
            RayT iRay = wRay.worldToIndexF(*grid);
            // clip to bounds.
            if (iRay.clip(treeIndexBbox) == false) {
                compositeOp(image, i, width, height, 0.0f, 0.0f);
                return;
            }
            // integrate...
            const float dt = 0.5f;
            float       transmittance = 1.0f;
            for (float t = iRay.t0(); t < iRay.t1(); t += dt) {
                float sigma = acc.getValue(CoordT::Floor(iRay(t))) * 0.1f;
                transmittance *= 1.0f - sigma * dt;
            }
            // write transmittance.
            compositeOp(image, i, width, height, 0.0f, 1.0f - transmittance);
        }
    };
    /*
    {
        float durationAvg = 0;
        for (int i = 0; i < numIterations; ++i) {
            float duration = renderImage(false, renderOp, width, height, h_outImage, h_grid);
            //std::cout << "Duration(NanoVDB-Host) = " << duration << " ms" << std::endl;
            durationAvg += duration;
        }
        durationAvg /= numIterations;
        std::cout << "Average Duration(NanoVDB-Host) = " << durationAvg << " ms" << std::endl;

        saveImage("raytrace_fog_volume-nanovdb-host.pfm", width, height, (float*)imageBuffer.data());
    }
    */
    handle.deviceUpload();

    auto* d_grid = handle.deviceGrid<float>();
    if (!d_grid)
        throw std::runtime_error("GridHandle does not contain a valid device grid");

    imageBuffer.deviceUpload();
    float* d_outImage = reinterpret_cast<float*>(imageBuffer.deviceData());

    {
        float durationAvg = 0;
        for (int i = 0; i < numIterations; ++i) {
            float duration = renderImage(true, renderOp, width, height, d_outImage, d_grid);
            //std::cout << "Duration(NanoVDB-Cuda) = " << duration << " ms" << std::endl;
            durationAvg += duration;
        }
        durationAvg /= numIterations;
        std::cout << "Average Duration(NanoVDB-Cuda) = " << durationAvg << " ms" << std::endl;

        imageBuffer.deviceDownload();
        //saveImage("raytrace_fog_volume-nanovdb-cuda.pfm", width, height, (float*)imageBuffer.data());
        //save_image((float*)imageBuffer.data(), height, width, "raytrace_fog_volume.bmp");
    }
}


void convert_image(float* image, int height, int width, uint8_t* imagee)
{
    int j = 0;
    for (int i = 0; i < width * height; i++) {
        image[i] *= 255.0f;


        imagee[j] = image[i];
        imagee[j + 1] = image[i];
        imagee[j + 2] = image[i];
        j += 3;
    }
}

#include "nanovdb/util/GridBuilder.h"

void PrepareRender(uint8_t* image, int3 dims) {
    //nanovdb::GridHandle<BufferT> handle;
    //handle = nanovdb::createFogVolumeSphere<float, BufferT>(100.0f, nanovdb::Vec3R(-20, 0, 0), 1.0f, 3.0f, nanovdb::Vec3R(0), "sphere");

    auto handle = nanovdb::createLevelSetSphere<float,BufferT>(/*radius*/25.0f,/*center*/nanovdb::Vec3R(-20, 0, 0),
        /*voxelsize*/1.0f,/*halfwidth*/3.0f,/*origin*/nanovdb::Vec3R(0), "sphere",nanovdb::ChecksumMode::Partial);

    const int numIterations = 5;

    const int width = dims.x;
    const int height = dims.y;
    BufferT   imageBuffer;
    imageBuffer.init(width * height * sizeof(float));

    runNanoVDB2(handle, numIterations, width, height, imageBuffer);

    convert_image((float*)imageBuffer.data(), height, width, image);
}
#endif