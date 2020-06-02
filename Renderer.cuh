#ifndef __RENDERER
#define __RENDERER

#include "Libraries.h"

__device__ float2 rotate(float2 p, float a)
{
    return make_float2(p.x * cos(a) - p.y * sin(a),
        p.y * cos(a) + p.x * sin(a));
}

// GPU volumetric raymarcher
__global__ void render_pixel(uint8_t* image, float* volume,
    float* temper, int3 img_dims, int3 vol_dims, float step_size,
    float3 light_dir, float3 cam_pos, float rotation, int steps)
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




    bool _SMOKE = true;
    //RENDER SMOKE
    if (_SMOKE) {
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
}

void render_fluid(uint8_t* render_target, int3 img_dims,
    float* d_volume, float* temper, int3 vol_dims,
    float step_size, float3 light_dir, float3 cam_pos, float rotation, int STEPS) {

    float measured_time = 0.0f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 block(32, 32);
    dim3 grid((img_dims.x + 32 - 1) / 32, (img_dims.y + 32 - 1) / 32);

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
        step_size, light_dir, cam_pos, rotation, STEPS);

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