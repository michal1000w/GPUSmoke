#ifndef __RENDERER
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
    float* temper,float* collision, int3 img_dims, int3 vol_dims, float step_size,
    float3 light_dir, float3 cam_pos, float rotation, int steps,
    float Fire_Max_temp = 5.0f, bool Smoke_And_Fire = false, float density_influence = 0.2, float fire_multiply = 0.5f,
    bool render_shadows = true, float transparency_compensation = 1.0f, float shadow_quality = 1.0f,
    bool render_collision = true)
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
            float coll = get_cellF2(ray_pos, vd, collision); //dla obiektów
            ray_pos += ray_dir * empty_step_size * 3.0f;
            // Don't bother with occlusion ray if theres nothing there
            if (c_density >= occ_thresh || temp >= occ_thresh || coll >= occ_thresh) break;
            step++;
            if (step == empty_steps) goto NOTHING;
        }

        ray_pos -= ray_dir * (empty_step_size * 3.0f);

        steps /= 2;

        float R, G, B;
        R = G = B = 0.0f;
        // Trace ray through volume
#pragma unroll
        for (int step = 0; step < steps; step++) {
            // At each step, cast occlusion ray towards light source
            float c_density = get_cellF2(ray_pos, vd, volume) * density_influence;
            float temp = get_cellF2(ray_pos, vd, temper); //dla ognia
            float coll = get_cellF2(ray_pos, vd, collision); //dla obiektów
            if (c_density > 1.0) c_density = MAX_DENSITY; //bo �le si� renderuje beta
            float3 occ_pos = ray_pos;
            ray_pos += ray_dir * step_size;
            // Don't bother with occlusion ray if theres nothing there
            if (c_density < occ_thresh && temp < occ_thresh && coll < occ_thresh && render_collision) continue;
            else if (c_density < occ_thresh && temp < occ_thresh && !render_collision) continue;
            float transparency = 1.0;

            if (coll > 0 && render_collision) {
#pragma unroll
                for (int occ = 0; occ < 2; occ++) {
                    transparency *= maxf(get_cellF2(occ_pos, vd, collision), 0.0);
                    if (transparency > 1.0) transparency = 1.0; //beta
                    if (transparency < occ_thresh) break;
                    occ_pos += dir_to_light * step_size;
                }

                d_accum *= transparency;

                const int pixel = 3 * (y * img_dims.x + x);
                image[pixel + 0] = (uint8_t)(255.0 * d_accum);
                image[pixel + 1] = (uint8_t)(255.0 * d_accum);
                image[pixel + 2] = (uint8_t)(255.0 * d_accum);
                return;
            }
            else {

                if (render_shadows) {
#pragma unroll
                    for (int occ = 0; occ < int((steps / 2) * shadow_quality); occ++) {
                        transparency *= fmax(1.0 - get_cellF2(occ_pos, vd, volume), 0.0);
                        if (transparency > 1.0) transparency = 1.0; //beta
                        if (transparency < occ_thresh) break;
                        occ_pos += dir_to_light * step_size;
                    }
                }
                else {
                    transparency = transparency_compensation;
                }
            }


            d_accum *= fmax(1.0 - c_density, 0.0);
            d_accum2 *= (fmax(1 - temp, 0.0f) / Fire_Max_temp);

            if (Smoke_And_Fire) {
                if (fire_multiply == 0) {
                    R += (d_accum2 * 1.0f * temp);
                    G += (d_accum2 * 0.45f * temp);
                    B += (d_accum2 * 0.2f * temp);
                }
                else {
                    R += d_accum2 * 1.0f * temp * mix(transparency, 1 / maxf(c_density, 0.01), fire_multiply);
                    G += d_accum2 * 0.45f * temp * mix(transparency, 1 / maxf(c_density, 0.01), fire_multiply);
                    B += d_accum2 * 0.2f * temp * mix(transparency, 1 / maxf(c_density, 0.01), fire_multiply);
                }

            }


            R += d_accum * c_density * transparency;
            G += d_accum * c_density * transparency;
            B += d_accum * c_density * transparency;

            if (d_accum < occ_thresh) d_accum = 0;
            if (d_accum < occ_thresh && d_accum2 < occ_thresh) break;

        }

        const int pixel = 3 * (y * img_dims.x + x);
        image[pixel + 0] = (uint8_t)(fmin(255.0 * R, 255.0));
        image[pixel + 1] = (uint8_t)(fmin(255.0 * G, 255.0));
        image[pixel + 2] = (uint8_t)(fmin(255.0 * B, 255.0));
    }

    return;

NOTHING:
    const int pixel = 3 * (y * img_dims.x + x);
    image[pixel + 0] = (uint8_t)(0);
    image[pixel + 1] = (uint8_t)(0);
    image[pixel + 2] = (uint8_t)(0);
    return;
}


void render_fluid(uint8_t* render_target, int3 img_dims,
    float* d_volume, float* temper, float* coll, int3 vol_dims,
    float step_size, float3 light_dir, float3 cam_pos, float rotation, int STEPS, float Fire_Max_Temp = 5.0f, 
    bool Smoke_and_fire = false, float density_influence = 0.2, float fire_multiply = 0.5f,
    bool legacy_renderer = false, bool render_shadows = true, float transparency_compensation = 1.0f,
    float shadow_quality = 1.0f, bool render_collision = true) {

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
            device_img, d_volume, temper, coll, img_dims, vol_dims,
            step_size, light_dir, cam_pos, rotation, STEPS, Fire_Max_Temp, Smoke_and_fire,
            density_influence, fire_multiply, render_shadows, transparency_compensation,
            shadow_quality, render_collision);


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


























































__global__ void render_pixel_new(curandState* globalState, uint8_t* image, float* volume,
    float* temper, float* collision, int3 img_dims, int3 vol_dims, float step_size,
    float3 light_dir, float3 cam_pos, float rotation, int steps,
    float Fire_Max_temp = 5.0f, bool Smoke_And_Fire = false, float density_influence = 0.2, float fire_multiply = 0.5f,
    bool render_shadows = true, float transparency_compensation = 1.0f, float shadow_quality = 1.0f,
    bool render_collision = true, float random_noise = 0)
{
    step_size *= 512.0 / float(steps); //beta

    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x >= img_dims.x || y >= img_dims.y) return;



    int ind = (x * y) % 1024;
    curandState localState = globalState[ind];
    float RANDOM = curand_uniform(&localState);
    globalState[ind] = localState;
    random_noise = RANDOM;




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
    //randomness
    v_center.x += random_noise * step_size;
    v_center.z -= random_noise * step_size;
    v_center.y += random_noise * step_size;


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

    int empty_steps = steps / 2;
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
            float coll = get_cellF2(ray_pos, vd, collision); //dla obiektów
            ray_pos += ray_dir * empty_step_size * 3.0f;
            // Don't bother with occlusion ray if theres nothing there
            if (c_density >= occ_thresh || temp >= occ_thresh || coll >= occ_thresh) break;
            step++;
            if (step == empty_steps) goto NOTHING;
        }

        ray_pos -= ray_dir * (empty_step_size * 3.0f);

        steps /= 2;

        float R, G, B;
        R = G = B = 0.0f;
        // Trace ray through volume
#pragma unroll
        for (int step = 0; step < steps; step++) {
            // At each step, cast occlusion ray towards light source
            float c_density = get_cellF2(ray_pos, vd, volume) * density_influence;
            float temp = get_cellF2(ray_pos, vd, temper); //dla ognia
            float coll = get_cellF2(ray_pos, vd, collision); //dla obiektów
            if (c_density > 1.0) c_density = MAX_DENSITY; //bo �le si� renderuje beta
            float3 occ_pos = ray_pos;
            ray_pos += ray_dir * step_size;
            // Don't bother with occlusion ray if theres nothing there
            if (c_density < occ_thresh && temp < occ_thresh && coll < occ_thresh && render_collision) continue;
            else if (c_density < occ_thresh && temp < occ_thresh && !render_collision) continue;
            float transparency = 1.0;

            if (coll > 0 && render_collision) {
#pragma unroll
                for (int occ = 0; occ < 2; occ++) {
                    transparency *= maxf(get_cellF2(occ_pos, vd, collision), 0.0);
                    if (transparency > 1.0) transparency = 1.0; //beta
                    if (transparency < occ_thresh) break;
                    occ_pos += dir_to_light * step_size;
                }

                d_accum *= transparency;

                const int pixel = 3 * (y * img_dims.x + x);
                image[pixel + 0] = (uint8_t)(255.0 * d_accum);
                image[pixel + 1] = (uint8_t)(255.0 * d_accum);
                image[pixel + 2] = (uint8_t)(255.0 * d_accum);
                return;
            }
            else {

                if (render_shadows) {
#pragma unroll
                    for (int occ = 0; occ < int((steps / 2) * shadow_quality); occ++) {
                        transparency *= fmax(1.0 - get_cellF2(occ_pos, vd, volume), 0.0);
                        if (transparency > 1.0) transparency = 1.0; //beta
                        if (transparency < occ_thresh) break;
                        occ_pos += dir_to_light * step_size;
                    }
                }
                else {
                    transparency = transparency_compensation;
                }
            }


            d_accum *= fmax(1.0 - c_density, 0.0);
            d_accum2 *= (fmax(1 - temp, 0.0f) / Fire_Max_temp);

            if (Smoke_And_Fire) {
                if (fire_multiply == 0) {
                    R += (d_accum2 * 1.0f * temp);
                    G += (d_accum2 * 0.45f * temp);
                    B += (d_accum2 * 0.2f * temp);
                }
                else {
                    R += d_accum2 * 1.0f * temp * mix(transparency, 1 / maxf(c_density, 0.01), fire_multiply);
                    G += d_accum2 * 0.45f * temp * mix(transparency, 1 / maxf(c_density, 0.01), fire_multiply);
                    B += d_accum2 * 0.2f * temp * mix(transparency, 1 / maxf(c_density, 0.01), fire_multiply);
                }

            }


            R += d_accum * c_density * transparency;
            G += d_accum * c_density * transparency;
            B += d_accum * c_density * transparency;

            if (d_accum < occ_thresh) d_accum = 0;
            if (d_accum < occ_thresh && d_accum2 < occ_thresh) break;

        }

        const int pixel = 3 * (y * img_dims.x + x);
        image[pixel + 0] = (uint8_t)(fmin(255.0 * R, 255.0));
        image[pixel + 1] = (uint8_t)(fmin(255.0 * G, 255.0));
        image[pixel + 2] = (uint8_t)(fmin(255.0 * B, 255.0));
    }

    return;

NOTHING:
    const int pixel = 3 * (y * img_dims.x + x);
    image[pixel + 0] = (uint8_t)(0);
    image[pixel + 1] = (uint8_t)(0);
    image[pixel + 2] = (uint8_t)(0);
    return;
}






template <typename T>
__global__ void average(T* dst, T* src, int3 vd) {
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= vd.x || y >= vd.y) return;

    const int pixel = 3 * (y * vd.x + x);
    dst[pixel + 0] = maxf(dst[pixel+0], (dst[pixel+0]+src[pixel+0])/2.0f);
    dst[pixel + 1] = maxf(dst[pixel+1], (dst[pixel+1]+src[pixel+1])/2.0f);
    dst[pixel + 2] = maxf(dst[pixel+2], (dst[pixel+2]+src[pixel+2])/2.0f);
}

__global__ void setup_kernel(curandState* state, unsigned long seed)
{
    int id = threadIdx.x;
    curand_init(seed, id, 0, &state[id]);
}



void render_fluid_new(uint8_t* render_target, int3 img_dims,
    float* d_volume, float* temper, float* coll, int3 vol_dims,
    float step_size, float3 light_dir, float3 cam_pos, float rotation, int STEPS, float Fire_Max_Temp = 5.0f,
    bool Smoke_and_fire = false, float density_influence = 0.2, float fire_multiply = 0.5f,
    bool legacy_renderer = false, bool render_shadows = true, float transparency_compensation = 1.0f,
    float shadow_quality = 1.0f, bool render_collision = true, int RenderSamples = 8) {

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

    uint8_t* device_img_temp;
    cudaMalloc((void**)&device_img_temp, img_bytes);

    if (0 == device_img || 0 == device_img_temp)
    {
        printf("couldn't allocate GPU memory\n");
        return;
    }


    int N = 1024;
    dim3 tpb(N, 1, 1);
    curandState* devStates;
    cudaMalloc(&devStates, N * sizeof(curandState));
    setup_kernel << < 1, tpb >> > (devStates, time(NULL));


    //first pass
    render_pixel_new << <grid, block >> > ( devStates,
        device_img, d_volume, temper, coll, img_dims, vol_dims,
        step_size, light_dir, cam_pos, rotation, STEPS, Fire_Max_Temp, Smoke_and_fire,
        density_influence, fire_multiply, render_shadows, transparency_compensation,
        shadow_quality, render_collision);

    for (int sample = 0; sample < RenderSamples-1; sample++) {
        render_pixel_new << <grid, block >> > ( devStates,
            device_img_temp, d_volume, temper, coll, img_dims, vol_dims,
            step_size, light_dir, cam_pos, rotation, STEPS, Fire_Max_Temp, Smoke_and_fire,
            density_influence, fire_multiply, render_shadows, transparency_compensation,
            shadow_quality, render_collision);

        average << < grid, block >> > (device_img, device_img_temp, img_dims);

       
    }


    // Read image back
    cudaMemcpy(render_target, device_img, img_bytes, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop, 0);
    cudaThreadSynchronize();
    cudaEventElapsedTime(&measured_time, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::cout << "Render Time: " << measured_time << "  --  ";
    cudaFree(device_img);
    cudaFree(device_img_temp);
}
#endif