#include <stdio.h>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <conio.h>
#include <cuda_fp16.h>
#include "cutil_math.h"

#include "HugeScaleSolver.cu"


// A couple IO utility functions

std::string pad_number(int n)
{
    std::ostringstream ss;
    ss << std::setw(7) << std::setfill('0') << n;
    return ss.str();
}

void save_image(uint8_t* pixels, int3 img_dims, std::string name) {
    std::ofstream file(name, std::ofstream::binary);
    if (file.is_open()) {
        file << "P6\n" << img_dims.x << " " << img_dims.y << "\n" << "255\n";
        file.write((char*)pixels, img_dims.x * img_dims.y * 3);
        file.close();
    }
    else {
        std::cout << "Could not open file :(\n";
    }
}









// Runs a single iteration of the simulation
void simulate_fluid(fluid_state& state, std::vector<OBJECT>& object_list, int ACCURACY_STEPS = 35)
{
    float AMBIENT_TEMPERATURE = 0.0f;//0.0f
    float BUOYANCY = 1.0f; //1.0f

    float measured_time = 0.0f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int s = 8;//8
    dim3 block(s, s, s);
    dim3 grid((state.dim.x + s - 1) / s,
        (state.dim.y + s - 1) / s,
        (state.dim.z + s - 1) / s);

    cudaEventRecord(start, 0);

    advection << <grid, block >> > (
        state.velocity->readTarget(),
        state.velocity->readTarget(),
        state.velocity->writeTarget(),
        state.dim, state.time_step, 1.0);//1.0
    state.velocity->swap();

    advection << <grid, block >> > (
        state.velocity->readTarget(),
        state.temperature->readTarget(),
        state.temperature->writeTarget(),
        state.dim, state.time_step, 0.998);//0.998
    state.temperature->swap();

    advection << <grid, block >> > (  //zanikanie
        state.velocity->readTarget(),
        state.density->readTarget(),
        state.density->writeTarget(),
        state.dim, state.time_step, 0.995);//0.9999
    state.density->swap();

    buoyancy << <grid, block >> > (
        state.velocity->readTarget(),
        state.temperature->readTarget(),
        state.density->readTarget(),
        state.velocity->writeTarget(),
        AMBIENT_TEMPERATURE, state.time_step, 1.0f, state.f_weight, state.dim);
    state.velocity->swap();

    float3 location = state.impulseLoc;


    /////Z - glebia
    /////X - lewo prawo
    /////Y - gora dol
    float MOVEMENT_SIZE = 9.0;//90.0
    float MOVEMENT_SPEED = 10.0;
    bool MOVEMENT = true;
    if (MOVEMENT) {
        location.x += MOVEMENT_SIZE * 2.0 * sinf(-0.04f * MOVEMENT_SPEED * float(state.step));//-0.003f
        //location.y += cosf(-0.03f * float(state.step));//-0.003f
        location.z += MOVEMENT_SIZE * cosf(-0.02f * MOVEMENT_SPEED * float(state.step));//-0.003f
    }

    for (int i = 0; i < object_list.size(); i++) {
        OBJECT current = object_list[i];
        if (current.get_type() == "smoke")
            object_list.erase(object_list.begin() + i); //remove emmiter from the list

        float3 SIZEE = make_float3(current.get_size(), current.get_size(), current.get_size());

        wavey_impulse << < grid, block >> > (
            state.temperature->readTarget(),
            current.get_location(), SIZEE,
            state.impulseTemp, current.get_initial_velocity(), current.get_velocity_frequence(),
            state.dim
            );
        wavey_impulse << < grid, block >> > (
            state.density->readTarget(),
            current.get_location(), SIZEE,
            state.impulseDensity, current.get_initial_velocity() * (1.0 / current.get_initial_velocity()), current.get_velocity_frequence(),
            state.dim
            );
    }



    divergence << <grid, block >> > (
        state.velocity->readTarget(),
        state.diverge, state.dim, 0.5);//0.5

// clear pressure
    impulse << <grid, block >> > (
        state.pressure->readTarget(),
        make_float3(0.0), 1000000.0f,
        0.0f, state.dim);

    for (int i = 0; i < ACCURACY_STEPS; i++)
    {
        pressure_solve << <grid, block >> > (
            state.diverge,
            state.pressure->readTarget(),
            state.pressure->writeTarget(),
            state.dim, -1.0);
        state.pressure->swap();
    }

    subtract_pressure << <grid, block >> > (
        state.velocity->readTarget(),
        state.velocity->writeTarget(),
        state.pressure->readTarget(),
        state.dim, 1.0);
    state.velocity->swap();

    cudaEventRecord(stop, 0);
    cudaThreadSynchronize();
    cudaEventElapsedTime(&measured_time, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::cout << "Simulation Time: " << measured_time << "  ||";
}





// Runs a single iteration of the simulation
void simulate_fluid(fluid_state_huge& state, std::vector<OBJECT>& object_list, int ACCURACY_STEPS = 35)
{
    float AMBIENT_TEMPERATURE = 0.0f;//0.0f
    float BUOYANCY = 1.0f; //1.0f

    float measured_time = 0.0f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int s = 8;//8
    dim3 block(s, s, s);
    dim3 grid((state.dim.x + s - 1) / s,
        (state.dim.y + s - 1) / s,
        (state.dim.z + s - 1) / s);

    cudaEventRecord(start, 0);

    advection << <grid, block >> > (
        state.velocity->readTarget(),
        state.velocity->readTarget(),
        state.velocity->writeTarget(),
        state.dim, state.time_step, 1.0);//1.0
    state.velocity->swap();

    advection << <grid, block >> > (
        state.velocity->readTarget(),
        state.temperature->readTarget(),
        state.temperature->writeTarget(),
        state.dim, state.time_step, 0.998);//0.998
    state.temperature->swap();

    advection << <grid, block >> > (  //zanikanie
        state.velocity->readTarget(),
        state.density->readTarget(),
        state.density->writeTarget(),
        state.dim, state.time_step, 0.995);//0.9999
    state.density->swap();

    buoyancy << <grid, block >> > (
        state.velocity->readTarget(),
        state.temperature->readTarget(),
        state.density->readTarget(),
        state.velocity->writeTarget(),
        AMBIENT_TEMPERATURE, state.time_step, 1.0f, state.f_weight, state.dim);
    state.velocity->swap();

    float3 location = state.impulseLoc;


    /////Z - glebia
    /////X - lewo prawo
    /////Y - gora dol
    float MOVEMENT_SIZE = 9.0;//90.0
    float MOVEMENT_SPEED = 10.0;
    bool MOVEMENT = true;
    if (MOVEMENT) {
        location.x += MOVEMENT_SIZE * 2.0 * sinf(-0.04f * MOVEMENT_SPEED * float(state.step));//-0.003f
        //location.y += cosf(-0.03f * float(state.step));//-0.003f
        location.z += MOVEMENT_SIZE * cosf(-0.02f * MOVEMENT_SPEED * float(state.step));//-0.003f
    }

    for (int i = 0; i < object_list.size(); i++) {
        OBJECT current = object_list[i];
        if (current.get_type() == "smoke")
            object_list.erase(object_list.begin() + i); //remove emmiter from the list

        float3 SIZEE = make_float3(current.get_size(), current.get_size(), current.get_size());

        wavey_impulse << < grid, block >> > (
            state.temperature->readTarget(),
            current.get_location(), SIZEE,
            state.impulseTemp, current.get_initial_velocity(), current.get_velocity_frequence(),
            state.dim
            );
        wavey_impulse << < grid, block >> > (
            state.density->readTarget(),
            current.get_location(), SIZEE,
            state.impulseDensity, current.get_initial_velocity() * (1.0 / current.get_initial_velocity()), current.get_velocity_frequence(),
            state.dim
            );
    }



    divergence << <grid, block >> > (
        state.velocity->readTarget(),
        state.diverge, state.dim, 0.5);//0.5

// clear pressure
    impulse << <grid, block >> > (
        state.pressure->readTarget(),
        make_float3(0.0), 1000000.0f,
        0.0f, state.dim);

    for (int i = 0; i < ACCURACY_STEPS; i++)
    {
        pressure_solve << <grid, block >> > (
            state.diverge,
            state.pressure->readTarget(),
            state.pressure->writeTarget(),
            state.dim, -1.0);
        state.pressure->swap();
    }

    subtract_pressure << <grid, block >> > (
        state.velocity->readTarget(),
        state.velocity->writeTarget(),
        state.pressure->readTarget(),
        state.dim, 1.0);
    state.velocity->swap();

    cudaEventRecord(stop, 0);
    cudaThreadSynchronize();
    cudaEventElapsedTime(&measured_time, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::cout << "Simulation Time: " << measured_time << "  ||";
}






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






void Medium_Scale(int3 vol_d, int3 img_d, uint8_t* img, float3 light, std::vector<OBJECT>& object_list, float3 cam, int ACCURACY_STEPS, int FRAMES, int STEPS) {
    fluid_state state(vol_d);

    state.impulseLoc = make_float3(0.5 * float(vol_d.x),
        0.5 * float(vol_d.y) - 170.0,
        0.5 * float(vol_d.z));
    state.impulseTemp = 20.0;//4.0
    state.impulseDensity = 0.6;//0.35
    state.impulseRadius = 18.0;//18.0
    state.f_weight = 0.05;
    state.time_step = 0.1;

    dim3 full_grid(vol_d.x / 8 + 1, vol_d.y / 8 + 1, vol_d.z / 8 + 1);
    dim3 full_block(8, 8, 8);


    for (int f = 0; f <= FRAMES; f++) {

        std::cout << "\rFrame " << f + 1 << "  -  ";

        render_fluid(
            img, img_d,
            state.density->readTarget(),
            state.temperature->readTarget(),
            vol_d, 1.0, light, cam, 0.0 * float(state.step),
            STEPS);

        save_image(img, img_d, "output/R" + pad_number(f + 1) + ".ppm");
        for (int st = 0; st < 1; st++) {
            simulate_fluid(state, object_list, ACCURACY_STEPS);
            state.step++;
        }
    }

    delete[] img;

    printf("CUDA: %s\n", cudaGetErrorString(cudaGetLastError()));

    cudaThreadExit();
}

void Huge_Scale(int3 vol_d, int3 img_d, uint8_t* img, float3 light, std::vector<OBJECT>& object_list, float3 cam, int ACCURACY_STEPS, int FRAMES, int STEPS) {
    fluid_state_huge state(vol_d);

    state.impulseLoc = make_float3(0.5 * float(vol_d.x),
        0.5 * float(vol_d.y) - 170.0,
        0.5 * float(vol_d.z));
    state.impulseTemp = 20.0;//4.0
    state.impulseDensity = 0.6;//0.35
    state.impulseRadius = 18.0;//18.0
    state.f_weight = 0.05;
    state.time_step = 0.1;

    dim3 full_grid(vol_d.x / 8 + 1, vol_d.y / 8 + 1, vol_d.z / 8 + 1);
    dim3 full_block(8, 8, 8);


    for (int f = 0; f <= FRAMES; f++) {

        std::cout << "\rFrame " << f + 1 << "  -  ";

        if (_kbhit()) {
            std::cout << "Stopping simulation\n";
            break;
        }

        render_fluid(
            img, img_d,
            state.density->readTarget(),
            state.temperature->readTarget(),
            vol_d, 1.0, light, cam, 0.0 * float(state.step),
            STEPS);

        save_image(img, img_d, "output/R" + pad_number(f + 1) + ".ppm");
        for (int st = 0; st < 1; st++) {
            simulate_fluid(state, object_list, ACCURACY_STEPS);
            state.step++;
        }
    }

    delete[] img;

    printf("CUDA: %s\n", cudaGetErrorString(cudaGetLastError()));

    cudaThreadExit();
}

int main(int argc, char* args[])
{

    int DOMAIN_RESOLUTION = 550;
    int FRAMES = 450;
    int STEPS = 196; //512
    int ACCURACY_STEPS = 16; //8
    float ZOOM = 1.8; //1.0
    std::vector<OBJECT> object_list;







    const int3 vol_d = make_int3(DOMAIN_RESOLUTION, DOMAIN_RESOLUTION, DOMAIN_RESOLUTION); //Domain resolution
    const int3 img_d = make_int3(720, 720, 0);




    //adding emmiters
    object_list.push_back(OBJECT("emmiter", 16.0f, 500, 100, make_float3(vol_d.x * 0.25, 10.0, 200.0)));
    object_list.push_back(OBJECT("smoke", 10, 500, 100, make_float3(vol_d.x * 0.5, 10.0, 200.0)));




    float3 cam;
    cam.x = static_cast<float>(vol_d.x) * 0.5;
    cam.y = static_cast<float>(vol_d.y) * 0.5;
    cam.z = static_cast<float>(vol_d.z) * -0.4 * (1.0 / ZOOM);//0.0   minus do ty�u, plus do przodu
    float3 light;
    //X - lewo prawo
    //Y - g�ra d�
    //Z - prz�d ty�
    light.x = 5.0;//0.1
    light.y = 1.0;//1.0
    light.z = -0.5;//-0.5

    uint8_t* img = new uint8_t[3 * img_d.x * img_d.y];

    std::cout << "Clearing previous frames\n";
    std::system("erase_imgs.sh");

    if (DOMAIN_RESOLUTION <= 450)
        Medium_Scale(vol_d, img_d, img, light, object_list, cam, ACCURACY_STEPS, FRAMES, STEPS);
    else
        Huge_Scale(vol_d, img_d, img, light, object_list, cam, ACCURACY_STEPS, FRAMES, STEPS);
    
    std::cout << "Rendering animation video..." << std::endl;
    std::system("make_video.sh");
    std::system("pause");

    return 0;
}