#ifndef FLUIDD
#define FLUIDD

#include "IO.h"
#include "Simulation.cuh"
#include "Renderer.cuh"



void Medium_Scale(int3 vol_d, int3 img_d, uint8_t* img, 
    float3 light, std::vector<OBJECT>& object_list, float3 cam, 
    int ACCURACY_STEPS, int FRAMES, int STEPS, float Dissolve_rate, 
    float Ambient_temp, float Fire_Max_Temp, bool Smoke_and_Fire,
    float time_step) {
    
    
    fluid_state state(vol_d);

    //VDB
    
    //VDB


    state.f_weight = 0.05;
    state.time_step = time_step;// 0.1;

    dim3 full_grid(vol_d.x / 8 + 1, vol_d.y / 8 + 1, vol_d.z / 8 + 1);
    dim3 full_block(8, 8, 8);

    bool DEBUG = true;
    for (int f = 0; f <= FRAMES; f++) {

        std::cout << "\rFrame " << f + 1 << "  -  ";

        render_fluid(
            img, img_d,
            state.density->readTarget(),
            state.temperature->readTarget(),
            vol_d, 1.0, light, cam, 0.0 * float(state.step),
            STEPS, Fire_Max_Temp, Smoke_and_Fire);

        save_image(img, img_d, "output/R" + pad_number(f + 1) + ".ppm");
        for (int st = 0; st < 1; st++) {
            simulate_fluid(state, object_list, ACCURACY_STEPS, DEBUG, f, Dissolve_rate, Ambient_temp);
            state.step++;
            //DEBUG = false;
        }
    }

    delete[] img;

    printf("CUDA: %s\n", cudaGetErrorString(cudaGetLastError()));

    cudaThreadExit();
}
/*
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
*/

#endif