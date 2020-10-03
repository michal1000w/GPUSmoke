#include "IO.h"
#include "Simulation.cuh"
#include "Renderer.cuh"


void Medium_Scale(int3 vol_d, int3 img_d, uint8_t* img, float3 light, std::vector<OBJECT>& object_list, float3 cam, int ACCURACY_STEPS, int FRAMES, int STEPS) {
    fluid_state state(vol_d);
   
    state.f_weight = 0.05;
    state.time_step = 0.1;

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
            STEPS);

        save_image(img, img_d, "output/R" + pad_number(f + 1) + ".ppm");
        for (int st = 0; st < 1; st++) {
            simulate_fluid(state, object_list, ACCURACY_STEPS, DEBUG, f);
            state.step++;
            //DEBUG = false;
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

    int DOMAIN_RESOLUTION = 450;
    int FRAMES = 300;
    int STEPS = 96; //512 Rendering Samples
    int ACCURACY_STEPS = 8; //8
    float ZOOM = 1.8; //1.0
    std::vector<OBJECT> object_list;

    float Dissolve = 0.95;





    const int3 vol_d = make_int3(DOMAIN_RESOLUTION, DOMAIN_RESOLUTION, DOMAIN_RESOLUTION); //Domain resolution
    const int3 img_d = make_int3(720, 720, 0);




    //adding emmiters
    object_list.push_back(OBJECT("emmiter", 18.0f, 50, 0.9, 5 ,0.9, make_float3(vol_d.x * 0.25, 10.0, 200.0)));
    object_list.push_back(OBJECT("emmiter", 18.0f, 50, 0.6, 5, 0.9, make_float3(vol_d.x * 0.5, 10.0, 200.0)));
    object_list.push_back(OBJECT("emmiter", 18.0f, 50, 0.3, 5, 0.9, make_float3(vol_d.x * 0.75, 10.0, 200.0)));
    //object_list.push_back(OBJECT("smoke", 10, 50, 0.9, 50, 1.0, make_float3(vol_d.x * 0.5, 10.0, 200.0)));




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
    else {
        std::cout << "Domain resolution over 450^3 not supported yet" << std::endl;
        //Huge_Scale(vol_d, img_d, img, light, object_list, cam, ACCURACY_STEPS, FRAMES, STEPS);
    }
    
    std::cout << "Rendering animation video..." << std::endl;
    std::system("make_video.sh");
    //std::system("pause");

    return 0;
}