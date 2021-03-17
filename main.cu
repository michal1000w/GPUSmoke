#define GUI


//#include "Solver.cuh"
#include "Window.h"

extern Solver solver;



/////////////////////////////////
//section experimental


/////////////////////////////////



int main(int argc, char* args[]) {
    //srand(1);
    int devicesCount;
    cudaGetDeviceCount(&devicesCount);
    std::cout << "Found " << devicesCount << " devices:" << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    for (int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex)
    {
        cudaDeviceProp deviceProperties;
        cudaGetDeviceProperties(&deviceProperties, deviceIndex);
        std::cout << deviceProperties.name << "   ->  " << deviceProperties.totalGlobalMem << std::endl;
    }
    int Best_Device_Index = 0;
    long long Memory = 0;
    for (int deviceIndex = 0; deviceIndex < devicesCount; deviceIndex++) {
        cudaDeviceProp deviceProperties;
        cudaGetDeviceProperties(&deviceProperties, deviceIndex);
        if (deviceProperties.totalGlobalMem > Memory){
            Memory = deviceProperties.totalGlobalMem;
            Best_Device_Index = deviceIndex;
        }
    }
    std::cout << "----------------------------------------" << std::endl;
    cudaSetDevice(Best_Device_Index);
    std::cout << "Choosing device: " << Best_Device_Index << std::endl;
#ifdef EXPERIMENTAL
    solver.Initialize();
#ifdef OBJECTS_EXPERIMENTAL
    //std::cout << "Generating example scene" << std::endl;
    solver.ExampleScene(true);
#else
    solver.ExportVDBScene();
#endif
    //solver.ExampleScene(true);
    float Window_Resolution[2] = { 1600, 800 };
    float Image_Resolution[2] = { 700, 700 };
    std::cout << "Setting image resolution" << std::endl;
    solver.setImageResolution(Image_Resolution[0], Image_Resolution[1]);

    solver.Initialize_Simulation();
    Window(Window_Resolution);
    solver.Clear_Simulation_Data();
    //std::cout << "Rendering animation video..." << std::endl;
    //std::system("make_video.sh");
#else
#ifdef GUI

    std::cout << "Hello" << std::endl;

    float Image_Resolution[2] = { 640, 640 };
    const int3 img_d = make_int3(Image_Resolution[0], Image_Resolution[1], 0);

    uint8_t* img = new uint8_t[3 * img_d.x * img_d.y];

    Window(Image_Resolution);
#else
    initialize();
#endif
#endif
    return 0;
}