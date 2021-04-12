#define GUI


//#include "Solver.cuh"
#include "Window.h"

extern Solver solver;



/////////////////////////////////
//section experimental
void EnableP2Psharing(unsigned int devices_count = 1) {
    std::cout << "Enabling P2P sharing..." << std::endl;
    for (unsigned int i = 0; i < devices_count; i++) {
        for (unsigned int j = 0; j < devices_count; j++) {
            int is_able = NULL;
            cudaSetDevice(i);
            cudaDeviceCanAccessPeer(&is_able, i, j);
            if (is_able) {
                checkCudaErrors(cudaDeviceEnablePeerAccess(j, 0));
                std::cout << "Enabled P2P sharing for: " << i << std::endl;
            }
        }
    }
}

/////////////////////////////////



int main(int argc, char* argv[]) {
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

    if (false) {
        cudaDeviceProp deviceProperties;
        cudaGetDeviceProperties(&deviceProperties, Best_Device_Index);
        cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, deviceProperties.persistingL2CacheMaxSize); /* Set aside max possible size of L2 cache for persisting accesses */
        std::cout << "Setting L2 max cache: " << deviceProperties.persistingL2CacheMaxSize << std::endl;
    }

    EnableP2Psharing(devicesCount);

#ifdef EXPERIMENTAL
    if (argc <= 1) {
        if (false) {
            std::cout << "Using All (" << devicesCount << ") devices" << std::endl;
            solver.Initialize(devicesCount);
        }
        else {
            std::cout << "Using (" << 1 << ") device" << std::endl;
            solver.Initialize(1);
        }
    }
    else {
        std::cout << "Using " << std::stoi(argv[1]) << " devices" << std::endl;
        solver.Initialize(std::stoi(argv[1]));
    }
#ifdef OBJECTS_EXPERIMENTAL
    std::cout << "Generating example scene" << std::endl;
    solver.ExampleScene(true);
#else
    solver.ExportVDBScene();
#endif
    //solver.ExampleScene(true);
    float Window_Resolution[2] = { 1600, 800 };
    float Image_Resolution[2] = { 700, 900 };
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