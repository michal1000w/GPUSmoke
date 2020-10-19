#define GUI


//#include "Solver.cuh"
#include "Window.h"

extern Solver solver;



int main(int argc, char* args[]) {
#ifdef EXPERIMENTAL
    solver.Initialize();
    solver.ClearCache();
    solver.ExportVDBScene();
    //solver.ExampleScene();
    float Window_Resolution[2] = { 1600, 640 };
    float Image_Resolution[2] = { 640, 640 };
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