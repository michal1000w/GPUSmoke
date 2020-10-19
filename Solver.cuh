#pragma once
#include "IO.h"
#include "Simulation.cuh"
#include "Renderer.cuh"



#define EXPERIMENTAL
#define OBJECTS_EXPERIMENTAL




#ifdef EXPERIMENTAL

class Solver {
public:
    int ACCURACY_STEPS;//
    float Smoke_Dissolve;//
    float Ambient_Temperature;//
    float Fire_Max_Temperature;//
    int STEPS;//
    bool Smoke_And_Fire;//
    int3 New_DOMAIN_RESOLUTION;//
    float speed;
    std::vector<OBJECT> object_list;
    bool preserve_object_list;
private:
    int3 DOMAIN_RESOLUTION;
    int FRAMES;
    float Image_Resolution[2];
    float ZOOM;
    int3 vol_d;
    int3 img_d;
    uint8_t* img;
    float time_step;
    float rotation;

    float3 Camera;
    float3 Light;
    fluid_state* state;
    dim3 full_grid;
    dim3 full_block;
public:
    void UpdateTimeStep() {
        time_step = speed * 0.1; //chyba dobre
    }
    void UpdateDomainResolution() {
        if (New_DOMAIN_RESOLUTION.x * New_DOMAIN_RESOLUTION.y * New_DOMAIN_RESOLUTION.z <= 490*490*490){
            DOMAIN_RESOLUTION = New_DOMAIN_RESOLUTION;
            std::cout << "New domain resolution set\n";
            }
        else
            std::cout << "\nError!!!\nDomain Size too Big\n";

        vol_d = make_int3(DOMAIN_RESOLUTION.x, DOMAIN_RESOLUTION.y, DOMAIN_RESOLUTION.z); //Domain resolution
    }
    unsigned int frame;
    void setImageResolution(unsigned int x, unsigned int y) {
        Image_Resolution[0] = x;
        Image_Resolution[1] = y;
        img_d = make_int3(Image_Resolution[0], Image_Resolution[1], 0);
    }

    int3 getImageResoltion() const {
        return img_d;
    }

    void ClearCache() {
        std::cout << "Clearing previous frames\n";
        //std::system("erase_imgs.sh");
        std::system("rm ./output/*.bmp");
        std::system("rm ./output/cache/*");
        std::cout << "Done";
    }

    void ExportVDBScene() {
        std::cout << "Exporting scene\n";
        export_vdb("sphere", vol_d);
        std::cout << "Done\n";


        clock_t startTime = clock();
        
        GRID3D sphere(vol_d.x, vol_d.y, vol_d.z);
        load_vdb("sphere", vol_d, sphere);
        std::cout << "Loaded in : " << double(clock() - startTime) << std::endl;
        if (false) {
            OBJECT SPHERE("vdb", 18.0f, 50, 0.9, 5, 0.9, make_float3(vol_d.x * 0.25, 10.0, 200.0));
            SPHERE.load_density_grid(sphere, 3.0);
            object_list.push_back(SPHERE);
            SPHERE.free();
        }
        else {
            OBJECT SPHERE("vdbsingle", 18.0f, 50, 0.9, 5, 0.9, make_float3(vol_d.x * 0.25, 10.0, 200.0));
            SPHERE.load_density_grid(sphere, 6.0);
            object_list.push_back(SPHERE);
            SPHERE.free();
        }
        
        sphere.free();
        
    }

    void ExampleScene() {
        //adding emmiters
        if (!preserve_object_list) {
            object_list.push_back(OBJECT("emmiter", 18.0f, 50, 0.9, 5, 0.9, make_float3(vol_d.x * 0.25, 10.0, 200.0), object_list.size()));
            object_list.push_back(OBJECT("emmiter", 18.0f, 50, 0.6, 5, 0.9, make_float3(vol_d.x * 0.5, 10.0, 200.0), object_list.size()));
            object_list.push_back(OBJECT("emmiter", 18.0f, 50, 0.3, 5, 0.9, make_float3(vol_d.x * 0.75, 10.0, 200.0), object_list.size()));
            object_list.push_back(OBJECT("smoke", 10, 50, 0.9, 50, 1.0, make_float3(vol_d.x * 0.5, 30.0, 200.0), object_list.size()));
        }
    }

    void Initialize() {
        openvdb::initialize();

        srand(0);
        //simulation settings
        DOMAIN_RESOLUTION = make_int3(64, 64, 64);
        New_DOMAIN_RESOLUTION = DOMAIN_RESOLUTION;
        ACCURACY_STEPS = 8; //8
        object_list;

        Smoke_Dissolve = 0.995f; //0.995f
        Ambient_Temperature = 0.0f; //0.0f
        speed = 1.0; //1.0


        //rendering settings
        FRAMES = 500;
        Fire_Max_Temperature = 50.0f;
        Image_Resolution[0] = 640;
        Image_Resolution[1] = 640;
        STEPS = 100; //512 Rendering Samples
        ZOOM = 0.45; //1.8
        Smoke_And_Fire = true;



        time_step = speed * 0.1; //chyba dobre


        vol_d = make_int3(DOMAIN_RESOLUTION.x, DOMAIN_RESOLUTION.y, DOMAIN_RESOLUTION.z); //Domain resolution
        img_d = make_int3(Image_Resolution[0], Image_Resolution[1], 0);





        //X - lewo prawo
        //Y - g�ra d�
        //Z - prz�d ty�
        setCamera(static_cast<float>(vol_d.x) * 0.5,
            static_cast<float>(vol_d.y) * 0.5,
            static_cast<float>(vol_d.z) * -0.4 * (1.0 / ZOOM));

        setLight(5.0, 1.0, -0.5);

    }

    Solver() {
        std::cout << "Create Solver Instance" << std::endl;
        preserve_object_list = true;
    }
    
    void setCamera(float x, float y, float z) {
        Camera.x = x;
        Camera.y = y;
        Camera.z = z;
    }

    float3 getCamera() {
        return Camera;
    }

    void setLight(float x, float y, float z) {
        Light.x = x;
        Light.y = y;
        Light.z = z;
    }

    void setRotation(float rotation = 0.0f) {
        this->rotation = rotation;
    }

    float getRotation() const {
        return this->rotation;
    }

    void Initialize_Simulation() {
        state = new fluid_state(vol_d);


        state->f_weight = 0.05;
        state->time_step = time_step;// 0.1;

        full_grid = dim3(vol_d.x / 8 + 1, vol_d.y / 8 + 1, vol_d.z / 8 + 1);
        full_block = dim3(8, 8, 8);

        img = new uint8_t[3 * img_d.x * img_d.y];
    }

    void Clear_Simulation_Data() {
        delete state;
        delete[] img;

        if (!preserve_object_list) {
            for (auto i : object_list) {
                if (i.get_type() == "vdb" || i.get_type() == "vdbs")
                    i.cudaFree();
                i.free();
            }
            object_list.clear();
        }

        printf("CUDA: %s\n", cudaGetErrorString(cudaGetLastError()));

        cudaThreadExit();
    }

    void Simulation_Frame(unsigned int frame = 0) {
        unsigned int f = frame;
        std::cout << "\rFrame " << f + 1 << "  -  ";

        for (int st = 0; st < 1; st++) {
            simulate_fluid(*state, object_list, ACCURACY_STEPS, false, f, Smoke_Dissolve, Ambient_Temperature);
            state->step++;
        }


        render_fluid(
            img, img_d,
            state->density->readTarget(),
            state->temperature->readTarget(),
            vol_d, 1.0, Light, Camera, rotation,
            STEPS, Fire_Max_Temperature, Smoke_And_Fire);

        //save_image(img, img_d, "output/R" + pad_number(f + 1) + ".ppm");
        generateBitmapImage(img, img_d.x, img_d.y, ("output/R" + pad_number(f + 1) + ".bmp").c_str());

        if (false) {
            GRID3D* arr = new GRID3D();
            GRID3D* arr_temp = new GRID3D();
            arr->set_pointer(state->density->readToGrid());
            arr_temp->set_pointer(state->temperature->readToGrid());
            export_openvdb("frame." + std::to_string(f), vol_d, arr, arr_temp);
            arr->free();
            arr_temp->free();
            delete arr;
            delete arr_temp;
        }
    }
    unsigned char* loadImgData() {
        return img;
    }
};

#else

void Medium_Scale(int3 vol_d, int3 img_d, uint8_t* img,
    float3 light, std::vector<OBJECT>& object_list, float3 cam,
    int ACCURACY_STEPS, int FRAMES, int STEPS, float Dissolve_rate,
    float Ambient_temp, float Fire_Max_Temp, bool Smoke_and_Fire,
    float time_step) {


    fluid_state state(vol_d);


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


#ifndef NOSAVE
        GRID3D* arr = new GRID3D();
        GRID3D* arr_temp = new GRID3D();
        arr->set_pointer(state.density->readToGrid());
        arr_temp->set_pointer(state.temperature->readToGrid());
        export_openvdb("frame." + std::to_string(f), vol_d, arr, arr_temp);
        arr->free();
        arr_temp->free();
#endif
    }

    delete[] img;

    printf("CUDA: %s\n", cudaGetErrorString(cudaGetLastError()));

    cudaThreadExit();
}

int initialize() {
    openvdb::initialize();
    srand(0);
    //simulation settings
    int3 DOMAIN_RESOLUTION = make_int3(256, 600, 256);
    int ACCURACY_STEPS = 8; //8
    std::vector<OBJECT> object_list;

    float Smoke_Dissolve = 0.995f; //0.995f
    float Ambient_Temperature = 0.0f; //0.0f
    float speed = 1.0; //1.0




    //rendering settings
    int FRAMES = 500;
    float Fire_Max_Temperature = 50.0f;
    float Image_Resolution[2] = { 640, 640 };
    int STEPS = 100; //512 Rendering Samples
    float ZOOM = 0.45; //1.8
    bool Smoke_And_Fire = true;



    float time_step = 0.1; //0.1
    time_step = speed * 0.1; //chyba dobre


    const int3 vol_d = make_int3(DOMAIN_RESOLUTION.x, DOMAIN_RESOLUTION.y, DOMAIN_RESOLUTION.z); //Domain resolution
    const int3 img_d = make_int3(Image_Resolution[0], Image_Resolution[1], 0);








    /////////VDB
#define VDBB
#ifdef VDBB
    export_vdb("sphere", vol_d);



    clock_t startTime = clock();
    GRID3D sphere = load_vdb("sphere", vol_d);
    std::cout << "Loaded in : " << double(clock() - startTime) << std::endl;

    if (false) {
        OBJECT SPHERE("vdb", 18.0f, 50, 0.9, 5, 0.9, make_float3(vol_d.x * 0.25, 10.0, 200.0));
        SPHERE.load_density_grid(sphere, 3.0);
        object_list.push_back(SPHERE);
    }
    else {
        OBJECT SPHERE("vdbsingle", 18.0f, 50, 0.9, 5, 0.9, make_float3(vol_d.x * 0.25, 10.0, 200.0));
        SPHERE.load_density_grid(sphere, 6.0);
        object_list.push_back(SPHERE);
    }
#else


    ////////////////
    if (EXAMPLE__ == 1) {
        //adding emmiters
        object_list.push_back(OBJECT("emmiter", 18.0f, 50, 0.9, 5, 0.9, make_float3(vol_d.x * 0.25, 10.0, 200.0)));
        object_list.push_back(OBJECT("emmiter", 18.0f, 50, 0.6, 5, 0.9, make_float3(vol_d.x * 0.5, 10.0, 200.0)));
        object_list.push_back(OBJECT("emmiter", 18.0f, 50, 0.3, 5, 0.9, make_float3(vol_d.x * 0.75, 10.0, 200.0)));
        object_list.push_back(OBJECT("smoke", 10, 50, 0.9, 50, 1.0, make_float3(vol_d.x * 0.5, 30.0, 200.0)));
    }
    else if (EXAMPLE__ == 2) {
        object_list.push_back(OBJECT("emmiter", 18.0f, 50, 0.6, 5, 0.9, make_float3(/*left-right*/vol_d.x * 0.5, 10.0, /*front-back*/200)));
        ZOOM = 1.2; //1.8
    }
#endif


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
    std::system("rm ./output/cache/*");

    if (DOMAIN_RESOLUTION.x * DOMAIN_RESOLUTION.y * DOMAIN_RESOLUTION.z <= 100000000)
        Medium_Scale(vol_d, img_d, img, light, object_list, cam, ACCURACY_STEPS, FRAMES, STEPS, Smoke_Dissolve, Ambient_Temperature, Fire_Max_Temperature, Smoke_And_Fire, time_step);
    else {
        std::cout << "Domain resolution over 450^3 not supported yet" << std::endl;
        //Huge_Scale(vol_d, img_d, img, light, object_list, cam, ACCURACY_STEPS, FRAMES, STEPS);
    }

    for (int i = 0; i < object_list.size(); i++) { //czyszczenie pamięci GPU
        object_list[i].cudaFree();
    }

    std::cout << "Rendering animation video..." << std::endl;
    std::system("make_video.sh");
    //std::system("pause");

    return 0;
}

#endif