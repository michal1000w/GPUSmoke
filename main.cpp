#include "fluid.cuh"

int main()
{
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
    export_vdb("sphere", vol_d);



    clock_t startTime = clock();
    GRID3D sphere = load_vdb("sphere", vol_d);
    std::cout << "Loaded in : " << double(clock() - startTime) / (double)CLOCKS_PER_SEC << "s" << std::endl;

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



    //renderImage("sphere", 2);
    //exit(1);
    ////////////////

    //adding emmiters
    //object_list.push_back(OBJECT("emmiter", 18.0f, 50, 0.9, 5 ,0.9, make_float3(vol_d.x * 0.25, 10.0, 200.0)));
    //object_list.push_back(OBJECT("emmiter", 18.0f, 50, 0.6, 5, 0.9, make_float3(vol_d.x * 0.5, 10.0, 200.0)));
    //object_list.push_back(OBJECT("emmiter", 18.0f, 50, 0.3, 5, 0.9, make_float3(vol_d.x * 0.75, 10.0, 200.0)));
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