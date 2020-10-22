#pragma once
#include "IO.h"
#include "Simulation.cuh"
#include "Renderer.cuh"



#define EXPERIMENTAL
#define OBJECTS_EXPERIMENTAL




#ifdef EXPERIMENTAL

std::string trim(const std::string& str,
    const std::string& whitespace = " \t")
{
    const auto strBegin = str.find_first_not_of(whitespace);
    if (strBegin == std::string::npos)
        return ""; // no content

    const auto strEnd = str.find_last_not_of(whitespace);
    const auto strRange = strEnd - strBegin + 1;

    return str.substr(strBegin, strRange);
}

std::string reduce(const std::string& str,
    const std::string& fill = " ",
    const std::string& whitespace = " \t")
{
    // trim first
    auto result = trim(str, whitespace);

    // replace sub ranges
    auto beginSpace = result.find_first_of(whitespace);
    while (beginSpace != std::string::npos)
    {
        const auto endSpace = result.find_first_not_of(whitespace, beginSpace);
        const auto range = endSpace - beginSpace;

        result.replace(beginSpace, range, fill);

        const auto newStart = beginSpace + fill.length();
        beginSpace = result.find_first_of(whitespace, newStart);
    }

    return result;
}

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
    int SAMPLE_SCENE;
    int EXPORT_END_FRAME;
    int EXPORT_START_FRAME;
    char EXPORT_FOLDER[100] = { 0 };
    char SAVE_FOLDER[100] = { 0 };
    char OPEN_FOLDER[100] = { 0 };
    bool EXPORT_VDB;
    bool SIMULATE;
    bool DONE_FRAME;
    float DIVERGE_RATE = 0.5f;
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
    bool what_it_is(std::string in) {
        if (in == "true" || in == "True") {
            return true;
        }
        else if (in == "false" || in == "False") {
            return false;
        }
        else {
            return 2;
        }
    }

    std::vector<std::string> getFilesList(std::string directory) {
        return get_file_list(directory);
    }

    void SaveSceneToFile(std::string fielname) {
        std::vector<std::string> lines;
        //header
        lines.push_back("#name=JFLOW");
        lines.push_back("#version=" + VERSION);
        lines.push_back("#type=" + RELEASE_STATUS);
        //simulation info
        lines.push_back("#scene_type=" + std::to_string(SAMPLE_SCENE));
        lines.push_back("#domain_resolution_x=" +
            std::to_string(DOMAIN_RESOLUTION.x));
        lines.push_back("#domain_resolution_y=" +
            std::to_string(DOMAIN_RESOLUTION.y));
        lines.push_back("#domain_resolution_z=" +
            std::to_string(DOMAIN_RESOLUTION.z));
        lines.push_back("#ambient_temperature=" + std::to_string(Ambient_Temperature));
        lines.push_back("#smoke_dissolve=" + std::to_string(Smoke_Dissolve));
        lines.push_back("#simulation_accuracy=" + std::to_string(ACCURACY_STEPS));
        //render info
        lines.push_back("#fire_and_smoke_render=" + std::to_string(Smoke_And_Fire));
        lines.push_back("#fire_emmision_rate=" + std::to_string(Fire_Max_Temperature));
        lines.push_back("#render_samples=" + std::to_string(STEPS));
        lines.push_back("#preserve_object_list=" + std::to_string(preserve_object_list));
        //cache info
        std::string FOLDER = EXPORT_FOLDER;
        FOLDER = trim(FOLDER);
        lines.push_back("#output_cache=" + FOLDER);
        lines.push_back("#end_frame=" + std::to_string(EXPORT_END_FRAME));
        lines.push_back("#start_frame=" + std::to_string(EXPORT_START_FRAME));

        for (int i = 0; i < object_list.size(); i++) {
            std::string current = "#object={";

            current += object_list[i].get_type() + ";";
            current += std::to_string(object_list[i].location.x) + ";";
            current += std::to_string(object_list[i].location.y) + ";";
            current += std::to_string(object_list[i].location.z) + ";";
            current += std::to_string(object_list[i].size) + ";";
            current += std::to_string(object_list[i].velocity_frequence) + ";"; //nowe
            current += std::to_string(object_list[i].force_strength) + ";"; //nowe
            current += std::to_string(object_list[i].force_direction[0]) + ";";
            current += std::to_string(object_list[i].force_direction[1]) + ";";
            current += std::to_string(object_list[i].force_direction[2]) + ";";
            current += std::to_string(object_list[i].square) + "";
            current += "}";

            lines.push_back(current);
        }

        save_scene_to_file(fielname, lines);
    }


    void LoadSceneFromFile(std::string filename, bool DEBUG = true) {
        std::vector<std::string> lines = load_scene_from_file(filename);

        if (lines[0] == "ERROR") return;

        std::cout << "Loading scene\n";
        preserve_object_list = false;
        frame = 0;


        std::vector<std::string> podzial;
        for (std::string i : lines) {
            std::string fragment = "";
            std::string fragment2 = "";
            int i1 = 1;
            int length = i.length();
            while (i1 < length) {
                if (i[i1] == '=')
                    break;
                fragment += i[i1];
                i1++;
            }
            i1++;
            while (i1 < length) {
                fragment2 += i[i1];
                i1++;
            }
            podzial.push_back(fragment);
            podzial.push_back(fragment2);
        }


        std::cout << lines[0] << std::endl;
        if (lines[0].find("JFLOW") == -1) {
            std::cout << "Invalid JFlow file" << std::endl;
            return;
        }

        Clear_Simulation_Data();

        int zrobione = 0;

        for (int line = 0; line < lines.size(); line++) {
            if (lines[line][0] != '#') continue;

            std::cout << podzial[line*2] << "  ->  " << podzial[line*2+1] << std::endl;

            if (podzial[line * 2] == "scene_type") {
                if (podzial[line * 2 + 1] == "ELEMENTS") {
                    SAMPLE_SCENE = 0;
                }
                else {
                    SAMPLE_SCENE = stoi(podzial[line * 2 + 1]);
                }
            }
            if (podzial[line * 2] == "domain_resolution_x") {   
                New_DOMAIN_RESOLUTION.x = stoi(podzial[line * 2 + 1]);
            }
            if (podzial[line * 2] == "domain_resolution_y") {
                New_DOMAIN_RESOLUTION.y = stoi(podzial[line * 2 + 1]);
            }
            if (podzial[line * 2] == "domain_resolution_z") {
                New_DOMAIN_RESOLUTION.z = stoi(podzial[line * 2 + 1]);
            }

            if (podzial[line * 2] == "ambient_temperature") {
                Ambient_Temperature = stof(podzial[line * 2 + 1]);
            }
            if (podzial[line * 2] == "smoke_dissolve") {
                Smoke_Dissolve = stof(podzial[line * 2 + 1]);
            }
            if (podzial[line * 2] == "simulation_accuracy") {
                ACCURACY_STEPS = stoi(podzial[line * 2 + 1]);
            }
            if (podzial[line * 2] == "fire_and_smoke_render") {
                Smoke_And_Fire = what_it_is(podzial[line * 2 + 1]);
            }
            if (podzial[line * 2] == "fire_emmision_rate") {
                Fire_Max_Temperature = stoi(podzial[line * 2 + 1]);
            }
            if (podzial[line * 2] == "render_samples") {
                STEPS = stoi(podzial[line * 2 + 1]);
            }
            if (podzial[line * 2] == "preserve_object_list") {
                preserve_object_list = what_it_is(podzial[line * 2 + 1]);
            }
            if (podzial[line * 2] == "output_cache") {
                for (int j = 0; j < 100; j++)
                    EXPORT_FOLDER[j] = 0;
                for (int j = 0; j < podzial[line * 2 + 1].length(); j++) {
                    EXPORT_FOLDER[j] = podzial[line * 2 + 1][j];
                }
            }
            if (podzial[line * 2] == "end_frame") {
                EXPORT_END_FRAME = stoi(podzial[line * 2 + 1]);
            }
            if (podzial[line * 2] == "start_frame") {
                EXPORT_START_FRAME = stoi(podzial[line * 2 + 1]);
            }

            if (podzial[line * 2] == "object") {
                int i1 = 1;
                std::vector<std::string> attributes;
                std::string current = "";
                while (i1 < podzial[line * 2 + 1].length()) {
                    if (podzial[line * 2 + 1][i1] == ';' || podzial[line * 2 + 1][i1] == '}') {
                        attributes.push_back(current);
                        current = "";
                        i1++;
                        continue;
                    }
                    current += podzial[line * 2 + 1][i1];
                    i1++;
                }

                object_list.push_back(OBJECT(attributes[0],/*name*/
                    stof(attributes[4]),/*size*/
                    50,  /*initial velocity*/
                    stof(attributes[5]), /*velocity frequence*/
                    5, 0.9,/*temperature, density*/
                    make_float3(stof(attributes[1]), stof(attributes[2]), stof(attributes[3])), /*position*/
                    object_list.size()/*ID*/
                ));
                object_list[object_list.size() - 1].force_strength = stof(attributes[6]);
                object_list[object_list.size() - 1].force_direction[0] = stof(attributes[7]);
                object_list[object_list.size() - 1].force_direction[1] = stof(attributes[8]);
                object_list[object_list.size() - 1].force_direction[2] = stof(attributes[9]);
                object_list[object_list.size() - 1].square = stoi(attributes[10]);

            }

        }
        UpdateDomainResolution();
        Initialize_Simulation();
    }


    int3 getDomainResolution() const {
        return DOMAIN_RESOLUTION;
    }
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
        std::string FOLDER = EXPORT_FOLDER;
        FOLDER = trim(FOLDER);
        //std::system("rm ./output/cache/*");
        std::system(("rm ./" + FOLDER + "*").c_str());

        std::experimental::filesystem::create_directory(FOLDER);
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
        if (SAMPLE_SCENE == 2) {
            OBJECT SPHERE("vdb", 18.0f, 50, 0.9, 5, 0.9, make_float3(vol_d.x * 0.25, 10.0, 200.0));
            SPHERE.load_density_grid(sphere, 3.0);
            object_list.push_back(SPHERE);
            SPHERE.free();
        }
        else if (SAMPLE_SCENE == 1){
            OBJECT SPHERE("vdbsingle", 18.0f, 50, 0.9, 5, 0.9, make_float3(vol_d.x * 0.25, 10.0, 200.0));
            SPHERE.load_density_grid(sphere, 6.0);
            object_list.push_back(SPHERE);
            SPHERE.free();
        }
        
        sphere.free();
        
    }

    void ExampleScene(bool force = false) {
        //adding emitters
        if (!preserve_object_list || force) {
            object_list.push_back(OBJECT("emitter", 18.0f, 50, 0.9, 5, 0.9, make_float3(vol_d.x * 0.25, 10.0, vol_d.z/2.0), object_list.size()));
            //object_list.push_back(OBJECT("emitter", 18.0f, 50, 0.6, 5, 0.9, make_float3(vol_d.x * 0.5, 0.0, vol_d.z / 2.0), object_list.size()));
            object_list.push_back(OBJECT("emitter", 18.0f, 50, 0.3, 5, 0.9, make_float3(vol_d.x * 0.75, 10.0, vol_d.z / 2.0), object_list.size()));
        }
    }

    void Initialize() {
        openvdb::initialize();
        DONE_FRAME = true;

        srand(0);
        //simulation settings
        DOMAIN_RESOLUTION = make_int3(96, 490, 96);
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
            static_cast<float>(vol_d.z) * -1.1 * (1.0 / ZOOM));

        setLight(5.0, 1.0, -0.5);

    }

    Solver() {
        std::cout << "Create Solver Instance" << std::endl;
        SAMPLE_SCENE = 0;
        EXPORT_START_FRAME = 0;
        EXPORT_END_FRAME = 500;
        std::string folder = "output/cache/";
        for (int i = 0; i < folder.length(); i++)
            EXPORT_FOLDER[i] = folder[i];
        EXPORT_VDB = false;
        Initialize();
        ExampleScene(true);
        preserve_object_list = true;
        SIMULATE = true;
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
        DONE_FRAME = true;
    }

    void Clear_Simulation_Data() {
        delete state;
        delete[] img;

        if (!preserve_object_list || /*SAMPLE_SCENE == 1 || */SAMPLE_SCENE == 2) {
            for (auto i : object_list) {
                //if (i.get_type() == "vdb" || i.get_type() == "vdbs")
                    //i.cudaFree();
                if (i.get_type() != "vdb")
                    i.free();
            }
            object_list.clear();
        }

        printf("\nCUDA: %s\n", cudaGetErrorString(cudaGetLastError()));

        cudaThreadExit();
    }

    void Simulation_Frame() {
        DONE_FRAME = false;
        unsigned int f = frame;
        std::cout << "\rFrame " << f + 1 << "  -  ";
        for (int st = 0; st < 1; st++) {
            simulate_fluid(*state, object_list, ACCURACY_STEPS, 
                false, f, Smoke_Dissolve, Ambient_Temperature,
                DIVERGE_RATE);
            state->step++;
        }

        render_fluid(
            img, img_d,
            state->density->readTarget(),
            state->temperature->readTarget(),
            vol_d, 1.0, Light, Camera, rotation,
            STEPS, Fire_Max_Temperature, Smoke_And_Fire);

        
        
        generateBitmapImage(img, img_d.x, img_d.y, ("output/R" + pad_number(f + 1) + ".bmp").c_str());
        

        if (EXPORT_VDB && frame >= EXPORT_START_FRAME) {
            GRID3D* arr = new GRID3D();
            GRID3D* arr_temp = new GRID3D();
            arr->set_pointer(state->density->readToGrid());
            arr_temp->set_pointer(state->temperature->readToGrid());
            std::string FOLDER = EXPORT_FOLDER;
            FOLDER = trim(FOLDER);
            export_openvdb(FOLDER,"frame." + std::to_string(f), vol_d, arr, arr_temp, false);
            //arr->free();
            //arr_temp->free();
            delete arr;
            delete arr_temp;
            if (frame >= EXPORT_END_FRAME)
                EXPORT_VDB = false;
        }
        frame++;
        DONE_FRAME = true;
    }
    std::thread* spawn() {
        if (SIMULATE)
            return new std::thread([this] { this->Simulation_Frame(); });
        else
            return new std::thread();
    }
    unsigned char* loadImgData() {
        return img;
    }
};
Solver solver;
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
        //adding emitters
        object_list.push_back(OBJECT("emitter", 18.0f, 50, 0.9, 5, 0.9, make_float3(vol_d.x * 0.25, 10.0, 200.0)));
        object_list.push_back(OBJECT("emitter", 18.0f, 50, 0.6, 5, 0.9, make_float3(vol_d.x * 0.5, 10.0, 200.0)));
        object_list.push_back(OBJECT("emitter", 18.0f, 50, 0.3, 5, 0.9, make_float3(vol_d.x * 0.75, 10.0, 200.0)));
        object_list.push_back(OBJECT("smoke", 10, 50, 0.9, 50, 1.0, make_float3(vol_d.x * 0.5, 30.0, 200.0)));
    }
    else if (EXAMPLE__ == 2) {
        object_list.push_back(OBJECT("emitter", 18.0f, 50, 0.6, 5, 0.9, make_float3(/*left-right*/vol_d.x * 0.5, 10.0, /*front-back*/200)));
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