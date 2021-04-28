#ifndef __IO
#define __IO
#include "Libraries.h"
#include <string>
#include <iostream>
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#include <filesystem>
#include <experimental/filesystem>
#include <nanovdb/util/Primitives.h>

#include <vector>

#include <tbb/tbb.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/atomic.h>

#include "BPPointCloud.h"

//#include "OpenVDB/tinyvdbio.h"
#include <openvdb/tools/Dense.h>
#include <openvdb/tools/DenseSparseTools.h>
#include <openvdb/Types.h>
#include <cuda_runtime.h>








#include <stdio.h>
const int BYTES_PER_PIXEL = 3; /// red, green, & blue
const int FILE_HEADER_SIZE = 14;
const int INFO_HEADER_SIZE = 40;
void generateBitmapImage(unsigned char* image, int height, int width, char* imageFileName);
unsigned char* createBitmapFileHeader(int height, int stride);
unsigned char* createBitmapInfoHeader(int height, int width);


void generateBitmapImage(unsigned char* image, int height, int width, const char* imageFileName)
{

    int widthInBytes = width * BYTES_PER_PIXEL;

    unsigned char padding[3] = { 0, 0, 0 };
    int paddingSize = (4 - (widthInBytes) % 4) % 4;

    int stride = (widthInBytes)+paddingSize;

    std::experimental::filesystem::create_directory("./output");

    FILE* imageFile = fopen(imageFileName, "wb");

    
    unsigned char* fileHeader = createBitmapFileHeader(height, stride);
    fwrite(fileHeader, 1, FILE_HEADER_SIZE, imageFile);

    unsigned char* infoHeader = createBitmapInfoHeader(height, width);
    fwrite(infoHeader, 1, INFO_HEADER_SIZE, imageFile);
    
    
    //////
    int j = height - 1;
    for (int i = 0; i < height; i++) {
        fwrite(image + (j * widthInBytes), BYTES_PER_PIXEL, width, imageFile);
        fwrite(padding, 1, paddingSize, imageFile);
        j--;
    }
    
    fclose(imageFile);
}

unsigned char* createBitmapFileHeader(int height, int stride)
{
    int fileSize = FILE_HEADER_SIZE + INFO_HEADER_SIZE + (stride * height);

    static unsigned char fileHeader[] = {
        0,0,     /// signature
        0,0,0,0, /// image file size in bytes
        0,0,0,0, /// reserved
        0,0,0,0, /// start of pixel array
    };

    fileHeader[0] = (unsigned char)('B');
    fileHeader[1] = (unsigned char)('M');
    fileHeader[2] = (unsigned char)(fileSize);
    fileHeader[3] = (unsigned char)(fileSize >> 8);
    fileHeader[4] = (unsigned char)(fileSize >> 16);
    fileHeader[5] = (unsigned char)(fileSize >> 24);
    fileHeader[10] = (unsigned char)(FILE_HEADER_SIZE + INFO_HEADER_SIZE);

    return fileHeader;
}

unsigned char* createBitmapInfoHeader(int height, int width)
{
    
    static unsigned char infoHeader[] = {
        0,0,0,0, /// header size
        0,0,0,0, /// image width
        0,0,0,0, /// image height
        0,0,     /// number of color planes
        0,0,     /// bits per pixel
        0,0,0,0, /// compression
        0,0,0,0, /// image size
        0,0,0,0, /// horizontal resolution
        0,0,0,0, /// vertical resolution
        0,0,0,0, /// colors in color table
        0,0,0,0, /// important color count
    };
    infoHeader[0] = (unsigned char)(INFO_HEADER_SIZE);
    infoHeader[4] = (unsigned char)(width);
    infoHeader[5] = (unsigned char)(width >> 8);
    infoHeader[6] = (unsigned char)(width >> 16);
    infoHeader[7] = (unsigned char)(width >> 24);
    infoHeader[8] = (unsigned char)(height);
    infoHeader[9] = (unsigned char)(height >> 8);
    infoHeader[10] = (unsigned char)(height >> 16);
    infoHeader[11] = (unsigned char)(height >> 24);
    infoHeader[12] = (unsigned char)(1);
    infoHeader[14] = (unsigned char)(BYTES_PER_PIXEL * 8);

    return infoHeader;
}














std::mutex mtx;

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














#define THREADED_SAVE

int copy(openvdb::FloatGrid::Accessor& accessor,
    float* grid_src_arr, int3 dim, int sizee, int n) {
    Sleep(n);
    int end = (n * sizee) + (sizee - 1);
    if (end > dim.x) {
        end = dim.x;
    }
    for (int x = n * sizee; x < end; x++)
        for (int y = 0; y < dim.y; y++) {
            for (int z = 0; z < dim.z; z++) {
                accessor.setValue(openvdb::Coord(x, y, z), grid_src_arr[z * dim.y * dim.x + y * dim.x + x]);
            }
        }
    return 1;
}

void create_grid_mt_old(openvdb::FloatGrid& grid_dst, GRID3D* grid_src, const openvdb::Vec3f& c) {
    using ValueT = typename openvdb::FloatGrid::ValueType;
    const ValueT outside = grid_dst.background();
    int padding = int(openvdb::math::RoundUp(openvdb::math::Abs(outside)));

    //get bounding box
    int3 dim = grid_src->get_resolution();

    grid_dst.tree().clearAllAccessors();



    // Get a voxel accessor.

    float* grid_src_arr = grid_src->get_grid();

    int THREADS = 12;//8


    //std::vector<typename openvdb::tree::ValueAccessorRW<openvdb::FloatTree::ValueType,true>> accessors;
    //for (int i = 0; i < THREADS; i++)
        //accessors.push_back(openvdb::tree::ValueAccessorRW<openvdb::FloatTree::ValueType, true>(grid_dst.getAccessor()));

    int sizee = ceil((double)dim.x / (double)THREADS);

    //////////////////////
    std::vector<openvdb::FloatGrid::Accessor> accessors;
    for (int i = 0; i < THREADS; i++) {
        accessors.push_back(grid_dst.getAccessor());
    }
    //////////////////////
    //tworzenie w¹tków
    std::vector<std::thread> pool;

    for (int i = 0; i < THREADS; i++) {
        std::thread T(copy, accessors[i], grid_src_arr, dim, sizee, i);
        pool.push_back(move(T));
    }
    for (auto& T : pool)
        if (T.joinable())
            T.join();
    pool.clear();



    //grid_dst.tree().combine(bGrid->tree(), Local::diff);


    /*
    */
    for (int i = 0; i < THREADS; i++)
        accessors[0].clear();
    accessors.clear();

    grid_dst.tree().clearAllAccessors();
    delete[] grid_src_arr;
    grid_src->free();

    openvdb::tools::signedFloodFill(grid_dst.tree());
    grid_dst.setTransform(
        openvdb::math::Transform::createLinearTransform(/*voxel size=*/0.1));
}




// value iterator points.
struct Local {
    float* M;
    Local(float* mat) : M(mat) {}
    inline void operator()(const openvdb::FloatGrid::ValueAllIter& iter) const {
        //iter.setValue(M[iter]);
        iter.setValue(2.5);
    }
    static inline void add(const float& a, const float& b, float& result) {
        result = a + b;
    }
    static inline void maks(const float& a, const float& b, float& result) {
        result = max(a,b);
    }

    static inline void noise(const openvdb::FloatGrid::ValueAllIter& iter) {
        float noise = float(rand() % 1000) / 1000.0;
        float influence = 0.9f;
        noise *= influence;
        iter.setValue(*iter * noise);
    }
};






openvdb::FloatGrid::Ptr create_grid_mt(openvdb::FloatGrid::Ptr& grid_dst, GRID3D* grid_src, const openvdb::Vec3f& c, int INDEX, bool DEBUG = false) {
    using ValueT = typename openvdb::FloatGrid::ValueType;
    const ValueT outside = grid_dst->background();
    const ValueT inside = -outside;
    int padding = int(openvdb::math::RoundUp(openvdb::math::Abs(outside)));

    //get bounding box
    int3 dim = grid_src->get_resolution();

    grid_dst->tree().clearAllAccessors();

    if (DEBUG)
        std::cout << "Initialize...\n";

    const auto processor_count = std::thread::hardware_concurrency();

    int THREADS = processor_count / 4; //4

    // Get a voxel accessor.
    float* grid_src_arr = nullptr;
    if (INDEX == 0 || INDEX == 2)
        grid_src_arr = grid_src->get_grid();
    else
        grid_src_arr = grid_src->get_grid_temp();






    openvdb::math::Coord dim2(grid_src->resolution.x,grid_src->resolution.y, grid_src->resolution.z);
    openvdb::tools::Dense<float> dense(dim2);

    float *data = dense.data();
    //std::cout << "value at origin: " << data[0] << std::endl;


    ////////////////////// current ~720
    
    tbb::task_scheduler_init initt(THREADS);

    tbb::parallel_for(tbb::blocked_range<int>(0, dim.x), [&](tbb::blocked_range<int> tbbx) {
        for (int x = tbbx.begin(); x < tbbx.end(); x++) {

            for (int y = 0; y < dim.y; y++) {
                for (int z = 0; z < dim.z; z++) {
                    float val = grid_src_arr[z * dim.y * dim.x + y * dim.x + x];
                    if (val < 0.025) continue; //beta
                    data[z * dim.y * dim.x + y * dim.x + x] = val;
                }
            }
        }
        });





    openvdb::tools::copyFromDense<openvdb::tools::Dense<float>, openvdb::FloatGrid>(dense, *grid_dst, 0.001);

    grid_dst->pruneGrid(0);

    float voxel_size = 0.1;
    auto transform = openvdb::math::Transform::createLinearTransform(/*voxel size=*/voxel_size); //Skala œwiatowa
    const openvdb::math::Vec3d offset(float(-dim.x) / 2., 0, float(-dim.z) / 2.);
    transform->preTranslate(offset); //center the grid
    transform->postRotate(1.571, openvdb::math::X_AXIS); // poprawa rotacji dla blendera
    grid_dst->setTransform(
        transform
        ); 

    //grid_dst = dest;
    
    return grid_dst;

}































openvdb::FloatGrid::Ptr create_grid_gpu(openvdb::FloatGrid::Ptr& grid_dst, float* grid_src,GRID3D* grid_info, const openvdb::Vec3f& c, int INDEX, bool DEBUG = false) {
    using ValueT = typename openvdb::FloatGrid::ValueType;
    const ValueT outside = grid_dst->background();
    const ValueT inside = -outside;
    int padding = int(openvdb::math::RoundUp(openvdb::math::Abs(outside)));

    //get bounding box
    int3 dim = grid_info->get_resolution();

    grid_dst->tree().clearAllAccessors();

    if (DEBUG)
        std::cout << "Initialize...\n";

    
    openvdb::math::Coord dim2(grid_info->resolution.z, grid_info->resolution.y, grid_info->resolution.x);
    openvdb::tools::Dense<float> dense(dim2);

    float* data = dense.data();

    //copying
    cudaMemcpyAsync(data, grid_src, grid_info->size() * sizeof(float), cudaMemcpyDeviceToHost);


    openvdb::tools::copyFromDense<openvdb::tools::Dense<float>, openvdb::FloatGrid>(dense, *grid_dst, 0.025);

    grid_dst->pruneGrid(0);

    float voxel_size = 0.1;
    auto transform = openvdb::math::Transform::createLinearTransform(/*voxel size=*/voxel_size); //Skala œwiatowa
    const openvdb::math::Vec3d offset(float(-dim.x) / 2., 0, float(-dim.z) / 2.);
    transform->preTranslate(offset); //center the grid
    transform->postRotate(1.571, openvdb::math::X_AXIS); // poprawa rotacji dla blendera
    grid_dst->setTransform(
        transform
    );


    return grid_dst;

}


int export_openvdb_experimental(std::string folder, std::string filename, int3 domain_resolution,
    GRID3D* grid_info, float* density, float* temperature, float* flame, bool DEBUG = false) {
    filename = folder + filename + ".vdb";

    std::cout << "|| Saving OpenVDB: ";
    clock_t startTime = clock();

    std::cout << "" << filename;//<< std::endl;

    std::vector < float* > grids_src;
    std::vector <openvdb::FloatGrid::Ptr> grids_dst;
    openvdb::GridPtrVecPtr grids(new openvdb::GridPtrVec);
    /////////////////////////////////////
    openvdb::FloatGrid::Ptr grid_density =
        openvdb::FloatGrid::create(/*background value=*/0.0);
    openvdb::FloatGrid::Ptr grid_temperature =
        openvdb::FloatGrid::create(/*background value=*/0.0);
    openvdb::FloatGrid::Ptr grid_flame =
        openvdb::FloatGrid::create(/*background value=*/0.0);

    if (DEBUG)
        std::cout << "Grids prepared" << std::endl;
    ////////////////////////////////////////////////////////
    grid_density->setName("density");
    grid_temperature->setName("temperature");
    grid_flame->setName("flame");
    ////////////////////////////////////////////////////////



    grids_src.push_back(density);
    grids_src.push_back(temperature);
    grids_src.push_back(flame);
    grids_dst.push_back(grid_density);
    grids_dst.push_back(grid_temperature);
    grids_dst.push_back(grid_flame);
    ////////////////////////////////////////////////////////

    if (DEBUG)
        std::cout << "Starting threads" << std::endl;

    std::mutex mtx1;
    tbb::parallel_for(0, 3, [&](int i) {
        //for (int i = 0; i < 3; i++){
#ifndef THREADED_SAVE
        create_grid_sthr(*grids_dst[i], grids_src[i], /*center=*/openvdb::Vec3f(0, 0, 0), DEBUG);
        //create_grid_mt(*grids_dst[i], grids_src[i], /*center=*/openvdb::Vec3f(0, 0, 0));
#else
        auto upgrid = create_grid_gpu(grids_dst[i], grids_src[i], grid_info, /*center=*/openvdb::Vec3f(0, 0, 0), i, DEBUG);
        //auto upgrid = create_grid_gpu(grids_dst[i], grids_src[i], /*center=*/openvdb::Vec3f(0, 0, 0), i, DEBUG);


#endif

        //grids_dst[i]->saveFloatAsHalf();


        upgrid->saveFloatAsHalf();
        upgrid->pruneGrid(0);

        //sparse it up
        //0 -> lossles
        //more is smaller
        //grids_dst[i]->pruneGrid(0); //beta 0.1



        std::lock_guard<std::mutex> lock(mtx1);
        //grids->push_back(grids_dst[i]);
        grids->push_back(upgrid);
        }
    );

    if (DEBUG)
        std::cout << "Grids copied" << std::endl;

    //std::cout << (clock() - startTime);
    //startTime = clock();

    ////////////////////////////////////////////////////////




    //kolejnoœæ
    openvdb::io::File file(filename);
    file.setCompression(openvdb::OPENVDB_FILE_VERSION_BLOSC_COMPRESSION //     9.6 ->  600
                        //openvdb::OPENVDB_FILE_VERSION_BOOL_LEAF_OPTIMIZATION //9 -> 1400
    );
    file.write({ grid_density, grid_temperature, grid_flame });
    file.close();
    /*
    */
    /*
    std::ofstream ofile(filename, std::ios_base::binary);
    openvdb::io::Stream(ofile).write(*grids);
    */
    if (DEBUG)
        std::cout << "Grids saved" << std::endl;

    grid_info->free();
    grids_dst.clear();
    grids_src.clear();
    grids->clear();
    std::cout << " ; " << (clock() - startTime) << "     \n";

    //    grid->clear();

    return 1;
}













































void create_grid_sthr(openvdb::FloatGrid& grid_dst, GRID3D* grid_src, const openvdb::Vec3f& c, int INDEX, bool DEBUG = false) {


    using ValueT = typename openvdb::FloatGrid::ValueType;
    const ValueT outside = grid_dst.background();
    int padding = int(openvdb::math::RoundUp(openvdb::math::Abs(outside)));

    //get bounding box
    int3 dim = grid_src->get_resolution();

    if (DEBUG)
        std::cout << dim.x << ";" << dim.y << ";" << dim.z << std::endl;

    // Get a voxel accessor.
    float* grid_src_arr = nullptr;
    if (INDEX == 0 || INDEX == 2)
        grid_src_arr = grid_src->get_grid();
    else
        grid_src_arr = grid_src->get_grid_temp();
    
    if (DEBUG)
        std::cout << "Copying..." << std::endl;

    typename openvdb::FloatGrid::Accessor accessor = grid_dst.getAccessor();
    for (int x = 0; x < dim.x; x++) {
        for (int y = 0; y < dim.y; y++) {
            for (int z = 0; z < dim.z; z++) {
                if (grid_src_arr[z * dim.y * dim.x + y * dim.x + x] < 0.025) continue; //beta
                accessor.setValue(openvdb::Coord(x, y, z), grid_src_arr[z * dim.y * dim.x + y * dim.x + x]);
            }
        }
    }
    if (DEBUG)
        std::cout << "Clearing\n";
    //delete[] grid_src_arr;  //to poprzednio by³o
    //grid_src->free();
    if (DEBUG)
        std::cout << "Cleared\n";
    auto tree = grid_dst.tree();
    tree.clearAllAccessors();
    if (DEBUG)
        std::cout << "Transforming\n";
    openvdb::tools::signedFloodFill(tree);



    float voxel_size = 0.1;
    auto transform = openvdb::math::Transform::createLinearTransform(/*voxel size=*/voxel_size); //Skala œwiatowa
    const openvdb::math::Vec3d offset(float(-dim.x) / 2., 0, float(-dim.z) / 2.);
    transform->preTranslate(offset); //center the grid
    transform->postRotate(1.571, openvdb::math::X_AXIS); // poprawa rotacji dla blendera
    grid_dst.setTransform(
        transform
    );
}







int export_openvdb(std::string folder,std::string filename, int3 domain_resolution, 
                    GRID3D* grid_dst,GRID3D* grid_temperature, bool DEBUG = false) {
    filename = folder + filename + ".vdb";
    
    std::cout << "|| Saving OpenVDB: ";
    clock_t startTime = clock();
    
    std::cout << "" << filename ;//<< std::endl;

    std::vector < GRID3D* > grids_src;
    std::vector <openvdb::FloatGrid::Ptr> grids_dst;
    openvdb::GridPtrVecPtr grids(new openvdb::GridPtrVec);
    /////////////////////////////////////
    openvdb::FloatGrid::Ptr grid =
        openvdb::FloatGrid::create(/*background value=*/0.0);
    openvdb::FloatGrid::Ptr grid_temp2 =
        openvdb::FloatGrid::create(/*background value=*/0.0);
    openvdb::FloatGrid::Ptr grid_temp3 =
        openvdb::FloatGrid::create(/*background value=*/0.0);

    if (DEBUG)
        std::cout << "Grids prepared" << std::endl;
    ////////////////////////////////////////////////////////
    grid->setName("density");
    grid_temp2->setName("flame");
    grid_temp3->setName("temperature");
    ////////////////////////////////////////////////////////
    GRID3D* grid_temp = new GRID3D(1);
    //grid_temp->set_pointer(grid_dst);
    //grid_temp = grid_dst;



    grids_src.push_back(grid_dst);
    grids_src.push_back(grid_dst);
    grids_src.push_back(grid_temperature);
    grids_dst.push_back(grid);
    grids_dst.push_back(grid_temp2);
    grids_dst.push_back(grid_temp3);
    ////////////////////////////////////////////////////////

    if (DEBUG)
        std::cout << "Starting threads" << std::endl;
    
    //for (int i = 0; i < grids_src.size(); i++) {
    std::mutex mtx1;
    tbb::parallel_for(0, 3, [&](int i) {
    //for (int i = 0; i < 3; i++){
#ifndef THREADED_SAVE
        create_grid_sthr(*grids_dst[i], grids_src[i], /*center=*/openvdb::Vec3f(0, 0, 0),DEBUG);
        //create_grid_mt(*grids_dst[i], grids_src[i], /*center=*/openvdb::Vec3f(0, 0, 0));
#else
        auto upgrid = create_grid_mt(grids_dst[i], grids_src[i], /*center=*/openvdb::Vec3f(0, 0, 0),i, DEBUG);
        //auto upgrid = create_grid_gpu(grids_dst[i], grids_src[i], /*center=*/openvdb::Vec3f(0, 0, 0), i, DEBUG);


#endif
        
        //grids_dst[i]->saveFloatAsHalf();

        
        upgrid->saveFloatAsHalf();
        upgrid->pruneGrid(0);

        //sparse it up
        //0 -> lossles
        //more is smaller
        //grids_dst[i]->pruneGrid(0); //beta 0.1



        std::lock_guard<std::mutex> lock(mtx1);
        //grids->push_back(grids_dst[i]);
        grids->push_back(upgrid);
        }
    );
    
    if (DEBUG)
        std::cout << "Grids copied" << std::endl;

    //std::cout << (clock() - startTime);
    //startTime = clock();

    ////////////////////////////////////////////////////////
    



    //kolejnoœæ
    openvdb::io::File file(filename);
    file.setCompression(openvdb::OPENVDB_FILE_VERSION_BLOSC_COMPRESSION //     9.6 ->  600
                        //openvdb::OPENVDB_FILE_VERSION_BOOL_LEAF_OPTIMIZATION //9 -> 1400
                        );
    file.write({ grid, grid_temp2, grid_temp3 });
    file.close();
    /*
    */
    /*
    std::ofstream ofile(filename, std::ios_base::binary);
    openvdb::io::Stream(ofile).write(*grids);
    */
    if (DEBUG)
        std::cout << "Grids saved" << std::endl;
    

    grid_temp->free();
    grid_dst->free();

    grids_dst.clear();
    grids_src.clear();
    grids->clear();
    grid->clearGridClass();
    grid->clearMetadata();
    grid->clear();
    grid_temp2->clear();
    grid_temp2->clearGridClass();
    grid_temp2->clearMetadata();
    grid_temp3->clear();
    grid_temp3->clearGridClass();
    grid_temp3->clearMetadata();
    std::cout << " ; "<< (clock() - startTime) << "     \n";

//    grid->clear();

    return 1;
}





int load_vdb(std::string filename, int3 domain_resolution,GRID3D& outputt , bool DEBUG = false) {
    filename = "input//" + filename + ".nvdb";

    //nanovdb::FloatGrid* output = nullptr;
    try {
        // returns a GridHandle using CUDA for memory management.
        auto handle = nanovdb::io::readGrids(filename);

        auto* grid = handle[0].grid<float>(); // get a (raw) pointer to a NanoVDB grid of value type float on the CPU
        auto object = grid->getAccessor();


        std::cout << domain_resolution.x * domain_resolution.y * domain_resolution.z << std::endl;
        
        std::cout << grid->activeVoxelCount() << std::endl;
        auto dims = grid->indexBBox().dim();
        std::cout << "X: " << dims.x() << " Y: " << dims.y() << " Z: " << dims.z() << std::endl;
        int3 dims3 = make_int3(dims.x(), dims.y(), dims.x());
        
        float mini, maxi;
        grid->tree().extrema(mini, maxi);
        std::cout << "Min: " << mini << " Max: " << maxi << std::endl;
        //for (int x = 0; x < domain_resolution.x; x++)
        
        int THREADS = 8;//8
        int sizee = ceil((double)dims3.x / (double)THREADS);
        for (int n = 0; n < THREADS; n++){
            int end = (n * sizee) + (sizee - 1);
            if (end > dims3.x) {
                end = dims3.x;
            }
            for (int x = n * sizee; x < end; x++)
                for (int y = 0; y < dims3.y; y++)
                    for (int z = 0; z < dims3.z; z++) {
                        if (DEBUG)
                            std::cout << "\r" << object.getValue(nanovdb::Coord(x, y, z)) << " , ";
                        outputt.set(x, y, z, object.getValue(nanovdb::Coord(x, y, z)));
                        outputt.set_temp(x, y, z, object.getValue(nanovdb::Coord(x, y, z)));
                    }
            }
        //handle.clear();
    }
    catch (const std::exception& e) {
        std::cout << "[Import] An exception occurred: \"" << e.what() << "\"" << std::endl;
    }
    return true;
}


using BufferT = nanovdb::CudaDeviceBuffer;


int export_vdb(std::string filename, int3 domain_resolution) {
    filename = "input//" + filename + ".nvdb";
    try {
        float size = min(min(domain_resolution.x, domain_resolution.y),domain_resolution.z);
        size /= 3;
        auto handle = nanovdb::createFogVolumeSphere<float, BufferT>(
            size, nanovdb::Vec3R(domain_resolution.x/2.0, 0.0, domain_resolution.z/2), 1.0f, 3.0f, nanovdb::Vec3R(0), "sphere");
       
        nanovdb::io::writeGrid(filename, handle, nanovdb::io::Codec::BLOSC);
    }
    catch (const std::exception& e) {
        std::cout << "[Export] An exception occurred: \"" << e.what() << "\"" << std::endl;
    }
    return 0;
}







int save_scene_to_file(std::string filename, std::vector<std::string> lines) {
    //filename = "scenes\\" + filename + ".txt";
    
    std::ifstream myfile(filename);
    if (myfile.is_open()) {
        std::cout << "\nError!!!\n";
        std::cout << "File already exists\n";
        return 1;
    }
    else {
        myfile.close();

        std::fstream  savefile;
        savefile.open(filename, std::ios::out);

        if (savefile.is_open()) {
            for (int i = 0; i < lines.size(); i++) {
                savefile << lines[i] << "\n";
            }
        }
        else {
            std::cout << "\nError!!!\n";
            std::cout << "Cannot open file\n";
            return 1;
        }
        savefile.close();
    }
    return 0;
}

std::vector<std::string> load_scene_from_file(std::string filename) {
    //filename = "scenes\\" + filename + ".txt";

    std::vector<std::string> lines;

    std::string line;
    std::ifstream myfile(filename);
    if (myfile.is_open())
    {
        while (std::getline(myfile, line))
        {
            lines.push_back(line);
        }
        myfile.close();
    }
    else {
        std::cout << "Error\n";
        std::cout << "Cannot open: " << filename << std::endl;
        lines.push_back("ERROR");
        return lines;
    }
    return lines;
}
#endif
