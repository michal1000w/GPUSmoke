#ifndef __IO
#define __IO
#include "Libraries.h"
#include <vector>


//#include "OpenVDB/tinyvdbio.h"

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
    delete[] infoHeader;
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
    /*
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
    */
    unsigned char* infoHeader = new unsigned char[17+36];
    for (int i = 0; i < 17 + 36; i++)
        infoHeader[i] = 0;
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



void create_grid_sthr(openvdb::FloatGrid& grid_dst, GRID3D* grid_src, const openvdb::Vec3f& c) {
    bool DEBUG = false;
    
    using ValueT = typename openvdb::FloatGrid::ValueType;
    const ValueT outside = grid_dst.background();
    int padding = int(openvdb::math::RoundUp(openvdb::math::Abs(outside)));

    //get bounding box
    int3 dim = grid_src->get_resolution();

    if (DEBUG)
    std::cout << dim.x << ";" << dim.y << ";" << dim.z << std::endl;

    // Get a voxel accessor.

    float* grid_src_arr = grid_src->get_grid();
    if (DEBUG)
    std::cout << "Copying..." << std::endl;

    typename openvdb::FloatGrid::Accessor accessor = grid_dst.getAccessor();
    for (int x = 0; x < dim.x; x++) {
        for (int y = 0; y < dim.y; y++) {
            for (int z = 0; z < dim.z; z++) {
                accessor.setValue(openvdb::Coord(x, y, z), grid_src_arr[z * dim.y * dim.x + y * dim.x + x]);
            }
        }
    }
    if (DEBUG)
    std::cout << "Clearing\n";
delete[] grid_src_arr;
//grid_src->free();
if (DEBUG)
std::cout << "Cleared\n";
auto tree = grid_dst.tree();
tree.clearAllAccessors();
if (DEBUG)
std::cout << "Transforming\n";
openvdb::tools::signedFloodFill(tree);
grid_dst.setTransform(
    openvdb::math::Transform::createLinearTransform(/*voxel size=*/0.1));
}















int copy(openvdb::FloatGrid::Accessor& accessor,
    float* grid_src_arr,int3 dim, int sizee, int n) {
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


struct Local {
    static inline void diff(const float& a, const float& b, float& result) {
        result = a + b;
    }
};

void create_grid_mt(openvdb::FloatGrid& grid_dst, GRID3D* grid_src, const openvdb::Vec3f& c) {
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
    std::vector<typename openvdb::FloatGrid::Accessor> accessors;
    for (int i = 0; i < THREADS; i++) {
        accessors.push_back(grid_dst.getAccessor());
    }
    //////////////////////
    //tworzenie w¹tków
    std::vector<std::thread> pool;

    /*
    for (int i = 0; i < THREADS; i++) {
        copy(accessors[i], grid_src_arr, dim, sizee, i);
    }
    */
    for (int i = 0; i < THREADS; i++) {
        std::thread T(copy,accessors[i],grid_src_arr,dim, sizee, i);
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

int export_openvdb(std::string folder,std::string filename, int3 domain_resolution, GRID3D* grid_dst, GRID3D* grid_temp, bool DEBUG = true) {
    filename = folder + filename + ".vdb";
    
    std::cout << " || Saving OpenVDB:  ";
    clock_t startTime = clock();
    
    std::cout << "\n" << filename << std::endl;

    std::vector < GRID3D* > grids_src;
    std::vector <openvdb::FloatGrid::Ptr> grids_dst;
    openvdb::GridPtrVecPtr grids(new openvdb::GridPtrVec);
    /////////////////////////////////////
    openvdb::FloatGrid::Ptr grid =
        openvdb::FloatGrid::create(/*background value=*/0.0);
    openvdb::FloatGrid::Ptr grid_temp2 =
        openvdb::FloatGrid::create(/*background value=*/0.0);

    if (DEBUG)
        std::cout << "Grids prepared" << std::endl;
    ////////////////////////////////////////////////////////
    grid->setName("density");
    grid_temp2->setName("temperature");
    ////////////////////////////////////////////////////////
    grids_src.push_back(grid_dst);
    grids_src.push_back(grid_temp);
    grids_dst.push_back(grid);
    grids_dst.push_back(grid_temp2);
    ////////////////////////////////////////////////////////

    if (DEBUG)
        std::cout << "Starting threads" << std::endl;
    
    //for (int i = 0; i < grids_src.size(); i++) {
    std::mutex mtx1;
    concurrency::parallel_for(0, 2, [&](int i) {
    //for (int i = 0; i < 2; i++){
        create_grid_sthr(*grids_dst[i], grids_src[i], /*center=*/openvdb::Vec3f(0, 0, 0));
        //grids->at(i)->saveFloatAsHalf();
        grids_dst[i]->saveFloatAsHalf();

        std::lock_guard<std::mutex> lock(mtx1);
        grids->push_back(grids_dst[i]);
        });
    
    if (DEBUG)
        std::cout << "Grids copied" << std::endl;

    std::cout << (clock() - startTime);
    startTime = clock();

    ////////////////////////////////////////////////////////
    
    /*
    openvdb::io::File file(filename);
    file.setCompression(openvdb::OPENVDB_FILE_VERSION_BLOSC_COMPRESSION);
    file.write({ grid, grid_temp2 });
    file.close();
    */
    
    std::ofstream ofile(filename, std::ios_base::binary);
    openvdb::io::Stream(ofile).write(*grids);
    if (DEBUG)
        std::cout << "Grids saved" << std::endl;
    
    grids_dst.clear();
    grids_src.clear();
    grids->clear();
    grid->clearGridClass();
    grid->clearMetadata();
    grid->clear();
    grid_temp2->clear();
    grid_temp2->clearGridClass();
    grid_temp2->clearMetadata();
    std::cout << " ; "<< (clock() - startTime) << "     ";

//    grid->clear();

    return 1;
}





int export_openvdb_old(std::string filename, int3 domain_resolution, GRID3D* grid_dst, GRID3D* grid_temp, bool DEBUG = false) {
    filename = "output/cache/" + filename + ".vdb";

    std::cout << " || Saving OpenVDB:  ";
    //clock_t startTime = clock();


    openvdb::GridPtrVecPtr grids(new openvdb::GridPtrVec);
    /////////////////////////////////////
    openvdb::FloatGrid::Ptr grid =
        openvdb::FloatGrid::create(/*background value=*/0.0);
    clock_t startTime = clock();



    create_grid_sthr(*grid, grid_dst, /*center=*/openvdb::Vec3f(0, 0, 0));

    grid_dst->free();
    // Associate some metadata with the grid.
    //grid->insertMeta("radius", openvdb::FloatMetadata(50.0));
    ////////////////////////////////////////////////////////
    openvdb::FloatGrid::Ptr grid_temp2 =
        openvdb::FloatGrid::create(/*background value=*/0.0);
    //clock_t startTime = clock();

    create_grid_sthr(*grid_temp2, grid_temp, /*center=*/openvdb::Vec3f(0, 0, 0));

    grid_temp->free();

    ////////////////////////////////////////////////////////
    //reduce size
    grid_temp2->saveFloatAsHalf();
    grid->saveFloatAsHalf();
    ////////////////////////////////////////////////////////
    grid->setName("density");
    grids->push_back(grid);
    grid_temp2->setName("temperature");
    grids->push_back(grid_temp2);
    ////////////////////////////////////////////////////////



    std::cout << (clock() - startTime);
    startTime = clock();

    ////////////////////////////////////////////////////////

    /*
    openvdb::io::File file(filename);
    file.setCompression(openvdb::OPENVDB_FILE_VERSION_BLOSC_COMPRESSION);
    file.write({ grid, grid_temp2 });
    file.close();
    */


    std::ofstream ofile(filename, std::ios_base::binary);
    openvdb::io::Stream(ofile).write(*grids);

    grids->clear();
    grid->clearGridClass();
    grid->clearMetadata();
    grid->clear();
    grid_temp2->clear();
    grid_temp2->clearGridClass();
    grid_temp2->clearMetadata();
    std::cout << " ; " << (clock() - startTime) << "     ";

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


#endif
