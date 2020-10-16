#ifndef __IO
#define __IO
#include "Libraries.h"
#include <vector>
//#include "OpenVDB/tinyvdbio.h"



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



void create_grid_old(openvdb::FloatGrid& grid_dst, GRID3D* grid_src, const openvdb::Vec3f& c) {
    using ValueT = typename openvdb::FloatGrid::ValueType;
    const ValueT outside = grid_dst.background();
    int padding = int(openvdb::math::RoundUp(openvdb::math::Abs(outside)));

    //get bounding box
    int3 dim = grid_src->get_resolution();

    // Get a voxel accessor.

    float* grid_src_arr = grid_src->get_grid();
//#define MULTI
#ifndef MULTI
    typename openvdb::FloatGrid::Accessor accessor = grid_dst.getAccessor();
    for (int x = 0; x < dim.x; x++) {
#else
    int THREADS = 24;//8
    std::vector<typename openvdb::FloatGrid::Accessor> accessors;
    //std::vector<typename openvdb::tree::ValueAccessorRW<openvdb::FloatTree::ValueType,true>> accessors;
    for (int i = 0; i < THREADS; i++)
        //accessors.push_back(openvdb::tree::ValueAccessorRW<openvdb::FloatTree::ValueType, true>(grid_dst.getAccessor()));
        accessors.push_back(grid_dst.getAccessor());
    int sizee = ceil((double)dim.x / (double)THREADS);
    concurrency::parallel_for(0, THREADS, [&](int n) {
        int end = (n * sizee) + (sizee - 1);
        if (end > dim.x) {
            end = dim.x;
        }
        for (int x = n * sizee; x < end; x++)
            for (int y = 0; y < dim.y; y++) {
                for (int z = 0; z < dim.z; z++) {
                    //accessor.setValue(openvdb::Coord(x, y, z), grid_src(x, y, z));
                    accessors[n].setValue(openvdb::Coord(x, y, z), grid_src_arr[z * dim.y * dim.x + y * dim.x + x]);
                }
            }
    });
    for (int i = 0; i < THREADS; i++)
        accessors[0].clear();
    accessors.clear();
#endif
#ifdef MULTI
#else
        for (int y = 0; y < dim.y; y++) {
            for (int z = 0; z < dim.z; z++) {
                //accessor.setValue(openvdb::Coord(x, y, z), grid_src(x, y, z));
                accessor.setValue(openvdb::Coord(x, y, z), grid_src_arr[z * dim.y * dim.x + y * dim.x + x]);
            }
        }
    }
#endif
    delete[] grid_src_arr;
    auto tree = grid_dst.tree();
    tree.clearAllAccessors();
    openvdb::tools::signedFloodFill(tree);
}

void create_grid(openvdb::FloatGrid& grid_dst, GRID3D* grid_src, const openvdb::Vec3f& c) {
    using ValueT = typename openvdb::FloatGrid::ValueType;
    const ValueT outside = grid_dst.background();
    int padding = int(openvdb::math::RoundUp(openvdb::math::Abs(outside)));

    //get bounding box
    int3 dim = grid_src->get_resolution();

    // Get a voxel accessor.

    float* grid_src_arr = grid_src->get_grid();
    //#define MULTI
#ifndef MULTI
    typename openvdb::FloatGrid::Accessor accessor = grid_dst.getAccessor();
    for (int x = 0; x < dim.x; x++) {
#else
    int THREADS = 24;//8
    std::vector<typename openvdb::FloatGrid::Accessor> accessors;
    //std::vector<typename openvdb::tree::ValueAccessorRW<openvdb::FloatTree::ValueType,true>> accessors;
    for (int i = 0; i < THREADS; i++)
        //accessors.push_back(openvdb::tree::ValueAccessorRW<openvdb::FloatTree::ValueType, true>(grid_dst.getAccessor()));
        accessors.push_back(grid_dst.getAccessor());
    int sizee = ceil((double)dim.x / (double)THREADS);
    concurrency::parallel_for(0, THREADS, [&](int n) {
        int end = (n * sizee) + (sizee - 1);
        if (end > dim.x) {
            end = dim.x;
        }
        for (int x = n * sizee; x < end; x++)
            for (int y = 0; y < dim.y; y++) {
                for (int z = 0; z < dim.z; z++) {
                    //accessor.setValue(openvdb::Coord(x, y, z), grid_src(x, y, z));
                    accessors[n].setValue(openvdb::Coord(x, y, z), grid_src_arr[z * dim.y * dim.x + y * dim.x + x]);
                }
            }
        });
    for (int i = 0; i < THREADS; i++)
        accessors[0].clear();
    accessors.clear();
#endif
#ifdef MULTI
#else
    for (int y = 0; y < dim.y; y++) {
        for (int z = 0; z < dim.z; z++) {
            //accessor.setValue(openvdb::Coord(x, y, z), grid_src(x, y, z));
            accessor.setValue(openvdb::Coord(x, y, z), grid_src_arr[z * dim.y * dim.x + y * dim.x + x]);
        }
    }
    }
#endif
delete[] grid_src_arr;
grid_src->free();
auto tree = grid_dst.tree();
tree.clearAllAccessors();
openvdb::tools::signedFloodFill(tree);
}

int export_openvdb(std::string filename, int3 domain_resolution, GRID3D* grid_dst, GRID3D* grid_temp, bool DEBUG = false) {
    filename = "output/cache/" + filename + ".vdb";
    
    std::cout << " || Saving OpenVDB:  ";
    clock_t startTime = clock();
    

    std::vector < GRID3D* > grids_src;
    std::vector <openvdb::FloatGrid::Ptr> grids_dst;
    openvdb::GridPtrVecPtr grids(new openvdb::GridPtrVec);
    /////////////////////////////////////
    openvdb::FloatGrid::Ptr grid =
        openvdb::FloatGrid::create(/*background value=*/0.0);
    openvdb::FloatGrid::Ptr grid_temp2 =
        openvdb::FloatGrid::create(/*background value=*/0.0);

    ////////////////////////////////////////////////////////
    grid->setName("density");
    grid_temp2->setName("temperature");
    ////////////////////////////////////////////////////////
    grids_src.push_back(grid_dst);
    grids_src.push_back(grid_temp);
    grids_dst.push_back(grid);
    grids_dst.push_back(grid_temp2);
    ////////////////////////////////////////////////////////
    
    //for (int i = 0; i < grids_src.size(); i++) {
    std::mutex mtx;
    concurrency::parallel_for(0, 2, [&](int i) {
        create_grid(*grids_dst[i], grids_src[i], /*center=*/openvdb::Vec3f(0, 0, 0));
        //grids->at(i)->saveFloatAsHalf();
        grids_dst[i]->saveFloatAsHalf();

        std::lock_guard<std::mutex> lock(mtx);
        grids->push_back(grids_dst[i]);
        });
    


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



    create_grid(*grid, grid_dst, /*center=*/openvdb::Vec3f(0, 0, 0));

    grid_dst->free();
    // Associate some metadata with the grid.
    //grid->insertMeta("radius", openvdb::FloatMetadata(50.0));
    ////////////////////////////////////////////////////////
    openvdb::FloatGrid::Ptr grid_temp2 =
        openvdb::FloatGrid::create(/*background value=*/0.0);
    //clock_t startTime = clock();

    create_grid(*grid_temp2, grid_temp, /*center=*/openvdb::Vec3f(0, 0, 0));

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











GRID3D load_vdb(std::string filename, int3 domain_resolution, bool DEBUG = false) {
    filename = "input//" + filename + ".nvdb";

    GRID3D outputt(domain_resolution.x, domain_resolution.y, domain_resolution.z);
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
        concurrency::parallel_for(0, THREADS, [&](int n) {
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
            });
            
    }
    catch (const std::exception& e) {
        std::cout << "[Import] An exception occurred: \"" << e.what() << "\"" << std::endl;
    }
    return outputt;
}


using BufferT = nanovdb::CudaDeviceBuffer;


int export_vdb(std::string filename, int3 domain_resolution) {
    filename = "input//" + filename + ".nvdb";
    try {
        auto handle = nanovdb::createFogVolumeSphere<float, BufferT>(100.0f, nanovdb::Vec3R(100, 40, 100), 1.0f, 3.0f, nanovdb::Vec3R(0), "sphere");
       
        nanovdb::io::writeGrid(filename, handle, nanovdb::io::Codec::BLOSC);
    }
    catch (const std::exception& e) {
        std::cout << "[Export] An exception occurred: \"" << e.what() << "\"" << std::endl;
    }
    return 0;
}






extern void runNanoVDB(nanovdb::GridHandle<BufferT>& handle, int numIterations, int width, int height, BufferT& imageBuffer);
void renderImage(std::string filename, int ac) {
    filename = "input//" + filename + ".nvdb";
    try {
        nanovdb::GridHandle<BufferT> handle;
        if (ac > 1) {
            handle = nanovdb::io::readGrid<BufferT>(filename);
            std::cout << "Loaded NanoVDB grid[" << handle.gridMetaData()->gridName() << "]...\n";
        }
        else {
            handle = nanovdb::createFogVolumeSphere<float, BufferT>(100.0f, nanovdb::Vec3R(100, 100, 100), 1.0f, 3.0f, nanovdb::Vec3R(0), "sphere");
        }

        if (handle.gridMetaData()->isFogVolume() == false) {
            throw std::runtime_error("Grid must be a fog volume");
        }

        const int numIterations = 50;

        const int width = 1024;
        const int height = 1024;
        BufferT   imageBuffer;
        imageBuffer.init(width * height * sizeof(float));

        runNanoVDB(handle, numIterations, width, height, imageBuffer);
    }
    catch (const std::exception& e) {
        std::cerr << "An exception occurred: \"" << e.what() << "\"" << std::endl;
    }
}

#endif
