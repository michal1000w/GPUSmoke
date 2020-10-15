#ifndef __IO
#define __IO
#include "Libraries.h"
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



void create_grid(openvdb::FloatGrid& grid_dst, GRID3D& grid_src, const openvdb::Vec3f& c) {
    using ValueT = typename openvdb::FloatGrid::ValueType;
    const ValueT outside = grid_dst.background();
    int padding = int(openvdb::math::RoundUp(openvdb::math::Abs(outside)));

    //get bounding box
    int3 dim = grid_src.get_resolution();

    // Get a voxel accessor.
    typename openvdb::FloatGrid::Accessor accessor = grid_dst.getAccessor();

    float* grid_src_arr = grid_src.get_grid();
//#define MULTI
#ifndef MULTI
    for (int x = 0; x < dim.x; x++) {
#else
    int THREADS = 2;//8
    int sizee = ceil((double)dim.x / (double)THREADS);
    concurrency::parallel_for(0, THREADS, [&](int n) {
        int end = (n * sizee) + (sizee - 1);
        if (end > dim.x) {
            end = dim.x;
        }
        for (int x = n * sizee; x < end; x++)
#endif
            for (int y = 0; y < dim.y; y++) {
                for (int z = 0; z < dim.z; z++) {
                    //accessor.setValue(openvdb::Coord(x, y, z), grid_src(x, y, z));
                    accessor.setValue(openvdb::Coord(x, y, z), grid_src_arr[z * dim.y * dim.x + y * dim.x + x]);
                }
            }
#ifdef MULTI
    });
#else
    }
#endif
    //delete[] grid_src_arr;
    openvdb::tools::signedFloodFill(grid_dst.tree());
}

int export_openvdb(std::string filename, int3 domain_resolution, GRID3D& grid_dst, bool DEBUG = false) {
    filename = "output/cache/" + filename + ".vdb";
    
    std::cout << " || Saving OpenVDB:  ";
    //clock_t startTime = clock();
    
    //std::cout << "Done" << std::endl;
    // Create a FloatGrid and populate it with a narrow-band
    // signed distance field of a sphere.
    openvdb::FloatGrid::Ptr grid =
        openvdb::FloatGrid::create(/*background value=*/0.0);


    
    clock_t startTime = clock();

    //std::cout << "Grid created" << std::endl;
    create_grid(*grid, grid_dst, /*center=*/openvdb::Vec3f(0, 0, 0));
    //std::cout << "Grid copied" << std::endl;

    // Associate some metadata with the grid.
    //grid->insertMeta("radius", openvdb::FloatMetadata(50.0));
    // Name the grid "LevelSetSphere".
    std::cout << (clock() - startTime);
    startTime = clock();
    
    grid->setName("density");
    openvdb::io::File(filename).write({ grid });
    
    std::cout << " ; "<< (clock() - startTime);

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
