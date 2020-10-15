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




GRID3D export_openvdb(std::string filename, int3 domain_resolution, bool DEBUG = false) {
    filename = "input//" + filename + ".nvdb";

    GRID3D output(domain_resolution.x, domain_resolution.y, domain_resolution.z);
    
    openvdb::initialize();
    
    // Create a FloatGrid and populate it with a narrow-band
    // signed distance field of a sphere.
    //openvdb::FloatGrid::Ptr grid = openvdb::tools::createLevelSetSphere<openvdb::FloatGrid>(/*radius=*/50.0, /*center=*/openvdb::Vec3f(1.5, 2, 3),/*voxel size=*/0.5, /*width=*/4.0);
    // Associate some metadata with the grid.
    //grid->insertMeta("radius", openvdb::FloatMetadata(50.0));
    // Name the grid "LevelSetSphere".
    //grid->setName("LevelSetSphere");
    // Create a VDB file object and write out the grid.
    //openvdb::io::File("mygrids.vdb").write({ grid });

    return output;
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
