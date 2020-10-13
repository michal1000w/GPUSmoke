#ifndef __IO
#define __IO
#include "Libraries.h"
//#include "OpenVDB/tinyvdbio.h"
//#include <openvdb/openvdb.h>
#include "third_party/openvdb/nanovdb/nanovdb/NanoVDB.h"
#include <windows.h>
#include <ppl.h>
#include <thread>
#include "OpenVDB-old/tinyvdbio.h"
#include <nanovdb/util/GridBuilder.h>
#include <nanovdb/util/IO.h>
#include <nanovdb/util/CudaDeviceBuffer.h>
#include <nanovdb/util/GridBuilder.h>

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

GRID3D load_vdb(std::string filename, int3 domain_resolution, bool DEBUG = false) {
    filename = "input//" + filename + ".nvdb";

    GRID3D outputt(domain_resolution.x, domain_resolution.y, domain_resolution.z);
    //nanovdb::FloatGrid* output = nullptr;
    try {
        // returns a GridHandle using CUDA for memory management.
        auto handle = nanovdb::io::readGrids(filename);

        auto* grid = handle[0].grid<float>(); // get a (raw) pointer to a NanoVDB grid of value type float on the CPU
        auto output = grid->getAccessor();
        //for (int x = 0; x < domain_resolution.x; x++)
        int THREADS = 8;//8
        int sizee = ceil((double)domain_resolution.x / (double)THREADS);
        concurrency::parallel_for(0, THREADS, [&](int n) {
            int end = (n * sizee) + (sizee - 1);
            if (end > domain_resolution.x) {
                end = domain_resolution.x;
            }
            for (int x = n * sizee; x < end; x++)
                for (int y = 0; y < domain_resolution.y; y++)
                    for (int z = 0; z < domain_resolution.z; z++) {
                        if (DEBUG)
                            std::cout << "\r" << output.getValue(nanovdb::Coord(x, y, z)) << " , ";
                        outputt.set(x, y, z, output.getValue(nanovdb::Coord(x, y, z)));
                    }
        });
    }
    catch (const std::exception& e) {
        std::cout << "[Import] An exception occurred: \"" << e.what() << "\"" << std::endl;
    }
    return outputt;
}

GRID3D load_vdb3(std::string filename, int3 domain_resolution, bool DEBUG = false) {
    filename = "input//" + filename + ".nvdb";

    GRID3D outputt(domain_resolution.x, domain_resolution.y, domain_resolution.z);
    //nanovdb::FloatGrid* output = nullptr;
    try {
        // returns a GridHandle using CUDA for memory management.
        auto handle = nanovdb::io::readGrids(filename);

        auto* grid = handle[0].grid<float>(); // get a (raw) pointer to a NanoVDB grid of value type float on the CPU
        auto object = grid->getAccessor();

        std::cout << grid->activeVoxelCount() << std::endl;
        std::cout << domain_resolution.x * domain_resolution.y * domain_resolution.z << std::endl;
        
        //for (int x = 0; x < domain_resolution.x; x++)
        int THREADS = 8;//8
        int sizee = ceil((double)domain_resolution.x / (double)THREADS);
        concurrency::parallel_for(0, THREADS, [&](int n) {
            int end = (n * sizee) + (sizee - 1);
            if (end > domain_resolution.x) {
                end = domain_resolution.x;
            }
            for (int x = n * sizee; x < end; x++)
                for (int y = 0; y < domain_resolution.y; y++)
                    for (int z = 0; z < domain_resolution.z; z++) {
                        if (DEBUG)
                            std::cout << "\r" << object.getValue(nanovdb::Coord(x, y, z)) << " , ";
                        outputt.set(x, y, z, object.getValue(nanovdb::Coord(x, y, z)));
                    }
            });
    }
    catch (const std::exception& e) {
        std::cout << "[Import] An exception occurred: \"" << e.what() << "\"" << std::endl;
    }
    return outputt;
}


GRID3D load_vdb2(std::string filename, int3 domain_resolution, bool DEBUG = false) {
    filename = "input//" + filename + ".nvdb";

    GRID3D outputt(domain_resolution.x, domain_resolution.y, domain_resolution.z);
    //nanovdb::FloatGrid* output = nullptr;
    try {
        // returns a GridHandle using CUDA for memory management.
        auto handle = nanovdb::io::readGrid<nanovdb::CudaDeviceBuffer>(filename);

        cudaStream_t stream; // Create a CUDA stream to allow for asynchronous copy of pinned CUDA memory.
        cudaStreamCreate(&stream);

        handle.deviceUpload(stream, false); // Copy the NanoVDB grid to the GPU asynchronously

        const nanovdb::FloatGrid* grid = handle.deviceGrid<float>();

        
        int THREADS = 8;//8
        int sizee = ceil((double)domain_resolution.x / (double)THREADS);
        concurrency::parallel_for(0, THREADS, [&](int n) {
            int end = (n * sizee) + (sizee - 1);
            if (end > domain_resolution.x) {
                end = domain_resolution.x;
            }
            for (int x = n * sizee; x < end; x++)
                for (int y = 0; y < domain_resolution.y; y++)
                    for (int z = 0; z < domain_resolution.z; z++) {
                        if (DEBUG)
                            std::cout << "\r" << grid->tree().getValue(nanovdb::Coord(x, y, z)) << " , ";
                        outputt.set(x, y, z, grid->tree().getValue(nanovdb::Coord(x, y, z)));
                    }
            });
        cudaStreamDestroy(stream);
    }
    catch (const std::exception& e) {
        std::cerr << "An exception occurred: \"" << e.what() << "\"" << std::endl;
    }
    return outputt;
}

int export_vdb(std::string filename, int3 domain_resolution) {
    filename = "input//" + filename + ".nvdb";
    try {
        std::vector<nanovdb::GridHandle<>> handles;
        // Create multiple NanoVDB grids of various types
        handles.push_back(nanovdb::createLevelSetSphere<float>(1000.0f));
        
        //handles.push_back(nanovdb::createLevelSetTorus<float>(20.0f, 10.0f));
        //handles.push_back(nanovdb::createLevelSetBox<float>(domain_resolution.x, domain_resolution.y, domain_resolution.z));
        //handles.push_back(nanovdb::createLevelSetBBox<float>(domain_resolution.x, domain_resolution.y, domain_resolution.z, 1.0f));
        //handles.push_back(nanovdb::createPointSphere<float>(1, 0.2f));

        auto* dstGrid = handles[0].grid<float>(); // Get a (raw) pointer to the NanoVDB grid form the GridManager.
        if (!dstGrid)
            throw std::runtime_error("Export GridHandle does not contain a grid with value type float");

        // Access and print out a single value (inside the level set) from both grids
        //printf("NanoVDB cpu: %4.2f\n", dstGrid->tree().getValue(nanovdb::Coord(99, 0, 0)));

        nanovdb::io::writeGrids<nanovdb::HostBuffer, std::vector>(filename, handles, nanovdb::io::Codec::BLOSC); // Write the NanoVDB grids to file and throw if writing fails
        //nanovdb::io::writeGrids<nanovdb::HostBuffer, std::vector>(filename, handles);
    }
    catch (const std::exception& e) {
        std::cout << "[Export] An exception occurred: \"" << e.what() << "\"" << std::endl;
    }
    return 0;
}


using BufferT = nanovdb::CudaDeviceBuffer;


int export_vdb2(std::string filename, int3 domain_resolution) {
    filename = "input//" + filename + ".nvdb";
    try {
        auto handle = nanovdb::createFogVolumeSphere<float, BufferT>(100.0f, nanovdb::Vec3R(-20, 0, 0), 1.0f, 3.0f, nanovdb::Vec3R(0), "sphere");
       
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
            handle = nanovdb::createFogVolumeSphere<float, BufferT>(100.0f, nanovdb::Vec3R(-20, 0, 0), 1.0f, 3.0f, nanovdb::Vec3R(0), "sphere");
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
