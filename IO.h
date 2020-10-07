#ifndef __IO
#define __IO
#include "Libraries.h"
#include "OpenVDB/tinyvdbio.h"

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

int load_vdb(std::string filename) {
    filename = "input//" + filename + ".vdb";

    // 1. Parse VDB header
    tinyvdb::VDBHeader header;
    std::string warn;
    std::string err;
    tinyvdb::VDBStatus status = tinyvdb::ParseVDBHeader(filename, &header, &err);

    if (status != tinyvdb::TINYVDBIO_SUCCESS) {
        if (!err.empty()) {
            std::cerr << err << std::endl;
        }
        return EXIT_FAILURE;
    }

    // 2. Read Grid descriptors
    std::map<std::string, tinyvdb::GridDescriptor> gd_map;

    status = tinyvdb::ReadGridDescriptors(filename, header, &gd_map, &err);
    if (status != tinyvdb::TINYVDBIO_SUCCESS) {
        if (!err.empty()) {
            std::cerr << err << std::endl;
        }
        return EXIT_FAILURE;
    }

    std::cout << "# of grid descriptors = " << gd_map.size() << std::endl;

    // 3. Read Grids
    status = tinyvdb::ReadGrids(filename, header, gd_map, &warn, &err);
    if (!warn.empty()) {
        std::cout << warn << std::endl;
    }
    if (status != tinyvdb::TINYVDBIO_SUCCESS) {
        if (!err.empty()) {
            std::cerr << err << std::endl;
        }
        return EXIT_FAILURE;
    }

    std::system("pause");
    std::system("cls");
    std::cout << "Load OK" << std::endl;

    std::cout << "Grids:" << std::endl;
    for (const auto& desc : gd_map) {
        std::cout << "    -name: " << desc.second.GridName() << "\n";
    }

    return EXIT_SUCCESS;
}
#endif
