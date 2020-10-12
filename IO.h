#ifndef __IO
#define __IO
#include "Libraries.h"
//#include "OpenVDB/tinyvdbio.h"
//#include <openvdb/openvdb.h>
#include "third_party/openvdb/nanovdb/nanovdb/NanoVDB.h"


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

    return 0;
}
#endif
