#ifndef __LIBRARIES
#define __LIBRARIES

#include <stdio.h>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <conio.h>

//#include <cuda_fp16.h> //to siê gryzie
#include "cutil_math.h"


//#include "Fluid_Kernels.cuh"
//#include "Unified_Buffer.cpp"
#include "HugeScaleSolver.cu"








#include "third_party/openvdb/nanovdb/nanovdb/NanoVDB.h"
#include <windows.h>
#include <ppl.h>
#include <thread>
//#include "OpenVDB-old/tinyvdbio.h"
#include <nanovdb/util/GridBuilder.h>
#include <nanovdb/util/IO.h>
#include <nanovdb/util/CudaDeviceBuffer.h>
#include <nanovdb/util/GridBuilder.h>

#endif