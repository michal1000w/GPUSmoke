# GPUSmoke

Interactive speed GPU rendering of smoke

## Introduction

To create nice looking smoke is an expensive operation. 
This code leverages NVIDIA GPUs to create smoke simulations quickly and efficiently using CUDA.

## Getting Started

### Prerequisites

1. Download and install [Visual Studio 2019](https://visualstudio.microsoft.com/vs/)
2. Download and install [CUDA Toolkit 11.3](https://developer.nvidia.com/cuda-downloads)

### Building

1. Clone the repo
2. Build the solution
3. `gpufluid.exe` will be in `x64` or `x86`, `/Debug` or `/Release` depending on your build configuration

### Running Simulation
1. Create a new folder named `output` next to the `gpufluid.exe`
2. Create a new folder named `cache` in `output` folder
3. Run `gpufluid.exe`

## GUI
1. W - zoom in
2. S - zoom out
3. A - go left
4. D - go right
5. Left mouse button + A/D rotate camera
6. Scrool zomm in/out
7. Q - camera up
8. Z - camera down
9. R - reset simulation
10. F - stop exporting

## TODO list
### Short Term
* [x] Render temperature
* [x] Simple Graphical Interface
* [x] Sliders with most important factors
* [x] Camera and light rotation

### Short-Medium Term
* [x] Sparse and Multi-Threaded vdb saving
* [ ] Host memory usage for Huge Scale Simulations (resolution over 512^3)
* [ ] Adaptive Domain - probably won't be faster in case of smoke simulation, however I'll need to do more research
* [x] Object Collision
    * [x] Initial sphere collision
    * [x] More complex geometry
* [x] Simple Forces --> (Wind, Turbulance, Noise)
* [x] Preset creator (LOAD/SAVE)
* [x] Fix Density over 1.0 issue
* [ ] Volume upsampling - In the research phaze
    * [x] Wavelet noise upsampling
    * [ ] Custom Neural Network upsampling

### Medium-Long Term
* [ ] Smoke Colors
* [x] OpenVDB IO
    * [x] Import
    * [x] Export
    * Works but slowly
* [x] Obj import
* [ ] Simple Blender integration
* [x] Faster and better Render Engine

### Long Term
* [ ] Smoke adaptive resolution
* [ ] Liquids!!!
* [ ] FLIP fluid solver
* [ ] Sparse Volume optimization
* [ ] FLIP whitewater particles
* [ ] FLIP mesh solver

## Authors

Michał Wieczorek
