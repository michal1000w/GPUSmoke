# GPUSmoke

Interactive speed GPU rendering of smoke

## Introduction

To create nice looking smoke is an expensive operation. 
This code leverages NVIDIA GPUs to create smoke simulations quickly and efficiently using CUDA.

## Getting Started

### Prerequisites

1. Download and install [Visual Studio 2019](https://visualstudio.microsoft.com/vs/)
2. Download and install [CUDA Toolkit 10.1 Update 2](https://developer.nvidia.com/cuda-10.1-download-archive-update2)
3. Download, install and add to path [FFmpeg](https://www.ffmpeg.org/download.html)

### Building

1. Clone the repo
2. Build the solution
3. `gpufluid.exe` will be in `x64` or `x86`, `/Debug` or `/Release` depending on your build configuration

### Running Simulation
1. Create a new folder named `output` next to the `gpufluid.exe`
2. Run `gpufluid.exe`

## TODO list
### Short Term
1. Render temperature
2. Simple Graphical Interface
3. Sliders with most important factors
4. Camera and light rotation

### Short-Medium Term
1. Host memory usage for Huge Scale Simulations (resolution over 512^3)
2. Adaptive Domain
3. Object Collision
4. Simple Forces --> (Wind, Turbulance, Noise)
5. Preset creator (LOAD/SAVE)
6. Fix Density over 1.0 issue

### Medium-Long Term
1. Smoke Colors
2. OpenVDB import/export
3. Obj import
4. Simple Blender integration
5. Faster and better Render Engine

### Long Term
1. Smoke adaptive resolution
2. Liquids!!!
3. FLIP fluid solver
4. Sparse Volume optimization
5. FLIP whitewater particles
6. FLIP mesh solver

## Authors

Michał Wieczorek
