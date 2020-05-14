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
* [ ] Render temperature
* [ ] Simple Graphical Interface
* [ ] Sliders with most important factors
* [ ] Camera and light rotation

### Short-Medium Term
* [ ] Host memory usage for Huge Scale Simulations (resolution over 512^3)
* [ ] Adaptive Domain
* [ ] Object Collision
* [ ] Simple Forces --> (Wind, Turbulance, Noise)
* [ ] Preset creator (LOAD/SAVE)
* [ ] Fix Density over 1.0 issue

### Medium-Long Term
* [ ] Smoke Colors
* [ ] OpenVDB import/export
* [ ] Obj import
* [ ] Simple Blender integration
* [ ] Faster and better Render Engine

### Long Term
* [ ] Smoke adaptive resolution
* [ ] Liquids!!!
* [ ] FLIP fluid solver
* [ ] Sparse Volume optimization
* [ ] FLIP whitewater particles
* [ ] FLIP mesh solver

## Authors

Michał Wieczorek
