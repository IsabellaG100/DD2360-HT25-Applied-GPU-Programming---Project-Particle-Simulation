# DD2360-HT25-Applied-GPU-Programming
# Project-Particle-Simulation
This is our repository for the project in Applied GPU programming. The assignment is to improve the particle simulations (iPIC3D-mini) code's efficiency. Code taken from https://github.com/KTH-ScaLab/DD2360-HT25.git

# Project Overview
This project focuses on accelerating a simplified particle-in-cell plasma simulation, IPIC3D-mini, by offloading computationally intensive parts of the application from the CPU to the GPU using CUDA. The primary goal is to improve performance by exploiting the inherent data parallelism of particle-based computations.

In particular, the particle mover and particle-to-grid (P2G) interpolation routines are targeted, as they dominate the overall runtime and operate over large numbers of independent particles. Performance profiling is used throughout the project to guide optimization decisions and to support a comparison between the CPU and GPU implementations.

# Repository Structure
```text
├── src/                  # Source files (CPU and GPU implementations)
│   ├── Particles.cu      # Particle mover and P2G kernels
│   ├── EMfield.cpp
│   └── ...
├── include/              # Header files
├── inputfiles/           # Simulation input files (e.g. GEM_2D, GEM_3D)
├── bin/                  # Compiled binaries
├── scripts/              # Profiling and validation scripts
├── Makefile
└── README.md
```

# Build instructions
We recommend running the code in Google Colab. The repository with instructions can be found here:
https://colab.research.google.com/drive/1DW0B0zMKLNIHjdvToutz4Xy3X2gB32GM?usp=sharing

If you want to run the code on your local machine:
## Requirements to run on local machine:
* Nvidia GPU with CUDA support
* CUDA Toolkit (tested with CUDA 12.x)
* C++ compiler with C++11 support

## Download
```
git clone https://github.com/IsabellaG100/DD2360-HT25-Applied-GPU-Programming---Project-Particle-Simulation
```

## Compile
```
make
```

## Create folder to store output
```
mkdir data
```

## Run the code
Choose which one of the input files you want to use (GEM_2D or GEM_3D)
```
!./bin/miniPIC.out inputfiles/GEM_2D.inp # GEM_2D input file
./bin/miniPIC.out inputfiles/GEM_3D.inp # GEM_3D input file
```

# Visualize results in ParaView
Output files will be stored in the data folder if you created one.
Load the output file you want to visualize into ParaView.

# Profiling and/or validating results
Visit Google Colab Notebook and follow instuctions:
https://colab.research.google.com/drive/1DW0B0zMKLNIHjdvToutz4Xy3X2gB32GM?usp=sharing

# GPU Implementation Overview
The GPU implementation parallelizes particle updates by assigning one thread per particle, allowing particle position and velocity updates to be computed independently. Particle data are transferred to device memory prior to kernel execution, and results are copied back to the host after computation.

The particle-to-grid (P2G) interpolation step is also offloaded to the GPU. Since multiple particles may contribute to the same grid cell, atomic operations are used to ensure correctness during accumulation.

# Profiling and Performance Analysis
Performance profiling is conducted using NVIDIA Nsight Systems. Profiling is used to analyze kernel execution times, CPU–GPU synchronization points, and memory transfer overheads. These measurements help identify performance bottlenecks and guide optimization decisions in the GPU implementation.

Profiling results are also used to contextualize runtime comparisons between the CPU and GPU versions of the application.

# Validation and Results
The correctness of the GPU implementation is validated by comparing simulation outputs against the original CPU reference implementation. Output fields and particle quantities are written in VTK legacy format and compared on a field-by-field basis.

Due to differences in floating-point execution order, parallel reductions, and atomic operations, bitwise identical results are not expected. Instead, numerical equivalence within acceptable tolerances is assessed.

# Course Context
This project was developed as part of the DD2360 Applied GPU Programming course at KTH Royal Institute of Technology.