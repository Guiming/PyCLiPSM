# PyCLiPSM
PyCLiPSM: Harnessing heterogeneous computing resources on CPUs and GPUs for accelerated digital soil mapping. Find more detail about the code in this paper: https://onlinelibrary.wiley.com/doi/full/10.1111/tgis.12730 (an author's copy is included in this repository "2021 - TGIS - PyCLiPSM.pdf").

## Citation
Zhang, G, Zhu, A‐X, Liu, J, Guo, S, Zhu, Y. PyCLiPSM: Harnessing heterogeneous computing resources on CPUs and GPUs for accelerated digital soil mapping. Transactions in GIS. 2021; 00: 1– 23. https://doi.org/10.1111/tgis.12730

## Environment
Linux/Windows/Mac

## Set up computing environment:
1. Install GPU drivers. If you are using NVIDIA GPUs, OpenCL support is included in the driver (https://developer.nvidia.com/opencl). If you are using AMD GPU/CPU, install appropriate OpenCL drivers from ADM (This link provides helpful pointers https://github.com/microsoft/LightGBM/issues/1342). If you are using Intel GPU/CPU, install appropriate OpenCL drivers from Intel (e.g., https://software.intel.com/en-us/articles/opencl-drivers).    
2. It's assumed that you already have Python (version 2.7) installed. Anaconda is recommended for installing Python https://www.anaconda.com/distribution/. 
3. Install Python GDAL (https://pypi.org/project/GDAL/).
4. Once OpenCL drivers, Python and Python GDAL are properly installed, you can use pip to install the PyOpenCL package (https://pypi.org/project/pyopencl/). 
5. Run pyopencl_test.py to test if PyOpenCL is working properly: python pyopencl_test.py.

## Run PyCLiPSM with sample data:
1. Run pyopencl_test.py to see a list of available OpenCL computing platforms/devices on your computer: python pyopencl_test.py
2. Change configurations in the OPENCL_CONFIG variable in utility/config.py accordingly, as well as the OPENCL_CONFIG variable in PyCLiPSM_main.py.
3. Run PyCLiPSM_main.py to get a sense of how to use PyCLiPSM (using example data provided): python PyCLiPSM_main.py

## Use PyCLiPSM for your own application:
1. Prepare soil sample data and environmental covariate data following the example data in the "data" directory
2. Change parameters (e.g., data directory, data file names, etc.) in PyCLiPSM_main.py or PyCLiPSM_tiled_main.py accordingly
3. Run PyCLiPSM_main.py or PyCLiPSM_tiled_main.py: python PyCLiPSM_main.py or python PyCLiPSM_tiled_main.py

## License
Copyright 2021 Guiming Zhang. Distributed under MIT license.

## Contact
guiming.zhang@du.edu
