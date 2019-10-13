# PyCLiPSM

## Set up computing environment:
1. Install GPU drivers. If you are using NVIDIA GPUs, OpenCL support is included in the driver (https://developer.nvidia.com/opencl). If you are using AMD GPU/CPU, install appropriate OpenCL drivers from ADM. If you are using Intel GPU/CPU, install appropriate OpenCL drivers from Intel (e.g., https://software.intel.com/en-us/articles/opencl-drivers).    
2. Once the above OpenCL drivers are properly installed, you can use pip to install the PyOpenCL package (https://pypi.org/project/pyopencl/). 
3. Run pyopencl_test.py to see a list of available OpenCL computing platforms/devices on your computer: python pyopencl_test.py
4. Change configurations in the OPENCL_CONFIG variable in utility/config.py accordingly, as well as the OPENCL_CONFIG variable in PyCLiPSM_main.py.
5. Run PyCLiPSM_main.py to get a sense of how to use PyCLiPSM (using sample data provided): python PyCLiPSM_main.py

## Contact
guiming.zhang@du.edu
