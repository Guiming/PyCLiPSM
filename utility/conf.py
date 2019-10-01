# Author: Guiming Zhang - guiming.zhang@du.edu
# Last update: Oct. 1 2019
### this is a configuration file specifying various parameters

import sys, os
print 'Python version:', sys.version

NAIVE = False ## naive implementation of the iPSM algorithm

#### debug mode
DEBUG_FLAG = False

N_INTERVALS = 300
N_HIST_BINS = 50

#### measurement level of environmental covariates
MSR_LEVELS = {'0':'nominal', '1':'ordinal', '2':'interval', '3':'ratio', '4':'count'}
MSR_LEVEL_NOMINAL = MSR_LEVELS['0']
MSR_LEVEL_ORDINAL = MSR_LEVELS['1']
MSR_LEVEL_INTERVAL = MSR_LEVELS['2']
MSR_LEVEL_RATIO = MSR_LEVELS['3']
MSR_LEVEL_COUNT = MSR_LEVELS['4']

### OpenCL platform and device specification (change according to the outputs from running pyopencl_test.py)

OPENCL_CONFIG = {'Platform': 'Intel(R) OpenCL', 'Device':'Intel(R) Core(TM) i9-8950HK CPU @ 2.90GHz'}
#OPENCL_CONFIG = {'Platform': 'NVIDIA CUDA', 'Device':'Quadro P2000'}


## run opencl on single cpu core? if True, must use CPU device
SINGLE_CPU = False

#### number of CPU processes (parallel processing using pathos.multiprocesses)
N_PROCESS = 4

#### determine available device memory size using pyopencl
import pyopencl as cl
GLOBAL_MEM_SIZE = 0.0
DEVICE_TYPE = 'CPU'
DEVICE = None
def updateInfo():
    PLT_IDX = -1 # platform index
    DVC_IDX = -1 # device index
    BRK = False
    for platform in cl.get_platforms():
        PLT_IDX += 1
        if platform.name == OPENCL_CONFIG['Platform']:
            for device in platform.get_devices():
                DVC_IDX += 1
                if device.name == OPENCL_CONFIG['Device']:
                    BRK = True
                    break # break inner loop
        if BRK: break # break outer loop
    DEVICE = cl.get_platforms()[PLT_IDX].get_devices()[DVC_IDX]
    global GLOBAL_MEM_SIZE
    GLOBAL_MEM_SIZE = DEVICE.global_mem_size
    global DEVICE_TYPE
    DEVICE_TYPE = cl.device_type.to_string(DEVICE.type)
updateInfo()

if DEVICE_TYPE == 'GPU' and SINGLE_CPU:
    print 'Warning: Cannot run on single CPU thread on a GPU device. Exiting...'
    sys.exit(1)

# percentage of memory used for iPSM
MEM_PCT = 0.8
CL_CHUNK_SIZE = 20000 # initial chunk size in number of raster pixels; always change in program

#### path to .c file containing opencl kernel functions
iPSM_KERNEL_FN = os.path.dirname(os.path.realpath(__file__)) + os.sep +'ipsm_kernel.c'
