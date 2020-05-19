# Author: Guiming Zhang - guiming.zhang@du.edu
# Last update: May. 19 2020
### this is a configuration file specifying various parameters

import sys, os
print 'Python version:', sys.version

MULTITHREAD_READ = False ## read raster with multithreads

TILE_READ = False
## if None, will be determined automatically
TILE_XSIZE = None # 45036 ## setting specific to covariates_10m.vrt (BlockXsize of the underlying geotiff)
TILE_YSIZE = None #128 * 10 # None # 128 * 2 ## multiple of Blocksize in vrt
#https://stackoverflow.com/questions/41742162/gdal-readasarray-for-vrt-extremely-slow

NAIVE = False ## naive implementation of the iPSM algorithm

## CL (OpenCL) implementations
RUN_CL_GPU = True
RUN_CL_GPU_NAIVE = False ## run naive implementaton on GPU
RUN_CL_CPU = True
RUN_CL_CPU_NAIVE = False ## run naive implementaton on CPU
RUN_CL_CPU1 = True
RUN_CL_CPU1_NAIVE = False ## run naive implementaton on 1 CPU

## MP (MultiProcessing) implementations
RUN_MP_CPU = False
RUN_MP_CPU_NAIVE = False ## run naive implementaton on CPU
RUN_MP_CPU1 = False
RUN_MP_CPU1_NAIVE = False ## run naive implementaton on 1 CPU

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
NOMINAL_KEYWORD_IN_FN = ['geo', 'geology']

### OpenCL platform and device specification (change according to the outputs from running pyopencl_test.py)

#OPENCL_CONFIG = {'Platform': 'Intel(R) OpenCL', 'Device':'Intel(R) Core(TM) i9-8950HK CPU @ 2.90GHz'}
OPENCL_CONFIG = {'Platform': 'NVIDIA CUDA', 'Device':'Quadro P2000'}


## run opencl on single cpu core? if True, must use CPU device
SINGLE_CPU = False

#### number of CPU processes (parallel processing using pathos.multiprocesses)
N_PROCESS = 4

#### determine available device memory size using pyopencl
import pyopencl as cl
HOST_MEM_SIZE = 0.0
DEVICE_MEM_SIZE = 0.0
DEVICE_TYPE = 'CPU'
DEVICE = None
DEVICE_MAX_WORK_ITEM_SIZES = 256
def updateInfo():
    ## Host info.
    for platform in cl.get_platforms():
        for device in platform.get_devices():
            if 'CPU' in device.name:
                global HOST_MEM_SIZE
                HOST_MEM_SIZE = device.global_mem_size
                break

    ## Device info.
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
    global DEVICE_MEM_SIZE
    DEVICE_MEM_SIZE = DEVICE.global_mem_size
    global DEVICE_TYPE
    DEVICE_TYPE = cl.device_type.to_string(DEVICE.type)
    global DEVICE_MAX_WORK_ITEM_SIZES
    DEVICE_MAX_WORK_ITEM_SIZES = DEVICE.max_work_item_sizes

updateInfo()

if DEVICE_TYPE == 'GPU' and SINGLE_CPU:
    print 'Warning: Cannot run on single CPU thread on a GPU device. Exiting...'
    sys.exit(1)

# percentage of host, device memory used for iPSM
HOST_MEM_PCT = 0.5 # percent of available mem for iPSM
DEVICE_MEM_PCT = 0.6
CL_CHUNK_SIZE = 20000 # initial chunk size in number of raster pixels; always change in program

#### path to .c file containing opencl kernel functions
iPSM_KERNEL_FN = os.path.dirname(os.path.realpath(__file__)) + os.sep +'ipsm_kernel.c'

## time keeping
TIME_KEEPING_DICT = {'total': 0, 'parts':{'read':[0], 'write':[0], 'data_transfer':[0], 'compute':[0]}}
MEM_USAGE_DICT = {'time': 0.0,'percent':[0], 'used':[0]}
