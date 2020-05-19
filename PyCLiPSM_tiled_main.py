# Author: Guiming Zhang
# Last update: May 19 2020

import time, os, sys, json, platform
import numpy as np
from utility import iPSM_tiled, util, conf

def main():
     ### data directories
    rootDir = os.path.dirname(os.path.realpath(__file__))
    print 'working directory:', rootDir

    ## INPUTS
    vrtfn = rootDir + os.sep + 'data' + os.sep + 'covariates' + os.sep + 'covariates_1km.vrt'
    samplefn = rootDir + os.sep + 'data' + os.sep + 'samples' + os.sep + 'synthetic_samples_1000.csv'

    ## OUTPUTS
    soilmapfn = rootDir + os.sep + 'outputs' + os.sep + 'propertyMap.tif'
    uncertmapfn = rootDir + os.sep + 'outputs' + os.sep + 'uncertaintyMap.tif'

    ### CL - GPU
    print '\n\n## CL - GPU'
    if conf.RUN_CL_GPU:
        ## change this according to outputs from running pyopencl_test.py
        conf.OPENCL_CONFIG = {'Platform': 'NVIDIA CUDA', 'Device':'Quadro P2000'}
        conf.updateInfo()

        T0 = time.time()
        ipsm = iPSM_tiled.iPSM(vrtfn, samplefn, uncthreshold = 1.0, outfns = [soilmapfn, uncertmapfn])
        ipsm.predict_opencl()

    if conf.RUN_CL_CPU:
        print '\n\n## CL - CPU - MULTIPLE'
        ## change this according to outputs from running pyopencl_test.py
        conf.OPENCL_CONFIG = {'Platform': 'Intel(R) OpenCL', 'Device':'Intel(R) Core(TM) i9-8950HK CPU @ 2.90GHz'}
        conf.updateInfo()

        ipsm = iPSM_tiled.iPSM(vrtfn, samplefn, uncthreshold = 1.0, outfns = [soilmapfn, uncertmapfn])
        ipsm.predict_opencl()

if __name__ == "__main__":
    main()
