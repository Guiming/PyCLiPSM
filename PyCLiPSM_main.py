# Author: Guiming Zhang - guiming.zhang@du.edu
# Last update: Oct 1 2019

import os, time, sys, platform
import numpy as np
from scipy import stats
from utility import raster, points, util, conf, iPSM

def main():
    ### data directories
    rootDir = os.path.dirname(os.path.realpath(__file__))
    print 'working directory:', rootDir

    sampleDir = rootDir + os.sep + 'data' + os.sep + 'samples'
    samplefn = 'synthetic_samples_1000.csv'

    ascDir = rootDir + os.sep + 'data' + os.sep + 'covariates' + os.sep + '1km'
    asciifns = ['acc10_n_std.tif', 'arid_std.tif', 'evap_std.tif', 'geo_ma.tif', 'im_std.tif', \
            'ndvi_std.tif', 'plan_n_std.tif', 'prof_n_std.tif', 'rain_std.tif',\
            'slopetan_n_std.tif', 'tem_std.tif', 'twi_std.tif']

    t0 = time.time()
    ### read in soil samples
    soilsamples = points.Points()
    soilsamples.readFromCSV(sampleDir + os.sep + samplefn)

    ### read in environmental data layers
    predictors = util.readEnvDataLayers(ascDir, asciifns)

    print 'reading samples and covariates took', time.time() - t0, 's\n\n'

    t0 = time.time()
    X_te = util.extractCovariatesAtPoints(predictors, soilsamples).T
    X_te = None # mapping
    print 'extracting covariate values at sample locations took', time.time() - t0, 's\n\n'

    ### ipsm ###
    ipsm = iPSM.iPSM(predictors, soilsamples)

    print '## CL - GPU'
    ## change this according to outputs from running pyopencl_test.py
    conf.OPENCL_CONFIG = {'Platform': 'NVIDIA CUDA', 'Device':'Quadro P2000'}
    conf.SINGLE_CPU = False
    conf.updateInfo()
    t0 = time.time()
    y_map = ipsm.predict_opencl(X_te, predict_class = False)
    print 'ipsm.predict_opencl() GPU PARALLEL took', time.time()-t0, 's\n\n'
    if y_map is not None and np.shape(y_map)[0] == predictors[0].getData().size:
        propertymap = predictors[0].copySelf()
        propertymap.filename = 'property_ipsm_GPU'
        propertymap.updateRasterData(y_map[:,0])
        propertymap.writeRasterGDAL('outputs' + os.sep + propertymap.filename + '.tif')

        uncertaintymap = predictors[0].copySelf()
        uncertaintymap.filename = 'uncert_ipsm_GPU'
        uncertaintymap.updateRasterData(y_map[:,1])
        uncertaintymap.writeRasterGDAL('outputs' + os.sep + uncertaintymap.filename + '.tif')

    print '## CL - CPU'
    ## change this according to outputs from running pyopencl_test.py
    conf.OPENCL_CONFIG = {'Platform': 'Intel(R) OpenCL', 'Device':'Intel(R) Core(TM) i9-8950HK CPU @ 2.90GHz'}
    conf.SINGLE_CPU = False
    conf.updateInfo()
    t0 = time.time()
    y_map = ipsm.predict_opencl(X_te, predict_class = False)
    print 'ipsm.predict_opencl() CPU PARALLEL took', time.time()-t0, 's\n\n'
    if y_map is not None and np.shape(y_map)[0] == predictors[0].getData().size:
        propertymap = predictors[0].copySelf()
        propertymap.filename = 'property_ipsm_CPU'
        propertymap.updateRasterData(y_map[:,0])
        propertymap.writeRasterGDAL('outputs' + os.sep + propertymap.filename + '.tif')

        uncertaintymap = predictors[0].copySelf()
        uncertaintymap.filename = 'uncert_ipsm_CPU'
        uncertaintymap.updateRasterData(y_map[:,1])
        uncertaintymap.writeRasterGDAL('outputs' + os.sep + uncertaintymap.filename + '.tif')

    print '## CL - CPU - SINGLE'
    ## change this according to outputs from running pyopencl_test.py
    conf.OPENCL_CONFIG = {'Platform': 'Intel(R) OpenCL', 'Device':'Intel(R) Core(TM) i9-8950HK CPU @ 2.90GHz'}
    conf.SINGLE_CPU = True
    conf.updateInfo()
    t0 = time.time()
    y_map = ipsm.predict_opencl(X_te, predict_class = False, single_cpu = conf.SINGLE_CPU)
    print 'ipsm.predict_opencl() CPU SEQUENTIAL took', time.time()-t0, 's\n\n'
    if y_map is not None and np.shape(y_map)[0] == predictors[0].getData().size:
        propertymap = predictors[0].copySelf()
        propertymap.filename = 'property_ipsm_CL_CPU1'
        propertymap.updateRasterData(y_map[:,0])
        propertymap.writeRasterGDAL('outputs' + os.sep + propertymap.filename + '.tif')

        uncertaintymap = predictors[0].copySelf()
        uncertaintymap.filename = 'uncert_ipsm_CL_CPU1'
        uncertaintymap.updateRasterData(y_map[:,1])
        uncertaintymap.writeRasterGDAL('outputs' + os.sep + uncertaintymap.filename + '.tif')


if __name__ == "__main__":
    T0 = time.time()
    main()
    print 'IN TOTAL IT TOOK', time.time() - T0, 's\n\n'
