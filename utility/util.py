# Author: Guiming Zhang
# Last update: August 8 2019

import os, time, sys
import numpy as np
import points, raster, conf

def extractCovariatesAtPoints(rasters, samples):
    #t0 = time.time()
    ''' extract values at sample points from a list of raster GIS data layers
        it is assumed all rasters have the same spatial extent
    '''
    try:
        N_PNTS = samples.size
        N_LYRS = len(rasters)

        nd = 0 # num of NoData
        rs = []
        cs = []
        for p in range(N_PNTS):
            x = samples.xycoords[p,0]
            y = samples.xycoords[p,1]
            r,c = rasters[0].xy2RC(x, y)
            rs.append(r)
            cs.append(c)

        vals = np.zeros((N_LYRS, N_PNTS))
        for i in range(N_LYRS):
            vals[i, :] = rasters[i].getData2D()[rs, cs]

        return vals

    except Exception as e:
        raise

def readEnvDataLayers(ascdir, asciifns):
    ''' read in environmental data layers
        ascdir: directory containing .asc files
        asciifns: list of ascii file names
        return: list of rasters, standardized to [0, 1] or []-0.5, 0.5] if needed
    '''
    envmaps = [] # hold environmental variables
    matrix = [] # hold data for PCA
    for asciifn in asciifns:  ## presume all covariates are continuous
        envmap = raster.Raster()
        envmap.readFromAscii(ascdir + os.sep + asciifn)
        envmaps.append(envmap)
    return envmaps
