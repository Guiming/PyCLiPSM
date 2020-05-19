# Author: Guiming Zhang
# Last update: April 16 2020

import os, time, sys
import numpy as np
import json
import matplotlib
matplotlib.use('Qt4Agg')
from matplotlib import pyplot as plt
from pathos.multiprocessing import ProcessingPool as Pool
import points, raster, conf

def runtime_pie_chart(jsonfn, figfn=None):
    ''' json format:
    dict = {'total': 0, 'parts':{'read':[], 'write':[], 'data_transfer':[], 'compute':[]}}
    '''
    with open(jsonfn, 'r') as fp:
        times = json.load(fp)

        total = times['total']

        read = np.array(times['parts']['read']).sum()
        write = np.array(times['parts']['write']).sum()
        compute = np.array(times['parts']['compute']).sum()
        data_transfer = np.array(times['parts']['data_transfer']).sum()

        other = total - (read + write + data_transfer + compute)
        parts = np.array([read, write, data_transfer, compute, other])

        print 'total:', total, parts.sum()
        print 'RUNTIME BREAKDOWN:', parts
        print 'RUNTIME BREAKDOWN(%):', parts / parts.sum() * 100

        # Pie chart, where the slices will be ordered and plotted counter-clockwise:
        labels = ['Read', 'Write', 'Data Transfer', 'Compute', 'Other']
        sizes = parts / parts.sum() * 100
        explode = (0, 0, 0, 0.1, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

        fig, ax = plt.subplots()
        ax.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                shadow=False, startangle=360)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.title(jsonfn.split('.')[0].split(os.sep)[-1] + ' - Total time: ' + str(int(total*100)/100.0) + ' s')
        #plt.legend(loc = 'best')
        #plt.show()
        #'''
        if figfn is None:
            figfn = jsonfn.replace('.json','.png')
        plt.savefig(figfn, dpi=300)
        #'''
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

def readEnvDataLayers(rasterdir, rasterfns, multithread = conf.MULTITHREAD_READ):
    ''' read in environmental data layers
        ascdir: directory containing .asc files
        asciifns: list of ascii file names
        return: list of rasters, standardized to [0, 1] or []-0.5, 0.5] if needed
    '''
    rst_list = None
    if multithread:
        def threadReadingByFile(rasterfn):
            ''' each thread reads a separate raster
                using multiprocess pool
            '''
            import gdal, gdalconst, conf
            import numpy as np
            rst = None
            if 'geo' in rasterfn:
                rst = raster.Raster(msrlevel = conf.MSR_LEVEL_NOMINAL)
            else:
                rst = raster.Raster(msrlevel = conf.MSR_LEVEL_RATIO)
            rst.readRasterGDAL(rasterfn)
            return rst
        ## multi-thread reading
        n_threads = len(rasterfns)
        MP_pool = Pool(n_threads)
        fns = []
        for i in range(len(rasterfns)):
            fns.append(rasterdir + os.sep + rasterfns[i])
        rst_list = MP_pool.map(threadReadingByFile, fns)
        MP_poo.clear()
    else:
        rst_list = [] # hold environmental variables
        for rasterfn in rasterfns:  ## presume all covariates are continuous
            rst = None
            if 'geo' in rasterfn:
                rst = raster.Raster(msrlevel = conf.MSR_LEVEL_NOMINAL)
            else:
                rst = raster.Raster(msrlevel = conf.MSR_LEVEL_RATIO)
            rst.readRasterGDAL(rasterdir + os.sep + rasterfn)
            #print envmap.getData().size
            rst_list.append(rst)

    return rst_list
