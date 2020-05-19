# Author: Guiming Zhang
# Last update: May 19 2020
#http://www.paolocorti.net/2012/03/08/gdal_virtual_formats/
#https://stackoverflow.com/questions/41742162/gdal-readasarray-for-vrt-extremely-slow
#https://gdal.org/development/rfc/rfc26_blockcache.html


import gdal, gdalconst, glob, psutil
import os, sys, time
import random, math
import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool
import conf

class VRTBuilder:
    '''
    '''
    def __init__(self):
        '''
        '''
    def buildVRT(self, srcFilelist, outVrt):
        vrt_options = gdal.BuildVRTOptions(separate=True, VRTNodata=-9999)
        gdal.BuildVRT(outVrt, srcFilelist, options=vrt_options)

class tiledRasterReader:
    '''
    '''
    def __init__(self, srcRasterfile, xoff=0, yoff=0, xsize=None, ysize=None):
        '''
        '''
        self.srcRasterfile = srcRasterfile
        gdal.SetCacheMax(2**30) # 1 GB
        self.ds = gdal.Open(self.srcRasterfile, gdalconst.GA_ReadOnly)
        self.fileList = self.ds.GetFileList()[1:]
        self.measurement_level_ints = []
        for fn in self.fileList:
            # default level of measurement
            msrlevel = conf.MSR_LEVEL_RATIO
            for keyword in conf.NOMINAL_KEYWORD_IN_FN:
                if keyword in fn:
                    msrlevel = conf.MSR_LEVEL_NOMINAL
                    break
            for key in conf.MSR_LEVELS:
                if conf.MSR_LEVELS[key] == msrlevel:
                    self.measurement_level_ints.append(int(key))
                    break
        self.measurement_level_ints = np.array(self.measurement_level_ints)

        self.nbands = self.ds.RasterCount
        self.nrows = self.ds.RasterYSize
        self.ncols = self.ds.RasterXSize
        self.geotransfrom = self.ds.GetGeoTransform()
        self.projection = self.ds.GetProjection()

        band = self.ds.GetRasterBand(1)
        self.nodata = band.GetNoDataValue()

        self.block_ysize_base = band.GetBlockSize()[0]
        self.block_xsize_base = gdal.Open(self.fileList[0], gdalconst.GA_ReadOnly).GetRasterBand(1).GetBlockSize()[0]

        self.__N_TilesRead = 0
        self.xoff, self.yoff = xoff, yoff

        if xsize is None:
            self.xsize = self.block_xsize_base
        elif xsize > self.ncols:
            print 'tile xsize exceeds RasterXsize', self.ncols
            sys.exit(1)
        else:
            self.xsize = xsize

        if ysize is None:
            self.ysize = self.block_ysize_base
        elif ysize > self.nrows:
            print 'tile xsize exceeds RasterYsize', self.nrows
            sys.exit(1)
        else:
            self.ysize = ysize


        ## estimated data size (in MB)
        self.estimate_TotalSize_MB = self.estimateTileSize_MB(self.nrows, self.ncols)
        self.estimate_TileSize_MB = self.estimateTileSize_MB(self.xsize, self.ysize)

        self.statistics = np.zeros((self.nbands, 4))
        for i in range(self.nbands):
            self.statistics[i] = self.ds.GetRasterBand(i+1).GetStatistics(0, 1)

        self.MP_pool = None

    def estimateTileSize_MB(self, xsize=None, ysize=None):
        '''
        '''
        if xsize is None:
            xsize = self.xsize
        if ysize is None:
            ysize = self.ysize
        return np.array([1.0]).astype('float32').nbytes / 1024.0**2 * xsize * ysize  * self.nbands

    def readWholeRaster(self, multithread = conf.MULTITHREAD_READ):
        data  = None
        if multithread:
            def threadReadingByBand(i, rasterfile):
                ''' each thread reads a whole band
                    using multiprocess pool
                '''
                import gdal, gdalconst, psutil, conf
                import numpy as np
                ds = gdal.Open(rasterfile, gdalconst.GA_ReadOnly)
                data = ds.GetRasterBand(i).ReadAsArray()
                return data

            # optimal for multi-thread reading by band
            n_threads = self.nbands
            if self.MP_pool is None:
                self.MP_pool = Pool(n_threads)

            ## multi-thread reading by band
            band_idx = range(1, n_threads + 1)
            fns = np.array([self.srcRasterfile]).repeat(n_threads)
            data = self.MP_pool.map(threadReadingByBand, band_idx, fns)
            data = np.stack(data, axis=0)
            self.MP_pool.clear()
        else:
            data = self.ds.ReadAsArray(xoff=0, yoff=0, xsize=None, ysize=None)

        ## nodatavalues
        data[data < self.nodata] = self.nodata
        return data

    def readNextTile(self, xsize=None, ysize=None, multithread = conf.MULTITHREAD_READ):
        ## update xsize and ysize if needed
        ## PLEASE specify xsize, ysize ONLY ONCE (when reading the first tile)
        if xsize is not None: self.xsize = xsize
        if ysize is not None: self.ysize = ysize

        N_BLOCK_X = int(math.ceil(self.ncols*1.0/self.xsize))
        y = int(self.__N_TilesRead / N_BLOCK_X)
        x = self.__N_TilesRead - y * N_BLOCK_X

        self.xoff = min(x * self.xsize, self.ncols)
        xsize = min(self.xsize, self.ncols - self.xoff)

        self.yoff = min(y * self.ysize, self.nrows)
        ysize = min(self.ysize, self.nrows - self.yoff)

        if self.xoff == self.ncols or self.yoff == self.nrows:
            return (None, self.xoff, self.yoff, 0, 0)

        data = None
        if multithread: ## multi-thread read
            def threadReadingByBand(i, param, rasterfile):
                ''' each thread reads a band, with tile dimension spec in param
                    using multiprocess pool
                '''
                import gdal, gdalconst
                import numpy as np
                ds = gdal.Open(rasterfile, gdalconst.GA_ReadOnly)
                data = ds.GetRasterBand(i).ReadAsArray(xoff=param[0], yoff=param[1], win_xsize=param[2], win_ysize=param[3])
                return data

            # optimal for multi-thread reading by band
            n_threads = self.nbands# - 1
            if self.MP_pool is None:
                self.MP_pool = Pool(n_threads)

            ## multi-thread reading by band
            params = []
            for i in range(n_threads):
                params.append([self.xoff, self.yoff, xsize, ysize])
            fns = np.array([self.srcRasterfile]).repeat(n_threads)
            band_idx = range(1, self.nbands + 1)
            data = self.MP_pool.map(threadReadingByBand, band_idx, params, fns)
            data = np.stack(data, axis=0)
            self.MP_pool.clear()

        else: ## single-thread read
            data = self.ds.ReadAsArray(xoff=self.xoff, yoff=self.yoff, xsize=xsize, ysize=ysize)
        ## nodatavalues
        data[data < self.nodata] = self.nodata

        self.__N_TilesRead += 1

        return (data, self.xoff, self.yoff, xsize, ysize)

    def reset(self):
        ''' reset after reading tiles
        '''
        self.xoff, self.yoff = 0, 0
        self.__N_TilesRead = 0

    def extractByXY(self, x, y):
        ''' Extract raster value by x, y coordinates
        '''
        xoff = int((x - self.geotransfrom[0]) / self.geotransfrom[1])
        yoff = int((y - self.geotransfrom[3]) / self.geotransfrom[5])
        return self.ds.ReadAsArray(xoff, yoff, 1, 1)

    def extractByRC(c, r):
        '''Extract raster value by row, col
        '''
        return self.ds.ReadAsArray(c, r, 1, 1)

    def close(self):
        self.ds = None
        if self.MP_pool is not None:
            self.MP_pool.clear()

class tiledRasterWriter:
    '''
    '''
    def __init__(self, outRasterfile, nrows, ncols, geotransfrom, projection, nodata=-9999.0):
        '''
        '''
        self.nrows, self.ncols, self.nodata = nrows, ncols, nodata
        driver = gdal.GetDriverByName('GTiff')
        self.ds = driver.Create(outRasterfile, self.ncols, self.nrows, 1, gdal.GDT_Float32, options = [ 'COMPRESS=LZW', 'BIGTIFF=YES' ])
        self.ds.SetGeoTransform(geotransfrom)
        self.ds.SetProjection(projection)

        self.band = self.ds.GetRasterBand(1)
        self.band.SetNoDataValue(self.nodata)

    def WriteWholeRaster(self, data):
        self.band.WriteArray(data, xoff=0, yoff=0)
        self.band.FlushCache()
        data = None

    def writeTile(self, data, xoff, yoff):
        self.band.WriteArray(data, xoff=xoff, yoff=yoff)
        self.band.FlushCache()
        data = None

    ## Have to call this to write to disc
    def close(self):
        stats = self.band.GetStatistics(0, 1)
        self.band.SetStatistics(stats[0], stats[1], stats[2], stats[3])
        #self.band.FlushCache()
        self.band = None
        self.ds = None
