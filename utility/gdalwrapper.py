# Author: Guiming Zhang
# Last update: April 12 2020
#http://www.paolocorti.net/2012/03/08/gdal_virtual_formats/
#https://stackoverflow.com/questions/41742162/gdal-readasarray-for-vrt-extremely-slow
#https://gdal.org/development/rfc/rfc26_blockcache.html


import gdal, gdalconst, glob, psutil
import os, sys, time
import random, math
import numpy as np
from pathos.multiprocessing import ProcessingPool as Pool
#import multiprocess
#from multiprocess import Process
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
        #print xsize, ysize
        self.srcRasterfile = srcRasterfile
        gdal.SetCacheMax(2**30) # 1 GB
        self.ds = gdal.Open(self.srcRasterfile, gdalconst.GA_ReadOnly)
        self.fileList = self.ds.GetFileList()[1:]
        #print self.fileList
        #print self.ds.GetRasterBand(1).CreateMaskBand(0)
        #print self.ds.GetRasterBand(1).GetMaskBand().ReadAsArray().max()
        #sys.exit(0)
        ## collect measurement level(s) of the raster bands
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

        ## collect statistics
        #print 'started collecting statistics...'
        #t0 = time.time()
        self.statistics = np.zeros((self.nbands, 4))
        for i in range(self.nbands):
            self.statistics[i] = self.ds.GetRasterBand(i+1).GetStatistics(0, 1)
        #print 'collecting statistics took', time.time()-t0, 's'

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
            #print '...multithread thread reading...'
            def threadReadingByTile(param, srcRasterfile):
                ''' each thread reads param[1] of rows
                    starting at row param[0]
                    using multiprocess pool
                '''
                import gdal, gdalconst
                import numpy as np
                ds = gdal.Open(srcRasterfile, gdalconst.GA_ReadOnly)
                data = ds.ReadAsArray(xoff=0, yoff=param[0], xsize=None, ysize=param[1])
                #print param, data.shape
                return data

            def threadReadingByBand(i, rasterfile):
                ''' each thread reads a whole band
                    using multiprocess pool
                '''
                import gdal, gdalconst, psutil, conf
                import numpy as np
                ds = gdal.Open(rasterfile, gdalconst.GA_ReadOnly)
                data = ds.GetRasterBand(i).ReadAsArray()
                #print param, data.shape
                #global conf.MEM_USAGE_DICT
                conf.MEM_USAGE_DICT['percent'].append(dict(psutil.virtual_memory()._asdict())['percent'])
                #global conf.MEM_USAGE_DICT
                conf.MEM_USAGE_DICT['used'].append(dict(psutil.virtual_memory()._asdict())['used']/1024.0**2)
                return data

            # optimal for multi-thread reading by band
            n_threads = self.nbands
            if self.MP_pool is None:
                self.MP_pool = Pool(n_threads)
                conf.MEM_USAGE_DICT['percent'].append(dict(psutil.virtual_memory()._asdict())['percent'])
                conf.MEM_USAGE_DICT['used'].append(dict(psutil.virtual_memory()._asdict())['used']/1024.0**2)
            ## multi-thread reading by tile
            '''
            yoff = 0
            ysize = int(self.nrows / n_threads)
            params = []
            for i in range(n_threads):
                if i < n_threads - 1:
                    param = [yoff + i * ysize, ysize]
                else:
                    param = [yoff + i * ysize, self.nrows - i * ysize]
                params.append(param)
            fns = np.array([self.srcRasterfile]).repeat(n_threads)
            #print params, '\n', fns

            data = self.MP_pool.map(threadReadingByTile, params, fns)

            axis = 1 ## multiple covariates
            if len(np.array(data[0]).shape) == 2: ## single covariate
                axis = 0
            data = np.concatenate((np.concatenate(np.array(data[0:-1]), axis = axis), \
                                  np.array(data[-1])), axis = axis)
            '''

            ## multi-thread reading by band
            band_idx = range(1, n_threads + 1)
            fns = np.array([self.srcRasterfile]).repeat(n_threads)
            data = self.MP_pool.map(threadReadingByBand, band_idx, fns)
            data = np.stack(data, axis=0)
            #print data.shape
            conf.MEM_USAGE_DICT['percent'].append(dict(psutil.virtual_memory()._asdict())['percent'])
            conf.MEM_USAGE_DICT['used'].append(dict(psutil.virtual_memory()._asdict())['used']/1024.0**2)
            self.MP_pool.clear()
        else:
            #print '...single thread reading...'
            data = self.ds.ReadAsArray(xoff=0, yoff=0, xsize=None, ysize=None)

        ## nodatavalues
        data[data < self.nodata] = self.nodata
        conf.MEM_USAGE_DICT['percent'].append(dict(psutil.virtual_memory()._asdict())['percent'])
        conf.MEM_USAGE_DICT['used'].append(dict(psutil.virtual_memory()._asdict())['used']/1024.0**2)
        return data

    def readNextTile(self, xsize=None, ysize=None, multithread = conf.MULTITHREAD_READ):
        ## update xsize and ysize if needed
        ## PLEASE specify xsize, ysize ONLY ONCE (when reading the first tile)
        if xsize is not None: self.xsize = xsize
        if ysize is not None: self.ysize = ysize

        #print

        #N_BLOCK_X = int(math.ceil(self.ncols/self.xsize + 0.5))
        N_BLOCK_X = int(math.ceil(self.ncols*1.0/self.xsize))
        y = int(self.__N_TilesRead / N_BLOCK_X)
        x = self.__N_TilesRead - y * N_BLOCK_X

        #print self.ncols/self.xsize
        #print N_BLOCK_X
        #print self.__N_TilesRead

        #print x, y
        #if x > int(self.ncols/self.xsize) or y > int(self.nrows/self.ysize):
        #    return (None, self.xoff, self.yoff, 0, 0)

        #print self.__N_TilesRead, x, y,

        self.xoff = min(x * self.xsize, self.ncols)
        xsize = min(self.xsize, self.ncols - self.xoff)

        self.yoff = min(y * self.ysize, self.nrows)
        ysize = min(self.ysize, self.nrows - self.yoff)

        #if x > int(self.ncols/self.xsize) or y > int(self.nrows/self.ysize):
        if self.xoff == self.ncols or self.yoff == self.nrows:
            return (None, self.xoff, self.yoff, 0, 0)

        #print self.xoff, self.yoff, xsize, ysize
        data = None

        '''
        data_b1 = self.ds.GetRasterBand(1).ReadAsArray(xoff=self.xoff, yoff=self.yoff, win_xsize=xsize, win_ysize=ysize)
        ## values in this tile is all nodata
        if np.sum(data_b1 != self.nodata) == 0:
            print '... all values are nodata'
            self.__N_TilesRead += 1
            return (np.array([self.nodata]), self.xoff, self.yoff, xsize, ysize)
        print '... NOT all values are nodata'
        '''
        if multithread: ## multi-thread read
            def threadReading(param, srcRasterfile):
                ''' each thread reads param[2] of cols and param[3] of rows
                    starting at col, row at param[0], param[1]
                '''
                import gdal, gdalconst
                import numpy as np
                ds = gdal.Open(srcRasterfile, gdalconst.GA_ReadOnly)
                data = ds.ReadAsArray(xoff=param[0], yoff=param[1], xsize=param[2], ysize=param[3])
                #print param, data.shape
                return data

            def threadReadingByBand(i, param, rasterfile):
                ''' each thread reads a band, with tile dimension spec in param
                    using multiprocess pool
                '''
                #print param, rasterfile
                import gdal, gdalconst
                import numpy as np
                ds = gdal.Open(rasterfile, gdalconst.GA_ReadOnly)
                data = ds.GetRasterBand(i).ReadAsArray(xoff=param[0], yoff=param[1], win_xsize=param[2], win_ysize=param[3])
                #data = ds.ReadAsArray(xoff=param[0], yoff=param[1], xsize=param[2], ysize=param[3])
                #print param, data.shape
                return data


            # optimal for multi-thread reading by band
            n_threads = self.nbands# - 1
            if self.MP_pool is None:
                self.MP_pool = Pool(n_threads)

            ## multi-thread reading by tile
            '''
            _xoff, _yoff = self.xoff, self.yoff
            _ysize = int(ysize / n_threads)
            params = []
            for i in range(n_threads):
                if i < n_threads - 1:
                    param = [_xoff, _yoff + i * _ysize, xsize, _ysize]
                else:
                    param = [_xoff, _yoff + i * _ysize, xsize, ysize - i * _ysize]
                params.append(param)
            fns = np.array([self.srcRasterfile]).repeat(n_threads)
            #print params, '\n', fns

            data = self.MP_pool.map(threadReading, params, fns)
            axis = 1 ## multiple covariates
            if len(np.array(data[0]).shape) == 2: ## single covariate
                axis = 0
            data = np.concatenate((np.concatenate(np.array(data[0:-1]), axis = axis), \
                                  np.array(data[-1])), axis = axis)
            '''
            ## multi-thread reading by band
            params = []
            for i in range(n_threads):
                params.append([self.xoff, self.yoff, xsize, ysize])
            fns = np.array([self.srcRasterfile]).repeat(n_threads)
            band_idx = range(1, self.nbands + 1)
            #print params
            #print self.fileList
            data = self.MP_pool.map(threadReadingByBand, band_idx, params, fns)
            #print type(data)
            #data = self.MP_pool.map(threadReadingByBand, band_idx, params, self.fileList)
            #data.insert(0, data_b1)
            data = np.stack(data, axis=0)
            #print data.shape
            self.MP_pool.clear()

        else: ## single-thread read
            #print '...single thread reading...'
            data = self.ds.ReadAsArray(xoff=self.xoff, yoff=self.yoff, xsize=xsize, ysize=ysize)
        ## nodatavalues
        data[data < self.nodata] = self.nodata

        self.__N_TilesRead += 1

        return (data, self.xoff, self.yoff, xsize, ysize)

    def reset(self):
        ''' reset after reading tiles
        '''
        #self.xoff, self.yoff, self.xsize, self.ysize = 0, 0, 256, 256
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

def testReadWholeRaster():
    '''
    '''
    Dir = r'D:\OneDrive - University of Denver\Data\Pyclipsm\anhui'
    srcRasterfile = Dir + os.sep + 'covariates' + os.sep + 'covariates_90m.vrt'

    t0 = time.time()
    reader = tiledRasterReader(srcRasterfile)
    print 'constructing reader took', time.time()-t0, 's'

    t0 = time.time()
    dataS = reader.readWholeRaster(multithread=False)
    print 'SINGLE-thread reading took', time.time()-t0, 's'
    print dataS.shape
    dataS = None

    t0 = time.time()
    dataM = reader.readWholeRaster(multithread=True)
    print 'MULTI-thread reading took', time.time()-t0, 's'
    print dataM.shape
    dataM = None

    if dataS is not None and dataM is not None:
        print 'equality test:', np.sum(dataS != dataM)

def TestReadRasterTiles():
    '''Test tiled raster reader and writer
    '''
    Dir = r'D:\OneDrive - University of Denver\Data\Pyclipsm\anhui'
    t0 = time.time()
    vrtfn = Dir + os.sep + 'covariates' + os.sep + 'covariates_90m.vrt'
    outfn = Dir + os.sep + 'covariates' + os.sep + 'tiledOutput_90m.tif'

    ## raster reader
    reader = tiledRasterReader(vrtfn, xsize = conf.TILE_XSIZE, ysize = conf.TILE_YSIZE)

    print 'COVARIATE DATA SIZE:', reader.estimate_TotalSize_MB,'MB'
    print 'HOST MEM SIZE:', conf.HOST_MEM_SIZE / 1024.0**2, 'MB'
    host_mem_avail = dict(psutil.virtual_memory()._asdict())['available']
    print 'HOST MEM AVAIL:', host_mem_avail / 1024.0**2, 'MB'
    host_mem_quota = conf.HOST_MEM_PCT * host_mem_avail / 1024.0**2
    print 'HOST MEM QUOTA:', host_mem_quota, 'MB'

    idx = 0
    print 'tile size:', reader.xsize, 'x', reader.ysize
    print 'raster size:', reader.ncols, 'x', reader.nrows
    print 'COVARIATE TILE SIZE:', reader.estimateTileSize_MB(), 'MB'
    #sys.exit(0)
    FLAG_TEST_READWHOLERASTER = False
    FLAG_WRITETIF = True


    if FLAG_TEST_READWHOLERASTER:
        ## read in the whole raster
        t0 = time.time()
        data = reader.readWholeRaster()
        print 'read whole raster SINGLE thread took', time.time()-t0, 's'
        print data.shape


        if FLAG_WRITETIF:
            ## write out the whole raster
            writer = tiledRasterWriter(outfn, reader.nrows, reader.ncols, reader.geotransfrom, reader.projection, reader.nodata)
            if len(data.shape) == 2:
                writer.WriteWholeRaster(data)
            else:
                writer.WriteWholeRaster(data[idx])
            ## have to call this to FlushCache to disc
            writer.close()

        #data = None

        t0 = time.time()
        data2 = reader.readWholeRaster(multithread=True)
        print 'read whole raster MULTIPLE threads took', time.time()-t0, 's'

        if FLAG_WRITETIF:
            ## write out the whole raster
            writer = tiledRasterWriter(outfn.replace('.tif','_mt.tif'), reader.nrows, reader.ncols, reader.geotransfrom, reader.projection, reader.nodata)
            if len(data2.shape) == 2:
                writer.WriteWholeRaster(data2)
            else:
                writer.WriteWholeRaster(data2[idx])
            ## have to call this to FlushCache to disc
            writer.close()

        #data2 = None

        if data is not None and data2 is not None:
            print 'Read Whole Raster equality test:', np.sum(data != data2), '\n'

    #sys.exit(0)

    if FLAG_WRITETIF:
        ## writer for writing out tiles of raster
        writer = tiledRasterWriter(outfn.replace('.tif','_tile.tif'), reader.nrows, reader.ncols, reader.geotransfrom, reader.projection, reader.nodata)


    ## read in the first tile of raster
    reader_mt = tiledRasterReader(vrtfn, xsize = conf.TILE_XSIZE, ysize = conf.TILE_YSIZE)
    T_total = 0.0
    t0 = time.time()
    data, xoff, yoff, _xsize, _ysize = reader.readNextTile()
    T_total += time.time()-t0
    print 'single thread reading took', time.time()-t0, 's'

    T_total_mt = 0.0
    t0 = time.time()
    data_mt, xoff, yoff, _xsize, _ysize = reader_mt.readNextTile(multithread=True)
    T_total_mt += time.time()-t0
    print 'multi-thread reading took', time.time()-t0, 's'

    while data is not None and data_mt is not None:
        print 'tile', xoff, '/', reader.ncols, yoff, '/', reader.nrows#, data.shape, time.time()-t0, 's'
        print 'equality test:', np.sum(data != data_mt), '\n'

        if FLAG_WRITETIF:
            ## write out the current tile of raster
            t0 = time.time()
            if len(data.shape) == 1:
                writer.writeTile(np.ones((_ysize, _xsize))*reader.nodata, xoff, yoff)
            elif len(data.shape) == 2:
                writer.writeTile(data, xoff, yoff)
            else:
                writer.writeTile(data[idx], xoff, yoff)
            print 'writing took', time.time()-t0, 's\n'
        ## read in the next tile
        t0 = time.time()
        data, xoff, yoff, _xsize, _ysize = reader.readNextTile()
        T_total += time.time()-t0
        print 'single thread reading took', time.time()-t0, 's'

        t0 = time.time()
        data_mt, xoff, yoff, _xsize, _ysize = reader_mt.readNextTile(multithread=True)
        T_total_mt += time.time()-t0
        print 'multi-thread reading took', time.time()-t0, 's'

        #if data is not None and data_mt is not None:
        #    print 'equality test:', np.sum(data != data_mt), '\n'

    print 'read tiles SINGLE thread took', T_total, 's\n'
    print 'read tiles MULTIPLE threads took', T_total_mt, 's\n'

    if FLAG_WRITETIF:
        ## close raster writer - have to call this to FlushCache to disc
        writer.close()

    ## finally, close reader
    reader.close()
    reader_mt.close()

def main():
    ''' Place holder for test drivers
    '''
    testReadWholeRaster()
    #TestReadRasterTiles()

if __name__ == "__main__":
    T0 = time.time()
    main()
    print 'IN TOTAL IT TOOK', time.time() - T0, 's'
