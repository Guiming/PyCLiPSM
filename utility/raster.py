# Author: Guiming Zhang
# Last update: August 8 2019

import numpy as np
import numpy.ma as ma
import os, time, sys
import matplotlib as mpl
mpl.use('Qt4Agg') # https://stackoverflow.com/questions/53529044/importerror-no-module-named-pyqt5
import matplotlib.pyplot as plt
from matplotlib import gridspec

import gdal, gdalconst, osr

import conf, util, gaussian_kde

class Raster:
    '''
    this is a class representing a raster GIS data layer
    '''
    __data2D = None # 2D array holding data values, including NoData values
    __data1D = None # serialized (row-wise) 1D array holding non-NoData values
    __coords1D = None # N by 2 array holding the (x,y) coordinates of data points in __data1D, needed by regression kriging
    __prjString = None # projection
    __sigTimestampe = 0 #signature timestamp, ms
    __measurement_level = None # nominal, ordinal, interval, ratio, count
    msrInt = None # 0, 1, 2, 3, 4
    filename = None
    density = None

    density_sample = None
    density_sample_weighted = None

    min = None
    max = None
    mean = None
    std = None

    def __init__(self, msrlevel = conf.MSR_LEVEL_RATIO):
        self.__sigTimestampe = int(time.time()*1000) # in ms
        self.__measurement_level = msrlevel
        self.filename = "NA"

        for key in conf.MSR_LEVELS:
            if conf.MSR_LEVELS[key] == msrlevel:
                self.msrInt = int(key)

    def __serialize2Dto1D(self):
        ''' private member function
            serialize raster data from 2d [including NoData values] to 1d [excluding NoData values]
        '''
        try:
            '''
            self.__data1D = []
            # serialize non-NoData values
            for row in self.__data2D:
                for val in row:
                    if val != self.nodatavalue:
                        self.__data1D.append(val)
            self.__data1D = np.array(self.__data1D)
            '''
            tmp = self.__data2D.flatten()
            self.__data1D = tmp[tmp != self.nodatavalue]

        except Exception as e:
            raise

    def getCoords(self):
        ''' N by 2 array holding the (x,y) coordinates of data points in __data1D,
            needed by regression kriging
        '''
        try:
            if self.__coords1D is None:
                self.__coords1D = []
                for i in range(self.nrows):
                    for j in range(self.ncols):
                        val = self.__data2D[i, j]
                        if val != self.nodatavalue:
                            x, y = self.rc2XY(i, j)
                            self.__coords1D.append([x, y])
                self.__coords1D = np.array(self.__coords1D)

            return self.__coords1D

        except Exception as e:
            raise

    def readRasterGDAL(self, fn, pdf = False):
        ''' Use GDAL for reading raster
        '''
        # this allows GDAL to throw Python Exceptions
        gdal.UseExceptions()
        print 'started reading', fn
        t0 = time.time()
        self.filename = os.path.basename(fn)

        try:
            # open dataset
            ds = gdal.Open(fn, gdalconst.GA_ReadOnly)
        except RuntimeError, e:
            print e
            sys.exit(1)

        '''
        rasterSRS = osr.SpatialReference()
        rasterSRS.ImportFromWkt(ds.GetProjection())
        self.__prjString = rasterSRS.ExportToWkt()
        '''
        self.__prjString = ds.GetProjection()


        #print self.__prjString
        gtrsfm = ds.GetGeoTransform()

        self.xllcorner = gtrsfm[0]

        self.cellsize = gtrsfm[1]

        try:
            band_num = 1
            srcband = ds.GetRasterBand(band_num)
        except RuntimeError, e:
            # for example, try GetRasterBand(10)
            print 'Band ( %i ) not found' % band_num
            print e
            sys.exit(1)

        #print 'band.getstatistics():', srcband.GetStatistics(0,1)
        self.min, self.max, self.mean, self.std = srcband.GetStatistics(0,1)

        self.nodatavalue = srcband.GetNoDataValue()
        self.__data2D = srcband.ReadAsArray()

        self.nrows, self.ncols = self.__data2D.shape
        self.yllcorner = gtrsfm[3] + gtrsfm[5] * self.nrows

        self.__serialize2Dto1D()

        # to save space?
        #self.__data2D = None

        # close dataset
        ds = None

        if pdf:
            self.__computePopulationDistribution()

        print 'done reading took', time.time()-t0, 's'

    def getData(self):
        ''' return a deep copy of the serialized 1d data
        '''
        return np.copy(self.__data1D)

    def getData2D(self):
        ''' return a deep copy of the 2d data
        '''
        return np.copy(self.__data2D)

    def __computePopulationDistribution(self):
        ''' compute frequency distributions histogram for NOMINAL/ORDINAL or pdf for INTERVAL/RATIO
        '''
        print 'computePopulationDistribution() called'
        xmin = self.min
        xmax = self.max

        if self.__measurement_level in [conf.MSR_LEVEL_NOMINAL, conf.MSR_LEVEL_ORDINAL]:
            y, x = np.histogram(self.__data1D, range = (xmin-0.5, xmax+0.5), bins = int(xmax-xmin)+1, density = True)
            #print xmin, xmax
            #print x
            #print y
        else:
            x = np.linspace(xmin, xmax, conf.N_INTERVALS)
            if self.__data1D.size < 10000:
                gkde = gaussian_kde.gaussian_kde(self.__data1D)
                y = gkde.evaluate(x)
            else:
                #print '# of cells exceeds 100000. Randomly select 100000 to estimate population distribution'
                gkde = gaussian_kde.gaussian_kde(np.random.choice(self.__data1D, size=10000, replace=False))
                y = gkde.evaluate(x)
            #print y
        self.density = y

    def computeSampleDistribution(self, points):
        ''' compute sample frequency distributions histogram for NOMINAL/ORDINAL or pdf for INTERVAL/RATIO
        '''
        print 'computeSampleDistribution() called'

        xmin = self.min
        xmax = self.max
        #print xmin, xmax

        vals = util.extractCovariatesAtPoints([self], points)[0]
        if self.__measurement_level in [conf.MSR_LEVEL_NOMINAL, conf.MSR_LEVEL_ORDINAL]:
            y1, x = np.histogram(vals, range = (xmin-0.5, xmax+0.5), bins = int(xmax-xmin)+1, density = True)
            #print np.unique(vals)
            #print x, y1
            if np.std(points.weights) != 0: # unequal weights
                y2, x = np.histogram(vals, weights = points.weights, bins = int(xmax-xmin)+1, density = True)
            else:
                y2 = np.copy(y1)
        else:
            x = np.linspace(xmin, xmax, conf.N_INTERVALS)

            gkde = gaussian_kde.gaussian_kde(vals)
            y1 = gkde.evaluate(x)

            if np.std(points.weights) != 0:  # unequal weights
                gkde = gaussian_kde.gaussian_kde(vals, weights=points.weights)
                y2 = gkde.evaluate(x)
            else:
                y2 = np.copy(y1)

        self.density_sample = y1
        self.density_sample_weighted = y2

    def updateRasterData(self, data1d):
        ''' update raster data by passing a 1d array, exclusing NoData values
        '''
        # check dimension
        dim = np.shape(data1d)
        if dim[0] > self.nrows * self.ncols:
            print 'cannot deserialize 1D to 2D, too many data'
            sys.exit(1)
        try:
            '''
            idx = 0
            for row in range(self.nrows):
                for col in range(self.ncols):
                    val = self.__data2D[row][col]
                    if val != self.nodatavalue:
                        self.__data2D[row][col] = data1d[idx]
                        idx += 1
            '''
            self.__data2D[self.__data2D != self.nodatavalue] = data1d
            self.__data1D = np.copy(data1d)


            self.__computeStatistics()

            if self.density is not None: self.__computePopulationDistribution()

        except Exception as e:
            raise

    def updateRasterData2D(self, data2d):
        ''' update raster data by passing a 2d array, including NoData values
        '''
        # check dimension
        dim = np.shape(data2d)
        if dim[0] != self.nrows or dim[1] != self.ncols:
            print 'dimension of the 2D array does not match dimension of the raster'
            sys.exit(1)
        try:
            self.__data2D = np.copy(data2d)
            self.__serialize2Dto1D()

            self.__computeStatistics()

            if self.density is not None: self.__computePopulationDistribution()

        except Exception as e:
            raise

    def __computeStatistics(self):
        if self.__data1D.size > 2:
            self.min = self.__data1D.min()
            self.max = self.__data1D.max()
            self.mean = self.__data1D.mean()
            self.std = self.__data1D.std()

    def getMsrLevel(self):
        ''' return measurement level
        '''
        return self.__measurement_level

    def setMsrLevel(self, msrlevel):
        ''' set measurement level
        '''
        self.__measurement_level = msrlevel

    def copySelf(self):
        ''' deep copy a raster object
        '''
        try:
            raster = Raster()
            raster.ncols = self.ncols
            raster.nrows = self.nrows
            raster.xllcorner = self.xllcorner
            raster.yllcorner = self.yllcorner
            raster.cellsize = self.cellsize
            raster.nodatavalue = self.nodatavalue

            raster.min = self.min
            raster.max = self.max
            raster.mean = self.mean
            raster.std = self.std

            raster.__data2D = np.copy(self.__data2D)
            raster.__data1D = np.copy(self.__data1D)

            raster.filename = self.filename

            if self.__prjString is not None:
                raster.__prjString = self.__prjString #np.copy(self.__prjString)

            return raster
        except Exception as e:
            raise

    def writeRasterGDAL(self, fn = None):
        '''write raster using gdal
        '''
        # only write geotiff
        if fn is None:
            fn = self.filename + '.tif'
        if fn[-4:] != '.tif':
            fn += 'tif'

        # this allows GDAL to throw Python Exceptions
        gdal.UseExceptions()
        #print 'started writing', fn
        #t0 = time.time()
        #print self.__prjString
        try:
            driver = gdal.GetDriverByName('GTiff')
            outRaster = driver.Create(fn, self.ncols, self.nrows, 1, gdal.GDT_Float32, options = [ 'COMPRESS=LZW', 'BIGTIFF=YES' ])
            outRaster.SetGeoTransform((self.xllcorner, self.cellsize, 0, self.cellsize * self.nrows + self.yllcorner, 0, -1 * self.cellsize))
            outband = outRaster.GetRasterBand(1)
            outband.SetNoDataValue(self.nodatavalue)
            outband.WriteArray(self.getData2D())

            if np.sum(self.getData2D() != self.nodatavalue) > 2:
                stats = outband.GetStatistics(0, 1)
                outband.SetStatistics(stats[0], stats[1], stats[2], stats[3])

            outRasterSRS = osr.SpatialReference()
            outRasterSRS.ImportFromWkt(self.__prjString)
            outRaster.SetProjection(outRasterSRS.ExportToWkt())
            outband.FlushCache()

        except RuntimeError, e:
            print e
            sys.exit(1)

        #print 'writing ', fn, 'took', time.time()-t0, 's'

    def printInfo(self):
        ''' print out basic info.
        '''
        print self.filename
        print '------HEADER----------------------'
        print 'ncols', self.ncols
        print 'nrows', self.nrows
        print 'xllcorner', self.xllcorner
        print 'yllcorner', self.yllcorner
        print 'cellsize', self.cellsize
        print 'nodatavalue', self.nodatavalue

        print '------STATS----------------------'
        print 'measurement_level', self.__measurement_level, self.msrInt
        '''
        print 'min', np.min(self.__data1D)
        print 'max', np.max(self.__data1D)
        print 'mean', np.mean(self.__data1D)
        print 'std', np.std(self.__data1D)
        '''
        print 'min', self.min
        print 'max', self.max
        print 'mean', self.mean
        print 'std', self.std

        '''
        print '------DATA----------------------'
        print '2D', np.shape(self.__data2D)
        print self.__data2D
        print '1D', np.shape(self.__data1D)
        print self.__data1D
        '''
        print '\n'

    def xy2RC(self, x, y):
        ''' convert x, y coordinates to row, col
        '''
        row = self.nrows - 1 - int((y - self.yllcorner) / self.cellsize)
        col = int((x - self.xllcorner) / self.cellsize)

        if (row >= 0 and row < self.nrows) and (col >= 0 and col < self.ncols):
            return row, col
        else:
            if conf.DEBUG_FLAG: print '(x, y) out of bound'
            return -1, -1

    def rc2XY(self, row, col):
        ''' convert row, col to x, y coordinates
        '''
        if row < 0 or row >= self.nrows or col < 0 or col >= self.ncols:
            if conf.DEBUG_FLAG: print '(row, col) out of bound'
            return -1, -1
        y = self.yllcorner + (self.nrows - 0.5 - row) * self.cellsize
        x = self.xllcorner + (col + 0.5) * self.cellsize
        return x, y

    ## enhanced version
    def rc2POS(self, row, col):
        ''' convert row, col to index in the 1d array
        '''
        pos = -1
        if row >= self.nrows or col >= self.ncols or row < 0 or col < 0: # updated on 5/15/2018
            if conf.DEBUG_FLAG: print '(row, col) out of bound'
            return pos
        if self.__data2D[row][col] == self.nodatavalue: # NoData at (row, col)
            if conf.DEBUG_FLAG: print 'NoData at (row, col)'
            return pos

        pos = np.sum(self.__data2D[0:row]!=self.nodatavalue)
        pos += np.sum(self.__data2D[row][0:col+1]!=self.nodatavalue)
        #for j in range(col + 1): # bug fixed
        #    if self.__data2D[row][j] != self.nodatavalue:
        #        pos += 1
        return pos-1

    def pos2RC(self, pos):
        ''' convert index in the 1d array to row, col
        '''
        idx = -1
        if pos > len(self.__data1D) or pos < 0: # out of bound
            if conf.DEBUG_FLAG: print 'pos out of bound'
            return idx
        for row in range(self.nrows):
            for col in range(self.ncols):
                if self.__data2D[row][col] != self.nodatavalue:
                    idx += 1
                    if idx == pos:
                        return row, col

    def xy2POS(self, x, y):
        ''' convert x, y to index in the 1d array
        '''
        r, c = self.xy2RC(x, y)
        return self.rc2POS(r, c)

    def pos2XY(self, pos):
        ''' convert x, y to index in the 1d array
        '''
        r, c = self.pos2RC(pos)
        return self.rc2XY(r, c)

    def createRaster(self, data, xoff, yoff, geotransform, projection, nodata=-9999.0, filename='out.tif'):
        '''create raster from a tile read from a larger raster
            relative location is represented by xoff (col), yoff (row)
        '''
        self.filename = filename
        self.__prjString = projection

        self.nrows, self.ncols = data.shape
        self.cellsize = geotransform[1]

        self.xllcorner = geotransform[0] + self.cellsize * xoff
        self.yllcorner = geotransform[3] - self.cellsize * (yoff + self.nrows)

        self.nodatavalue = nodata
        self.__data2D = np.copy(data)

        self.__serialize2Dto1D()
        self.__computeStatistics()

def main():
    ''' TEST DRIVER
    '''
    '''
    cellsize = 10
    ascDir = r'D:\OneDrive - University of Denver\Data\Pyclipsm\heshan\asc'
    asciifn = 'elevation_' + str(cellsize) + '.asc'
    tiffn = asciifn.replace('.asc','.tif')
    '''
    '''
    rst = Raster()
    rst.readRasterGDAL(ascDir + os.sep + asciifn)
    rst.printInfo()
    rst.writeRasterGDAL(ascDir + os.sep + tiffn)
    '''
    '''
    rst = Raster()
    rst.readRasterGDAL('D:/OneDrive - University of Denver/Data/Pyclipsm/anhui/covariates/90m/acc10_n_std.tif')
    rst.printInfo()
    #rst.writeRasterGDAL(ascDir + os.sep + r'tif\test2.tif')
    '''
    Dir = r'D:\OneDrive - University of Denver\Data\Pyclipsm\anhui\covariates\10m'
    rst = Raster()
    rst.readRasterGDAL(Dir + os.sep + 'geo_ma.tif')
    rst.xllcorner = 301603.878
    rst.yllcorner = 3253219.08
    rst.printInfo()
    rst.writeRasterGDAL(Dir + os.sep + 'geo_ma1.tif')

    rst = Raster()
    rst.readRasterGDAL(Dir + os.sep + 'ndvi_std.tif')
    rst.xllcorner = 301603.878
    rst.yllcorner = 3253219.08
    rst.printInfo()
    rst.writeRasterGDAL(Dir + os.sep + 'ndvi_std1.tif')


if __name__ == "__main__":
    T0 = time.time()
    main()
    print 'IN TOTAL IT TOOK', time.time() - T0, 's\n\n'
