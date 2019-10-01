# Author: Guiming Zhang
# Last update: August 8 2019

import numpy as np
import numpy.ma as ma
from matplotlib import gridspec
import csv, time, random, os, sys, math
import random
from sets import Set
import conf, raster, util, gaussian_kde

class Points:
    '''
    this is a class representing field samples
    each point has x, y coordinates and only one attribute
    '''
    ids = None # point ids
    xycoords = None # 2D array holding data values, including NoData values
    attributes = None # attribute values at point locations
    weights = None    # weights
    predictions = None
    size = None

    ### Added 09/08/2017, optional
    covariates_at_points = None # to hold evironmental variates values at points
             # avoid reading complete data layers to save memory (e.g., large study area)

    __header = None
    __sigTimestampe = 0 #signature timestamp, ms

    def __init__(self):
        self.__sigTimestampe = int(time.time()*1000) # in ms

    def readFromCSV(self, csvfn):
        ''' create points from a csv file (without covariates_at_points)
        '''
        try:
            f = open(csvfn, 'r')
            csvreader = csv.reader(f, delimiter=',')

            self.__header = next(csvreader, None)  # skip the headers
            #print self.__header

            data = []
            for row in csvreader:
                data.append(row)
            f.close()

            data = np.asarray(data, 'float')
            #print data

            self.ids = data[:, 0]
            self.xycoords = data[:, 1:3]
            self.attributes = data[:, 3]
            self.weights = data[:, 4]
            self.predictions = data[:, 5]

            ## read in covariates at points, if any
            N_COLS = data.shape[1]
            if N_COLS > 6:
                self.covariates_at_points = data[:,6:N_COLS]

            self.size = self.weights.size

            '''
            print np.shape(self.xycoords)
            print np.shape(self.attributes)
            print np.shape(self.weights)
            '''
        except Exception as e:
            raise

    def generateRandom(self, N, mask, val = None):
        ''' genrate N random points that is within the extent specified by mask (a raster)
            with uniform probability
        '''
        print 'generating random points...'
        try:
            xmin = mask.xllcorner
            xlen = mask.cellsize * mask.ncols
            ymin = mask.yllcorner
            ylen = mask.cellsize * mask.nrows
            vals = mask.getData()

            self.ids = range(0, N, 1)
            self.attributes = np.zeros(N)
            self.xycoords = np.zeros((N, 2))
            self.weights = np.ones(N)
            self.predictions = np.zeros(N)

            #'''
            idxs = np.random.choice(range(0, mask.getData().size), N)
            for i in range(N):
                print i
                x, y = mask.pos2XY(idxs[i])
                self.xycoords[i, 0] = x
                self.xycoords[i, 1] = y
                if val is None:
                    self.attributes[i] = mask.getData()[idxs[i]]
                else:
                    self.attributes[i] = val

            self.size = N
            self.__header = ['id', 'x', 'y', 'attribute', 'weight', 'prediction']
        except Exception as e:
            raise

    def generateBiasedRandom(self, mask, seedxy = None, N=10, decay = 1.0, dist_threshold = 0.1):
        ''' genrate N random points that is within the extent specified by mask (a raster)
            with decreasing probability away from the seed point
            TO BE IMPLEMENTED
        '''
        try:
            xmin = mask.xllcorner
            xlen = mask.cellsize * mask.ncols
            ymin = mask.yllcorner
            ylen = mask.cellsize * mask.nrows
            vals = mask.getData()

            self.ids = range(0, N, 1)
            self.attributes = np.zeros(N)
            self.xycoords = np.zeros((N, 2))
            self.weights = np.ones(N)
            self.predictions = np.zeros(N)

            ### Added 09/08/2017
            dmax = math.sqrt(xlen**2 + ylen**2)

            idxs = range(0, mask.getData().size)

            idxsA = [] # the key is to determine the selected ids, i.e., pos of pixels
            if seedxy is None: # no seed point provided
                idx_seed = random.sample(idxs, 1)[0]
                x, y = mask.pos2XY(idx_seed)
                seedxy = [x, y]

            cnt = 0
            while cnt < N:
                print '##in while loop - generateBiasedRandom()'
                ### generate 10 * N random points
                delta = dist_threshold * dmax
                xys = np.array([seedxy[0] + (np.random.uniform(size=20*N) - 0.5) * delta, seedxy[1] + (np.random.uniform(size=20*N) - 0.5) * delta]).T
                #print xys.shape, xys[0]
                dists = np.sqrt(np.sum((xys - np.array(seedxy))**2, axis=1))
                dists[dists == 0] = 0.01
                prob = 1.0 / dists**decay
                prob = prob / np.sum(prob)  #
                #print prob

                idxsA = np.random.choice(range(20*N), size=N, replace=False, p=prob)

                for i in range(len(idxsA)):
                    xy_tmp = xys[idxsA[i]]
                    pos = mask.xy2POS(xy_tmp[0], xy_tmp[1])
                    #print 'i = ', i
                    if pos != -1 and cnt < N:
                        self.xycoords[cnt] = xys[idxsA[i]]
                        self.attributes[cnt] = mask.getData()[pos]
                        #print 'cnt', cnt
                        cnt = cnt + 1

            self.size = N
            self.__header = ['id', 'x', 'y', 'attribute', 'weight', 'prediction']
        except Exception as e:
            raise

    def generateStratifiedRandom(self, N, mask):
        ''' genrate N stratified random points that is within the extent specified by mask (a raster)
        '''
        try:
            msrlevel = mask.getMsrLevel()
            if not (msrlevel == conf.MSR_LEVEL_NOMINAL or msrlevel == conf.MSR_LEVEL_ORDINAL):
                print 'strata raster must be nominal or ordinal'
                sys.exit(1)

            xmin = mask.xllcorner
            xlen = mask.cellsize * mask.ncols
            ymin = mask.yllcorner
            ylen = mask.cellsize * mask.nrows
            vals = mask.getData()
            uvals = np.unique(vals)

            proportions = []
            for val in uvals:
                proportions.append(np.sum(vals==val)*1.0/vals.size)

            self.ids = range(0, N, 1)
            self.attributes = np.zeros(N)
            self.xycoords = np.zeros((N, 2))
            self.weights = np.ones(N)
            self.predictions = np.zeros(N)

            counter = 0
            for i in range(uvals.size):
                Ni = int(N * proportions[i] + 0.5)
                idx = np.where(vals == uvals[i])[0]
                idx_chosen = np.random.choice(idx, size=Ni, replace=True)

                #print uvals[i], len(idx_chosen)

                for j in range(len(idx_chosen)):
                    if counter < N: # in case of round up error
                        x, y = mask.pos2XY(idx_chosen[j])
                        self.xycoords[counter, 0] = x + (random.random() - 0.499) * mask.cellsize
                        self.xycoords[counter, 1] = y + (random.random() - 0.499) * mask.cellsize
                        self.attributes[counter] = vals[idx_chosen[j]]
                        counter += 1
            while counter < N: # in case of round down error
                pos = np.random.choice(range(vals.size), size=1, replace=False)
                x, y = mask.pos2XY(pos)
                self.xycoords[counter, 0] = x + (random.random() - 0.499) * mask.cellsize
                self.xycoords[counter, 1] = y + (random.random() - 0.499) * mask.cellsize
                self.attributes[counter] = vals[pos]
                counter += 1

            print np.min(self.xycoords, axis = 0)
            #print self.xycoords.max()
            #print counter

            self.size = N
            self.__header = ['id', 'x', 'y', 'attribute', 'weight', 'prediction']

        except Exception as e:
            raise

    def generateRegular(self, grideSize, mask):
        ''' generate regular grid points (with random start)
            within the extent specified by mask (a raster)
        '''
        try:
            xmin = mask.xllcorner
            xmax = xmin + mask.cellsize * mask.ncols
            ymin = mask.yllcorner
            ymax = ymin + mask.cellsize * mask.nrows
            vals = mask.getData()

            attributes = []
            xycoords = [[],[]]
            weights = []

            # random starting point
            x0 = xmin + random.random() * (xmax - xmin)
            y0 = ymin + random.random() * (ymax - ymin)

            xs = [] # possible x coords of the sampled points
            x = x0
            while x < xmax:
                xs.append(x)
                x = x + grideSize
            x = x0 - grideSize
            while x > xmin:
                xs.append(x)
                x = x - grideSize

            ys = [] # possible x coords of the sampled points
            y = y0
            while y < ymax:
                ys.append(y)
                y = y + grideSize
            y = y0 - grideSize
            while y > ymin:
                ys.append(y)
                y = y - grideSize

            # combine x, y coords to locate grid points
            for x in xs:
                for y in ys:
                    pos = mask.xy2POS(x, y)
                    if pos != -1: # skip those NoData locations
                        xycoords[0].append(x)
                        xycoords[1].append(y)
                        attributes.append(vals[pos])
                        weights.append(1.0)
            self.xycoords = np.array(xycoords).T
            self.attributes = np.array(attributes)
            self.weights = np.array(weights)
            self.size = self.weights.size
            self.ids = range(0, self.size, 1)
            self.predictions = np.zeros(self.size)

            self.__header = ['id', 'x', 'y', 'attribute', 'weight', 'prediction']
        except Exception as e:
            raise

    def updateWeights(self, weights):
        ''' update weights of the points
        '''
        try:
            if np.shape(weights) != np.shape(self.weights):
                print 'weights dimension does not match'
                sys.exit(1)
            self.weights = np.copy(weights)
        except Exception as e:
            raise

    def copySelf(self):
        ''' copy points themselves
        '''
        try:
            points = Points()
            points.ids = np.copy(self.ids)
            points.xycoords = np.copy(self.xycoords)
            points.attributes = np.copy(self.attributes)
            points.weights = np.copy(self.weights)
            points.predictions = np.copy(self.predictions)
            points.__header = np.copy(self.__header)
            points.size = len(points.weights)

            ## copy covariates values at points, if any
            if self.covariates_at_points is not None:
                points.covariates_at_points = np.copy(self.covariates_at_points)

            return points

        except Exception as e:
            raise

    def writeCSV(self, csvfn):
        ''' write points to csv file
        '''
        try:
            f = open(csvfn, 'wb')
            writer = csv.writer(f)
            if self.covariates_at_points is not None and len(self.__header) == 6:
                for i in range(self.covariates_at_points.shape[1]):
                    self.__header.append('VAR'+str(i+1))
            writer.writerow(self.__header)
            data = np.concatenate((np.array([self.ids]).T, self.xycoords), axis = 1)
            data = np.concatenate((data, np.array([self.attributes]).T), axis = 1)
            data = np.concatenate((data, np.array([self.weights]).T), axis = 1)
            data = np.concatenate((data, np.array([self.predictions]).T), axis = 1)
            #print data.shape

            ## write covariates_at_points as well, if any
            if self.covariates_at_points is not None:
                data = np.concatenate((data, self.covariates_at_points), axis = 1)
            #print data.shape

            writer.writerows(data)
            f.close()

        except Exception as e:
            raise
