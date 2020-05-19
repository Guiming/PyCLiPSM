# Author: Guiming Zhang
# Last update: May 15 2020

## iPSM implementation accommodating covariates read in tiles
## i.e., covariate layers do not need to be loaded into the main memory entirely
## This is helpful if mapping over large areas at fine spatial resolution (size of the covariates exceeds main memory capability)
## ONLY SUPPORT THE OPTIMIZED OPENCL IMPLEMENTATIONS

import os, time, sys, psutil, math
import numpy as np

import pyopencl as cl
import gdalwrapper, raster, points, util, conf
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '0'

### a class implementing the iPSM approach to predictive mapping (Zhu et al 2015)
class iPSM:
    ''' a class implementing the iPSM approach to predictive mapping (Zhu et al 2015)
    '''

    __envrasters = None # a list of environmental raster data layers for the study area
    __msrInts = None   # the int code of measurement level of each layer
    __soilsamples = None # a points object
    __uncthreshold = 1.0 # uncertainty threshold below which prediction is made

    __sigTimestampe = 0 #signature timestamp, ms
    __single_cpu = False ## run predict_opencl() on single cpu core?


    def __init__(self, covariatesfn, soilsamplesfn, uncthreshold = 1.0, outfns = None):
        self.__sigTimestampe = int(time.time()*1000) # in ms

        self.__tileRasterReader = gdalwrapper.tiledRasterReader(covariatesfn, xsize = conf.TILE_XSIZE, ysize = conf.TILE_YSIZE)
        self.__msrInts = np.copy(self.__tileRasterReader.measurement_level_ints)

        soilsamples = points.Points()
        t0 = time.time()
        soilsamples.readFromCSV(soilsamplesfn)
        conf.TIME_KEEPING_DICT['parts']['read'].append(time.time()-t0)
        ## Extract covairate values at sample locations, if missing in the sample file
        if soilsamples.covariates_at_points is None:
            soilsamples.covariates_at_points = np.zeros((soilsamples.size, self.__tileRasterReader.nbands))
            for i in range(soilsamples.size):
                t0 = time.time()
                soilsamples.covariates_at_points[i] = self.__tileRasterReader.extractByXY(soilsamples.xycoords[i][0], soilsamples.xycoords[i][1]).flatten()
                conf.TIME_KEEPING_DICT['parts']['read'].append(time.time()-t0)
        self.__soilsamples = soilsamples
        self.__uncthreshold = uncthreshold

        if outfns is None:
            soilmapfn = soilsamplesfn.replace('.csv', '_soilmap.tif')
            uncertmapfn = soilsamplesfn.replace('.csv', '_uncertmap.tif')
            self.__outfns = [soilmapfn, uncertmapfn]
        else:
            self.__outfns = outfns

        ## variables needed in predict_opencl_atom()
        self.__samples_stats_collected = False
        self.__samples_SD_evs = None
        self.__samples_X = None
        self.__sample_weights = None
        self.__sample_attributes = None
        self.__nrows_samples = None

    def predict_opencl_atom(self, X, predict_class = False, single_cpu = conf.SINGLE_CPU, opencl_config = conf.OPENCL_CONFIG):
        ''' PyOpenCL implementation of the iPSM approach
            return: a vector of predictions, eacn for a row in X
        '''
        print 'predict_opencl_atom() was called'
        try:
            t0 = time.time()
            c_evs = np.int32(self.__tileRasterReader.nbands)

            # standard deviation of each variable (over the whole study area)
            Std_evs = self.__tileRasterReader.statistics[:,3]
            SD_evs = Std_evs.reshape(c_evs).astype(np.float32)

            r, c = np.shape(X)
            nrows_X = np.int32(r)
            ncols_X = np.int32(c)

            X = X.reshape(nrows_X*ncols_X).astype(np.float32)

            MSRLEVES = self.__tileRasterReader.measurement_level_ints.reshape(c_evs).astype(np.int32)

            if not self.__samples_stats_collected:
                samples_X = self.__soilsamples.covariates_at_points.T

                nrows_samples = np.int32(samples_X.shape[1])
                self.__nrows_samples = nrows_samples

                samples_SD_evs = np.zeros((nrows_samples, c_evs))
                AVG_evs = self.__tileRasterReader.statistics[:,2]

                for i in range(nrows_samples):
                    delta = samples_X[:,i].T - AVG_evs
                    tmp = Std_evs**2 + delta**2
                    samples_SD_evs[i] = np.sqrt(tmp)

                self.__samples_SD_evs = np.array(samples_SD_evs).reshape(nrows_samples*c_evs).astype(np.float32)
                self.__samples_X = np.array(samples_X).T.reshape(nrows_samples*c_evs).astype(np.float32)

                # sample weights
                self.__sample_weights = self.__soilsamples.weights.reshape(nrows_samples).astype(np.float32)

                # sample attributes
                self.__sample_attributes = self.__soilsamples.attributes.reshape(nrows_samples).astype(np.float32)
                self.__samples_stats_collected = True

            # hold predictions for instances in X
            X_predictions = np.zeros(nrows_X).astype(np.float32)
            # hold prediction uncertainties for instances in X
            X_uncertainties = np.zeros(nrows_X).astype(np.float32)
            print 'preparation on HOST took', time.time() - t0, 's'

            ##### config computing platform and device
            for platform in cl.get_platforms():
                #print platform.name
                if platform.name == conf.OPENCL_CONFIG['Platform']:
                    PLATFORM = platform
                    # Print each device per-platform
                    for device in platform.get_devices():
                        #print device.name
                        if device.name == conf.OPENCL_CONFIG['Device']:
                            DEVICE = device
                            break

            # opencl context
            ctx = cl.Context([DEVICE])
            # opencl command queue
            queue = cl.CommandQueue(ctx)

            ##### allocate memory space on device
            mf = cl.mem_flags
            t0 = time.time()
            #evs_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=evs)
            SD_evs_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=SD_evs)
            X_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=X)
            MSRLEVES_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=MSRLEVES)
            sample_X_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.__samples_X)

            ## added 09/06/2017
            samples_SD_evs_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.__samples_SD_evs)

            sample_weights_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.__sample_weights)
            sample_attributes_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=self.__sample_attributes)
            X_predictions_g = cl.Buffer(ctx, mf.WRITE_ONLY, X_predictions.nbytes)
            X_uncertainties_g = cl.Buffer(ctx, mf.WRITE_ONLY, X_uncertainties.nbytes)
            queue.finish()
            t1 = time.time()-t0
            conf.TIME_KEEPING_DICT['parts']['data_transfer'].append(t1)
            print 'allocate and copy from HOST to DEVICE took', t1, 's'
            X = None

            ##### build opencl kernel from code in the file
            f = open(conf.iPSM_KERNEL_FN, 'r')
            fstr = "".join(f.readlines())
            fstr = fstr.replace("#define N_SAMPLES 100", "#define N_SAMPLES " + str(self.__nrows_samples))
            prg = cl.Program(ctx, fstr).build()

            ##### opencl computation
            threshold = np.float32(self.__uncthreshold)

            if predict_class:
                mode = np.int32(1)
            else:
                mode = np.int32(0)

            print X_predictions.shape

            ## improved version, 09/06/2017
            if not single_cpu:
                t0 = time.time()
                completeEvent = \
                prg.iPSM_Predict(queue, X_predictions.shape, None, nrows_X, ncols_X, self.__nrows_samples, mode, \
                                 threshold, MSRLEVES_g, samples_SD_evs_g, SD_evs_g, X_g, sample_X_g, sample_weights_g, sample_attributes_g, \
                                 X_predictions_g, X_uncertainties_g)
                queue.finish()
                t1 = time.time() - t0
                conf.TIME_KEEPING_DICT['parts']['compute'].append(t1)
                print 'kernel took', t1, 's'
                #print queue.finish()


            ## added on Oct. 7, 2018 [sequential version - CPU]
            else:
                print 'SINGLE_CPU iPSM.predict_opencl() called'
                t0 = time.time()
                completeEvent = \
                prg.iPSM_Predict_Sequential(queue, (1,), (1,), nrows_X, ncols_X, self.__nrows_samples, mode, \
                                 threshold, MSRLEVES_g, samples_SD_evs_g, SD_evs_g, X_g, sample_X_g, sample_weights_g, sample_attributes_g, \
                                 X_predictions_g, X_uncertainties_g)
                queue.finish()
                t1 = time.time() - t0
                conf.TIME_KEEPING_DICT['parts']['compute'].append(t1)
                print 'kernel took', t1, 's'
                #print queue.finish()

            #### wait until completions
            events = [completeEvent]
            queue.finish()
            print 'up to events finished kernel took', time.time() - t0, 's'
            #print queue.finish()

            ##### copy result data
            t0 = time.time()
            cl.enqueue_copy(queue, X_predictions, X_predictions_g, wait_for = events)#.wait()
            #print queue.finish()
            cl.enqueue_copy(queue, X_uncertainties, X_uncertainties_g)
            queue.finish()
            t1 = time.time() - t0
            conf.TIME_KEEPING_DICT['parts']['data_transfer'].append(t1)
            print 'copy from DEVICE to HOST took', t1, 's'
            y = np.vstack((X_predictions, X_uncertainties)).T
            #print y
            return y

        except Exception as e:
            raise

    def predict_opencl_tile(self, X, predict_class = False, single_cpu = conf.SINGLE_CPU,  opencl_config = conf.OPENCL_CONFIG):
        ''' PyOpenCL implementation of the iPSM approach
            return: a vector of predictions, each for a row in X
        '''
        print 'predict_opencl_tile() was called'
        try:
            print 'DEVICE MEM SIZE:', conf.DEVICE_MEM_SIZE / 1024.0**2, 'MB'
            device_mem_quota = conf.DEVICE_MEM_PCT * conf.DEVICE_MEM_SIZE
            if conf.DEVICE_TYPE == 'CPU':
                device_mem_avail = dict(psutil.virtual_memory()._asdict())['available']
                print 'DEVICE (CPU) MEM AVAIL:', device_mem_avail / 1024.0**2, 'MB'
                device_mem_quota = min(device_mem_quota, device_mem_avail *conf.DEVICE_MEM_PCT)
            ## predict PIECE BY PIECE to avoid blowing up the GPU memory
            print 'DEVICE MEM QUOTA:', device_mem_quota / 1024.0**2, 'MB'
            print 'X SIZE:', X.nbytes / 1024.0**2, 'MB'
            print ''

            N_LOCS = X.shape[0]

            y = np.zeros((N_LOCS, 2))

            ## CHUNK_SIZE SHOULD BE MULTIPLE OF DEVICE_MAX_WORK_ITEM_SIZES[0], e.g., x 1024
            if N_LOCS <= conf.DEVICE_MAX_WORK_ITEM_SIZES[0]:
                N_CHUNKS = 1
                conf.CL_CHUNK_SIZE = N_LOCS
            else:
                conf.CL_CHUNK_SIZE = conf.DEVICE_MAX_WORK_ITEM_SIZES[0] * int(math.floor(min(device_mem_quota, X.nbytes) / X[0].nbytes / conf.DEVICE_MAX_WORK_ITEM_SIZES[0]))
                N_CHUNKS = int(math.ceil(N_LOCS * 1.0 / conf.CL_CHUNK_SIZE))

            print 'N_CHUNKS:', N_CHUNKS, 'CL_CHUNK_SIZE:', conf.CL_CHUNK_SIZE, '(# of prediction locations)'

            n_accum_locs = 0
            counter = 1
            while n_accum_locs < N_LOCS:
                lower_idx = n_accum_locs
                upper_idx = min(n_accum_locs + conf.CL_CHUNK_SIZE, N_LOCS)
                X_chunk = X[lower_idx: upper_idx]

                print 'X_chunk.shape:', X_chunk.shape, 'X_chunk.nbytes:', X_chunk.nbytes / 1024.0**2, 'MB'

                t0 = time.time()
                y_chunk = self.predict_opencl_atom(X_chunk, predict_class, single_cpu, opencl_config)
                print 'ipsm.predict_opencl() MAP on chunk', counter, 'out of ', N_CHUNKS, ' took', time.time() - t0, 's\n'

                y[lower_idx: upper_idx] = y_chunk

                n_accum_locs += conf.CL_CHUNK_SIZE
                counter += 1
            return y

        except Exception as e:
            raise

    def predict_opencl(self, predict_class = False, single_cpu = conf.SINGLE_CPU,  opencl_config = conf.OPENCL_CONFIG):
        ''' PyOpenCL implementation of the iPSM approach
            return: a vector of predictions, each for a row in X
        '''
        print 'predict_opencl() was called'
        try:
            conf.HOST_MEM_SIZE / 1024.0**2, 'MB'
            print 'HOST MEM SIZE:', conf.HOST_MEM_SIZE / 1024.0**2, 'MB'
            host_mem_avail = dict(psutil.virtual_memory()._asdict())['available']
            print 'HOST MEM AVAIL:', host_mem_avail / 1024.0**2, 'MB'

            host_mem_quota = conf.HOST_MEM_PCT * host_mem_avail / 1024.0**2
            ## read in covariates tile by tile to avoid blowing up host memory

            print 'HOST MEM QUOTA:', host_mem_quota, 'MB'
            print 'COVARIATE DATA SIZE (EST.):', self.__tileRasterReader.estimate_TotalSize_MB, 'MB'
            #print 'SINGLE_CPU:', single_cpu
            print ''


            if not conf.TILE_READ and self.__tileRasterReader.estimate_TotalSize_MB <= host_mem_quota:
                template_raster = raster.Raster()
                template_data = None
                #print 'started reading in ENTIRE covariates into host memory...'
                t0 = time.time()
                data = self.__tileRasterReader.readWholeRaster()
                t1 = time.time()-t0
                conf.TIME_KEEPING_DICT['parts']['read'].append(t1)
                print 'done reading in ENTIRE covariates took', t1, 's'
                X = []
                if len(data.shape) == 2: # single covariate
                    template_data = data
                    tmp_data = data.flatten()
                    X.append(tmp_data[tmp_data != self.__tileRasterReader.nodata])
                else: # 2+ covariates
                    template_data = data[0]
                    for i in range(data.shape[0]):
                        tmp_data = data[i].flatten()
                        X.append(tmp_data[tmp_data != self.__tileRasterReader.nodata])
                X = np.array(X).T
                y = self.predict_opencl_tile(X, predict_class, single_cpu, opencl_config)


                template_raster.createRaster(template_data, xoff=0, yoff=0, \
                                            geotransform = self.__tileRasterReader.geotransfrom, \
                                            projection = self.__tileRasterReader.projection, \
                                            nodata = self.__tileRasterReader.nodata)
                ## write predicted soil map
                template_raster.updateRasterData(y[:,0])
                t0 = time.time()
                template_raster.writeRasterGDAL(self.__outfns[0])
                conf.TIME_KEEPING_DICT['parts']['write'].append(time.time()-t0)
                ## write prediction uncertainty map
                template_raster.updateRasterData(y[:,1])
                t1 = time.time()
                template_raster.writeRasterGDAL(self.__outfns[1])
                conf.TIME_KEEPING_DICT['parts']['write'].append(time.time()-t1)
                print 'done writing result took', time.time()-t0, 's\n'

            else: ## covariates cannot fit in host memory
                ## read in covariates tile by tile
                if conf.TILE_XSIZE is None or conf.TILE_YSIZE is None:
                    #if conf.MULTITHREAD_READ:# and conf.DEVICE_TYPE == 'CPU':
                    host_mem_quota /= 4
                    factor = int(host_mem_quota / self.__tileRasterReader.estimateTileSize_MB() * (self.__tileRasterReader.ysize/self.__tileRasterReader.block_ysize_base))
                    print factor, self.__tileRasterReader.block_ysize_base, \
                          self.__tileRasterReader.block_ysize_base * factor
                    self.__tileRasterReader.ysize = min(self.__tileRasterReader.block_ysize_base * factor, self.__tileRasterReader.nrows)
                print 'tile size:', self.__tileRasterReader.xsize, 'x', self.__tileRasterReader.ysize, \
                        self.__tileRasterReader.estimateTileSize_MB(), 'MB'
                print 'raster size:', self.__tileRasterReader.ncols, 'x', self.__tileRasterReader.nrows

                ## writer for writing out tiles of predicted soil map
                soilmapwriter = gdalwrapper.tiledRasterWriter(self.__outfns[0], \
                                                      self.__tileRasterReader.nrows, \
                                                      self.__tileRasterReader.ncols, \
                                                      self.__tileRasterReader.geotransfrom, \
                                                      self.__tileRasterReader.projection, \
                                                      self.__tileRasterReader.nodata)
                ## writer for writing out tiles of uncertainty map
                uncertmapwriter = gdalwrapper.tiledRasterWriter(self.__outfns[1], \
                                                      self.__tileRasterReader.nrows, \
                                                      self.__tileRasterReader.ncols, \
                                                      self.__tileRasterReader.geotransfrom, \
                                                      self.__tileRasterReader.projection, \
                                                      self.__tileRasterReader.nodata)
                ## prediction tile by tile
                template_raster = raster.Raster()
                template_data = None
                X = []

                t0 = time.time()
                data, xoff, yoff, xsize, ysize = self.__tileRasterReader.readNextTile()
                t1 = time.time()
                conf.TIME_KEEPING_DICT['parts']['read'].append(t1 - t0)
                print 'done reading in tile', self.__tileRasterReader.xoff, '/', self.__tileRasterReader.ncols, \
                                             self.__tileRasterReader.yoff, '/', self.__tileRasterReader.nrows,\
                                             'took', t1 - t0,'s\n'
                while data is not None:
                    if len(data.shape) == 1:
                        template_data = np.ones((ysize, xsize)) * self.__tileRasterReader.nodata
                    elif len(data.shape) == 2: # single covariate
                        template_data = data
                        tmp_data = data.flatten()
                        X.append(tmp_data[tmp_data != self.__tileRasterReader.nodata])
                    else: # 2+ covariates
                        template_data = data[0]
                        for i in range(data.shape[0]):
                            tmp_data = data[i].flatten()
                            X.append(tmp_data[tmp_data != self.__tileRasterReader.nodata])
                    X = np.array(X).T

                    y = None
                    ## prediction
                    if X.size > 0:
                        y = self.predict_opencl_tile(X, predict_class, single_cpu, opencl_config)

                    ##############################
                    t0 = time.time()
                    template_raster.createRaster(template_data, xoff=xoff, yoff=yoff, \
                                                geotransform = self.__tileRasterReader.geotransfrom, \
                                                projection = self.__tileRasterReader.projection, \
                                                nodata = self.__tileRasterReader.nodata)
                    ## write tile of predicted soil map
                    if X.size > 0: template_raster.updateRasterData(y[:,0])
                    t1 = time.time()
                    soilmapwriter.writeTile(template_raster.getData2D(), xoff, yoff)
                    conf.TIME_KEEPING_DICT['parts']['write'].append(time.time() - t1)

                    ## write tile of prediction uncertainty map
                    if X.size > 0: template_raster.updateRasterData(y[:,1])
                    t2 = time.time()
                    uncertmapwriter.writeTile(template_raster.getData2D(), xoff, yoff)
                    conf.TIME_KEEPING_DICT['parts']['write'].append(time.time() - t2)
                    print 'done writing out tile took', time.time()-t0,'s\n'
                    #################################

                    # reset X
                    X = []
                    t0 = time.time()
                    # read in the next tile
                    data, xoff, yoff, xsize, ysize = self.__tileRasterReader.readNextTile()
                    t1 = time.time()
                    conf.TIME_KEEPING_DICT['parts']['read'].append(t1 - t0)
                    print 'done reading in tile', self.__tileRasterReader.xoff, '/', self.__tileRasterReader.ncols, \
                                                 self.__tileRasterReader.yoff, '/', self.__tileRasterReader.nrows,\
                                                 'took', t1 - t0,'s\n'

                ## have to call this to FlushCache() to disc
                t0 = time.time()
                soilmapwriter.close()
                uncertmapwriter.close()
                conf.TIME_KEEPING_DICT['parts']['write'].append(time.time() - t0)

        except Exception as e:
            raise
