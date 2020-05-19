# Author: Guiming Zhang
# Last update: May 19 2020

import os, time, sys, psutil, math
import numpy as np
#import dill
from pathos.multiprocessing import ProcessingPool as Pool

import pyopencl as cl
import raster, points, util, conf
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

    __env_data_size = 0  ## size of 1d covariate data

    __single_cpu = False ## run predict_opencl() on single cpu core?


    def __init__(self, envrasters, soilsamples, uncthreshold = 1.0):
        self.__sigTimestampe = int(time.time()*1000) # in ms

        self.__envrasters = envrasters

        msr_codes = []
        for envraster in envrasters:
            msr_codes.append(envraster.msrInt)
        self.__msrInts = np.array(msr_codes)
        #print '\nMEASUREENT LEVELS:', self.__msrInts, '\n'

        self.__soilsamples = soilsamples
        self.__uncthreshold = uncthreshold

        self.__env_data_size = self.__envrasters[0].getData().nbytes * len(self.__envrasters)

        ## variables needed in predict_opencl_atom()
        self.__samples_stats_collected = False
        self.__samples_SD_evs = None
        self.__samples_X = None
        self.__sample_weights = None
        self.__sample_attributes = None
        self.__nrows_samples = None

    def __simLoc2SampleV0(self, loc_ev, sample_ev, evs, SD_evs):
            ''' compute similarity between a location to a sample
                evs: environmental variable values for the study area
                SD_evs: std of evs
                return: a similarity value
            '''
            try:
                M = SD_evs.size  # number of environmental variables

                sim = np.zeros(M)

                for i in range(M):

                    evi = loc_ev[i]
                    evj= sample_ev[i]
                    msrlevel = self.__envrasters[i].getMsrLevel

                    if msrlevel == conf.MSR_LEVEL_ORDINAL or msrlevel == conf.MSR_LEVEL_NOMINAL:
                        if evi == evj:
                            sim_i = 1.0
                        else:
                            sim_i = 0.0
                    else: # interval or ratio variables
                        SD_ev = SD_evs[i]
                        ev = evs[:,i]
                        SD_evj = np.sqrt(np.mean((ev - evj) ** 2))
                        sim_i = np.exp(-0.5 * (evi - evj) ** 2 / (SD_ev ** 2 / SD_evj) ** 2)

                    sim[i] = sim_i
                return np.min(sim) ## limiting factor
                #'''

            except Exception as e:
                raise

    ## faster alternative
    def __simLoc2Sample(self, loc_ev, sample_ev, REVS, SD_evs, AVG_evs, SUM_DIF_SQ_AVG):
            ''' compute similarity between a location to a sample
                evs: environmental variable values for the study area
                SD_evs: std of evs
                return: a similarity value
            '''
            try:
                M = SD_evs.size  # number of environmental variables
                sim = np.zeros(M)

                for i in range(M):
                    evi = loc_ev[i]
                    evj= sample_ev[i]

                    msrlevel = self.__envrasters[i].getMsrLevel()
                    if msrlevel == conf.MSR_LEVEL_NOMINAL or msrlevel == conf.MSR_LEVEL_ORDINAL:
                        if evi == evj:
                            sim_i = 1.0
                        else:
                            sim_i = 0.0
                    else: # interval or ratio variables
                        SD_ev = SD_evs[i]
                        delta = sample_ev[i] - AVG_evs[i]
                        tmp = SUM_DIF_SQ_AVG[i] + REVS * delta**2
                        SD_evj = np.sqrt(tmp/REVS)
                        sim_i = np.exp(-0.5 * (evi - evj) ** 2 / (SD_ev ** 2 / SD_evj) ** 2)

                    sim[i] = sim_i
                return np.min(sim) ## limiting factor

            except Exception as e:
                raise

    def __simLoc2SamplesV0(self, loc_ev, evs, SD_evs):
            ''' compute similarity between a location to N samples
                return: a vector of similarity values, each to a sample
            '''
            try:
                sample_evs = util.extractCovariatesAtPoints(self.__envrasters, self.__soilsamples)
                sample_evs = np.array(sample_evs).T

                # number of samples
                N = np.shape(sample_evs)[0]

                # similarity btw a loc to N samples
                sim = np.zeros(N)

                # compute similarities
                t0 = time.time()
                for i in range(N):
                    sim[i] = self.__simLoc2SampleV0(loc_ev, sample_evs[i], evs, SD_evs)
                conf.TIME_KEEPING_DICT['parts']['compute'].append(time.time()-t0)
                return sim

            except Exception as e:
                raise

     ## faster alternative
    def __simLoc2Samples(self, loc_ev, sample_evs, REVS, SD_evs, AVG_evs, SUM_DIF_SQ_AVG):
            ''' compute similarity between a location to N samples
                return: a vector of similarity values, each to a sample
            '''
            try:
                # number of samples
                N = np.shape(sample_evs)[0]

                # similarity btw a loc to N samples
                sim_s = np.zeros(N)

                t0 = time.time()
                # compute similarities
                for p in range(N):
                    ### INLINE, INSTEAD OF FUNCITON called
                    M = SD_evs.size  # number of environmental variables
                    sim = np.zeros(M)

                    for i in range(M):
                        evi = loc_ev[i]
                        evj= sample_evs[p][i]

                        msrlevel = self.__envrasters[i].getMsrLevel()
                        if msrlevel == conf.MSR_LEVEL_NOMINAL or msrlevel == conf.MSR_LEVEL_ORDINAL:
                            if evi == evj:
                                sim_i = 1.0
                            else:
                                sim_i = 0.0
                        else: # interval or ratio variables
                            SD_ev = SD_evs[i]
                            delta = sample_evs[p][i] - AVG_evs[i]
                            tmp = SUM_DIF_SQ_AVG[i] + REVS * delta**2
                            SD_evj = np.sqrt(tmp/REVS)
                            sim_i = np.exp(-0.5 * (evi - evj) ** 2 / (SD_ev ** 2 / SD_evj) ** 2)

                        sim[i] = sim_i
                    sim_s[p] = np.min(sim) ## limiting factor

                conf.TIME_KEEPING_DICT['parts']['compute'].append(time.time()-t0)
                return sim_s

            except Exception as e:
                raise

    def __simLocs2Samples(self, X, parallel = True, nprocess = conf.N_PROCESS):
        ''' compute similarity between locations to predict and samples
            return: a matrix of similarity values, each row is a location, each column is a sample
        '''
        ## this import is necessary [on Windows]:
        # http://stackoverflow.com/questions/28445373/python-import-numpy-as-np-from-outer-code-gets-lost-within-my-own-user-defined
        import numpy as np
        import raster, points, util, conf
        def simLoc2SamplesV0(loc_ev, datapkg): # this function is needed for parallel computing using multiprocessing
            import conf
            # unpack data in datapkg
            t0 = time.time()
            sample_evs = datapkg[0]
            evs = datapkg[1]
            SD_evs = datapkg[2]
            conf.TIME_KEEPING_DICT['parts']['data_transfer'].append(time.time()-t0)
            # number of environmental variables
            M = SD_evs.size
            # number of samples
            N = np.shape(sample_evs)[0]
            sim = np.zeros(N)
            t0 = time.time()
            for i in range(N): # for each sample
                sim0 = np.zeros(M)
                sample_ev = sample_evs[i]
                for j in range(M): # for each environmental variable
                    evi = loc_ev[j]
                    evj= sample_ev[j]
                    msrlevel = self.__envrasters[j].getMsrLevel()
                    if msrlevel == conf.MSR_LEVEL_NOMINAL or msrlevel == conf.MSR_LEVEL_ORDINAL:
                        if evi == evj:
                            sim_i = 1.0
                        else:
                            sim_i = 0.0
                    else:
                        SD_ev = SD_evs[j]
                        ev = evs[:,j]
                        SD_evj = np.sqrt(np.mean((ev - evj) ** 2))
                        sim_i = np.exp(-0.5 * (evi - evj) ** 2 / (SD_ev ** 2 / SD_evj) ** 2)
                    sim0[j] = sim_i
                sim[i] = np.min(sim0) ## limiting factor
            conf.TIME_KEEPING_DICT['parts']['compute'].append(time.time()-t0)
            return sim

        def simLoc2Samples(loc_ev, datapkg): # this function is needed for parallel computing using multiprocessing
            import conf ## IMPORTANT - makes **conf.MSR_LEVELS** visible
            # unpack data in datapkg
            t0 = time.time()
            sample_evs = datapkg[0]
            REVS = datapkg[1]
            SD_evs = datapkg[2]
            AVG_evs = datapkg[3]
            SUM_DIF_SQ_AVG = datapkg[4]
            # Guiming 3/31/2019
            MSRLEVES = datapkg[5]
            conf.TIME_KEEPING_DICT['parts']['data_transfer'].append(time.time()-t0)
            # number of environmental variables
            M = SD_evs.size
            # number of samples
            N = np.shape(sample_evs)[0]

            sim = np.zeros(N)
            t0 = time.time()
            for i in range(N): # for each sample
                sim0 = np.zeros(M)
                sample_ev = sample_evs[i]

                for j in range(M): # for each environmental variable
                    evi = loc_ev[j]
                    evj= sample_ev[j]
                    # Guiming 3/31/2019 - SAVES MEM, NO NEED TO DISPATCH self.__envrasters TO EACH THREAD
                    msrlevel = MSRLEVES[j]
                    ## this line below does not work without ** import conf ** at the begining of this function
                    if msrlevel == conf.MSR_LEVEL_NOMINAL or msrlevel == conf.MSR_LEVEL_ORDINAL:
                    #if msrlevel == 'nominal' or msrlevel == 'ordinal':
                        if evi == evj:
                            sim_i = 1.0
                        else:
                            sim_i = 0.0
                    else:
                        SD_ev = SD_evs[j]
                        delta = sample_ev[j] - AVG_evs[j]
                        tmp = SUM_DIF_SQ_AVG[j] + REVS * delta**2
                        SD_evj = np.sqrt(tmp/REVS)
                        sim_i = np.exp(-0.5 * (evi - evj) ** 2 / (SD_ev ** 2 / SD_evj) ** 2)

                    sim0[j] = sim_i
                sim[i] = np.min(sim0) ## limiting factor
            conf.TIME_KEEPING_DICT['parts']['compute'].append(time.time()-t0)
            return sim

        try:
            # do dimension match check here
            if np.shape(X)[1] != len(self.__envrasters):
                print 'dimension mismatch in computing similarity in iPSM'
                sys.exit(1)

            msr_levels = []
            if conf.NAIVE:
                evs = np.zeros((self.__envrasters[0].getData().size, len(self.__envrasters)))
            SD_evs = np.zeros(len(self.__envrasters))
            AVG_evs = np.zeros(len(self.__envrasters))
            for i in range(len(self.__envrasters)):
                if conf.NAIVE:
                    evs[:, i] = self.__envrasters[i].getData().T
                msr_levels.append(self.__envrasters[i].getMsrLevel())
                SD_evs[i] = self.__envrasters[i].std
                AVG_evs[i] = self.__envrasters[i].mean

            NROWS = np.shape(X)[0]

            REVS = self.__envrasters[0].getData().size
            SUM_DIF_SQ_AVG = REVS * SD_evs**2

            samples_evs = util.extractCovariatesAtPoints(self.__envrasters, self.__soilsamples)
            samples_evs = np.array(samples_evs).T

            if not parallel:
                sim = np.zeros((NROWS, self.__soilsamples.size))
                for i in range(NROWS):
                    if conf.NAIVE: ## naive implementaton
                        sim[i,:] = self.__simLoc2SamplesV0(X[i], evs, SD_evs)
                    else: ## with optimizations
                        sim[i,:] = self.__simLoc2Samples(X[i], samples_evs, REVS, SD_evs, AVG_evs, SUM_DIF_SQ_AVG)
            else:
                datapkg = []
                for i in range(NROWS):
                    if conf.NAIVE: ## naive implementaton
                        datapkg.append([samples_evs, evs, SD_evs])
                    else:
                        # Guiming 3/31/2019
                        datapkg.append([samples_evs, REVS, SD_evs, AVG_evs, SUM_DIF_SQ_AVG, msr_levels])

                #print 'n process', nprocess
                pool = Pool(nprocess)

                t0 = time.time()
                if conf.NAIVE: ## naive implementaton
                    sim = np.array(pool.map(simLoc2SamplesV0, X, datapkg))
                else:
                    sim = np.array(pool.map(simLoc2Samples, X, datapkg))
                conf.TIME_KEEPING_DICT['parts']['compute'].append(time.time()-t0)
                pool.clear()

            return sim

        except Exception as e:
            raise

    def predict(self, X = None, predict_class = False, parallel = True, nprocess = conf.N_PROCESS):
        ''' make prediction for every instance [cell] in X
            return: a raster correspoinding to envrasters, or a column vector corresponding to X
        '''
        try:
            if X is None: # if X is not provided, make prediction for the whole study area
                X = []
                for raster in self.__envrasters:
                    X.append(raster.getData())
                X = np.array(X).T#.astype(np.float32)
            K = np.shape(X)[0] # number of locations to make prediction

            # compute similarity matrix
            #print 'compute similarity matrix...'
            #t0 = time.time()
            simMatrix = self.__simLocs2Samples(X, parallel = parallel, nprocess = nprocess)
            #print 'computing similarity matrix took', 1000.0*(time.time()-t0), 'ms'
            #print 'simMatrix', simMatrix[0]

            #print 'prediction...'
            # now make prediction
            y_hat = np.zeros((K,2))
            N = self.__soilsamples.size # number of soil samples
            attributes = self.__soilsamples.attributes

            # sample weights
            weights = self.__soilsamples.weights
            weights = weights / np.max(weights)

            t0 = time.time()
            if predict_class: # predict class, nearest neighbor approach
                for i in range(K):
                    sim_row = simMatrix[i, :] * weights
                    sim_row = sim_row[sim_row >= 1.0 - self.__uncthreshold]
                    max_sim = sim_row.max()
                    y_hat[i, 0] = attributes[sim_row == max_sim][0]
                    y_hat[i, 1] = 1.0 - max_sim
            else: # predicit property, weighted average
                for i in range(K):
                    sim_row = simMatrix[i, :] * weights
                    threshold = 1.0 - self.__uncthreshold
                    tmp_row = sim_row[sim_row >= threshold]
                    y_hat[i, 0] = np.sum(attributes[sim_row >= threshold] * tmp_row)/tmp_row.sum()
                    y_hat[i, 1] = 1.0 - tmp_row.max()
            #print 'prediction took', 1000.0*(time.time()-t0), 'ms'
            conf.TIME_KEEPING_DICT['parts']['compute'].append(time.time()-t0)

            return y_hat

        except Exception as e:
            raise

    def predict_opencl_naive(self, X = None, predict_class = False, single_cpu = conf.SINGLE_CPU, opencl_config = conf.OPENCL_CONFIG):
        ''' PyOpenCL implementation of the iPSM approach
            return: a vector of predictions, eacn for a row in X
        '''
        try:
            t0 = time.time()
            ##### prepare data
            # covariates values over the whole study area
            r_evs = np.int32(self.__envrasters[0].getData().size)
            c_evs = np.int32(len(self.__envrasters))
            evs = np.zeros((r_evs, c_evs))
            SD_evs = np.zeros(c_evs)
            for i in range(len(self.__envrasters)):
                evs[:, i] = self.__envrasters[i].getData().T
                SD_evs[i] = self.__envrasters[i].std

            # standard deviation of each variable (over the whole study area)
            SD_evs = SD_evs.reshape(c_evs).astype(np.float32)
            #print SD_evs, SD_evs.shape
            evs = evs.reshape(r_evs*c_evs).astype(np.float32)
            #print evs, evs.shape, r_evs, c_evs

            # covariates values at prediction locations
            if X is None: # if X is not provided, make prediction for the whole study area
                r, c = r_evs, c_evs
                nrows_X = np.int32(r)
                ncols_X = np.int32(c)
                X = evs
            else:
                r, c = np.shape(X)
                nrows_X = np.int32(r)
                ncols_X = np.int32(c)
                X = X.reshape(nrows_X*ncols_X).astype(np.float32)

            #print X, X.shape, nrows_X, ncols_X

            MSRLEVES = self.__msrInts.reshape(c_evs).astype(np.int32)
            #print MSRLEVES, MSRLEVES.shape

            # covariates values at sample locations
            if self.__soilsamples.covariates_at_points is None:
                samples_X = util.extractCovariatesAtPoints(self.__envrasters, self.__soilsamples)
            else:
                samples_X = self.__soilsamples.covariates_at_points.T

            nrows_samples = np.int32(samples_X.shape[1])

            samples_X = np.array(samples_X).T.reshape(nrows_samples*c_evs).astype(np.float32)
            #print samples_X, samples_X.shape, nrows_samples, c_evs

            # sample weights
            sample_weights = self.__soilsamples.weights.reshape(nrows_samples).astype(np.float32)
            #print sample_weights, sample_weights.shape, nrows_samples

            # sample attributes
            sample_attributes = self.__soilsamples.attributes.reshape(nrows_samples).astype(np.float32)
            #print sample_attributes, sample_attributes.shape, nrows_samples

            # hold predictions for instances in X
            X_predictions = np.zeros(nrows_X).astype(np.float32)
            # hold prediction uncertainties for instances in X
            X_uncertainties = np.zeros(nrows_X).astype(np.float32)
            print 'preparation on HOST took', time.time() - t0, 's'

            ##### config computing platform and device
            for platform in cl.get_platforms():
                if platform.name == conf.OPENCL_CONFIG['Platform']:
                    PLATFORM = platform
                    # Print each device per-platform
                    for device in platform.get_devices():
                        if device.name == conf.OPENCL_CONFIG['Device']:
                            DEVICE = device
                            break
            print DEVICE.name, 'on', PLATFORM.name

            # opencl context
            ctx = cl.Context([DEVICE])
            # opencl command queue
            queue = cl.CommandQueue(ctx)

            ##### allocate memory space on device
            mf = cl.mem_flags
            t0 = time.time()
            evs_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=evs)
            SD_evs_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=SD_evs)
            X_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=X)
            MSRLEVES_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=MSRLEVES)
            sample_X_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=samples_X)
            sample_weights_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=sample_weights)
            sample_attributes_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=sample_attributes)
            X_predictions_g = cl.Buffer(ctx, mf.WRITE_ONLY, X_predictions.nbytes)
            X_uncertainties_g = cl.Buffer(ctx, mf.WRITE_ONLY, X_uncertainties.nbytes)
            queue.finish()
            t1 = time.time()
            conf.TIME_KEEPING_DICT['parts']['data_transfer'].append(t1-t0)
            print 'allocation and copy from HOST to DEVICE took', t1 - t0, 's'
            X = None

            ##### build opencl kernel from code in the file
            f = open(conf.iPSM_KERNEL_FN, 'r')
            fstr = "".join(f.readlines())
            fstr = fstr.replace("#define N_SAMPLES 100", "#define N_SAMPLES " + str(nrows_samples))
            prg = cl.Program(ctx, fstr).build()

            ##### opencl computation

            threshold = np.float32(self.__uncthreshold)

            if predict_class:
                mode = np.int32(1)
            else:
                mode = np.int32(0)

            if not single_cpu:
                t0 = time.time()
                completeEvent = \
                prg.iPSM_Predict_naive(queue, X_predictions.shape, None, r_evs, nrows_X, ncols_X, nrows_samples, mode, \
                                 threshold, MSRLEVES_g, evs_g, SD_evs_g, X_g, sample_X_g, sample_weights_g, sample_attributes_g, \
                                 X_predictions_g, X_uncertainties_g)
                queue.finish()
                #queue.flush()
                t1 = time.time()
                conf.TIME_KEEPING_DICT['parts']['compute'].append(t1-t0)
                print 'kernel took', t1 - t0, 's'

            else:
                print 'SINGLE_CPU iPSM.predict_opencl() called'
                t0 = time.time()
                completeEvent = \
                prg.iPSM_Predict_Sequential_naive(queue, (1,), (1,), r_evs, nrows_X, ncols_X, nrows_samples, mode, \
                                 threshold, MSRLEVES_g, evs_g, SD_evs_g, X_g, sample_X_g, sample_weights_g, sample_attributes_g, \
                                 X_predictions_g, X_uncertainties_g)
                queue.finish()
                t1 = time.time()
                conf.TIME_KEEPING_DICT['parts']['compute'].append(t1-t0)
                print 'kernel took', t1 - t0, 's'

            #### wait until completions
            events = [completeEvent]
            queue.finish()
            #queue.flush()
            print 'up to events finished kernel took', time.time() - t0, 's'

            ##### copy result data
            t0 = time.time()
            cl.enqueue_copy(queue, X_predictions, X_predictions_g, wait_for = events)#.wait())
            cl.enqueue_copy(queue, X_uncertainties, X_uncertainties_g)
            queue.finish()
            t1 = time.time()
            conf.TIME_KEEPING_DICT['parts']['data_transfer'].append(t1-t0)
            print 'copy from DEVICE to HOST took', t1 - t0, 's'
            y = np.vstack((X_predictions, X_uncertainties)).T
            #print y
            return y

        except Exception as e:
            raise

    def predict_opencl_atom(self, X = None, predict_class = False, single_cpu = conf.SINGLE_CPU, opencl_config = conf.OPENCL_CONFIG):
        ''' PyOpenCL implementation of the iPSM approach
            return: a vector of predictions, eacn for a row in X
        '''
        print 'predict_opencl_atom() was called'
        try:
            t0 = time.time()
            ##### prepare data
            # covariates values over the whole study area
            r_evs = np.int32(self.__envrasters[0].getData().size)
            c_evs = np.int32(len(self.__envrasters))

            Std_evs = np.zeros(len(self.__envrasters))
            AVG_evs = np.zeros(len(self.__envrasters))
            for i in range(len(self.__envrasters)):
                Std_evs[i] = self.__envrasters[i].std
                AVG_evs[i] = self.__envrasters[i].mean

            SD_evs = Std_evs.reshape(c_evs).astype(np.float32)

            # covariates values at prediction locations
            if X is None: # if X is not provided, make prediction for the whole study area
                X = []
                for raster in self.__envrasters:
                    X.append(raster.getData())
                X = np.array(X).T

            r, c = np.shape(X)
            nrows_X = np.int32(r)
            ncols_X = np.int32(c)

            X = X.reshape(nrows_X*ncols_X).astype(np.float32)

            MSRLEVES = self.__msrInts.reshape(c_evs).astype(np.int32)
            #print MSRLEVES, MSRLEVES.shape

            if not self.__samples_stats_collected:
                # covariates values at sample locations
                if self.__soilsamples.covariates_at_points is None:
                    samples_X = util.extractCovariatesAtPoints(self.__envrasters, self.__soilsamples)
                else:
                    samples_X = self.__soilsamples.covariates_at_points.T#[0:c_evs].T ## prone to bug
                nrows_samples = np.int32(samples_X.shape[1])
                self.__nrows_samples = nrows_samples

                samples_SD_evs = np.zeros((nrows_samples, c_evs))

                for i in range(nrows_samples):
                    delta = samples_X[:,i].T - AVG_evs
                    tmp = Std_evs**2  + delta**2
                    samples_SD_evs[i] = np.sqrt(tmp)

                #print '\nsamples_SD_evs:', samples_SD_evs, '\n'

                self.__samples_SD_evs = np.array(samples_SD_evs).reshape(nrows_samples*c_evs).astype(np.float32)

                self.__samples_X = np.array(samples_X).T.reshape(nrows_samples*c_evs).astype(np.float32)
                #print 'samples_X:', samples_X.shape, samples_X.min()

                # sample weights
                self.__sample_weights = self.__soilsamples.weights.reshape(nrows_samples).astype(np.float32)
                #print 'sample_weights:', sample_weights.shape, sample_weights.min()

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
            print DEVICE.name, 'on', PLATFORM.name
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
            t1 = time.time()
            conf.TIME_KEEPING_DICT['parts']['data_transfer'].append(t1-t0)
            print 'allocation and copy from HOST to DEVICE took', t1 - t0, 's'
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

            if not single_cpu:
                t0 = time.time()
                completeEvent = \
                prg.iPSM_Predict(queue, X_predictions.shape, None, nrows_X, ncols_X, self.__nrows_samples, mode, \
                                 threshold, MSRLEVES_g, samples_SD_evs_g, SD_evs_g, X_g, sample_X_g, sample_weights_g, sample_attributes_g, \
                                 X_predictions_g, X_uncertainties_g)
                queue.finish()
                t1 = time.time()
                conf.TIME_KEEPING_DICT['parts']['compute'].append(t1-t0)
                print 'kernel took', t1 - t0, 's'
                #print queue.finish()

            else:
                print 'SINGLE_CPU iPSM.predict_opencl() called'
                t0 = time.time()
                completeEvent = \
                prg.iPSM_Predict_Sequential(queue, (1,), (1,), nrows_X, ncols_X, self.__nrows_samples, mode, \
                                 threshold, MSRLEVES_g, samples_SD_evs_g, SD_evs_g, X_g, sample_X_g, sample_weights_g, sample_attributes_g, \
                                 X_predictions_g, X_uncertainties_g)
                queue.finish()
                t1 = time.time()
                conf.TIME_KEEPING_DICT['parts']['compute'].append(t1-t0)
                print 'kernel took', t1 - t0, 's'
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
            t1 = time.time()
            conf.TIME_KEEPING_DICT['parts']['data_transfer'].append(t1-t0)
            print 'copy from DEVICE to HOST took', t1 - t0, 's'
            y = np.vstack((X_predictions, X_uncertainties)).T
            #print y
            return y

        except Exception as e:
            raise

    def predict_opencl(self, X = None, predict_class = False, single_cpu = conf.SINGLE_CPU,  opencl_config = conf.OPENCL_CONFIG):
        ''' PyOpenCL implementation of the iPSM approach
            return: a vector of predictions, each for a row in X
        '''
        print 'predict_opencl() was called'
        try:
            ## predict PIECE BY PIECE to avoid blowing up the GPU memory

            print 'DEVICE MEM SIZE:', conf.DEVICE_MEM_SIZE / 1024.0**2, 'MB'
            device_mem_quota = conf.DEVICE_MEM_PCT * conf.DEVICE_MEM_SIZE
            if conf.DEVICE_TYPE == 'CPU':
                device_mem_avail = dict(psutil.virtual_memory()._asdict())['available']
                print 'DEVICE (CPU) MEM AVAIL:', device_mem_avail / 1024.0**2, 'MB'
                device_mem_quota = min(device_mem_quota, device_mem_avail *conf.DEVICE_MEM_PCT)
            ## predict PIECE BY PIECE to avoid blowing up the GPU memory
            print 'DEVICE MEM QUOTA:', device_mem_quota / 1024.0**2, 'MB'

            print 'ENV DATA SIZE:', self.__env_data_size / 1024.0**2, 'MB'
            if X is not None: print 'X SIZE:', X.nbytes / 1024.0**2, 'MB'
            print 'SINGLE_CPU:', single_cpu

            if X is None: ## predict map
                N_LOCS = self.__envrasters[0].getData().size
                X = []
                for raster in self.__envrasters:
                    X.append(raster.getData())
                X = np.array(X).T
            else: # X is not None
                N_LOCS = X.shape[0]

            y = np.zeros((N_LOCS, 2))

            ## CHUNK_SIZE SHOULD BE MULTIPLE OF DEVICE_MAX_WORK_ITEM_SIZES[0], e.g., x 1024
            if N_LOCS <= conf.DEVICE_MAX_WORK_ITEM_SIZES[0]:
                N_CHUNKS = 1
                conf.CL_CHUNK_SIZE = N_LOCS
            else:
                conf.CL_CHUNK_SIZE = conf.DEVICE_MAX_WORK_ITEM_SIZES[0] * int(math.floor(min(device_mem_quota, X.nbytes) / X[0].nbytes / conf.DEVICE_MAX_WORK_ITEM_SIZES[0]))
                N_CHUNKS = int(math.ceil(N_LOCS * 1.0 / conf.CL_CHUNK_SIZE))

            print 'N_LOCS:', N_LOCS, 'N_CHUNKS:', N_CHUNKS, 'CL_CHUNK_SIZE:', conf.CL_CHUNK_SIZE

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
