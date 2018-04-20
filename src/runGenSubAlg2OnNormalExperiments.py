# Run experiments discussed in Section 8.3.2 in Bjarkason (2018),
# testing gen. subspace iteration Algorithm 2 when using it to
# approximate normal matrices JT*J. Also compare Algorithm 2 with
# the pinched and prolonged sketching methods.
# Coded by: Elvar K. Bjarkason (2018)

import scipy as sp
from scipy.sparse import csr_matrix
from RSVDmethods import *
from oversamplingMethods import *
from genTestMatrices import *
from math import floor
import matplotlib
from matplotlib import pyplot as plt
import os
import multiprocessing
from multiprocessing import Pool
import time

# Get Spectral norm error of rank-p approximation of normal matrix:
def StandardSpectralNorm(A, sk2, Vk):
    norm2 = sp.linalg.norm(sp.dot(A.T, A) - Vk.dot( sp.dot( sp.diag(sk2) , Vk.transpose() ) ) , ord=2, axis=None, keepdims=False )
    return norm2
   
# Get Spectral norm error for optimal rank-p approximation of normal matrix:   
def optSpectarlNormErrRankp(A, p):
    # Find spectral norm error of optimal rank-p approximation:
    sfullp1 = rank_p_SVD(A, p+1)[1]
    return sfullp1[-1]**2.
        
def main(A, ktrunc, matName, subfolder, inneriter, OversVec, viewMax):
    print 'Tests using matrix : ', matName
    time0 = time.clock()
    
    # Evaluate Spectral norm difference between JT*J and optimal rank-p approximation:
    SpecOptRankP = optSpectarlNormErrRankp(A, ktrunc)
    
    ViewVec = range(2, viewMax + 1)
    # Store mean errors for plots:
    MeanErrOnJ      = sp.zeros(len(ViewVec))
    MeanErrOnJT     = sp.zeros(len(ViewVec))
    MeanErrNystrom  = sp.zeros(len(ViewVec))
    MeanErrPinched  = sp.zeros(len(ViewVec))
    
    for ell in OversVec:
        print ktrunc, ell
        ell = int(ell)
        for indx, views in enumerate(ViewVec):
            ErrsOnJ     = sp.zeros(inneriter)
            ErrsOnJT    = sp.zeros(inneriter)
            ErrsNystrom = sp.zeros(inneriter)
            ErrsPinched = sp.zeros(inneriter)
            for i in range(0,inneriter):
                # Apply gen (half) subspace iteration method to A:
                [sk, Uk, Vk] = randGenHalfSubIter(A, None, ktrunc, ell, views)
                # Find Spectral norm error:
                SpecErr = StandardSpectralNorm(A, sk**2., Vk)
                # Store relative Spectral error:
                ErrsOnJ[i] = sp.absolute(SpecErr/SpecOptRankP  - 1.)
                
                # Apply gen (half) subspace iteration method to AT:
                [sk, Vk, Uk] = randGenHalfSubIter(A.transpose(), None, ktrunc, ell, views)
                # Find Spectral norm error:
                SpecErr = StandardSpectralNorm(A, sk**2., Vk)
                # Store relative Spectral error:
                ErrsOnJT[i] = sp.absolute(SpecErr/SpecOptRankP  - 1.)
                
                # Apply Nystrom approach:
                [sk2, Vk] = nystromOnNormal(A, ktrunc, ell, views)
                # Find Spectral norm error:
                SpecErr = StandardSpectralNorm(A, sk2, Vk)
                # Store relative Spectral error:
                ErrsNystrom[i] = sp.absolute(SpecErr/SpecOptRankP  - 1.)
                
                # Apply Pinched approach:
                [sk2, Vk] = pinchedOnNormal(A, ktrunc, ell, views)
                # Find Spectral norm error:
                SpecErr = StandardSpectralNorm(A, sk2, Vk)
                # Store relative Spectral error:
                ErrsPinched[i] = sp.absolute(SpecErr/SpecOptRankP  - 1.)
    
            
            # Store mean values:
            MeanErrOnJ[indx]  = sp.mean(ErrsOnJ)
            MeanErrOnJT[indx] = sp.mean(ErrsOnJT)
            MeanErrNystrom[indx] = sp.mean(ErrsNystrom)
            MeanErrPinched[indx] = sp.mean(ErrsPinched)
        
        # Save mean values and Tvec:
        sp.save(subfolder + '/' + 'ViewVec.npy', ViewVec)
        sp.save(subfolder + '/' + 'MeanSpecErrOnJ_'     + str(ell) + 'over.npy', MeanErrOnJ)
        sp.save(subfolder + '/' + 'MeanSpecErrOnJT_'    + str(ell) + 'over.npy', MeanErrOnJT)
        sp.save(subfolder + '/' + 'MeanSpecErrNystrom_' + str(ell) + 'over.npy', MeanErrNystrom)
        sp.save(subfolder + '/' + 'MeanSpecErrPinched_' + str(ell) + 'over.npy', MeanErrPinched)
        print 'Time for ', matName, 'Rank ', ktrunc, 'Over ', ell , ' :', time.clock() - time0
    
def runTestForMat(inputs):
    [rank, matName, MainSubfolder, inneriter, OversVec, N]  = inputs
    
    # Get specified test matrix:
    if matName == 'lowRankMedNoiseR10':
        A = sp.load('TestMatrices/lowRankMedNoiseR10.npy')
    elif matName == 'lowRankHiNoiseR10':
        A = sp.load('TestMatrices/lowRankHiNoiseR10.npy')
    elif matName == 'polySlowR10':
        A = matrixPolyDecay(N, 1., R=10)
    elif matName == 'polyFastR10':
        A = matrixPolyDecay(N, 2., R=10)
    elif matName == 'expSlowR10':
        A = matrixExpDecay(N, 0.25, R=10)
    elif matName == 'expFastR10':
        A = matrixExpDecay(N, 1., R=10)
    elif matName == 'SDlarge':
####        A = sp.load('TestMatrices/SD_6135obs_10years_24kParams.npy')
        # For use the following as a surrogate for large Jacobian (SD):
        A = sp.diag(sp.load('TestMatrices/singvalsSDlarge.npy')[0:1000])
    
    # Create folder for saving results:
    matName = matName + '_Rank' + str(rank)
    subfolder = MainSubfolder + matName
    newpath = os.getcwd() + '/' + subfolder
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    
    # Run subspace iteration tests for test matrix:
    temptime = time.clock()
    main(A, rank, matName, subfolder, inneriter, OversVec, viewMax=8)
    print matName, 'Time spent on Test', time.clock() - temptime
    
if __name__ == "__main__":
    multiprocessing.freeze_support()
    # Create folder for saving results:
    MainSubfolder = 'Results/GenSubOnNormal/'
    newpath = os.getcwd() + '/' + MainSubfolder
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    
    # Run tests for generating Figure 3:
    # Names of test matrices:
    mats = ['lowRankMedNoiseR10', 'lowRankHiNoiseR10', 'polySlowR10', 'polyFastR10']
    # Ranks for the approximations:
    ranks = [10]
    # Number of Monte Carlo runs used to estimate average or expected performance:
    inneriter = 500
    
    N = 1000
    tstart = time.clock()
    # Run for ranks in parallel:
    OversVec = [5, 10, 50]
    for rank in ranks:
        print OversVec
        iterableInput = [ [rank, matName, MainSubfolder, inneriter, OversVec, N]  for matName in mats  ]
        pool = Pool(processes=2)
        pool.map(runTestForMat, iterableInput)
        pool.terminate()
    print 'Time taken : ', time.clock() - tstart
    
    
    # Run tests for generating Figure 4:
    # Names of test matrices:
    mats = ['SDlarge']
    # Ranks for the approximations:
    ranks = [1, 10, 50]
    # Number of Monte Carlo runs used to estimate average or expected performance:
    inneriter = 500
    
    # Run subspace iteration tests:
    N = 1000   # size of square test matrices
    tstart = time.clock()
    # Run for ranks in parallel:
    OversVec = [5, 10, 50]
    for rank in ranks:
        print OversVec
        iterableInput = [ [rank, matName, MainSubfolder, inneriter, OversVec, N]  for matName in mats  ]
        pool = Pool(processes=2)
        pool.map(runTestForMat, iterableInput)
        pool.terminate()
    print 'Time taken : ', time.clock() - tstart
    






