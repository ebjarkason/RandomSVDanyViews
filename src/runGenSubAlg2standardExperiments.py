# Run experiments discussed in Section 8.3.1 in Bjarkason (2018),
# testing gen. subspace iteration Algorithm 2.
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

# Get Spectral and Frobenius norm errors of rank-p approximation:
def SpecFrobErrs(A, sk, Uk, Vk):
    Specerr = sp.linalg.norm(A - Uk.dot( sp.dot( sp.diag(sk) , Vk.transpose() ) ) , ord=2, axis=None, keepdims=False )
    Froberr = sp.linalg.norm(A - Uk.dot( sp.dot( sp.diag(sk) , Vk.transpose() ) ) , ord='fro', axis=None, keepdims=False )
    return Specerr, Froberr

# Get Spectral and Frobenius norm errors for optimal rank-p approximation:    
def optSpecFrobErrsRankp(A, p):
    # Find accurate rank-p TSVD:
    [Ufullp, sfullp, Vfullp] = rank_p_SVD(A, p)
    # Evaluate Spec and Frob norm differences between A and optimal rank-p approximation:
    SpecOptRankP, FrobOptRankP = SpecFrobErrs(A, sfullp, Ufullp, Vfullp)
    return SpecOptRankP, FrobOptRankP
    
        
def main(A, ktrunc, matName, subfolder, inneriter, OversVec, viewMax):
    print 'Tests using matrix : ', matName
    time0 = time.clock()
    
    # Evaluate Spectral and Frobenius norm difference between A and optimal rank-p approximation:
    SpecOptRankP, FrobOptRankP = optSpecFrobErrsRankp(A, ktrunc)
    
    ViewVec = range(2, viewMax + 1)
    # Store mean errors for plots:
    SpecMeanErrOnJ      = sp.zeros(len(ViewVec))
    FrobMeanErrOnJ      = sp.zeros(len(ViewVec))
    
    for ell in OversVec:
        print ktrunc, ell
        ell = int(ell)
        for indx, views in enumerate(ViewVec):
            SpecErrsOnJ     = sp.zeros(inneriter)
            SpecErrsOnJT    = sp.zeros(inneriter)
            FrobErrsOnJ     = sp.zeros(inneriter)
            FrobErrsOnJT    = sp.zeros(inneriter)
            for i in range(0,inneriter):
                # Apply gen (half) subspace iteration method to A:
                [sk, Uk, Vk] = randGenHalfSubIter(A, None, ktrunc, ell, views)
                # Find Spectral and Frobenius norm errors:
                [SpecErr, FrobErr] = SpecFrobErrs(A, sk, Uk, Vk)
                # Store relative Spectral error:
                SpecErrsOnJ[i] = sp.absolute(SpecErr/SpecOptRankP  - 1.)
                # Store relative Frob error:
                FrobErrsOnJ[i] = sp.absolute(FrobErr/FrobOptRankP  - 1.)             
            
            # Store mean values:
            SpecMeanErrOnJ[indx]  = sp.mean(SpecErrsOnJ)
            FrobMeanErrOnJ[indx]  = sp.mean(FrobErrsOnJ)
        
        # Save mean values and ViewVec:
        sp.save(subfolder + '/' + 'ViewVec.npy', ViewVec)
        sp.save(subfolder + '/' + 'MeanSpecErrOnJ_'     + str(ell) + 'over.npy', SpecMeanErrOnJ)
        sp.save(subfolder + '/' + 'MeanFrobErrOnJ_'     + str(ell) + 'over.npy', FrobMeanErrOnJ)
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
        # For GitHub use the following as a surrogate for large Jacobian (SD):
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
    MainSubfolder = 'Results/GenSubError/'
    newpath = os.getcwd() + '/' + MainSubfolder
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    
    # Names of test matrices:
    mats = ['SDlarge', 'lowRankMedNoiseR10', 'lowRankHiNoiseR10', 'polySlowR10', 'polyFastR10', 'expSlowR10']
    # Ranks for the approximations:
    ranks = [10]
    # Number of Monte Carlo runs used to estimate average or expected performance:
    inneriter = 50
    
    # Run subspace iteration tests:
    N = 1000  # size of square test matrices
    tstart = time.clock()
    # Run for ranks in parallel:
    OversVec = [10]
    for rank in ranks:
        print OversVec
        iterableInput = [ [rank, matName, MainSubfolder, inneriter, OversVec, N]  for matName in mats  ]
        pool = Pool(processes=2)
        pool.map(runTestForMat, iterableInput)
        pool.terminate()
    print 'Time taken : ', time.clock() - tstart
    






