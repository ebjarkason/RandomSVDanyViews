# Code for running randomized 1-view experiments
#
# These tests are the same type of tests used by Tropp et al (2017), 
# "PRACTICAL SKETCHING ALGORITHMS FOR LOW-RANK MATRIX APPROXIMATION",
# for comparing the performance of various 1-view methods.
#
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
    
# Get Frobenius norm error of rank-p approximation:
def FrobNormError(A, sk, Uk, Vk):
    Froberr = sp.linalg.norm(A - Uk.dot( sp.dot( sp.diag(sk) , Vk.transpose() ) ) , ord='fro', axis=None, keepdims=False )
    return Froberr

# Get Frobenius norm error for optimal rank-p approximation:    
def optFrobErrRankp(A, p):
    # Find accurate rank-p TSVD:
    [Ufullp, sfullp, Vfullp] = rank_p_SVD(A, p)
    # Evaluate Frobenius norm difference between A and optimal rank-p approximation:
    FrobOptRankP = FrobNormError(A, sfullp, Ufullp, Vfullp)
    return FrobOptRankP
    
def runStandard1viewTroppTests(A, ktrunc, matName, subfolder, inneriter, Tvec, FrobOptRankP, RunOracle):    
    # Store mean errors for plots:
    MeanErrORACLE   = sp.zeros(len(Tvec))
    MeanErrBEST     = sp.zeros(len(Tvec))
    MeanErrFlat     = sp.zeros(len(Tvec))
    MeanErrMedium   = sp.zeros(len(Tvec))
    MeanErrRapid    = sp.zeros(len(Tvec))
    
    for Tindx, T in enumerate(Tvec):
        print ktrunc, T
        T = int(T)
        ErrsFlat     = sp.zeros(inneriter) + 1.e99
        ErrsMedium   = sp.zeros(inneriter) + 1.e99
        ErrsRapid    = sp.zeros(inneriter) + 1.e99
        # For storing "Oracle" performance as used by Tropp et al. (2017), i.e. 
        # find the lowest error ErrMin[i] of any pair (ell1, ell2) over a Monte Carlo iteration i,
        # then find average of ErrMin:
        ErrsORACLE   = sp.zeros(inneriter) + 1.e99
        # For storing "best" performance, i.e. find mean error for best fixed (ell1, ell2) pair:
        ErrsBEST = sp.zeros((int(floor(T/2)) - ktrunc + 1, inneriter)) + 1.e99
        
        # Oversampling parameters using schemes proposed by Tropp et al. (2017)
        ell1Flat, ell2Flat = overparamsFlatSpectrum(ktrunc, T)
        ell1Med, ell2Med = overparamsModerateDecay(ktrunc, T)
        ell1Rapid, ell2Rapid = overparamsRapidDecay(ktrunc, T)
        for i in range(0,inneriter):
            if RunOracle:
                ErrMin = 1.e99
                for ell1 in range(0, int(floor(T/2)) - ktrunc + 1):
                    ell2 = T - ell1 - 2*ktrunc
                    [sk1view, Uk1view, Vk1view] = rand1viewTropp(A, A.transpose(), ktrunc, ell1, ell2, ell1, SketchOrth=False)
                    # Find Frob norm error:
                    Frob1view = FrobNormError(A, sk1view, Uk1view, Vk1view)
                    # Store relative Frob error:
                    RelErr = sp.absolute(Frob1view/FrobOptRankP  - 1.)
                    # Store for finding "best" error:
                    ErrsBEST[ell1, i] = RelErr
                    if RelErr < ErrMin:
                        ErrMin = RelErr + 0.
                    if (ell1==ell1Flat) and (ell2==ell2Flat):
                        ErrsFlat[i] = RelErr
                    if (ell1==ell1Med) and (ell2==ell2Med):
                        ErrsMedium[i] = RelErr
                    if (ell1==ell1Rapid) and (ell2==ell2Rapid):
                        ErrsRapid[i] = RelErr
                ErrsORACLE[i] = ErrMin
            else:
                [sk1view, Uk1view, Vk1view] = rand1viewTropp(A, A.transpose(), ktrunc, ell1Flat, ell2Flat, ell1Flat, SketchOrth=False)
                # Find Frob norm error:
                Frob1view = FrobNormError(A, sk1view, Uk1view, Vk1view)
                # Store relative Frob error:
                ErrsFlat[i] = sp.absolute(Frob1view/FrobOptRankP  - 1.)
                
                [sk1view, Uk1view, Vk1view] = rand1viewTropp(A, A.transpose(), ktrunc, ell1Med, ell2Med, ell1Med, SketchOrth=False)
                # Find Frob norm error:
                Frob1view = FrobNormError(A, sk1view, Uk1view, Vk1view)
                # Store relative Frob error:
                ErrsMedium[i] = sp.absolute(Frob1view/FrobOptRankP  - 1.)
                
                [sk1view, Uk1view, Vk1view] = rand1viewTropp(A, A.transpose(), ktrunc, ell1Rapid, ell1Rapid, ell1Rapid, SketchOrth=False)
                # Find Frob norm error:
                Frob1view = FrobNormError(A, sk1view, Uk1view, Vk1view)
                # Store relative Frob error:
                ErrsRapid[i] = sp.absolute(Frob1view/FrobOptRankP  - 1.)
                    
        # Store mean values:
        MeanErrFlat[Tindx]          = sp.mean(ErrsFlat)
        MeanErrMedium[Tindx]        = sp.mean(ErrsMedium)
        MeanErrRapid[Tindx]         = sp.mean(ErrsRapid)
        # Oracle error as used by Tropp et al. (2017):
        MeanErrORACLE[Tindx]        = sp.mean(ErrsORACLE) # same as using sp.mean(sp.amin(ErrsBEST, axis=0))
        # "Best" error:
        MeanErrBEST[Tindx]    = sp.amin(sp.mean(ErrsBEST, axis=1))
    
    # Save mean values:
    if RunOracle:
        sp.save(subfolder + '/' + 'TroppMeanFrobErrStandardORACLE.npy', MeanErrORACLE)
        sp.save(subfolder + '/' + 'TroppMeanFrobErrStandardBEST.npy', MeanErrBEST)
    sp.save(subfolder + '/' + 'TroppMeanFrobErrFlat.npy', MeanErrFlat)
    sp.save(subfolder + '/' + 'TroppMeanFrobErrMedium.npy', MeanErrMedium)
    sp.save(subfolder + '/' + 'TroppMeanFrobErrRapid.npy', MeanErrRapid)
        
def runLcut1viewTroppTests(A, ktrunc, matName, subfolder, inneriter, Tvec, FrobOptRankP, RunOracle):    
    # Store mean errors for plots:
    MeanErrEll1Ell2Equal                    = sp.zeros(len(Tvec))
    MeanErrEll1Ell2EqualEllCutHalf          = sp.zeros(len(Tvec))
    MeanErrEll1Ell2EqualEllCutQuarter       = sp.zeros(len(Tvec))
    MeanErrEll1Ell2EqualEllCutThreeFourths  = sp.zeros(len(Tvec))
    MeanErrEll1Ell2EqualEllCutORACLE        = sp.zeros(len(Tvec))
    MeanErrEll1Ell2EqualEllCutBEST          = sp.zeros(len(Tvec))
    MeanErrEll1Ell2EqualEllCutMinVar        = sp.zeros(len(Tvec))
    
    for Tindx, T in enumerate(Tvec):
        print ktrunc, T
        T = int(T)
        ErrsEll1Ell2Equal                     = sp.zeros(inneriter) + 1.e99
        ErrsEll1Ell2EqualEllCutHalf           = sp.zeros(inneriter) + 1.e99
        ErrsEll1Ell2EqualEllCutQuarter        = sp.zeros(inneriter) + 1.e99
        ErrsEll1Ell2EqualEllCutThreeFourths   = sp.zeros(inneriter) + 1.e99
        ErrsEll1Ell2EqualEllCutMinVar         = sp.zeros(inneriter) + 1.e99
        # For storing "Oracle" performance as used by Tropp et al. (2017), i.e. 
        # find the lowest error ErrMin[i] of any (ellCut) over a Monte Carlo iteration i,
        # then find average of ErrMin:
        ErrsEll1Ell2EqualEllCutORACLE    = sp.zeros(inneriter) + 1.e99
        # For storing "best" performance, i.e. find mean error for best fixed (ellCut):
        ErrsEll1Ell2EqualEllCutBEST = sp.zeros((int(floor(T/2)) - ktrunc + 1, inneriter)) + 1.e99
        # Use ell1 = ell2 (or ell1 = ell2 - 1)
        ell1 = int(floor(T/2)) - ktrunc
        ell2 = T - ell1 - 2*ktrunc
        for i in range(0,inneriter):
            if RunOracle:
                # Estimate "ORACLE" error using 1-view Tropp method using ellCut and ell1=ell2 (sometimes ell1=ell2-1):
                # Also store errors for selected values of ellCut
                ErrMin = 1.e99
                for ellCut in range(0, ell1 + 1):
                    [sk1view, Uk1view, Vk1view] = rand1viewTropp(A, A.transpose(), ktrunc, ell1, ell2, ellCut, SketchOrth=False)
                    # Find Frob norm error:
                    Frob1view = FrobNormError(A, sk1view, Uk1view, Vk1view)
                    # Store relative Frob error:
                    RelErr = sp.absolute(Frob1view/FrobOptRankP  - 1.)
                    # Store for finding "best" error:
                    ErrsEll1Ell2EqualEllCutBEST[ellCut, i] = RelErr
                    if RelErr < ErrMin:
                        ErrMin = RelErr + 0.
                    if ellCut == ell1:
                        ErrsEll1Ell2Equal[i] = RelErr
                    if ellCut == int(floor(ell1/2)):
                        ErrsEll1Ell2EqualEllCutHalf[i] = RelErr
                    if ellCut == int(floor(ell1/4)):
                        ErrsEll1Ell2EqualEllCutQuarter[i] = RelErr
                    if ellCut == int(floor(3.*ell1/4.)):
                        ErrsEll1Ell2EqualEllCutThreeFourths[i] = RelErr
                ErrsEll1Ell2EqualEllCutORACLE[i] = ErrMin
            else:
                [sk1view, Uk1view, Vk1view] = rand1viewTropp(A, A.transpose(), ktrunc, ell1, ell2, ell1, SketchOrth=False)
                # Find Frob norm error:
                Frob1view = FrobNormError(A, sk1view, Uk1view, Vk1view)
                # Store relative Frob error:
                ErrsEll1Ell2Equal[i] = sp.absolute(Frob1view/FrobOptRankP  - 1.)
                
                [sk1view, Uk1view, Vk1view] = rand1viewTropp(A, A.transpose(), ktrunc, ell1, ell2, int(floor(ell1/2)), SketchOrth=False)
                # Find Frob norm error:
                Frob1view = FrobNormError(A, sk1view, Uk1view, Vk1view)
                # Store relative Frob error:
                ErrsEll1Ell2EqualEllCutHalf[i] = sp.absolute(Frob1view/FrobOptRankP  - 1.)
                
                [sk1view, Uk1view, Vk1view] = rand1viewTropp(A, A.transpose(), ktrunc, ell1, ell2, int(floor(ell1/4)), SketchOrth=False)
                # Find Frob norm error:
                Frob1view = FrobNormError(A, sk1view, Uk1view, Vk1view)
                # Store relative Frob error:
                ErrsEll1Ell2EqualEllCutQuarter[i] = sp.absolute(Frob1view/FrobOptRankP  - 1.)
                
                [sk1view, Uk1view, Vk1view] = rand1viewTropp(A, A.transpose(), ktrunc, ell1, ell2, int(floor(3.*ell1/4.)), SketchOrth=False)
                # Find Frob norm error:
                Frob1view = FrobNormError(A, sk1view, Uk1view, Vk1view)
                # Store relative Frob error:
                ErrsEll1Ell2EqualEllCutThreeFourths[i] = sp.absolute(Frob1view/FrobOptRankP  - 1.)
                
            # Using MINVAR ellcut:
            [sk1view, Uk1view, Vk1view] = rand1viewTroppMinVar(A, A.transpose(), ktrunc, ell1, ell2, SketchOrth=False)
            # Find Frob norm error:
            Frob1view = FrobNormError(A, sk1view, Uk1view, Vk1view)
            # Store relative Frob error:
            ErrsEll1Ell2EqualEllCutMinVar[i] = sp.absolute(Frob1view/FrobOptRankP  - 1.)
                   
        # Store mean values:
        MeanErrEll1Ell2Equal[Tindx]                    = sp.mean(ErrsEll1Ell2Equal)
        MeanErrEll1Ell2EqualEllCutHalf[Tindx]          = sp.mean(ErrsEll1Ell2EqualEllCutHalf)
        MeanErrEll1Ell2EqualEllCutQuarter[Tindx]       = sp.mean(ErrsEll1Ell2EqualEllCutQuarter)
        MeanErrEll1Ell2EqualEllCutThreeFourths[Tindx]  = sp.mean(ErrsEll1Ell2EqualEllCutThreeFourths)
        MeanErrEll1Ell2EqualEllCutMinVar[Tindx]        = sp.mean(ErrsEll1Ell2EqualEllCutMinVar)
        # Oracle error as used by Tropp et al. (2017):
        MeanErrEll1Ell2EqualEllCutORACLE[Tindx]        = sp.mean(ErrsEll1Ell2EqualEllCutORACLE) # same as using sp.mean(sp.amin(ErrsEll1Ell2EqualEllCutBEST, axis=0))
        # "Best" error:
        MeanErrEll1Ell2EqualEllCutBEST[Tindx]    = sp.amin(sp.mean(ErrsEll1Ell2EqualEllCutBEST, axis=1))
    
    # Save mean values:
    sp.save(subfolder + '/' + 'TroppMeanFrobErrEll1Ell2Equal.npy', MeanErrEll1Ell2Equal)
    sp.save(subfolder + '/' + 'TroppMeanFrobErrEll1Ell2EqualEllCutHalf.npy', MeanErrEll1Ell2EqualEllCutHalf)
    sp.save(subfolder + '/' + 'TroppMeanFrobErrEll1Ell2EqualEllCutQuarter.npy', MeanErrEll1Ell2EqualEllCutQuarter)
    sp.save(subfolder + '/' + 'TroppMeanFrobErrEll1Ell2EqualEllCutThreeFourths.npy', MeanErrEll1Ell2EqualEllCutThreeFourths)
    sp.save(subfolder + '/' + 'TroppMeanFrobErrEll1Ell2EqualEllCutMinVar.npy', MeanErrEll1Ell2EqualEllCutMinVar)
    if RunOracle:
        sp.save(subfolder + '/' + 'TroppMeanFrobErrEll1Ell2EqualEllCutORACLE.npy', MeanErrEll1Ell2EqualEllCutORACLE)
        sp.save(subfolder + '/' + 'TroppMeanFrobErrEll1Ell2EqualEllCutBEST.npy', MeanErrEll1Ell2EqualEllCutBEST)

def run1viewBwzTests(A, ktrunc, matName, subfolder, inneriter, Tvec, FrobOptRankP, RunOracle):
    Nr, Nc = A.shape
    
    # Store mean errors for plots:
    MeanErrBwzOracle  = sp.zeros(len(Tvec))
    MeanErrBwzBest  = sp.zeros(len(Tvec))
    MeanErrBwzOracleV2  = sp.zeros(len(Tvec))
    MeanErrBwzBestV2  = sp.zeros(len(Tvec))
    MeanErrBwzV2Afac8  = sp.zeros(len(Tvec))
    
    for Tindx, T in enumerate(Tvec):
        print ktrunc, T
        T = int(T)
        StorageCost = T*(Nr + Nc)  # STORAGE COST of sketches and sampling matrices for 1-view Tropp
        ErrsBwzV2Afac8 = sp.zeros(inneriter) + 1.e99
        # For storing Oracle performance as used by Tropp et al. (2017), i.e. 
        # find the lowest error ErrMin[i] of any pair (t - ktrunc, s) over a Monte Carlo iteration i,
        # then find average of ErrMin:
        ErrsBwzOracle = sp.zeros(inneriter) + 1.e99
        ErrsBwzOracleV2 = sp.zeros(inneriter) + 1.e99
        # For storing "best" performance, i.e. find mean error for best fixed (t - ktrunc, s) pair:
        ErrsBwzBest = sp.zeros((int(floor(T/2)) - ktrunc + 1, inneriter)) + 1.e99
        ErrsBwzBestV2 = sp.zeros((int(floor(T/2)) - ktrunc + 1, inneriter)) + 1.e99
        for i in range(0,inneriter):
            # Run extended SRFT 1-view scheme using equations (6.18-19) 
            # in Bjarkason (2018):
            afac = 0.8
            t = int(floor(ktrunc + afac*(T/2. - ktrunc)))
            # Choose s so that the Bwz storage cost is similar to 
            # StorageCost of 1-view Tropp method:
            s = round((StorageCost + 1. - (2.*t + 1.)*(Nc+Nr))**(0.5) - 1.)
            s = int(min(min(Nc, Nr), s))
            # Low-rank approximation using Bwz algorithm V2:
            [sBwz, UBwz, VBwz] = rand1viewBwzImproved(A, ktrunc, t - ktrunc, s, SketchOrth=False)
            # Find Frob norm error:
            Frob1view = FrobNormError(A, sBwz, UBwz, VBwz)
            # Store relative Frob error:
            ErrsBwzV2Afac8[i] = sp.absolute(Frob1view/FrobOptRankP  - 1.)
            
            if RunOracle:
                t = ktrunc
                s = round((StorageCost + 1. - (2.*t + 1.)*(Nc+Nr))**(0.5) - 1.)
                s = int(min(min(Nc, Nr), s))
                while (t <= s) and (StorageCost >= -1. + (2.*t + 1.)*(Nc+Nr)): 
  #                  print 's', s 
  #                  print 'StorageCostRatio', ((2.*t + 1.)*(Nc + Nr) + s*(s + 2.))/StorageCost 
                    # Low-rank approximation using Bwz algorithm, see
                    # equation (6.17) in Bjarkason (2018):
                    [sBwz, UBwz, VBwz] = rand1viewBwz(A, ktrunc, t - ktrunc, s, SketchOrth=False)
                    # Find Frob norm error:
                    Frob1view = FrobNormError(A, sBwz, UBwz, VBwz)
                    # Store relative Frob error:
                    RelErr = sp.absolute(Frob1view/FrobOptRankP  - 1.)
                    # Store for finding "best" and "Oracle" errors:
                    ErrsBwzBest[t - ktrunc, i] = RelErr
                    if RelErr < ErrsBwzOracle[i]:
                        ErrsBwzOracle[i] = RelErr
                        
                    # Low-rank approximation using Bwz algorithm V2, see
                    # equation (6.18) in Bjarkason (2018):
                    [sBwz, UBwz, VBwz] = rand1viewBwzImproved(A, ktrunc, t - ktrunc, s, SketchOrth=False)
                    # Find Frob norm error:
                    Frob1view = FrobNormError(A, sBwz, UBwz, VBwz)
                    # Store relative Frob error:
                    RelErr = sp.absolute(Frob1view/FrobOptRankP  - 1.)
                    # Store for finding "best" and "Oracle" errors:
                    ErrsBwzBestV2[t - ktrunc, i] = RelErr
                    if RelErr < ErrsBwzOracleV2[i]:
                        ErrsBwzOracleV2[i] = RelErr    
                
                    t += 1
                    if (StorageCost >= -1. + (2.*t + 1.)*(Nc+Nr)):
                        s = round((StorageCost + 1. - (2.*t + 1.)*(Nc+Nr))**(0.5) - 1.)
                        s = int(min(min(Nc, Nr), s))
            
        # Store mean values:
        MeanErrBwzV2Afac8[Tindx]      = sp.mean(ErrsBwzV2Afac8)
        # Oracle error as used by Tropp et al. (2017):
        MeanErrBwzOracle[Tindx]     = sp.mean(ErrsBwzOracle) # same as using sp.mean(sp.amin(ErrsBwzBest, axis=0))
        MeanErrBwzOracleV2[Tindx]   = sp.mean(ErrsBwzOracleV2) # same as using sp.mean(sp.amin(ErrsBwzBestV2, axis=0))
        # "Best" error:
        MeanErrBwzBest[Tindx]      = sp.amin(sp.mean(ErrsBwzBest, axis=1))
        MeanErrBwzBestV2[Tindx]    = sp.amin(sp.mean(ErrsBwzBestV2, axis=1))
    
    # Save mean values:
    if RunOracle:
        sp.save(subfolder + '/' + 'MeanFrobErrBwzORACLE.npy',  MeanErrBwzOracle)
        sp.save(subfolder + '/' + 'MeanFrobErrBwzORACLEv2.npy',  MeanErrBwzOracleV2)
        sp.save(subfolder + '/' + 'MeanFrobErrBwzBest.npy',  MeanErrBwzBest)
        sp.save(subfolder + '/' + 'MeanFrobErrBwzBestv2.npy',  MeanErrBwzBestV2)
    sp.save(subfolder + '/' + 'MeanFrobErrBwzV2Afac8.npy',  MeanErrBwzV2Afac8)

def main(A, ktrunc, matName, subfolder, inneriter, Tvec):
    print 'Tests using matrix : ', matName
    
    if matName[0:7] == 'SDlarge':
        RunOracle = False
    else:
        RunOracle = True
    
    # Save sketch budgets Tvec:
    sp.save(subfolder + '/' + 'Tvec.npy',Tvec)
    
    # Evaluate Frobenius norm difference between A and optimal rank-p approximation:
    FrobOptRankP = optFrobErrRankp(A, ktrunc)
    
    temptime = time.clock()
    # Evaluate performance of baseline 1-view method proposed by 
    # Tropp et al. (2017) (Algorithm 7 in Bjarkason (2018) with ellCut = ell1):
    print '####################################################################'
    print 'Running baseline Tropp 1-view experiments'
    print '####################################################################'
    runStandard1viewTroppTests(A, ktrunc, matName, subfolder, inneriter, Tvec, FrobOptRankP, RunOracle)
    print matName, 'Time spent on baseline Tropp', time.clock() - temptime
 
    temptime = time.clock()
    # Evaluate performance of 1-view Algorithm 7 in Bjarkason (2018)
    # using ellCut and ell1=ell2 (sometimes ell1=ell2-1):
    print '####################################################################'
    print 'Running Lcut Alg. 7 1-view experiments'
    print '####################################################################'
    runLcut1viewTroppTests(A, ktrunc, matName, subfolder, inneriter, Tvec, FrobOptRankP, RunOracle)
    print matName, 'Time spent on Lcut Alg. 7', time.clock() - temptime
    
    temptime = time.clock()
    # Evaluate performance of extended SRFT schemes, see Section 6.7:
    print '####################################################################'
    print 'Running Bwz 1-view experiments'
    print '####################################################################'
    run1viewBwzTests(A, ktrunc, matName, subfolder, inneriter, Tvec, FrobOptRankP, RunOracle)
    print matName, 'Time spent on Bwz', time.clock() - temptime
    

def runTestForMat(inputs):  
    [rank, matName, MainSubfolder, inneriter, N] = inputs
    
    # Vector of sketch budgets T = 2p + ell1 + ell2
    TvecOver = sp.around( (2*5 + 6) * ( 2**(sp.linspace(0, 4.0, 10) ) ) ) - ( 2*5 + 6 )
    Tvec = TvecOver + ( 2* rank + 6 )
    print 'Sketch budgets T', Tvec
    
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
    
    # Run 1-view tests for test matrix:
    main(A, rank, matName, subfolder, inneriter, Tvec)
    
if __name__ == "__main__":
    multiprocessing.freeze_support()
    # Create folder for saving results:
    MainSubfolder = 'Results/1viewSampling/'
    newpath = os.getcwd() + '/' + MainSubfolder
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    
    # Names of test matrices:
    mats = ['lowRankMedNoiseR10', 'lowRankHiNoiseR10', 'polySlowR10', 'polyFastR10', 'expSlowR10', 'expFastR10', 'SDlarge', 'SDlarge', 'SDlarge']
    # Ranks for the approximations:
    ranks = [5]*len(mats)
    ranks[-2] = 1
    ranks[-1] = 50
    # Number of Monte Carlo runs used to estimate average or expected performance:
    inneriter = 50
    
    # Run 1-view tests:
    N = 1000 # size of square test matrices
    iterableInput = [ [ranks[indx], mats[indx], MainSubfolder, inneriter, N]  for indx in range(0,len(mats))  ]
    pool = Pool(processes=2)
    pool.map(runTestForMat, iterableInput)
    pool.terminate()







