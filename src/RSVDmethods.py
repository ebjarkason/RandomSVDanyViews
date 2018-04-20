# Randomized TSVD routines
# Coded by: Elvar K. Bjarkason (2018)

import scipy as sp
from scipy.linalg import cholesky, solve_triangular, svd, eigh
import multiprocessing
from multiprocessing import Pool
from scipy.sparse.linalg import LinearOperator, aslinearoperator
from numpy.linalg import lstsq
from scipy.fftpack import dct

# rank_p_SVD(X, p):
# Return rank-p TSVD of matrix X, Uxp * diag(rxp) * Vxp^T.
# ----------------------------------------------------------------------
# INPUTS:
# X: (Nr by Nc) matrix
# p: rank of TSVD
# ----------------------------------------------------------------------
# RETURNS:
# Uxp: (Nr by p) matrix, where the ith column is the ith left singular vector
# rxp: top p singular values of X
# Vxp: (Nc by p) matrix, where the ith column is the ith right singular vector
# ----------------------------------------------------------------------
def rank_p_SVD(X, p):
    [Ux, rx, Vhx] = svd(X, full_matrices = False, compute_uv=True, overwrite_a=False, check_finite=True)
    return Ux[:,0:p], rx[0:p], Vhx[0:p,:].transpose()

# rank_p_EVD(X, p):
# Return rank-p TEVD of matrix X, Vxp * diag(rxp) * Vxp^T.
# ----------------------------------------------------------------------
# INPUTS:
# X: (N by N) matrix
# p: rank of TEVD
# ----------------------------------------------------------------------
# RETURNS:
# rxp: top p eigenvalues of X
# Vxp: (N by p) matrix, where the ith column is the ith eigenvector
# ----------------------------------------------------------------------    
def rank_p_EVD(X, p):
    [rx, Vx] = eigh(X)
    rx = rx[::-1]
    Vx = Vx[:,::-1]
    return rx[0:p], Vx[:,0:p]

# randGenHalfSubIter(A, AT, ktrunc, ell, views): 
# Algorithm 2 in Bjarkason (2018)
# A general randomized algorithm for estimating a TSVD of matrix A,
# by accesing the matrix views >= 2 times and using a (half) subspace iteration scheme.
# ----------------------------------------------------------------------
# INPUTS:
# A: (Nr by Nc) LinearOperator (for A*matrix) or matrix
# AT: (Nc by Nr) LinearOperator (for AT*matrix), matrix or None
# !!! Note on A and AT: When specifying A as a dense matrix, then 
# set AT=None. Other option is to specify A and AT as linear operators for
# evaluating A times a matrix and A transposed times a matrix.
# ktrunc: (integer > 0) number of retained singular values
# ell: (integer >= 0) amount of oversampling
# views: (integer >= 2) the number of matrix views.
# ----------------------------------------------------------------------
# RETURNS: approximate rank-ktrunc TSVD, Uk*diag(sk)*Vk^T, of A:
# sk: approximate top ktrunc singular values of A
# Uk: (Nr by ktrunc) matrix, where the ith column approximates the ith left singular vector
# Vk: (Nc by ktrunc) matrix, where the ith column approximates the ith right singular vector
# ----------------------------------------------------------------------
def randGenHalfSubIter(A, AT, ktrunc, ell, views):
    if (views < 2):
        print 'Exited without result: views needs to be 2 or greater'
        raise SystemExit
    # Perform general (half) subspace iteration:
    Nr, Nc = A.shape
    Qr = sp.random.randn(Nc, ktrunc + ell)
    if AT is None:
        for j in range(0, views):
            if (j%2) == 0:
                Qc = A.dot(Qr)
                # Orthonormalize:
                [Qc, Rc] = sp.linalg.qr(Qc, mode='economic')
            else:
                Qr = sp.dot( Qc.transpose(), A).transpose()
                # Orthonormalize:
                [Qr, Rr] = sp.linalg.qr(Qr, mode='economic')
    else:
        A = aslinearoperator(A)
        AT = aslinearoperator(AT)
        for j in range(0, views):
            if (j%2) == 0:
                Qc = A.matmat(Qr)
                # Orthonormalize:
                [Qc, Rc] = sp.linalg.qr(Qc, mode='economic')
            else:
                Qr = AT.matmat(Qc)
                # Orthonormalize:
                [Qr, Rr] = sp.linalg.qr(Qr, mode='economic')   
    if (views%2) == 0: 
        [Vhatk, sk, Uhatk] = rank_p_SVD(Rr, ktrunc)
    else:
        [Uhatk, sk, Vhatk] = rank_p_SVD(Rc, ktrunc)
    Uk = Qc.dot(Uhatk)
    Vk = Qr.dot(Vhatk)
    return sk, Uk, Vk

# constructQrOrQc(A, sample, views): 
# Algorithm 3 in Bjarkason (2018)
# Construct an orthonormal basis for the approximate range of matrix A, Qc, 
# or the co-range of A, Qr.
# ----------------------------------------------------------------------
# INPUTS:
# A: (Nr by Nc) dense matrix
# sample: number of sampling vectors for sketching A
# views: (integer > 0) the number of matrix views.
# ----------------------------------------------------------------------
# RETURNS: Qr or Qc
# Qr: (Nc by sample) matrix whose columns form an orthonormal basis for 
# the approximate co-range of A, returned if views is even
# Qc: (Nr by sample) matrix whose columns form an orthonormal basis for 
# the approximate range of A, returned if views is odd
# ----------------------------------------------------------------------    
def constructQrOrQc(A, sample, views):
    Nr, Nc = A.shape
    Qr = sp.random.randn(Nc, sample)
    for j in range(0, views):
        if (j%2) == 0:
            Qc = A.dot(Qr)
            # Orthonormalize:
            Qc = sp.linalg.qr(Qc, mode='economic')[0]
        else:
            Qr = sp.dot( Qc.transpose(), A).transpose()
            # Orthonormalize:
            Qr = sp.linalg.qr(Qr, mode='economic')[0]
    # Return Qr or Qc:
    if (views%2) == 0: 
        return Qr
    else:
        return Qc

# nystromOnNormal(A, ktrunc, ell, views): 
# Algorithm 4 in Bjarkason (2018)
# Randomized algorithm for estimating a TEVD of matrix AT*A, using
# prolonged or Nystrom-type sketch and by accesing the matrix A views times.
# ----------------------------------------------------------------------
# INPUTS:
# A: (Nr by Nc) dense matrix
# ktrunc: (integer  > 0) number of retained singular values
# ell: (integer  >= 0) amount of oversampling
# views: (integer >= 2) the number of matrix views.
# ----------------------------------------------------------------------
# RETURNS: approximate rank-ktrunc TEVD, Vk*diag(sk)*Vk^T, of AT*A:
# sk: approximate top ktrunc eigenvalues of AT*A
# Vk: (Nc by ktrunc) matrix, where the ith column approximates the ith eigenvector
# ----------------------------------------------------------------------
def nystromOnNormal(A, ktrunc, ell, views):
    if (views < 2):
        print 'Exited without result: views needs to be 2 or greater'
        raise SystemExit
    # Find approximate TEVD of normal matrix AT*A, using Nystrom-type sketch:
    if (views == 2):
        Nr, Nc = A.shape
        Qr = sp.random.randn(Nc, ktrunc + ell)
        Qr = sp.linalg.orth(Qr)
    elif (views%2) == 0:
        Qr = constructQrOrQc(A, ktrunc + ell, views-2)
    else:
        Qr = constructQrOrQc(A.transpose(), ktrunc + ell, views-2)
    Yr = A.dot(Qr)
    Yr = sp.dot(Yr.transpose(), A).transpose()
    nu = 2.2e-16 * sp.linalg.norm(Yr, ord=2, axis=None, keepdims=False)
    Yr += nu*Qr
    B = sp.dot(Qr.transpose(), Yr)
    C = cholesky( 0.5*(B + B.transpose()) , lower=True , overwrite_a=False , check_finite=True )
    F = solve_triangular(C , Yr.transpose() , trans=0 , lower=True , unit_diagonal=False, overwrite_b=False, debug=False, check_finite=False) 
    [skhat, Vk] = rank_p_SVD(F, ktrunc)[1::]
    sk = skhat**2. - nu
    sk[sk < 0.] = 0.
    return sk, Vk

# pinchedOnNormal(A, ktrunc, ell, views): 
# Algorithm 5 in Bjarkason (2018)
# Randomized algorithm for estimating a TEVD of matrix AT*A, using
# pinched sketch and by accesing the matrix A views times.
# ----------------------------------------------------------------------
# INPUTS:
# A: (Nr by Nc) dense matrix
# ktrunc: (integer > 0) number of retained singular values
# ell: (integer >= 0) amount of oversampling
# views: (integer >= 2) the number of matrix views.
# ----------------------------------------------------------------------
# RETURNS: approximate rank-ktrunc TEVD, Vk*diag(sk)*Vk^T, of AT*A:
# sk: approximate top ktrunc eigenvalues of AT*A
# Vk: (Nc by ktrunc) matrix, where the ith column approximates the ith eigenvector
# ----------------------------------------------------------------------
def pinchedOnNormal(A, ktrunc, ell, views):
    if (views < 2):
        print 'Exited without result: views needs to be 2 or greater'
        raise SystemExit
    # Find approximate TEVD of normal matrix AT*A, using pinched sketch:
    if (views%2) == 0:
        Qr = constructQrOrQc(A.transpose(), ktrunc + ell, views-1)
    else:
        Qr = constructQrOrQc(A, ktrunc + ell, views-1)
    B = A.dot(Qr)
    [sk, Vkhat] = rank_p_EVD(sp.dot(B.transpose(), B), ktrunc)
    Vk = Qr.dot(Vkhat)
    return  sk, Vk    
    
# RangeOrCoRangeSketch(A, ktrunc, ell, ellCut, SketchOrth, option):
# A function used within the 1-view methods when using multiprocessing to 
# estimate the range and co-range of matrix A simultaneously in parallel 
# (e.g. for parallel adjoint and direct solves).
# ----------------------------------------------------------------------
# INPUTS:
# inputs = [A, ktrunc, ell, ellCut, SketchOrth, option], 
# where
# A: (Nr by Nc) LinearOperator (for A*matrix)
# ktrunc: (integer) number of retained singular values
# ell: (integer) amount of oversampling for range of A
# ellCut: (integer) cutoff parameter (ell >= ellCut >= 0)
# SketchOrth: (True or False) determines whether to orthonormalize the 
# random sampling matrices used to form the range sketch
# option: 'rangeTropp', 'corangeTropp', 'rangeWoolfe', 'corangeWoolfe', or 'rangeMinVar'
# ----------------------------------------------------------------------
# RETURNS:
# if option=='rangeTropp': Q
# if option=='corangeTropp': Omega^T, Y^T
# if option=='rangeWoolfe': Q
# if option=='corangeWoolfe': Q, Omega^T, Y^T
# if option=='rangeMinVar': Q
# else: Q, Omega, Y
# where Omega is an (Nc by ktrunc+ell) Gaussian random sampling matrix,
# Y is an (Nr by ktrunc+ell) range sketching matrix and Q is an (Nr by ktrunc+ellCut)
# matrix with orthonormal columns which form a basis that spans the space
# given by the principal ktrunc+ellCut left singular vectors of Y.
# ----------------------------------------------------------------------
def RangeOrCoRangeSketch(inputs):
    [A, ktrunc, ell, ellCut, SketchOrth, option] = inputs
    Nr, Nc = A.shape
    # Form random matrix for sketching matrix A:
    Omega = sp.random.randn(Nc, ktrunc+ell)
    # Orthonormalize:
    if SketchOrth:
        Omega = sp.linalg.orth(Omega)
    # Form range sketch:
    Y = A.matmat(Omega)
    if option=='corangeTropp':
        return Omega.transpose(), Y.transpose()
    else:
        # Find orthonormal basis for the range sketch:
        if (ellCut < ell) or (option=='rangeMinVar'):
            Q, R = sp.linalg.qr(Y, mode='economic')
            Qhat = rank_p_SVD(R, ktrunc+ellCut)[0]
            Q = Q.dot(Qhat)
        else:
            Q = sp.linalg.qr(Y, mode='economic')[0]   
        if option=='corangeWoolfe':
            return Q, Omega.transpose(), Y.transpose()
        elif (option=='rangeWoolfe') or (option=='rangeTropp') or (option=='rangeMinVar'): 
            return Q
        else:
            return Q, Omega, Y
            
# rand1viewTropp(A, AT, ktrunc, ell1, ell2, ellCut, SketchOrth=False, Parallel=False): 
# Algorithm 7 in Bjarkason (2018)
# A randomized 1-view method for estimating a TSVD  of a matrix A.
# Based on a 1-view algorithm by Tropp et al. (2017)
# "PRACTICAL SKETCHING ALGORITHMS FOR LOW-RANK MATRIX APPROXIMATION",
# with modifications to improve accuracy when ell1 and ell2 are similar. 
# ----------------------------------------------------------------------
# INPUTS:
# A: (Nr by Nc) LinearOperator (for A*matrix) or matrix
# AT: (Nc by Nr) LinearOperator (for AT*matrix) or matrix
# (A and AT can be specified as a dense/sparse matrix (for test purposes)),
# ktrunc: (integer > 0) number of retained singular values
# ell1: (integer) amount of oversampling for range
# ell2: (integer) amount of oversampling for co-range (ell2 >= ell1 >= 0)
# ellCut: (integer) cutoff parameter (ell1 >= ellCut >= 0)
# SketchOrth: (True or False) determines whether to orthonormalize the 
# random sampling matrices used to form the range and co-range sketches
# Parallel: (True or False) When True, then find the range and co-range 
# sketches simultaneously in parallel. Use Parallel=False, for serial version
# but NOTE! that this is just for testing purposes, since using pool.map has
# overheads which make the parallel implementation slow for small test
# problems which use an explicit input matrix.
# The 1-view method, with Parallel=True, is meant for applications
# where there are separate functions (which are computationally expensive) 
# for evaluating A times a matrix and AT times a matrix. 
# ----------------------------------------------------------------------
# RETURNS: approximate rank-ktrunc TSVD, Uk*diag(sk)*Vk^T, of A:
# sk: approximate top ktrunc singular values of A
# Uk: (Nr by ktrunc) matrix, where the ith column approximates the ith left singular vector
# Vk: (Nc by ktrunc) matrix, where the ith column approximates the ith right singular vector
# ----------------------------------------------------------------------       
def rand1viewTropp(A, AT, ktrunc, ell1, ell2, ellCut, SketchOrth=False, Parallel=False):
    A = aslinearoperator(A)
    AT = aslinearoperator(AT)
    if Parallel:
        # Parallel evaluation of range and co-range sketches:
        pool = Pool(processes=2)
        SketchResults = pool.map(RangeOrCoRangeSketch, 
            [ [A, ktrunc, ell1, ellCut, SketchOrth, 'rangeTropp'], 
            [AT, ktrunc, ell2, ell2, SketchOrth, 'corangeTropp'] ])
        pool.terminate()
        [Qc, [OmegaCT, YrT]]  = SketchResults
    else:
        # Serial evaluation of range and co-range sketches:
        Qc = RangeOrCoRangeSketch([A, ktrunc, ell1, ellCut, SketchOrth, 'rangeTropp'])
        [OmegaCT, YrT] = RangeOrCoRangeSketch([AT, ktrunc, ell2, ell2, SketchOrth, 'corangeTropp'])
    
    [Qhat, Rhat] = sp.linalg.qr(OmegaCT.dot(Qc), mode='economic')
    # Solve small linear problem Rhat*X = (Qhat^T)*YrT, where Rhat is upper triangular:
    X = solve_triangular(Rhat, sp.dot(Qhat.transpose(), YrT), trans=0, lower=False)
    # SVD of small matrix X:
    [Uhatk, sk, Vk] = rank_p_SVD(X, ktrunc)
    Uk = Qc.dot(Uhatk)
    return sk, Uk, Vk

# rand1viewWoolfeVariant(A, AT, ktrunc, ell1, ell2, ellCut, SketchOrth=False, Parallel=False): 
# Algorithm 8 in Bjarkason (2018)
# A randomized 1-view method for estimating a TSVD  of a matrix A.
# Randomized 1-view algorithm which is based on an algorithm proposed by
# Woolfe et al. (2008) "A fast randomized algorithm for the approximation of matrices"
# ----------------------------------------------------------------------
# INPUTS:
# A: (Nr by Nc) LinearOperator (for A*matrix) or matrix
# AT: (Nc by Nr) LinearOperator (for AT*matrix) or matrix
# (A and AT can be specified as a dense/sparse matrix (for test purposes)),
# ktrunc: (integer > 0) number of retained singular values
# ell1: (integer) amount of oversampling for range
# ell2: (integer) amount of oversampling for co-range (ell2 >= ell1 >= 0)
# ellCut: (integer) cutoff parameter (ell1 >= ellCut >= 0)
# SketchOrth: (True or False) determines whether to orthonormalize the 
# random sampling matrices used to form the range and co-range sketches
# Parallel: (True or False) When True, then find the range and co-range 
# sketches simultaneously in parallel. Use Parallel=False, for serial version
# but NOTE! that this is just for testing purposes, since using pool.map has
# overheads which make the parallel implementation slow for small test
# problems which use an explicit input matrix.
# The 1-view method, with Parallel=True, is meant for applications
# where there are separate functions (which are computationally expensive) 
# for evaluating A times a matrix and AT times a matrix. 
# ----------------------------------------------------------------------
# RETURNS: approximate rank-ktrunc TSVD, Uk*diag(sk)*Vk^T, of A:
# sk: approximate top ktrunc singular values of A
# Uk: (Nr by ktrunc) matrix, where the ith column approximates the ith left singular vector
# Vk: (Nc by ktrunc) matrix, where the ith column approximates the ith right singular vector    
def rand1viewWoolfeVariant(A, AT, ktrunc, ell1, ell2, ellCut, SketchOrth=False, Parallel=False):
    A = aslinearoperator(A)
    AT = aslinearoperator(AT)
    if Parallel:
        # Parallel evaluation of range and co-range sketches:
        pool = Pool(processes=2)
        SketchResults = pool.map(RangeOrCoRangeSketch, 
            [ [A, ktrunc, ell1, ellCut, SketchOrth, 'rangeWoolfe'], 
            [AT, ktrunc, ell2, ellCut, SketchOrth, 'corangeWoolfe'] ])
        pool.terminate()
        [Qc, [Qr, OmegaCT, YrT]]  = SketchResults
    else:
        # Serial evaluation of range and co-range sketches:
        Qc = RangeOrCoRangeSketch([A, ktrunc, ell1, ellCut, SketchOrth, 'rangeWoolfe'])
        [Qr, OmegaCT, YrT] = RangeOrCoRangeSketch([AT, ktrunc, ell2, ellCut, SketchOrth, 'corangeWoolfe'])
    
    # Solve small least-squares problem (OmegaCT*Qc)X = YrT*Qr:
    [Qhat, Rhat] = sp.linalg.qr(OmegaCT.dot(Qc), mode='economic')
    X = solve_triangular(Rhat, sp.dot(Qhat.transpose(), YrT.dot(Qr)), trans=0, lower=False)
    # SVD of small matrix X:
    [Uhatk, sk, Vhatk] = rank_p_SVD(X, ktrunc)
    Uk = Qc.dot(Uhatk)
    Vk = Qr.dot(Vhatk)
    return sk, Uk, Vk
    
# getXsingvals(ktrunc, ellCut, OmegaCTdotQc, YrT):
# ----------------------------------------------------------------------
# INPUTS:
# ktrunc: (integer > 0) number of retained singular values
# ellCut:  (integer) oversampling cutoff parameter (ell1 >= ellCut >= 0)
# OmegaCTdotQc: (ktrunc + ell2 by ktrunc + ell1) matrix (= OmegaCT * Qc)
# YrT: (ktrunc + ell2 by Nc) matrix (transpose of co-range sketch)
# ----------------------------------------------------------------------
# RETURNS:
# sk: vector of length ktrunc of estimated singular values
# ---------------------------------------------------------------------- 
def getXsingvals(ktrunc, ellCut, OmegaCTdotQc, YrT):
    [Qhat, Rhat] = sp.linalg.qr(OmegaCTdotQc[:,0:(ktrunc+ellCut)], mode='economic')
    # Solve small linear problem Rhat*X = (Qhat^T)*YrT, where Rhat is upper triangular:
    X = solve_triangular(Rhat, sp.dot(Qhat.transpose(), YrT), trans=0, lower=False)
    # SVD of small matrix X:
    sk = rank_p_SVD(X, ktrunc)[1]
    return sk

# findMinVarEllCut(singMat):
# Find "minimu variance" ellCut parameter 
# using equation(6.11) in Bjarkason (2018)
# ----------------------------------------------------------------------
# INPUTS:
# singMat: (ktrunc by Nell) matrix, where the ith column contains
# estimated singular values for the ith ellCut parameter
# ----------------------------------------------------------------------
# RETURNS:
# ellCut: (integer >= 0) "minimu variance" ellCut parameter
# ---------------------------------------------------------------------- 
def findMinVarEllCut(singMat):
    Nk, Nell = singMat.shape 
    # Estimate Local Variance of Singular Values:
    singVarEstVec = sp.zeros(Nell-1)
    for i in range(0, Nell-1):
        if i == 0:
            si = singMat[:, i]
            # Normalized singular values:
            sTestMat = sp.zeros((Nk, 2))
            sTestMat[:, 0] = si/si
            sTestMat[:, 1] = singMat[:, i+1]/si
            singVarEstVec[i] = sp.var(sTestMat)
        else:
            si = singMat[:, i]
            # Normalized singular values:
            sTestMat = sp.zeros((Nk, 3))
            sTestMat[:, 0] = singMat[:, i-1]/si
            sTestMat[:, 1] = si/si
            sTestMat[:, 2] = singMat[:, i+1]/si
            singVarEstVec[i] = sp.var(sTestMat)
    # Find minimum variance Index:
    ellCut = sp.argmin(singVarEstVec)
    return ellCut  

# getMinVarEllCut(ktrunc, ell1, OmegaCTdotQc, YrT):
# ----------------------------------------------------------------------
# INPUTS:
# ktrunc: (integer > 0) number of retained singular values
# ell1: (intege > 0)) amount of oversampling for range
# OmegaCTdotQc: (ktrunc + ell2 by ktrunc + ell1) matrix (= OmegaCT * Qc)
# YrT: (ktrunc + ell2 by Nc) matrix
# ----------------------------------------------------------------------
# RETURNS:
# ellCut: (integer >= 0) "minimu variance" ellCut parameter
# ----------------------------------------------------------------------         
def getMinVarEllCut(ktrunc, ell1, OmegaCTdotQc, YrT):
    ellCutVec = range(0, ell1 + 1)
    singMat = sp.zeros((ktrunc, len(ellCutVec)))
    for ellCut in ellCutVec:
        singMat[:, ellCut] = getXsingvals(ktrunc, ellCut, OmegaCTdotQc, YrT)
    ellCut = findMinVarEllCut(singMat)
    return ellCut

# rand1viewTroppMinVar(A, AT, ktrunc, ell1, ell2, SketchOrth=False, Parallel=False): 
# A modified version of the randomized 1-view Algorithm 7 in Bjarkason (2018),
# which uses a "minimum variance" ellCut parameter found using
# equation (6.11) in Bjarkason (2018)
# ----------------------------------------------------------------------
# INPUTS:
# A: (Nr by Nc) LinearOperator (for A*matrix) or matrix
# AT: (Nc by Nr) LinearOperator (for AT*matrix) or matrix
# (A and AT can also be specified as a dense/sparse matrix (for test purposes)),
# ktrunc: (integer > 0) number of retained singular values
# ell1: (integer) amount of oversampling for range
# ell2: (integer) amount of oversampling for co-range (ell2 >= ell1 >= 0)
# SketchOrth: (True or False) determines whether to orthonormalize the 
# random sampling matrices used to form the range and co-range sketches
# Parallel: (True or False) When True, then find the range and co-range 
# sketches simultaneously in parallel. Use Parallel=False, for serial version
# but NOTE! that this is just for testing purposes, since using pool.map has
# overheads which make the parallel implementation slow for small test
# problems which use an explicit input matrix.
# The 1-view method, with Parallel=True, is meant for applications
# where there are separate functions (which are computationally expensive) 
# for evaluating A times a matrix and AT times a matrix. 
# ----------------------------------------------------------------------
# RETURNS: approximate rank-ktrunc TSVD, Uk*diag(sk)*Vk^T, of A:
# sk: approximate top ktrunc singular values of A
# Uk: (Nr by ktrunc) matrix, where the ith column approximates the ith left singular vector
# Vk: (Nc by ktrunc) matrix, where the ith column approximates the ith right singular vector
# ----------------------------------------------------------------------     
def rand1viewTroppMinVar(A, AT, ktrunc, ell1, ell2, SketchOrth=False, Parallel=False):
    A = aslinearoperator(A)
    AT = aslinearoperator(AT)
    if Parallel:
        # Parallel evaluation of range and co-range sketches:
        pool = Pool(processes=2)
        SketchResults = pool.map(RangeOrCoRangeSketch, 
            [ [A, ktrunc, ell1, ell1, SketchOrth, 'rangeMinVar'], 
            [AT, ktrunc, ell2, ell2, SketchOrth, 'corangeTropp'] ])
        pool.terminate()
        [Qc, [OmegaCT, YrT]]  = SketchResults
    else:
        # Serial evaluation of range and co-range sketches:
        Qc = RangeOrCoRangeSketch([A, ktrunc, ell1, ell1, SketchOrth, 'rangeMinVar'])
        [OmegaCT, YrT] = RangeOrCoRangeSketch([AT, ktrunc, ell2, ell2, SketchOrth, 'corangeTropp'])
       
    # Find "minimum variance" ellCut:
    OmegaCTdotQc = OmegaCT.dot(Qc)
    ellCut = getMinVarEllCut(ktrunc, ell1, OmegaCTdotQc, YrT)
    
    Qc = Qc[:,0:(ktrunc+ellCut)]
    [Qhat, Rhat] = sp.linalg.qr(OmegaCTdotQc[:,0:(ktrunc+ellCut)], mode='economic')
    # Solve small linear problem Rhat*X = (Qhat^T)*YrT, where Rhat is upper triangular:
    X = solve_triangular(Rhat, sp.dot(Qhat.transpose(), YrT), trans=0, lower=False)
    # SVD of small matrix X:
    [Uhatk, sk, Vk] = rank_p_SVD(X, ktrunc)
    Uk = Qc.dot(Uhatk)
    return sk, Uk, Vk


# rand1viewBwz(A, ktrunc, ell, s, SketchOrth=False): 
# An randomized extended 1-view method for estimating a low-rank approximation of a matrix A.
# Implementation of 1-view algorithm outlined in section 7.3.2 in Tropp et al. (2017)
# "PRACTICAL SKETCHING ALGORITHMS FOR LOW-RANK MATRIX APPROXIMATION".
# The algorithm uses an additional random sketch using a SRFT sampling matrix. 
# NOTE: Naive implementation just for testing.
# ----------------------------------------------------------------------
# INPUTS:
# A: (Nr by Nc) dense matrix
# ktrunc: (integer > 0) number of retained singular values
# ell: (integer >= 0) amount of oversampling for range and co-range
# s: (integer >= ktrunc + ell) s - ktrunc specifies the amount of oversampling used for the SRFT sketch
# SketchOrth: (True or False) determines whether to orthonormalize the 
# random sampling matrices used to form the range and co-range sketches 
# ----------------------------------------------------------------------
# RETURNS: approximate rank-ktrunc approximation, U*diag(sbwz)*V^T, of A:
# sbwz: vector of length ktrunc
# U: (Nr by ktrunc) matrix
# V: (Nc by ktrunc) matrix
# ----------------------------------------------------------------------   
def rand1viewBwz(A, ktrunc, ell, s, SketchOrth=False):
    # Serial evaluation of range and co-range sketches:
    Qc = RangeOrCoRangeSketch([aslinearoperator(A), ktrunc, ell, ell, SketchOrth, 'rangeTropp'])
    Qr = RangeOrCoRangeSketch([aslinearoperator(A.transpose()), ktrunc, ell, ell, SketchOrth, 'rangeTropp'])
    
    Nr, Nc = A.shape
    # Generate extra information for SRFT matrices:
    dPhi = sp.random.choice([-1, 1], Nr, replace=True)
    IdxPhi = sp.random.choice(Nr, s, replace=False)
    dXi = sp.random.choice([-1, 1], Nc, replace=True)
    IdxXi = sp.random.choice(Nc, s, replace=False)
    # Evaluate extra random sketch, Z = Phi * A * Xi (where Phi^T and Xi are SRFT matrices):
    AXi = dct(A * dXi, n=None, axis=1)[:, IdxXi]
    Z = dct(AXi.transpose() * dPhi, n=None, axis=1)[:, IdxPhi].transpose()
    
    # Extra triangular factorizations:
    PhiQc = dct(Qc.transpose() * dPhi, n=None, axis=1)[:, IdxPhi].transpose()
    U1, T1 = sp.linalg.qr(PhiQc, mode='economic')
    QrTXi = dct(Qr.transpose() * dXi, n=None, axis=1)[:, IdxXi]
    U2, T2 = sp.linalg.qr(QrTXi.transpose(), mode='economic')
    
    # Find rank-p approximation:
    Ubwz, sbwz, Vbwz = rank_p_SVD(sp.dot(U1.transpose(), Z.dot(U2)), ktrunc)
    U = Qc.dot(solve_triangular(T1, Ubwz, trans=0, lower=False))
    V = Qr.dot(solve_triangular(T2, Vbwz, trans=0, lower=False))
    return sbwz, U, V

# rand1viewBwzImproved(A, AT, ktrunc, ell, s, SketchOrth=False): 
# A randomized 1-view method for estimating a TSVD approximation of a matrix A.
# An improved implementation of 1-view algorithm outlined in section 7.3.2 in Tropp et al. (2017)
# "PRACTICAL SKETCHING ALGORITHMS FOR LOW-RANK MATRIX APPROXIMATION".
# See equation (6.18) in section 6 in Bjarkason (2018).
# The algorithm uses an additional random sketch using a SRFT sampling matrix. 
# NOTE: Naive implementation just for testing.
# ----------------------------------------------------------------------
# INPUTS:
# A: (Nr by Nc) dense matrix
# ktrunc: (integer > 0) number of retained singular values
# ell: (integer >= 0) amount of oversampling for range and co-range
# s: (integer >= ktrunc + ell) s - ktrunc specifies the amount of oversampling used for the SRFT sketch
# SketchOrth: (True or False) determines whether to orthonormalize the 
# random sampling matrices used to form the range and co-range sketches 
# ----------------------------------------------------------------------
# RETURNS: approximate rank-ktrunc TSVD approximation, Uk*diag(sk)*Vk^T, of A:
# sk: approximate top ktrunc singular values of A
# Uk: (Nr by ktrunc) matrix, where the ith column approximates the ith left singular vector
# Vk: (ktrunc by Nc) matrix, where the ith column approximates the ith right singular vector
# ---------------------------------------------------------------------- 
def rand1viewBwzImproved(A, ktrunc, ell, s, SketchOrth=False):
    # Serial evaluation of range and co-range sketches:
    Qc = RangeOrCoRangeSketch([aslinearoperator(A), ktrunc, ell, ell, SketchOrth, 'rangeTropp'])
    Qr = RangeOrCoRangeSketch([aslinearoperator(A.transpose()), ktrunc, ell, ell, SketchOrth, 'rangeTropp'])
    
    Nr, Nc = A.shape
    # Generate extra information for SRFT matrices:
    dPhi = sp.random.choice([-1, 1], Nr, replace=True)
    IdxPhi = sp.random.choice(Nr, s, replace=False)
    dXi = sp.random.choice([-1, 1], Nc, replace=True)
    IdxXi = sp.random.choice(Nc, s, replace=False)
    # Evaluate extra random sketch, Z = Phi * A * Xi (where Phi^T and Xi are SRFT matrices):
    AXi = dct(A * dXi, n=None, axis=1)[:, IdxXi]
    Z = dct(AXi.transpose() * dPhi, n=None, axis=1)[:, IdxPhi].transpose()
    
    # Extra triangular factorizations:
    PhiQc = dct(Qc.transpose() * dPhi, n=None, axis=1)[:, IdxPhi].transpose()
    U1, T1 = sp.linalg.qr(PhiQc, mode='economic')
    QrTXi = dct(Qr.transpose() * dXi, n=None, axis=1)[:, IdxXi]
    U2, T2 = sp.linalg.qr(QrTXi.transpose(), mode='economic')
    
    # Find rank-p approximation:
    X = sp.dot(U1.transpose(), Z.dot(U2))
    X = solve_triangular(T1, X, trans=0, lower=False)
    X = solve_triangular(T2, X.transpose(), trans=0, lower=False).transpose()
    Uhat, sk, Vhat = rank_p_SVD(X, ktrunc)
    Uk = Qc.dot(Uhat)
    Vk = Qr.dot(Vhat)
    return sk, Uk, Vk     
    
       
    
# getXsingvalsWoolfe(ktrunc, ellCut, OmegaCTdotQc, YrT):
# ----------------------------------------------------------------------
# INPUTS:
# ktrunc: (integer > 0) number of retained singular values
# ellCut:  (integer) oversampling cutoff parameter (ell1 >= ellCut >= 0)
# OmegaCTdotQc: (ktrunc + ell2 by ktrunc + ell1) matrix (= OmegaCT * Qc)
# YrT: (ktrunc + ell2 by Nc) matrix (transpose of co-range sketch)
# ----------------------------------------------------------------------
# RETURNS:
# sk: vector of length ktrunc of estimated singular values
# ---------------------------------------------------------------------- 
def getXsingvalsWoolfe(ktrunc, ellCut, OmegaCTdotQc, YrTQr):
    [Qhat, Rhat] = sp.linalg.qr(OmegaCTdotQc[:,0:(ktrunc+ellCut)], mode='economic')
    # Solve small linear problem Rhat*X = (Qhat^T)*YrT, where Rhat is upper triangular:
    X = solve_triangular(Rhat, sp.dot(Qhat.transpose(), YrTQr), trans=0, lower=False)
    # SVD of small matrix X:
    sk = rank_p_SVD(X, ktrunc)[1]
    return sk 

# getMinVarEllCutWoolfe(ktrunc, ell1, OmegaCTdotQc, YrTQr):
# ----------------------------------------------------------------------
# INPUTS:
# ktrunc: (integer > 0) number of retained singular values
# ell1: (integer) amount of oversampling for range
# OmegaCTdotQc: (ktrunc + ell2 by ktrunc + ell1) matrix (= OmegaCT * Qc)
# YrTQr: (ktrunc + ell2 by ktrunc + ell2) matrix
# ----------------------------------------------------------------------
# RETURNS:
# ellCut: (integer) "minimu variance" ellCut parameter
# ----------------------------------------------------------------------         
def getMinVarEllCutWoolfe(ktrunc, ell1, OmegaCTdotQc, YrTQr):
    ellCutVec = range(0, ell1 + 1)
    singMat = sp.zeros((ktrunc, len(ellCutVec)))
    for ellCut in ellCutVec:
        singMat[:, ellCut] = getXsingvalsWoolfe(ktrunc, ellCut, OmegaCTdotQc, YrTQr)
    ellCut = findMinVarEllCut(singMat)
    return ellCut

      
# rand1viewTroppMinVar(A, AT, ktrunc, ell1, ell2, SketchOrth=False, Parallel=False): 
# A modified version of the randomized 1-view Algorithm 8 in Bjarkason (2018),
# which uses a "minimum variance" ellCut parameter found using
# equation (6.11) in Bjarkason (2018)
# ----------------------------------------------------------------------
# INPUTS:
# A: (Nr by Nc) LinearOperator (for A*matrix) or matrix
# AT: (Nc by Nr) LinearOperator (for AT*matrix) or matrix
# (A and AT can also be specified as a dense/sparse matrix (for test purposes)),
# ktrunc: (integer > 0) number of retained singular values
# ell1: (integer) amount of oversampling for range
# ell2: (integer) amount of oversampling for co-range (ell2 >= ell1 >= 0)
# SketchOrth: (True or False) determines whether to orthonormalize the 
# random sampling matrices used to form the range and co-range sketches
# Parallel: (True or False) When True, then find the range and co-range 
# sketches simultaneously in parallel. Use Parallel=False, for serial version
# but NOTE! that this is just for testing purposes, since using pool.map has
# overheads which make the parallel implementation slow for small test
# problems which use an explicit input matrix.
# The 1-view method, with Parallel=True, is meant for applications
# where there are separate functions (which are computationally expensive) 
# for evaluating A times a matrix and AT times a matrix. 
# ----------------------------------------------------------------------
# RETURNS: approximate rank-ktrunc TSVD, Uk*diag(sk)*Vk^T, of A:
# sk: approximate top ktrunc singular values of A
# Uk: (Nr by ktrunc) matrix, where the ith column approximates the ith left singular vector
# Vk: (ktrunc by Nc) matrix, where the ith column approximates the ith right singular vector
# ----------------------------------------------------------------------     
def rand1viewWoolfeMinVar(A, AT, ktrunc, ell1, ell2, SketchOrth=False, Parallel=False):
    A = aslinearoperator(A)
    AT = aslinearoperator(AT)
    if Parallel:
        # Parallel evaluation of range and co-range sketches:
        pool = Pool(processes=2)
        SketchResults = pool.map( RangeOrCoRangeSketch , 
            [ [A, ktrunc, ell1, ell1, SketchOrth, 'rangeMinVar'], 
            [AT, ktrunc, ell2, ell2, SketchOrth, 'corangeWoolfe'] ])
        pool.terminate()
        [Qc, [Qr, OmegaCT, YrT]]  = SketchResults
    else:
        # Serial evaluation of range and co-range sketches:
        Qc = RangeOrCoRangeSketch([A, ktrunc, ell1, ell1, SketchOrth, 'rangeMinVar'])
        [Qr, OmegaCT, YrT] = RangeOrCoRangeSketch([AT, ktrunc, ell2, ell2, SketchOrth, 'corangeWoolfe'])
       
    # Find "minimum variance" ellCut:
    OmegaCTdotQc = OmegaCT.dot(Qc)
    YrTQr = YrT.dot(Qr)
    ellCut = getMinVarEllCutWoolfe(ktrunc, ell1, OmegaCTdotQc, YrTQr)
    
    Qc = Qc[:,0:(ktrunc+ellCut)]
    [Qhat, Rhat] = sp.linalg.qr(OmegaCTdotQc[:,0:(ktrunc+ellCut)], mode='economic')
    # Solve small linear problem Rhat*X = (Qhat^T)*YrT, where Rhat is upper triangular:
    X = solve_triangular(Rhat, sp.dot(Qhat.transpose(), YrTQr), trans=0, lower=False)
    # SVD of small matrix X:
    [Uhatk, sk, Vhatk] = rank_p_SVD(X, ktrunc)
    Uk = Qc.dot(Uhatk)
    Vk = Qr.dot(Vhatk)
    return sk, Uk, Vk
    
    
# randGenBlockKrylov(A, AT, ktrunc, ell, views): 
# Algorithm 9 in Bjarkason (2018)
# A general randomized algorithm for estimating a TSVD of matrix A,
# by accesing the matrix views >= 2 times and using a generalized 
# block Krylov scheme.
# ----------------------------------------------------------------------
# INPUTS:
# A: (Nr by Nc) LinearOperator (for A*matrix) or matrix
# AT: (Nc by Nr) LinearOperator (for AT*matrix), matrix or None
# !!! Note on A and AT: When specifying A as a dense matrix, then 
# set AT=None. Other option is to specify A and AT as linear operators for
# evaluating A times a matrix and A transposed times a matrix.
# ktrunc: (integer > 0) number of retained singular values
# ell: (integer >= 0) amount of oversampling
# views: (integer >= 2) the number of matrix views.
# ----------------------------------------------------------------------
# RETURNS: approximate rank-ktrunc TSVD, Uk*diag(sk)*Vk^T, of A:
# sk: approximate top ktrunc singular values of A
# Uk: (Nr by ktrunc) matrix, where the ith column approximates the ith left singular vector
# Vk: (Nc by ktrunc) matrix, where the ith column approximates the ith right singular vector
# ----------------------------------------------------------------------    
def randGenBlockKrylov(A, AT, ktrunc, ell, views):
    if (views < 2):
        print 'Exited without result: views needs to be 2 or greater'
        raise SystemExit
    Nr, Nc = A.shape
    Qr = sp.random.randn(Nc, ktrunc + ell)
    # Set up matrices for storing Krylov subspaces for
    # the range Kc (views is even) or the co-range Kr (views is odd):
    if (views%2) == 0:
        Kc = sp.zeros((Nr, (views/2)*(ktrunc+ell)))
    else:
        Kr = sp.zeros((Nc, ((views-1)/2)*(ktrunc+ell)))
    if AT is None:
        for j in range(0, views-1):
            if (j%2) == 0:
                Qc = A.dot(Qr)
                if j < views-2:
                    # Orthonormalize:
                    Qc = sp.linalg.qr(Qc, mode='economic')[0]
                if (views%2) == 0:
                    lowerIndx = (j/2)*(ktrunc+ell)
                    upperIndx = ((j/2) + 1)*(ktrunc+ell)
                    Kc[:, lowerIndx:upperIndx] = Qc
            else:
                Qr = sp.dot( Qc.transpose(), A).transpose()
                if j < views-2:
                    # Orthonormalize:
                    Qr = sp.linalg.qr(Qr, mode='economic')[0]
                if (views%2) == 1:
                    lowerIndx = ((j-1)/2)*(ktrunc+ell)
                    upperIndx = (((j-1)/2) + 1)*(ktrunc+ell)
                    Kr[:, lowerIndx:upperIndx] = Qr
    else:
        A = aslinearoperator(A)
        AT = aslinearoperator(AT)
        for j in range(0, views-1):
            if (j%2) == 0:
                Qc = A.matmat(Qr)
                if j < views-2:
                    # Orthonormalize:
                    Qc = sp.linalg.qr(Qc, mode='economic')[0]
                if (views%2) == 0:
                    lowerIndx = (j/2)*(ktrunc+ell)
                    upperIndx = ((j/2) + 1)*(ktrunc+ell)
                    Kc[:, lowerIndx:upperIndx] = Qc
            else:
                Qr = AT.matmat(Qc)
                if j < views-2:
                    # Orthonormalize:
                    Qr = sp.linalg.qr(Qr, mode='economic')[0]
                if (views%2) == 1:
                    lowerIndx = ((j-1)/2)*(ktrunc+ell)
                    upperIndx = (((j-1)/2) + 1)*(ktrunc+ell)
                    Kr[:, lowerIndx:upperIndx] = Qr
    if (views%2) == 0: 
        Qc = sp.linalg.qr(Kc, mode='economic')[0]
        if AT is None:
            Qr = sp.dot( Qc.transpose(), A).transpose()
        else:
            Qr = AT.matmat(Qc)
        [Qr, Rr] = sp.linalg.qr(Qr, mode='economic')
        [Vhatk, sk, Uhatk] = rank_p_SVD(Rr, ktrunc)
    else:
        Qr = sp.linalg.qr(Kr, mode='economic')[0]
        if AT is None:
            Qc = A.dot(Qr)
        else:
            Qc = A.matmat(Qr)
        [Qc, Rc] = sp.linalg.qr(Qc, mode='economic')
        [Uhatk, sk, Vhatk] = rank_p_SVD(Rc, ktrunc)
    Uk = Qc.dot(Uhatk)
    Vk = Qr.dot(Vhatk)
    return sk, Uk, Vk