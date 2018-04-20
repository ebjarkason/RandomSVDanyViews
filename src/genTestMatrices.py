# Functions for generating some test matrices
# Coded by: Elvar K. Bjarkason (2018)

import scipy as sp

# matrixLowRankPlusNoise
# ---------------------------------------------------------
# INPUTS:
# N: matrix dimension
# eta: noise level
# R: number of diagonal elements equal to 1, default 10
# ---------------------------------------------------------
# RETURNS:
# A: N by N matrix (low-rank with noise)
# ---------------------------------------------------------
def matrixLowRankPlusNoise(N, eta, R=10):
    A = sp.zeros((N, N))
    A[0:R,0:R] = sp.eye(R)
    G = sp.random.randn(N, N)
    A += ( ( (eta * R) / (2.*(N**2.) ) )**(0.5) ) * ( G + G.transpose() )
    return A
    
# matrixPolyDecay
# ---------------------------------------------------------
# INPUTS:
# N: matrix dimension
# p: decay rate
# R: number of diagonal elements equal to 1, default 10
# ---------------------------------------------------------
# RETURNS:
# A: N by N matrix with Poly decaying spectrum
# ---------------------------------------------------------
def matrixPolyDecay(N, p, R=10):
    diagels = sp.zeros(N)
    diagels[0:R] = 1.
    diagels[R::] = ( sp.linspace(2, N - R + 1, N - R) )**(-p)
    A = sp.diag(diagels)
    return A
    
# matrixExpDecay
# ---------------------------------------------------------
# INPUTS:
# N: matrix dimension
# q: decay rate
# R: number of diagonal elements equal to 1, default 10
# ---------------------------------------------------------
# RETURNS:
# A: N by N matrix with Exp decaying spectrum
# ---------------------------------------------------------
def matrixExpDecay(N, q, R=10):
    diagels = sp.zeros(N)
    diagels[0:R] = 1.
    diagels[R::] = 10.**( - sp.linspace(1, N - R, N - R) * q)
    A = sp.diag(diagels)
    return A