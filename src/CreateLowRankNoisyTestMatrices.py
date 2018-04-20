# Create low-rank plus noise test matrices
# Coded by: Elvar K. Bjarkason (2018)

import scipy as sp
from genTestMatrices import *

N = 1000

# Generate and save medium noise test matrix:
sp.save('TestMatrices/lowRankMedNoiseR10.npy', matrixLowRankPlusNoise(N, 1.e-2, R=10))

# Generate and save medium noise test matrix:
sp.save('TestMatrices/lowRankHiNoiseR10.npy', matrixLowRankPlusNoise(N, 1., R=10))