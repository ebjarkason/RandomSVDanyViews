# Oversampling methods:
# Functions for choosing 1-view oversampling parameters
# given a target rank r and total sketch size T.
# The following oversampling schemes were proposed by Tropp et al. (2017),
# "PRACTICAL SKETCHING ALGORITHMS FOR LOW-RANK MATRIX APPROXIMATION"
#
# Note: the following functions are for the case where the matrix
# is real valued. For the case where the matrix is complex 
# refer to Tropp et al. (2018).
#
# Coded by: Elvar K. Bjarkason (2018)

import scipy as sp
from math import floor

# overparamsFlatSpectrum
# Oversampling scheme for a real valued matrix with a "flat" singular spectrum
# ---------------------------------------------------------
# INPUTS:
# r: fixed target rank of approximation
# T: total sketch size, i.e. T = 2*r + ell1 + ell2
# ---------------------------------------------------------
# RETURNS:
# ell1: oversampling parameter for range sketch (ell1 >= 0)
# ell2: oversampling parameter for co-range sketch (ell2 >= ell1)
# ---------------------------------------------------------
def overparamsFlatSpectrum(r, T):
    # Select oversampling parameters which are useful for a "flat" spectrum
    # see equation (4.8) in Tropp et al (2017)
    alpha = 1 # alpha parameter when using real numbers (see Tropp et al (2017))
    if T > (2*r + 3*alpha + 2):
        val = ( r * (T - r - 2.) * (1. - 2./(T - 1.)) )**(0.5)
        val -= r - 1.
        val /= T - 2*r -1.
        val *= T - 1.
        kstar = int( max(r + 2, floor(val) ) )
        ell1 = kstar - r
        ell2 = T - ell1 - 2*r
    else:
        ell1 = int( floor( (T - 2*r)/3 ) )
        ell2 = T - ell1 - 2*r
    return ell1, ell2    

# overparamsModerateDecay
# Oversampling scheme for a real valued matrix with a "moderately" decaying singular spectrum
# ---------------------------------------------------------
# INPUTS:
# r: fixed target rank of approximation
# T: total sketch size, i.e. T = 2*r + ell1 + ell2
# ---------------------------------------------------------
# RETURNS:
# ell1: oversampling parameter for range sketch (ell1 >= 0)
# ell2: oversampling parameter for co-range sketch (ell2 >= ell1)
# ---------------------------------------------------------
def overparamsModerateDecay(r, T):
    # Select oversampling parameters which are useful for a "moderately" decaying spectrum
    # see equation (4.9) in Tropp et al (2017)
    alpha = 1 # alpha parameter when using real numbers (see Tropp et al (2017))
    if T > (2*r + 3*alpha + 2):
        kstar = int( max( r + alpha + 1, floor((T-alpha)/3) ) )
        ell1 = kstar - r
        ell2 = T - ell1 - 2*r
    else:
        ell1 = int( floor( (T - 2*r)/3 ) )
        ell2 = T - ell1 - 2*r
    return ell1, ell2
    
# overparamsRapidDecay
# Oversampling scheme for a real valued matrix with a "rapidly decaying singular spectrum
# ---------------------------------------------------------
# INPUTS:
# r: fixed target rank of approximation
# T: total sketch size, i.e. T = 2*r + ell1 + ell2
# ---------------------------------------------------------
# RETURNS:
# ell1: oversampling parameter for range sketch (ell1 >= 0)
# ell2: oversampling parameter for co-range sketch (ell2 >= ell1)
# ---------------------------------------------------------
def overparamsRapidDecay(r, T):
    # Select oversampling parameters which are useful for a "rapidly" decaying spectrum
    # see equation (4.10) in Tropp et al (2017)
    alpha = 1 # alpha parameter when using real numbers (see Tropp et al (2017))
    if T > (2*r + 3*alpha + 2):
        kstar = int( floor( (T - alpha - 1)/2 ) )
        ell1 = kstar - r
        ell2 = T - ell1 - 2*r
    else:
        ell1 = int( floor( (T - 2*r)/3 ) )
        ell2 = T - ell1 - 2*r
    return ell1, ell2