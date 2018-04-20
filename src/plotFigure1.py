# Plot Figure 1 showing the normalized singular spectrum of the test matrices.
# Coded by: Elvar K. Bjarkason (2018)

import scipy as sp
from scipy.linalg import svd
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
import os, string
from math import ceil, floor
from genTestMatrices import *
from RSVDmethods import *

def plotFigureSingVals(FigFolder, s, plotStylei, subfigIndx):
    font = {'family' : 'sans-serif',
        'weight' : 'normal',
        'size'   : 14}

    matplotlib.rc('font', **font)
    
    # set tick width
    matplotlib.rcParams['xtick.major.size'] = 7
    matplotlib.rcParams['xtick.major.width'] = 1.0
    matplotlib.rcParams['xtick.minor.size'] = 3
    matplotlib.rcParams['xtick.minor.width'] = 1.0
    matplotlib.rcParams['ytick.major.size'] = 7
    matplotlib.rcParams['ytick.major.width'] = 1.0
    matplotlib.rcParams['ytick.minor.size'] = 3
    matplotlib.rcParams['ytick.minor.width'] = 1.0
    matplotlib.rcParams['axes.linewidth'] = 1.0 
    
    # Make LATEX font the same as text font:
    matplotlib.rcParams['mathtext.fontset'] = 'custom'
    matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
    matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
    matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

    
    fig = plt.figure(1, figsize=(10, 4))
    ax = plt.subplot(subfigIndx)
    plt.plot(range(1, len(s)+1), s, plotStylei, linewidth=2.0)
    plt.yscale('log')
    plt.xscale('log')
    ax.set_yticks(sp.logspace(-8, 0, num=5, base=10))
    plt.xlabel('Singular Value Index, $i$', labelpad=10)
    plt.ylabel('Scaled Singular Value, $\lambda_i/\lambda_1$', labelpad=10)
    # Set axis limits:
    plt.ylim(1.e-8, 3.)
    plt.xlim(0, 1000)
    ax.minorticks_off()  
    ax.tick_params(axis='both', bottom=True, top=True, left=True, right=True, direction='in')    
    if subfigIndx == 121:
        ax.legend(['LowRankMedNoise', 'LowRankHiNoise', '$S_\mathrm{D}$'], numpoints=1, loc="lower left", prop={'size': 12})
    if subfigIndx == 122:
        ax.legend(['PolySlow', 'PolyFast', 'ExpSlow', 'ExpFast'], numpoints=1, loc="lower left", prop={'size': 12})
    fig.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99, wspace=0.4, hspace=0.4)
    fig.savefig(FigFolder + '/Figure1.pdf', format='pdf',bbox_inches='tight')

def singvalsTestMat(matName):
    N = 1000
    
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
 #   elif matName == 'SDlarge':
 #       A = sp.load('TestMatrices/SD_6135obs_10years_24kParams.npy')  
    
    if matName == 'SDlarge':
        s = sp.load('TestMatrices/singvalsSDlarge.npy')
    else:
        s = svd(A, full_matrices = False, compute_uv=True, overwrite_a=False, check_finite=True)[1]
    return s

if __name__ == "__main__":

    mats = ['lowRankMedNoiseR10', 'lowRankHiNoiseR10', 'polySlowR10', 'polyFastR10', 'expSlowR10', 'expFastR10', 'SDlarge']
    plotStyle = ['r-', 'k--', 'r-', 'k--', 'b-.', 'g:', 'b-.']
    subfigIndices = [121]*len(mats)
    subfigIndices[2:6] = [122]*4
    
    for i, matName in enumerate(mats):
        FigFolder = 'Figures/'
        newpath = os.getcwd() + '/' + FigFolder
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        s = singvalsTestMat(matName)
        # Plot:
        plotFigureSingVals(FigFolder, s/s[0], plotStyle[i], subfigIndices[i])
        
        
        
        
        