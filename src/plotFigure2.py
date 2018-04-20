# Plot Figure 2 showing the improvement in accuracy when increasing
# the number of views used by the gen. subspace iteration Algorithm 2.
# Coded by: Elvar K. Bjarkason (2018)

import scipy as sp
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
import os, string
from math import ceil, floor

def plotFigure2(subfolder, FigFolder, rank, subfigIndx, subFigName, Indx, oversample, FigNumber, markerStyle, markerColour):
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
    matplotlib.rcParams['ytick.minor.size'] = 0
    matplotlib.rcParams['ytick.minor.width'] = 0.0
    matplotlib.rcParams['axes.linewidth'] = 1.0
    
    # Make LATEX font the same as text font:
    matplotlib.rcParams['mathtext.fontset'] = 'custom'
    matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
    matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
    matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

    # Get Spectral errors:
    MeanSpecErrOnJ     = sp.load(subfolder + '/' + 'MeanSpecErrOnJ_'     + str(int(oversample)) + 'over.npy')
    MeanFrobErrOnJ     = sp.load(subfolder + '/' + 'MeanFrobErrOnJ_'     + str(int(oversample)) + 'over.npy')
    # Get sketch sizes:
    ViewVec = sp.load(subfolder + '/' + 'ViewVec.npy')
    
    fig = plt.figure(FigNumber, figsize=(10, 4))
    ax = plt.subplot(121)
    plt.plot(ViewVec, MeanSpecErrOnJ, markerStyle+markerColour, markersize=8, markerfacecolor=markerColour, markeredgecolor=markerColour, markeredgewidth=2, linewidth=1.5)
    plt.yscale('log')
    # For y-ticks:
    plt.yticks(sp.logspace(-14, 0, 8))
    plt.xticks(sp.linspace(2, 8, 7))
    plt.xlabel('Views, $v$', labelpad=10)
    plt.ylabel('Relative Spectral Error', labelpad=10)
    # Set y axis limits:
    plt.ylim(1.e-14, 1.e1)
    plt.xlim(1.5, 8.5)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.text(-0.3, 1.1, '('+string.ascii_lowercase[0]+')', transform=ax.transAxes, weight='bold')
    
    ax = plt.subplot(122)
    plt.plot(ViewVec, MeanFrobErrOnJ, markerStyle+markerColour, markersize=8, markerfacecolor=markerColour, markeredgecolor=markerColour, markeredgewidth=2, linewidth=1.5)
    plt.yscale('log')
    # For y-ticks:
    plt.yticks(sp.logspace(-14, 0, 8))
    plt.xticks(sp.linspace(2, 8, 7))
    plt.xlabel('Views, $v$', labelpad=10)
    plt.ylabel('Relative Frobenius Error', labelpad=10)
    # Set y axis limits:
    plt.ylim(1.e-14, 1.e1)
    plt.xlim(1.5, 8.5)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.text(-0.3, 1.1, '('+string.ascii_lowercase[1]+')', transform=ax.transAxes, weight='bold')
    
    ax.legend(['$S_\mathrm{D}$', 'LowRankMedNoise', 'LowRankHiNoise', 'PolySlow', 'PolyFast'], numpoints=1, loc="upper right", prop={'size': 12})#, edgecolor='k')
    fig.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99, wspace=0.4, hspace=0.4)
    fig.savefig(FigFolder + '/Figure2.pdf', format='pdf',bbox_inches='tight')
    

if __name__ == "__main__":
    MainSubfolder = 'Results/GenSubError/'
    
    mats = ['SDlarge', 'lowRankMedNoiseR10', 'lowRankHiNoiseR10', 'polySlowR10', 'polyFastR10']
    subFigNames = ['SDlarge', 'LowRankMedNoise', 'LowRankHiNoise', 'PolyDecaySlow', 'PolyDecayFast']
    markerStyles = ['o-', 'v--', 'x-.', '+:', 's--', 'd:', '<:',]
    markerColours = ['b', 'r', 'g', 'k', 'k', 'c', 'm']
    ranks = [10]
    OversVec = [10]
    # For plotting specify sub-figure indices:
    subfigIndices = [331, 332, 333, 334, 335, 336, 337, 338, 339]
    
    FigNumber = 1
    for rank in ranks:
        for oversample in OversVec:
            FigFolder = 'Figures'
            newpath = os.getcwd() + '/' + FigFolder
            if not os.path.exists(newpath):
                os.makedirs(newpath)
            for i, matName in enumerate(mats):    
                matName = matName + '_Rank' + str(rank)
                subfolder = MainSubfolder + matName 
                    
                # Plot:
                plotFigure2(subfolder, FigFolder, rank, subfigIndices[i], subFigNames[i], i, oversample, FigNumber, markerStyles[i], markerColours[i])
            FigNumber += 1
            
            
            
            