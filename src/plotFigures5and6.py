# Plot Figures 5 and 6 comparing the gen. subspace iteration Algorithm 2, 
# in Bjarkason (2018), with the gen. block Krylov Algorithm 9.
# Coded by: Elvar K. Bjarkason (2018)

import scipy as sp
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
import os, string
from math import ceil, floor

def plotFigures5and6(subfolder, FigFolder, rank, subfigIndx, subFigName, Indx, oversample, FigNumber, markerStyle, markerColour, matName):
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
    matplotlib.rcParams['axes.linewidth'] = 1.0 #set the value globally
    
    # Make LATEX font the same as text font:
    matplotlib.rcParams['mathtext.fontset'] = 'custom'
    matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
    matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
    matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

    # Get Spectral errors:
    MeanSpecErrOnJHalfMore     = sp.load(subfolder + '/' + 'MeanSpecErrOnJHalfMore_'     + str(int(oversample)) + 'over.npy')
    MeanSpecErrOnJ     = sp.load(subfolder + '/' + 'MeanSpecErrOnJ_'     + str(int(oversample)) + 'over.npy')
    BlkMeanSpecErrOnJ     = sp.load(subfolder + '/' + 'BlkMeanSpecErrOnJ_'     + str(int(oversample)) + 'over.npy')
    # Get sketch sizes:
    ViewVec = sp.load(subfolder + '/' + 'ViewVec.npy')
    
    # Get min and max values:
    Errs = [MeanSpecErrOnJ, BlkMeanSpecErrOnJ, MeanSpecErrOnJHalfMore]
    ymin = 1.e99
    ymax = -1.e99
    for Err in Errs:
        emax = max(Err)
        if emax > ymax:
            ymax = emax
        emin = min(Err)
        if emin < ymin:
            ymin = emin
    
    fig = plt.figure(1, figsize=(12, 12))
    ax = plt.subplot(subfigIndx)
    plt.plot(ViewVec, BlkMeanSpecErrOnJ, 'bo-', markersize=8, markerfacecolor="None", markeredgecolor='b', markeredgewidth=2, linewidth=1.5)
    plt.plot(ViewVec, MeanSpecErrOnJ, 'ks--', markersize=8, markerfacecolor="None", markeredgecolor='k', markeredgewidth=2, linewidth=1.5)
    plt.plot(ViewVec, MeanSpecErrOnJHalfMore, 'rv-.', markersize=8, markerfacecolor="None", markeredgecolor='r', markeredgewidth=2, linewidth=1.5)
    plt.yscale('log')
    # For y-ticks:
    plt.yticks(sp.logspace(-16, 4, 6))
    plt.xticks(sp.linspace(2, 10, 5))
    plt.xlabel('Views, $v$', labelpad=10)
    plt.ylabel('Relative Spectral Error', labelpad=10)
    # Set y axis limits:
    plt.xlim(1.5, 10.5)
    plt.ylim(1.e-16, 3.0*ymax)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.text(-0.3, 1.1, '('+string.ascii_lowercase[Indx]+')', transform=ax.transAxes, weight='bold')
    ax.set_title(subFigName + ' ($p=$' + str(rank) + ')', y=1.02)
    if Indx == 4:
        ax.legend([ 'Block Krylov Alg. 9 ($l=10$)', 'Subspace Alg. 2 ($l=10$)', 'Subspace Alg. 2, $50$% more sampling'], numpoints=1, loc=9, bbox_to_anchor=(0.4, -0.24), ncol=5, frameon=False)
    fig.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99, wspace=0.4, hspace=0.4)
    fig.savefig(FigFolder + '/Figure5.pdf', format='pdf',bbox_inches='tight')
    
    


    fig = plt.figure(2, figsize=(12, 12))
    ax = plt.subplot(subfigIndx)
    plt.plot((ViewVec + (ViewVec/2) - 1)*(rank+oversample+0.)/(2*rank), BlkMeanSpecErrOnJ, 'bo-', markersize=8, markerfacecolor="None", markeredgecolor='b', markeredgewidth=2, linewidth=1.5)
    plt.plot(ViewVec*(rank+oversample+0.)/(2.*rank), MeanSpecErrOnJ, 'ks--', markersize=8, markerfacecolor="None", markeredgecolor='k', markeredgewidth=2, linewidth=1.5)
    plt.plot((3./2.)*ViewVec*(rank+oversample+0.)/(2.*rank), MeanSpecErrOnJHalfMore, 'rv-.', markersize=8, markerfacecolor="None", markeredgecolor='r', markeredgewidth=2, linewidth=1.5)
    plt.yscale('log')
    # For y-ticks:
    plt.yticks(sp.logspace(-16, 4, 6))
    plt.xticks(sp.linspace(0, 10, 6))
    plt.xlabel('Matrix-Vector Multiplications / $2p$', labelpad=10)
    plt.ylabel('Relative Spectral Error', labelpad=10)
    # Set y axis limits:
    plt.xlim(0., 10.)
    plt.ylim(1.e-16, 3.0*ymax)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.text(-0.3, 1.1, '('+string.ascii_lowercase[Indx]+')', transform=ax.transAxes, weight='bold')
    ax.set_title(subFigName + ' ($p=$' + str(rank) + ')', y=1.02)
    if Indx == 4:
        ax.legend([ 'Block Krylov Alg. 9 ($l=10$)', 'Subspace Alg. 2 ($l=10$)', 'Subspace Alg. 2, $50$% more sampling'], numpoints=1, loc=9, bbox_to_anchor=(0.4, -0.24), ncol=5, frameon=False)
    fig.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99, wspace=0.4, hspace=0.4)
    fig.savefig(FigFolder + '/Figure6.pdf', format='pdf',bbox_inches='tight')
    
    

if __name__ == "__main__":
    MainSubfolder = 'Results/GenPowError/'
    
    mats = ['lowRankMedNoiseR10', 'lowRankHiNoiseR10', 'polySlowR10', 'polySlowR10', 'SDlarge', 'SDlarge']
    subFigNames = ['LowRankMedNoise', 'LowRankHiNoise', 'PolySlow', 'PolySlow', '$S_\mathrm{D}$', '$S_\mathrm{D}$']
    markerStyles = ['o-', 'v--', 'x-.', '+:', 's--', 'd:', '<:',]
    markerColours = ['b', 'r', 'g', 'k', 'k', 'c', 'm']
    ranks = [10]*len(mats)
    ranks[3] = 50
    ranks[5] = 50
    OversVec = [10]
    # For plotting specify sub-figure indices:
    subfigIndices = [331, 332, 333, 334, 335, 336, 337, 338, 339]
    
    FigNumber = 1
    for oversample in OversVec:
        FigFolder = 'Figures'
        newpath = os.getcwd() + '/' + FigFolder
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        for i, matName in enumerate(mats):    
            rank = ranks[i]
            matName = matName + '_Rank' + str(rank)
            subfolder = MainSubfolder + matName 
                
            # Plot:
            plotFigures5and6(subfolder, FigFolder, rank, subfigIndices[i], subFigNames[i], i, oversample, FigNumber, markerStyles[i], markerColours[i], matName)
        FigNumber += 1
                
            
            
            
            