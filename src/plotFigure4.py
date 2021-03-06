# Plot Figure 3 comparing the gen. subspace iteration Algorithm 2, 
# in Bjarkason (2018), with the prolonged and pinched sketching schemes
# when approximating a normal matrix JT*J.
# Coded by: Elvar K. Bjarkason (2018)

import scipy as sp
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
import os, string
from math import ceil, floor

def plotFigure4(subfolder, FigFolder, rank, subFigName, Indx, oversample, FigNumber,  FigRows, FigCols):
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
    matplotlib.rcParams['axes.linewidth'] = 1.0 #set the value globally
    
    # Make LATEX font the same as text font:
    matplotlib.rcParams['mathtext.fontset'] = 'custom'
    matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
    matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
    matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

    # Get Spectral errors:
    MeanErrOnSD     = sp.load(subfolder + '/' + 'MeanSpecErrOnJ_'     + str(int(oversample)) + 'over.npy')
    MeanErrOnSDT    = sp.load(subfolder + '/' + 'MeanSpecErrOnJT_'    + str(int(oversample)) + 'over.npy')
    MeanErrNystrom  = sp.load(subfolder + '/' + 'MeanSpecErrNystrom_' + str(int(oversample)) + 'over.npy')
    MeanErrPinched  = sp.load(subfolder + '/' + 'MeanSpecErrPinched_' + str(int(oversample)) + 'over.npy')
    # Get sketch sizes:
    ViewVec = sp.load(subfolder + '/' + 'ViewVec.npy')
    
    # Get min and max values:
    Errs = [MeanErrOnSD, MeanErrOnSDT, MeanErrNystrom, MeanErrPinched]
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
    ax = plt.subplot(FigRows, FigCols, FigNumber)
    plt.plot(ViewVec, MeanErrNystrom, 'k-.', markersize=11, markerfacecolor='k',markeredgecolor='k', markeredgewidth=2, linewidth=2.0)
    plt.plot(ViewVec, MeanErrPinched, 'k--', markersize=11, markerfacecolor='g',markeredgecolor='g', markeredgewidth=2, linewidth=2.0)
    plt.plot(ViewVec, MeanErrOnSD, 'rv', markersize=11, markerfacecolor='r',markeredgecolor='r', linewidth=1.5)
    plt.plot(ViewVec, MeanErrOnSDT,  'bo', markersize=11, markerfacecolor='b',markeredgecolor='b', linewidth=1.5)
    plt.yscale('log')
    # For y-ticks:
    ytksMin = floor(sp.log10(MeanErrNystrom[4]*0.5))
    ytksMax = ceil(sp.log10(MeanErrPinched[0]*2.))
    plt.xlabel('Views, $v$', labelpad=10)
    plt.ylabel('Relative Spectral Error', labelpad=10)
    # Set y axis limits:
    if sp.absolute(ytksMin - ytksMax) < 12.5:
        plt.yticks(sp.logspace(-16, 4, 11))
    else:
        plt.yticks(sp.logspace(-16, 4, 6))
    plt.ylim(10.**(ytksMin), 10.**(ytksMax))
    plt.xlim(1.5, 6.5)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.minorticks_off()
    ax.set_title('$S_\mathrm{D}$' + ' ($p=$'+str(rank)+', $l = $' + str(oversample) + ')', y=1.02)
    ax.text(-0.3, 1.1, '('+string.ascii_lowercase[Indx]+')', transform=ax.transAxes, weight='bold')
    if Indx == 7:
        ax.legend(['Prolonged', 'Pinched', 'Alg. 2 on $J$', 'Alg. 2 on $J^*$'], numpoints=1, loc=9, bbox_to_anchor=(0.4, -0.24), ncol=5, frameon=False)
    fig.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99, wspace=0.4, hspace=0.4)
    fig.savefig(FigFolder + '/Figure4.pdf', format='pdf',bbox_inches='tight')
    

if __name__ == "__main__":
    MainSubfolder = 'Results/GenSubOnNormal/'
    
    mats = ['SDlarge']
    subFigNames = ['SDlarge']*3
    ranks = [1, 10, 50]
    OversVec = [5, 10, 50]
    
    FigNumber = 1
    FigCols = len(OversVec)
    FigRows = len(ranks)
    FigFolder = 'Figures'
    for i, rank in enumerate(ranks):    
        matName = mats[0] + '_Rank' + str(rank)
        subfolder = MainSubfolder + matName 
        for oversample in OversVec:
            # Plot:
            plotFigure4(subfolder, FigFolder, rank, subFigNames[i], FigNumber-1, oversample, FigNumber, FigRows, FigCols)
            FigNumber += 1
            
            
            
            