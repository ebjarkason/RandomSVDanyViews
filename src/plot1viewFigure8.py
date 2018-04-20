# Plot Figure 8 comparing 1-view schemes using approx. equal sampling
# for the range and co-range sketches, i.e. Algorithm 7 in Bjarkason (2018)
# with ell1=ell2, with the baseline 1-view algorithm proposed by Tropp et al. (2017).
# Coded by: Elvar K. Bjarkason (2018)

import scipy as sp
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
import os, string
from math import ceil, floor

def plotFigure8(subfolder, FigFolder, rank, subfigIndx, subFigName, Indx):
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
    
    # Get Frob errors:
    MeanErrFlat    = sp.load(subfolder + '/' + 'TroppMeanFrobErrFlat.npy')
    MeanErrMedium  = sp.load(subfolder + '/' + 'TroppMeanFrobErrMedium.npy')
    MeanErrRapid   = sp.load(subfolder + '/' + 'TroppMeanFrobErrRapid.npy')
    MeanErrEll1Ell2EqualEllCutHalf = sp.load(subfolder + '/' + 'TroppMeanFrobErrEll1Ell2EqualEllCutHalf.npy')
    MeanErrEll1Ell2EqualEllCutMinVar     = sp.load(subfolder + '/' + 'TroppMeanFrobErrEll1Ell2EqualEllCutMinVar.npy')
    # Get sketch sizes:
    Tvec = sp.load(subfolder + '/' + 'Tvec.npy')
    
    # Get min and max values:
    Errs = [MeanErrFlat, MeanErrMedium, MeanErrRapid, MeanErrEll1Ell2EqualEllCutHalf, MeanErrEll1Ell2EqualEllCutMinVar]
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
    plt.plot(Tvec - 2*rank, MeanErrFlat, 'rs--', markersize=8, markerfacecolor="None", markeredgecolor='r', markeredgewidth=2, linewidth=1.5)
    plt.plot(Tvec - 2*rank, MeanErrMedium, 'yo:', markersize=8, markerfacecolor='y', markeredgecolor='y', markeredgewidth=2, linewidth=1.5)
    plt.plot(Tvec - 2*rank, MeanErrRapid, 'gx-.', markersize=8, markerfacecolor='g', markeredgecolor='g', markeredgewidth=2, linewidth=1.5)
    plt.plot(Tvec - 2*rank, MeanErrEll1Ell2EqualEllCutHalf,  'kv--', markersize=8, markerfacecolor="None", markeredgecolor='k', markeredgewidth=2, linewidth=1.5)
    plt.plot(Tvec - 2*rank, MeanErrEll1Ell2EqualEllCutMinVar,  'b<-.', markersize=8, markerfacecolor="None", markeredgecolor='b', markeredgewidth=2, linewidth=1.5)
    plt.yscale('log')
    plt.xscale('log', basex=2)
    # For y-ticks:
    ytksMin = ceil(sp.log10(ymax))
    ytksMax = floor(sp.log10(ymin))
    if sp.absolute(ytksMin - ytksMax) < 5.:
        plt.yticks(sp.logspace(-10, 2, 13))
    else:
        plt.yticks(sp.logspace(-10, 2, 7))
    xtks = sp.logspace(2, 8, num=7, base=2)
    plt.xticks(xtks, xtks)
    plt.xlabel('$l_1 + l_2$', labelpad=10)
    plt.ylabel('Relative Frobenius Error', labelpad=10)
    # Set y axis limits:
    plt.ylim(max(1.e-10, ymin*0.6), ymax*1.4)
    plt.xlim(4, 300)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.minorticks_off()
    ax.set_title(subFigName, y=1.02)
    if Indx == 7:
        ax.legend(['FLAT', 'DECAY', 'RAPID', '$l_\mathrm{c} = \lfloor l_1/2 \\rfloor$', '$l_\mathrm{c}$ Min. Var.'], numpoints=1, loc=9, bbox_to_anchor=(0.4, -0.24), ncol=5, frameon=False)
    ax.text(-0.3, 1.1, '('+string.ascii_lowercase[Indx]+')', transform=ax.transAxes, weight='bold')
    fig.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99, wspace=0.4, hspace=0.4)
    fig.savefig(FigFolder + '/Figure8.pdf', format='pdf',bbox_inches='tight')
    

if __name__ == "__main__":
    MainSubfolder = 'Results/1viewSampling/'
    
    mats = ['lowRankMedNoiseR10', 'lowRankHiNoiseR10', 'polySlowR10', 'polyFastR10', 'expSlowR10', 'expFastR10', 'SDlarge', 'SDlarge', 'SDlarge']
    subFigNames = ['LowRankMedNoise', 'LowRankHiNoise', 'PolySlow', 'PolyFast', 'ExpSlow', 'ExpFast', '$S_\mathrm{D}$ ($p = 1$)', '$S_\mathrm{D}$ ($p = 5$)', '$S_\mathrm{D}$ ($p = 50$)']
    ranks = [5]*9
    ranks[6] = 1
    ranks[8] = 50
    print ranks
    # For plotting specify sub-figure indices:
    subfigIndices = [331, 332, 333, 334, 335, 336, 337, 338, 339]
    
    FigFolder = 'Figures'
    newpath = os.getcwd() + '/' + FigFolder
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    for i, matName in enumerate(mats):  
        rank = ranks[i]
        matName = matName + '_Rank' + str(rank)
        subfolder = MainSubfolder + matName 
            
        # Plot:
        plotFigure8(subfolder, FigFolder, rank, subfigIndices[i], subFigNames[i], i)