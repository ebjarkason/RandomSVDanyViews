# Plot Figures 9 and 10 comparing the "best" performance of selected
# 1-view schemes with some practical implementations.
# Coded by: Elvar K. Bjarkason (2018)

import scipy as sp
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
import os, string
from math import ceil, floor

def plotFigures9and10(subfolder, FigFolder, rank, subfigIndx, subFigName, Indx):
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
#    matplotlib.rcParams['text.usetex'] = True
#    matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{mathtools}']
#    matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']
    matplotlib.rcParams['mathtext.fontset'] = 'custom'
    matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
    matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
    matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'

    # Get Frob errors:
    MeanErrStandardTroppORACLE = sp.load(subfolder + '/' + 'TroppMeanFrobErrStandardORACLE.npy')
    MeanErrStandardTroppBEST = sp.load(subfolder + '/' + 'TroppMeanFrobErrStandardBEST.npy')
    MeanErrEll1Ell2EqualEllCutORACLE = sp.load(subfolder + '/' + 'TroppMeanFrobErrEll1Ell2EqualEllCutORACLE.npy')
    MeanErrEll1Ell2EqualEllCutBEST = sp.load(subfolder + '/' + 'TroppMeanFrobErrEll1Ell2EqualEllCutBEST.npy')
    MeanErrBwzORACLE = sp.load(subfolder + '/' + 'MeanFrobErrBwzORACLE.npy')
    MeanErrBwzBEST = sp.load(subfolder + '/' + 'MeanFrobErrBwzBEST.npy')
    MeanErrBwzORACLEV2 = sp.load(subfolder + '/' + 'MeanFrobErrBwzORACLEv2.npy')
    MeanErrBwzBESTV2 = sp.load(subfolder + '/' + 'MeanFrobErrBwzBESTv2.npy')
    MeanErrBwzV2Afac8 = sp.load(subfolder + '/' + 'MeanFrobErrBwzV2Afac8.npy')
    MeanErrEll1Ell2EqualEllCutMinVar     = sp.load(subfolder + '/' + 'TroppMeanFrobErrEll1Ell2EqualEllCutMinVar.npy')
    # Get sketch sizes:
    Tvec = sp.load(subfolder + '/' + 'Tvec.npy')
    
    # Get min and max values:
    Errs = [MeanErrStandardTroppBEST, MeanErrEll1Ell2EqualEllCutBEST, MeanErrBwzBEST, MeanErrBwzBESTV2, MeanErrBwzV2Afac8, MeanErrEll1Ell2EqualEllCutMinVar]
    ymin = 1.e99
    ymax = -1.e99
    for Err in Errs:
        emax = max(Err)
        if emax > ymax:
            ymax = emax
        emin = min(Err)
        if emin < ymin:
            ymin = emin
    
    fig = plt.figure(rank, figsize=(12, 12))
    ax = plt.subplot(subfigIndx)
    plt.plot(Tvec, MeanErrStandardTroppBEST, 'b-', markersize=8, markerfacecolor=None,markeredgecolor='r', linewidth=1.5)
    plt.plot(Tvec, MeanErrEll1Ell2EqualEllCutBEST, 'k--', markersize=8, markerfacecolor=None,markeredgecolor='k', linewidth=1.5)
    plt.plot(Tvec, MeanErrBwzBEST, 'gd:', markersize=8, markerfacecolor="None", markeredgecolor='g', markeredgewidth=2, linewidth=1.5)
    plt.plot(Tvec, MeanErrBwzBESTV2, 'rx:', markersize=8, markerfacecolor='r', markeredgecolor='r', markeredgewidth=2, linewidth=1.5)
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
    plt.xlabel('$T$', labelpad=10)
    plt.ylabel('Relative Frobenius Error', labelpad=10)
    # Set y axis limits:
    plt.ylim(max(1.e-10, ymin*0.6), ymax*1.4)
    plt.xlim(14, 300)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.minorticks_off()
    ax.set_title(subFigName, y=1.02)
    ax.text(-0.3, 1.1, '('+string.ascii_lowercase[Indx]+')', transform=ax.transAxes, weight='bold')
    if Indx == 4:
        ax.legend(['Alg. 7, $l_\mathrm{c} = l_1$', 'Alg. 7, $l_1 \simeq l_2$', 'Bwz (6.17)', 'Bwz2 (6.18)'], numpoints=1, loc=9, bbox_to_anchor=(0.4, -0.24), ncol=4, frameon=False)#, prop={'size': 12}))
    fig.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99, wspace=0.4, hspace=0.4)
    fig.savefig(FigFolder + '/Figure9.pdf', format='pdf',bbox_inches='tight',dpi=300)
    
    
    fig = plt.figure(rank+100, figsize=(12, 12))
    ax = plt.subplot(subfigIndx)
    plt.plot(Tvec, MeanErrEll1Ell2EqualEllCutBEST, 'k--', markersize=8, markerfacecolor=None,markeredgecolor='k', linewidth=1.5)
    plt.plot(Tvec, MeanErrEll1Ell2EqualEllCutMinVar,  'b<-.', markersize=8, markerfacecolor="None", markeredgecolor='b', markeredgewidth=2, linewidth=1.5)
    plt.plot(Tvec, MeanErrBwzBESTV2, 'rx:', markersize=8, markerfacecolor='r', markeredgecolor='r', markeredgewidth=2, linewidth=1.5)
    plt.plot(Tvec, MeanErrBwzV2Afac8, 'ko:', markersize=8, markerfacecolor="None", markeredgecolor='k', markeredgewidth=2, linewidth=1.5)
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
    plt.xlabel('$T$', labelpad=10)
    plt.ylabel('Relative Frobenius Error', labelpad=10)
    # Set y axis limits:
    plt.ylim(max(1.e-10, ymin*0.6), ymax*1.4)
    plt.xlim(14, 300)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.minorticks_off()
    ax.set_title(subFigName, y=1.02)
    ax.text(-0.3, 1.1, '('+string.ascii_lowercase[Indx]+')', transform=ax.transAxes, weight='bold')
    if Indx == 4:
        ax.legend(['Best Alg. 7, $l_1 \simeq l_2$', 'Alg. 7, $l_\mathrm{c}$ Min. Var.', 'Best Bwz2 (6.18)', 'Bwz2 (6.18-19)'], numpoints=1, loc=9, bbox_to_anchor=(0.4, -0.24), ncol=4, frameon=False)#, prop={'size': 12}))
    fig.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99, wspace=0.4, hspace=0.4)
    fig.savefig(FigFolder + '/Figure10.pdf', format='pdf',bbox_inches='tight',dpi=300)
    

if __name__ == "__main__":
    MainSubfolder = 'Results/1viewSampling/'
    
    mats = ['lowRankMedNoiseR10', 'lowRankHiNoiseR10', 'polySlowR10', 'polyFastR10', 'expSlowR10', 'expFastR10']
    subFigNames = ['LowRankMedNoise', 'LowRankHiNoise', 'PolySlow', 'PolyFast', 'ExpSlow', 'ExpFast']
    ranks = [5]
    # For plotting specify sub-figure indices:
    subfigIndices = [331, 332, 333, 334, 335, 336, 337, 338, 339]
    
    for rank in ranks:
        FigFolder = 'Figures'
        newpath = os.getcwd() + '/' + FigFolder
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        for i, matName in enumerate(mats):    
            matName = matName + '_Rank' + str(rank)
            subfolder = MainSubfolder + matName 
                
            # Plot:
            plotFigures9and10(subfolder, FigFolder, rank, subfigIndices[i], subFigNames[i], i)