# Load in a number of libraries we will use
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import copy
import pickle
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib import colors
import matplotlib
import os

use_latex   = False

if use_latex:
    
    from matplotlib import rc
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    rc('text', usetex=True)
    titlesize   = 14
    labelsize   = 12
    addendum    = "_latex"
    pad         = -20
    bigsize     = 22
    smallsize   = 10
    tinysize    = 8
    
else:
    
    matplotlib.style.use('default')
    titlesize   = 12
    labelsize   = 10
    addendum    = ""
    pad         = -25
    bigsize     = 18
    smallsize   = 8
    tinysize    = 6

root_directory = os.path.dirname(os.path.realpath(__file__))
    
plt.close('all')

T           = 1000
repeats     = 100

Ns          = [50,75,100,150,250,500,1000]

plt.figure(figsize=(12,4))

RMSE_N_sparse    = np.zeros((len(Ns)))
RMSE_N_dense     = np.zeros((len(Ns)))
RMSE_N_EnKF      = np.zeros((len(Ns)))
RMSE_N_EnKF_sa   = np.zeros((len(Ns)))

MCSE_N_sparse    = np.zeros((len(Ns)))
MCSE_N_dense     = np.zeros((len(Ns)))
MCSE_N_EnKF      = np.zeros((len(Ns)))
MCSE_N_EnKF_sa   = np.zeros((len(Ns)))


# Check both ensemble sizes
for i,N in enumerate(Ns):
    
    # Reset RMSE containers
    RMSEs_sparse    = np.zeros((repeats,T))
    RMSEs_dense     = np.zeros((repeats,T))
    RMSEs_EnKF      = np.zeros((repeats,T))
    RMSEs_EnKF_sa   = np.zeros((repeats,T))
    
    colors1 = ['xkcd:orangish red','xkcd:cerulean','xkcd:grass green','xkcd:tangerine']
    colors2 = ['xkcd:crimson','xkcd:cobalt','xkcd:pine','xkcd:deep orange']
    colors2 = ['xkcd:tangerine','xkcd:sky blue','xkcd:pine','xkcd:deep orange']

    # Load in the results from every repeat directory
    for rep in range(repeats):
        
        # Load results dictionary
        dct     = pickle.load(
            open(
                "filter_comparison_output_dictionary"+"_N="+str(N).zfill(4)+"_rep="+str(rep).zfill(4)+".p",
                "rb"))
        
        # Store the results
        RMSEs_sparse[rep,:]     = copy.copy(np.asarray(dct["RMSE_list_sparse"]))
        RMSEs_dense[rep,:]      = copy.copy(np.asarray(dct["RMSE_list_dense"]))
        RMSEs_EnKF[rep,:]       = copy.copy(np.asarray(dct["RMSE_list_EnKF_empirical"]))
        RMSEs_EnKF_sa[rep,:]    = copy.copy(np.asarray(dct["RMSE_list_EnKF_semiempirical"]))
        
    RMSE_N_sparse[i]    = np.mean(RMSEs_sparse)
    RMSE_N_dense[i]     = np.mean(RMSEs_dense)
    RMSE_N_EnKF[i]      = np.mean(RMSEs_EnKF)
    RMSE_N_EnKF_sa[i]   = np.mean(RMSEs_EnKF_sa)
    
    MCSE_N_sparse[i]    = np.std(RMSEs_sparse)/np.sqrt(T*repeats)*1.96
    MCSE_N_dense[i]     = np.std(RMSEs_dense)/np.sqrt(T*repeats)*1.96
    MCSE_N_EnKF[i]      = np.std(RMSEs_EnKF)/np.sqrt(T*repeats)*1.96
    MCSE_N_EnKF_sa[i]   = np.std(RMSEs_EnKF_sa)/np.sqrt(T*repeats)*1.96
    
plt.plot(
    Ns,
    RMSE_N_dense,
    color = colors1[0],
    label   = 'EnTF (dense)',
    marker  = 'o')

plt.plot(
    Ns,
    RMSE_N_EnKF,
    color = colors1[1],
    label   = 'EnKF (empirical)',
    ls      = '--',
    zorder  = 10,
    markersize = 10,
    marker  = 'x')

    
plt.plot(
    Ns,
    RMSE_N_sparse,
    color = colors2[0],
    label   = 'EnTF (sparse)',
    marker  = 's')

plt.plot(
    Ns,
    RMSE_N_EnKF_sa,
    color = colors2[1],
    label   = 'EnKF (semi-empirical)',
    marker  = 'v')

plt.legend(frameon = False, fontsize = labelsize)

plt.xlabel('ensemble size (log scale)', fontsize = labelsize)
plt.ylabel('time-average RMSE', fontsize = labelsize)

ylim = plt.gca().get_ylim()
plt.ylim([ylim[0],0.8])
plt.xlim([50,250])
plt.xscale('log')
plt.gca().set_xticks(Ns, fontsize = labelsize)
plt.gca().set_xticklabels(Ns, fontsize = labelsize)
plt.yticks(fontsize = labelsize)

plt.savefig('filter_comparison_ensemble_repeats='+str(repeats)+addendum+'.png',dpi=600,bbox_inches='tight')
plt.savefig('filter_comparison_ensemble_repeats='+str(repeats)+addendum+'.pdf',dpi=600,bbox_inches='tight')
    