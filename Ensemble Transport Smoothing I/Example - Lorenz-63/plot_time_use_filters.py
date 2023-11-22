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

use_latex   = True

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



time_N_sparse    = np.zeros((len(Ns)))
time_N_dense     = np.zeros((len(Ns)))
time_N_EnKF      = np.zeros((len(Ns)))
time_N_EnKF_sa   = np.zeros((len(Ns)))

colors1 = ['xkcd:orangish red','xkcd:cerulean','xkcd:grass green','xkcd:tangerine']
colors2 = ['xkcd:crimson','xkcd:cobalt','xkcd:pine','xkcd:deep orange']
colors2 = ['xkcd:tangerine','xkcd:sky blue','xkcd:pine','xkcd:deep orange']

# Check both ensemble sizes
for i,N in enumerate(Ns):
    
    # Reset time containers
    times_sparse    = np.zeros((repeats,T))
    times_dense     = np.zeros((repeats,T))
    times_EnKF      = np.zeros((repeats,T))
    times_EnKF_sa   = np.zeros((repeats,T))
    
    # Load in the results from every repeat directory
    for rep in range(repeats):
        
        print("N = "+str(N)+" | "+"rep = "+str(rep))
        
        # Load results dictionary
        dct     = pickle.load(
            open(
                "filter_comparison_output_dictionary"+"_N="+str(N).zfill(4)+"_rep="+str(rep).zfill(4)+".p",
                "rb"))
        
        # Store the results
        times_sparse[rep,:]     = copy.copy(np.asarray(dct["time_list_sparse"]))
        times_dense[rep,:]      = copy.copy(np.asarray(dct["time_list_dense"]))
        times_EnKF[rep,:]       = copy.copy(np.asarray(dct["time_list_EnKF_empirical"]))
        times_EnKF_sa[rep,:]    = copy.copy(np.asarray(dct["time_list_EnKF_semiempirical"]))
        
        
    time_N_sparse[i]    = np.mean(times_sparse)
    time_N_dense[i]     = np.mean(times_dense)
    time_N_EnKF[i]      = np.mean(times_EnKF)
    time_N_EnKF_sa[i]   = np.mean(times_EnKF_sa)
    
    
plt.figure(figsize=(12,4))

ind = np.arange(len(Ns))

ax1 = plt.gca()
# ax2 = ax1.twinx()

ax1.bar(
    ind - 0.3,
    time_N_EnKF,
    color = colors1[1],
    label   = 'EnKF (empirical)',
    width = 0.15)

ax1.bar(
    ind - 0.1,
    time_N_EnKF_sa,
    color = colors2[1],
    label   = 'EnKF (semi-empirical)',
    width = 0.15)

ax1.bar(
    ind + 0.1,
    time_N_dense,
    color = colors1[0],
    label   = 'EnTF (dense)',
    width = 0.15)

ax1.bar(
    ind + 0.3,
    time_N_sparse,
    color = colors2[0],
    label   = 'EnTF (sparse)',
    width = 0.15)

ax1.set_yscale("log")

ax1.set_xticks(ind)
ax1.set_xticklabels([str(N) for N in Ns])
ax1.set_xlabel("ensemble size", fontsize = labelsize)
ax1.set_ylabel("average computational time"+"\n"+"per update [s]", fontsize = labelsize)

plt.legend(frameon = False, fontsize = labelsize, loc = "upper center", bbox_to_anchor=(0.5, 1.12), ncols = 4)


plt.savefig('L63_time_demand'+addendum+'.png',dpi=600,bbox_inches='tight')
plt.savefig('L63_time_demand'+addendum+'.pdf',dpi=600,bbox_inches='tight')


