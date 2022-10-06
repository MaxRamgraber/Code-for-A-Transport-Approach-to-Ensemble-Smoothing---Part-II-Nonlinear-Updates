import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import copy
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

plt.close('all')

from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

# Then we prepare a new colormap, mainly for cosmetic purposes.
cmap    = "turbo"
T = 30

def fillwrap(x,y=None):
    if y is None:
        y = x
    return np.asarray(list(x)+list(np.flip(y)))

xfill   = np.asarray(list(np.arange(T))+list(np.flip(np.arange(T))))

# define algorithms

en_algs = ['EnKS','EnTS (joint-analysis)','EnRTSS (single-pass)','EnTS (backward, single-pass)',\
    'EnRTSS (multi-pass)','EnTS (backward, multi-pass)','EnFIT (multi-pass)','EnTS (forward, multi-pass)']
algs_titles = ['EnKS', 'EnTS (dense)','EnRTSS\n (single-pass)','EnTS-BW\n (single-pass)',\
    'EnRTSS\n (multi-pass)','EnTS-BW\n (multi-pass)','EnFIT\n (multi-pass)','EnTS-FW\n (multi-pass)']

Ns = [100,1000]

# load results
subdct = pickle.load(open("autoregressive_processed.p","rb"))

# determine min and max values
attributes = ['mean_RMSE']
for attr in attributes:
    attr_min = []
    attr_max = []
    for N in Ns:
        for (k,alg) in enumerate(en_algs):
            attr_min.append(np.min(subdct[N][alg][attr]))
            attr_max.append(np.max(subdct[N][alg][attr]))
    subdct[attr+' min'] = np.min(attr_min)*0.9
    subdct[attr+' max'] = np.max(attr_max)*1.1
    
# Now, let us plot the results
colors = ['xkcd:orangish red','xkcd:orangish red','xkcd:cerulean','xkcd:cerulean','xkcd:grass green','xkcd:grass green','xkcd:yellow orange','xkcd:yellow orange']
styles = [':','--',':','--',':','--',':','--']

plt.figure(figsize=(8,2.5))
gs  = GridSpec(nrows=1, ncols=len(Ns))# width_ratios = [2]*2)#, wspace = 0.1, hspace = 0.1)

# Plot the RMSE
for (i,N) in enumerate(Ns):
    plt.subplot(gs[0,i])
    for (k,alg) in enumerate(en_algs):
        plt.plot(np.arange(T)+1,subdct[N][alg]['mean_RMSE'], color=colors[k], label=algs_titles[k], linestyle=styles[k])
    plt.xlim((0.5,T+0.5))
    plt.ylim((subdct['mean_RMSE min'], subdct['mean_RMSE max']))
    plt.ylabel('Error in ensemble mean') # posterior mean$') #: $|\bar{X} - E[X|y^*]\|_2$')
    plt.title('$N = '+str(N)+'$')
    if i==1:
        plt.legend(
            ncol            = 2,
            loc             = 'upper right',
            bbox_to_anchor  = (1.0, 1.01),
            frameon         = False,
            fancybox        = False, 
            shadow          = False,
            fontsize        = 8)

plt.savefig('smoothing_mean.pdf',dpi=600,bbox_inches='tight')
