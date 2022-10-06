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

Ns = [100,1000]

# load results
dct  = pickle.load(open("autoregressive_model_results.p","rb"))

subdct  = {}
for N in Ns:
    
    subdct[N] = {}

    # add KS results
    subdct[N]['KS'] = {'cov_mean':[]}
    for seed in list(dct.keys()):
        subdct[N]['KS']['cov_mean'].append(dct[seed]['KS']['cov'])
    subdct[N]['KS']['mean_RMSE'] = np.zeros((T,))
    subdct[N]['KS']['cov_mean']  = np.mean(np.asarray(subdct[N]['KS']['cov_mean']), axis=0)
    subdct[N]['KS']['cov_RMSE']  = np.zeros((T,T))
    subdct[N]['KS']['cov_bias']  = np.zeros((T,T))

    # process results for each algorithm
    for alg in en_algs:

        # add elements to subdct for mean
        subdct[N][alg] = {'mean_mean': [], 'mean_RMSE': [], 'cov_mean': [], 'cov_RMSE': [], 'cov_bias': []}
        for seed in list(dct.keys()):
            subdct[N][alg]['mean_mean'].append(dct[seed][N][alg]['mean'])
            subdct[N][alg]['mean_RMSE'].append(dct[seed][N][alg]['mean'] - dct[seed]['KS']['mean'][:,0])
            subdct[N][alg]['cov_mean'].append(dct[seed][N][alg]['cov'])
            subdct[N][alg]['cov_RMSE'].append(dct[seed][N][alg]['cov'] - dct[seed]['KS']['cov'])

        # average elements
        subdct[N][alg]['mean_mean'] = np.mean(np.asarray(subdct[N][alg]['mean_mean']), axis=0)
        subdct[N][alg]['mean_RMSE'] = np.sqrt(np.mean(np.abs(np.asarray(subdct[N][alg]['mean_RMSE']))**2, axis=0))
        subdct[N][alg]['cov_mean']  = np.mean(np.asarray(subdct[N][alg]['cov_mean']), axis=0)
        subdct[N][alg]['cov_bias']  = subdct[N][alg]['cov_mean'] - subdct[N]['KS']['cov_mean']
        subdct[N][alg]['cov_RMSE']  = np.sqrt(np.mean(np.abs(np.asarray(subdct[N][alg]['cov_RMSE']))**2, axis=0))
    
# Store the results
pickle.dump(subdct,open('autoregressive_processed.p','wb'))
    
