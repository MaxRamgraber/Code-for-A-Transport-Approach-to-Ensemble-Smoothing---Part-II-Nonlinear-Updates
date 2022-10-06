import pickle
import matplotlib.pyplot as plt
import numpy as np
import copy
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

plt.close('all')

# load data
dct     = pickle.load(open("autoregressive_model_results.p","rb"))
subdct  = {}

algs = ['EnKS','EnRTSS (single-pass)','EnRTSS (multi-pass)','EnFIT (multi-pass)']
algs_keys = ['EnTS-JA','EnTS-BW-sp','EnTS-BW-mp','EnTS-FW-mp']

for N in [100,1000]:
    
    # create dictionary to save results for each algorithm
    subdct[N] = {}
    for (k,alg) in enumerate(algs):
        subdct[N][algs_keys[k]] =  {'signal' : [], 'gain' : []}
        
    # save results in dictionary
    for seed in list(dct.keys()):
        for (k,alg) in enumerate(algs):
            subdct[N][algs_keys[k]]['signal']  .append(np.abs(dct[seed][N][alg]['signal']))
            subdct[N][algs_keys[k]]['gain']    .append(np.abs(dct[seed][N][alg]['map']))
            
    # Average signals across random seeds
    for (k,alg) in enumerate(algs):
        subdct[N][algs_keys[k]]['signal']      = np.mean(np.asarray(subdct[N][algs_keys[k]]['signal']), axis = 0)
        subdct[N][algs_keys[k]]['gain']        = np.mean(np.asarray(subdct[N][algs_keys[k]]['gain']),   axis = 0)
        
    # average across samples
    for (k,alg) in enumerate(algs):
        subdct[N][algs_keys[k]]['signal']      = np.mean(np.asarray(subdct[N][algs_keys[k]]['signal']), axis = 1)
        subdct[N][algs_keys[k]]['gain']        = np.mean(np.asarray(subdct[N][algs_keys[k]]['gain']),   axis = 1)
        
    # squeeze any arrays
    for (k,alg) in enumerate(algs):
        subdct[N][algs_keys[k]]['signal']  = np.squeeze(subdct[N][algs_keys[k]]['signal'])
        subdct[N][algs_keys[k]]['gain']    = np.squeeze(subdct[N][algs_keys[k]]['gain'])
        
# Store the results
pickle.dump(subdct,open('autoregressive_signalgain.p','wb'))