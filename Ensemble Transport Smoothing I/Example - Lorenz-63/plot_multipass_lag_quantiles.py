# Load in a number of libraries we will use
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import copy
import pickle
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

plt.rc('font', family='serif') # sans-serif
plt.rc('text', usetex=True)

plt.rcParams['text.latex.preamble'] = [
       r'\usepackage{siunitx}',   # i need upright \micro symbols, but you need...
       r'\sisetup{detect-all}',   # ...this to force siunitx to actually use your fonts
       r'\usepackage{helvet}',    # set the normal font here
       r'\usepackage{sansmath}',  # load up the sansmath so that math -> helvet
       r'\sansmath'               # <- tricky! -- gotta actually tell tex to use!
]  
    
plt.close('all')

N           = 1000
T           = 1000
maxlag      = 100

Ns          = [50,100,175,250,375,500,750,1000]
repeats     = 100

colors      = ['xkcd:grass green','xkcd:cerulean','xkcd:orangish red','xkcd:grass green','xkcd:cerulean','xkcd:orangish red']

plt.figure(figsize=(12,6))
gs  = GridSpec(nrows = 2, ncols = 2, height_ratios=[1,1], hspace = 0.1, wspace = 0.3)


plt.title(r'$\bf{A}$: Lorenz-63 smoothing results', loc='left', fontsize=11)

labels      = [
    'backward smoother (single-pass)',
    'backward smoother (multi-pass)',
    'joint-analysis smoother']

linestyles  = [
    'solid',
    'solid',
    'solid',
    '--',
    '--',
    '--']

dct_res     = {
    'EnRTS'     : [],
    'EnRTS_mp'  : [],
    'EnKS'      : [],
    'EnKF'      : []}

dct_res_MCSE    = {
    'EnRTS'     : [],
    'EnRTS_mp'  : [],
    'EnKS'      : [],
    'EnKF'      : []}

dct_lag_quantiles   = {
    'EnRTS_mp'  : [],
    'EnKS'      : [],
    'EnKF'      : []}

    
for idx,strng in enumerate(['EnRTS','EnRTS_mp','EnKS']): #enumerate(['TM_BW','TM_BW_mp','TM_JA','EnRTS','EnRTS_mp','EnKS']):
    
    for N in Ns:
        
        RMSE_list_collate   = []
        MCSE_list_collate   = []
        
        quantiles           = []
        
        for rep in range(repeats):
    
            # Load result dictionary
            output_dictionary   = pickle.load(
                open(
                    strng + '_smoother_N='+str(N).zfill(4)+'_rep='+str(rep).zfill(4)+'.p',
                    'rb'))
        
            RMSE_list   = output_dictionary['RMSE_list']
            
            RMSE_list_collate.append(np.mean(RMSE_list))
            MCSE_list_collate.append(np.std(RMSE_list)/np.sqrt(T)*1.96)
            
            if idx > 0:
                quantiles.append(
                    np.nanmean(output_dictionary['X_s_q'],axis=0))
            
        dct_res[strng].append(np.mean(RMSE_list_collate))
        dct_res_MCSE[strng].append(np.mean(MCSE_list_collate))
        
        if idx > 0:
            quantiles    = np.mean(
                np.asarray(quantiles),
                axis = 0)
    
        if idx > 0:
            dct_lag_quantiles[strng].append(copy.copy(quantiles))
    
# =============================================================================
# Plot lagged quantiles
# =============================================================================

plt.subplot(gs[0,0])

xpos    = [-0.175,1.05]
ypos    = [-1.35,1.06]
xdif    = np.abs(np.diff(xpos))
ydif    = np.abs(np.diff(ypos))

plt.text(xpos[0],ypos[1]+0.1,r'$\bf{A}$: fixed-lag smoothing (multi-pass backwards smoother)', 
    transform=plt.gca().transAxes, fontsize=10,color='xkcd:grey',
    verticalalignment='top',horizontalalignment='left')

plt.gca().annotate('', xy=(xpos[0], ypos[0]), xycoords='axes fraction', xytext=(xpos[0], ypos[1]), 
                    arrowprops=dict(color='xkcd:silver',headlength=1,headwidth=0,width=1))

plt.gca().annotate('', xy=(xpos[0], ypos[1]), xycoords='axes fraction', xytext=(xpos[1], ypos[1]), 
                    arrowprops=dict(color='xkcd:silver',headlength=1,headwidth=0,width=1))

plt.gca().annotate('', xy=(xpos[1], ypos[1]), xycoords='axes fraction', xytext=(xpos[1], ypos[0]), 
                    arrowprops=dict(color='xkcd:silver',headlength=1,headwidth=0,width=1))

plt.gca().annotate('', xy=(xpos[1], ypos[0]), xycoords='axes fraction', xytext=(xpos[0], ypos[0]), 
                    arrowprops=dict(color='xkcd:silver',headlength=1,headwidth=0,width=1))


plt.ylabel("time-averaged error quantiles")



plt.fill(
    list(np.arange(101)) + list(np.flip(np.arange(101))),
    list(dct_lag_quantiles['EnRTS_mp'][0][:,1]) + list(np.flip(dct_lag_quantiles['EnRTS_mp'][0][:,2])),
    color       = "xkcd:cerulean",
    label       = "$25$\% - $75$\% quantile ($N = 50$)",
    alpha       = 0.5,
    edgecolor   = "None",
    zorder      = -1)

plt.fill(
    list(np.arange(101)) + list(np.flip(np.arange(101))),
    list(dct_lag_quantiles['EnRTS_mp'][0][:,0]) + list(np.flip(dct_lag_quantiles['EnRTS_mp'][0][:,1])),
    color       = "xkcd:cerulean",
    label       = "$5$\% - $95$\% quantile ($N = 50$)",
    alpha       = 0.2,
    edgecolor   = "None",
    zorder      = -2)

plt.plot(
    np.arange(101),
    dct_lag_quantiles['EnRTS_mp'][0][:,2],
    color       = "xkcd:cerulean",
    label       = "$50$\% quantile ($N = 50$)",
    alpha       = 1,
    zorder      = 0)

plt.fill(
    list(np.arange(101)) + list(np.flip(np.arange(101))),
    list(dct_lag_quantiles['EnRTS_mp'][0][:,2]) + list(np.flip(dct_lag_quantiles['EnRTS_mp'][0][:,3])),
    color       = "xkcd:cerulean",
    alpha       = 0.5,
    edgecolor   = "None",
    zorder      = -1)

plt.fill(
    list(np.arange(101)) + list(np.flip(np.arange(101))),
    list(dct_lag_quantiles['EnRTS_mp'][0][:,3]) + list(np.flip(dct_lag_quantiles['EnRTS_mp'][0][:,4])),
    color       = "xkcd:cerulean",
    alpha       = 0.2,
    edgecolor   = "None",
    zorder      = -2)

plt.legend(frameon=False)

plt.gca().set_xticklabels([])



plt.subplot(gs[1,0])

plt.ylabel("time-averaged error quantiles")
plt.xlabel("smoothing lag")


plt.fill(
    list(np.arange(101)) + list(np.flip(np.arange(101))),
    list(dct_lag_quantiles['EnRTS_mp'][-1][:,1]) + list(np.flip(dct_lag_quantiles['EnRTS_mp'][-1][:,2])),
    color       = "xkcd:cerulean",
    label       = "$25$\% - $75$\% quantile ($N = 1000$)",
    alpha       = 0.5,
    edgecolor   = "None",
    zorder      = -4)

plt.fill(
    list(np.arange(101)) + list(np.flip(np.arange(101))),
    list(dct_lag_quantiles['EnRTS_mp'][-1][:,0]) + list(np.flip(dct_lag_quantiles['EnRTS_mp'][-1][:,1])),
    color       = "xkcd:cerulean",
    label       = "$5$\% - $95$\% quantile ($N = 1000$)",
    alpha       = 0.2,
    edgecolor   = "None",
    zorder      = -5)

plt.plot(
    np.arange(101),
    dct_lag_quantiles['EnRTS_mp'][-1][:,2],
    color       = "xkcd:cerulean",
    label       = "$50$\% quantile ($N = 1000$)",
    alpha       = 1,
    zorder      = -3)

plt.fill(
    list(np.arange(101)) + list(np.flip(np.arange(101))),
    list(dct_lag_quantiles['EnRTS_mp'][-1][:,2]) + list(np.flip(dct_lag_quantiles['EnRTS_mp'][-1][:,3])),
    color       = "xkcd:cerulean",
    alpha       = 0.5,
    edgecolor   = "None",
    zorder      = -4)

plt.fill(
    list(np.arange(101)) + list(np.flip(np.arange(101))),
    list(dct_lag_quantiles['EnRTS_mp'][-1][:,3]) + list(np.flip(dct_lag_quantiles['EnRTS_mp'][-1][:,4])),
    color       = "xkcd:cerulean",
    alpha       = 0.2,
    edgecolor   = "None",
    zorder      = -5)

plt.legend(frameon=False)



plt.subplot(gs[0,1])

# plt.title(r'$\bf{C}$: fixed-lag smoothing (joint-analysis smoother)', loc='left', fontsize=11)

# xpos    = [-0.175,1.05]
# ypos    = [-1.4,1.075]
# xdif    = np.abs(np.diff(xpos))
# ydif    = np.abs(np.diff(ypos))

plt.text(xpos[0],ypos[1]+0.1,r'$\bf{B}$: fixed-lag smoothing (joint-analysis smoother)', 
    transform=plt.gca().transAxes, fontsize=10,color='xkcd:grey',
    verticalalignment='top',horizontalalignment='left')

plt.gca().annotate('', xy=(xpos[0], ypos[0]), xycoords='axes fraction', xytext=(xpos[0], ypos[1]), 
                    arrowprops=dict(color='xkcd:silver',headlength=1,headwidth=0,width=1))

plt.gca().annotate('', xy=(xpos[0], ypos[1]), xycoords='axes fraction', xytext=(xpos[1], ypos[1]), 
                    arrowprops=dict(color='xkcd:silver',headlength=1,headwidth=0,width=1))

plt.gca().annotate('', xy=(xpos[1], ypos[1]), xycoords='axes fraction', xytext=(xpos[1], ypos[0]), 
                    arrowprops=dict(color='xkcd:silver',headlength=1,headwidth=0,width=1))

plt.gca().annotate('', xy=(xpos[1], ypos[0]), xycoords='axes fraction', xytext=(xpos[0], ypos[0]), 
                    arrowprops=dict(color='xkcd:silver',headlength=1,headwidth=0,width=1))

plt.ylabel("time-averaged error quantiles")
# plt.xlabel("smoothing lag")


plt.fill(
    list(np.arange(101)) + list(np.flip(np.arange(101))),
    list(dct_lag_quantiles['EnKS'][0][:,1]) + list(np.flip(dct_lag_quantiles['EnKS'][0][:,2])),
    color       = "xkcd:orangish red",
    label       = "$25$\% - $75$\% quantile ($N = 50$)",
    alpha       = 0.5,
    edgecolor   = "None",
    zorder      = -1)

plt.fill(
    list(np.arange(101)) + list(np.flip(np.arange(101))),
    list(dct_lag_quantiles['EnKS'][0][:,0]) + list(np.flip(dct_lag_quantiles['EnKS'][0][:,1])),
    color       = "xkcd:orangish red",
    label       = "$5$\% - $95$\% quantile ($N = 50$)",
    alpha       = 0.2,
    edgecolor   = "None",
    zorder      = -2)

plt.plot(
    np.arange(101),
    dct_lag_quantiles['EnKS'][0][:,2],
    color       = "xkcd:orangish red",
    label       = "$50$\% quantile ($N = 50$)",
    alpha       = 1,
    zorder      = 0)

plt.fill(
    list(np.arange(101)) + list(np.flip(np.arange(101))),
    list(dct_lag_quantiles['EnKS'][0][:,2]) + list(np.flip(dct_lag_quantiles['EnKS'][0][:,3])),
    color       = "xkcd:orangish red",
    alpha       = 0.5,
    edgecolor   = "None",
    zorder      = -1)

plt.fill(
    list(np.arange(101)) + list(np.flip(np.arange(101))),
    list(dct_lag_quantiles['EnKS'][0][:,3]) + list(np.flip(dct_lag_quantiles['EnKS'][0][:,4])),
    color       = "xkcd:orangish red",
    alpha       = 0.2,
    edgecolor   = "None",
    zorder      = -2)

plt.legend(frameon=False)

plt.gca().set_xticklabels([])



plt.subplot(gs[1,1])


plt.fill(
    list(np.arange(101)) + list(np.flip(np.arange(101))),
    list(dct_lag_quantiles['EnKS'][-1][:,1]) + list(np.flip(dct_lag_quantiles['EnKS'][-1][:,2])),
    color       = "xkcd:orangish red",
    label       = "$25$\% - $75$\% quantile ($N = 1000$)",
    alpha       = 0.5,
    edgecolor   = "None",
    zorder      = -4)

plt.fill(
    list(np.arange(101)) + list(np.flip(np.arange(101))),
    list(dct_lag_quantiles['EnKS'][-1][:,0]) + list(np.flip(dct_lag_quantiles['EnKS'][-1][:,1])),
    color       = "xkcd:orangish red",
    label       = "$5$\% - $95$\% quantile ($N = 1000$)",
    alpha       = 0.2,
    edgecolor   = "None",
    zorder      = -5)

plt.plot(
    np.arange(101),
    dct_lag_quantiles['EnKS'][-1][:,2],
    color       = "xkcd:orangish red",
    label       = "$50$\% quantile ($N = 1000$)",
    alpha       = 1,
    zorder      = -3)

plt.fill(
    list(np.arange(101)) + list(np.flip(np.arange(101))),
    list(dct_lag_quantiles['EnKS'][-1][:,2]) + list(np.flip(dct_lag_quantiles['EnKS'][-1][:,3])),
    color       = "xkcd:orangish red",
    alpha       = 0.5,
    edgecolor   = "None",
    zorder      = -4)

plt.fill(
    list(np.arange(101)) + list(np.flip(np.arange(101))),
    list(dct_lag_quantiles['EnKS'][-1][:,3]) + list(np.flip(dct_lag_quantiles['EnKS'][-1][:,4])),
    color       = "xkcd:orangish red",
    alpha       = 0.2,
    edgecolor   = "None",
    zorder      = -5)

plt.legend(frameon=False)

plt.ylabel("time-averaged error quantiles")
plt.xlabel("smoothing lag")

plt.savefig('multipass_lag_quantiles.png',dpi=600,bbox_inches='tight')
plt.savefig('multipass_lag_quantiles.pdf',dpi=600,bbox_inches='tight')
