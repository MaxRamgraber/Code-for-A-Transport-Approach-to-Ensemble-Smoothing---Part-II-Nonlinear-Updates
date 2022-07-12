# Load in a number of libraries we will use
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import copy
import pickle
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
    
plt.close('all')

N           = 1000
T           = 1000
maxlag      = 100

Ns          = [50,100,175,250,375,500,750,1000]
repeats     = 100

colors      = ['xkcd:grass green','xkcd:cerulean','xkcd:orangish red','xkcd:grass green','xkcd:cerulean','xkcd:orangish red']

plt.figure(figsize=(12,8))
gs  = GridSpec(nrows = 2, ncols = 2, height_ratios=[1,1], hspace = 0.4)

plt.subplot(gs[0,:])

plt.title(r'$\bf{A}$: Lorenz-63 linear smoothing results vs. ensemble size', loc='left', fontsize=11)

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
    'EnTF'      : []}

dct_res_MCSE    = {
    'EnRTS'     : [],
    'EnRTS_mp'  : [],
    'EnKS'      : [],
    'EnTF'      : []}

dct_lag_quantiles   = {
    'EnRTS_mp'  : [],
    'EnKS'      : [],
    'EnTF'      : []}

for N in Ns:
    
    RMSE_list_collate   = []
    MCSE_list_collate   = []
    
    quantiles           = []
    
    for rep in range(repeats):

        # Load result dictionary
        output_dictionary   = pickle.load(
            open(
                'EnTF_N='+str(N).zfill(4)+'_rep='+str(rep).zfill(4)+'.p',
                'rb'))
    
        RMSE_list   = output_dictionary['RMSE_list']
        
        RMSE_list_collate.append(np.mean(RMSE_list))
        MCSE_list_collate.append(np.std(RMSE_list)/np.sqrt(T)*1.96)
        
        quantiles.append(
            np.nanmean(output_dictionary['X_a'],axis=0))
        
    dct_res['EnTF'].append(np.mean(RMSE_list_collate))
    dct_res_MCSE['EnTF'].append(np.mean(MCSE_list_collate))
    
    quantiles    = np.mean(
        np.asarray(quantiles),
        axis = 0)
    
    dct_lag_quantiles['EnTF'].append(copy.copy(quantiles))

plt.plot(
    Ns,
    dct_res['EnTF'],
    marker      = "x",
    color       = "xkcd:grey",
    alpha       = 1,
    label       = "EnTF / EnKF")
plt.fill(
    list(Ns) + list(np.flip(Ns)),
    list(np.asarray(dct_res['EnTF']) + np.asarray(dct_res_MCSE['EnTF'])) + list(np.flip(np.asarray(dct_res['EnTF']) - np.asarray(dct_res_MCSE['EnTF']))),
    color       = "xkcd:grey",
    alpha       = 0.5,
    edgecolor   = "None",
    zorder      = -2)

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

    plt.plot(
        Ns,
        dct_res[strng],
        marker      = "x",
        color       = colors[idx],
        linestyle   = linestyles[idx],
        alpha       = 1,
        label       = labels[idx])
    plt.fill(
        list(Ns) + list(np.flip(Ns)),
        list(np.asarray(dct_res[strng]) + np.asarray(dct_res_MCSE[strng])) + list(np.flip(np.asarray(dct_res[strng]) - np.asarray(dct_res_MCSE[strng]))),
        color       = colors[idx],
        alpha       = 0.5,
        edgecolor   = "None",
        zorder      = -2)
    
    
    
        
plt.legend(loc='upper right',frameon=False)
plt.gca().set_xticks(Ns)
plt.gca().set_xticklabels(Ns)

plt.xlabel('ensemble size')
plt.ylabel('average ensemble RMSE')


plt.subplot(gs[1,:])

plt.title(r'$\bf{B}$: Equivalence of the linear single-pass BW-EnTS and EnRTS', loc='left', fontsize=11)

dct_EnRTS = pickle.load(open("EnRTS_smoother_N=1000.p","rb"))
dct_TM_BWS = pickle.load(open("TM_BW_smoother_N=1000.p","rb"))


# Now plot the TM reference

RMSEs   = dct_TM_BWS['X_s'][450:550,:,:] - dct_TM_BWS['synthetic_truth'][450:550,:][:,np.newaxis,:]
RMSEs   = np.mean(RMSEs**2,axis = -1)
RMSEs   = np.sqrt(RMSEs)

q05     = np.quantile(RMSEs,    q = 0.05,     axis = -1)
q25     = np.quantile(RMSEs,    q = 0.25,     axis = -1)
q50     = np.quantile(RMSEs,    q = 0.50,     axis = -1)
q75     = np.quantile(RMSEs,    q = 0.75,     axis = -1)
q95     = np.quantile(RMSEs,    q = 0.95,     axis = -1)

plt.fill(
    list(np.arange(450,550,1)) + list(np.flip(np.arange(450,550,1))),
    list(q75) + list(np.flip(q95)),
    color       = "xkcd:cerulean",
    alpha       = 0.2,
    edgecolor   = "None",
    zorder      = -5)

plt.fill(
    list(np.arange(450,550,1)) + list(np.flip(np.arange(450,550,1))),
    list(q50) + list(np.flip(q75)),
    color       = "xkcd:cerulean",
    alpha       = 0.5,
    edgecolor   = "None",
    zorder      = -5)

plt.fill(
    list(np.arange(450,550,1)) + list(np.flip(np.arange(450,550,1))),
    list(q05) + list(np.flip(q25)),
    color       = "xkcd:cerulean",
    alpha       = 0.2,
    label       = "BW-EnTS (5% - 95%)",
    edgecolor   = "None",
    zorder      = -5)

plt.fill(
    list(np.arange(450,550,1)) + list(np.flip(np.arange(450,550,1))),
    list(q25) + list(np.flip(q50)),
    color       = "xkcd:cerulean",
    label       = "BW-EnTS (25% - 75%)",
    alpha       = 0.5,
    edgecolor   = "None",
    zorder      = -5)

plt.plot(
    np.arange(450,550,1),
    q50,
    color       = "xkcd:cerulean",
    label       = "BW-EnTS (50%)",
    alpha       = 1)

RMSEs   = dct_EnRTS['X_EnRTS'][450:550,:,:] - dct_EnRTS['synthetic_truth'][450:550,:][:,np.newaxis,:]
RMSEs   = np.mean(RMSEs**2,axis = -1)
RMSEs   = np.sqrt(RMSEs)


q05     = np.quantile(RMSEs,    q = 0.05,     axis = -1)
q25     = np.quantile(RMSEs,    q = 0.25,     axis = -1)
q50     = np.quantile(RMSEs,    q = 0.50,     axis = -1)
q75     = np.quantile(RMSEs,    q = 0.75,     axis = -1)
q95     = np.quantile(RMSEs,    q = 0.95,     axis = -1)


plt.fill(
    list(np.arange(450,550,1)) + list(np.flip(np.arange(450,550,1))),
    list(q05) + list(np.flip(q25)),
    color       = "xkcd:grass green",
    alpha       = 0.2,
    edgecolor   = "None",
    label       = "EnRTSS (5% - 95%)",
    zorder      = -5)

plt.fill(
    list(np.arange(450,550,1)) + list(np.flip(np.arange(450,550,1))),
    list(q25) + list(np.flip(q50)),
    color       = "xkcd:grass green",
    label       = "EnRTSS (25% - 75%)",
    alpha       = 0.5,
    edgecolor   = "None",
    zorder      = -5)

plt.plot(
    np.arange(450,550,1),
    q50,
    color       = "xkcd:grass green",
    label       = "EnRTSS (50%)",
    alpha       = 1,
    zorder      = -3)

plt.fill(
    list(np.arange(450,550,1)) + list(np.flip(np.arange(450,550,1))),
    list(q50) + list(np.flip(q75)),
    color       = "xkcd:grass green",
    alpha       = 0.5,
    edgecolor   = "None",
    zorder      = -5)

plt.fill(
    list(np.arange(450,550,1)) + list(np.flip(np.arange(450,550,1))),
    list(q75) + list(np.flip(q95)),
    color       = "xkcd:grass green",
    alpha       = 0.2,
    edgecolor   = "None",
    zorder      = -5)

from matplotlib.patches import Patch
from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0], [0], color='xkcd:grass green', lw=1, label='EnRTSS (50%)'),
    Patch(facecolor='xkcd:grass green', edgecolor='None', alpha = 0.5, label='EnRTSS (25% - 75%)'),
    Patch(facecolor='xkcd:grass green', edgecolor='None', alpha = 0.2, label='EnRTSS (5% - 95%)'),
    Line2D([0], [0], color='xkcd:cerulean', lw=1, label='BW-EnTS (50%)'),
    Patch(facecolor='xkcd:cerulean', edgecolor='None', alpha = 0.5, label='BW-EnTS (25% - 75%)'),
    Patch(facecolor='xkcd:cerulean', edgecolor='None', alpha = 0.2, label='BW-EnTS (5% - 95%)')]


plt.legend(handles=legend_elements, loc='upper right',frameon=False,ncol = 2)

plt.xlabel("time steps")
plt.ylabel("ensemble RMSE")

plt.savefig('linear_L63_results_line.png',dpi=600,bbox_inches='tight')
plt.savefig('linear_L63_results_line.pdf',dpi=600,bbox_inches='tight')
    