# Load in a number of libraries we will use
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import copy
import pickle
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib

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
    
plt.close('all')

N           = 1000
T           = 1000
maxlag      = 100

Ns          = [50,100,175,250,375,500,750,1000]
repeats     = 100

colors      = [
    'xkcd:grass green',
    'xkcd:cerulean',
    'xkcd:orangish red',
    'xkcd:grass green',
    'xkcd:cerulean',
    'xkcd:orangish red']
markers     = ['o','x','v','o','x','v']

plt.figure(figsize=(12,12))
gs  = GridSpec(nrows = 3, ncols = 4, hspace = 0.4)

plt.subplot(gs[0,:])

plt.title(r'$\bf{A}$: Lorenz-63 linear smoothing results vs. ensemble size', loc='left', fontsize=titlesize)

labels      = [
    'backward smoother (single-pass)',
    'backward smoother (multi-pass)',
    'dense smoother']

linestyles  = [
    'solid',
    'solid',
    'solid',
    '--',
    '--',
    '--']


# dct_EnRTS = pickle.load(open("EnRTS_smoother_N=1000.p","rb"))
# dct_TM_BWS = pickle.load(open("TM_BW_smoother_N=1000.p","rb"))

# raise Exception

# dct_res     = {
#     'TM_BW'     : [],
#     'TM_BW_mp'  : [],
#     'TM_JA'     : [],
#     'EnRTS'     : [],
#     'EnRTS_mp'  : [],
#     'EnKS'      : []}

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

# """

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
        MCSE_list_collate += copy.deepcopy(RMSE_list)
        
        quantiles.append(
            np.nanmean(output_dictionary['X_a'],axis=0))
        
    MCSE_list_collate = np.std(RMSE_list)/np.sqrt(T*repeats)*1.96
        
    dct_res['EnTF'].append(np.mean(RMSE_list_collate))
    dct_res_MCSE['EnTF'].append(np.mean(MCSE_list_collate))
    
    quantiles    = np.mean(
        np.asarray(quantiles),
        axis = 0)
    
    dct_lag_quantiles['EnTF'].append(copy.copy(quantiles))

plt.plot(
    Ns,
    dct_res['EnTF'],
    marker      = "s",
    color       = "xkcd:grey",
    alpha       = 1,
    zorder      = 10,
    label       = "EnTF / EnKF")
plt.fill(
    list(Ns) + list(np.flip(Ns)),
    list(np.asarray(dct_res['EnTF']) + np.asarray(dct_res_MCSE['EnTF'])) + list(np.flip(np.asarray(dct_res['EnTF']) - np.asarray(dct_res_MCSE['EnTF']))),
    color       = "xkcd:grey",
    alpha       = 0.25,
    zorder      = 5,
    edgecolor   = "None")

# raise Exception

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
            MCSE_list_collate   += copy.deepcopy(RMSE_list)
            
            if idx > 0:
                quantiles.append(
                    np.nanmean(output_dictionary['X_s_q'],axis=0))
            
        dct_res[strng].append(np.mean(RMSE_list_collate))
        
        MCSE_list_collate = np.std(MCSE_list_collate)/np.sqrt(T*repeats)*1.96
        
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
        color       = colors[idx],
        linestyle   = linestyles[idx],
        marker      = markers[idx],
        zorder      = 10,
        alpha       = 1,
        label       = labels[idx])
    plt.fill(
        list(Ns) + list(np.flip(Ns)),
        list(np.asarray(dct_res[strng]) + np.asarray(dct_res_MCSE[strng])) + list(np.flip(np.asarray(dct_res[strng]) - np.asarray(dct_res_MCSE[strng]))),
        color       = colors[idx],
        alpha       = 0.25,
        edgecolor   = "None",
        zorder      = 5)
    
# """
    
 
plt.legend(loc='upper right',frameon=False, fontsize = labelsize)
plt.gca().set_xticks(Ns)
plt.gca().set_xticklabels(Ns)

plt.xticks(fontsize = labelsize)
plt.yticks(fontsize = labelsize)

plt.xlabel('ensemble size', fontsize = labelsize)
plt.ylabel('time-average RMSE', fontsize = labelsize)


plt.subplot(gs[1,:])

plt.title(r'$\bf{B}$: Equivalence of the linear single-pass BW-EnTS and EnRTS', loc='left', fontsize=titlesize)

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
    label       = "BW-EnTS ($5$\% - $95$\%)",
    edgecolor   = "None",
    zorder      = -5)

plt.fill(
    list(np.arange(450,550,1)) + list(np.flip(np.arange(450,550,1))),
    list(q25) + list(np.flip(q50)),
    color       = "xkcd:cerulean",
    label       = "BW-EnTS ($25$\% - $75$\%)",
    alpha       = 0.5,
    edgecolor   = "None",
    zorder      = -5)

plt.plot(
    np.arange(450,550,1),
    q50,
    color       = "xkcd:cerulean",
    label       = "BW-EnTS ($50$\%)",
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
    label       = "EnRTSS ($5$\% - $95$\%)",
    zorder      = -5)

plt.fill(
    list(np.arange(450,550,1)) + list(np.flip(np.arange(450,550,1))),
    list(q25) + list(np.flip(q50)),
    color       = "xkcd:grass green",
    label       = "EnRTSS ($25$\% - $75$\%)",
    alpha       = 0.5,
    edgecolor   = "None",
    zorder      = -5)

plt.plot(
    np.arange(450,550,1),
    q50,
    color       = "xkcd:grass green",
    label       = "EnRTSS ($50$\%)",
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

# plt.legend(frameon=False,ncol=2)

from matplotlib.patches import Patch
from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0], [0], color='xkcd:grass green', lw=1, label='EnRTSS ($50$\%)'),
    Patch(facecolor='xkcd:grass green', edgecolor='None', alpha = 0.5, label='EnRTSS ($25$\% - $75$\%)'),
    Patch(facecolor='xkcd:grass green', edgecolor='None', alpha = 0.2, label='EnRTSS ($5$\% - $95$\%)'),
    Line2D([0], [0], color='xkcd:cerulean', lw=1, label='BW-EnTS ($50$\%)'),
    Patch(facecolor='xkcd:cerulean', edgecolor='None', alpha = 0.5, label='BW-EnTS ($25$\% - $75$\%)'),
    Patch(facecolor='xkcd:cerulean', edgecolor='None', alpha = 0.2, label='BW-EnTS ($5$\% - $95$\%)')]


plt.legend(handles=legend_elements, loc='upper right',frameon=False,ncol = 2, fontsize = labelsize)

plt.xlabel("time steps", fontsize = labelsize)
plt.ylabel("ensemble error quantiles", fontsize = labelsize)

# Get the limits
xlims = plt.gca().get_xlim()
plt.gca().set_xlim(xlims)
ylims = plt.gca().get_ylim()
plt.gca().set_ylim(ylims)

snapshots = [460,490,506,536]


# Draw an annotation arrow
plt.gca().annotate('', 
    xy=((snapshots[0]-xlims[0])/(xlims[1]-xlims[0]), 0), 
    xycoords='axes fraction', 
    xytext=(0.125, -0.4), 
    arrowprops=dict(arrowstyle = '-',color='xkcd:dark grey'))
plt.gca().annotate('', 
    xy=((snapshots[0]-xlims[0])/(xlims[1]-xlims[0]), (q50[snapshots[0]-450]-ylims[0])/(ylims[1]-ylims[0])), 
    xycoords='axes fraction', 
    xytext=((snapshots[0]-xlims[0])/(xlims[1]-xlims[0]), 0), 
    arrowprops=dict(arrowstyle = '-',color='xkcd:dark grey'))

plt.gca().annotate('', 
    xy=((snapshots[1]-xlims[0])/(xlims[1]-xlims[0]), 0), 
    xycoords='axes fraction', 
    xytext=(0.375, -0.4), 
    arrowprops=dict(arrowstyle = '-',color='xkcd:dark grey'))
plt.gca().annotate('', 
    xy=((snapshots[1]-xlims[0])/(xlims[1]-xlims[0]), (q50[snapshots[1]-450]-ylims[0])/(ylims[1]-ylims[0])), 
    xycoords='axes fraction', 
    xytext=((snapshots[1]-xlims[0])/(xlims[1]-xlims[0]), 0), 
    arrowprops=dict(arrowstyle = '-',color='xkcd:dark grey'))

plt.gca().annotate('', 
    xy=((snapshots[2]-xlims[0])/(xlims[1]-xlims[0]), 0), 
    xycoords='axes fraction', 
    xytext=(0.625, -0.4), 
    arrowprops=dict(arrowstyle = '-',color='xkcd:dark grey'))
plt.gca().annotate('', 
    xy=((snapshots[2]-xlims[0])/(xlims[1]-xlims[0]), (q50[snapshots[2]-450]-ylims[0])/(ylims[1]-ylims[0])), 
    xycoords='axes fraction', 
    xytext=((snapshots[2]-xlims[0])/(xlims[1]-xlims[0]), 0), 
    arrowprops=dict(arrowstyle = '-',color='xkcd:dark grey'))

plt.gca().annotate('', 
    xy=((snapshots[3]-xlims[0])/(xlims[1]-xlims[0]), 0), 
    xycoords='axes fraction', 
    xytext=(0.875, -0.4), 
    arrowprops=dict(arrowstyle = '-',color='xkcd:dark grey'))
plt.gca().annotate('', 
    xy=((snapshots[3]-xlims[0])/(xlims[1]-xlims[0]), (q50[snapshots[3]-450]-ylims[0])/(ylims[1]-ylims[0])), 
    xycoords='axes fraction', 
    xytext=((snapshots[3]-xlims[0])/(xlims[1]-xlims[0]), 0), 
    arrowprops=dict(arrowstyle = '-',color='xkcd:dark grey'))




#%%

# Plot Lorenz-63 snapshots

def lorenz_dynamics(t, Z, beta=8/3, rho=28, sigma=10):
    
    if len(Z.shape) == 1: # Only one particle
    
        dZ1ds   = - sigma*Z[0] + sigma*Z[1]
        dZ2ds   = - Z[0]*Z[2] + rho*Z[0] - Z[1]
        dZ3ds   = Z[0]*Z[1] - beta*Z[2]
        
        dyn     = np.asarray([dZ1ds, dZ2ds, dZ3ds])
        
    else:
        
        dZ1ds   = - sigma*Z[...,0] + sigma*Z[...,1]
        dZ2ds   = - Z[...,0]*Z[...,2] + rho*Z[...,0] - Z[...,1]
        dZ3ds   = Z[...,0]*Z[...,1] - beta*Z[...,2]

        dyn     = np.column_stack((dZ1ds, dZ2ds, dZ3ds))

    return dyn

# Finds value of y for a given x using step size h
# and initial value y0 at x0.
def rk4(Z,fun,t=0,dt=1,nt=1):#(x0, y0, x, h):
    
    """
    Parameters
        t       : initial time
        Z       : initial states
        fun     : function to be integrated
        dt      : time step length
        nt      : number of time steps
    
    """
    
    # Prepare array for use
    if len(Z.shape) == 1: # We have only one particle, convert it to correct format
        Z       = Z[np.newaxis,:]
        
    # Go through all time steps
    for i in range(nt):
        
        # Calculate the RK4 values
        k1  = fun(t + i*dt,           Z);
        k2  = fun(t + i*dt + 0.5*dt,  Z + dt/2*k1);
        k3  = fun(t + i*dt + 0.5*dt,  Z + dt/2*k2);
        k4  = fun(t + i*dt + dt,      Z + dt*k3);
    
        # Update next value
        Z   += dt/6*(k1 + 2*k2 + 2*k3 + k4)
    
    return Z


import scipy.stats
D   = 3
T = 4000

# Create the array for the synthetic reference
attractor         = np.zeros((T,1,D))

# Initiate it with standard Gaussian samples
attractor[0,0,:]  = scipy.stats.norm.rvs(size=3)

# Simulate the spinup and simulation period
for t in np.arange(0,T-1,1):
     
    # Make a Lorenz forecast
    attractor[t+1,:,:] = rk4(
        Z           = copy.copy(attractor[t,:,:]),
        fun         = lorenz_dynamics,
        t           = 0,
        dt          = 0.01,
        nt          = 2)
    
# Remove the unnecessary particle index
attractor     = attractor[:,0,:]
    
# Discard the spinup
attractor     = attractor[1000:,:]



ax = plt.subplot(gs[2,0], projection='3d')

ax.plot3D(
    attractor[:,0], 
    attractor[:,1], 
    attractor[:,2], 
    'gray',
    alpha = 0.5)

ax.scatter3D(
    dct_TM_BWS['synthetic_truth'][snapshots[0],0],
    dct_TM_BWS['synthetic_truth'][snapshots[0],1],
    dct_TM_BWS['synthetic_truth'][snapshots[0],2],
    marker = "x",
    color = "red")




ax = plt.subplot(gs[2,1], projection='3d')

ax.plot3D(
    attractor[:,0], 
    attractor[:,1], 
    attractor[:,2], 
    'gray',
    alpha = 0.5)

ax.scatter3D(
    dct_TM_BWS['synthetic_truth'][snapshots[1],0],
    dct_TM_BWS['synthetic_truth'][snapshots[1],1],
    dct_TM_BWS['synthetic_truth'][snapshots[1],2],
    marker = "x",
    color = "red")



ax = plt.subplot(gs[2,2], projection='3d')

ax.plot3D(
    attractor[:,0], 
    attractor[:,1], 
    attractor[:,2], 
    'gray',
    alpha = 0.5)

ax.scatter3D(
    dct_TM_BWS['synthetic_truth'][snapshots[2],0],
    dct_TM_BWS['synthetic_truth'][snapshots[2],1],
    dct_TM_BWS['synthetic_truth'][snapshots[2],2],
    marker = "x",
    color = "red")



ax = plt.subplot(gs[2,3], projection='3d')

ax.plot3D(
    attractor[:,0], 
    attractor[:,1], 
    attractor[:,2], 
    'gray',
    alpha = 0.5)

ax.scatter3D(
    dct_TM_BWS['synthetic_truth'][snapshots[3],0],
    dct_TM_BWS['synthetic_truth'][snapshots[3],1],
    dct_TM_BWS['synthetic_truth'][snapshots[3],2],
    marker = "x",
    color = "red")













plt.savefig('tmp_results_L63_linear'+addendum+'.png',dpi=600,bbox_inches='tight')
plt.savefig('tmp_results_L63_linear'+addendum+'.pdf',dpi=600,bbox_inches='tight')
    