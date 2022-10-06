import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import copy
from matplotlib.gridspec import GridSpec
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
algs = ['KS','EnKS','EnTS (joint-analysis)','EnRTSS (single-pass)','EnTS (backward, single-pass)',\
    'EnRTSS (multi-pass)','EnTS (backward, multi-pass)','EnFIT (multi-pass)','EnTS (forward, multi-pass)']
algs_titles = ['KS', 'EnKS', 'EnTS','EnRTSS','EnTS-BW','EnRTSS','EnTS-BW','EnFIT','EnTS-FW']

Ns = [100,1000]

# load results
case   = ''#'_lownoise'
subdct = pickle.load(open("autoregressive_processed"+case+".p","rb"))

# determine min and max values
attributes = ['cov_mean','cov_bias','cov_RMSE']
for attr in attributes:
    attr_min = []
    attr_max = []
    for N in Ns:
        for (k,alg) in enumerate(algs):
            attr_min.append(np.min(subdct[N][alg][attr]))
            attr_max.append(np.max(subdct[N][alg][attr]))
    subdct[attr+' min'] = np.min(attr_min)
    subdct[attr+' max'] = np.max(attr_max)
    
# Now, let us plot the results
plt.figure(figsize=(11,8))
gs  = GridSpec(nrows = 3*len(Ns), ncols=10, width_ratios = [1]*9+[0.1])#, wspace = 0.1, hspace = 0.1)

for (i,N) in enumerate(Ns):    
    plt.subplot(gs[3*i+0,0])
    plt.gca().annotate('', xy=(-0.32, -2.68-0.02), xycoords='axes fraction', xytext=(-0.32, 1.04+0.02), 
                arrowprops=dict(arrowstyle = '-',color='xkcd:dark grey'))
    plt.gca().annotate('', xy=(-0.1, 1.04), xycoords='axes fraction', xytext=(-0.36, 1.04), 
                arrowprops=dict(arrowstyle = '-',color='xkcd:dark grey'))
    plt.gca().annotate('', xy=(-0.1, -2.68), xycoords='axes fraction', xytext=(-0.36, -2.68), 
                arrowprops=dict(arrowstyle = '-',color='xkcd:dark grey'))
    plt.gca().text(-0.42, -0.8, '$N='+str(N)+'$', transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='center',horizontalalignment='center',color='xkcd:dark grey',rotation=90)
    
    # Plot the covariance for each algorithm
    for (k,alg) in enumerate(algs):
        plt.subplot(gs[3*i+0,k])
        if i==0: # only add title to top row
            plt.title(algs_titles[k], fontsize=9)
        plt.imshow(
            subdct[N][alg]['cov_mean'],
            cmap    = cmap,
            vmin    = subdct['cov_mean min'],
            vmax    = subdct['cov_mean max'])
        plt.xticks([])
        plt.yticks([])
        if k == 0:
            plt.ylabel('Avg. covariance', fontsize=8)
    
    plt.subplot(gs[3*i+0,len(algs)])
    plt.axis('off')
    ax = plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height*0.05, box.width, box.height*0.9])
    axColor = plt.axes([box.x0 - box.width*0.05, box.y0 + box.height*0.05, 0.01, box.height*0.9])
    cbar = plt.colorbar(cax = axColor, orientation="vertical")
    cbar.ax.set_yticklabels(["{:4.2f}".format(i) for i in cbar.get_ticks()])
    
    # Plot the bias in the covariance
    for (k,alg) in enumerate(algs):
        plt.subplot(gs[3*i+1,k])
        plt.imshow(
            subdct[N][alg]['cov_bias'], 
            cmap    = cmap+'_r',
            vmin    = subdct['cov_bias min'],
            vmax    = subdct['cov_bias max'])
        plt.xticks([])
        plt.yticks([])
        if k == 0:
            plt.ylabel('Bias', fontsize=8)
    
    plt.subplot(gs[3*i+1,len(algs)])
    plt.axis('off')
    ax = plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height*0.05, box.width, box.height*0.9])
    axColor = plt.axes([box.x0 - box.width*0.05, box.y0 + box.height*0.05, 0.01, box.height*0.9])
    cbar = plt.colorbar(cax = axColor, orientation="vertical")
    cbar.ax.set_yticklabels(["{:4.2f}".format(i) for i in cbar.get_ticks()])
    
    # Plot the RMSE in the covariance
    for (k,alg) in enumerate(algs):
        plt.subplot(gs[3*i+2,k])
        plt.imshow(
            subdct[N][alg]['cov_RMSE'],
            cmap    = cmap,
            vmin    = subdct['cov_RMSE min'],
            vmax    = subdct['cov_RMSE max'])
        plt.xticks([])
        plt.yticks([])
        if k==0:
            plt.ylabel('RMSE', fontsize=8)
    
    plt.subplot(gs[3*i+2,len(algs)])
    plt.axis('off')
    ax = plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height*0.05, box.width, box.height*0.9])
    axColor = plt.axes([box.x0 - box.width*0.05, box.y0 + box.height*0.05, 0.01, box.height*0.9])
    cbar = plt.colorbar(cax = axColor, orientation="vertical")
    cbar.ax.set_yticklabels(["{:4.2f}".format(i) for i in cbar.get_ticks()])

# add labels
plt.subplot(gs[0,0])
plt.gca().annotate('', xy=(-0.02, 1.3), xycoords='axes fraction', xytext=(1.02, 1.3),
            arrowprops=dict(arrowstyle = '-',color='xkcd:dark grey'))#'xkcd:orangish red'))
plt.gca().annotate('', xy=(.0, 1.34), xycoords='axes fraction', xytext=(0., 1.2),
            arrowprops=dict(arrowstyle = '-',color='xkcd:dark grey'))#'xkcd:orangish red'))
plt.gca().annotate('', xy=(1, 1.34), xycoords='axes fraction', xytext=(1., 1.2),
            arrowprops=dict(arrowstyle = '-',color='xkcd:dark grey'))#'xkcd:orangish red'))
plt.gca().text(0.5, 1.4, 'Reference', transform=plt.gca().transAxes, fontsize=9,
        verticalalignment='center',horizontalalignment='center',color='xkcd:dark grey')#'xkcd:orangish red')

# color_subtypes = ['xkcd:orangish red','xkcd:cerulean','xkcd:grass green','xkcd:yellow orange']
# title_subtypes = ['Dense','Backward single-pass','Backward multi-pass','Forward multi-pass']
# index_subtypes = [1,3,5,7]
color_subtypes = ['xkcd:cerulean','xkcd:grass green','xkcd:yellow orange']
title_subtypes = ['Backward single-pass','Backward multi-pass','Forward multi-pass']
index_subtypes = [3,5,7]
n_subtypes = len(color_subtypes)

for i in range(n_subtypes):
    plt.subplot(gs[0,index_subtypes[i]])
    rightx  = 2.2
    plt.gca().annotate('', xy=(-0.02, 1.3), xycoords='axes fraction', xytext=(rightx+0.02, 1.3), 
                arrowprops=dict(arrowstyle = '-',color=color_subtypes[i]))
    plt.gca().annotate('', xy=(.0, 1.34), xycoords='axes fraction', xytext=(0., 1.2), 
                arrowprops=dict(arrowstyle = '-',color=color_subtypes[i]))
    plt.gca().annotate('', xy=(rightx, 1.34), xycoords='axes fraction', xytext=(rightx, 1.2), 
                arrowprops=dict(arrowstyle = '-',color=color_subtypes[i]))
    plt.gca().text((rightx+0.02)/2, 1.4, title_subtypes[i], transform=plt.gca().transAxes, fontsize=9,
            verticalalignment='center',horizontalalignment='center',color=color_subtypes[i])

# plt.subplot(gs[0,3])
# rightx  = 2.2
# plt.gca().annotate('', xy=(-0.02, 1.43), xycoords='axes fraction', xytext=(rightx+0.02, 1.43), 
#             arrowprops=dict(arrowstyle = '-',color='))
# plt.gca().annotate('', xy=(.0, 1.46), xycoords='axes fraction', xytext=(0., 1.2), 
#             arrowprops=dict(arrowstyle = '-',color='xkcd:cerulean'))
# plt.gca().annotate('', xy=(rightx, 1.46), xycoords='axes fraction', xytext=(rightx, 1.2), 
#             arrowprops=dict(arrowstyle = '-',color='xkcd:cerulean'))
# plt.gca().text((rightx+0.02)/2, 1.50, '', transform=plt.gca().transAxes, fontsize=9,
#         verticalalignment='center',horizontalalignment='center',color='xkcd:cerulean')

# plt.subplot(gs[0,5])
# rightx  = 2.2
# plt.gca().annotate('', xy=(-0.02, 1.43), xycoords='axes fraction', xytext=(rightx+0.02, 1.43), 
#             arrowprops=dict(arrowstyle = '-',color=''))
# plt.gca().annotate('', xy=(.0, 1.46), xycoords='axes fraction', xytext=(0., 1.2), 
#             arrowprops=dict(arrowstyle = '-',color='xkcd:grass green'))
# plt.gca().annotate('', xy=(rightx, 1.46), xycoords='axes fraction', xytext=(rightx, 1.2), 
#             arrowprops=dict(arrowstyle = '-',color='xkcd:grass green'))
# plt.gca().text((rightx+0.02)/2, 1.50, 'backward multi-pass smoothers', transform=plt.gca().transAxes, fontsize=9,
#         verticalalignment='center',horizontalalignment='center',color='xkcd:grass green')

# plt.subplot(gs[0,7])
# rightx  = 2.2
# plt.gca().annotate('', xy=(-0.02, 1.43), xycoords='axes fraction', xytext=(rightx+0.02, 1.43), 
#             arrowprops=dict(arrowstyle = '-',color=''))
# plt.gca().annotate('', xy=(.0, 1.46), xycoords='axes fraction', xytext=(0., 1.2), 
#             arrowprops=dict(arrowstyle = '-',color='xkcd:yellow orange'))
# plt.gca().annotate('', xy=(rightx, 1.46), xycoords='axes fraction', xytext=(rightx, 1.2), 
#             arrowprops=dict(arrowstyle = '-',color='xkcd:yellow orange'))
# plt.gca().text((rightx+0.02)/2, 1.50, 'forward multi-pass smoothers', transform=plt.gca().transAxes, fontsize=9,
#         verticalalignment='center',horizontalalignment='center',color='xkcd:yellow orange')

plt.savefig('smoothing_covariances'+case+'.pdf',dpi=600,bbox_inches='tight')

#%%

