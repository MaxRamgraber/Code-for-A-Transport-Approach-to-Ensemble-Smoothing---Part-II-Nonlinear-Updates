import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
import copy
from matplotlib.gridspec import GridSpec
import scipy.linalg
import turbo_colormap

plt.close('all')

# Let's seek an analytical solution
D       = 1     # Number of state dimensions
O       = D     # Number of observation dimensions

# Then we prepare a new colormap, mainly for cosmetic purposes.
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", 
    [   "xkcd:dark grey",
        "xkcd:silver",
        "xkcd:cerulean",
        "xkcd:grass green",
        "xkcd:goldenrod",
        "xkcd:orangish red"])

cmap   = "turbo"



order   = 3

# Forecast operator
A       = np.identity(D)*0.9
   
# Forecast error covariance matrix
Q       = np.identity(D)

# Get the prior mean and cov
mut     = np.ones((D,1))*10
covt    = np.identity(D)*4

mu_pri  = np.ones((D,1))*10
cov_pri = np.identity(D)*4

# Get the observation operator and likelihood
H       = np.zeros((O,D))
np.fill_diagonal(H,1)

# Get the observation error
covstd  = np.ones((D,D))*2
R       = np.zeros((O,O))
np.fill_diagonal(R,covstd**2)

# Get the number of total time steps
T       = 30

# Pre-allocate a complete multivariate normal
muT     = np.zeros((int(T*D),1))
covT    = np.zeros((int(T*D),int(T*D)))

# Fill in the prior
muT[:D,0]   = copy.copy(mut)
covT[:D,:D] = copy.copy(covt)

# Get the true dynamics
x_true          = np.zeros((int(T*D),1))
x_true[:D,0]    = 10
for t in np.arange(1,T,1):
    x_true[int(t*D):int((t+1)*D),0] = np.dot(
        A,
        x_true[int((t-1)*D):int((t)*D),:]) 

# Create fake observations
x_obs       = copy.copy(x_true) 
x_obs[:,0]  += np.asarray([covstd[0,0] if i%2==0 else -covstd[0,0] for i in range(T)])


def fillwrap(x,y=None):
    
    if y is None:
        y = x

    return np.asarray(list(x)+list(np.flip(y)))



xfill   = np.asarray(list(np.arange(T))+list(np.flip(np.arange(T))))



dct     = pickle.load(open("autoregressive_model_results_cubic.p","rb"))


subdct  = {}

for N in [100,1000]:
    
    subdct[N]   = {
        'KS'            : [],
        'EnTS-JA'       : [],
        'EnTS-BW-sp'    : [],
        'EnTS-BW-mp'    : [],
        'div KS'        : [],
        'div EnTS-JA'   : [],
        'div EnTS-BW-sp': [],
        'div EnTS-BW-mp': []}

    KS      = []
    
    for seed in list(dct.keys()):
        
        subdct[N]['KS']             .append(dct[seed]['KS']['cov'])
        subdct[N]['EnTS-JA']        .append(dct[seed][N]['EnTS (joint-analysis)']['cov'])
        subdct[N]['EnTS-BW-sp']     .append(dct[seed][N]['EnTS (backward, single-pass)']['cov'])
        subdct[N]['EnTS-BW-mp']     .append(dct[seed][N]['EnTS (backward, multi-pass)']['cov'])
        
        subdct[N]['div KS']         .append(dct[seed]['KS']['cov']                              - dct[seed]['KS']['cov'])
        subdct[N]['div EnTS-JA']    .append(dct[seed][N]['EnTS (joint-analysis)']['cov']        - dct[seed]['KS']['cov'])
        subdct[N]['div EnTS-BW-sp'] .append(dct[seed][N]['EnTS (backward, single-pass)']['cov'] - dct[seed]['KS']['cov'])
        subdct[N]['div EnTS-BW-mp'] .append(dct[seed][N]['EnTS (backward, multi-pass)']['cov']  - dct[seed]['KS']['cov'])
        
    
        
    # Average
    subdct[N]['KS']             = np.mean(np.asarray(subdct[N]['KS']),          axis = 0)
    subdct[N]['EnTS-JA']        = np.mean(np.asarray(subdct[N]['EnTS-JA']),     axis = 0)
    subdct[N]['EnTS-BW-sp']     = np.mean(np.asarray(subdct[N]['EnTS-BW-sp']),  axis = 0)
    subdct[N]['EnTS-BW-mp']     = np.mean(np.asarray(subdct[N]['EnTS-BW-mp']),  axis = 0)
    
    subdct[N]['div KS']         = np.mean(np.abs(np.asarray(subdct[N]['div KS'])),          axis = 0)
    subdct[N]['div EnTS-JA']    = np.mean(np.abs(np.asarray(subdct[N]['div EnTS-JA'])),     axis = 0)
    subdct[N]['div EnTS-BW-sp'] = np.mean(np.abs(np.asarray(subdct[N]['div EnTS-BW-sp'])),  axis = 0)
    subdct[N]['div EnTS-BW-mp'] = np.mean(np.abs(np.asarray(subdct[N]['div EnTS-BW-mp'])),  axis = 0)
    
    subdct[N]['cov min']        = np.min([
        np.min(subdct[N]['KS']), 
       np.min(subdct[N]['EnTS-JA']), 
        np.min(subdct[N]['EnTS-BW-sp']),
        np.min(subdct[N]['EnTS-BW-mp'])])
    
    subdct[N]['cov max']        = np.max([
        np.max(subdct[N]['KS']), 
        np.max(subdct[N]['EnTS-JA']), 
        np.max(subdct[N]['EnTS-BW-sp']),
        np.max(subdct[N]['EnTS-BW-mp'])])
    
    subdct[N]['div min']        = np.min([
        np.min(subdct[N]['div KS']), 
        np.min(subdct[N]['div EnTS-JA']), 
        np.min(subdct[N]['div EnTS-BW-sp']),
        np.min(subdct[N]['div EnTS-BW-mp'])])
    
    subdct[N]['div max']        = np.max([
        np.max(subdct[N]['div KS']), 
        np.max(subdct[N]['div EnTS-JA']), 
        np.max(subdct[N]['div EnTS-BW-sp']),
        np.max(subdct[N]['div EnTS-BW-mp'])])
    
    # raise Exception


#%%


plt.figure(figsize=(9,8))

gs  = GridSpec(nrows = 4, ncols = 5, width_ratios = [1]*4+[0.1])#, wspace = 0.1, hspace = 0.1)

# Plot the Kalman Smoother reference
plt.subplot(gs[0,0])
plt.title('KS \n (analytical)', fontsize=12)
plt.imshow(
    subdct[100]['KS'],
    cmap    = cmap,
    vmin    = subdct[100]['cov min'],
    vmax    = subdct[100]['cov max'])
plt.xticks([])
plt.yticks([])
plt.ylabel('covariance', fontsize=12)

plt.gca().annotate('', xy=(-0.02, 1.38), xycoords='axes fraction', xytext=(1.02, 1.38), 
            arrowprops=dict(arrowstyle = '-',color='xkcd:orangish red'))
plt.gca().annotate('', xy=(.0, 1.4), xycoords='axes fraction', xytext=(0., 1.2), 
            arrowprops=dict(arrowstyle = '-',color='xkcd:orangish red'))
plt.gca().annotate('', xy=(1, 1.4), xycoords='axes fraction', xytext=(1., 1.2), 
            arrowprops=dict(arrowstyle = '-',color='xkcd:orangish red'))
plt.gca().text(0.5, 1.45, 'reference', transform=plt.gca().transAxes, fontsize=12,
        verticalalignment='center',horizontalalignment='center',color='xkcd:orangish red')


plt.gca().annotate('', xy=(-0.28, -1.25-0.02), xycoords='axes fraction', xytext=(-0.28, 1.+0.02), 
            arrowprops=dict(arrowstyle = '-',color='xkcd:dark grey'))
plt.gca().annotate('', xy=(-0.1, 1.), xycoords='axes fraction', xytext=(-0.3, 1.), 
            arrowprops=dict(arrowstyle = '-',color='xkcd:dark grey'))
plt.gca().annotate('', xy=(-0.1, -1.25), xycoords='axes fraction', xytext=(-0.3, -1.25), 
            arrowprops=dict(arrowstyle = '-',color='xkcd:dark grey'))
plt.gca().text(-0.35, -0.1, '$N=100$', transform=plt.gca().transAxes, fontsize=12,
        verticalalignment='center',horizontalalignment='center',color='xkcd:dark grey',rotation=90)

# Plot the TM Kalman Smoother
plt.subplot(gs[0,1])
plt.title('EnTS-JA', fontsize=12)
plt.imshow(
    subdct[100]['EnTS-JA'],
    cmap    = cmap,
    vmin    = subdct[100]['cov min'],
    vmax    = subdct[100]['cov max'])
plt.xticks([])
plt.yticks([])

rightx  = 1.

plt.gca().annotate('', xy=(-0.02, 1.38), xycoords='axes fraction', xytext=(rightx+0.02, 1.38), 
            arrowprops=dict(arrowstyle = '-',color='xkcd:cerulean'))
plt.gca().annotate('', xy=(.0, 1.4), xycoords='axes fraction', xytext=(0., 1.2), 
            arrowprops=dict(arrowstyle = '-',color='xkcd:cerulean'))
plt.gca().annotate('', xy=(rightx, 1.4), xycoords='axes fraction', xytext=(rightx, 1.2), 
            arrowprops=dict(arrowstyle = '-',color='xkcd:cerulean'))
plt.gca().text((rightx+0.02)/2, 1.375, 'joint-analysis \n smoothers', transform=plt.gca().transAxes, fontsize=12,
        verticalalignment='center',horizontalalignment='center',color='xkcd:cerulean')




# Plot the TM RTS (multi-pass)
plt.subplot(gs[0,2])
plt.title('EnTS-BW\n (single-pass)', fontsize=12)
plt.imshow(
    subdct[100]['EnTS-BW-sp'],
    cmap    = cmap,
    vmin    = subdct[100]['cov min'],
    vmax    = subdct[100]['cov max'])
plt.xticks([])
plt.yticks([])

rightx  = 2.25
plt.gca().annotate('', xy=(-0.02, 1.38), xycoords='axes fraction', xytext=(rightx+0.02, 1.38), 
            arrowprops=dict(arrowstyle = '-',color='xkcd:grass green'))
plt.gca().annotate('', xy=(.0, 1.4), xycoords='axes fraction', xytext=(0., 1.2), 
            arrowprops=dict(arrowstyle = '-',color='xkcd:grass green'))
plt.gca().annotate('', xy=(rightx, 1.4), xycoords='axes fraction', xytext=(rightx, 1.2), 
            arrowprops=dict(arrowstyle = '-',color='xkcd:grass green'))
plt.gca().text((rightx+0.02)/2, 1.45, 'backward smoothers', transform=plt.gca().transAxes, fontsize=12,
        verticalalignment='center',horizontalalignment='center',color='xkcd:grass green')

# Plot the TM RTS (multi-pass)
plt.subplot(gs[0,3])
plt.title('EnTS-BW\n (multi-pass)', fontsize=12)
plt.imshow(
    subdct[100]['EnTS-BW-mp'],
    cmap    = cmap,
    vmin    = subdct[100]['cov min'],
    vmax    = subdct[100]['cov max'])
plt.xticks([])
plt.yticks([])

plt.subplot(gs[0,4])
plt.axis('off')
ax = plt.gca()
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height*0.05, box.width, box.height*0.9])
axColor = plt.axes([box.x0 - box.width*1.05, box.y0 + box.height*0.05, 0.01, box.height*0.9])
cbar = plt.colorbar(cax = axColor, orientation="vertical",label="variance")
cbar.set_label(label="variance",size=12)
cbar.ax.set_yticklabels(["{:4.2f}".format(i) for i in cbar.get_ticks()])

#%%

# Plot the Kalman Smoother reference
plt.subplot(gs[1,0])
plt.imshow(
    subdct[100]['div KS'],
    cmap    = cmap,
    vmin    = subdct[100]['div min'],
    vmax    = subdct[100]['div max'])
plt.xticks([])
plt.yticks([])
plt.ylabel('mismatch', fontsize=12)

# Plot the TM Kalman Smoother
plt.subplot(gs[1,1])
plt.imshow(
    subdct[100]['div EnTS-JA'],
    cmap    = cmap,
    vmin    = subdct[100]['div min'],
    vmax    = subdct[100]['div max'])
plt.xticks([])
plt.yticks([])


# Plot the TM RTS (single-pass)
plt.subplot(gs[1,2])
plt.imshow(
    subdct[100]['div EnTS-BW-sp'],
    cmap    = cmap,
    vmin    = subdct[100]['div min'],
    vmax    = subdct[100]['div max'])
plt.xticks([])
plt.yticks([])

# Plot the TM RTS (multi-pass)
plt.subplot(gs[1,3])
plt.imshow(
    subdct[100]['div EnTS-BW-mp'],
    cmap    = cmap,
    vmin    = subdct[100]['div min'],
    vmax    = subdct[100]['div max'])
plt.xticks([])
plt.yticks([])

plt.subplot(gs[1,4])
plt.axis('off')
ax = plt.gca()
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height*0.05, box.width, box.height*0.9])
axColor = plt.axes([box.x0 - box.width*1.05, box.y0 + box.height*0.05, 0.01, box.height*0.9])
# plt.colorbar(cax = axColor, orientation="vertical")
cbar = plt.colorbar(cax = axColor, orientation="vertical",label="variance")
cbar.set_label(label="variance",size=12)
cbar.ax.set_yticklabels(["{:4.2f}".format(i) for i in cbar.get_ticks()])



#%%

# Plot the Kalman Smoother reference
plt.subplot(gs[2,0])
# plt.title('KS \n (analytical)', fontsize=12)
plt.imshow(
    subdct[1000]['KS'],
    cmap    = cmap,
    vmin    = subdct[1000]['cov min'],
    vmax    = subdct[1000]['cov max'])
plt.xticks([])
plt.yticks([])
plt.ylabel('covariance', fontsize=12)

plt.gca().annotate('', xy=(-0.28, -1.25-0.02), xycoords='axes fraction', xytext=(-0.28, 1.+0.02), 
            arrowprops=dict(arrowstyle = '-',color='xkcd:dark grey'))
plt.gca().annotate('', xy=(-0.1, 1.), xycoords='axes fraction', xytext=(-0.3, 1.), 
            arrowprops=dict(arrowstyle = '-',color='xkcd:dark grey'))
plt.gca().annotate('', xy=(-0.1, -1.25), xycoords='axes fraction', xytext=(-0.3, -1.25), 
            arrowprops=dict(arrowstyle = '-',color='xkcd:dark grey'))
plt.gca().text(-0.35, -0.1, '$N=1000$', transform=plt.gca().transAxes, fontsize=12,
        verticalalignment='center',horizontalalignment='center',color='xkcd:dark grey',rotation=90)


# Plot the TM Kalman Smoother
plt.subplot(gs[2,1])
plt.imshow(
    subdct[1000]['EnTS-JA'],
    cmap    = cmap,
    vmin    = subdct[1000]['cov min'],
    vmax    = subdct[1000]['cov max'])
plt.xticks([])
plt.yticks([])


# Plot the TM RTS (single-pass)
plt.subplot(gs[2,2])
# plt.title('TM RTS \n (single-pass)', fontsize=12)
plt.imshow(
    subdct[1000]['EnTS-BW-sp'],
    cmap    = cmap,
    vmin    = subdct[1000]['cov min'],
    vmax    = subdct[1000]['cov max'])
plt.xticks([])
plt.yticks([])

# Plot the TM RTS (multi-pass)
plt.subplot(gs[2,3])
# plt.title('TM RTS \n (multi-pass)', fontsize=12)
plt.imshow(
    subdct[1000]['EnTS-BW-mp'],
    cmap    = cmap,
    vmin    = subdct[1000]['cov min'],
    vmax    = subdct[1000]['cov max'])
plt.xticks([])
plt.yticks([])

plt.subplot(gs[2,4])
plt.axis('off')
ax = plt.gca()
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height*0.05, box.width, box.height*0.9])
axColor = plt.axes([box.x0 - box.width*1.05, box.y0 + box.height*0.05, 0.01, box.height*0.9])
cbar = plt.colorbar(cax = axColor, orientation="vertical",label="variance")
cbar.set_label(label="variance",size=12)
cbar.ax.set_yticklabels(["{:4.2f}".format(i) for i in cbar.get_ticks()])


# Plot the Kalman Smoother reference
plt.subplot(gs[3,0])
plt.imshow(
    subdct[1000]['div KS'],
    cmap    = cmap,
    vmin    = subdct[1000]['div min'],
    vmax    = subdct[1000]['div max'])
plt.xticks([])
plt.yticks([])
plt.ylabel('mismatch', fontsize=12)

# Plot the TM Kalman Smoother
plt.subplot(gs[3,1])
plt.imshow(
    subdct[1000]['div EnTS-JA'],
    cmap    = cmap,
    vmin    = subdct[1000]['div min'],
    vmax    = subdct[1000]['div max'])
plt.xticks([])
plt.yticks([])

# Plot the TM RTS (single-pass)
plt.subplot(gs[3,2])
plt.imshow(
    subdct[1000]['div EnTS-BW-sp'],
    cmap    = cmap,
    vmin    = subdct[1000]['div min'],
    vmax    = subdct[1000]['div max'])
plt.xticks([])
plt.yticks([])

# Plot the TM RTS (multi-pass)
plt.subplot(gs[3,3])
plt.imshow(
    subdct[1000]['div EnTS-BW-mp'],
    cmap    = cmap,
    vmin    = subdct[1000]['div min'],
    vmax    = subdct[1000]['div max'])
plt.xticks([])
plt.yticks([])

plt.subplot(gs[3,4])
plt.axis('off')
ax = plt.gca()
box = ax.get_position()
ax.set_position([box.x0, box.y0 + box.height*0.05, box.width, box.height*0.9])
axColor = plt.axes([box.x0 - box.width*1.05, box.y0 + box.height*0.05, 0.01, box.height*0.9])
# plt.colorbar(cax = axColor, orientation="vertical")
cbar = plt.colorbar(cax = axColor, orientation="vertical",label="variance")
cbar.set_label(label="variance",size=12)
cbar.ax.set_yticklabels(["{:4.2f}".format(i) for i in cbar.get_ticks()])

#%%




plt.savefig('smoothing_covariances'+'_order='+str(order)+'.png',dpi=600,bbox_inches='tight')
plt.savefig('smoothing_covariances'+'_order='+str(order)+'.pdf',dpi=600,bbox_inches='tight')


