import pickle
import matplotlib.pyplot as plt
import numpy as np
import copy
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter

plt.close('all')


dct     = pickle.load(open("autoregressive_model_results.p","rb"))

subdct  = {}

for N in [100,1000]:
    
    subdct[N]   = {
        'EnTS-JA'       : {'signal' : [], 'gain' : []},
        'EnTS-BW-sp'    : {'signal' : [], 'gain' : []},
        'EnTS-BW-mp'    : {'signal' : [], 'gain' : []}}

    for seed in list(dct.keys()):
        
        subdct[N]['EnTS-JA']['signal']      .append(np.abs(dct[seed][N]['EnTS (joint-analysis)']['signal']))
        subdct[N]['EnTS-JA']['gain']        .append(np.abs(dct[seed][N]['EnTS (joint-analysis)']['map']))
        subdct[N]['EnTS-BW-sp']['signal']   .append(np.abs(dct[seed][N]['EnTS (backward, single-pass)']['signal']))
        subdct[N]['EnTS-BW-sp']['gain']     .append(np.abs(dct[seed][N]['EnTS (backward, single-pass)']['map']))
        subdct[N]['EnTS-BW-mp']['signal']   .append(np.abs(dct[seed][N]['EnTS (backward, multi-pass)']['signal']))
        subdct[N]['EnTS-BW-mp']['gain']     .append(np.abs(dct[seed][N]['EnTS (backward, multi-pass)']['map']))
     
    
        
    # Average signals across random seeds
    subdct[N]['EnTS-JA']['signal']      = np.mean(np.asarray(subdct[N]['EnTS-JA']['signal']),       axis = 0)
    subdct[N]['EnTS-BW-sp']['signal']   = np.mean(np.asarray(subdct[N]['EnTS-BW-sp']['signal']),    axis = 0)
    subdct[N]['EnTS-BW-mp']['signal']   = np.mean(np.asarray(subdct[N]['EnTS-BW-mp']['signal']),    axis = 0)
    
    # Average signals across samples
    subdct[N]['EnTS-JA']['signal']      = np.mean(np.asarray(subdct[N]['EnTS-JA']['signal']),       axis = -1)
    subdct[N]['EnTS-BW-sp']['signal']   = np.mean(np.asarray(subdct[N]['EnTS-BW-sp']['signal']),    axis = -1)
    subdct[N]['EnTS-BW-mp']['signal']   = np.mean(np.asarray(subdct[N]['EnTS-BW-mp']['signal']),    axis = -1)
    

    # Average gains across random seeds
    subdct[N]['EnTS-JA']['gain']      = np.mean(np.asarray(subdct[N]['EnTS-JA']['gain']),       axis = 0)
    subdct[N]['EnTS-BW-sp']['gain']   = np.mean(np.asarray(subdct[N]['EnTS-BW-sp']['gain']),    axis = 0)
    subdct[N]['EnTS-BW-mp']['gain']   = np.mean(np.asarray(subdct[N]['EnTS-BW-mp']['gain']),    axis = 0)
    
    # Average gains across samples
    subdct[N]['EnTS-JA']['gain']      = np.mean(np.asarray(subdct[N]['EnTS-JA']['gain']),       axis = -1)
    subdct[N]['EnTS-BW-sp']['gain']   = np.mean(np.asarray(subdct[N]['EnTS-BW-sp']['gain']),    axis = -1)
    subdct[N]['EnTS-BW-mp']['gain']   = np.mean(np.asarray(subdct[N]['EnTS-BW-mp']['gain']),    axis = -1)
    

plt.figure(figsize=(10,7.4))
gs  = GridSpec(nrows = 3, ncols = 1)



# =============================================================================
# Joint analysis smoother
# =============================================================================

plt.subplot(gs[2,0])
ax1     = plt.gca()
ax2     = ax1.twinx()

alg     = "TM_KS"#"EnKSr"
plt.title(r"$\bf{C}$: joint-analysis smoother", loc='left')

# Plot the map
ax1.plot(np.arange(30)+1,subdct[1000]['EnTS-JA']['gain'],color='xkcd:crimson')
ax1.plot(np.arange(30)+1,subdct[100]['EnTS-JA']['gain'],color='xkcd:orangish red')

# Plot the signal
ax2.plot(np.arange(30)+1,subdct[1000]['EnTS-JA']['signal'],color='xkcd:crimson',linestyle='--')
ax2.plot(np.arange(30)+1,subdct[100]['EnTS-JA']['signal'],color='xkcd:orangish red',linestyle='--')

ax1.set_ylabel('mean abs. gain')
ax2.set_ylabel('mean abs. signal')
plt.xlabel('time steps (final smoothing pass)')
# plt.gca().set_xticklabels([])

# ax1.yaxis.label.set_color('xkcd:cerulean')
# ax2.yaxis.label.set_color('xkcd:orangish red')

ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

ax1.relim()
ax2.relim()

ax1.set_ylim([0,ax1.get_ylim()[1]*1.3])
ax2.set_ylim([0,ax2.get_ylim()[1]*1.3])


# Create a custom legend
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color='xkcd:orangish red',  label='signal $N=100$',linestyle='--'),
                   Line2D([0], [0], color='xkcd:crimson',       label='signal $N=1000$',linestyle='--'),
                   Line2D([0], [0], color='xkcd:orangish red',  label='gain $N=100$'),
                   Line2D([0], [0], color='xkcd:crimson',       label='gain $N=1000$')]
ax1.legend(
    handles         = legend_elements, 
    ncol            = 4,
    loc             = 'upper right',
    bbox_to_anchor  = (1.0, 1.05),
    frameon         = False,
    fancybox        = False, 
    shadow          = False)


xlim    = ax1.get_xlim()

ax1.set_xlabel('time steps (final smoothing pass)')


# =============================================================================
# EnRTS single-pass
# =============================================================================

plt.subplot(gs[1,0])
ax1     = plt.gca()
ax1.set_xlim(xlim)
ax2     = ax1.twinx()

alg     = "RTS_sp"#"EnRTSr"
plt.title(r"$\bf{B}$: backward smoother (single-pass)", loc='left')

# Plot the map
ax1.plot(np.arange(29)+1,subdct[1000]['EnTS-BW-sp']['gain'][:-1],color='xkcd:pine')
ax1.plot(np.arange(29)+1,subdct[100]['EnTS-BW-sp']['gain'][:-1],color='xkcd:grass green')

# Plot the signal
ax2.plot(np.arange(29)+1,subdct[1000]['EnTS-BW-sp']['signal'][:-1],color='xkcd:pine',linestyle='--')
ax2.plot(np.arange(29)+1,subdct[100]['EnTS-BW-sp']['signal'][:-1],color='xkcd:grass green',linestyle='--')

ax1.set_ylabel('mean abs. gain')
ax2.set_ylabel('mean abs. signal')
plt.gca().set_xticklabels([])

# ax1.yaxis.label.set_color('xkcd:cerulean')
# ax2.yaxis.label.set_color('xkcd:orangish red')

ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

ax1.relim()
ax2.relim()

ax1.set_ylim([0,ax1.get_ylim()[1]*1.3])
ax2.set_ylim([0,ax2.get_ylim()[1]*1.3])

# Create a custom legend
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color='xkcd:grass green',   label='signal $N=100$',linestyle='--'),
                    Line2D([0], [0], color='xkcd:pine',          label='signal $N=1000$',linestyle='--'),
                    Line2D([0], [0], color='xkcd:grass green',   label='gain $N=100$'),
                    Line2D([0], [0], color='xkcd:pine',          label='gain $N=1000$')]
ax1.legend(
    handles         = legend_elements, 
    ncol            = 4,
    loc             = 'upper right',
    bbox_to_anchor  = (1.0, 1.05),
    frameon         = False,
    fancybox        = False, 
    shadow          = False)

# =============================================================================
# EnRTS multi-pass
# =============================================================================

plt.subplot(gs[0,0])
ax1     = plt.gca()
ax1.set_xlim(xlim)
ax2     = ax1.twinx()

alg     = "RTS_mp"#"EnRTSr_mp"
plt.title(r"$\bf{A}$: backward smoother (multi-pass)", loc='left')
# Plot the map
ax1.plot(np.arange(29)+1,subdct[1000]['EnTS-BW-mp']['gain'][:-1],color='xkcd:cobalt')
ax1.plot(np.arange(29)+1,subdct[100]['EnTS-BW-mp']['gain'][:-1],color='xkcd:cerulean')

# Plot the signal
ax2.plot(np.arange(29)+1,subdct[1000]['EnTS-BW-mp']['signal'][:-1],color='xkcd:cobalt',linestyle='--')
ax2.plot(np.arange(29)+1,subdct[100]['EnTS-BW-mp']['signal'][:-1],color='xkcd:cerulean',linestyle='--')

ax1.set_ylabel('mean abs. gain')
ax2.set_ylabel('mean abs. signal')
plt.xlabel('time steps (final smoothing pass)')
plt.gca().set_xticklabels([])

# ax1.yaxis.label.set_color('xkcd:cerulean')
# ax2.yaxis.label.set_color('xkcd:orangish red')

ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

ax1.relim()
ax2.relim()

ax1.set_ylim([0,ax1.get_ylim()[1]*1.3])
ax2.set_ylim([0,ax2.get_ylim()[1]*1.3])

# Create a custom legend
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], color='xkcd:cerulean',      label='signal $N=100$',linestyle='--'),
                   Line2D([0], [0], color='xkcd:cobalt',        label='signal $N=1000$',linestyle='--'),
                   Line2D([0], [0], color='xkcd:cerulean',      label='gain $N=100$'),
                   Line2D([0], [0], color='xkcd:cobalt',        label='gain $N=1000$')]
ax1.legend(
    handles         = legend_elements, 
    ncol            = 4,
    loc             = 'upper right',
    bbox_to_anchor  = (1.0, 1.05),
    frameon         = False,
    fancybox        = False, 
    shadow          = False)



plt.savefig('signal_vs_gain.png',dpi=600,bbox_inches='tight')
plt.savefig('signal_vs_gain.pdf',dpi=600,bbox_inches='tight')
