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
subdct  = pickle.load(open("autoregressive_signalgain.p","rb"))

algs = ['EnKS','EnRTSS (single-pass)','EnRTSS (multi-pass)','EnFIT (multi-pass)']
algs_keys = ['EnTS-JA','EnTS-BW-sp','EnTS-BW-mp','EnTS-FW-mp']

# plot results
plt.figure(figsize=(8,8))
gs  = GridSpec(nrows=5, ncols=1)

# =============================================================================
# Joint analysis smoother
# =============================================================================

plt.subplot(gs[4,0])
ax1     = plt.gca()
ax2     = ax1.twinx()

plt.title(r"Dense smoother", loc='left')

# Plot the map
ax1.plot(np.arange(30)+1,subdct[1000]['EnTS-JA']['gain'],color='xkcd:crimson',marker="^")
ax1.plot(np.arange(30)+1,subdct[100]['EnTS-JA']['gain'],color='xkcd:orangish red',marker='v')

# Plot the signal
ax2.plot(np.arange(30)+1,subdct[1000]['EnTS-JA']['signal'],color='xkcd:crimson',linestyle='--',marker='+')
ax2.plot(np.arange(30)+1,subdct[100]['EnTS-JA']['signal'],color='xkcd:orangish red',linestyle='--',marker='x')

ax1.set_ylabel('Mean abs. gain')
ax2.set_ylabel('Mean abs. signal')
#plt.xlabel('Time steps (final smoothing pass)')
#plt.gca().set_xticklabels([])

ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

ax1.relim()
ax2.relim()

ax1.set_xlim([0.5,30.5])
ax2.set_xlim([0.5,30.5])
ax1.set_ylim([0,ax1.get_ylim()[1]*1.3])
ax2.set_ylim([0,ax2.get_ylim()[1]*1.3])

# Create a custom legend
legend_elements = [Line2D([0], [0], color='xkcd:orangish red',  label='signal $N=100$',linestyle='--',marker='x'),
                   Line2D([0], [0], color='xkcd:crimson',       label='signal $N=1000$',linestyle='--',marker='+'),
                   Line2D([0], [0], color='xkcd:orangish red',  label='gain $N=100$',marker='v'),
                   Line2D([0], [0], color='xkcd:crimson',       label='gain $N=1000$',marker='^')]
ax1.legend(
    handles         = legend_elements, 
    ncol            = 4,
    loc             = 'upper right',
    bbox_to_anchor  = (1.0, 1.05),
    frameon         = False,
    fancybox        = False, 
    shadow          = False)

xlim    = ax1.get_xlim()
ax1.set_xlabel('Time steps (final smoothing pass)')

#xlim    = ax1.get_xlim()
#ax1.set_xlabel('time steps (final smoothing pass)')

# =============================================================================
# Forward smoother
# =============================================================================

plt.subplot(gs[3,0])
ax1     = plt.gca()
ax2     = ax1.twinx()
plt.title(r"Forward mutli-pass smoother - Contribution from state $\mathbf{B}_s(\mathbf{X}_{s-1}^* - \mathbf{X}_{s-1})$", loc='left')

# Plot the map
ax1.plot(np.arange(30)+1,subdct[1000]['EnTS-FW-mp']['gain'][:,1],color='xkcd:yellow orange',marker='v')
ax1.plot(np.arange(30)+1,subdct[100]['EnTS-FW-mp']['gain'][:,1],color='xkcd:light mustard',marker='^')

# Plot the signal
ax2.plot(np.arange(30)+1,subdct[1000]['EnTS-FW-mp']['signal'][:,1],color='xkcd:yellow orange',linestyle='--',marker='+')
ax2.plot(np.arange(30)+1,subdct[100]['EnTS-FW-mp']['signal'][:,1],color='xkcd:light mustard',linestyle='--',marker='x')

ax1.set_ylabel('Mean abs. gain')
ax2.set_ylabel('Mean abs. signal')
#plt.xlabel('time steps (final smoothing pass)')
plt.gca().set_xticklabels([])

ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

ax1.relim()
ax2.relim()

ax1.set_xlim([0.5,30.5])
ax2.set_xlim([0.5,30.5])
ax1.set_ylim([0,ax1.get_ylim()[1]*1.3])
ax2.set_ylim([0,ax2.get_ylim()[1]*1.3])


# Create a custom legend
legend_elements = [Line2D([0], [0], color='xkcd:light mustard',  label='signal $N=100$',linestyle='--',marker='x'),
                   Line2D([0], [0], color='xkcd:yellow orange',  label='signal $N=1000$',linestyle='--',marker='+'),
                   Line2D([0], [0], color='xkcd:light mustard',  label='gain $N=100$',marker='v'),
                   Line2D([0], [0], color='xkcd:yellow orange',  label='gain $N=1000$',marker='^')]
ax1.legend(
    handles         = legend_elements, 
    ncol            = 4,
    loc             = 'upper right',
    bbox_to_anchor  = (1.0, 1.05),
    frameon         = False,
    fancybox        = False, 
    shadow          = False)



plt.subplot(gs[2,0])
ax1     = plt.gca()
ax2     = ax1.twinx()
plt.title(r"Forward multi-pass smoother - Contribution from observations $\mathbf{K}_s(\mathbf{y}_t^* - \mathbf{Y}_t)$", loc='left')

# Plot the map
ax1.plot(np.arange(30)+1,subdct[1000]['EnTS-FW-mp']['gain'][:,0],color='xkcd:yellow orange',marker='v')
ax1.plot(np.arange(30)+1,subdct[100]['EnTS-FW-mp']['gain'][:,0],color='xkcd:light mustard',marker='^')

# Plot the signal
ax2.plot(np.arange(30)+1,subdct[1000]['EnTS-FW-mp']['signal'][:,0],color='xkcd:yellow orange',linestyle='--',marker='+')
ax2.plot(np.arange(30)+1,subdct[100]['EnTS-FW-mp']['signal'][:,0],color='xkcd:light mustard',linestyle='--',marker='x')

ax1.set_ylabel('Mean abs. gain')
ax2.set_ylabel('Mean abs. signal')
plt.gca().set_xticklabels([])

ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

ax1.relim()
ax2.relim()

ax1.set_xlim([0.5,30.5])
ax2.set_xlim([0.5,30.5])
ax1.set_ylim([0,ax1.get_ylim()[1]*1.3])
ax2.set_ylim([0,ax2.get_ylim()[1]*1.3])

# Create a custom legend
legend_elements = [Line2D([0], [0], color='xkcd:light mustard',  label='signal $N=100$',linestyle='--',marker='x'),
                   Line2D([0], [0], color='xkcd:yellow orange',  label='signal $N=1000$',linestyle='--',marker='+'),
                   Line2D([0], [0], color='xkcd:light mustard',  label='gain $N=100$',marker='v'),
                   Line2D([0], [0], color='xkcd:yellow orange',  label='gain $N=1000$',marker='^')]
ax1.legend(
    handles         = legend_elements, 
    ncol            = 4,
    loc             = 'upper right',
    bbox_to_anchor  = (1.0, 1.05),
    frameon         = False,
    fancybox        = False, 
    shadow          = False)

# =============================================================================
# EnRTS single-pass
# =============================================================================

plt.subplot(gs[1,0])
ax1     = plt.gca()
ax1.set_xlim(xlim)
ax2     = ax1.twinx()

alg     = "RTS_sp"#"EnRTSr"
plt.title(r"Backward single-pass smoother", loc='left')

# Plot the map
ax1.plot(np.arange(29)+1,subdct[1000]['EnTS-BW-sp']['gain'][:-1],color='xkcd:pine',marker='^')
ax1.plot(np.arange(29)+1,subdct[100]['EnTS-BW-sp']['gain'][:-1],color='xkcd:grass green',marker='v')

# Plot the signal
ax2.plot(np.arange(29)+1,subdct[1000]['EnTS-BW-sp']['signal'][:-1],color='xkcd:pine',linestyle='--',marker='+')
ax2.plot(np.arange(29)+1,subdct[100]['EnTS-BW-sp']['signal'][:-1],color='xkcd:grass green',linestyle='--',marker='x')

ax1.set_ylabel('Mean abs. gain')
ax2.set_ylabel('Mean abs. signal')
plt.gca().set_xticklabels([])

# ax1.yaxis.label.set_color('xkcd:cerulean')
# ax2.yaxis.label.set_color('xkcd:orangish red')

ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

ax1.relim()
ax2.relim()

ax1.set_xlim([0.5,30.5])
ax2.set_xlim([0.5,30.5])
ax1.set_ylim([0,ax1.get_ylim()[1]*1.3])
ax2.set_ylim([0,ax2.get_ylim()[1]*1.3])

# Create a custom legend
legend_elements = [Line2D([0], [0], color='xkcd:grass green',   label='signal $N=100$',linestyle='--',marker='x'),
                    Line2D([0], [0], color='xkcd:pine',          label='signal $N=1000$',linestyle='--',marker='+'),
                    Line2D([0], [0], color='xkcd:grass green',   label='gain $N=100$',marker='v'),
                    Line2D([0], [0], color='xkcd:pine',          label='gain $N=1000$',marker='^')]
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

plt.title(r"Backward multi-pass smoother", loc='left')
# Plot the map
ax1.plot(np.arange(30)+1,subdct[1000]['EnTS-BW-mp']['gain'],color='xkcd:cobalt',marker='^')
ax1.plot(np.arange(30)+1,subdct[100]['EnTS-BW-mp']['gain'],color='xkcd:cerulean',marker='v')

# Plot the signal
ax2.plot(np.arange(30)+1,subdct[1000]['EnTS-BW-mp']['signal'],color='xkcd:cobalt',linestyle='--',marker='+')
ax2.plot(np.arange(30)+1,subdct[100]['EnTS-BW-mp']['signal'],color='xkcd:cerulean',linestyle='--',marker='x')

ax1.set_ylabel('Mean abs. gain')
ax2.set_ylabel('Mean abs. signal')
plt.xlabel('time steps (final smoothing pass)')
plt.gca().set_xticklabels([])

ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

ax1.relim()
ax2.relim()

ax1.set_xlim([0.5,30.5])
ax2.set_xlim([0.5,30.5])
ax1.set_ylim([0,ax1.get_ylim()[1]*1.3])
ax2.set_ylim([0,ax2.get_ylim()[1]*1.3])

# Create a custom legend
legend_elements = [Line2D([0], [0], color='xkcd:cerulean',      label='signal $N=100$',linestyle='--',marker='x'),
                   Line2D([0], [0], color='xkcd:cobalt',        label='signal $N=1000$',linestyle='--',marker='+'),
                   Line2D([0], [0], color='xkcd:cerulean',      label='gain $N=100$',marker='v'),
                   Line2D([0], [0], color='xkcd:cobalt',        label='gain $N=1000$',marker='^')]
ax1.legend(
    handles         = legend_elements, 
    ncol            = 4,
    loc             = 'upper right',
    bbox_to_anchor  = (1.0, 1.05),
    frameon         = False,
    fancybox        = False, 
    shadow          = False)

plt.tight_layout()
plt.savefig('signal_vs_gain.pdf',dpi=600,bbox_inches='tight')
