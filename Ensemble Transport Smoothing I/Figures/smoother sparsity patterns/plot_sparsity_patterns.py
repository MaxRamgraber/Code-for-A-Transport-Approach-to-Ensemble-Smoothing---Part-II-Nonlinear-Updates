import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import scipy.stats
import turbo_colormap

cmap = matplotlib.cm.get_cmap('turbo')

plt.close('all')

plt.figure(figsize=(12,9))

gs  = matplotlib.gridspec.GridSpec(nrows = 3, ncols = 3, height_ratios = [1,1,0.05], hspace = 0.25)

#%%

# Plot legend

from matplotlib import gridspec

gs2     = gridspec.GridSpecFromSubplotSpec(
    nrows           = 1,
    ncols           = 2,
    wspace          = 0.0,
    width_ratios    = [1,1.5],
    subplot_spec    = gs[2,:])

plt.subplot(gs2[0,0])

ax = plt.gca()

norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
cb1 = matplotlib.colorbar.ColorbarBase(
    ax, 
    cmap        = cmap,
    ticks       = [-1, 1],
    norm        = norm,
    orientation = 'horizontal')

cb1.set_label("$\leq \qquad \qquad s \qquad \qquad \leq$", labelpad=-10)

plt.gca().xaxis.set_ticks_position('top')
plt.gca().xaxis.set_label_position('top')
cb1.ax.set_xticklabels(['$1$', '$t$'])  # horizontal colorbar

plt.show()



plt.subplot(gs2[0,1])

plt.text(
    x       = 0.1,
    y       = 0.5,
    s       = "$\mathbf{Y}$",
    fontsize= 12,
    ha      = "left",
    va      = "center",
    transform = plt.gca().transAxes,
    color   = cmap(0.9))

plt.text(
    x       = 0.15,
    y       = 0.5,
    s       = "coefficient block for $\mathbf{Y}_{s}$",
    fontsize= 10,
    ha      = "left",
    va      = "center",
    transform = plt.gca().transAxes)


plt.text(
    x       = 0.55,
    y       = 0.5,
    s       = "$\mathbf{X}$",
    fontsize= 12,
    ha      = "left",
    va      = "center",
    transform = plt.gca().transAxes,
    color   = cmap(0.1))

plt.text(
    x       = 0.6,
    y       = 0.5,
    s       = "coefficient block for $\mathbf{X}_{s}$",
    fontsize= 10,
    ha      = "left",
    va      = "center",
    transform = plt.gca().transAxes)


plt.gca().axis("off")

















#%%

plt.subplot(gs[0,0])

plt.title(r'$\bf{A}$: dense EnTS (ordering A)', loc='left', fontsize=10)

ax  = plt.gca()

T   = 13

for t in range(T+1):
    
    for s in range(t+1):
        
        if s == 0:
            
            plt.text(
                x       = s,
                y       = T+1-t,
                s       = "$\mathbf{Y}$",
                fontsize= 12,
                ha      = "center",
                va      = "center",
                color   = cmap((T-s-1)/(T-1)),
                alpha   = 1)
            
        else:
            
            plt.text(
                x       = s,
                y       = T+1-t,
                s       = "$\mathbf{X}$",
                fontsize= 12,
                ha      = "center",
                va      = "center",
                color   = cmap((T-s)/(T-1)),
                alpha   = 1)
            
ax.set_xlim([-1,T+1])
ax.set_ylim([0,T+2])

plt.axis('off')

#%%

plt.subplot(gs[0,1])

plt.title(r'$\bf{B}$: sparse EnTS (ordering A)', loc='left', fontsize=10)

ax  = plt.gca()

T   = 13

for t in range(T+1):
    
    for s in range(t+1):
        
        if s+1 >= t:
            alpha   = 1.
        else:
            alpha   = 0.15
        
        if s == 0:
            
            plt.text(
                x       = s,
                y       = T+1-t,
                s       = "$\mathbf{Y}$",
                fontsize= 12,
                ha      = "center",
                va      = "center",
                color   = cmap((T-s-1)/(T-1)),
                alpha   = alpha)
            
        else:
            
            plt.text(
                x       = s,
                y       = T+1-t,
                s       = "$\mathbf{X}$",
                fontsize= 12,
                ha      = "center",
                va      = "center",
                color   = cmap((T-s)/(T-1)),
                alpha   = alpha)
            
ax.set_xlim([-1,T+1])
ax.set_ylim([0,T+2])

plt.axis('off')

#%%

plt.subplot(gs[0,2])

plt.title(r'$\bf{C}$: separated EnTS (ordering A)', loc='left', fontsize=10)

ax  = plt.gca()

T   = 13

for t in range(T+1):
    
    for s in range(t+1):
        
        if s+1 >= t:
            alpha   = 1.
        else:
            alpha   = 0.15
        
        if s == 0:
            
            if alpha > 0.15:
            
                plt.text(
                    x       = s,
                    y       = T+1-t,
                    s       = "$\mathbf{Y}$",
                    fontsize= 12,
                    ha      = "center",
                    va      = "center",
                    color   = cmap((T-s-1)/(T-1)),
                    alpha   = alpha)
                
        else:
            
            if alpha > 0.15:
            
                plt.text(
                    x       = s,
                    y       = T+1-t,
                    s       = "$\mathbf{X}$",
                    fontsize= 12,
                    ha      = "center",
                    va      = "center",
                    color   = cmap((T-s)/(T-1)),
                    alpha   = alpha)
            
        if s == t and s != T:
            
            plt.plot(
                [s-0.5,s+2-0.5,s+2-0.5,s-0.5,s-0.5],
                [T-t-0.4,T-t-0.4,T-t+2-0.4,T-t+2-0.4,T-t-0.4],
                color   = cmap((T-s-1)/(T-1)),
                lw      = 1)
            
            if t == 0:
                var     = 'Y'
            else:
                var     = 'X'
            
            if t == 0:
            
                plt.text(
                    x       = s+2-0.1,
                    y       = T-t+1+0.1,
                    s       = "map for $(\mathbf{"+var+"}_{"+str(T-t+1)+"},\mathbf{X}_{"+str(T-t+1)+"})$",
                    ha      = "left",
                    va      = "center",
                    color   = cmap((T-s-1)/(T-1)),
                    fontsize= 8)
                
            elif t > 0 and t < 7:
                
                plt.text(
                    x       = s+2-0.1,
                    y       = T-t+1+0.1,
                    s       = "map for $(\mathbf{"+var+"}_{"+str(T-t+1)+"},\mathbf{X}_{"+str(T-t-1+1)+"})$",
                    ha      = "left",
                    va      = "center",
                    color   = cmap((T-s-1)/(T-1)),
                    fontsize= 8)
                
            else:
                
                plt.text(
                    x       = s-0.6,
                    y       = T-t-0.1,
                    s       = "map for $(\mathbf{"+var+"}_{"+str(T-t+1)+"},\mathbf{X}_{"+str(T-t-1+1)+"})$",
                    ha      = "right",
                    va      = "center",
                    color   = cmap((T-s-1)/(T-1)),
                    fontsize= 8)
            
ax.set_xlim([-1,T+1])
ax.set_ylim([0,T+2])

plt.axis('off')


#%%

plt.subplot(gs[1,0])

plt.title(r'$\bf{D}$: dense EnTS (ordering B)', loc='left', fontsize=10)

ax  = plt.gca()

T   = 13

for t in range(T+1):
    
    for s in range(t+1):
        
        if s == 0:
            
            plt.text(
                x       = s,
                y       = T+1-t,
                s       = "$\mathbf{Y}$",
                fontsize= 12,
                ha      = "center",
                va      = "center",
                color   = cmap((T-s-1)/(T-1)),
                alpha   = alpha)
            
        else:
            
            plt.text(
                x       = s,
                y       = T+1-t,
                s       = "$\mathbf{X}$",
                fontsize= 12,
                ha      = "center",
                va      = "center",
                color   = cmap((s-1)/(T-1)),
                alpha   = alpha)
            
ax.set_xlim([-1,T+1])
ax.set_ylim([0,T+2])

plt.axis('off')

#%%

plt.subplot(gs[1,1])

plt.title(r'$\bf{E}$: sparse EnTS (ordering B)', loc='left', fontsize=10)

ax  = plt.gca()

T   = 13

for t in range(T+1):
    
    for s in range(t+1):
        
        if s == 0:
            
            plt.text(
                x       = s,
                y       = T+1-t,
                s       = "$\mathbf{Y}$",
                fontsize= 12,
                ha      = "center",
                va      = "center",
                color   = cmap((T-s-1)/(T-1)),
                alpha   = alpha)
            
        else:
            
            if s == t or s+1 >= t:
                alpha   = 1.
            else:
                alpha   = 0.15
            
            plt.text(
                x       = s,
                y       = T+1-t,
                s       = "$\mathbf{X}$",
                fontsize= 12,
                ha      = "center",
                va      = "center",
                color   = cmap((s-1)/(T-1)),
                alpha   = alpha)
        
            
ax.set_xlim([-1,T+1])
ax.set_ylim([0,T+2])

plt.axis('off')

#%%

plt.subplot(gs[1,2])

plt.title(r'$\bf{F}$: defective sparsity (ordering B)', loc='left', fontsize=10)

ax  = plt.gca()

T   = 13

for t in range(T+1):
    
    for s in range(t+1):
        
        if s == 0:
            
            plt.text(
                x       = s,
                y       = T+1-t,
                s       = "$\mathbf{Y}$",
                fontsize= 12,
                ha      = "center",
                va      = "center",
                color   = cmap((T-s-1)/(T-1)),
                alpha   = alpha)
            
        else:
            
            if s == t:
                alpha   = 1.
            else:
                alpha   = 0.15
            
            plt.text(
                x       = s,
                y       = T+1-t,
                s       = "$\mathbf{X}$",
                fontsize= 12,
                ha      = "center",
                va      = "center",
                color   = cmap((s-1)/(T-1)),
                alpha   = alpha)
        
ax.set_xlim([-1,T+1])
ax.set_ylim([0,T+2])

plt.axis('off')


plt.savefig('sparsity_patterns.png',dpi=600,bbox_inches='tight')
plt.savefig('sparsity_patterns.pdf',dpi=600,bbox_inches='tight')