import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import turbo_colormap

plt.rc('font', family='serif') # sans-serif
plt.rc('text', usetex=True)

plt.rcParams['text.latex.preamble'] = [
       r'\usepackage{siunitx}',   # i need upright \micro symbols, but you need...
       r'\sisetup{detect-all}',   # ...this to force siunitx to actually use your fonts
       r'\usepackage{helvet}',    # set the normal font here
       r'\usepackage{sansmath}',  # load up the sansmath so that math -> helvet
       r'\sansmath'               # <- tricky! -- gotta actually tell tex to use!
]  

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", 
    ["xkcd:midnight blue",
     "xkcd:sky blue",
     "xkcd:light sky blue"])

plt.close('all')

Nnodes  = 5

plt.figure(figsize=(12,11))
gs  = matplotlib.gridspec.GridSpec(nrows=4,ncols=3,height_ratios=[0.1,0.3,1.,1.],wspace=0.1)

#%%

plt.subplot(gs[0,:])

cmap = matplotlib.cm.get_cmap('turbo')
import matplotlib
norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
cb1 = matplotlib.colorbar.ColorbarBase(
    plt.gca(), 
    cmap        = cmap,
    ticks       = [-1, 1],
    norm        = norm,
    orientation = 'horizontal')

cb1.set_label("algorithm progress", labelpad=-8,fontsize = 12)

plt.gca().xaxis.set_ticks_position('top')
plt.gca().xaxis.set_label_position('top')
cb1.ax.set_xticklabels(['start', 'end'],fontsize = 12)  # horizontal colorbar


plt.gca().annotate('', xy=(0.1, 1.5), xycoords='axes fraction', xytext=(0.425, 1.5), 
            arrowprops=dict(arrowstyle = '-',color='xkcd:dark grey'))
plt.gca().annotate('', xy=(0.9, 1.5), xycoords='axes fraction', xytext=(0.575, 1.5), 
            arrowprops=dict(arrowstyle = '->',color='xkcd:dark grey'))

xpos    = np.column_stack((
    np.cos(np.linspace(2*np.pi,0,Nnodes+1)[:-1]+np.pi/2),
    np.sin(np.linspace(2*np.pi,0,Nnodes+1)[:-1]+np.pi/2) )) * 2


ypos    = np.column_stack((
    np.cos(np.linspace(2*np.pi,0,Nnodes+1)[:-1]+np.pi/2),
    np.sin(np.linspace(2*np.pi,0,Nnodes+1)[:-1]+np.pi/2) )) 

n = 0

filter_x    = []
filter_y    = []

T           = 20.

#%%


square_x    = np.asarray([+0.5,+0.5,-0.5,-0.5])*0.8
square_y    = np.asarray([+0.5,-0.5,-0.5,+0.5])*0.8

triang_x    = np.asarray([-0.5,0,+0.5])*0.8
triang_y    = np.asarray([-0.5,+0.5,-0.5])*0.8

circ_x      = np.cos(np.linspace(-np.pi,np.pi,36))*0.5*0.8
circ_y      = np.sin(np.linspace(-np.pi,np.pi,36))*0.5*0.8


plt.subplot(gs[1,:])


plt.fill(1 + triang_x, 0 + triang_y, color=cmap(0.25), edgecolor="None")

plt.fill(2 + circ_x, 0 + circ_y, color=cmap(0.3), edgecolor="None")

plt.gca().annotate('', xy=(2.15, 0), xytext=(0.85, 0), 
        arrowprops=dict(arrowstyle = '->',color='xkcd:dark grey'))

plt.gca().text(3, 0, "filtering forecast", ha="left", va="center",zorder=10,color="k",fontsize=10)


plt.fill(12 + triang_x, 0.5 + triang_y, color=cmap(0.55), edgecolor="None")

plt.fill(12 + circ_x, -0.5 + circ_y, color=cmap(0.5), edgecolor="None")

plt.gca().annotate('', xy=(12, 0.65), xytext=(12, -0.65), 
        arrowprops=dict(arrowstyle = '->',color='xkcd:dark grey'))

plt.gca().text(13, 0, "filtering update", ha="left", va="center",zorder=10,color="k",fontsize=10)

plt.fill(21 + np.asarray(list(triang_x[:2])+list(triang_x[1:]+1)), 0.5 + np.asarray(list(triang_y[:2])+list(triang_y[1:])), color=cmap(0.55), edgecolor="None")

plt.fill(22 + circ_x, -0.5 + circ_y, color=cmap(0.5), edgecolor="None")

plt.gca().annotate('', xy=(22, 0.65), xytext=(22, -0.65), 
        arrowprops=dict(arrowstyle = '->',color='xkcd:dark grey'))

plt.gca().text(23, 0, "dense \n"+"smoothing update", ha="left", va="center",zorder=10,color="k",fontsize=10)


plt.fill(32 + square_x, 0.5 + square_y, color=cmap(0.75), edgecolor="None")

plt.fill(32 + square_x, -0.5 + square_y, color=cmap(0.75), edgecolor="None")

plt.fill(33 + triang_x, 0.5 + triang_y, color=cmap(0.8), edgecolor="None")

plt.fill(33 + square_x, -0.5 + square_y, color=cmap(0.8), edgecolor="None")

plt.gca().annotate('', xy=(33.15, 0.5), xytext=(31.85, 0.5), 
        arrowprops=dict(arrowstyle = '->',color='xkcd:dark grey'))

plt.gca().annotate('', xy=(33.15, -0.5), xytext=(31.85, -0.5), 
        arrowprops=dict(arrowstyle = '->',color='xkcd:dark grey'))

plt.gca().text(34, 0, "forward \n"+"smoothing update", ha="left", va="center",zorder=10,color="k",fontsize=10)




plt.fill(43 + square_x, 0.5 + square_y, color=cmap(0.8), edgecolor="None")

plt.fill(43 + square_x, -0.5 + square_y, color=cmap(0.8), edgecolor="None")

plt.fill(44 + triang_x, 0.5 + triang_y, color=cmap(0.75), edgecolor="None")

plt.fill(44 + square_x, -0.5 + square_y, color=cmap(0.75), edgecolor="None")

plt.gca().annotate('', xy=(42.85, 0.5), xytext=(44.15, 0.5), 
        arrowprops=dict(arrowstyle = '->',color='xkcd:dark grey'))

plt.gca().annotate('', xy=(42.85, -0.5), xytext=(44.15, -0.5), 
        arrowprops=dict(arrowstyle = '->',color='xkcd:dark grey'))

plt.gca().text(45, 0, "backward \n"+"smoothing update", ha="left", va="center",zorder=10,color="k",fontsize=10)



# Dummy point to make axes work
plt.fill(52 + circ_x, 0 + circ_y, color=cmap(0.3),alpha = 0.)
plt.fill(0 + circ_x, 0 + circ_y, color=cmap(0.3),alpha = 0.)

plt.axis('equal')

ylim    = plt.gca().get_ylim()
plt.gca().set_ylim([ylim[0]-0.3,ylim[1]-0.3])

plt.axis("off")


#%%

plt.subplot(gs[2,0])


xpos    = np.column_stack((
    np.cos(np.linspace(2*np.pi,0,Nnodes+1)[:-1]+np.pi/2),
    np.sin(np.linspace(2*np.pi,0,Nnodes+1)[:-1]+np.pi/2) )) * 2


ypos    = np.column_stack((
    np.cos(np.linspace(2*np.pi,0,Nnodes+1)[:-1]+np.pi/2),
    np.sin(np.linspace(2*np.pi,0,Nnodes+1)[:-1]+np.pi/2) )) 

n = 0

square_x    = np.asarray([+0.5,+0.5,-0.5,-0.5])*0.8
square_y    = np.asarray([+0.5,-0.5,-0.5,+0.5])*0.8

filter_x    = []
filter_y    = []

counter     = -1
maxcounter  = (T-1)*2

cmap = matplotlib.cm.get_cmap('turbo')

# Go through all time steps
for t in np.arange(1,T,1):
    
    counter += 1
    
    if t != T-1:
        plt.fill(t + circ_x, t + circ_y, color=cmap(counter/maxcounter), edgecolor="None")
    
    if t > 1:
    
        counter += 1
        
        trapezoid   = np.column_stack((
            np.asarray(list(triang_x[:2] + 1) + list(triang_x[1:] + t - 1)),
            np.asarray(list(triang_y[:2] + t) + list(triang_y[1:] + t))))
        
        plt.fill(trapezoid[:,0], trapezoid[:,1], color=cmap(counter/maxcounter), edgecolor="None")

plt.title(r'$\bf{A}$: dense smoother', loc='left', fontsize=12)
plt.axis('equal')

plt.ylabel('conditioned on data', labelpad=-15,fontsize = 12)
ax = plt.gca()
ax.set_xticks([1,T-2])
ax.set_xticklabels(['',''])
ax.set_yticks([2,T-1])
ax.set_yticklabels(['$\mathbf{y}_{1}^{*}$','$\mathbf{y}_{1:t}^{*}$'],fontsize = 12)


#%%

plt.subplot(gs[2,1])


xpos    = np.column_stack((
    np.cos(np.linspace(2*np.pi,0,Nnodes+1)[:-1]+np.pi/2),
    np.sin(np.linspace(2*np.pi,0,Nnodes+1)[:-1]+np.pi/2) )) * 2


ypos    = np.column_stack((
    np.cos(np.linspace(2*np.pi,0,Nnodes+1)[:-1]+np.pi/2),
    np.sin(np.linspace(2*np.pi,0,Nnodes+1)[:-1]+np.pi/2) )) 

n = 0

square_x    = np.asarray([+0.5,+0.5,-0.5,-0.5])*0.8
square_y    = np.asarray([+0.5,-0.5,-0.5,+0.5])*0.8

filter_x    = []
filter_y    = []


counter     = -1
maxcounter  = (T-1)*2

cmap = matplotlib.cm.get_cmap('turbo')

# Go through all time steps
for t in np.arange(1,T,1):
    
    counter += 1
    
    if t != T-1:
        plt.fill(t + circ_x, t + circ_y, color=cmap(counter/maxcounter), edgecolor="None")
    
    if t > 1:
    
        counter += 1
        
        trapezoid   = np.column_stack((
            np.asarray(list(triang_x[:2] + 1) + list(triang_x[1:] + t - 1)),
            np.asarray(list(triang_y[:2] + t) + list(triang_y[1:] + t))))
        
        plt.fill(trapezoid[:,0], trapezoid[:,1], color=cmap(counter/maxcounter), alpha = 0.1, edgecolor="None")
    
    
        if t - 5 <= 1:
            
            xs    = triang_x[:2]
            
        else:
            
            xs    = np.ones(2)*(-0.5)*0.8
    
        trapezoid   = np.column_stack((
            np.asarray(list(xs + np.maximum(1, t - 5)) + list(triang_x[1:] + t - 1)),
            np.asarray(list(triang_y[:2] + t) + list(triang_y[1:] + t))))
        
        plt.fill(trapezoid[:,0], trapezoid[:,1], color=cmap(counter/maxcounter), edgecolor="None")

plt.title(r'$\bf{B}$: dense smoother (fixed-lag)', loc='left', fontsize=12)
plt.axis('equal')

ax = plt.gca()
ax.set_xticks([1,T-2])
ax.set_xticklabels(['',''])
ax.set_yticks([2,T-1])
ax.set_yticklabels(['',''])



#%%

plt.subplot(gs[2,2])

xpos    = np.column_stack((
    np.cos(np.linspace(2*np.pi,0,Nnodes+1)[:-1]+np.pi/2),
    np.sin(np.linspace(2*np.pi,0,Nnodes+1)[:-1]+np.pi/2) )) * 2


ypos    = np.column_stack((
    np.cos(np.linspace(2*np.pi,0,Nnodes+1)[:-1]+np.pi/2),
    np.sin(np.linspace(2*np.pi,0,Nnodes+1)[:-1]+np.pi/2) )) 

n = 0

square_x    = np.asarray([+0.5,+0.5,-0.5,-0.5])*0.8
square_y    = np.asarray([+0.5,-0.5,-0.5,+0.5])*0.8

filter_x    = []
filter_y    = []

counter     = -1
maxcounter  = (T-1)*2

cmap = matplotlib.cm.get_cmap('turbo')

# Go through all time steps
for t in np.arange(1,T,1):
    
    counter += 1
    
    plt.fill(t + circ_x, t + circ_y, color=cmap(counter/maxcounter), edgecolor="None")
    
    counter += 1
    
    plt.fill(t + triang_x, t + 1 + triang_y, color=cmap(counter/maxcounter), edgecolor="None")
    
    if t > 1:
    
        trapezoid   = np.column_stack((
            np.asarray(list(triang_x[:2] + 1) + list(triang_x[1:] + t)),
            np.asarray(list(triang_y[:2] + t + 1) + list(triang_y[1:] + t + 1))))
        
        plt.fill(trapezoid[:,0], trapezoid[:,1], color=cmap(counter/maxcounter), alpha = 0.1, edgecolor="None")
    
    for s in [5.,10.,13.]:
    
        if t == s+1:   
            
            xs    = -np.ones(2)/2*0.8 
        
            trapezoid   = np.column_stack((
                np.asarray(list(xs + t - 1) + list(triang_x[1:] + t)),
                np.asarray(list(triang_y[:2] + t + 1) + list(triang_y[1:] + t + 1))))
            
            plt.fill(trapezoid[:,0], trapezoid[:,1], color=cmap(counter/maxcounter), edgecolor="None")
            
            
    
        elif t > s:
            
            plt.fill(s + square_x, t + 1 + square_y, color=cmap(counter/maxcounter), edgecolor="None")
    
plt.title(r'$\bf{C}$: fixed-point smoother', loc='left', fontsize=12)
plt.axis('equal')


ax = plt.gca()
ax.set_xticks([1,T-1])
ax.set_xticklabels(['',''])
ax.set_yticks([2,T])
ax.set_yticklabels(['',''])

#%%

plt.subplot(gs[3,0])

counter     = -1
maxcounter  = 208

cmap = matplotlib.cm.get_cmap('turbo')

# Go through all time steps
for t in np.arange(1,T,1):

    counter += 1
    
    plt.fill(t + circ_x, t + circ_y, color=cmap(counter/maxcounter), edgecolor="None")
    
    for s in np.arange(1,t,1):
        
        counter += 1
        
        plt.fill(s + square_x, t + 1 + square_y, color=cmap(counter/maxcounter), edgecolor="None")
        
    counter += 1
    
    plt.fill(t + triang_x, t + 1 + triang_y, color=cmap(counter/maxcounter), edgecolor="None")
    

plt.title(r'$\bf{D}$: forward smoother', loc='left', fontsize=12)
plt.axis('equal')

plt.xlabel('state block', labelpad=-10,fontsize = 12)
plt.ylabel('conditioned on data', labelpad=-15,fontsize = 12)
ax = plt.gca()
ax.set_xticks([1,T-1])
ax.set_xticklabels(['$\mathbf{x}_{1}$','$\mathbf{x}_{t}$'],fontsize = 12)
ax.set_yticks([2,T])
ax.set_yticklabels(['$\mathbf{y}_{1}^{*}$','$\mathbf{y}_{1:t}^{*}$'],fontsize = 12)

# raise Exception


#%%


plt.subplot(gs[3,1])


xpos    = np.column_stack((
    np.cos(np.linspace(2*np.pi,0,Nnodes+1)[:-1]+np.pi/2),
    np.sin(np.linspace(2*np.pi,0,Nnodes+1)[:-1]+np.pi/2) )) * 2


ypos    = np.column_stack((
    np.cos(np.linspace(2*np.pi,0,Nnodes+1)[:-1]+np.pi/2),
    np.sin(np.linspace(2*np.pi,0,Nnodes+1)[:-1]+np.pi/2) )) 

n = 0

square_x    = np.asarray([+0.5,+0.5,-0.5,-0.5])*0.8
square_y    = np.asarray([+0.5,-0.5,-0.5,+0.5])*0.8

filter_x    = []
filter_y    = []

counter     = -1
maxcounter  = (T-1)*2+(T-1)

cmap = matplotlib.cm.get_cmap('turbo')

# Go through all time steps
for t in np.arange(1,T,1):
    
    counter += 1
    
    plt.fill(t + circ_x, t + circ_y, color=cmap(counter/maxcounter), edgecolor="None")
    
    counter += 1
    
    plt.fill(t + triang_x, t + 1 + triang_y, color=cmap(counter/maxcounter), edgecolor="None")
    
# Go through all time steps
for s in np.arange(t-1,0,-1):
    
    counter += 1
    
    plt.fill(s + square_x, t + 1 + square_y, color=cmap(counter/maxcounter), edgecolor="None")

plt.title(r'$\bf{E}$: backward smoother (single-pass)', loc='left', fontsize=12)
plt.axis('equal')

plt.xlabel('state block', labelpad=-10,fontsize = 12)
ax = plt.gca()
ax.set_xticks([1,T-1])
ax.set_xticklabels(['$\mathbf{x}_{1}$','$\mathbf{x}_{t}$'],fontsize = 12)
ax.set_yticks([2,T])
ax.set_yticklabels(['',''])



#%%

plt.subplot(gs[3,2])


xpos    = np.column_stack((
    np.cos(np.linspace(2*np.pi,0,Nnodes+1)[:-1]+np.pi/2),
    np.sin(np.linspace(2*np.pi,0,Nnodes+1)[:-1]+np.pi/2) )) * 2


ypos    = np.column_stack((
    np.cos(np.linspace(2*np.pi,0,Nnodes+1)[:-1]+np.pi/2),
    np.sin(np.linspace(2*np.pi,0,Nnodes+1)[:-1]+np.pi/2) )) 

n = 0

square_x    = np.asarray([+0.5,+0.5,-0.5,-0.5])*0.8
square_y    = np.asarray([+0.5,-0.5,-0.5,+0.5])*0.8

filter_x    = []
filter_y    = []

counter     = -1
maxcounter  = 79

cmap = matplotlib.cm.get_cmap('turbo')

# Go through all time steps
for t in np.arange(1,T,1):
    
    # s = t
    
    counter += 1
    
    plt.fill(t + circ_x, t + circ_y, color=cmap(counter/maxcounter), edgecolor="None")
    
    counter += 1
    
    plt.fill(t + triang_x, t + 1 + triang_y, color=cmap(counter/maxcounter), edgecolor="None")
    
    
    if (t+1)%5 == 0:
        
        for s in np.arange(t-1,0,-1):
            
            counter += 1
            
            plt.fill(s + square_x, t + 1 + square_y, color=cmap(counter/maxcounter), edgecolor="None")
    

        
plt.title(r'$\bf{F}$: backward smoother (multi-pass)', loc='left', fontsize=12)
# plt.title('backward smoother (multi-pass)')
plt.axis('equal')

plt.xlabel('state block', labelpad=-10,fontsize = 12)
# plt.ylabel('time scubscript $s$', labelpad=-10)
ax = plt.gca()
ax.set_xticks([1,T-1])
ax.set_xticklabels(['$\mathbf{x}_{1}$','$\mathbf{x}_{t}$'],fontsize = 12)
ax.set_yticks([2,T])
ax.set_yticklabels(['',''])


plt.savefig('smoother_types.png',dpi=600,bbox_inches='tight')
plt.savefig('smoother_types.pdf',dpi=600,bbox_inches='tight')












