import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import turbo_colormap
import colorsys

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", 
    ["xkcd:midnight blue",
     "xkcd:sky blue",
     "xkcd:light sky blue"])

def whiten(color):
    
    color   = np.asarray(color)
    
    color   = color + (np.ones(len(color))-color)*fac
    
    return color

color1  = 'xkcd:cerulean'
color2  = '#62BEED'

cmap = matplotlib.cm.get_cmap('turbo')

plt.close('all')

Nnodes  = 5

plt.figure(figsize=(12,7))
gs  = matplotlib.gridspec.GridSpec(nrows=3,ncols=3,width_ratios=[1,1,1],height_ratios=[0.1,1,1])

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

cb1.set_label("algorithm progress", labelpad=-3)

plt.gca().xaxis.set_ticks_position('top')
plt.gca().xaxis.set_label_position('top')
cb1.ax.set_xticklabels(['start', 'end'])  # horizontal colorbar

plt.gca().annotate('', xy=(0.1, 1.75), xycoords='axes fraction', xytext=(0.4, 1.75), 
            arrowprops=dict(arrowstyle = '-',color='xkcd:dark grey'))
plt.gca().annotate('', xy=(0.9, 1.75), xycoords='axes fraction', xytext=(0.6, 1.75), 
            arrowprops=dict(arrowstyle = '->',color='xkcd:dark grey'))

figidx = 0
    
if figidx == 0:
    fac = 0.
else:
    fac = 0.75

plt.subplot(gs[figidx+1,:])

labels  = [
    '_{'+str(1)+'}',
    '_{'+str(2)+'}',
    '\dots$',
    '_{t-1}',
    '_{t}']

Nnodes  = 5

xscale  = 6

xpos    = np.column_stack((
    np.linspace(0,xscale,Nnodes),
    np.ones(Nnodes)))


ypos    = np.column_stack((
    np.linspace(0,xscale,Nnodes),
    np.zeros(Nnodes)+0.25))


skipnode = 2

offsets     = np.asarray([
    [np.cos(-np.pi/2),  np.sin(-np.pi/2)],
    [np.cos(np.pi/4),  np.sin(np.pi/4)-0.1],
    [np.cos(3*np.pi/4),  np.sin(3*np.pi/4)-0.5]])*0.3

offsets     = np.asarray([
    [-1,1],
    [0,0],
    [1,-1]])*0.3


colorcounter = -1
colorcounter_max = 8

pos     = np.row_stack((
    xpos,ypos))

if figidx == 0:

    plt.gca().text(
        2.9, 
        1, 
        'states $\mathbf{x}$', 
        ha          = "center", 
        va          = "center",
        zorder      = 10,
        color       = [0.1,0.1,0.1],
        fontsize    = 10,
        rotation    = 315)
    
    plt.gca().text(
        2.9, 
        +0.2, 
        'predictions $\mathbf{y}$', 
        ha          = "center", 
        va          = "center",
        zorder      = 10,
        color       = [0.1,0.1,0.1],
        fontsize    = 10,
        rotation    = 315)
    
    
    
    plt.gca().text(
        4, 
        +0.2, 
        'dimensions', 
        ha          = "center", 
        va          = "center",
        zorder      = 10,
        color       = 'xkcd:grey',
        fontsize    = 10,
        rotation    = 315)

    plt.plot(
        np.asarray([xpos[0,0]+offsets[0,0]+0.15,xpos[0,0]+offsets[0,0], xpos[0,0]+offsets[2,0],xpos[0,0]+offsets[2,0]+0.15])+4.15,
        np.asarray([xpos[0,1]+offsets[0,1],xpos[0,1]+offsets[0,1], xpos[0,1]+offsets[2,1], xpos[0,1]+offsets[2,1]])-0.75,
        color   = 'xkcd:grey',
        zorder  = -5)
    plt.plot(
        np.asarray([xpos[0,0]+offsets[1,0]+0.15,xpos[0,0]+offsets[1,0]])+4.15,
        np.asarray([xpos[0,1]+offsets[1,1], xpos[0,1]+offsets[1,1]])-0.75,
        color   = 'xkcd:grey',
        zorder  = -5)

for n in range(Nnodes):
    
    if n != skipnode:
        
        colorcounter += 1
    
    
        textcolor = colorsys.rgb_to_hsv(
            cmap(colorcounter/colorcounter_max)[0], 
            cmap(colorcounter/colorcounter_max)[1], 
            cmap(colorcounter/colorcounter_max)[2])
        
        textcolor = colorsys.hsv_to_rgb(
            0, 
            0, 
            1 - textcolor[2])
        
        
        if figidx == 1 and n == 4:
            
            # textcolor = tuple(list(textcolor)+[0.5])
            plt.gca().add_patch(plt.Circle(xpos[n,:]+offsets[0,:], 0.15, color=cmap(colorcounter/colorcounter_max)))
            if n not in [0,Nnodes-1]:
                plt.gca().text(xpos[n,0]+offsets[0,0], xpos[n,1]+offsets[0,1], '$a$', ha="center", va="center",zorder=10,color=textcolor,fontsize=10)
            else:
                plt.gca().text(xpos[n,0]+offsets[0,0], xpos[n,1]+offsets[0,1], '$a$', ha="center", va="center",zorder=10,color=[0.95,0.95,0.95],fontsize=10)
            
            plt.gca().add_patch(plt.Circle(xpos[n,:]+offsets[1,:], 0.15, color=cmap(colorcounter/colorcounter_max)))
            if n not in [0,Nnodes-1]:
                plt.gca().text(xpos[n,0]+offsets[1,0], xpos[n,1]+offsets[1,1], '$b$', ha="center", va="center",zorder=10,color=textcolor,fontsize=10)
            else:
                plt.gca().text(xpos[n,0]+offsets[1,0], xpos[n,1]+offsets[1,1], '$b$', ha="center", va="center",zorder=10,color=[0.95,0.95,0.95],fontsize=10)
                
            plt.gca().add_patch(plt.Circle(xpos[n,:]+offsets[2,:], 0.15, color=cmap(colorcounter/colorcounter_max)))
            if n not in [0,Nnodes-1]:
                plt.gca().text(xpos[n,0]+offsets[2,0], xpos[n,1]+offsets[2,1], '$c$', ha="center", va="center",zorder=10,color=textcolor,fontsize=10)
            else:
                plt.gca().text(xpos[n,0]+offsets[2,0], xpos[n,1]+offsets[2,1], '$c$', ha="center", va="center",zorder=10,color=[0.95,0.95,0.95],fontsize=10)
        
        else:
        
            # textcolor = tuple(list(textcolor)+[0.5])
            plt.gca().add_patch(plt.Circle(xpos[n,:]+offsets[0,:], 0.15, color=whiten(cmap(colorcounter/colorcounter_max))))
            if n not in [0,Nnodes-1]:
                plt.gca().text(xpos[n,0]+offsets[0,0], xpos[n,1]+offsets[0,1], '$a$', ha="center", va="center",zorder=10,color=whiten(textcolor),fontsize=10)
            else:
                plt.gca().text(xpos[n,0]+offsets[0,0], xpos[n,1]+offsets[0,1], '$a$', ha="center", va="center",zorder=10,color=whiten([0.95,0.95,0.95]),fontsize=10)
            
            plt.gca().add_patch(plt.Circle(xpos[n,:]+offsets[1,:], 0.15, color=whiten(cmap(colorcounter/colorcounter_max))))
            if n not in [0,Nnodes-1]:
                plt.gca().text(xpos[n,0]+offsets[1,0], xpos[n,1]+offsets[1,1], '$b$', ha="center", va="center",zorder=10,color=whiten(textcolor),fontsize=10)
            else:
                plt.gca().text(xpos[n,0]+offsets[1,0], xpos[n,1]+offsets[1,1], '$b$', ha="center", va="center",zorder=10,color=whiten([0.95,0.95,0.95]),fontsize=10)
                
            plt.gca().add_patch(plt.Circle(xpos[n,:]+offsets[2,:], 0.15, color=whiten(cmap(colorcounter/colorcounter_max))))
            if n not in [0,Nnodes-1]:
                plt.gca().text(xpos[n,0]+offsets[2,0], xpos[n,1]+offsets[2,1], '$c$', ha="center", va="center",zorder=10,color=whiten(textcolor),fontsize=10)
            else:
                plt.gca().text(xpos[n,0]+offsets[2,0], xpos[n,1]+offsets[2,1], '$c$', ha="center", va="center",zorder=10,color=whiten([0.95,0.95,0.95]),fontsize=10)

        plt.gca().text(
            xpos[n,0]+offsets[1,0], 
            xpos[n,1]+offsets[1,1]+0.65, 
            'timestep $'+str(labels[n][2:-1])+'$', 
            ha          = "center", 
            va          = "center",
            zorder      = 10,
            color       = [0.1,0.1,0.1],
            fontsize    = 10)

        line = np.column_stack((
            np.linspace(
                list(xpos[n,0]+offsets[:,0])[0],
                list(xpos[n,0]+offsets[:,0])[-1],
                101),
            np.linspace(
                list(xpos[n,1]+offsets[:,1])[0],
                list(xpos[n,1]+offsets[:,1])[-1],
                101) - np.sin(np.linspace(-np.pi,0,101))*0.35))
        
        from scipy.interpolate import interp1d
        
        itp     = interp1d(
            np.linspace(0,1,101),
            np.linspace(
                list(xpos[n,0]+offsets[:,0])[0],
                list(xpos[n,0]+offsets[:,0])[-1],
                101))
        
        import scipy.stats
        
        positions   = np.column_stack((
            xpos[n,0]+offsets[:,0],
            xpos[n,1]+offsets[:,1]))
        
        from scipy.interpolate import CubicSpline
        
        if n != Nnodes-1 and n != 1:
            
            tinyoffset = [0.0,-0.0,0.]
            
            tinieroffset = [0.02,-0.02,0.]
        
            for i in range(3):
                
                for j in range(3):

                    normratio   = np.linalg.norm(positions[j,:]-positions[i,:] + np.asarray([1.5,0]))
                    normratio   = (normratio-0.22)/normratio
                
                    plt.arrow(
                        positions[i,0],
                        positions[i,1],
                        (np.asarray(positions[j,0]-positions[i,0])+1.5)*normratio,
                        (np.asarray(positions[j,1]-positions[i,1]))*normratio,
                        head_width = 0.05,
                        ec=whiten(cmap((colorcounter+2)/colorcounter_max)),
                        fc=whiten(cmap((colorcounter+2)/colorcounter_max)),
                        width=0.01,
                        zorder=-0.5)
                
        elif n != Nnodes-1:
            
            for i in range(3):
                
                for j in range(3):
                    
                    normratio   = np.linalg.norm(positions[j,:]-positions[i,:] + np.asarray([1.5,0]))
                    normratio   = (normratio-0.22)/normratio
                
                    plt.plot(
                        [positions[i,0],positions[i,0]+(np.asarray(positions[j,0]-positions[i,0])+1.5)*normratio*0.5],
                        [positions[i,1],positions[i,1]+(np.asarray(positions[j,1]-positions[i,1]))*normratio*0.5],
                        lw = 2,
                        color=whiten(cmap((colorcounter+2)/colorcounter_max)),
                        zorder=-0.5)
                    
                    
                    normratio   = np.linalg.norm(positions[j,:]-positions[i,:] + np.asarray([1.5,0]))
                    normratio   = (normratio-0.22)/normratio
                
                    plt.arrow(
                        positions[i,0]+(np.asarray(positions[j,0]-positions[i,0])+1.5)*normratio*0.5+1.5,
                        positions[i,1]+(np.asarray(positions[j,1]-positions[i,1]))*normratio*0.5,
                        (np.asarray(positions[j,0]-positions[i,0])+1.5)*normratio*0.5,
                        (np.asarray(positions[j,1]-positions[i,1]))*normratio*0.5,
                        head_width = 0.05,
                        ec=whiten(cmap((colorcounter+3)/colorcounter_max)),
                        fc=whiten(cmap((colorcounter+3)/colorcounter_max)),
                        width=0.01,
                        zorder=-0.5)
            
            
        textcolor = colorsys.rgb_to_hsv(
            cmap(colorcounter/colorcounter_max)[0], 
            cmap(colorcounter/colorcounter_max)[1], 
            cmap(colorcounter/colorcounter_max)[2])
        
        textcolor = colorsys.hsv_to_rgb(
            0, 
            0, 
            1 - textcolor[2])
        
        
        colorcounter += 1

        if figidx == 1 and n == 4:

            # textcolor = tuple(list(textcolor)+[0.5])
            plt.gca().add_patch(plt.Circle(ypos[n,:]+offsets[0,:], 0.15, color=whiten(cmap(colorcounter/colorcounter_max)),zorder=-3))
            if n not in [0,Nnodes-1]:
                plt.gca().text(ypos[n,0]+offsets[0,0], ypos[n,1]+offsets[0,1], '$a$', ha="center", va="center",zorder=10,color=textcolor,fontsize=10)
            else:
                plt.gca().text(ypos[n,0]+offsets[0,0], ypos[n,1]+offsets[0,1], '$a$', ha="center", va="center",zorder=10,color=[0.95,0.95,0.95],fontsize=10)
            
            plt.gca().add_patch(plt.Circle(ypos[n,:]+offsets[1,:], 0.15, color=cmap(colorcounter/colorcounter_max),zorder=-3))
            if n not in [0,Nnodes-1]:
                plt.gca().text(ypos[n,0]+offsets[1,0], ypos[n,1]+offsets[1,1], '$b$', ha="center", va="center",zorder=10,color=whiten(textcolor),fontsize=10)
            else:
                plt.gca().text(ypos[n,0]+offsets[1,0], ypos[n,1]+offsets[1,1], '$b$', ha="center", va="center",zorder=10,color=whiten([0.95,0.95,0.95]),fontsize=10)
                
            plt.gca().add_patch(plt.Circle(ypos[n,:]+offsets[2,:], 0.15, color=whiten(cmap(colorcounter/colorcounter_max)),zorder=-3))
            if n not in [0,Nnodes-1]:
                plt.gca().text(ypos[n,0]+offsets[2,0], ypos[n,1]+offsets[2,1], '$c$', ha="center", va="center",zorder=10,color=whiten(textcolor),fontsize=10)
            else:
                plt.gca().text(ypos[n,0]+offsets[2,0], ypos[n,1]+offsets[2,1], '$c$', ha="center", va="center",zorder=10,color=whiten([0.95,0.95,0.95]),fontsize=10)
        
        else:
            
            # textcolor = tuple(list(textcolor)+[0.5])
            plt.gca().add_patch(plt.Circle(ypos[n,:]+offsets[0,:], 0.15, color=whiten(cmap(colorcounter/colorcounter_max)),zorder=-3))
            if n not in [0,Nnodes-1]:
                plt.gca().text(ypos[n,0]+offsets[0,0], ypos[n,1]+offsets[0,1], '$a$', ha="center", va="center",zorder=10,color=whiten(textcolor),fontsize=10)
            else:
                plt.gca().text(ypos[n,0]+offsets[0,0], ypos[n,1]+offsets[0,1], '$a$', ha="center", va="center",zorder=10,color=whiten([0.95,0.95,0.95]),fontsize=10)
            
            plt.gca().add_patch(plt.Circle(ypos[n,:]+offsets[1,:], 0.15, color=whiten(cmap(colorcounter/colorcounter_max)),zorder=-3))
            if n not in [0,Nnodes-1]:
                plt.gca().text(ypos[n,0]+offsets[1,0], ypos[n,1]+offsets[1,1], '$b$', ha="center", va="center",zorder=10,color=whiten(textcolor),fontsize=10)
            else:
                plt.gca().text(ypos[n,0]+offsets[1,0], ypos[n,1]+offsets[1,1], '$b$', ha="center", va="center",zorder=10,color=whiten([0.95,0.95,0.95]),fontsize=10)
                
            plt.gca().add_patch(plt.Circle(ypos[n,:]+offsets[2,:], 0.15, color=whiten(cmap(colorcounter/colorcounter_max)),zorder=-3))
            if n not in [0,Nnodes-1]:
                plt.gca().text(ypos[n,0]+offsets[2,0], ypos[n,1]+offsets[2,1], '$c$', ha="center", va="center",zorder=10,color=whiten(textcolor),fontsize=10)
            else:
                plt.gca().text(ypos[n,0]+offsets[2,0], ypos[n,1]+offsets[2,1], '$c$', ha="center", va="center",zorder=10,color=whiten([0.95,0.95,0.95]),fontsize=10)
            
        plt.arrow(
            pos[n,0]+offsets[0,0],
            pos[n,1]-1+0.7+offsets[0,1]+0.3,
            0,
            -1.03*0.5,
            head_width = 0.05,
            ec=whiten(cmap(colorcounter/colorcounter_max)),
            fc=whiten(cmap(colorcounter/colorcounter_max)),
            width=0.01,
            zorder=-1)
        
        if figidx == 1 and n == 4:
        
            plt.arrow(
                pos[n,0]+offsets[1,0],
                pos[n,1]-1+0.7+offsets[1,1]+0.3,
                0,
                -1.03*0.5,
                head_width = 0.05,
                ec=cmap(colorcounter/colorcounter_max),
                fc=cmap(colorcounter/colorcounter_max),
                width=0.01,
                zorder=-1)
        
        else:
            
            plt.arrow(
                pos[n,0]+offsets[1,0],
                pos[n,1]-1+0.7+offsets[1,1]+0.3,
                0,
                -1.03*0.5,
                head_width = 0.05,
                ec=whiten(cmap(colorcounter/colorcounter_max)),
                fc=whiten(cmap(colorcounter/colorcounter_max)),
                width=0.01,
                zorder=-1)
        
        plt.arrow(
            pos[n,0]+offsets[2,0],
            pos[n,1]-1+0.7+offsets[2,1]+0.3,
            0,
            -1.03*0.5,
            head_width = 0.05,
            ec=whiten(cmap(colorcounter/colorcounter_max)),
            fc=whiten(cmap(colorcounter/colorcounter_max)),
            width=0.01,
            zorder=-1)
        
    else:
        
        colorcounter += 1
        
        plt.gca().add_patch(plt.Circle(xpos[n,:]+np.asarray([-0.45,0]), 0.03, color=whiten(cmap(colorcounter/colorcounter_max))))
        
        plt.gca().add_patch(plt.Circle(xpos[n,:]+np.asarray([-0.45,0])+np.asarray([-0.1,0]), 0.03, color=whiten(cmap(colorcounter/colorcounter_max))))
        
        plt.gca().add_patch(plt.Circle(xpos[n,:]+np.asarray([-0.45,0])+np.asarray([+0.1,0]), 0.03, color=whiten(cmap(colorcounter/colorcounter_max))))

            
        plt.gca().add_patch(plt.Circle(xpos[n,:]+np.asarray([+0.25,0]), 0.03, color=whiten(cmap(colorcounter/colorcounter_max))))
        
        plt.gca().add_patch(plt.Circle(xpos[n,:]+np.asarray([+0.25,0])+np.asarray([-0.1,0]), 0.03, color=whiten(cmap(colorcounter/colorcounter_max))))
        
        plt.gca().add_patch(plt.Circle(xpos[n,:]+np.asarray([+0.25,0])+np.asarray([+0.1,0]), 0.03, color=whiten(cmap(colorcounter/colorcounter_max))))


    
plt.gca().relim()
plt.gca().autoscale_view()
plt.axis('equal')

plt.tick_params(left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False)

if figidx == 0:
    plt.ylabel('$\mathbf{A}$: Lorenz-63 graph',loc="bottom")
else:
    plt.ylabel('$\mathbf{B}$: analysis graph for $y_{t}^{b}$',loc="bottom")
    
    
    
    
#%%

figidx = 1

if figidx == 0:
    fac = 0.
else:
    fac = 0.75


plt.subplot(gs[figidx+1,0])




labels  = [
    '_{'+str(1)+'}',
    '_{'+str(2)+'}',
    '\dots$',
    '_{t-1}',
    '_{t}']

Nnodes  = 5

xscale  = 6

xpos    = np.column_stack((
    np.linspace(0,xscale,Nnodes),
    np.ones(Nnodes)))


ypos    = np.column_stack((
    np.linspace(0,xscale,Nnodes),
    np.zeros(Nnodes)+0.25))


skipnode = 2

offsets     = np.asarray([
    [np.cos(-np.pi/2),  np.sin(-np.pi/2)],
    [np.cos(np.pi/4),  np.sin(np.pi/4)-0.1],
    [np.cos(3*np.pi/4),  np.sin(3*np.pi/4)-0.5]])*0.3

offsets     = np.asarray([
    [-1,1],
    [0,0],
    [1,-1]])*0.3


colorcounter = -1
colorcounter_max = 8

pos     = np.row_stack((
    xpos,ypos))


for n in range(Nnodes):
    
    if n != skipnode:
        
        colorcounter += 1
    
    
        textcolor = colorsys.rgb_to_hsv(
            cmap(colorcounter/colorcounter_max)[0], 
            cmap(colorcounter/colorcounter_max)[1], 
            cmap(colorcounter/colorcounter_max)[2])
        
        textcolor = colorsys.hsv_to_rgb(
            0, 
            0, 
            1 - textcolor[2])
        
        
        if figidx == 1 and n == 4:
            
            # textcolor = tuple(list(textcolor)+[0.5])
            plt.gca().add_patch(plt.Circle(xpos[n,:]+offsets[0,:], 0.15, color=cmap(colorcounter/colorcounter_max)))
            if n not in [0,Nnodes-1]:
                plt.gca().text(xpos[n,0]+offsets[0,0], xpos[n,1]+offsets[0,1], '$a$', ha="center", va="center",zorder=10,color=textcolor,fontsize=10)
            else:
                plt.gca().text(xpos[n,0]+offsets[0,0], xpos[n,1]+offsets[0,1], '$a$', ha="center", va="center",zorder=10,color=[0.95,0.95,0.95],fontsize=10)
            
            plt.gca().add_patch(plt.Circle(xpos[n,:]+offsets[1,:], 0.15, color=cmap(colorcounter/colorcounter_max)))
            if n not in [0,Nnodes-1]:
                plt.gca().text(xpos[n,0]+offsets[1,0], xpos[n,1]+offsets[1,1], '$b$', ha="center", va="center",zorder=10,color=textcolor,fontsize=10)
            else:
                plt.gca().text(xpos[n,0]+offsets[1,0], xpos[n,1]+offsets[1,1], '$b$', ha="center", va="center",zorder=10,color=[0.95,0.95,0.95],fontsize=10)
                
            plt.gca().add_patch(plt.Circle(xpos[n,:]+offsets[2,:], 0.15, color=cmap(colorcounter/colorcounter_max)))
            if n not in [0,Nnodes-1]:
                plt.gca().text(xpos[n,0]+offsets[2,0], xpos[n,1]+offsets[2,1], '$c$', ha="center", va="center",zorder=10,color=textcolor,fontsize=10)
            else:
                plt.gca().text(xpos[n,0]+offsets[2,0], xpos[n,1]+offsets[2,1], '$c$', ha="center", va="center",zorder=10,color=[0.95,0.95,0.95],fontsize=10)
        
        else:
        
            plt.gca().add_patch(plt.Circle(xpos[n,:]+offsets[0,:], 0.15, color=whiten(cmap(colorcounter/colorcounter_max))))

            plt.gca().add_patch(plt.Circle(xpos[n,:]+offsets[1,:], 0.15, color=whiten(cmap(colorcounter/colorcounter_max))))

            plt.gca().add_patch(plt.Circle(xpos[n,:]+offsets[2,:], 0.15, color=whiten(cmap(colorcounter/colorcounter_max))))
            if n == 3:
                if n not in [0,Nnodes-1]:
                    plt.gca().text(xpos[n,0]+offsets[2,0], xpos[n,1]+offsets[2,1], '$c$', ha="center", va="center",zorder=10,color=whiten(textcolor),fontsize=10)
                else:
                    plt.gca().text(xpos[n,0]+offsets[2,0], xpos[n,1]+offsets[2,1], '$c$', ha="center", va="center",zorder=10,color=whiten([0.95,0.95,0.95]),fontsize=10)
        

        
        line = np.column_stack((
            np.linspace(
                list(xpos[n,0]+offsets[:,0])[0],
                list(xpos[n,0]+offsets[:,0])[-1],
                101),
            np.linspace(
                list(xpos[n,1]+offsets[:,1])[0],
                list(xpos[n,1]+offsets[:,1])[-1],
                101) - np.sin(np.linspace(-np.pi,0,101))*0.35))
        
        from scipy.interpolate import interp1d
        
        itp     = interp1d(
            np.linspace(0,1,101),
            np.linspace(
                list(xpos[n,0]+offsets[:,0])[0],
                list(xpos[n,0]+offsets[:,0])[-1],
                101))
        
        import scipy.stats
        
        if n == 4:
            
            plt.plot(
                list(xpos[n,0]+offsets[:2,0])+[list(xpos[n,0]+offsets[:2,0])[0]],
                list(xpos[n,1]+offsets[:2,1])+[list(xpos[n,1]+offsets[:2,1])[0]],
                color   = cmap(colorcounter/colorcounter_max),
                alpha   = 1,
                lw      = 2,
                zorder  = -1,
                ls      = '--')
            
            plt.plot(
                list(xpos[n,0]+offsets[1:,0])+[list(xpos[n,0]+offsets[1:,0])[0]],
                list(xpos[n,1]+offsets[1:,1])+[list(xpos[n,1]+offsets[1:,1])[0]],
                color   = cmap(colorcounter/colorcounter_max),
                alpha   = 1,
                lw      = 2,
                zorder  = -1,
                ls      = '--')
            
                
            plt.plot(
                itp(scipy.stats.beta.cdf(np.linspace(0,1,101),a=1.5,b=1.5)),
                line[:,1]+0.135,
                color   = cmap(colorcounter/colorcounter_max),
                lw      = 2,
                zorder  = -1,
                ls      = '--')
        
        positions   = np.column_stack((
            xpos[n,0]+offsets[:,0],
            xpos[n,1]+offsets[:,1]))

        from scipy.interpolate import CubicSpline
        
        if n != Nnodes-1 and n != 1:
            
            tinyoffset = [0.0,-0.0,0.]
            
            tinieroffset = [0.02,-0.02,0.]
        
            for i in range(3):
                
                for j in range(3):

                    normratio   = np.linalg.norm(positions[j,:]-positions[i,:] + np.asarray([1.5,0]))
                    normratio   = (normratio-0.22)/normratio
                
                    plt.arrow(
                        positions[i,0],
                        positions[i,1],
                        (np.asarray(positions[j,0]-positions[i,0])+1.5)*normratio,
                        (np.asarray(positions[j,1]-positions[i,1]))*normratio,
                        head_width = 0.05,
                        ec=whiten(cmap((colorcounter+2)/colorcounter_max)),
                        fc=whiten(cmap((colorcounter+2)/colorcounter_max)),
                        width=0.01,
                        zorder=-0.5)
                
        elif n != Nnodes-1:
            
            for i in range(3):
                
                for j in range(3):

                    normratio   = np.linalg.norm(positions[j,:]-positions[i,:] + np.asarray([1.5,0]))
                    normratio   = (normratio-0.22)/normratio
                
                
                
                    plt.plot(
                        [positions[i,0],positions[i,0]+(np.asarray(positions[j,0]-positions[i,0])+1.5)*normratio*0.5],
                        [positions[i,1],positions[i,1]+(np.asarray(positions[j,1]-positions[i,1]))*normratio*0.5],
                        lw = 2,
                        color=whiten(cmap((colorcounter+2)/colorcounter_max)),
                        zorder=-0.5)
                    
                    
                    normratio   = np.linalg.norm(positions[j,:]-positions[i,:] + np.asarray([1.5,0]))
                    normratio   = (normratio-0.22)/normratio
                
                    plt.arrow(
                        positions[i,0]+(np.asarray(positions[j,0]-positions[i,0])+1.5)*normratio*0.5+1.5,
                        positions[i,1]+(np.asarray(positions[j,1]-positions[i,1]))*normratio*0.5,
                        (np.asarray(positions[j,0]-positions[i,0])+1.5)*normratio*0.5,
                        (np.asarray(positions[j,1]-positions[i,1]))*normratio*0.5,
                        head_width = 0.05,
                        ec=whiten(cmap((colorcounter+3)/colorcounter_max)),
                        fc=whiten(cmap((colorcounter+3)/colorcounter_max)),
                        width=0.01,
                        zorder=-0.5)
            
            
        textcolor = colorsys.rgb_to_hsv(
            cmap(colorcounter/colorcounter_max)[0], 
            cmap(colorcounter/colorcounter_max)[1], 
            cmap(colorcounter/colorcounter_max)[2])
        
        textcolor = colorsys.hsv_to_rgb(
            0, 
            0, 
           1 - textcolor[2])
        
        colorcounter += 1

        if figidx == 1 and n == 4:

            # textcolor = tuple(list(textcolor)+[0.5])
            plt.gca().add_patch(plt.Circle(ypos[n,:]+offsets[0,:], 0.15, color=cmap(colorcounter/colorcounter_max),zorder=-3))
            if n not in [0,Nnodes-1]:
                plt.gca().text(ypos[n,0]+offsets[0,0], ypos[n,1]+offsets[0,1], '$a$', ha="center", va="center",zorder=10,color=textcolor,fontsize=10)
            else:
                plt.gca().text(ypos[n,0]+offsets[0,0], ypos[n,1]+offsets[0,1], '$a$', ha="center", va="center",zorder=10,color=[0.95,0.95,0.95],fontsize=10)
            
            plt.gca().add_patch(plt.Circle(ypos[n,:]+offsets[1,:], 0.15, color=whiten(cmap(colorcounter/colorcounter_max)),zorder=-3))
            if n not in [0,Nnodes-1]:
                plt.gca().text(ypos[n,0]+offsets[1,0], ypos[n,1]+offsets[1,1], '$b$', ha="center", va="center",zorder=10,color=whiten(textcolor),fontsize=10)
            else:
                plt.gca().text(ypos[n,0]+offsets[1,0], ypos[n,1]+offsets[1,1], '$b$', ha="center", va="center",zorder=10,color=whiten([0.95,0.95,0.95]),fontsize=10)
                
            plt.gca().add_patch(plt.Circle(ypos[n,:]+offsets[2,:], 0.15, color=whiten(cmap(colorcounter/colorcounter_max)),zorder=-3))
            if n not in [0,Nnodes-1]:
                plt.gca().text(ypos[n,0]+offsets[2,0], ypos[n,1]+offsets[2,1], '$c$', ha="center", va="center",zorder=10,color=whiten(textcolor),fontsize=10)
            else:
                plt.gca().text(ypos[n,0]+offsets[2,0], ypos[n,1]+offsets[2,1], '$c$', ha="center", va="center",zorder=10,color=whiten([0.95,0.95,0.95]),fontsize=10)
        
        else:
            
            plt.gca().add_patch(plt.Circle(ypos[n,:]+offsets[0,:], 0.15, color=whiten(cmap(colorcounter/colorcounter_max)),zorder=-3))

            plt.gca().add_patch(plt.Circle(ypos[n,:]+offsets[1,:], 0.15, color=whiten(cmap(colorcounter/colorcounter_max)),zorder=-3))
 
            if n == 3:
                plt.gca().add_patch(plt.Circle(ypos[n,:]+offsets[2,:], 0.15, color=whiten(cmap(colorcounter/colorcounter_max)),zorder=-3))
                if n not in [0,Nnodes-1]:
                    plt.gca().text(ypos[n,0]+offsets[2,0], ypos[n,1]+offsets[2,1], '$c$', ha="center", va="center",zorder=10,color=whiten(textcolor),fontsize=10)
                else:
                    plt.gca().text(ypos[n,0]+offsets[2,0], ypos[n,1]+offsets[2,1], '$c$', ha="center", va="center",zorder=10,color=whiten([0.95,0.95,0.95]),fontsize=10)
            
            
        if figidx == 1 and n == 4:
        
            
            plt.arrow(
                pos[n,0]+offsets[0,0],
                pos[n,1]-1+0.7+offsets[0,1]+0.3,
                0,
                -1.03*0.5,
                head_width = 0.05,
                ec=cmap(colorcounter/colorcounter_max),
                fc=cmap(colorcounter/colorcounter_max),
                width=0.01,
                zorder=-1)
        

        else:
            
            plt.arrow(
                pos[n,0]+offsets[0,0],
                pos[n,1]-1+0.7+offsets[0,1]+0.3,
                0,
                -1.03*0.5,
                head_width = 0.05,
                ec=whiten(cmap(colorcounter/colorcounter_max)),
                fc=whiten(cmap(colorcounter/colorcounter_max)),
                width=0.01,
                zorder=-1)
            
        plt.arrow(
            pos[n,0]+offsets[1,0],
            pos[n,1]-1+0.7+offsets[1,1]+0.3,
            0,
            -1.03*0.5,
            head_width = 0.05,
            ec=whiten(cmap(colorcounter/colorcounter_max)),
            fc=whiten(cmap(colorcounter/colorcounter_max)),
            width=0.01,
            zorder=-1)
        

        plt.arrow(
            pos[n,0]+offsets[2,0],
            pos[n,1]-1+0.7+offsets[2,1]+0.3,
            0,
            -1.03*0.5,
            head_width = 0.05,
            ec=whiten(cmap(colorcounter/colorcounter_max)),
            fc=whiten(cmap(colorcounter/colorcounter_max)),
            width=0.01,
            zorder=-1)
        
    else:
        
        colorcounter += 1
        
        plt.gca().add_patch(plt.Circle(xpos[n,:]+np.asarray([-0.45,0]), 0.03, color=whiten(cmap(colorcounter/colorcounter_max))))
        
        plt.gca().add_patch(plt.Circle(xpos[n,:]+np.asarray([-0.45,0])+np.asarray([-0.1,0]), 0.03, color=whiten(cmap(colorcounter/colorcounter_max))))
        
        plt.gca().add_patch(plt.Circle(xpos[n,:]+np.asarray([-0.45,0])+np.asarray([+0.1,0]), 0.03, color=whiten(cmap(colorcounter/colorcounter_max))))

            
        plt.gca().add_patch(plt.Circle(xpos[n,:]+np.asarray([+0.25,0]), 0.03, color=whiten(cmap(colorcounter/colorcounter_max))))
        
        plt.gca().add_patch(plt.Circle(xpos[n,:]+np.asarray([+0.25,0])+np.asarray([-0.1,0]), 0.03, color=whiten(cmap(colorcounter/colorcounter_max))))
        
        plt.gca().add_patch(plt.Circle(xpos[n,:]+np.asarray([+0.25,0])+np.asarray([+0.1,0]), 0.03, color=whiten(cmap(colorcounter/colorcounter_max))))


    
plt.gca().relim()
plt.gca().autoscale_view()
plt.axis('equal')

plt.xlim([4.75,6.795])

plt.tick_params(left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False)

if figidx == 0:
    plt.ylabel('$\mathbf{A}$: Lorenz-63 graph',loc="bottom")
else:
    plt.ylabel('$\mathbf{B}$: analysis graph for $y_{t}^{a}$',loc="bottom")
    
    
    
    
#%%
    
figidx = 1

if figidx == 0:
    fac = 0.
else:
    fac = 0.75


plt.subplot(gs[figidx+1,1])




labels  = [
    '_{'+str(1)+'}',
    '_{'+str(2)+'}',
    '\dots$',
    '_{t-1}',
    '_{t}']

Nnodes  = 5

xscale  = 6

xpos    = np.column_stack((
    np.linspace(0,xscale,Nnodes),
    np.ones(Nnodes)))


ypos    = np.column_stack((
    np.linspace(0,xscale,Nnodes),
    np.zeros(Nnodes)+0.25))


skipnode = 2

offsets     = np.asarray([
    [np.cos(-np.pi/2),  np.sin(-np.pi/2)],
    [np.cos(np.pi/4),  np.sin(np.pi/4)-0.1],
    [np.cos(3*np.pi/4),  np.sin(3*np.pi/4)-0.5]])*0.3

offsets     = np.asarray([
    [-1,1],
    [0,0],
    [1,-1]])*0.3


colorcounter = -1
colorcounter_max = 8

pos     = np.row_stack((
    xpos,ypos))


for n in range(Nnodes):
    
    if n != skipnode:
        
        colorcounter += 1
    
    
        textcolor = colorsys.rgb_to_hsv(
            cmap(colorcounter/colorcounter_max)[0], 
            cmap(colorcounter/colorcounter_max)[1], 
            cmap(colorcounter/colorcounter_max)[2])
        
        textcolor = colorsys.hsv_to_rgb(
            0, 
            0, 
            1 - textcolor[2])
        
        
        if figidx == 1 and n == 4:
            
            # textcolor = tuple(list(textcolor)+[0.5])
            plt.gca().add_patch(plt.Circle(xpos[n,:]+offsets[0,:], 0.15, color=cmap(colorcounter/colorcounter_max)))
            if n not in [0,Nnodes-1]:
                plt.gca().text(xpos[n,0]+offsets[0,0], xpos[n,1]+offsets[0,1], '$a$', ha="center", va="center",zorder=10,color=textcolor,fontsize=10)
            else:
                plt.gca().text(xpos[n,0]+offsets[0,0], xpos[n,1]+offsets[0,1], '$a$', ha="center", va="center",zorder=10,color=[0.95,0.95,0.95],fontsize=10)
            
            plt.gca().add_patch(plt.Circle(xpos[n,:]+offsets[1,:], 0.15, color=cmap(colorcounter/colorcounter_max)))
            if n not in [0,Nnodes-1]:
                plt.gca().text(xpos[n,0]+offsets[1,0], xpos[n,1]+offsets[1,1], '$b$', ha="center", va="center",zorder=10,color=textcolor,fontsize=10)
            else:
                plt.gca().text(xpos[n,0]+offsets[1,0], xpos[n,1]+offsets[1,1], '$b$', ha="center", va="center",zorder=10,color=[0.95,0.95,0.95],fontsize=10)
                
            plt.gca().add_patch(plt.Circle(xpos[n,:]+offsets[2,:], 0.15, color=cmap(colorcounter/colorcounter_max)))
            if n not in [0,Nnodes-1]:
                plt.gca().text(xpos[n,0]+offsets[2,0], xpos[n,1]+offsets[2,1], '$c$', ha="center", va="center",zorder=10,color=textcolor,fontsize=10)
            else:
                plt.gca().text(xpos[n,0]+offsets[2,0], xpos[n,1]+offsets[2,1], '$c$', ha="center", va="center",zorder=10,color=[0.95,0.95,0.95],fontsize=10)
        
        else:
        
            # # textcolor = tuple(list(textcolor)+[0.5])
            plt.gca().add_patch(plt.Circle(xpos[n,:]+offsets[0,:], 0.15, color=whiten(cmap(colorcounter/colorcounter_max))))

            plt.gca().add_patch(plt.Circle(xpos[n,:]+offsets[1,:], 0.15, color=whiten(cmap(colorcounter/colorcounter_max))))

            if n == 3:
                plt.gca().add_patch(plt.Circle(xpos[n,:]+offsets[2,:], 0.15, color=whiten(cmap(colorcounter/colorcounter_max))))
                if n not in [0,Nnodes-1]:
                    plt.gca().text(xpos[n,0]+offsets[2,0], xpos[n,1]+offsets[2,1], '$c$', ha="center", va="center",zorder=10,color=whiten(textcolor),fontsize=10)
                else:
                    plt.gca().text(xpos[n,0]+offsets[2,0], xpos[n,1]+offsets[2,1], '$c$', ha="center", va="center",zorder=10,color=whiten([0.95,0.95,0.95]),fontsize=10)
        
        line = np.column_stack((
            np.linspace(
                list(xpos[n,0]+offsets[:,0])[0],
                list(xpos[n,0]+offsets[:,0])[-1],
                101),
            np.linspace(
                list(xpos[n,1]+offsets[:,1])[0],
                list(xpos[n,1]+offsets[:,1])[-1],
                101) - np.sin(np.linspace(-np.pi,0,101))*0.35))
        
        from scipy.interpolate import interp1d
        
        itp     = interp1d(
            np.linspace(0,1,101),
            np.linspace(
                list(xpos[n,0]+offsets[:,0])[0],
                list(xpos[n,0]+offsets[:,0])[-1],
                101))
        
        import scipy.stats
        
        
        if n == 4:
            
            plt.plot(
                list(xpos[n,0]+offsets[:,0])+[list(xpos[n,0]+offsets[:,0])[0]],
                list(xpos[n,1]+offsets[:,1])+[list(xpos[n,1]+offsets[:,1])[0]],
                color   = cmap(colorcounter/colorcounter_max),
                alpha   = 1,
                lw      = 2,
                zorder  = -1,
                ls      = '--')
            
            plt.plot(
                itp(scipy.stats.beta.cdf(np.linspace(0,1,101),a=1.5,b=1.5)),
                line[:,1]+0.135,
                color   = cmap(colorcounter/colorcounter_max),
                lw      = 2,
                zorder  = -1,
                ls      = "--")

        positions   = np.column_stack((
            xpos[n,0]+offsets[:,0],
            xpos[n,1]+offsets[:,1]))
        
        from scipy.interpolate import CubicSpline
        
        if n != Nnodes-1 and n != 1:
            
            tinyoffset = [0.0,-0.0,0.]
            
            tinieroffset = [0.02,-0.02,0.]
        
            for i in range(3):
                
                for j in range(3):

                    normratio   = np.linalg.norm(positions[j,:]-positions[i,:] + np.asarray([1.5,0]))
                    normratio   = (normratio-0.22)/normratio
                
                    plt.arrow(
                        positions[i,0],
                        positions[i,1],
                        (np.asarray(positions[j,0]-positions[i,0])+1.5)*normratio,
                        (np.asarray(positions[j,1]-positions[i,1]))*normratio,
                        head_width = 0.05,
                        ec=whiten(cmap((colorcounter+2)/colorcounter_max)),
                        fc=whiten(cmap((colorcounter+2)/colorcounter_max)),
                        width=0.01,
                        zorder=-0.5)
                
        elif n != Nnodes-1:
            
            for i in range(3):
                
                for j in range(3):

                    normratio   = np.linalg.norm(positions[j,:]-positions[i,:] + np.asarray([1.5,0]))
                    normratio   = (normratio-0.22)/normratio
                
                    plt.plot(
                        [positions[i,0],positions[i,0]+(np.asarray(positions[j,0]-positions[i,0])+1.5)*normratio*0.5],
                        [positions[i,1],positions[i,1]+(np.asarray(positions[j,1]-positions[i,1]))*normratio*0.5],
                        lw = 2,
                        color=whiten(cmap((colorcounter+2)/colorcounter_max)),
                        zorder=-0.5)
                    
                    
                    normratio   = np.linalg.norm(positions[j,:]-positions[i,:] + np.asarray([1.5,0]))
                    normratio   = (normratio-0.22)/normratio
                
                    plt.arrow(
                        positions[i,0]+(np.asarray(positions[j,0]-positions[i,0])+1.5)*normratio*0.5+1.5,
                        positions[i,1]+(np.asarray(positions[j,1]-positions[i,1]))*normratio*0.5,
                        (np.asarray(positions[j,0]-positions[i,0])+1.5)*normratio*0.5,
                        (np.asarray(positions[j,1]-positions[i,1]))*normratio*0.5,
                        head_width = 0.05,
                        ec=whiten(cmap((colorcounter+3)/colorcounter_max)),
                        fc=whiten(cmap((colorcounter+3)/colorcounter_max)),
                        width=0.01,
                        zorder=-0.5)
            
            
        textcolor = colorsys.rgb_to_hsv(
            cmap(colorcounter/colorcounter_max)[0], 
            cmap(colorcounter/colorcounter_max)[1], 
            cmap(colorcounter/colorcounter_max)[2])
        
        textcolor = colorsys.hsv_to_rgb(
            0, 
            0, 
           1 - textcolor[2])
        
        
        colorcounter += 1

        if figidx == 1 and n == 4:

            # textcolor = tuple(list(textcolor)+[0.5])
            plt.gca().add_patch(plt.Circle(ypos[n,:]+offsets[0,:], 0.15, color=whiten(cmap(colorcounter/colorcounter_max)),zorder=-3))
            if n not in [0,Nnodes-1]:
                plt.gca().text(ypos[n,0]+offsets[0,0], ypos[n,1]+offsets[0,1], '$a$', ha="center", va="center",zorder=10,color=whiten(textcolor),fontsize=10)
            else:
                plt.gca().text(ypos[n,0]+offsets[0,0], ypos[n,1]+offsets[0,1], '$a$', ha="center", va="center",zorder=10,color=whiten([0.95,0.95,0.95]),fontsize=10)
            
            plt.gca().add_patch(plt.Circle(ypos[n,:]+offsets[1,:], 0.15, color=cmap(colorcounter/colorcounter_max),zorder=-3))
            if n not in [0,Nnodes-1]:
                plt.gca().text(ypos[n,0]+offsets[1,0], ypos[n,1]+offsets[1,1], '$b$', ha="center", va="center",zorder=10,color=textcolor,fontsize=10)
            else:
                plt.gca().text(ypos[n,0]+offsets[1,0], ypos[n,1]+offsets[1,1], '$b$', ha="center", va="center",zorder=10,color=[0.95,0.95,0.95],fontsize=10)
                
            plt.gca().add_patch(plt.Circle(ypos[n,:]+offsets[2,:], 0.15, color=whiten(cmap(colorcounter/colorcounter_max)),zorder=-3))
            if n not in [0,Nnodes-1]:
                plt.gca().text(ypos[n,0]+offsets[2,0], ypos[n,1]+offsets[2,1], '$c$', ha="center", va="center",zorder=10,color=whiten(textcolor),fontsize=10)
            else:
                plt.gca().text(ypos[n,0]+offsets[2,0], ypos[n,1]+offsets[2,1], '$c$', ha="center", va="center",zorder=10,color=whiten([0.95,0.95,0.95]),fontsize=10)
        
        else:
            
            plt.gca().add_patch(plt.Circle(ypos[n,:]+offsets[0,:], 0.15, color=whiten(cmap(colorcounter/colorcounter_max)),zorder=-3))
  
            plt.gca().add_patch(plt.Circle(ypos[n,:]+offsets[1,:], 0.15, color=whiten(cmap(colorcounter/colorcounter_max)),zorder=-3))

            if n == 3:
                plt.gca().add_patch(plt.Circle(ypos[n,:]+offsets[2,:], 0.15, color=whiten(cmap(colorcounter/colorcounter_max)),zorder=-3))
                if n not in [0,Nnodes-1]:
                    plt.gca().text(ypos[n,0]+offsets[2,0], ypos[n,1]+offsets[2,1], '$c$', ha="center", va="center",zorder=10,color=whiten(textcolor),fontsize=10)
                else:
                    plt.gca().text(ypos[n,0]+offsets[2,0], ypos[n,1]+offsets[2,1], '$c$', ha="center", va="center",zorder=10,color=whiten([0.95,0.95,0.95]),fontsize=10)

        plt.arrow(
            pos[n,0]+offsets[0,0],
            pos[n,1]-1+0.7+offsets[0,1]+0.3,
            0,
            -1.03*0.5,
            head_width = 0.05,
            ec=whiten(cmap(colorcounter/colorcounter_max)),
            fc=whiten(cmap(colorcounter/colorcounter_max)),
            width=0.01,
            zorder=-1)
        
        if figidx == 1 and n == 4:
        
            plt.arrow(
                pos[n,0]+offsets[1,0],
                pos[n,1]-1+0.7+offsets[1,1]+0.3,
                0,
                -1.03*0.5,
                head_width = 0.05,
                ec=cmap(colorcounter/colorcounter_max),
                fc=cmap(colorcounter/colorcounter_max),
                width=0.01,
                zorder=-1)
            
        else:
            
            plt.arrow(
                pos[n,0]+offsets[1,0],
                pos[n,1]-1+0.7+offsets[1,1]+0.3,
                0,
                -1.03*0.5,
                head_width = 0.05,
                ec=whiten(cmap(colorcounter/colorcounter_max)),
                fc=whiten(cmap(colorcounter/colorcounter_max)),
                width=0.01,
                zorder=-1)
        
        plt.arrow(
            pos[n,0]+offsets[2,0],
            pos[n,1]-1+0.7+offsets[2,1]+0.3,
            0,
            -1.03*0.5,
            head_width = 0.05,
            ec=whiten(cmap(colorcounter/colorcounter_max)),
            fc=whiten(cmap(colorcounter/colorcounter_max)),
            width=0.01,
            zorder=-1)
        
    else:
        
        colorcounter += 1
        
        plt.gca().add_patch(plt.Circle(xpos[n,:]+np.asarray([-0.45,0]), 0.03, color=whiten(cmap(colorcounter/colorcounter_max))))
        
        plt.gca().add_patch(plt.Circle(xpos[n,:]+np.asarray([-0.45,0])+np.asarray([-0.1,0]), 0.03, color=whiten(cmap(colorcounter/colorcounter_max))))
        
        plt.gca().add_patch(plt.Circle(xpos[n,:]+np.asarray([-0.45,0])+np.asarray([+0.1,0]), 0.03, color=whiten(cmap(colorcounter/colorcounter_max))))

            
        plt.gca().add_patch(plt.Circle(xpos[n,:]+np.asarray([+0.25,0]), 0.03, color=whiten(cmap(colorcounter/colorcounter_max))))
        
        plt.gca().add_patch(plt.Circle(xpos[n,:]+np.asarray([+0.25,0])+np.asarray([-0.1,0]), 0.03, color=whiten(cmap(colorcounter/colorcounter_max))))
        
        plt.gca().add_patch(plt.Circle(xpos[n,:]+np.asarray([+0.25,0])+np.asarray([+0.1,0]), 0.03, color=whiten(cmap(colorcounter/colorcounter_max))))

plt.gca().relim()
plt.gca().autoscale_view()
plt.axis('equal')

plt.xlim([4.75,6.795])

plt.tick_params(left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False)

if figidx == 0:
    plt.ylabel('$\mathbf{A}$: Lorenz-63 graph',loc="bottom")
else:
    plt.ylabel('$\mathbf{C}$: analysis graph for $y_{t}^{b}$',loc="bottom")
        
    
    
#%%

figidx = 1

if figidx == 0:
    fac = 0.
else:
    fac = 0.75


plt.subplot(gs[figidx+1,2])


labels  = [
    '_{'+str(1)+'}',
    '_{'+str(2)+'}',
    '\dots$',
    '_{t-1}',
    '_{t}']

Nnodes  = 5

xscale  = 6

xpos    = np.column_stack((
    np.linspace(0,xscale,Nnodes),
    np.ones(Nnodes)))


ypos    = np.column_stack((
    np.linspace(0,xscale,Nnodes),
    np.zeros(Nnodes)+0.25))


skipnode = 2

offsets     = np.asarray([
    [np.cos(-np.pi/2),  np.sin(-np.pi/2)],
    [np.cos(np.pi/4),  np.sin(np.pi/4)-0.1],
    [np.cos(3*np.pi/4),  np.sin(3*np.pi/4)-0.5]])*0.3

offsets     = np.asarray([
    [-1,1],
    [0,0],
    [1,-1]])*0.3


colorcounter = -1
colorcounter_max = 8

pos     = np.row_stack((
    xpos,ypos))


for n in range(Nnodes):
    
    if n != skipnode:
        
        colorcounter += 1
    
    
        textcolor = colorsys.rgb_to_hsv(
            cmap(colorcounter/colorcounter_max)[0], 
            cmap(colorcounter/colorcounter_max)[1], 
            cmap(colorcounter/colorcounter_max)[2])
        
        textcolor = colorsys.hsv_to_rgb(
            0, 
            0, 
            1 - textcolor[2])
        
        
        if figidx == 1 and n == 4:
            
            # textcolor = tuple(list(textcolor)+[0.5])
            plt.gca().add_patch(plt.Circle(xpos[n,:]+offsets[0,:], 0.15, color=cmap(colorcounter/colorcounter_max)))
            if n not in [0,Nnodes-1]:
                plt.gca().text(xpos[n,0]+offsets[0,0], xpos[n,1]+offsets[0,1], '$a$', ha="center", va="center",zorder=10,color=textcolor,fontsize=10)
            else:
                plt.gca().text(xpos[n,0]+offsets[0,0], xpos[n,1]+offsets[0,1], '$a$', ha="center", va="center",zorder=10,color=[0.95,0.95,0.95],fontsize=10)
            
            plt.gca().add_patch(plt.Circle(xpos[n,:]+offsets[1,:], 0.15, color=cmap(colorcounter/colorcounter_max)))
            if n not in [0,Nnodes-1]:
                plt.gca().text(xpos[n,0]+offsets[1,0], xpos[n,1]+offsets[1,1], '$b$', ha="center", va="center",zorder=10,color=textcolor,fontsize=10)
            else:
                plt.gca().text(xpos[n,0]+offsets[1,0], xpos[n,1]+offsets[1,1], '$b$', ha="center", va="center",zorder=10,color=[0.95,0.95,0.95],fontsize=10)
                
            plt.gca().add_patch(plt.Circle(xpos[n,:]+offsets[2,:], 0.15, color=cmap(colorcounter/colorcounter_max)))
            if n not in [0,Nnodes-1]:
                plt.gca().text(xpos[n,0]+offsets[2,0], xpos[n,1]+offsets[2,1], '$c$', ha="center", va="center",zorder=10,color=textcolor,fontsize=10)
            else:
                plt.gca().text(xpos[n,0]+offsets[2,0], xpos[n,1]+offsets[2,1], '$c$', ha="center", va="center",zorder=10,color=[0.95,0.95,0.95],fontsize=10)
        
        else:
        
            # # textcolor = tuple(list(textcolor)+[0.5])
            plt.gca().add_patch(plt.Circle(xpos[n,:]+offsets[0,:], 0.15, color=whiten(cmap(colorcounter/colorcounter_max))))

            plt.gca().add_patch(plt.Circle(xpos[n,:]+offsets[1,:], 0.15, color=whiten(cmap(colorcounter/colorcounter_max))))
               
            if n == 3:
                plt.gca().add_patch(plt.Circle(xpos[n,:]+offsets[2,:], 0.15, color=whiten(cmap(colorcounter/colorcounter_max))))
                if n not in [0,Nnodes-1]:
                    plt.gca().text(xpos[n,0]+offsets[2,0], xpos[n,1]+offsets[2,1], '$c$', ha="center", va="center",zorder=10,color=whiten(textcolor),fontsize=10)
                else:
                    plt.gca().text(xpos[n,0]+offsets[2,0], xpos[n,1]+offsets[2,1], '$c$', ha="center", va="center",zorder=10,color=whiten([0.95,0.95,0.95]),fontsize=10)

        line = np.column_stack((
            np.linspace(
                list(xpos[n,0]+offsets[:,0])[0],
                list(xpos[n,0]+offsets[:,0])[-1],
                101),
            np.linspace(
                list(xpos[n,1]+offsets[:,1])[0],
                list(xpos[n,1]+offsets[:,1])[-1],
                101) - np.sin(np.linspace(-np.pi,0,101))*0.35))
        
        from scipy.interpolate import interp1d
        
        itp     = interp1d(
            np.linspace(0,1,101),
            np.linspace(
                list(xpos[n,0]+offsets[:,0])[0],
                list(xpos[n,0]+offsets[:,0])[-1],
                101))
        
        import scipy.stats
        
        if n == 4:
            
            plt.plot(
                list(xpos[n,0]+offsets[:2,0])+[list(xpos[n,0]+offsets[:2,0])[0]],
                list(xpos[n,1]+offsets[:2,1])+[list(xpos[n,1]+offsets[:2,1])[0]],
                color   = cmap(colorcounter/colorcounter_max),
                alpha   = 1,
                lw      = 2,
                zorder  = -1,
                ls      = "--")
            
            plt.plot(
                list(xpos[n,0]+offsets[1:,0])+[list(xpos[n,0]+offsets[1:,0])[0]],
                list(xpos[n,1]+offsets[1:,1])+[list(xpos[n,1]+offsets[1:,1])[0]],
                color   = cmap(colorcounter/colorcounter_max),
                alpha   = 1,
                lw      = 2,
                zorder  = -1,
                ls      = "--")
            
            plt.plot(
                itp(scipy.stats.beta.cdf(np.linspace(0,1,101),a=1.5,b=1.5)),
                line[:,1]+0.135,
                color   = cmap(colorcounter/colorcounter_max),
                lw      = 2,
                zorder  = -1,
                ls      = "--")

        positions   = np.column_stack((
            xpos[n,0]+offsets[:,0],
            xpos[n,1]+offsets[:,1]))
        
        from scipy.interpolate import CubicSpline
        
        if n != Nnodes-1 and n != 1:
            
            tinyoffset = [0.0,-0.0,0.]
            
            tinieroffset = [0.02,-0.02,0.]
        
            for i in range(3):
                
                for j in range(3):

                    normratio   = np.linalg.norm(positions[j,:]-positions[i,:] + np.asarray([1.5,0]))
                    normratio   = (normratio-0.22)/normratio
                
                    plt.arrow(
                        positions[i,0],
                        positions[i,1],
                        (np.asarray(positions[j,0]-positions[i,0])+1.5)*normratio,
                        (np.asarray(positions[j,1]-positions[i,1]))*normratio,
                        head_width = 0.05,
                        ec=whiten(cmap((colorcounter+2)/colorcounter_max)),
                        fc=whiten(cmap((colorcounter+2)/colorcounter_max)),
                        width=0.01,
                        zorder=-0.5)
                
        elif n != Nnodes-1:
            
            for i in range(3):
                
                for j in range(3):

                    normratio   = np.linalg.norm(positions[j,:]-positions[i,:] + np.asarray([1.5,0]))
                    normratio   = (normratio-0.22)/normratio
                
                
                
                    plt.plot(
                        [positions[i,0],positions[i,0]+(np.asarray(positions[j,0]-positions[i,0])+1.5)*normratio*0.5],
                        [positions[i,1],positions[i,1]+(np.asarray(positions[j,1]-positions[i,1]))*normratio*0.5],
                        lw = 2,
                        color=whiten(cmap((colorcounter+2)/colorcounter_max)),
                        zorder=-0.5)
                    
                    
                    normratio   = np.linalg.norm(positions[j,:]-positions[i,:] + np.asarray([1.5,0]))
                    normratio   = (normratio-0.22)/normratio
                
                    plt.arrow(
                        positions[i,0]+(np.asarray(positions[j,0]-positions[i,0])+1.5)*normratio*0.5+1.5,
                        positions[i,1]+(np.asarray(positions[j,1]-positions[i,1]))*normratio*0.5,
                        (np.asarray(positions[j,0]-positions[i,0])+1.5)*normratio*0.5,
                        (np.asarray(positions[j,1]-positions[i,1]))*normratio*0.5,
                        head_width = 0.05,
                        ec=whiten(cmap((colorcounter+3)/colorcounter_max)),
                        fc=whiten(cmap((colorcounter+3)/colorcounter_max)),
                        width=0.01,
                        zorder=-0.5)
            
            
        textcolor = colorsys.rgb_to_hsv(
            cmap(colorcounter/colorcounter_max)[0], 
            cmap(colorcounter/colorcounter_max)[1], 
            cmap(colorcounter/colorcounter_max)[2])
        
        textcolor = colorsys.hsv_to_rgb(
            0, 
            0, 
           1 - textcolor[2])
        
        colorcounter += 1

        if figidx == 1 and n == 4:

            # textcolor = tuple(list(textcolor)+[0.5])
            plt.gca().add_patch(plt.Circle(ypos[n,:]+offsets[0,:], 0.15, color=whiten(cmap(colorcounter/colorcounter_max)),zorder=-3))
            if n not in [0,Nnodes-1]:
                plt.gca().text(ypos[n,0]+offsets[0,0], ypos[n,1]+offsets[0,1], '$a$', ha="center", va="center",zorder=10,color=whiten(textcolor),fontsize=10)
            else:
                plt.gca().text(ypos[n,0]+offsets[0,0], ypos[n,1]+offsets[0,1], '$a$', ha="center", va="center",zorder=10,color=whiten([0.95,0.95,0.95]),fontsize=10)
            
            plt.gca().add_patch(plt.Circle(ypos[n,:]+offsets[1,:], 0.15, color=whiten(cmap(colorcounter/colorcounter_max)),zorder=-3))
            if n not in [0,Nnodes-1]:
                plt.gca().text(ypos[n,0]+offsets[1,0], ypos[n,1]+offsets[1,1], '$b$', ha="center", va="center",zorder=10,color=whiten(textcolor),fontsize=10)
            else:
                plt.gca().text(ypos[n,0]+offsets[1,0], ypos[n,1]+offsets[1,1], '$b$', ha="center", va="center",zorder=10,color=whiten([0.95,0.95,0.95]),fontsize=10)
                
            plt.gca().add_patch(plt.Circle(ypos[n,:]+offsets[2,:], 0.15, color=cmap(colorcounter/colorcounter_max),zorder=-3))
            if n not in [0,Nnodes-1]:
                plt.gca().text(ypos[n,0]+offsets[2,0], ypos[n,1]+offsets[2,1], '$c$', ha="center", va="center",zorder=10,color=textcolor,fontsize=10)
            else:
                plt.gca().text(ypos[n,0]+offsets[2,0], ypos[n,1]+offsets[2,1], '$c$', ha="center", va="center",zorder=10,color=[0.95,0.95,0.95],fontsize=10)
        
        else:
            
            # textcolor = tuple(list(textcolor)+[0.5])
            plt.gca().add_patch(plt.Circle(ypos[n,:]+offsets[0,:], 0.15, color=whiten(cmap(colorcounter/colorcounter_max)),zorder=-3))

            plt.gca().add_patch(plt.Circle(ypos[n,:]+offsets[1,:], 0.15, color=whiten(cmap(colorcounter/colorcounter_max)),zorder=-3))

            plt.gca().add_patch(plt.Circle(ypos[n,:]+offsets[2,:], 0.15, color=whiten(cmap(colorcounter/colorcounter_max)),zorder=-3))
            
            if n == 3:
                if n not in [0,Nnodes-1]:
                    plt.gca().text(ypos[n,0]+offsets[2,0], ypos[n,1]+offsets[2,1], '$c$', ha="center", va="center",zorder=10,color=whiten(textcolor),fontsize=10)
                else:
                    plt.gca().text(ypos[n,0]+offsets[2,0], ypos[n,1]+offsets[2,1], '$c$', ha="center", va="center",zorder=10,color=whiten([0.95,0.95,0.95]),fontsize=10)
            

            
        plt.arrow(
            pos[n,0]+offsets[0,0],
            pos[n,1]-1+0.7+offsets[0,1]+0.3,
            0,
            -1.03*0.5,
            head_width = 0.05,
            ec=whiten(cmap(colorcounter/colorcounter_max)),
            fc=whiten(cmap(colorcounter/colorcounter_max)),
            width=0.01,
            zorder=-1)
            
        plt.arrow(
            pos[n,0]+offsets[1,0],
            pos[n,1]-1+0.7+offsets[1,1]+0.3,
            0,
            -1.03*0.5,
            head_width = 0.05,
            ec=whiten(cmap(colorcounter/colorcounter_max)),
            fc=whiten(cmap(colorcounter/colorcounter_max)),
            width=0.01,
            zorder=-1)
        
        if figidx == 1 and n == 4:
        
            plt.arrow(
                pos[n,0]+offsets[2,0],
                pos[n,1]-1+0.7+offsets[2,1]+0.3,
                0,
                -1.03*0.5,
                head_width = 0.05,
                ec=cmap(colorcounter/colorcounter_max),
                fc=cmap(colorcounter/colorcounter_max),
                width=0.01,
                zorder=-1)
        
        else:
            
            plt.arrow(
                pos[n,0]+offsets[2,0],
                pos[n,1]-1+0.7+offsets[2,1]+0.3,
                0,
                -1.03*0.5,
                head_width = 0.05,
                ec=whiten(cmap(colorcounter/colorcounter_max)),
                fc=whiten(cmap(colorcounter/colorcounter_max)),
                width=0.01,
                zorder=-1)
        
    else:
        
        colorcounter += 1
        
        plt.gca().add_patch(plt.Circle(xpos[n,:]+np.asarray([-0.45,0]), 0.03, color=whiten(cmap(colorcounter/colorcounter_max))))
        
        plt.gca().add_patch(plt.Circle(xpos[n,:]+np.asarray([-0.45,0])+np.asarray([-0.1,0]), 0.03, color=whiten(cmap(colorcounter/colorcounter_max))))
        
        plt.gca().add_patch(plt.Circle(xpos[n,:]+np.asarray([-0.45,0])+np.asarray([+0.1,0]), 0.03, color=whiten(cmap(colorcounter/colorcounter_max))))

            
        plt.gca().add_patch(plt.Circle(xpos[n,:]+np.asarray([+0.25,0]), 0.03, color=whiten(cmap(colorcounter/colorcounter_max))))
        
        plt.gca().add_patch(plt.Circle(xpos[n,:]+np.asarray([+0.25,0])+np.asarray([-0.1,0]), 0.03, color=whiten(cmap(colorcounter/colorcounter_max))))
        
        plt.gca().add_patch(plt.Circle(xpos[n,:]+np.asarray([+0.25,0])+np.asarray([+0.1,0]), 0.03, color=whiten(cmap(colorcounter/colorcounter_max))))


    
plt.gca().relim()
plt.gca().autoscale_view()
plt.axis('equal')
xlim    = plt.gca().get_xlim()

plt.xlim([4.75,6.795])

plt.tick_params(left=False,
                bottom=False,
                labelleft=False,
                labelbottom=False)

if figidx == 0:
    plt.ylabel('$\mathbf{A}$: Lorenz-63 graph',loc="bottom")
else:
    plt.ylabel('$\mathbf{D}$: analysis graph for $y_{t}^{c}$',loc="bottom")
    

# Save the figure
plt.savefig('graph_Lorenz_resolved.png',dpi=600,bbox_inches='tight')
plt.savefig('graph_Lorenz_resolved.pdf',dpi=600,bbox_inches='tight')