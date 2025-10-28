import gsd.hoomd
import numpy as np
import matplotlib.pyplot as plt

from visuals import SuperEllipse
from coloring import ColorBlender, base_colors, color_blender
from coloring import ColorConn, ColorQG
from render import render_3d, animate


white_purp = color_blender(c00=base_colors['white'], c01=base_colors['red'], c10=base_colors['blue'], c11=base_colors['purple'])
X, Y = np.meshgrid(np.linspace(0, 1, 128), np.linspace(0, 1, 128))
img = white_purp(X, Y)

# Display the results
fig, ax = plt.subplots(figsize=(4, 4))
ax.imshow(img, origin='lower')
ax.set_xlabel('$Q_6$')
ax.set_xticks(np.linspace(0, 127, 6))
ax.set_xticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
ax.set_yticks(np.linspace(0, 127, 6))
ax.set_yticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
ax.set_ylabel('$C_6$')
fig.savefig("white-purp.png", dpi=600, bbox_inches='tight')


with gsd.hoomd.open('clust.gsd', 'r') as traj:

    q6_style = ColorQG(periodic=True)
    c6_style = ColorConn(periodic=True, calc_3d=True)
    style = ColorBlender(white_purp, q6_style, c6_style)
    
    # sometimes the it's helpful to specify a constant box size for 3D renders, especially with
    # small simulation boxes like this example
    L0 = traj[0].configuration.box[0]*2.5
    blend_fig = lambda snap: render_3d(snap, style=style, dark=True, figsize=5, dpi=500, L=L0)
    animate(traj, figure_maker=blend_fig, outpath="q6c6-clust.mp4", fps=10)