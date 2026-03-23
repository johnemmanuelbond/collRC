import gsd.hoomd
import numpy as np

from coloring import ColorC6Defects
from render import render_sphere, animate

view = np.array([0.3, -1, 0])
view = view/np.linalg.norm(view)

sphere_grad = lambda r: r / np.linalg.norm(r, axis=-1, keepdims=True)
with gsd.hoomd.open('sphere.gsd', 'r') as traj:

    L0 = traj[0].configuration.box[0]

    style = ColorC6Defects(surface_normal=None)
    fm1 = lambda snap: render_sphere(snap, style=style, dark=True, figsize=5, dpi=500, L=L0, view_dir = view)
    animate(traj, figure_maker=fm1, outpath="c6d-sphere-incorrect.mp4", fps=10)
    animate(traj, figure_maker=fm1, outpath="c6d-sphere-incorrect.webm", fps=10, codec='libvpx')

    style = ColorC6Defects(surface_normal=sphere_grad)
    fm2 = lambda snap: render_sphere(snap, style=style, dark=True, figsize=5, dpi=500, L=L0, view_dir = view)
    animate(traj, figure_maker=fm2, outpath="c6d-sphere-correct.mp4", fps=10)
    animate(traj, figure_maker=fm2, outpath="c6d-sphere-correct.webm", fps=10, codec='libvpx')

# moviepy supports rendering files to all kinds of extensions using all kinds of codecs.
# in this example we make both mp4 files (for slideshows) and webm files (for embedding in webpages)