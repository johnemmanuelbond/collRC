import gsd.hoomd

from visuals import SuperEllipse
from coloring import ColorEta0, ColorS2Defects
from coloring import ColorConn, ColorC4Defects
from render import render_npole, animate

# set up a style to render misorientation defects on top of a background colored by central area fraction
bg_style = ColorEta0(shape=SuperEllipse(ax=1.0, ay=0.5, n=20.0))
style = ColorS2Defects(shape=SuperEllipse(ax=1.0, ay=0.5, n=20.0), bgColor=bg_style)
fm1 = lambda snap: render_npole(snap, style=style, PEL='contour', dark=True, figsize=5, dpi=500)

# set up a style to render C4 defects on top of a background colored by the crystal connectivity
bg_style = ColorConn(shape=SuperEllipse(ax=1.0, ay=0.5, n=20.0), order=4, periodic=True, nei_cutoff=2.6)
style = ColorC4Defects(shape=SuperEllipse(ax=1.0, ay=0.5, n=20.0), bgColor=bg_style, periodic=True)
fm2 = lambda snap: render_npole(snap, style=style, PEL='contour', dark=True, figsize=5, dpi=500)

# render each on each gsd file
for f in ['rect1.gsd', 'rect2.gsd']:
    with  gsd.hoomd.open(f, 'r') as traj:
        animate(traj, figure_maker=fm1, outpath="s2d-"+f.replace('.gsd', '.mp4'), fps=10)
        animate(traj, figure_maker=fm2, outpath="c4d-"+f.replace('.gsd', '.mp4'), fps=10)