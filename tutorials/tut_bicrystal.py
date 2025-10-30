import numpy as np
import gsd.hoomd
from matplotlib.cm import hsv as hsv_map

from visuals import SuperEllipse, plot_principal_axes
from calc.locality import DEFAULT_CUTOFF
from calc import gyration_tensor
from coloring import ColorBondOrder, ColorConn, base_colors, color_gradient

# we define a color function which blends blue to white to red on a -1 to 1 scale
c1 = color_gradient(c1=base_colors['white'], c2=base_colors['red'])
c2 = color_gradient(c1=base_colors['white'], c2=base_colors['blue'])
rwb = lambda n: (c1(n).T*(n>0) + c2(-n).T*(n<=0)).T

disc = SuperEllipse(ax=0.5, ay=0.5, n=2.0)
class ColorDomains(ColorConn):
    """Class to identify and color particles by whether they are in a crystalline domain"""
    def __init__(self, order = 6, dark=True):
        # here in the constructor we initialize the base class with a lot of default 
        # values which make sense for our use case (no periodic boundaries, 2D discs, etc.)
        super().__init__(order=order, shape=disc, 
                         nei_cutoff=DEFAULT_CUTOFF, periodic=False, 
                         calc_3d=False, 
                         crystallinity_threshold=0.32, norm=order,
                         dark=dark)
        self._c = rwb # defining the color mapper happens in the constructor

    def calc_state(self):
        """find the neighbor matrix of each particle stretched by the shape parameters."""
        super().calc_state()
        # we can rely on the superclass to calculate basic bond order parameters and thus avoid
        # repeating sensitive code.
        psi6 = self.psi
        c6 = self.con

        # here we identify how closely particle environments align with each other, similar to C6
        psi_i, psi_j = np.meshgrid(psi6, psi6)
        cross_psi = psi_i*np.conjugate(psi_j)
        
        # find the particle whose environment aligns best with the most other particles
        best_ptcl = np.argmax(np.sum(np.real(cross_psi)>0.5, axis=-1))
        self.in_domain = np.real(cross_psi[:, best_ptcl])*c6 # weight by crystallinity
        # assign the color so that particles in the domain are positive (red), out of domain negative (blue)
        # we could just use in_domain directly, but this makes the color mapping more obvious to the eye
        self.ci = 1.0*(self.in_domain>0.5) -1.0*(self.in_domain<0)
    
if __name__ == "__main__":
    from render import render_npole,animate
    import matplotlib.pyplot as plt

    plt.style.use('dark_background')
    supfig, axs = plt.subplots(2,3, figsize=(7,4), dpi=500)

    # define an extended figure making function which places the principal moments of the biggest
    # crystal cluster overtop the render image.
    def figure_maker(snap):
        style = ColorDomains(dark=True)
        style.snap = snap
        fig,ax = render_npole(snap, style=style, PEL='contour', dark=True, figsize=5, dpi=500)
        if np.sum(style.in_domain>0.5) > 1:
            pts = snap.particles.position[style.in_domain>0.5]
            com = pts.mean(axis=0)
            gyr = gyration_tensor(pts)
            plot_principal_axes(gyr=gyr, com=com, ax=ax, color='green', lw=2.5)
        return fig, ax

    # now run the figure maker through animate for each file
    for f in ['qpole2.gsd','opole1.gsd']: 
        with gsd.hoomd.open(f, "r") as traj:
            animate(traj, figure_maker=figure_maker, outpath=f"xtal-domains-{f[:-4]}.webm", fps=10, codec='libvpx')