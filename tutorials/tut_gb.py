import numpy as np
import gsd.hoomd
from matplotlib.cm import hsv as hsv_map

from visuals import SuperEllipse, plot_principal_axes
from calc.locality import DEFAULT_CUTOFF
from calc import gyration_tensor, gyration_radius, ellipticity
from coloring import ColorBondOrder, ColorConn, base_colors, color_gradient

# we define a color function which blends blue to white to red on a -1 to 1 scale
c1 = color_gradient(c1=base_colors['white'], c2=base_colors['red'])
c2 = color_gradient(c1=base_colors['white'], c2=base_colors['blue'])
rwb = lambda n: (c1(n).T*(n>0) + c2(-n).T*(n<=0)).T

disc = SuperEllipse(ax=0.5, ay=0.5, n=2.0)
class ColorGBOrient(ColorConn):
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
        # we can rely on the superclass to calculate basic bond order parameters and
        # thus avoid repeating sensitive code.
        psi6 = self.psi
        c6 = self.con

        pts = self.snap.particles.position

        gyr = gyration_tensor(pts=pts)
        self.eps = ellipticity(gyr=gyr)

        self.is_defect = c6 < 0.95
        evals, evecs = np.linalg.eigh(gyr)
        pts_transformed = evecs.T @ pts.T
        d_x = np.sqrt(pts_transformed[-2]**2/evals[-2])
        d_y = np.sqrt(pts_transformed[-1]**2/evals[-1])

        self.ci = self.is_defect*(1.0-np.sqrt(d_x**2+d_y**2).clip(0,2))

        self.weights = 2.0-np.sqrt(d_x**2+d_y**2).clip(0,2)

        gyr = gyration_tensor(pts=pts[self.is_defect],
                              weights=self.weights[self.is_defect])
        eps = ellipticity(gyr=gyr)
        self.psi_gb = (1-np.abs(psi6.mean()))*np.exp(1j*np.angle(eps))

    def state_string(self, snap=None):
        """
        Customizing the state string by printing the ellipticity of the biggest
        crystal cluster in addition to the local (C6) and global (ψ6) order.
        """
        if snap is not None: self.snap=snap

        psig, gamma = 1-np.abs(self.psi_gb), np.angle(self.psi_gb)
        c, phi = 1-np.abs(self.eps), np.angle(self.eps)

        s1 = f"$\\psi_g = {psig:.2f}$\t$2\\gamma = {gamma/np.pi:.2f}\\pi$"
        s2 = f"$c_g = {c:.2f}$\t$2\\phi = {phi/np.pi:.2f}\\pi$"

        # s1 = f"$\\phi_{{gb}} = {np.abs(self.psi_gb):.2f}\\exp[{np.angle(self.psi_gb)/np.pi:.2f}i\\pi]$"
        # s2 = f"$\\varepsilon = {np.abs(self.eps):.2f}\\exp[{np.angle(self.eps)/np.pi:.2f}i\\pi]$"

        return f"{s1}\n{s2}"

if __name__ == "__main__":
    from render import render_npole,animate
    import matplotlib.pyplot as plt

    plt.style.use('dark_background')
    supfig, axs = plt.subplots(2,3, figsize=(7,4), dpi=500)

    # define an extended figure making function which places the
    # principal moments of the biggest crystal cluster overtop
    # the rendered image.
    def figure_maker(snap):
        style = ColorGBOrient(dark=True)
        style.snap = snap
        fig,ax = render_npole(snap, style=style, PEL='contour',
                              dark=True, figsize=5, dpi=500)
        pts = style.snap.particles.position[style.is_defect]
        weights = style.weights[style.is_defect]

        com = np.sum(pts*weights[:,np.newaxis],axis=0)/np.sum(weights)
        gyr = gyration_tensor(pts=pts, weights=weights)
        plot_principal_axes(gyr=gyr, com=com, ax=ax, color='orange', lw=2.5)

        pts = style.snap.particles.position
        com = pts.mean(axis=0)
        gyr = gyration_tensor(pts=pts)
        plot_principal_axes(gyr=gyr, com=com, ax=ax, color='green', lw=2.5)

        return fig, ax

    # now run the figure maker through animate for each file
    for f in ['opole5.gsd','opole4.gsd','qpole2.gsd','qpole1.gsd']: 
        with gsd.hoomd.open(f, "r") as traj:
            animate(traj, figure_maker=figure_maker,
                    outpath=f"GB-{f[:-4]}.mp4", fps=10, codec='mpeg4')
