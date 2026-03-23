import numpy as np
import gsd.hoomd
from matplotlib.cm import hsv as hsv_map
rainbow = lambda a: hsv_map(a).clip(0, 1)

from visuals import SuperEllipse
from calc import quat_to_angle, stretched_neighbors

from coloring import ColorBase, base_colors
white = base_colors['white']

class ColorNNN(ColorBase):
    """Color particles by their nth nearest neighbors."""
    def __init__(self, shape, dark = True, ptcl = 10, nth_neighbors=3):
        super().__init__(shape=shape, dark=dark)
        self._c = lambda n: rainbow(n)
        self._i = ptcl
        self._nn = nth_neighbors

    @classmethod
    def nth_neighbors(cls, nei, n=1):
        """Return boolean array of nth neighbors given a boolean array of 1st neighbors."""
        n_nei = nei
        old_nei = np.logical_or(nei, np.eye(nei.shape[0], dtype=bool))
        for _ in range(n-1):
            new_nei = (nei@n_nei)
            new_nei[old_nei] = False
            old_nei[new_nei] = True                
            n_nei = new_nei

        return n_nei

    def calc_state(self):
        """find the neighbor matrix of each particle stretched by the shape parameters."""
        super().calc_state()
        pts = self.snap.particles.position
        ang = quat_to_angle(self.snap.particles.orientation)
        nei = stretched_neighbors(pts, ang, rx=self._shape.ax, ry=self._shape.ay, neighbor_cutoff=2.8)
        self.nei = nei

    def local_colors(self, snap: gsd.hoomd.Frame = None):
        """
        Assign colors based on nth nearest neighbors.
        Notably this scheme `can't` use the :py:attr:`ci` attribute of most color classes,
        so we overwrite the ``local_colors`` method and return an RGBA array."""
        if snap is not None: self.snap = snap
        col = np.array([white]*self.snap.particles.N)
        col[self._i] = self._c(np.zeros(1))
        for n in range(1, self._nn+1):
            nn = self.nth_neighbors(self.nei, n=n)
            col[nn[self._i]] = self._c(np.array([(n+1)/(self._nn+2)]))
        return col
    
if __name__ == "__main__":
    from render import render_npole
    import matplotlib.pyplot as plt

    shape = SuperEllipse(ax=1.0, ay=0.5, n=20.0)
    plt.style.use('dark_background')
    supfig, axs = plt.subplots(2,3, figsize=(7,4), dpi=500)

    with gsd.hoomd.open("rect2.gsd", "r") as traj:

        for frame,axs_j in zip([0, 50], axs):
            for ptcl, ax_i in zip([101, 420, 203], axs_j):
                style = ColorNNN(shape=shape, ptcl=ptcl, dark=True)
                fig,ax = render_npole(traj[frame], style=style, PEL='contour', dark=True, figsize=5, dpi=500)
                fig.canvas.draw()

                # Extract RGB data from matplotlib figure
                w, h = fig.canvas.get_width_height()
                buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
                buf.shape = (h, w, 4)  # Height, width, RGBA channels

                plt.close(fig)  # Clean up memory

                # display rgba array on supfig axes
                ax_i.imshow(buf)
                ax_i.axis('off')
                ax_i.set_aspect('equal')

    supfig.savefig("nth-nearest-neighbors.png", bbox_inches='tight')