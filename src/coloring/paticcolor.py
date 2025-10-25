# -*- coding: utf-8 -*-
"""
Color schemes for nematic and tetratic orientational order.

Provides simple coloring classes that map  p-atic magnitude, phase,
defects and global summaries into RGB colors. Classes support
dark/light themes.
"""

import numpy as np
import gsd.hoomd
from matplotlib.cm import hsv as hsv_map

from visuals import SuperEllipse
from calc import local_patic, global_patic, neighbors, stretched_neighbors, quat_to_angle
from coloring import base_colors, color_gradient, ColorBase

# base color functions
_white_red = color_gradient(c1=base_colors['white'], c2=base_colors['red'])
_grey_red = color_gradient(c1=base_colors['grey'], c2=base_colors['red'])
_white_orange = color_gradient(c1=base_colors['white'], c2=base_colors['orange'])
_grey_orange = color_gradient(c1=base_colors['grey'], c2=base_colors['orange'])

_rainbow = lambda a: hsv_map(a).clip(0, 1)

# default geometry
_default_sphere = SuperEllipse(ax=0.5, ay=0.5, n=2.0)


class ColorByS2(ColorBase):
    """Color particles by local nematic magnitude (S2).

    :param shape: particle geometry
    :type shape: SuperEllipse
    :param dark: use dark theme if True
    :type dark: bool
    """

    def __init__(self, shape: SuperEllipse = _default_sphere, dark: bool = True):
        super().__init__(dark=dark)
        self._shape = shape
        self._c = _white_red if dark else _grey_red

    def calc_state(self):
        """Calculate both global and local nematic order parameters."""
        angles = quat_to_angle(self.snap.particles.orientation)
        self.ori = np.exp(1j * angles)
        
        self.nem_g = global_patic(angles, p=2)

        # Local nematic order around each particle
        pts = self.snap.particles.position
        self.nei = neighbors(pts, neighbor_cutoff=6*self._shape.ax)
        # snei = stretched_neighbors(pts, angles, rx=self._shape.ax, ry=self._shape.ay, neighbor_cutoff=2.6)
        # nnei = snei@(snei.T)
        # nnei[snei] = True
        # nnei[np.eye(nnei.shape[0], dtype=bool)] = False
        # self.nei = snei

        self.nem_l = local_patic(angles, self.nei, p=2)
        self.ci = np.abs(self.nem_l)

    # Use ColorBase.local_colors by default (ci is set in calc_state)

    def state_string(self, snap: gsd.hoomd.Frame = None):
        """
        :return: LaTeX-formatted summary string. i.e. ":math:`\\langle S_2\\rangle = 0.00`".
        :rtype: str
        """
        if snap is not None: self.snap = snap
        nem_l = np.abs(np.mean(self.nem_l))
        return f'$\\langle S_2\\rangle = {nem_l:.2f}$'


class ColorS2Phase(ColorByS2):
    """Color particles by the phase angle of the local p-atic director using a rainbow wheel. 

    :see: ColorByS2
    :param shift: Phase offset in color mapping
    :type shift: float
    """

    def __init__(self, shape: SuperEllipse = _default_sphere, dark: bool = True, shift: float = 0.0):
        super().__init__(shape=shape, dark=dark)
        self._shift = shift
        # use rainbow mapping
        self._c = lambda x: _rainbow(x)

    def calc_state(self):
        """Calculate both global and local nematic order parameters."""
        super().calc_state()
        self.ci = ((np.angle(self.nem_l) + np.pi) / (2 * np.pi) + self._shift) % 1.0
    # Use ColorBase.local_colors by default (ci is set in calc_state)


class ColorByS2g(ColorByS2):
    """Color all particles uniformly by global nematic order (S2).

    :see: ColorByS2
    """

    def calc_state(self):
        """Calculate both global and local nematic order parameters."""
        super().calc_state()
        self.ci = np.array([np.abs(self.nem_g)]*self.num_pts)
    # Use ColorBase.local_colors by default (ci is set in calc_state)

    def state_string(self, snap: gsd.hoomd.Frame = None):
        """
        :return: LaTeX-formatted summary string. i.e. ":math:`S_{2,g} = 0.00`".
        :rtype: str
        """
        if snap is not None: self.snap = snap
        return f'$S_{{2,g}} = {np.abs(self.nem_g):.2f}$'



class ColorByT4(ColorBase):
    """Color particles by local tetratic magnitude (T4) using orange scale.

    :param shape: particle geometry
    :type shape: SuperEllipse
    :param dark: use dark theme if True
    :type dark: bool
    """

    def __init__(self, shape: SuperEllipse = _default_sphere, dark: bool = True):
        super().__init__(dark=dark)
        self._shape = shape
        self._c = _white_orange if dark else _grey_orange

    def calc_state(self):
        """Calculate both global and local nematic order parameters."""
        angles = quat_to_angle(self.snap.particles.orientation)
        self.ori = np.exp(1j * angles)
        
        # Global nematic order and director
        self.tet_g = global_patic(angles, p=4)

        # Local nematic order around each particle
        pts = self.snap.particles.position
        self.nei = neighbors(pts, neighbor_cutoff=6*self._shape.ax)

        self.tet_l = local_patic(angles, self.nei, p=4)
        self.ci = np.abs(self.tet_l)

    # Use ColorBase.local_colors by default (ci is set in calc_state)
    
    def state_string(self, snap: gsd.hoomd.Frame = None):
        """
        :return: LaTeX-formatted summary string. i.e. ":math:`\\langle T_4\\rangle = 0.00`".
        :rtype: str
        """
        if snap is not None: self.snap = snap
        tet_l = np.abs(np.mean(self.tet_l))
        return f'$\\langle T_4\\rangle = {tet_l:.2f}$'


class ColorByT4g(ColorByT4):
    """Color all particles uniformly by global tetratic order (T4).

    :see: ColorByT4
    """

    def calc_state(self):
        """Calculate both global and local nematic order parameters."""
        super().calc_state()
        self.ci = np.array([np.abs(self.tet_g)]*self.num_pts)
    # Use ColorBase.local_colors by default (ci is set in calc_state)

    def state_string(self, snap: gsd.hoomd.Frame = None):
        """
        :return: LaTeX-formatted summary string. i.e. ":math:`T_{4,g} = 0.00`".
        :rtype: str
        """
        if snap is not None: self.snap = snap
        return f'$T_{{4,g}} = {np.abs(self.tet_g):.2f}$'



# ############################################################################################################
# # TESTING AND DEMONSTRATION
# ############################################################################################################

if __name__ == "__main__":
    
    # test code
    white = base_colors['white']
    
    from calc import stretched_neighbors

    class neicolor(ColorByS2):
        def __init__(self, shape = _default_sphere, dark = True, ptcl = 10):
            super().__init__(shape, dark)
            self._c = lambda n: _rainbow(n)
            self._i = ptcl

        def calc_state(self):
            super().calc_state()
            pts = self.snap.particles.position
            ang = quat_to_angle(self.snap.particles.orientation)
            nei = stretched_neighbors(pts, ang, rx=self._shape.ax, ry=self._shape.ay, neighbor_cutoff=2.8)
            self.nei = nei


        def local_colors(self, snap: gsd.hoomd.Frame = None):
            if snap is not None: self.snap = snap
            col = np.array([white]*self.snap.particles.N)
            col[self._i] = self._c(np.zeros(1))
            for n in range(1, 6):
                nn = self.nth_neighbors(self.nei, n=n)
                col[nn[self._i]] = self._c(np.array([(n+1)/10]))
            return col

        @classmethod
        def nth_neighbors(cls, nei, n=1):
            
            n_nei = nei
            old_nei = np.logical_or(nei, np.eye(nei.shape[0], dtype=bool))
            for _ in range(n-1):
                new_nei = (nei@n_nei)

                new_nei[old_nei] = False
                old_nei[new_nei] = True                

                n_nei = new_nei

            return n_nei


    # a = np.zeros((10,10))==1
    # for i,j in [(1,2),(2,4),(3,7),(4,8),(5,9)]:
    #     a[i,j] = True
    #     a[j,i] = True
    
    # for n in range(1,4):
    #     print(f"{n}-th neighbors:")
    #     print(neicolor.nth_neighbors(a, n=n).astype(int))
    #     print()

    # # make example movies
    # from render import render_npole

    # frame = gsd.hoomd.open("../render/test-rect1.gsd", mode='r')[1200]
    # rect = SuperEllipse(ax=1.0, ay=0.5, n=20)
    # style = neicolor(shape=rect, dark=False, ptcl=250)

    # fig,ax = render_npole(frame, style, figsize=4, dpi=200)
    # fig.savefig('test-nei-coloring.jpg',bbox_inches='tight')

    from render import render_npole, render_sphere, animate
    import traceback

    try:
        # Helper to render a rectangular trajectory with a given style
        def _make_rect_movie(gsd_path, outpath, style, fps=10, codec='mpeg4', istart=500, iend=1500, istride=10, sphere=False):
            try:
                frames = gsd.hoomd.open(gsd_path, mode='r')
            except Exception as _e:
                print(f"Could not open {gsd_path}: {_e}")
                traceback.print_exc()
                return
            sel = frames[istart:iend:istride]
            if sphere:
                L0 = frames[0].configuration.box[0]
                figure_maker = lambda snap: render_sphere(snap, style=style, view_dir=None, view_dist=100, dark=True, figsize=4, dpi=300, L=L0)
            else:
                figure_maker = lambda snap: render_npole(snap, style=style, PEL='contour', dark=True, figsize=4, dpi=300)
            animate(sel, outpath=outpath, figure_maker=figure_maker, fps=fps, codec=codec)

        # Create movies for the p-atic colorings
        for rect_gsd in ['../tests/test-rect1.gsd', '../tests/test-rect2.gsd']:
            if 'rect1' in rect_gsd: out = 'rect1'
            elif 'rect2' in rect_gsd: out = 'rect2'
            # ColorByS2
            style = ColorByS2(shape=SuperEllipse(ax=1.0, ay=0.5, n=20.0))
            _make_rect_movie(rect_gsd, f'../tests/s2-{out}.mp4', style)
            _make_rect_movie(rect_gsd, f'../docs/source/_static/s2-{out}.webm', style, codec='libvpx')

            # ColorByS2g
            style = ColorByS2g(shape=SuperEllipse(ax=1.0, ay=0.5, n=20.0))
            _make_rect_movie(rect_gsd, f'../tests/s2g-{out}.mp4', style)
            _make_rect_movie(rect_gsd, f'../docs/source/_static/s2g-{out}.webm', style, codec='libvpx')

            # ColorS2Phase
            style = ColorS2Phase(shape=SuperEllipse(ax=1.0, ay=0.5, n=20.0))
            _make_rect_movie(rect_gsd, f'../tests/s2p-{out}.mp4', style)
            _make_rect_movie(rect_gsd, f'../docs/source/_static/s2p-{out}.webm', style, codec='libvpx')

            # ColorByT4
            style = ColorByT4(shape=SuperEllipse(ax=1.0, ay=0.5, n=20.0))
            _make_rect_movie(rect_gsd, f'../tests/t4-{out}.mp4', style)
            _make_rect_movie(rect_gsd, f'../docs/source/_static/t4-{out}.webm', style, codec='libvpx')

            # ColorByT4g
            style = ColorByT4g(shape=SuperEllipse(ax=1.0, ay=0.5, n=20.0))
            _make_rect_movie(rect_gsd, f'../tests/t4g-{out}.mp4', style)
            _make_rect_movie(rect_gsd, f'../docs/source/_static/t4g-{out}.webm', style, codec='libvpx')

    except Exception as e:
        print('Agent movie creation skipped due to error:', e)
        traceback.print_exc()