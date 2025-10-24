# -*- coding: utf-8 -*-
"""
Color schemes for highlighting various kinds of defects
"""

from __future__ import annotations

import numpy as np
import gsd.hoomd

from calc import crystal_connectivity
from visuals import SuperEllipse
from coloring import base_colors, color_gradient, ColorBase
from coloring import ColorByS2, ColorByConn


# base color functions
_white_red = color_gradient(c1=base_colors['white'], c2=base_colors['red'])
_grey_red = color_gradient(c1=base_colors['grey'], c2=base_colors['red'])
_cyan = base_colors['cyan']
_gold = base_colors['gold']

# default geometry
_default_sphere = SuperEllipse(ax=0.5, ay=0.5, n=2.0)

class ColorS2Defects(ColorByS2):
    """Color nematic defects :math:`(S_{2,l} < 0.5)` cyan.

    :param bgColor: an optional ColorBase instance to provide background coloring which the defect colors will supercede.
    :type bgColor: ColorBase | None, optional
    :see: process.paticcolor.ColorByS2
    """

    def __init__(self, shape: SuperEllipse = _default_sphere, dark: bool = True, bgColor:ColorBase=None):
        super().__init__(shape=shape, dark=dark)
        self._c = _cyan
        if bgColor is None:
            self._bg = ColorBase(dark=dark)
        else:
            self._bg = bgColor

    def calc_state(self):
        """Identify nematic defects"""
        super().calc_state()

        # self.defects = np.abs(self.nem_l) < 0.5

        misorient = np.real(self.ori*self.ori*np.conjugate(self.nem_l)) < np.abs(self.nem_l)/2

        self.defects = misorient

        # is_nem = np.abs(self.nem_l) > 0.5
        # self.defects = np.logical_and(is_nem, misorient)

        # self.defects = np.real(self.ori*self.ori*np.conjugate(self.nem_g))/np.abs(self.nem_g) < 0.5



    def local_colors(self, snap: gsd.hoomd.Frame = None):
        """Return RGB colors mapping the background colorbase with defects in cyan.

        :return: (N,3) RGB array
        :rtype: ndarray
        """
        if snap is not None: self.snap = snap
        colors = self._bg.local_colors(snap=self.snap)
        colors[self.defects] = self._c
        return colors

    def state_string(self, snap: gsd.hoomd.Frame = None):
        """
        :return: LaTeX-formatted summary string. i.e. ":math:`N_{S2}=00%`".
        :rtype: str
        """
        if snap is not None: self.snap = snap
        def_percent = 100*self.defects.mean()
        old_str = self._bg.state_string(snap=self.snap)
        return f'{old_str}\n$N_{{S2}} = {def_percent:.0f}\\%$'


class ColorC6Defects(ColorByConn):
    """Color sixfold defects :math:`(C_{6} < 1)` to highight defects

    :param bgColor: an optional ColorBase instance to provide background coloring which the defect colors will supercede.
    :type bgColor: ColorBase | None, optional
    :see: process.psicolor.ColorByConn
    """

    def __init__(self, shape: SuperEllipse = _default_sphere,
                 surface_normal: callable = None,
                 dark: bool = True,
                 bgColor:ColorBase=None):
        super().__init__(shape=shape, order=6, dark=dark, surface_normal=surface_normal)
        # Set color mapping function based on background
        self._c = _white_red if dark else _grey_red
        if bgColor is None:
            self._bg = ColorBase(dark=dark)
        else:
            self._bg = bgColor

    def calc_state(self):
        """Identify C6 defects"""
        super().calc_state()
        self.defects = self.con<0.95

    def local_colors(self, snap: gsd.hoomd.Frame = None):
        """Return per-particle RGB colors highlighting defective particles in red.

        Mapping: for dark backgrounds this uses a white->red gradient; for light backgrounds a grey->red gradient. 

        :return: (N,3) array
        :rtype: ndarray
        """
        if snap is not None: self.snap = snap
        defect = (2*(1-self.con)).clip(0,1)[self.defects]
        colors = self._bg.local_colors(snap=self.snap)
        colors[self.defects] = self._c(defect)
        return colors

    def state_string(self, snap: gsd.hoomd.Frame = None):
        """
        :return: LaTeX-formatted summary string. i.e. ":math:`\\langle1-C_n\\rangle=0.00`".
        :rtype: str
        """
        if snap is not None: self.snap = snap
        old_str = self._bg.state_string(snap=self.snap)
        con_g = self.con.mean()
        return f'{old_str}\n$\\langle 1-C_6\\rangle = {1-con_g:.2f}$'
    

class ColorC4Defects(ColorByConn):
    """Color fourfold defects :math:`(C_{4} < 1)` to highlight defects.

    Notably the class has to recalculate the connectivity :py:meth:`calc.crystal_connectivity` using a different threshold :math:`\\Theta` to identify defects. 

    :param bgColor: an optional ColorBase instance to provide background coloring which the defect colors will supercede.
    :type bgColor: ColorBase | None, optional
    :see: process.psicolor.ColorByConn
    """

    def __init__(self, shape: SuperEllipse = _default_sphere,
                 surface_normal: callable = None,
                 dark: bool = True,
                 bgColor:ColorBase=None):
        super().__init__(shape=shape, order=4, dark=dark, surface_normal=surface_normal)
        # Set color mapping function based on background
        self._c = _gold
        if bgColor is None:
            self._bg = ColorBase(dark=dark)
        else:
            self._bg = bgColor

    def calc_state(self):
        """Identify C4 defects"""
        super().calc_state()
        
        if self._is_proj:
            c4 = crystal_connectivity(self.psi, self.nei, phase_rotate=self.rel_rot**self._n, norm=4, crystallinity_threshold=0.5)
        else:
            c4 = crystal_connectivity(self.psi, self.nei, norm=4, crystallinity_threshold=0.5)

        self.defects = c4<0.95

    def local_colors(self, snap: gsd.hoomd.Frame = None):
        """Return per-particle RGB colors highlighting defective particles in gold.

        :return: (N,3) array
        :rtype: ndarray
        """
        if snap is not None: self.snap = snap
        colors = self._bg.local_colors(snap=self.snap)
        colors[self.defects] = self._c
        return colors

    def state_string(self, snap: gsd.hoomd.Frame = None):
        """
        :return: LaTeX-formatted summary string. i.e. ":math:`N_{C4}=00%`".
        :rtype: str
        """
        if snap is not None: self.snap = snap
        def_percent = 100*self.defects.mean()
        old_str = self._bg.state_string(snap=self.snap)
        return f'{old_str}\n$N_{{C4}} = {def_percent:.0f}\\%$'


# ############################################################################################################
# # TESTING AND DEMONSTRATION
# ############################################################################################################

if __name__ == "__main__":
    
    from render import render_npole, render_sphere, animate
    from coloring import ColorByEta0, ColorBase
    import traceback

    try:
        def _make_movie(gsd_path, outpath, style, fps=10, codec='mpeg4', istart=0, iend=-1, istride=10, sphere=False):
            try:
                frames = gsd.hoomd.open(gsd_path, mode='r')
            except Exception as _e:
                print(f"Could not open {gsd_path}: {_e}")
                traceback.print_exc()
                return
            sel = frames[istart:iend:istride]
            if sphere:
                L0 = frames[0].configuration.box[0]
                figure_maker = lambda snap: render_sphere(snap, style=style, dark=True, figsize=4, dpi=300, L=L0)
            else:
                figure_maker = lambda snap: render_npole(snap, style=style, PEL='contour', dark=True, figsize=4, dpi=300)
            animate(sel, outpath=outpath, figure_maker=figure_maker, fps=fps, codec=codec)

        # # C6 defects on q-pole control
        # bg_style = ColorBase()
        # style = ColorC6Defects(bgColor=bg_style)
        # _make_movie('../tests/test-control.gsd', '../tests/c6d-qpole.mp4', style, istride=100)
        # _make_movie('../tests/test-control.gsd', '../docs/source/_static/c6d-qpole.webm', style, codec='libvpx', istride=100)

        # # C6 defects on sphere (small sample)
        # bg_style = ColorByConn()
        # style = ColorC6Defects(bgColor=bg_style)
        # _make_movie('../tests/test-sphere.gsd', '../tests/c6d-sphere.mp4', style, sphere=True, iend=100, istride=2)
        # _make_movie('../tests/test-sphere.gsd', '../docs/source/_static/c6d-sphere.webm', style, sphere=True, codec='libvpx', iend=100, istride=2)

        # C4 defects on rectangle (use tightened frame window)
        bg_style = ColorByConn(shape=SuperEllipse(ax=1.0, ay=0.5, n=20), order=4)
        style = ColorC4Defects(shape=SuperEllipse(ax=1.0, ay=0.5, n=20), bgColor=bg_style)
        _make_movie('../tests/test-rect.gsd', '../tests/c4d-rect.mp4', style, istart=500, iend=1500)
        _make_movie('../tests/test-rect.gsd', '../docs/source/_static/c4d-rect.webm', style, codec='libvpx', istart=500, iend=1500)

        # S2 defects on rectangle
        bg_style = ColorByEta0(shape=SuperEllipse(ax=1.0, ay=0.5, n=2.0))
        style = ColorS2Defects(shape=SuperEllipse(ax=1.0, ay=0.5, n=2.0), bgColor=bg_style)
        _make_movie('../tests/test-rect.gsd', '../tests/s2d-rect.mp4', style, istart=500, iend=1500)
        _make_movie('../tests/test-rect.gsd', '../docs/source/_static/s2d-rect.webm', style, codec='libvpx', istart=500, iend=1500)

        # Additional rectangle movie variants (new test rect1/rect2)
        bg_style = ColorByConn(shape=SuperEllipse(ax=1.0, ay=0.5, n=20), order=4)
        style = ColorC4Defects(shape=SuperEllipse(ax=1.0, ay=0.5, n=20), bgColor=bg_style)
        _make_movie('../tests/test-rect1.gsd', '../tests/c4d-rect1.mp4', style, istart=500, iend=1500)
        _make_movie('../tests/test-rect1.gsd', '../docs/source/_static/c4d-rect1.webm', style, codec='libvpx', istart=500, iend=1500)

        bg_style = ColorByEta0(shape=SuperEllipse(ax=1.0, ay=0.5, n=2.0))
        style = ColorS2Defects(shape=SuperEllipse(ax=1.0, ay=0.5, n=2.0), bgColor=bg_style)
        _make_movie('../tests/test-rect2.gsd', '../tests/s2d-rect2.mp4', style, istart=500, iend=1500)
        _make_movie('../tests/test-rect2.gsd', '../docs/source/_static/s2d-rect2.webm', style, codec='libvpx', istart=500, iend=1500)

    except Exception as e:
        print('Agent movie creation skipped due to error:', e)
        traceback.print_exc()