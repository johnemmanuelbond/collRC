# -*- coding: utf-8 -*-
"""
Color schemes for highlighting various kinds of defects
"""

from __future__ import annotations

import numpy as np
import gsd.hoomd

from calc.locality import DEFAULT_CUTOFF
from calc import crystal_connectivity
from visuals import SuperEllipse
from coloring import base_colors, color_gradient, ColorBase
from coloring import ColorS2, ColorConn


# base color functions
_white_red = color_gradient(c1=base_colors['white'], c2=base_colors['red'])
_grey_red = color_gradient(c1=base_colors['grey'], c2=base_colors['red'])
_cyan = base_colors['cyan']
_gold = base_colors['gold']

# default geometry
_default_sphere = SuperEllipse(ax=0.5, ay=0.5, n=2.0)

class ColorS2Defects(ColorS2):
    """Color nematic defects cyan.

    Highlights particles that disagree with their local nematic director and
    paints them cyan while delegating non-defect coloring to an optional
    background ColorBase instance.

    :param shape: particle geometry, defaults to sphere with diameter 1.0
    :type shape: :py:class:`SuperEllipse <visuals.shapes.SuperEllipse>`
    :param dark: dark theme flag
    :type dark: bool, optional
    :param bgColor: optional ColorBase instance providing background colors
    :type bgColor: ColorBase | None, optional
    :ivar defects: Boolean mask of defective particles.
    :type defects: ndarray
    """

    def __init__(self, shape: SuperEllipse = _default_sphere, dark: bool = True, bgColor:ColorBase=None):
        super().__init__(shape=shape, dark=dark)
        self._c = _cyan
        if bgColor is None:
            self._bg = ColorBase(dark=dark)
        else:
            self._bg = bgColor

    def calc_state(self):
        """
        Calls the parent :meth:`ColorS2.calc_state <coloring.paticcolor.ColorS2.calc_state>` to compute local nematic
        quantities, then marks particles as defects when their local
        orientation misaligns with the local director.
        """
        super().calc_state()

        # self.defects = np.abs(self.nem_l) < 0.5

        misorient = np.real(self.ori*self.ori*np.conjugate(self.nem_l)) < np.abs(self.nem_l)/2

        self.defects = misorient

        # is_nem = np.abs(self.nem_l) > 0.5
        # self.defects = np.logical_and(is_nem, misorient)

        # self.defects = np.real(self.ori*self.ori*np.conjugate(self.nem_g))/np.abs(self.nem_g) < 0.5

    def local_colors(self, snap: gsd.hoomd.Frame = None):
        """Return RGBA colors mapping the background colorbase with defects in cyan.

        :return: (N,4) RGBA array
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
        new_str = f'$N_{{S2}} = {def_percent:.0f}\\%$'
        if old_str == "": return new_str
        return f'{old_str}\n{new_str}'


class ColorC6Defects(ColorConn):
    """Color sixfold defects :math:`(C_{6} < 1)` to highlight defects.

    This class wraps a connectivity-based color style and overrides the
    per-particle colors to mark defective sites in red while allowing a
    configurable background color base.

    :param shape: particle geometry, defaults to sphere with diameter 1.0
    :type shape: :py:class:`SuperEllipse <visuals.shapes.SuperEllipse>`
    :param surface_normal: optional surface normal for projected calculations
    :type surface_normal: callable | None
    :param nei_cutoff: dimensionless neighbor cutoff distance, defaults to halfway between first and second neighbor shells
    :type nei_cutoff: float | None, optional
    :param periodic: whether to use periodic boundary conditions, defaults to False
    :type periodic: bool, optional
    :param norm: normalization factor for connectivity calculation, defaults to 6
    :type norm: float | None, optional
    :param crystallinity_threshold: threshold below which particle bonds are marked as defective, defaults to 0.32
    :type crystallinity_threshold: float, optional
    :param dark: dark theme flag
    :type dark: bool, optional
    :param bgColor: optional ColorBase instance providing background colors
    :type bgColor: ColorBase | None, optional
    :ivar defects: Boolean mask of defective particles.
    :type defects: ndarray
    :ivar con: Per-particle connectivity values inherited from :py:class:`ColorByConn <coloring.bondcolor.ColorByConn>`
    :type con: ndarray
    """

    def __init__(self, shape: SuperEllipse = _default_sphere,
                 surface_normal: callable = None,
                 nei_cutoff: float = DEFAULT_CUTOFF, periodic: bool = False,
                 norm: float = 6, crystallinity_threshold: float = 0.32,
                 dark: bool = True, bgColor:ColorBase=None):
        super().__init__(shape=shape, order=6, dark=dark, surface_normal=surface_normal, periodic=periodic, calc_3d=False, crystallinity_threshold=crystallinity_threshold, nei_cutoff=nei_cutoff, norm=norm)
        # Set color mapping function based on background
        self._c = _white_red if dark else _grey_red
        if bgColor is None:
            self._bg = ColorBase(dark=dark)
        else:
            self._bg = bgColor

    def calc_state(self):
        """
        Runs the parent's connectivity calculation then marks particles as
        defective when connectivity falls below a heuristic threshold.
        """
        super().calc_state()
        self.defects = self.con[:self.num_pts]<0.95

    def local_colors(self, snap: gsd.hoomd.Frame = None):
        """Return per-particle RGBA colors highlighting defective particles in red.

        Mapping: for dark backgrounds this uses a white->red gradient; for light backgrounds a grey->red gradient. 

        :return: (N,4) RGBA array
        :rtype: ndarray
        """
        if snap is not None: self.snap = snap
        defect = (2*(1-self.con[:self.num_pts])).clip(0,1)[self.defects]
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
        con_g = self.con[:self.num_pts].mean()
        new_str = f'$\\langle 1-C_6\\rangle = {1-con_g:.2f}$'
        if old_str == "": return new_str
        return f'{old_str}\n{new_str}'
    

class ColorC4Defects(ColorConn):
    """Color fourfold defects :math:`(C_{4} < 1)` to highlight defects.

    This variant recalculates connectivity with a tetratic (p=4) normalization
    and a tightened crystallinity threshold to detect fourfold defect sites.

    :param shape: particle geometry
    :type shape: :py:class:`SuperEllipse <visuals.shapes.SuperEllipse>`
    :param surface_normal: optional surface normal for projected calculations
    :type surface_normal: callable | None
    :param nei_cutoff: dimensionless neighbor cutoff distance, defaults to 2.6
    :type nei_cutoff: float | None, optional
    :param periodic: periodic boundary conditions flag
    :type periodic: bool, optional
    :param norm: normalization order for connectivity calculation, defaults to 4
    :type norm: float | None, optional
    :param crystallinity_threshold: threshold below which particle bonds are marked as defective, defaults to 0.5
    :type crystallinity_threshold: float, optional
    :param dark: dark theme flag
    :type dark: bool, optional
    :param bgColor: optional ColorBase instance providing background colors
    :type bgColor: ColorBase | None, optional
    :ivar defects: Boolean mask of defective particles.
    :type defects: ndarray
    """

    def __init__(self, shape: SuperEllipse = _default_sphere,
                 surface_normal: callable = None,
                 nei_cutoff: float = 2.6, periodic: bool = False,
                 norm: float = 4, crystallinity_threshold: float = 0.5,
                 dark: bool = True, bgColor:ColorBase=None):
        super().__init__(shape=shape, order=4, dark=dark, surface_normal=surface_normal, periodic=periodic, calc_3d=False, crystallinity_threshold=crystallinity_threshold, nei_cutoff=nei_cutoff, norm=norm)
        # Set color mapping function based on background
        self._c = _gold
        if bgColor is None:
            self._bg = ColorBase(dark=dark)
        else:
            self._bg = bgColor

    def calc_state(self):
        """
        Recomputes connectivity using :py:meth:`crystal_connectivity <calc.bond_order.crystal_connectivity>` with
        :code:`norm=4` and a tightened crystallinity threshold before marking
        defective sites.
        """
        super().calc_state()
        self.defects = self.con[:self.num_pts]<0.95

    def local_colors(self, snap: gsd.hoomd.Frame = None):
        """Return per-particle RGBA colors highlighting defective particles in gold.

        :return: (N,4) RGBA array
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
        new_str = f'$N_{{C4}} = {def_percent:.0f}\\%$'
        if old_str == "": return new_str
        return f'{old_str}\n{new_str}'


# ############################################################################################################
# # TESTING AND DEMONSTRATION
# ############################################################################################################

if __name__ == "__main__":
    
    from render import render_npole, render_sphere, animate
    from coloring import ColorBase, ColorEta0 
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
                figure_maker = lambda snap: render_sphere(snap, style=style, dark=True, figsize=5, dpi=500, L=L0)
            else:
                figure_maker = lambda snap: render_npole(snap, style=style, PEL='contour', dark=True, figsize=5, dpi=500)
            
            try:
                animate(sel, outpath=outpath, figure_maker=figure_maker, fps=fps, codec=codec)
            except Exception as _e:
                print(f"Could not make movie {outpath}: {_e}")
                traceback.print_exc()
                return

        # # C6 defects on q-pole control
        # bg_style = ColorBase()
        # style = ColorC6Defects(bgColor=bg_style)
        # _make_movie('../tests/test-control.gsd', '../tests/c6d-qpole.mp4', style, istride=100)
        # _make_movie('../tests/test-control.gsd', '../docs/source/_static/c6d-qpole.webm', style, codec='libvpx', istride=100)

        # # C6 defects on sphere (small sample)
        # bg_style = ColorConn(order=6, periodic=False)
        # style = ColorC6Defects(bgColor=bg_style)
        # _make_movie('../tests/test-sphere.gsd', '../tests/c6d-sphere.mp4', style, sphere=True, iend=100, istride=2)
        # _make_movie('../tests/test-sphere.gsd', '../docs/source/_static/c6d-sphere.webm', style, sphere=True, codec='libvpx', iend=100, istride=2)

        # C4 defects on rectangle (use tightened frame window)
        bg_style = ColorConn(shape=SuperEllipse(ax=1.0, ay=0.5, n=20.0), order=4, periodic=True, nei_cutoff=2.6)
        style = ColorC4Defects(shape=SuperEllipse(ax=1.0, ay=0.5, n=20.0), bgColor=bg_style, periodic=True)
        _make_movie('../tests/test-rect1.gsd', '../tests/c4d-rect1.mp4', style, istart=500, iend=1500)
        _make_movie('../tests/test-rect1.gsd', '../docs/source/_static/c4d-rect1.webm', style, codec='libvpx', istart=500, iend=1500)

        bg_style = ColorConn(shape=SuperEllipse(ax=1.0, ay=0.5, n=20.0), order=4, periodic=True, nei_cutoff=2.6)
        style = ColorC4Defects(shape=SuperEllipse(ax=1.0, ay=0.5, n=20.0), bgColor=bg_style, periodic=True)
        _make_movie('../tests/test-rect2.gsd', '../tests/c4d-rect2.mp4', style, istart=500, iend=1500)
        _make_movie('../tests/test-rect2.gsd', '../docs/source/_static/c4d-rect2.webm', style, codec='libvpx', istart=500, iend=1500)

        # # S2 defects on rectangle
        # bg_style = ColorEta0(shape=SuperEllipse(ax=1.0, ay=0.5, n=20.0))
        # style = ColorS2Defects(shape=SuperEllipse(ax=1.0, ay=0.5, n=20.0), bgColor=bg_style)
        # _make_movie('../tests/test-rect1.gsd', '../tests/s2d-rect1.mp4', style, istart=500, iend=1500)
        # _make_movie('../tests/test-rect1.gsd', '../docs/source/_static/s2d-rect1.webm', style, codec='libvpx', istart=500, iend=1500)

        # bg_style = ColorEta0(shape=SuperEllipse(ax=1.0, ay=0.5, n=20.0))
        # style = ColorS2Defects(shape=SuperEllipse(ax=1.0, ay=0.5, n=20.0), bgColor=bg_style)
        # _make_movie('../tests/test-rect2.gsd', '../tests/s2d-rect2.mp4', style, istart=500, iend=1500)
        # _make_movie('../tests/test-rect2.gsd', '../docs/source/_static/s2d-rect2.webm', style, codec='libvpx', istart=500, iend=1500)

    except Exception as e:
        print('Agent movie creation skipped due to error:', e)
        traceback.print_exc()