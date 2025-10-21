"""
Color schemes for highlighting various kinds of defects
"""

from __future__ import annotations

import numpy as np
import gsd.hoomd

from calc import crystal_connectivity
from visuals import SuperEllipse
from .base import base_colors, color_gradient, ColorBase, _gsd_match
from .paticcolor import ColorByS2
from .psicolor import ColorByConn


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
        self._bg.calc_state()
        self.defects = np.abs(self.nem_l)<0.5

    def local_colors(self, snap: gsd.hoomd.Frame):
        """Return RGB colors mapping the background colorbase with defects in cyan.

        :return: (N,3) RGB array
        :rtype: ndarray
        """
        if not _gsd_match(self.snap, snap):
            self.snap = snap
            self.calc_state()

        colors = self._bg.local_colors(snap)
        colors[self.defects] = self._c
        return colors
    
    def state_string(self, snap: gsd.hoomd.Frame):
        """
        :return: LaTeX-formatted summary string. i.e. ":math:`N_{S2}=00%`".
        :rtype: str
        """
        if not _gsd_match(self.snap, snap):
            self.snap = snap
            self.calc_state()
        def_percent = 100*self.defects.mean()
        old_str = self._bg.state_string(snap)
        return f'{old_str}\n$N_{{S2}} = {def_percent:.2f}\%$'


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
        self._bg.calc_state()
        self.defects = self.con<0.95

    def local_colors(self, snap: gsd.hoomd.Frame):
        """Return per-particle RGB colors highlighting defective particles in red.

        Mapping: for dark backgrounds this uses a white->red gradient; for light backgrounds a grey->red gradient. 

        :return: (N,3) array
        :rtype: ndarray
        """
        if not _gsd_match(self.snap, snap):
            self.snap = snap
            self.calc_state()
        defect = (3*(1-self.con)).clip(0,1)[self.defects]

        colors = self._bg.local_colors(snap)
        colors[self.defects] = self._c(defect)
        return colors

    def state_string(self, snap: gsd.hoomd.Frame):
        """
        :return: LaTeX-formatted summary string. i.e. ":math:`\\langle1-C_n\\rangle=0.00`".
        :rtype: str
        """
        if not _gsd_match(self.snap, snap):
            self.snap = snap
            self.calc_state()
        old_str = self._bg.state_string(snap)
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
        self._bg.calc_state()
        if self._is_proj:
            c4 = crystal_connectivity(self.psi, self.nei, phase_rotate=self.rel_rot**self._n, norm=4, crystallinity_threshold=0.5)
        else:
            c4 = crystal_connectivity(self.psi, self.nei, norm=4, crystallinity_threshold=0.5)

        self.defects = c4<0.95

    def local_colors(self, snap: gsd.hoomd.Frame):
        """Return per-particle RGB colors highlighting defective particles in gold.

        :return: (N,3) array
        :rtype: ndarray
        """
        if not _gsd_match(self.snap, snap):
            self.snap = snap
            self.calc_state()

        colors = self._bg.local_colors(snap)
        colors[self.defects] = self._c
        return colors

    def state_string(self, snap: gsd.hoomd.Frame):
        """
        :return: LaTeX-formatted summary string. i.e. ":math:`N_{C4}=00%`".
        :rtype: str
        """
        if not _gsd_match(self.snap, snap):
            self.snap = snap
            self.calc_state()
        
        def_percent = 100*self.defects.mean()
        old_str = self._bg.state_string(snap)
        return f'{old_str}\n$N_{{C4}} = {def_percent:.2f}\%$'


# ############################################################################################################
# # TESTING AND DEMONSTRATION
# ############################################################################################################

if __name__ == "__main__":
    
    # test code
    pass