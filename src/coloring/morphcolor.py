# -*- coding: utf-8 -*-
"""
Color schemes for morphology-based reaction coordinate visualizations.
"""

import numpy as np
import gsd.hoomd
from matplotlib.cm import hsv as hsv_map

from visuals import SuperEllipse
from coloring import base_colors, color_gradient, ColorBase
from calc import central_eta

# base color functions
_white_green = color_gradient(c1=base_colors['white'], c2=base_colors['green'])
_grey_green = color_gradient(c1=base_colors['grey'], c2=base_colors['green'])

_rainbow = lambda a: hsv_map(a).clip(0, 1)

# default geometry
_default_sphere = SuperEllipse(ax=0.5, ay=0.5, n=2.0)


class ColorByEta0(ColorBase):
    """Color all particles by the central area fraction.

    :param shape: particle geometry
    :type shape: SuperEllipse
    :param dark: use dark theme if True
    :type dark: bool
    """

    def __init__(self, shape: SuperEllipse = _default_sphere, dark: bool = True, jac='x'):
        super().__init__(dark=dark)
        self._shape = shape
        self._c = _white_green if dark else _grey_green
        self._jac = jac
        self._ap = shape.area

    def calc_state(self):
        """Calculate both global and local nematic order parameters."""
        pts = self.snap.particles.position
        box = self.snap.configuration.box
        self.eta0 = central_eta(pts, box, jac=self._jac, ptcl_area=self._ap)

    def local_colors(self, snap: gsd.hoomd.Frame = None):
        """Return RGB colors mapping local S2 magnitude (white/grey -> red).

        :return: (N,3) RGB array
        :rtype: ndarray
        """
        if snap is not None: self.snap = snap
        return np.array([self._c(np.abs(self.eta0))]*self.snap.particles.N)

    def state_string(self, snap: gsd.hoomd.Frame = None):
        """
        :return: LaTeX-formatted summary string. i.e. ":math:`\\langle S_2\\rangle = 0.00`".
        :rtype: str
        """
        if snap is not None: self.snap = snap
        return f'$\\eta_0 = {self.eta0:.2f}$'

# ############################################################################################################
# # TESTING AND DEMONSTRATION
# ############################################################################################################

if __name__ == "__main__":
    
    # test code
    pass