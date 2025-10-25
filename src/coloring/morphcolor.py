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

    This style computes a single scalar value for the central area
    fraction (:math:`\eta_0`) via :py:meth:`central_eta <calc.morphology.central_eta>` and applies a
    single-parameter color gradient to every particle.

    :param shape: particle geometry
    :type shape: :py:class:`SuperEllipse <visuals.shapes.SuperEllipse>`
    :param dark: use dark theme if True
    :type dark: bool, optional
    :param jac: Jacobian mode passed to :py:meth:`central_eta <calc.morphology.central_eta>`
    :type jac: str, optional
    :ivar eta0: The central area fraction computed from particle positions and box.
    :type eta0: scalar
    :ivar ci: Length-N numpy array containing :py:attr:`eta0` repeated for every particle; used by :py:meth:`ColorBase.local_colors <coloring.ColorBase.local_colors>`.
    :type ci: ndarray
    """

    def __init__(self, shape: SuperEllipse = _default_sphere, dark: bool = True, jac='x'):
        """Constructor"""
        super().__init__(dark=dark)
        self._shape = shape
        self._c = _white_green if dark else _grey_green
        self._jac = jac
        self._ap = shape.area

    def calc_state(self):
        """Caclulate the central area fraction using :py:meth:`central_eta <calc.morphology.central_eta>`."""
        pts = self.snap.particles.position
        box = self.snap.configuration.box
        self.eta0 = central_eta(pts, box, jac=self._jac, ptcl_area=self._ap)
        # expose a numeric array for the base-class mapper
        self.ci = np.array([self.eta0] * self.num_pts)

    # Use ColorBase.local_colors by default (ci is set in calc_state)

    def state_string(self, snap: gsd.hoomd.Frame = None):
        """
        :return: LaTeX-formatted summary string. i.e. ":math:`\\eta_0 = 0.00`".
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