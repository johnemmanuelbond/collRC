# -*- coding: utf-8 -*-
"""
Color schemes for morphology-based reaction coordinate visualizations.
"""

import numpy as np
import gsd.hoomd
from matplotlib.cm import hsv as hsv_map

from visuals import SuperEllipse
from coloring import base_colors, color_gradient, ColorBase
from calc import central_eta, gyration_radius, gyration_tensor, circularity, ellipticity

# base color functions
_white_green = color_gradient(c1=base_colors['white'], c2=base_colors['green'])
_grey_green = color_gradient(c1=base_colors['grey'], c2=base_colors['green'])
_white_blue = color_gradient(c1=base_colors['white'], c2=base_colors['blue'])
_grey_blue = color_gradient(c1=base_colors['grey'], c2=base_colors['blue'])
_white_lime = color_gradient(c1=base_colors['white'], c2=base_colors['lime'])
_grey_lime = color_gradient(c1=base_colors['grey'], c2=base_colors['lime'])


_rainbow = lambda a: hsv_map(a).clip(0, 1)

# default geometry
_default_sphere = SuperEllipse(ax=0.5, ay=0.5, n=2.0)


class ColorEta0(ColorBase):
    """Color all particles by the central area fraction.

    This style computes a single scalar value for the central area
    fraction (:math:`\\eta_0`) via :py:meth:`central_eta <calc.morphology.central_eta>` and applies a
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

    def __init__(self, shape: SuperEllipse = None, dark: bool = True, jac='x'):
        """Constructor"""
        super().__init__(dark=dark)
        self._shape = _default_sphere if shape is None else shape
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


class ColorRg(ColorBase):
    """Color all particles by the ensemble radius of gyration.

    This style computes a single scalar value for the ensemble radius of
    gyration (:math:`R_g`) via :py:meth:`gyration_radius <calc.morphology.gyration_radius>` and applies a
    single-parameter color gradient to every particle.

    Alternatively, and more expensively, users can pass `calc_tensor=True` to involve
    the full gyration tensor via :py:meth:`gyration_tensor <calc.morphology.gyration_tensor>`
    eigenvalues in the calculation.

    Particles are colored according to the global radius of gyration, normalized by a user-defined factor :py:attr:`rg_norm`: :py:attr:`ci = 2 - rg/rg_norm`.

    :param shape: particle geometry, defaults to a sphere of diameter 1.0
    :type shape: :py:class:`SuperEllipse <visuals.shapes.SuperEllipse>`
    :param calc_tensor: if True, use the gyration tensor eigenvalues to compute :math:`R_g`
    :type calc_tensor: bool, optional
    :param Rg_norm: Normalization factor for :math:`R_g` to compute color index, defaults to 1.0
    :type Rg_norm: scalar
    :param dark: use dark theme if True
    :type dark: bool, optional
    :ivar rg: The radius of gyration computed from particle positions.
    :type rg: scalar
    :ivar gyr: (if `calc_tensor=True`): The gyration tensor computed from particle positions.
    :type gyr: ndarray
    :ivar ci: Length-N numpy array containing :py:attr:`rg` repeated for every particle; used by :py:meth:`ColorBase.local_colors <coloring.ColorBase.local_colors>`.
    :type ci: ndarray
    """

    def __init__(self, shape: SuperEllipse = None,
                 calc_tensor=True, 
                 Rg_norm = 1.0,
                 dark: bool = True):
        """Constructor"""
        super().__init__(dark=dark)
        self._shape = _default_sphere if shape is None else shape
        self._c = _white_lime if dark else _grey_lime
        self._rg_norm = Rg_norm
        if calc_tensor:
            self.gyration_calc = self._gyr_rg
        else:
            self.gyration_calc = self._fast_rg

    def _fast_rg(self, pts):
        """Fast gyration radius using interparticle distances"""
        self.gyr = None
        self.rg = gyration_radius(pts)
        self.ci = np.array([1-self.rg/self._rg_norm]*self.num_pts)

    def _gyr_rg(self, pts):
        """Slow(er) gyration radius using gyration tensor eigenvalues"""
        self.gyr = gyration_tensor(pts)
        self.rg = np.sqrt(np.linalg.eigvalsh(self.gyr).sum())
        self.ci = np.array([2-self.rg/self._rg_norm]*self.num_pts).clip(0,1)

    def calc_state(self):
        """Caclulate the radius of gyration using either :py:meth:`radius_of_gyration <calc.morphology.radius_of_gyration>`,
        or by taking the eigenvalues of :py:meth:`gyration_tensor <calc.morphology.gyration_tensor>`."""
        pts = self.snap.particles.position
        self.gyration_calc(pts)
    # Use ColorBase.local_colors by default (ci is set in calc_state)

    def state_string(self, snap: gsd.hoomd.Frame = None):
        """
        :return: LaTeX-formatted summary string. i.e. ":math:`R_g/2a = 0.00`".
        :rtype: str
        """
        if snap is not None: self.snap = snap
        return f'$R_g/2a = {self.rg:.2f}$'


class ColorCirc(ColorRg):
    """
    Color all particles by their ensemble 'circiularity'.

    This style computes a single scalar value for the ensemble circularity
    :math:`c` via :py:meth:`circularity <calc.morphology.circularity>` and applies a
    single-parameter color gradient to every particle.
    
    :param shape: particle geometry, defaults to a sphere of diameter 1.0
    :type shape: :py:class:`SuperEllipse <visuals.shapes.SuperEllipse>`
    :param dark: use dark theme if True
    :type dark: bool, optional
    :ivar circ: The circularity computed from particle positions.
    :type circ: scalar
    :ivar ci: Length-N numpy array containing :py:attr:`circ` repeated for every particle; used by :py:meth:`ColorBase.local_colors <coloring.ColorBase.local_colors>`.
    :type ci: ndarray
    """
    def __init__(self, shape: SuperEllipse = None, dark: bool = True):
        """Constructor"""
        super().__init__(shape=shape, calc_tensor=True, dark=dark)

    def calc_state(self):
        """Calculate the ensemble circularity using :py:meth:`circularity <calc.morphology.circularity>`."""
        super().calc_state()
        self.circ = circularity(pts=None, gyr=self.gyr)
        # expose a numeric array for the base-class mapper
        self.ci = np.array([self.circ] * self.num_pts)

    def state_string(self, snap: gsd.hoomd.Frame = None):
        """
        :return: LaTeX-formatted summary string. i.e. ":math:`c = 0.00`".
        :rtype: str
        """
        if snap is not None: self.snap = snap
        return f'$c = {self.circ:.2f}$'



class ColorEpsPhase(ColorRg):
    """
    Color all particles by the complex phase of their ensemble 'ellipticity'.

    This style computes a single complex value for the ensemble ellipticity
    :math:`\\varepsilon` via :py:meth:`ellipticity <calc.morphology.ellipticity>` and applies a HSV color wheel
    to the complex phase   single-parameter color gradient to every particle.
    
    :param shape: particle geometry, defaults to a sphere of diameter 1.0
    :type shape: :py:class:`SuperEllipse <visuals.shapes.SuperEllipse>`
    :param dark: use dark theme if True
    :type dark: bool, optional
    :ivar eps: The ellipticity computed from particle positions.
    :type eps: complex
    :ivar ci: Length-N numpy array containing :py:attr:`eps` repeated for every particle; used by :py:meth:`ColorBase.local_colors <coloring.ColorBase.local_colors>`.
    :type ci: ndarray
    """
    def __init__(self, shape: SuperEllipse = None, shift=0.5, dark: bool = True):
        """Constructor"""
        super().__init__(shape=shape, calc_tensor=True, dark=dark)
        self._c = _rainbow
        self._shift = shift

    def calc_state(self):
        """Calculate the ensemble ellipticity using :py:meth:`ellipticity <calc.morphology.ellipticity>`."""
        super().calc_state()
        self.eps = ellipticity(pts=None, gyr=self.gyr)
        # expose a numeric array for the base-class mapper
        self.ci = ((np.array([np.angle(self.eps)]*self.num_pts) + np.pi) / (2*np.pi) + self._shift) % 1.0

    def state_string(self, snap: gsd.hoomd.Frame = None):
        """
        :return: LaTeX-formatted summary string. i.e. ":math:`\\varepsilon = 0.00\\exp[0.00i\\pi]`".
        :rtype: str
        """
        if snap is not None: self.snap = snap
        return f'$\\varepsilon = {np.abs(self.eps):.2f}\\exp[{np.angle(self.eps)/np.pi:.2f}i\\pi]$'

# ############################################################################################################
# # TESTING AND DEMONSTRATION
# ############################################################################################################

if __name__ == "__main__":
    
    # test code
    pass