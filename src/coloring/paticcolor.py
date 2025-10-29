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


class ColorS2(ColorBase):
    """Color particles by local nematic magnitude.

    Uses :py:meth:`local_patic <calc.orient_order.local_patic>` to compute
    a local nematic order parameter (:math:`S_2`) around each particle and maps its magnitude
    through a white->red (or grey->red) gradient.

    :param shape: particle geometry, defaults to a sphere of diameter 1.0
    :type shape: :py:class:`SuperEllipse <visuals.shapes.SuperEllipse>`
    :param dark: use dark theme if True
    :type dark: bool, optional
    :ivar ori: Complex director field (exp(i*theta)) computed from orientations
    :type ori: ndarray[complex]
    :ivar nem_g: Global nematic order (complex scalar) computed by :py:meth:`global_patic <calc.orient_order.global_patic>`.
    :type nem_g: complex
    :ivar nem_l: Local nematic order per particle (complex array) computed by :py:meth:`local_patic <calc.orient_order.local_patic>`.
    :type nem_l: ndarray[complex]
    :ivar nei: Neighborhood boolean matrix used to compute local order.
    :type nei: ndarray
    :ivar ci: Real-valued scalar field (:py:attr:`abs(nem_l)`) used by :py:meth:`ColorBase.local_colors <coloring.ColorBase.local_colors>`.
    :type ci: ndarray
    """

    def __init__(self, shape: SuperEllipse = None, dark: bool = True):
        """Constructor"""
        super().__init__(shape = shape)
        self._c = _white_red if dark else _grey_red

    def calc_state(self):
        """Calculate global and local nematic order parameters.

        This computes particle orientations, the global nematic order via
        :py:meth:`global_patic <calc.orient_order.global_patic>` and the local nematic order via
        :py:meth:`local_patic <calc.orient_order.local_patic>`. It also determines neighbors
        using :py:meth:`neighbors <calc.locality.neighbors>` so callers can inspect :py:attr:`nei`.
        """
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


class ColorS2Phase(ColorS2):
    """Color particles by the phase angle of the local nematic director using a rainbow wheel.

    This style converts the complex local nematic order into a phase in [0,1]
    (optionally shifted) and maps it to an HSV rainbow via the color mapper.

    :param shape: particle geometry, defaults to a sphere of diameter 1.0
    :type shape: :py:class:`SuperEllipse <visuals.shapes.SuperEllipse>`
    :param dark: use dark theme if True
    :type dark: bool
    :param shift: Phase offset (0-1) in color mapping
    :type shift: float
    :ivar ci: Phase per particle in [0,1] used by :meth:`ColorBase.local_colors <coloring.ColorBase.local_colors>`.
    :type ci: ndarray
    """

    def __init__(self, shape: SuperEllipse = None, dark: bool = True, shift: float = 0.0):
        """Constructor"""
        super().__init__(shape=shape)
        self._shift = shift
        # use rainbow mapping
        self._c = lambda x: _rainbow(x)

    def calc_state(self):
        """Compute parent state then cache the per-particle phase.

        Relies on :py:meth:`local_patic <calc.orient_order.local_patic>` populated by the parent
        implementation; converts complex local order to a normalized phase.
        """
        super().calc_state()
        self.ci = ((np.angle(self.nem_l) + np.pi) / (2 * np.pi) + self._shift) % 1.0
    # Use ColorBase.local_colors by default (ci is set in calc_state)


class ColorS2G(ColorS2):
    """Color all particles uniformly by global nematic order.

    This style computes the global nematic magnitude (:math:`S_{2,g}`) and exposes a uniform
    scalar :py:attr:`ci` so all particles receive the same color.

    :param shape: particle geometry
    :type shape: :py:class:`SuperEllipse <visuals.shapes.SuperEllipse>`
    :param dark: use dark theme if True
    :type dark: bool, optional
    :ivar nem_g: Global nematic order (complex scalar)
    :type nem_g: complex
    :ivar ci: Length-N array filled with :py:attr:`abs(nem_g)` used by :py:meth:`ColorBase.local_colors <coloring.ColorBase.local_colors>`.
    :type ci: ndarray
    """

    def calc_state(self):
        """Compute parent state then expose the uniform scalar based on the global magnitude."""
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



class ColorT4(ColorBase):
    """Color particles by local tetratic magnitude (:math:`T_4`) using an orange gradient.

    :param shape: particle geometry, defaults to a sphere of diameter 1.0
    :type shape: :py:class:`SuperEllipse <visuals.shapes.SuperEllipse>`
    :param dark: use dark theme if True
    :type dark: bool, optional
    :ivar ori: Complex director field (exp(i*theta)) computed from orientations
    :type ori: ndarray[complex]
    :ivar tet_g: Global tetratic order (complex scalar)
    :type tet_g: complex
    :ivar tet_l: Local tetratic order per particle (complex array)
    :type tet_l: ndarray[complex]
    :ivar nei: Neighborhood boolean matrix used to compute local order.
    :type nei: ndarray
    :ivar ci: Real-valued scalar field (:py:attr:`abs(tet_l)`) used by :py:meth:`ColorBase.local_colors <coloring.ColorBase.local_colors>`.
    :type ci: ndarray
    """

    def __init__(self, shape: SuperEllipse = None, dark: bool = True):
        """Constructor"""
        super().__init__(shape=shape)
        self._c = _white_orange if dark else _grey_orange

    def calc_state(self):
        """Compute tetratic global and local order.

        Uses :py:meth:`global_patic <calc.orient_order.global_patic>` and
        :py:meth:`local_patic <calc.orient_order.local_patic>` (with ``p=4``) to compute the
        tetratic order parameters and neighbor structure.
        """
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


class ColorT4G(ColorT4):
    """Color all particles uniformly by global tetratic order (:math:`T_{4,g}`).

    :param shape: particle geometry, defaults to a sphere of diameter 1.0
    :type shape: :py:class:`SuperEllipse <visuals.shapes.SuperEllipse>`
    :param dark: use dark theme if True
    :type dark: bool, optional
    :ivar tet_g: Global tetratic order (complex scalar)
    :type tet_g: complex
    :ivar ci: Length-N array filled with :py:attr:`abs(tet_g)` used by :py:meth:`ColorBase.local_colors <coloring.ColorBase.local_colors>`.
    :type ci: ndarray
    """

    def calc_state(self):
        """Compute parent state then expose the uniform scalar based on the global tetratic magnitude."""
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

    class neicolor(ColorS2):
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
