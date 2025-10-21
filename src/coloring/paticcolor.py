"""Color schemes for nematic and tetratic orientational order.

Provides simple coloring classes that map  p-atic magnitude, phase,
defects and global summaries into RGB colors. Classes support
dark/light themes.

WIP: Classes DO NOT currently support projected particle geometries
"""

import numpy as np
import gsd.hoomd
from matplotlib.cm import hsv as hsv_map

from visuals import SuperEllipse
from .base import base_colors, color_gradient, ColorBase, _gsd_match
from calc import local_patic, global_patic, neighbors, quat_to_angle, central_eta

# base color functions
_white_red = color_gradient(c1=base_colors['white'], c2=base_colors['red'])
_grey_red = color_gradient(c1=base_colors['grey'], c2=base_colors['red'])
_white_orange = color_gradient(c1=base_colors['white'], c2=base_colors['orange'])
_grey_orange = color_gradient(c1=base_colors['grey'], c2=base_colors['orange'])

# _white_green = color_gradient(c1=base_colors['white'], c2=base_colors['green'])
# _grey_green = color_gradient(c1=base_colors['grey'], c2=base_colors['green'])
# _white_cyan = color_gradient(c1=base_colors['white'], c2=base_colors['cyan'])
# _grey_cyan = color_gradient(c1=base_colors['grey'], c2=base_colors['cyan'])

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
        
        # Global nematic order and director
        Z, phi = global_patic(angles, p=2)
        self.nem_g = Z * np.exp(1j * 2 * phi)

        # Local nematic order around each particle
        pts = self.snap.particles.position
        self.nei = neighbors(pts, neighbor_cutoff=6*self.shape.ax)
        Zs, phis = local_patic(angles, self.nei, p=2)
        self.nem_l = Zs * np.exp(1j * 2 * phis)

    def local_colors(self, snap: gsd.hoomd.Frame):
        """Return RGB colors mapping local S2 magnitude (white/grey -> red).

        :return: (N,3) RGB array
        :rtype: ndarray
        """
        if not _gsd_match(self.snap, snap):
            self.snap = snap
            self.calc_state()
        return self._c(np.abs(self.nem_l))

    def state_string(self, snap: gsd.hoomd.Frame):
        """
        :return: LaTeX-formatted summary string. i.e. ":math:`\\langle S_2\\rangle = 0.00`".
        :rtype: str
        """
        if not _gsd_match(self.snap, snap):
            self.snap = snap
            self.calc_state()
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
        self._c = lambda ang: _rainbow(((ang + np.pi) / (2 * np.pi) + self._shift) % 1.0)

    def local_colors(self, snap: gsd.hoomd.Frame):
        """Return RGB colors mapping local S2 phase angle.
        
        :return: (N,3) RGB array
        :rtype: ndarray
        """
        if not _gsd_match(self.snap, snap):
            self.snap = snap
            self.calc_state()
        return self._c(np.angle(self.nem_l))


class ColorGlobalS2(ColorByS2):
    """Color all particles uniformly by global nematic order (S2).

    :see: ColorByS2
    """

    def local_colors(self, snap: gsd.hoomd.Frame):
        """Return RGB colors mapping global S2 magnitude (white/grey -> red).

        :return: (N,3) RGB array
        :rtype: ndarray
        """
        if not _gsd_match(self.snap, snap):
            self.snap = snap
            self.calc_state()
        return self._c([np.abs(self.nem_g)]*len(self.nem_l))

    def state_string(self, snap: gsd.hoomd.Frame):
        """
        :return: LaTeX-formatted summary string. i.e. ":math:`S_{2,g} = 0.00`".
        :rtype: str
        """
        if not _gsd_match(self.snap, snap):
            self.snap = snap
            self.calc_state()
        return f'$S_{{2,g}} = {np.abs(self.nem_g):.2f}$'



class ColorByT4(ColorByS2):
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
        Z, phi = global_patic(angles, p=2)
        self.tet_g = Z * np.exp(1j * 4 * phi)

        # Local nematic order around each particle
        pts = self.snap.particles.position
        self.nei = neighbors(pts, neighbor_cutoff=6*self.shape.ax)
        Zs, phis = local_patic(angles, self.nei, p=2)
        self.tet_l = Zs * np.exp(1j * 4 * phis)

    def local_colors(self, snap: gsd.hoomd.Frame):
        """Return RGB colors mapping local T4 magnitude (white/grey -> orange).

        :return: (N,3) RGB array
        :rtype: ndarray
        """
        if not _gsd_match(self.snap, snap):
            self.snap = snap
            self.calc_state()
        return self._c(np.abs(self.tet_l))

    def state_string(self, snap: gsd.hoomd.Frame):
        """
        :return: LaTeX-formatted summary string. i.e. ":math:`\\langle T_4\\rangle = 0.00`".
        :rtype: str
        """
        if not _gsd_match(self.snap, snap):
            self.snap = snap
            self.calc_state()
        tet_l = np.abs(np.mean(self.tet_l))
        return f'$\\langle T_4\\rangle = {tet_l:.2f}$'


class ColorGlobalT4(ColorByT4):
    """Color all particles uniformly by global tetratic order (T4).

    :see: ColorByT4
    """

    def local_colors(self, snap: gsd.hoomd.Frame):
        """Return RGB colors mapping global T4 magnitude (white/grey -> orange).

        :return: (N,3) RGB array
        :rtype: ndarray
        """
        if not _gsd_match(self.snap, snap):
            self.snap = snap
            self.calc_state()
        return self._c([np.abs(self.tet_g)]*len(self.tet_l))

    def state_string(self, snap: gsd.hoomd.Frame):
        """
        :return: LaTeX-formatted summary string. i.e. ":math:`T_{4,g} = 0.00`".
        :rtype: str
        """
        if not _gsd_match(self.snap, snap):
            self.snap = snap
            self.calc_state()
        return f'$T_{{4,g}} = {np.abs(self.tet_g):.2f}$'



# ############################################################################################################
# # TESTING AND DEMONSTRATION
# ############################################################################################################

if __name__ == "__main__":
    
    # test code
    pass