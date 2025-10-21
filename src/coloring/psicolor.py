"""Color schemes for order-parameter visualizations.

Provides coloring classes that map local and global bond-order,
connectivity and defect measures into RGB colors. Classes support
dark/light themes and projected geometries.

:module: process.psicolor
"""

import numpy as np
import gsd.hoomd

from matplotlib.cm import hsv as hsv_map

from visuals import SuperEllipse
from .base import base_colors, color_gradient, color_blender
from .base import ColorBase, _gsd_match

from calc.locality import DEFAULT_CUTOFF
from calc import neighbors, stretched_neighbors, tangent_connection, quat_to_angle
from calc import flat_bond_order, stretched_bond_order, projected_bond_order, crystal_connectivity

# Default sphere geometry for calculations
_default_sphere = SuperEllipse(ax=0.5, ay=0.5, n=2.0)

_white_red = color_gradient(c1=base_colors['white'], c2=base_colors['red'])
_grey_red = color_gradient(c1=base_colors['grey'], c2=base_colors['red'])
_white_blue = color_gradient(c1=base_colors['white'], c2=base_colors['blue'])
_grey_blue = color_gradient(c1=base_colors['grey'], c2=base_colors['blue'])
_white_purp = color_blender(c00=base_colors['white'], c01=base_colors['red'], c10=base_colors['blue'], c11=base_colors['purple'])
_grey_purp = color_blender(c00=base_colors['grey'], c01=base_colors['red'], c10=base_colors['blue'], c11=base_colors['purple'])

# Rainbow color mapping for phase visualization
_rainbow = lambda c: hsv_map(c).clip(0, 1)


class ColorByPsi(ColorBase):
    """Color particles by local b-fold bond-order :math:`\\psi_n`.

    :param shape: Particle geometry, defaults to a sphere with diameter 1.
    :type shape: SuperEllipse, optional
    :param surface_normal: Surface normal function for optional projected calculations, defaults to None
    :type surface_normal: callable, optional
    :param order: Bond-order symmetry, defaults to 6 (though 4 is also common)
    :type order: int, optional
    :param dark: whether to use the dark theme, default to True
    :type dark: bool, optional
    """

    def __init__(self, shape: SuperEllipse = _default_sphere,
                 surface_normal:callable = None,
                 order: int = 6, dark: bool = True):
        super().__init__(dark=dark)
        # Set color mapping function based on background
        self._c = _white_red if dark else _grey_red
            
        self._n = order
        self._shape = shape
        self._grad = surface_normal

        self._is_disc = (np.round(shape.aspect, 2) == 1 and np.round(shape.n, 2) == 2)
        self._is_proj = surface_normal is not None
        

    @property
    def shape(self) -> SuperEllipse:
        """The SuperEllipse shape used for particle geometry.

        :rtype: SuperEllipse
        """
        return self._shape

    @shape.setter
    def shape(self, shape: SuperEllipse):
        self._shape = shape
        self._is_disc = (np.round(shape.aspect, 2) == 1 and np.round(shape.n, 2) == 2)

    @property
    def surface_normal(self) -> callable:
        """Surface normal function for projected calculations.

        :rtype: callable or None
        """
        return self._grad

    @surface_normal.setter
    def surface_normal(self, grad: callable):
        self._grad = grad
        self._is_proj = grad is not None

    def calc_state(self):
        """
        Compute bond-order and neighbor structures and store to self. If a surface normal function is provided, use it for projected calculations.
        """
        pts = self.snap.particles.position
        cut = DEFAULT_CUTOFF * self.shape.ax * 2
        
        if self._is_disc:
            # For circular particles, use standard neighbor detection
            nei = neighbors(pts, neighbor_cutoff=cut)
            rot = None
            psi = flat_bond_order(pts, nei_bool=nei, order=self._n)
        elif self._is_proj:
            # For particles on curved surfaces, use projected bond order
            nei = neighbors(pts, neighbor_cutoff=cut)
            rot = tangent_connection(pts, self._grad)
            psi = projected_bond_order(pts, self._grad, nei_bool=nei, order=self._n)
        else:
            # For elliptical particles, account for orientation-dependent interactions
            angles = quat_to_angle(self.snap.particles.orientation)
            nei = stretched_neighbors(pts, angles, rx=self.shape.ax, ry=self.shape.ay, neighbor_cutoff=2.7)
            rot = None
            psi = stretched_bond_order(pts, angles, rx=self.shape.ax, ry=self.shape.ay, nei_bool=nei, order=self._n)
            
        self.nei     = nei
        self.rel_rot = rot
        self.psi     = psi

    def local_colors(self, snap: gsd.hoomd.Frame):
        """Return per-particle RGB colors encoding local :math:`|\\psi_n|`.

        Mapping: for dark backgrounds this uses a white->red gradient; for light backgrounds a grey->red gradient.

        :param snap: gsd frame
        :type snap: gsd.hoomd.Frame
        :return: (N,3) array of RGB colors
        :rtype: ndarray
        """
        if not _gsd_match(self.snap, snap):
            self.snap = snap
            self.calc_state()
        return self._c(np.abs(self.psi))

    def state_string(self, snap: gsd.hoomd.Frame):
        """
        :return: LaTeX-formatted summary string i.e. ":math:`|\\langle\\psi_n\\rangle|=0.00` ".
        :rtype: str
        """
        if not _gsd_match(self.snap, snap):
            self.snap = snap
            self.calc_state()
        psi_g = np.abs(np.mean(self.psi))
        return f'$|\\langle\\psi_6\\rangle| = {psi_g:.2f}$'

class ColorByGlobalPsi(ColorByPsi):
    """Color all particles by the global bond-order magnitude: :math:`|\\langle\\psi_n\\rangle|`.

    :see: ColorByPsi
    """

    def __init__(self, shape: SuperEllipse = _default_sphere,
                 order: int = 6, dark: bool = True):
        super().__init__(shape=shape, order=order, dark=dark)

    def local_colors(self, snap=None):
        """Return uniform RGB colors derived from the global :math:`|\\langle\\psi_n\\rangle|` average.

        Mapping follows the same white/grey -> red convention as :meth:`ColorByPsi.local_colors`, but applied to the global average :math:`|\\langle\\psi_n\\rangle|` so every particle is colored the same.

        :return: (N,3) array of RGB colors
        :rtype: ndarray
        """
        if not _gsd_match(self.snap, snap):
            self.snap = snap
            self.calc_state()
        # Use global average for all particles
        psi_g = np.abs(np.mean(self.psi)) * np.ones(self.snap.particles.N)
        return self._c(psi_g)


class ColorByPhase(ColorByPsi):
    """Color particles by phase (angle) of bond-order using a rainbow map.

    :see: ColorByPsi
    :param shift: Phase offset in color mapping
    :type shift: float
    """

    def __init__(self, shape: SuperEllipse = _default_sphere,
                 order: int = 6, shift: float = 0.6,
                 surface_normal: callable = None, dark: bool = True):
        super().__init__(shape=shape, order=order, dark=dark, surface_normal=surface_normal)
        # Override color function to use rainbow mapping of angles
        self._c = lambda ang: _rainbow(((ang + np.pi) / (2 * np.pi) + shift) % 1.0)

    def local_colors(self, snap=None):
        """Return per-particle RGB colors encoding the phase (angle) of :math:`\\psi_n`.

        Mapping: uses a rainbow (HSV) color wheel so the hue corresponds to the complex phase of each particle's local :math:`\\psi_n`. This makes misoriented grains and grain boundaries visible because different orientations map to distinct hues.

        :return: (N,3) array of RGB colors
        :rtype: ndarray
        """
        if not _gsd_match(self.snap, snap):
            self.snap = snap
            self.calc_state()
        if self._is_proj:
            # Adjust phase for projected bond order
            psi = self.psi * (self.rel_rot ** self._n)
        else: psi = self.psi
        return self._c(np.angle(psi))


class ColorByConn(ColorByPsi):
    """Color particles by local crystal connectivity (i.e. C6) as defined in :py:meth:`calc.crystal_connectivity`. If a surface normal function is provided, use it for projected calculations.

    :see: ColorByPsi
    """
    
    def __init__(self, shape: SuperEllipse = _default_sphere,
                 order: int = 6, surface_normal: callable = None,
                 dark: bool = True):
        super().__init__(shape=shape, order=order, dark=dark, surface_normal=surface_normal)
        # Use blue color scheme for connectivity
        self._c = _white_blue if dark else _grey_blue
    
    def calc_state(self):
        """
        Compute connectivity in addition to bond-order and store to self.
        """
        super().calc_state()
        if self._is_proj:
            c6 = crystal_connectivity(self.psi, self.nei, phase_rotate=self.rel_rot**self._n, norm=self._n)
        else:
            c6 = crystal_connectivity(self.psi, self.nei)

        self.con = c6

    def local_colors(self, snap=None):
        """Return per-particle RGB colors that reflect local connectivity (i.e. C6).

        Mapping: dark theme uses white->blue, light uses grey->blue. Higher connectivity maps to stronger blue tones indicating crystalline environments.

        :return: (N,3) array
        :rtype: ndarray
        """
        if not _gsd_match(self.snap, snap):
            self.snap = snap
            self.calc_state()
        return self._c(self.con)

    def state_string(self, snap):
        """
        :return: LaTeX-formatted summary string. i.e. ":math:`\\langle C_n\\rangle=0.00`".
        :rtype: str
        """
        if not _gsd_match(self.snap, snap):
            self.snap = snap
            self.calc_state()
        return f'$\\langle C_6\\rangle = {np.mean(self.con):.2f}$'


class ColorByGlobalConn(ColorByConn):
    """Color all particles by global connectivity, which is just the average local connectivity

    :see: ColorByPsi
    :see: ColorByConn
    """
    
    def __init__(self, shape: SuperEllipse = _default_sphere,
                 order: int = 6, surface_normal: callable = None,
                 dark: bool = True):
        super().__init__(shape=shape, order=order, dark=dark, surface_normal=surface_normal)

    def local_colors(self, snap=None):
        """Return per-particle RGB colors that reflect global connectivity (i.e. mean C6).

        Mapping: dark theme uses white->blue, light uses grey->blue. Higher connectivity maps to stronger blue tones indicating crystalline ensembles.

        :return: (N,3) array
        :rtype: ndarray
        """
        if not _gsd_match(self.snap, snap):
            self.snap = snap
            self.calc_state()
        # Use global average for all particles (note: fixed bug with missing parentheses)
        c6_g = np.mean(self.con) * np.ones(self.snap.particles.N)
        return self._c(c6_g)



class QpoleSuite(ColorByConn):
    """Two-parameter color mixing of global bond-order and crystalline connectivity.

    :return: RGB colors combining psi_g and connectivity
    :rtype: ndarray
    """
    
    def __init__(self, shape: SuperEllipse = _default_sphere,
                 order: int = 6, surface_normal: callable = None,
                 dark: bool = True):
        super().__init__(shape=shape, order=order, dark=dark, surface_normal=surface_normal)
        # Use two-parameter color mixing
        self._c = _white_purp if dark else _grey_purp

    def local_colors(self, snap=None):
        """Return RGB colors combining global bond order, :math:`|\\langle\\psi\\rangle|` magnitude and local connectivity.

        Two-parameter blending (white->purple/grey->purple) maps bond orientational order in red and connectivity in blue. Meaning grey/white states have no symmetry, blue states have high connectivity but a grain boundary, and purple states are perfect crystals with high connectivity and no grain boundaries.

        :return: (N,3) array
        :rtype: ndarray
        """
        if not _gsd_match(self.snap, snap):
            self.snap = snap
            self.calc_state()
        # Combine global psi magnitude with local connectivity
        psi_g = np.abs(np.mean(self.psi)) * np.ones(self.snap.particles.N)
        return self._c(psi_g, self.con)

    def state_string(self, snap):
        """
        :return: LaTeX-formatted summary string: i.e. :math:`|\\langle\\psi_n\\rangle|=0.00` / :math:`\\langle C_n\\rangle=0.00`
        :rtype: str
        """
        if not _gsd_match(self.snap, snap):
            self.snap = snap
            self.calc_state()
        psi_g = np.abs(np.mean(self.psi))
        con_g = np.mean(self.con)
        return f'$|\\langle\\psi_6\\rangle| = {psi_g:.2f}$\n$\\langle C_6\\rangle = {con_g:.2f}$'






############################################################################################################
# TESTING AND DEMONSTRATION
############################################################################################################

if __name__ == "__main__":
    
    # test code
    pass