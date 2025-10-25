# -*- coding: utf-8 -*-
"""
Color schemes for order-parameter visualizations.

Provides coloring classes that map local and global bond-order,
connectivity and defect measures into RGB colors. Classes support
dark/light themes and projected geometries.
"""

import numpy as np
import gsd.hoomd

from matplotlib.cm import hsv as hsv_map

from visuals import SuperEllipse
from coloring import base_colors, color_gradient, color_blender
from coloring import ColorBase

from calc.locality import DEFAULT_CUTOFF
from calc import neighbors, stretched_neighbors, tangent_connection, quat_to_angle, expand_around_pbc, box_to_matrix
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
    """Color particles by local n-fold bond-orientational order. .

    This class computes per-particle bond-orientational order :math:`\\psi_n` (complex
    numbers) and exposes both local and global statistics. It supports
    flat, projected and anisotropic (stretched) calculations and sets a
    canonical scalar :py:attr:`ci` (the magnitude of :math:`\\psi_n`) for the base
    color mapping.

    :param shape: Particle geometry, defaults to a sphere with diameter 1.0
    :type shape: :py:class:`SuperEllipse <visuals.shapes.SuperEllipse>`, optional
    :param surface_normal: Surface normal function for optional projected calculations, defaults to None
    :type surface_normal: callable, optional
    :param order: Bond-order symmetry, defaults to 6 (though 4 is also common)
    :type order: int, optional
    :param periodic: whether to apply periodic boundary conditions during neighbor search, default to False
    :type periodic: bool, optional
    :param dark: whether to use the dark theme, default to True
    :type dark: bool, optional
    :ivar nei: Neighbor boolean array/matrix used for local averages
    :type nei: ndarray
    :ivar rel_rot: `(if applicable)` Relative rotation factors for projected calculations
    :type rel_rot: ndarray[complex]
    :ivar psi: Per-particle complex bond-order values
    :type psi: ndarray[complex]
    :ivar psi_g: Global average bond-order (complex scalar)
    :type psi_g: complex
    :ivar ci: Real-valued scalar field (:py:attr:`abs(psi)`) used by :py:meth:`ColorBase.local_colors`.
    :type ci: ndarray
    """

    def __init__(self, shape: SuperEllipse = _default_sphere,
                 surface_normal:callable = None,
                 order: int = 6, periodic=False, dark: bool = True):
        """Constructor"""
        super().__init__(dark=dark)
        # Set color mapping function based on background
        self._c = _white_red if dark else _grey_red
            
        self._n = order
        self._shape = shape
        self._grad = surface_normal

        self._is_disc = (np.round(shape.aspect, 2) == 1 and np.round(shape.n, 2) == 2)
        self._is_proj = surface_normal is not None
        self._per = periodic
        

    @property
    def surface_normal(self) -> callable:
        """
        :return: Surface normal function for projected calculations.
        :rtype: callable | None
        """
        return self._grad

    @surface_normal.setter
    def surface_normal(self, grad: callable):
        """
        :param grad: Surface normal function for projected calculations.
        :type grad: callable
        """
        self._grad = grad
        self._is_proj = grad is not None

    def calc_state(self):
        """
        Compute bond-order and neighbor structures and store to self.

        Implementation notes:
        
        - For spherical/disc particles this uses :py:meth:`flat_bond_order <calc.bond_order.flat_bond_order>`.
        
        - For projected geometries it uses :py:meth:`projected_bond_order <calc.bond_order.projected_bond_order>` and computes a tangent connection via :py:meth:`tangent_connection <calc.locality.tangent_connection>`.
        
        - For anisotropic particles it uses the stretched variants (:py:meth:`stretched_neighbors <calc.locality.stretched_neighbors>` and :py:meth:`stretched_bond_order <calc.bond_order.stretched_bond_order>`).
        
        - If ``periodic`` is True, applies periodic boundary conditions during neighbor search using :py:meth:`expand_around_pbc <calc.locality.expand_around_pbc>`. IMPORTANT: this means that the instance variables have size N\' instead of size N. :py:attr:`ci` fields automatically adjust for this, but other instance variables (like :py:attr:`psi`) may need to be indexed accordingly.
        
        After computing the complex :py:attr:`psi`, this method sets :py:attr:`ci`
        to the per-particle magnitude :py:attr:`abs(self.psi)` so the base mapping
        (:py:meth:`ColorBase.local_colors <coloring.ColorBase.local_colors>`) can be used unchanged.
        """
        pts = self.snap.particles.position
        cut = DEFAULT_CUTOFF * self.shape.ax * 2
        
        if self._is_disc and not self._is_proj:
            # For circular particles, use standard neighbor detection
            if self._per:
                basis = box_to_matrix(self.snap.configuration.box)
                pts, _ = expand_around_pbc(pts, basis, max_dist=cut)

            nei = neighbors(pts, neighbor_cutoff=cut)
            rot = None
            psi = flat_bond_order(pts, nei_bool=nei, order=self._n)
        elif self._is_proj:
            # For particles on curved surfaces, use projected bond order
            if self._per:
                basis = box_to_matrix(self.snap.configuration.box)
                pts, _ = expand_around_pbc(pts, basis, max_dist=cut)
            nei = neighbors(pts, neighbor_cutoff=cut)
            rot = tangent_connection(pts, self._grad)
            psi = projected_bond_order(pts, self._grad, nei_bool=nei, order=self._n)
        else:
            # For elliptical particles, account for orientation-dependent interactions
            if not self._per:
                angles = quat_to_angle(self.snap.particles.orientation)
            else:
                basis = box_to_matrix(self.snap.configuration.box)
                pts, pad_idx = expand_around_pbc(pts, basis, max_dist=cut)
                angles = quat_to_angle(self.snap.particles.orientation)[pad_idx]

            nei = stretched_neighbors(pts, angles, rx=self.shape.ax, ry=self.shape.ay, neighbor_cutoff=2.7)
            rot = None
            psi = stretched_bond_order(pts, angles, rx=self.shape.ax, ry=self.shape.ay, nei_bool=nei, order=self._n)
            
        self.nei     = nei
        self.rel_rot = rot
        self.psi     = psi
        # Cache a canonical scalar field for the base class to map to colors
        self.ci = np.abs(self.psi)[:self.num_pts]

    # Use ColorBase.local_colors by default (ci is set in calc_state)

    def state_string(self, snap: gsd.hoomd.Frame = None):
        """
        :return: LaTeX-formatted summary string i.e. ":math:`\\langle|\\psi_n|\\rangle=0.00` ".
        :rtype: str
        """
        if snap is not None: self.snap = snap
        psi_l = np.mean(np.abs(self.psi[:self.num_pts]))
        return f'$\\langle|\\psi_6|\\rangle = {psi_l:.2f}$'

class ColorByGlobalPsi(ColorByPsi):
    """Color all particles by the global bond-order magnitude: :math:`|\\langle\\psi_n\\rangle|`.

    :param shape: Particle geometry
    :type shape: :py:class:`SuperEllipse <visuals.shapes.SuperEllipse>`, optional
    :param order: Bond-order symmetry
    :type order: int, optional
    :param dark: whether to use the dark theme
    :type dark: bool, optional
    :ivar psi_g: Global average bond-order magnitude
    :type psi_g: scalar
    :ivar ci: Length-N array filled with :py:attr:`psi_g` used by :py:meth:`ColorBase.local_colors <coloring.ColorBase.local_colors>`.
    :type ci: ndarray
    """

    def __init__(self, shape: SuperEllipse = _default_sphere,
                 order: int = 6, dark: bool = True):
        super().__init__(shape=shape, order=order, dark=dark)

    def calc_state(self):
        """Calculate both global and local bond-order parameters and expose a uniform scalar field (:py:attr:`ci`)."""
        # Ensure parent calculations run first so :py:attr:`psi` is populated
        super().calc_state()
        self.psi_g = np.abs(np.mean(self.psi[:self.num_pts]))
        self.ci = np.array([self.psi_g] * self.num_pts)


class ColorByPhase(ColorByPsi):
    """Color particles by phase (angle) of bond-order using a rainbow map.

    Converts the complex local :math:`\\psi_n` value into a phase and maps that
    phase to hue using an HSV/rainbow map. The :py:attr:`shift` parameter rotates the
    hue wheel for better visual separation in some datasets.

    :param shape: Particle geometry
    :type shape: :py:class:`SuperEllipse <visuals.shapes.SuperEllipse>`, optional
    :param order: Bond-order symmetry
    :type order: int, optional
    :param shift: Phase offset in color mapping (0-1 scale)
    :type shift: float
    :param surface_normal: Surface normal function for optional projected calculations, defaults to None
    :type surface_normal: callable, optional
    :param dark: whether to use the dark theme
    :type dark: bool, optional
    :ivar ci: Per-particle normalized phase in [0,1] used by :py:meth:`ColorBase.local_colors <coloring.ColorBase.local_colors>`.
    :type ci: ndarray
    """

    def __init__(self, shape: SuperEllipse = _default_sphere,
                 order: int = 6, shift: float = 0.6,
                 surface_normal: callable = None, dark: bool = True):
        """Constructor"""
        super().__init__(shape=shape, order=order, dark=dark, surface_normal=surface_normal)
        # store shift for phase mapping and use rainbow as color function
        self._shift = shift
        self._c = lambda x: _rainbow(x)
    
    def calc_state(self):
        """Compute parent state then cache the per-particle phase into :py:attr:`ci`."""
        super().calc_state()
        if self._is_proj:
            psi = self.psi * (self.rel_rot ** self._n)
        else:
            psi = self.psi
        self.ci = ((np.angle(psi[:self.num_pts]) + np.pi) / (2 * np.pi) + self._shift) % 1.0


class ColorByConn(ColorByPsi):
    """Color particles by local crystal connectivity (i.e. C6) as defined in :py:meth:`crystal_connectivity <calc.bond_order.crystal_connectivity>`.

    Uses :py:meth:`crystal_connectivity <calc.bond_order.crystal_connectivity>` applied to the previously computed
    :py:attr:`psi` and neighbor structure. The resulting connectivity is
    exposed as :py:attr:`ci` for the base color mapping.

    :param shape: Particle geometry
    :type shape: :py:class:`SuperEllipse <visuals.shapes.SuperEllipse>`, optional
    :param surface_normal: Surface normal function for optional projected calculations, defaults to None
    :type surface_normal: callable, optional
    :param order: Bond-order symmetry
    :type order: int, optional
    :param periodic: whether to apply periodic boundary conditions during neighbor search, default to False
    :type periodic: bool, optional
    :param dark: whether to use the dark theme
    :type dark: bool, optional
    :ivar con: Per-particle connectivity values
    :type con: ndarray
    :ivar ci: Per-particle scalar connectivity used by :py:meth:`ColorBase.local_colors`.
    :type ci: ndarray
    """
    
    def __init__(self, shape: SuperEllipse = _default_sphere,
                 surface_normal: callable = None,
                 order: int = 6, periodic = False, dark: bool = True):
        super().__init__(shape=shape, order=order, dark=dark, surface_normal=surface_normal, periodic=periodic)
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
        self.ci = c6[:self.num_pts]

    def state_string(self, snap: gsd.hoomd.Frame = None):
        """
        :return: LaTeX-formatted summary string. i.e. ":math:`\\langle C_n\\rangle=0.00`".
        :rtype: str
        """
        if snap is not None: self.snap = snap
        return f'$\\langle C_6\\rangle = {np.mean(self.con[:self.num_pts]):.2f}$'


class ColorByGlobalConn(ColorByConn):
    """Color all particles by global connectivity, which is just the average local connectivity

    :see: :py:class:`ColorByPsi`, :py:class:`ColorByConn`
    """
    
    def __init__(self, shape: SuperEllipse = _default_sphere,
                 order: int = 6, surface_normal: callable = None,
                 dark: bool = True):
        """Constructor"""
        super().__init__(shape=shape, order=order, dark=dark, surface_normal=surface_normal)

    def calc_state(self):
        """Calculate the average crystal connectivity and set as uniform field (:py:attr:`ci`)."""
        super().calc_state()
        con_g = np.mean(self.con[:self.num_pts])
        self.ci = np.array([con_g]*self.num_pts)
    # Use ColorBase.local_colors by default (ci is set in calc_state)




############################################################################################################
# TESTING AND DEMONSTRATION
############################################################################################################

if __name__ == "__main__":
    
    # make example movies
    from render import render_npole, render_sphere, animate
    from coloring import ColorBlender, base_colors, color_blender
    import traceback

    white_purp = color_blender(c00=base_colors['white'], c01=base_colors['red'], c10=base_colors['blue'], c11=base_colors['purple'])

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
                figure_maker = lambda snap: render_sphere(snap, style=style, dark=True, figsize=4, dpi=500, L=L0)
            else:
                figure_maker = lambda snap: render_npole(snap, style=style, PEL='contour', dark=True, figsize=4, dpi=500)

            try:
                animate(sel, outpath=outpath, figure_maker=figure_maker, fps=fps, codec=codec)
            except Exception as _e:
                print(f"Could not make movie {outpath}: {_e}")
                traceback.print_exc()
                return

        # Control / q-pole examples
        style = ColorByPhase()
        _make_movie('../tests/test-control.gsd', '../tests/phase-qpole.mp4', style, istride=100)
        _make_movie('../tests/test-control.gsd', '../docs/source/_static/phase-qpole.webm', style, codec='libvpx', istride=100)

        style = ColorByGlobalPsi()
        _make_movie('../tests/test-control.gsd', '../tests/psig-qpole.mp4', style, istride=100)
        _make_movie('../tests/test-control.gsd', '../docs/source/_static/psig-qpole.webm', style, codec='libvpx', istride=100)

        style = ColorByConn()
        _make_movie('../tests/test-control.gsd', '../tests/c6-qpole.mp4', style, istride=100)
        _make_movie('../tests/test-control.gsd', '../docs/source/_static/c6-qpole.webm', style, codec='libvpx', istride=100)

        style = ColorBlender(white_purp, ColorByGlobalPsi(), ColorByConn())
        _make_movie('../tests/test-control.gsd', '../tests/psi6c6-qpole.mp4', style, istride=100)
        _make_movie('../tests/test-control.gsd', '../docs/source/_static/psi6c6-qpole.webm', style, codec='libvpx', istride=100)


        # opole and sphere examples (reuse same style classes)
        style = ColorByPhase()
        _make_movie('../tests/test-opole1.gsd', '../tests/phase-opole1.mp4', style, istride=25)
        _make_movie('../tests/test-opole1.gsd', '../docs/source/_static/phase-opole1.webm', style, codec='libvpx', istride=25, iend=2500)

        style = ColorByPhase()
        _make_movie('../tests/test-opole2.gsd', '../tests/phase-opole2.mp4', style, istride=25)
        _make_movie('../tests/test-opole2.gsd', '../docs/source/_static/phase-opole2.webm', style, codec='libvpx', istride=25, iend=2500)

        style = ColorBlender(white_purp, ColorByGlobalPsi(), ColorByConn())
        _make_movie('../tests/test-opole1.gsd', '../tests/psi6c6-opole1.mp4', style, istride=25)
        _make_movie('../tests/test-opole1.gsd', '../docs/source/_static/psi6c6-opole1.webm', style, codec='libvpx', istride=25, iend=2500)

        style = ColorBlender(white_purp, ColorByGlobalPsi(), ColorByConn())
        _make_movie('../tests/test-opole2.gsd', '../tests/psi6c6-opole2.mp4', style, istride=25)
        _make_movie('../tests/test-opole2.gsd', '../docs/source/_static/psi6c6-opole2.webm', style, codec='libvpx', istride=25, iend=2500)


        style = ColorByPhase()
        _make_movie('../tests/test-sphere.gsd', '../tests/phase-sphere.mp4', style, sphere=True, iend=100, istride=2)
        _make_movie('../tests/test-sphere.gsd', '../docs/source/_static/phase-sphere.webm', style, codec='libvpx', sphere=True, iend=100, istride=2)

        style = ColorByConn()
        _make_movie('../tests/test-sphere.gsd', '../tests/c6-sphere.mp4', style, sphere=True, iend=100, istride=2)
        _make_movie('../tests/test-sphere.gsd', '../docs/source/_static/c6-sphere.webm', style, codec='libvpx', sphere=True, iend=100, istride=2)


        # rect examples
        style = ColorByConn(shape=SuperEllipse(ax=1.0, ay=0.5, n=20), order=4)
        _make_movie('../tests/test-rect1.gsd', '../tests/c4-rect1.mp4', style)
        _make_movie('../tests/test-rect1.gsd', '../docs/source/_static/c4-rect1.webm', style, codec='libvpx')

        style = ColorByConn(shape=SuperEllipse(ax=1.0, ay=0.5, n=20), order=4)
        _make_movie('../tests/test-rect2.gsd', '../tests/c4-rect2.mp4', style)
        _make_movie('../tests/test-rect2.gsd', '../docs/source/_static/c4-rect2.webm', style, codec='libvpx')

    except Exception as e:
        # Print the exception and full traceback so the caller can see where the
        # error originated (file name and line number). This makes debugging
        # failures in the demo code much easier than a single-line message.
        print('Agent movie creation skipped due to error:', e)
        traceback.print_exc()