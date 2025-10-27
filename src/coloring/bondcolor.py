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
from calc import flat_bond_order, stretched_bond_order, projected_bond_order, steinhardt_bond_order, crystal_connectivity

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


class ColorBondOrder(ColorBase):
    """Color particles by local n-fold bond-orientational order. .

    This class computes per-particle bond-orientational order :math:`\\psi_{n,j}` (complex numbers).
    In 2d, It supports flat, projected and anisotropic (stretched) calculations and sets a
    canonical scalar :py:attr:`ci` to the magnitude the local bond-order paramter :math:`|\\psi_{n,j}|`).

    Alternatively, users can specify this class to calculate per-particle 3d Steinhardt bond order
    :math:`q_{lm,j}` (complex arrays). In this case the canonica scalar :py:attr:`ci` is set to the
    rotationally invariant local bond-order parameter: :math:`q_{l,j} = \\sqrt{\\frac{4\\pi}{2l+1}\\sum_m|q_{lm,j}|^2}`.

    The :py:attr:`ci` field is then mapped to white->red / gret->red color gradients.

    :param order: Bond-order symmetry, defaults to 6 (though 4 is also common)
    :type order: int, optional
    :param shape: Particle geometry, defaults to a sphere with diameter 1.0
    :type shape: :py:class:`SuperEllipse <visuals.shapes.SuperEllipse>`, optional
    :param surface_normal: Surface normal function for optional projected calculations, defaults to None
    :type surface_normal: callable, optional
    :param nei_cutoff: Dimensionless neighbor cutoff distance (in units of particle diameter), defaults to DEFAULT_CUTOFF (halfway between first coodination shells in 2D)
    :type nei_cutoff: scalar, optional
    :param periodic: whether to apply periodic boundary conditions during neighbor search, default to False
    :type periodic: bool, optional
    :param calc_3d: whether to compute full 3d Steinhardt bond order, default to False
    :type calc_3d: bool, optional
    :param dark: whether to use the dark theme, default to True
    :type dark: bool, optional
    :ivar bond_order: Bond-order calculation method used internally
    :type bond_order: callable
    :ivar nei: Neighbor boolean array/matrix used for local averages
    :type nei: ndarray
    :ivar rel_rot: `(if applicable)` Relative rotation factors for projected calculations
    :type rel_rot: ndarray[complex]
    :ivar qlm: `(if applicable)` Per-particle, per-m spherical harmonic bond-order components
    :type qlm: ndarray[complex]
    :ivar psi: `(if applicable)` Per-particle complex bond-order values
    :type psi: ndarray[complex]
    :ivar ci: Real-valued scalar field (i.e. :py:attr:`abs(psi)`) used by :py:meth:`ColorBase.local_colors`.
    :type ci: ndarray
    """

    def __init__(self, order: int = 6,
                 shape: SuperEllipse = None,
                 surface_normal:callable = None,
                 nei_cutoff = DEFAULT_CUTOFF, periodic=False,
                 calc_3d=False, dark: bool = True):
        """Constructor"""
        super().__init__(dark=dark)
        # Set color mapping function based on background
        self._c = _white_red if dark else _grey_red
            
        self._n = order
        self._shape = _default_sphere if shape is None else shape
        self._grad = surface_normal
        self._cut = nei_cutoff

        self._is_disc = (np.round(self._shape.aspect, 2) == 1 and np.round(self._shape.n, 2) == 2)
        self._is_proj = surface_normal is not None and not calc_3d
        self._per = periodic
        self._3d = calc_3d
        self._stretched = (not self._is_disc) and (not self._3d)

        if self._3d:
            # For fully 3D systems, use Steinhardt bond order
            self.bond_order = self._steinhardt

        elif self._is_proj:
            # For particles on curved surfaces, (i.e. 2d embedded in 3d) use projected bond order
            self.bond_order = self._psi_projected
        
        elif self._is_disc:
            # For flat discs/spheres, use the simplest flat bond order
            self.bond_order = self._psi
        
        else:
            # For superellipsoidal particles (all other cases), account for orientation-dependent interactions
            self.bond_order = self._psi_stretched


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
        if self._3d: Warning("Setting surface_normal for 3d systems may lead to unexpected behavior.")
        self._grad = grad
        self._is_proj = grad is not None

    def _psi(self, pts, angles):
        """calc flat bond order"""
        cut = self._cut * self.shape.ax * 2
        self.nei = neighbors(pts, neighbor_cutoff=cut)
        self.rot = None
        self.qlm = None
        self.psi = flat_bond_order(pts, nei_bool=self.nei, order=self._n)
        # Cache a canonical scalar field for the base class to map to colors
        self.ci = np.abs(self.psi)[:self.num_pts]

    def _psi_stretched(self, pts, angles):
        """calc stretched bond order"""
        cut = self._cut # cutoff is dimensionles in stretched coordinates
        self.nei = stretched_neighbors(pts, angles, rx=self.shape.ax, ry=self.shape.ay, neighbor_cutoff=cut)
        self.rot = None
        self.qlm = None
        self.psi = stretched_bond_order(pts, angles, rx=self.shape.ax, ry=self.shape.ay, nei_bool=self.nei, order=self._n)
        # Cache a canonical scalar field for the base class to map to colors
        self.ci = np.abs(self.psi)[:self.num_pts]
    
    def _psi_projected(self, pts, angles):
        """calc projected bond order"""
        cut = self._cut * self.shape.ax * 2
        self.nei = neighbors(pts, neighbor_cutoff=cut)
        self.rot = tangent_connection(pts, self._grad)
        self.qlm = None
        self.psi = projected_bond_order(pts, self._grad, nei_bool=self.nei, order=self._n)
        # Cache a canonical scalar field for the base class to map to colors
        self.ci = self.psi[:self.num_pts]

    def _steinhardt(self, pts, angles):
        """calc steinhardt bond order"""
        cut = self._cut * self.shape.ax * 2
        self.nei = neighbors(pts, neighbor_cutoff=cut)
        self.rot = None
        self.qlm = steinhardt_bond_order(pts, nei_bool=self.nei, l=self._n)
        self.psi = None 
        # Cache a canonical scalar field for the base class to map to colors
        self.ci = np.sqrt(4*np.pi/(2*self._n + 1)*np.sum(self.qlm * np.conj(self.qlm), axis=-1))[:self.num_pts]

    def calc_state(self):
        """
        Compute bond-order and neighbor structures and store to self.

        Implementation notes:
        
        - For spherical/disc particles this uses :py:meth:`flat_bond_order <calc.bond_order.flat_bond_order>`. In this case the class variables :py:attr:`rel_rot` and :py:attr:`qlm` are set to :code:`None`.
        
        - For projected geometries it uses :py:meth:`projected_bond_order <calc.bond_order.projected_bond_order>` and computes a tangent connection via :py:meth:`tangent_connection <calc.locality.tangent_connection>`. In this case the class variable :py:attr:`qlm` is set to :code:`None`.
        
        - For anisotropic particles it uses the stretched variants (:py:meth:`stretched_neighbors <calc.locality.stretched_neighbors>` and :py:meth:`stretched_bond_order <calc.bond_order.stretched_bond_order>`). In this case the class variables :py:attr:`rel_rot` and :py:attr:`qlm` are set to :code:`None`.
        
        - In 3d, this uses :py:meth:`steinhardt_bond_order <calc.bond_order.steinhardt_bond_order>`. In this case the class variables :py:attr:`rel_rot` and :py:attr:`psi` are set to :code:`None`.
        
        - If ``periodic`` is True, applies periodic boundary conditions during neighbor search using :py:meth:`expand_around_pbc <calc.locality.expand_around_pbc>`. IMPORTANT: this means that the instance variables have size N\' instead of size N. :py:attr:`ci` fields automatically adjust for this, but other instance variables (like :py:attr:`psi`) may need to be indexed accordingly.
        
        In 2d the scalar field :py:attr:`ci` is set to the per-particle magnitude :py:attr:`abs(self.psi)`. Whereas in 3d it's set to the rotationally invariant local bond-order parameter: :math:`q_l`. In either istances the base mapping (:py:meth:`ColorBase.local_colors <coloring.ColorBase.local_colors>`) can be applied unchanged.
        """
        pts = self.snap.particles.position
        if self._stretched:
            angles = quat_to_angle(self.snap.particles.orientation)
        else: angles = np.zeros(self.num_pts)

        cut = self._cut * self.shape.ax * 2

        if self._per:
            basis = box_to_matrix(self.snap.configuration.box)
            pts, pad_idx = expand_around_pbc(pts, basis, max_dist=cut)
        else:
            pad_idx = np.arange(self.num_pts)
        
        self.bond_order(pts, angles[pad_idx])

    # Use ColorBase.local_colors by default (ci is set in calc_state)

    def state_string(self, snap: gsd.hoomd.Frame = None):
        """
        :return: LaTeX-formatted summary string i.e. ":math:`\\langle|\\psi_n|\\rangle=0.00` " / ":math:`\\langle q_n \\rangle=0.00`".
        :rtype: str
        """
        if snap is not None: self.snap = snap
        if self._3d:
            q_l = np.mean(np.sqrt(4*np.pi/(2*self._n + 1)*np.sum(self.qlm * np.conj(self.qlm), axis=-1))[:self.num_pts])
            return f'$\\langle\\q_{self._n}\\rangle = {q_l:.2f}$'
        else:
            psi_l = np.mean(np.abs(self.psi[:self.num_pts]))
            return f'$\\langle|\\psi_{self._n}|\\rangle = {psi_l:.2f}$'


class ColorPsiG(ColorBondOrder):
    """Color all particles by the global 2d bond-order magnitude: :math:`|\\langle\\psi_n\\rangle|`.

    :param order: Bond-order symmetry, defaults to 6 (though 4 is also common)
    :type order: int, optional
    :param shape: Particle geometry, defaults to a sphere with diameter 1.0
    :type shape: :py:class:`SuperEllipse <visuals.shapes.SuperEllipse>`, optional
    :param nei_cutoff: Dimensionless neighbor cutoff distance (in units of particle diameter), defaults to DEFAULT_CUTOFF (halfway between first coodination shells in 2D)
    :type nei_cutoff: scalar, optional
    :param periodic: whether to apply periodic boundary conditions during neighbor search, default to False
    :type periodic: bool, optional
    :param dark: whether to use the dark theme
    :type dark: bool, optional
    :ivar psi_g: Global average bond-order magnitude
    :type psi_g: scalar
    :ivar ci: Length-N array filled with :py:attr:`psi_g` used by :py:meth:`ColorBase.local_colors <coloring.ColorBase.local_colors>`.
    :type ci: ndarray
    """

    def __init__(self, order:int=6,
                 shape: SuperEllipse = None,
                 nei_cutoff: float = DEFAULT_CUTOFF, periodic:bool=False,
                 dark: bool = True):
        super().__init__(shape=shape, order=order, dark=dark, periodic=periodic, calc_3d=False, nei_cutoff=nei_cutoff, surface_normal=None)

    def calc_state(self):
        """Calculate both global and local bond-order parameters and expose a uniform scalar field (:py:attr:`ci`)."""
        # Ensure parent calculations run first so :py:attr:`psi` is populated
        super().calc_state()
        self.psi_g = np.abs(np.mean(self.psi[:self.num_pts]))
        self.ci = np.array([self.psi_g] * self.num_pts)

    def state_string(self, snap: gsd.hoomd.Frame = None):
        """
        :return: LaTeX-formatted summary string i.e. ":math:`|\\langle\\psi_n\\rangle|=0.00`".
        :rtype: str
        """
        if snap is not None: self.snap = snap
        return f'$|\\langle\\psi_{self._n}\\rangle| = {self.psi_g:.2f}$'


class ColorPsiPhase(ColorBondOrder):
    """Color particles by phase (angle) of bond-order using a rainbow map.

    Converts the complex local :math:`\\psi_n` value into a phase and maps that
    phase to hue using an HSV/rainbow map. The :py:attr:`shift` parameter rotates the
    hue wheel for better visual separation in some datasets.

    :param order: Bond-order symmetry, defaults to 6 (though 4 is also common)
    :type order: int, optional
    :param shape: Particle geometry, defaults to a sphere with diameter 1.0
    :type shape: :py:class:`SuperEllipse <visuals.shapes.SuperEllipse>`, optional
    :param surface_normal: Surface normal function for optional projected calculations, defaults to None
    :type surface_normal: callable, optional
    :param nei_cutoff: Dimensionless neighbor cutoff distance (in units of particle diameter), defaults to DEFAULT_CUTOFF (halfway between first coodination shells in 2D)
    :type nei_cutoff: scalar, optional
    :param periodic: whether to apply periodic boundary conditions during neighbor search, default to False
    :type periodic: bool, optional
    :param shift: Phase offset in color mapping (0-1 scale)
    :type shift: float
    :param dark: whether to use the dark theme
    :type dark: bool, optional
    :ivar ci: Per-particle normalized phase in [0,1] used by :py:meth:`ColorBase.local_colors <coloring.ColorBase.local_colors>`.
    :type ci: ndarray
    """

    def __init__(self, order: int = 6,
                 shape: SuperEllipse = None,
                 surface_normal: callable = None,
                 nei_cutoff: float = DEFAULT_CUTOFF, periodic:bool=False,
                 shift: float = 0.6, dark: bool = True):
        """Constructor"""
        super().__init__(shape=shape, order=order, dark=dark, periodic=periodic, calc_3d=False, nei_cutoff=nei_cutoff, surface_normal=surface_normal)
        # store shift for phase mapping and use rainbow as color function
        self._shift = shift
        self._c = lambda x: _rainbow(x)
    
    def calc_state(self):
        """Compute parent state then cache the per-particle phase into :py:attr:`ci`."""
        super().calc_state()
        if self._is_proj:
            psi = self.psi * (self.rel_rot[0] ** self._n)
        else:
            psi = self.psi
        self.ci = ((np.angle(psi[:self.num_pts]) + np.pi) / (2 * np.pi) + self._shift) % 1.0
    
    def state_string(self, snap: gsd.hoomd.Frame = None):
        """
        :return: LaTeX-formatted summary string i.e. ":math:`|\\langle\\psi_n\\rangle|=0.00`".
        :rtype: str
        """
        if snap is not None: self.snap = snap
        if self._is_proj: return ""
        else:
            psi_g = np.abs(np.mean(self.psi[:self.num_pts]))
            return f'$|\\langle\\psi_{self._n}\\rangle| = {psi_g:.2f}$'


class ColorQG(ColorBondOrder):
    """Color all particles by the global 3d rotationally invariant steinhardt bond-order magnitude: :math:`Q_{l} = \\sqrt{\\frac{4\\pi}{2l+1}\\sum_m|\\langle q_{lm,j}\\rangle|^2}`.

    :param order: Bond-order symmetry, defaults to 6 (though 4 is also common)
    :type order: int, optional
    :param shape: Particle geometry, defaults to a sphere with diameter 1.0
    :type shape: :py:class:`SuperEllipse <visuals.shapes.SuperEllipse>`, optional
    :param nei_cutoff: Dimensionless neighbor cutoff distance (in units of particle diameter), defaults to DEFAULT_CUTOFF (halfway between first coodination shells in 2D)
    :type nei_cutoff: scalar, optional
    :param periodic: whether to apply periodic boundary conditions during neighbor search, default to False
    :type periodic: bool, optional
    :param q_norm: Normalization factor to scale bond-order magnitude for coloring, defaults to 0.57
    :type q_norm: float, optional
    :param dark: whether to use the dark theme
    :type dark: bool, optional
    :ivar q_g: Global average bond-order magnitude
    :type q_g: scalar
    :ivar ci: Length-N array filled with :py:attr:`q_g/q_norm` used by :py:meth:`ColorBase.local_colors <coloring.ColorBase.local_colors>`.
    :type ci: ndarray
    """

    def __init__(self, order: int = 6,
                 shape: SuperEllipse = None,
                 nei_cutoff: float = DEFAULT_CUTOFF, periodic:bool=False,
                 q_norm:float = 0.57, dark: bool = True):
        super().__init__(shape=shape, order=order, dark=dark, periodic=periodic, calc_3d=True, nei_cutoff=nei_cutoff, surface_normal=None)
        self._qn = q_norm

    def calc_state(self):
        """Calculate both global and local bond-order parameters and expose a uniform scalar field (:py:attr:`ci`)."""
        # Ensure parent calculations run first so :py:attr:`psi` is populated
        super().calc_state()
        self.q_g = np.sqrt(4*np.pi/(2*6+1)*np.sum(np.abs(self.qlm[:self.num_pts,:].mean(axis=0))**2))
        self.ci = np.clip(np.array([self.q_g] * self.num_pts)/self._qn, 0, 1)
    
    def state_string(self, snap: gsd.hoomd.Frame = None):
        """
        :return: LaTeX-formatted summary string i.e. ":math:`Q_l=0.00`".
        :rtype: str
        """
        if snap is not None: self.snap = snap
        return f'$Q_{self._n} = {self.q_g:.2f}$'

class ColorConn(ColorBondOrder):
    """Color particles by local crystal connectivity (i.e. C6) for flat, stretched, projected, and 3D systems.

    Uses :py:meth:`crystal_connectivity <calc.bond_order.crystal_connectivity>` applied to the previously computed
    :py:attr:`psi` / :py:attr:`qlm` and neighbor structure. The resulting connectivity is exposed as :py:attr:`ci`
    for the base color mapping (white->blue / grey->blue).

    :param order: Bond-order symmetry
    :type order: int, optional
    :param shape: Particle geometry, defaults to a sphere with diameter 1.0
    :type shape: :py:class:`SuperEllipse <visuals.shapes.SuperEllipse>`, optional
    :param surface_normal: Surface normal function for optional projected calculations, defaults to None
    :type surface_normal: callable, optional
    :param nei_cutoff: Dimensionless neighbor cutoff distance (in units of particle diameter), defaults to DEFAULT_CUTOFF (halfway between first coodination shells in 2D)
    :type nei_cutoff: scalar, optional
    :param periodic: whether to apply periodic boundary conditions during neighbor search, default to False
    :type periodic: bool, optional
    :param calc_3d: whether to compute full 3d Steinhardt bond order for connectivity, default to False
    :type calc_3d: bool, optional
    :param norm: Normalization factor for connectivity calculation, defaults to `order` in 2d and `2*order` in 3d
    :type norm: float, optional
    :param crystallinity_threshold: Threshold for considering a particle "crystalline" in connectivity calculation, defaults to 0.32 in 2d and 0.5 in 3d
    :type crystallinity_threshold: float, optional
    :param dark: whether to use the dark theme
    :type dark: bool, optional
    :ivar connectivity: Connectivity calculation method used internally
    :type connectivity: callable
    :ivar con: Per-particle connectivity values
    :type con: ndarray
    :ivar ci: Per-particle scalar connectivity used by :py:meth:`ColorBase.local_colors`.
    :type ci: ndarray
    """
    
    def __init__(self, order: int = 6,
                 shape: SuperEllipse = _default_sphere,
                 surface_normal: callable = None,
                 nei_cutoff: float = DEFAULT_CUTOFF, periodic:bool = False,
                 calc_3d:bool=False,
                 norm:float = None, crystallinity_threshold:float=None,
                 dark: bool = True):
        super().__init__(shape=shape, order=order, dark=dark, surface_normal=surface_normal, periodic=periodic, calc_3d=calc_3d, nei_cutoff=nei_cutoff)
        # Use blue color scheme for connectivity
        self._c = _white_blue if dark else _grey_blue

        if norm is None and self._3d: self._norm = 2*order
        elif norm is None: self._norm = order
        else: self._norm = norm

        if calc_3d:
            self._thresh = 0.5 if crystallinity_threshold is None else crystallinity_threshold
            self._norm = 2*order if norm is None else norm
            self.connectivity = self._con_3d
        
        elif self._is_proj:
            self._thresh = 0.32 if crystallinity_threshold is None else crystallinity_threshold
            self._norm = order if norm is None else norm
            self.connectivity = self._con_proj
        
        else:
            self._thresh = 0.32 if crystallinity_threshold is None else crystallinity_threshold
            self._norm = order if norm is None else norm
            self.connectivity = self._con
    
    def _con(self):
        """calc flat connectivity"""
        return crystal_connectivity(self.psi, self.nei, norm=self._norm, crystallinity_threshold=self._thresh, calc_3d=False)

    def _con_proj(self):
        """calc projected connectivity"""
        phase_rotate = self.rel_rot ** self._n
        return crystal_connectivity(self.psi, self.nei, phase_rotate=phase_rotate, norm=self._norm, crystallinity_threshold=self._thresh, calc_3d=False)

    def _con_3d(self):
        """calc 3d connectivity"""
        return crystal_connectivity(self.qlm, self.nei, norm=self._norm, crystallinity_threshold=self._thresh, calc_3d=True)

    def calc_state(self):
        """
        Compute connectivity on top of bond-order and store to self.
        """
        super().calc_state()
        self.con = self.connectivity()
        self.ci = self.con[:self.num_pts]

    def state_string(self, snap: gsd.hoomd.Frame = None):
        """
        :return: LaTeX-formatted summary string. i.e. ":math:`\\langle C_n\\rangle=0.00`".
        :rtype: str
        """
        if snap is not None: self.snap = snap
        return f'$\\langle C_6\\rangle = {np.mean(self.con[:self.num_pts]):.2f}$'


class ColorConnG(ColorConn):
    """Color all particles by global connectivity, which is just the average local connectivity

    :see: :py:class:`ColorBondOrder`, :py:class:`ColorConn`
    """
    
    def __init__(self, shape: SuperEllipse = _default_sphere,
                 surface_normal: callable = None,
                 order: int = 6, nei_cutoff: float = DEFAULT_CUTOFF,
                 periodic:bool = False, calc_3d:bool=False,
                 norm:float = None, crystallinity_threshold:float=None,
                 dark: bool = True):
        """Constructor"""
        super().__init__(shape=shape, order=order, dark=dark, surface_normal=surface_normal, periodic=periodic, calc_3d=calc_3d, nei_cutoff=nei_cutoff, norm=norm, crystallinity_threshold=crystallinity_threshold)

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
    from render import render_npole, render_sphere, render_3d, animate
    from coloring import ColorBlender, base_colors, color_blender
    import traceback

    white_purp = color_blender(c00=base_colors['white'], c01=base_colors['red'], c10=base_colors['blue'], c11=base_colors['purple'])

    try:
        def _make_movie(gsd_path, outpath, style, fps=10, codec='mpeg4', istart=0, iend=-1, istride=10, sphere=False,clust=False):
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
            elif clust:
                L0 = frames[0].configuration.box[0]*2.5
                figure_maker = lambda snap: render_3d(snap, style=style, dark=True, figsize=5, dpi=500, L=L0)
            else:
                figure_maker = lambda snap: render_npole(snap, style=style, PEL='contour', dark=True, figsize=5, dpi=500)

            try:
                animate(sel, outpath=outpath, figure_maker=figure_maker, fps=fps, codec=codec)
            except Exception as _e:
                print(f"Could not make movie {outpath}: {_e}")
                traceback.print_exc()
                return

        # # Control / q-pole examples
        # style = ColorPsiPhase()
        # _make_movie('../tests/test-control.gsd', '../tests/phase-qpole.mp4', style, istride=100)
        # _make_movie('../tests/test-control.gsd', '../docs/source/_static/phase-qpole.webm', style, codec='libvpx', istride=100)

        # style = ColorPsiG()
        # _make_movie('../tests/test-control.gsd', '../tests/psig-qpole.mp4', style, istride=100)
        # _make_movie('../tests/test-control.gsd', '../docs/source/_static/psig-qpole.webm', style, codec='libvpx', istride=100)

        # style = ColorConn()
        # _make_movie('../tests/test-control.gsd', '../tests/c6-qpole.mp4', style, istride=100)
        # _make_movie('../tests/test-control.gsd', '../docs/source/_static/c6-qpole.webm', style, codec='libvpx', istride=100)

        style = ColorBlender(white_purp, ColorPsiG(), ColorConn())
        _make_movie('../tests/test-control.gsd', '../tests/psi6c6-qpole.mp4', style, istride=100)
        _make_movie('../tests/test-control.gsd', '../docs/source/_static/psi6c6-qpole.webm', style, codec='libvpx', istride=100)


        # # opole examples
        # style = ColorPsiPhase()
        # _make_movie('../tests/test-opole1.gsd', '../tests/phase-opole1.mp4', style, istride=25)
        # _make_movie('../tests/test-opole1.gsd', '../docs/source/_static/phase-opole1.webm', style, codec='libvpx', istride=25, iend=2500)

        # style = ColorPsiPhase()
        # _make_movie('../tests/test-opole2.gsd', '../tests/phase-opole2.mp4', style, istride=25)
        # _make_movie('../tests/test-opole2.gsd', '../docs/source/_static/phase-opole2.webm', style, codec='libvpx', istride=25, iend=2500)

        style = ColorBlender(white_purp, ColorPsiG(), ColorConn())
        _make_movie('../tests/test-opole1.gsd', '../tests/psi6c6-opole1.mp4', style, istride=25)
        _make_movie('../tests/test-opole1.gsd', '../docs/source/_static/psi6c6-opole1.webm', style, codec='libvpx', istride=25, iend=2500)

        style = ColorBlender(white_purp, ColorPsiG(), ColorConn())
        _make_movie('../tests/test-opole2.gsd', '../tests/psi6c6-opole2.mp4', style, istride=25)
        _make_movie('../tests/test-opole2.gsd', '../docs/source/_static/psi6c6-opole2.webm', style, codec='libvpx', istride=25, iend=2500)


        # # spherical surface examples
        # style = ColorPsiPhase()
        # _make_movie('../tests/test-sphere.gsd', '../tests/phase-sphere.mp4', style, sphere=True, iend=100, istride=2)
        # _make_movie('../tests/test-sphere.gsd', '../docs/source/_static/phase-sphere.webm', style, codec='libvpx', sphere=True, iend=100, istride=2)

        # style = ColorConn()
        # _make_movie('../tests/test-sphere.gsd', '../tests/c6-sphere.mp4', style, sphere=True, iend=100, istride=2)
        # _make_movie('../tests/test-sphere.gsd', '../docs/source/_static/c6-sphere.webm', style, codec='libvpx', sphere=True, iend=100, istride=2)


        # # rect examples
        # style = ColorConn(shape=SuperEllipse(ax=1.0, ay=0.5, n=20), order=4, periodic=True, nei_cutoff=2.6)
        # _make_movie('../tests/test-rect1.gsd', '../tests/c4-rect1.mp4', style)
        # _make_movie('../tests/test-rect1.gsd', '../docs/source/_static/c4-rect1.webm', style, codec='libvpx')

        # style = ColorConn(shape=SuperEllipse(ax=1.0, ay=0.5, n=20), order=4, periodic=True, nei_cutoff=2.6)
        # _make_movie('../tests/test-rect2.gsd', '../tests/c4-rect2.mp4', style)
        # _make_movie('../tests/test-rect2.gsd', '../docs/source/_static/c4-rect2.webm', style, codec='libvpx')


        # # clust examples
        # style = ColorQG(periodic=True)
        # _make_movie('../tests/test-32p.gsd', '../tests/Q6-clust.mp4', style, clust=True, istart=11000, iend=18000, istride=50)
        # _make_movie('../tests/test-32p.gsd', '../docs/source/_static/Q6-clust.webm', style, codec='libvpx', clust=True, istart=11000, iend=18000, istride=50)

        # style = ColorConn(periodic=True, calc_3d=True)
        # _make_movie('../tests/test-32p.gsd', '../tests/C6-clust.mp4', style, clust=True, istart=11000, iend=18000, istride=50)
        # _make_movie('../tests/test-32p.gsd', '../docs/source/_static/C6-clust.webm', style, codec='libvpx', clust=True, istart=11000, iend=18000, istride=50)

        style = ColorBlender(white_purp, ColorQG(periodic=True), ColorConn(periodic=True, calc_3d=True))
        _make_movie('../tests/test-32p.gsd', '../tests/Q6C6-clust.mp4', style, clust=True, istart=11000, iend=18000, istride=50)
        _make_movie('../tests/test-32p.gsd', '../docs/source/_static/Q6C6-clust.webm', style, codec='libvpx', clust=True, istart=11000, iend=18000, istride=50)

    except Exception as e:
        # Print the exception and full traceback so the caller can see where the
        # error originated (file name and line number). This makes debugging
        # failures in the demo code much easier than a single-line message.
        print('Agent movie creation skipped due to error:', e)
        traceback.print_exc()