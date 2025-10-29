# -*- coding: utf-8 -*-
"""
Base color scheming module for visualizing particle simulation data. Provides basic
color definitions, color mixing functions, and a base class for defining coloring
schemes based on particle state.
"""

import numpy as np
import gsd.hoomd
import matplotlib.pyplot as plt
import matplotlib.colors as mcol
from visuals import SuperEllipse

############################################################################################################
# BASE COLOR DEFINITIONS AND MIXING FUNCTIONS
############################################################################################################

base_colors = dict(
    white = np.array(mcol.to_rgba('white')),
    grey = np.array(mcol.to_rgba('darkgrey')),
    red = np.array(mcol.to_rgba('red')),
    blue = np.array(mcol.to_rgba('blue')),
    green = np.array(mcol.to_rgba('green')),
    lime = np.array(mcol.to_rgba('lime')),
    yellow = np.array(mcol.to_rgba('yellow')),
    gold = np.array(mcol.to_rgba('gold')),
    purple = np.array(mcol.to_rgba('magenta')),
    orange = np.array(mcol.to_rgba('orange')),
    cyan = np.array(mcol.to_rgba('cyan')),
)


# Helper function for broadcasting color mixing operations
# Takes scalar field x and color vector v, returns colored field
_cprod = lambda x, v: np.outer(x.flatten(), v).reshape(*x.shape, *v.shape)


def color_gradient(c1=base_colors['white'], c2=base_colors['red']):
    """Create a color gradient mapping function between two colors.

    :param c1: The starting color, defaults to white
    :type c1: np.ndarray, optional
    :param c2: The ending color, defaults to red
    :type c2: np.ndarray, optional
    :return: a function that maps a scalar field in [0,1] to colors between c1 and c2
    :rtype: function
    """
    e1 = c2 - c1
    c_func = lambda x: (c1 + _cprod(x, e1)).clip(0, 1)
    return c_func


def color_blender(c00=base_colors['white'], c01=base_colors['red'], c10=base_colors['blue'], c11=base_colors['purple']):
    """Create a color blending function based on two scalar fields.

    :param c00: Color at (0,0), defaults to white
    :type c00: np.ndarray, optional
    :param c01: Color at (0,1), defaults to red
    :type c01: np.ndarray, optional
    :param c10: Color at (1,0), defaults to blue
    :type c10: np.ndarray, optional
    :param c11: Color at (1,1), defaults to purple
    :type c11: np.ndarray, optional
    :return: a function that maps two scalar fields in [0,1]x[0,1] to colors blended between the four corner colors
    :rtype: function
    """
    e1 = c01-c00 # i.e. white to red
    e2 = c10-c00 # i.e. white to blue
    e3 = c01-c11 # i.e. purple to red
    e4 = c10-c11 # i.e. purple to blue

    c_func = lambda x, y: (_cprod((x+y)<1, c00)  + _cprod(x*(x+y<1), e1)      + _cprod(y*(x+y<1), e2) + 
                           _cprod((x+y)>=1, c11) + _cprod((1-y)*(x+y>=1), e3) + _cprod((1-x)*(x+y>=1), e4)).clip(0, 1)

    return c_func


############################################################################################################
# STATECOLOR BASE CLASS
############################################################################################################

# Default sphere geometry for calculations
_default_sphere = SuperEllipse(ax=0.5, ay=0.5, n=2.0)

def _gsd_match(gsd1: gsd.hoomd.Frame, gsd2: gsd.hoomd.Frame) -> bool:
    """ Check if two `GSD`<>_ frames are identical. Only checks positions, orientations, and types, not logged metadata.

    :param gsd1: a GSD frame
    :type gsd1: gsd.hoomd.Frame
    :param gsd2: another GSD frame
    :type gsd2: gsd.hoomd.Frame
    :return: True if frames are identical 
    :rtype: bool
    """

    try:
        pos_match = np.all(gsd1.particles.position == gsd2.particles.position)
        ori_match = np.all(gsd1.particles.orientation == gsd2.particles.orientation)
        typ_match = np.all(gsd1.particles.typeid == gsd2.particles.typeid)
        return (pos_match and ori_match and typ_match)
    except AttributeError:
        return False


class ColorBase():
    """
    Base class for particle coloring schemes.
    
    Provides basic infrastructure for coloring particles based on their state. Supports both dark and light background themes (grey and white particles respectively).

    StateColor has a :py:attr:`snap` attribute that caches the last GSD frame used for coloring, to avoid redundant computations.

    StateColor subclasses should implement the :py:meth:`calc_state`, :py:meth:`local_colors` and :py:meth:`state_string` methods to provide specific coloring logic and state descriptions.
    
    :param shape: Particle geometry used for calculations, defaults to a sphere of diameter 1.0
    :type shape: SuperEllipse
    :param dark: Whether to use a dark background theme, defaults to True
    :type dark: bool, optional
    :ivar ci: Canonical scalar field (length N) that will be mapped to colors. Subclasses should set this in :py:meth:`calc_state`.
    :type ci: ndarray
    """
    
    def __init__(self, shape: SuperEllipse = None, dark: bool = True):
        """
        Constructor
        """
        # Choose base color based on background theme
        if dark:
            self._c = lambda x: np.array([base_colors['white']] * x.shape[0])
        else:
            self._c = lambda x: np.array([base_colors['grey']] * x.shape[0])
        self._f = None
        self._shape = _default_sphere if shape is None else shape

    @property
    def snap(self) -> gsd.hoomd.Frame:
        """Get the current GSD frame used for coloring."""
        return self._f
    
    @snap.setter
    def snap(self, snap: gsd.hoomd.Frame):
        """Set the current GSD frame used for coloring."""
        if not _gsd_match(self._f, snap):
            self._f = snap
            try: self.calc_state()
            except AttributeError: pass
    
    @property
    def num_pts(self) -> int:
        """Get the number of particles in the current frame."""
        try:
            return self._f.particles.N
        except AttributeError:
            return 0

    @property
    def shape(self) -> SuperEllipse:
        """Get the shape used for coloring.

        :return: The particle geometry used by this style
        :rtype: :py:class:`SuperEllipse <visuals.shapes.SuperEllipse>`
        """
        return self._shape

    @shape.setter
    def shape(self, shape: SuperEllipse):
        """Set the shape used for coloring.

        :param shape: particle geometry
        :type shape: :py:class:`SuperEllipse <visuals.shapes.SuperEllipse>`
        """
        self._shape = shape

    def calc_state(self):
        """Calculate the state of the color mapping. Subclasses will overwrite by storing reaction coordinates."""
        self.ci = np.zeros(self.num_pts)

    def local_colors(self, snap: gsd.hoomd.Frame = None):
        """
        Generate colors for each particle based on local properties. Will not recalculate if the frame matches the cached one.
        
        :param snap: GSD frame containing particle data
        :type snap: gsd.hoomd.Frame
        :return: Array of RGBA colors for each particle
        :rtype: ndarray
        """
        if snap is not None: self.snap = snap
        return self._c(self.ci)

    def state_string(self, snap: gsd.hoomd.Frame = None):
        """
        Generate descriptive string about the system state. Will not recalculate if the frame matches the cached one.
        
        :param snap: GSD frame containing particle data
        :type snap: gsd.hoomd.Frame
        :return: Descriptive string about the system state
        :rtype: str
        """
        if snap is not None: self.snap = snap
        return ""

class ColorBlender(ColorBase):
    """Blend two coloring styles using a two-argument color blending function.

    This helper composes two :py:class:`ColorBase` instances and a blending
    callable that accepts two scalar fields (the :py:attr:`ci` arrays of the two
    styles) and returns an array of RGBA colors. The blender delegates
    state computation to both styles and then applies :py:meth:`c_func` to their
    computed scalar fields.

    :param c_func: Callable accepting two scalar fields :code:`(x, y)` and returning RGBA colors.
    :type c_func: callable
    :param mainstyle: Primary coloring style whose scalar field will be used as the first blend axis.
    :type mainstyle: ColorBase
    :param otherstyle: Secondary coloring style whose scalar field will be used as the second blend axis.
    :type otherstyle: ColorBase
    :ivar s1: The primary :py:class:`ColorBase` instance (same as :code:`mainstyle`).
    :type s1: :py:class:`ColorBase`
    :ivar s2: The secondary :py:class:`ColorBase` instance (same as :code:`otherstyle`).
    :type s2: :py:class:`ColorBase`
    :ivar c_func: The blending callable used to combine :py:attr:`s1.ci` and :py:attr:`s2.ci` into colors.
    :type c_func: callable
    """

    def __init__(self, c_func:callable, mainstyle:ColorBase, otherstyle:ColorBase):
        """Constructor"""
        super().__init__(shape = mainstyle._shape)
        self.s1 = mainstyle
        self.s2 = otherstyle
        self.c_func = c_func


    def calc_state(self):
        """Compute and cache the state for both constituent styles.

        This delegates to the two composed :py:class:`ColorBase` instances
        and ensures their :py:attr:`ci` fields are up-to-date. The blender does
        not itself store additional state beyond the two styles.

        :return: None
        """
        self.s1.snap = self.snap
        self.s2.snap = self.snap
        self.s1.calc_state()
        self.s2.calc_state()
    
    def local_colors(self, snap: gsd.hoomd.Frame = None):
        """Return blended RGBA colors for the provided (or cached) frame.

        If :py:attr:`snap` is provided the underlying styles will be updated with
        that frame before blending. The function :py:meth:`c_func` is expected
        to accept two scalar fields (:py:attr:`s1.ci`, :py:attr:`s2.ci`) and return an
        (N,4) RGBA array.

        :param snap: optional GSD frame to compute colors for
        :type snap: gsd.hoomd.Frame, optional
        :return: (N,4) array of RGBA colors produced by blending
        :rtype: ndarray
        """
        if snap is not None:
            self.s1.snap = snap
            self.s2.snap = snap
        return self.c_func(self.s1.ci, self.s2.ci)
    
    def state_string(self, snap: gsd.hoomd.Frame = None):
        """Return a combined descriptive string from both styles.

        If :py:attr:`snap` is provided the underlying styles will be updated with
        that frame before the strings are requested. The returned string is
        the primary style's state_string followed by the secondary's,
        separated by a newline.

        :param snap: optional GSD frame to compute state strings for
        :type snap: gsd.hoomd.Frame, optional
        :return: Combined human-readable state description
        :rtype: str
        """
        if snap is not None:
            self.s1.snap = snap
            self.s2.snap = snap
        return self.s1.state_string() + "\n" + self.s2.state_string()



if __name__ == "__main__":

    X, Y = np.meshgrid(np.linspace(0, 1, 256), np.linspace(0, 1, 256))

    # Simple demo of color_gradient and color_blender
    
    # Create a 1D gradient and show as an image
    img1 = color_gradient(base_colors['white'], base_colors['red'])(X)
    
    # Create a 2D blend using two scalar fields
    img2 = color_blender(base_colors['white'], base_colors['red'], base_colors['blue'], base_colors['purple'])(X,Y)

    # Create a 2D blend using two scalar fields
    img3 = color_blender(base_colors['white'], base_colors['blue'], base_colors['gold'], base_colors['green'])(X,Y)

    # Display the results
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img1.reshape(1, -1, 4), aspect='auto')
    axes[0].set_title('white->red')
    axes[0].axis('off')

    axes[1].imshow(img2, origin='lower')
    axes[1].set_title('white->purple')
    axes[1].axis('off')

    axes[2].imshow(img3, origin='lower')
    axes[2].set_title('white->green')
    axes[2].axis('off')

    fig.savefig("color_demo.png", dpi=600, bbox_inches='tight')