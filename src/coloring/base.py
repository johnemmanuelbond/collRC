"""
Color scheming module for visualizing particle simulation data.

This module provides color mapping functions and classes for visualizing various
physical properties of particle systems, including crystalline order, nematic order,
and electrical potential fields. It supports both dark and light backgrounds.
"""

import numpy as np
import gsd.hoomd
import matplotlib.pyplot as plt
import matplotlib.colors as mcol

############################################################################################################
# BASE COLOR DEFINITIONS AND MIXING FUNCTIONS
############################################################################################################

base_colors = dict(
    white = np.array(mcol.to_rgba('white')),
    grey = np.array(mcol.to_rgba('darkgrey')),
    red = np.array(mcol.to_rgba('red')),
    blue = np.array(mcol.to_rgba('blue')),
    green = np.array(mcol.to_rgba('green')),
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
    
    Provides basic infrastructure for coloring particles based on their state. Supports both dark and light background themes.

    StateColor has a ``snap`` attribute that caches the last GSD frame used for coloring, to avoid redundant computations.

    StateColor subclasses should implement the `local_colors` and `state_string` methods to provide specific coloring logic and state descriptions.
    
    :param dark: Whether to use a dark background theme, defaults to True
    :type dark: bool, optional
    """
    
    def __init__(self, dark: bool = True):
        # Choose base color based on background theme
        if dark:
            self._c = base_colors['white']
        else:
            self._c = base_colors['grey']
        self._f = None

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

    def calc_state(self):
        """Calculate the state of the color mapping. Subclasses will overwrite by storing reaction coords."""
        pass

    def local_colors(self, snap: gsd.hoomd.Frame = None):
        """
        Generate colors for each particle based on local properties. Will not recalculate if the frame matches the cached one.
        
        :param snap: GSD frame containing particle data
        :type snap: gsd.hoomd.Frame
        :return: Array of RGB colors for each particle
        :rtype: ndarray
        """
        if snap is not None: self.snap = snap
        return np.array([self._c] * self.snap.particles.N)

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


if __name__ == "__main__":

    # Simple demo of color_gradient and color_blender
    # Create a 1D gradient and show as an image
    grad = color_gradient(base_colors['white'], base_colors['red'])
    x = np.linspace(0, 1, 256)
    img1 = grad(x)

    # Create a 2D blend using two scalar fields
    blender = color_blender(base_colors['white'], base_colors['red'], base_colors['blue'], base_colors['purple'])
    nx = 128
    ny = 128
    X, Y = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny))
    img2 = blender(X, Y)

    # Create a 2D blend using two scalar fields
    blender = color_blender(base_colors['white'], base_colors['blue'], base_colors['gold'], base_colors['green'])
    nx = 128
    ny = 128
    X, Y = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny))
    img3 = blender(X, Y)

    print("Demo: gradient shape:", img1.shape)
    print("Demo: blender shape:", img2.shape)
    print("Demo: blender shape:", img3.shape)

    # Display the results
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img1.reshape(1, -1, 3), aspect='auto')
    axes[0].set_title('color_gradient')
    axes[0].axis('off')

    axes[1].imshow(img2, origin='lower')
    axes[1].set_title('color_blender')
    axes[1].axis('off')

    axes[2].imshow(img3, origin='lower')
    axes[2].set_title('color_blender 2')
    axes[2].axis('off')

    fig.savefig("color_demo.png", dpi=600, bbox_inches='tight')

