# -*- coding: utf-8 -*-
"""
Contains a few helper methods representing colloidal shapes in matplotlib. Includes a class to represent superelliptical shapes. Includes methods to plot them on matplotlib axes using real spatial metrics.
"""

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import transforms, patches
from calc import quat_to_angle


class SuperEllipse():
    """A class to contain helpful methods for characterizing the superellipses used in this module. Superellipses follow the equation:
    
    .. math::

        \\bigg|\\frac{x}{a_x}\\bigg|^n + \\bigg|\\frac{y}{a_y}\\bigg|^n = 1
    
    :param ax: one of the radii of the superellipse, defaults to 1.0
    :type ax: scalar, optional
    :param ay: the other radius of the superellipse, defaults to 1.0
    :type ay: scalar, optional
    :param n: the 'superellipse parameter' defines how sharp the corners are. :math:`n\\to\\infty` produces rectangles, :math:`n=2` gives ellipses, :math:`n\\to1` produces rhombuses, defaults to 2.0
    :type n: scalar, optional
    :ivar ax: one of the radii of the superellipse, defaults to 1.0
    :type ax: scalar, optional
    :ivar ay: the other radius of the superellipse, defaults to 1.0
    :type ay: scalar, optional
    :ivar n: the 'superellipse parameter' defines how sharp the corners are. :math:`n\\to\\infty` produces rectangles, :math:`n=2` gives ellipses, :math:`n\\to1` produces rhombuses, defaults to 2.0:
    :type n: scalar, optional

    """
    def __init__(self,ax:float=1.0,ay:float=1.0,n:float=2.0):
        """
        Constructor

        :param ax: one of the radii of the superellipse, defaults to 1.0
        :type ax: scalar, optional
        :param ay: the other radius of the superellipse, defaults to 1.0
        :type ay: scalar, optional
        :param n: the 'superellipse parameter' defines how sharp the corners are. :math:`n\\to\\infty` produces rectangles, :math:`n=2` gives ellipses, :math:`n\\to1` produces rhombuses, defaults to 2.0
        """        
        self.ax = max(ax,ay)
        self.ay = min(ax,ay)
        self.n = n
    
    @property
    def aspect(self) -> float:
        """
        :return: the aspect ratio of the superellipse
        :rtype: scalar
        """        
        try:
            return self.ax/self.ay
        except ZeroDivisionError:
            if self.ay==0 and self.ax==0:   return 1
            else: return np.nan
    
    @property
    def area(self) -> float:
        """
        :return: the area of the superellipse
        :rtype: scalar
        """        
        if self.n == 2.0:
            A = math.pi*self.ax*self.ay
        else:
            A = 4*self.ax*self.ay*(math.gamma(float(1+1/self.n)))**2/(math.gamma(float(1+2/self.n)))
        return A
    
    def surface(self,thetas:np.ndarray) -> np.ndarray:
        """returns points along the perimeter of the superellipse associated with polar angles

        :param thetas: an array of polar angles
        :type thetas: ndarray
        :return: the x- and y-positions of points along the perimeter of the superellipse
        :rtype: ndarray
        """        
        rad = self.ax*self.ay * ( np.abs(self.ax*np.sin(thetas))**self.n + np.abs(self.ay*np.cos(thetas))**self.n )**(-1/self.n)
        return np.array([rad*np.cos(thetas), rad*np.sin(thetas)]).T


_default_sphere = SuperEllipse(ax=0.5,ay=0.5,n=2.0)
def flat_patches(pts, orient, shape=_default_sphere,n_resolve=101):
    """Generates a list of matplotlib Patches which render SuperEllipse shapes in 2D at specified positions and orientations.

    :param pts: an :math:`[N, 2+]` array of points where each shape should be centered. Only the x- and y-coordinates are used.
    :type pts: ndarray
    :param orient: an :math:`[N, 4]` array of quaternions specifying the orientation of each shape
    :type orient: ndarray
    :param shape: the shape to be drawn at each point, defaults to a circle of diameter 1.0
    :type shape: SuperEllipse, optional
    :param n_resolve: the number of points to use to resolve the perimeter of each shape, defaults to 101
    :type n_resolve: int, optional
    :return: a list of the specified shapes at the specified points and orientations
    :rtype: list[Patch]
    """

    angles = quat_to_angle(orient)
    peri = shape.surface(np.linspace(0,2*np.pi,n_resolve,endpoint=False))
    
    all_patches = [patches.Polygon(peri, transform=transforms.Affine2D().rotate(a).translate(r[0], r[1]), zorder=1) 
                   for a, r in zip(angles, pts)]
    
    return all_patches
    


def projected_patches(pts, grads, shape=_default_sphere, n_resolve=101, view_dir = np.array([0,0,1]), view_dist=100.0, view_ref = 'z', parallax = False, centered=False):
    """Generates a list of matplotlib Patches which render SuperEllipse shapes in 3D projected onto an embedded 2D surface at specified positions and orientations.

    *WIP: still needs testing for acircular shapes on surfaces*

    :param pts: an :math:`[N, 3]` array of points of each shape in real space
    :type pts: ndarray
    :param grads: an :math:`[N, 3]` array of vectors specifying the surface gradient (normal vector) at each point
    :type orient: ndarray
    :param shape: the shape to be drawn at each point, defaults to a sphere of radius 0.5
    :type shape: SuperEllipse, optional
    :param n_resolve: the number of points to use to resolve the perimeter of each shape, defaults to 251
    :type n_resolve: int, optional
    :param view_dir: a :math:`[3,]` vector specifying the viewing direction, defaults to looking down the z-axis
    :type view_dir: ndarray, optional
    :param view_dist: the distance from the viewer to the origin, defaults to 100.0
    :type view_dist: float, optional
    :param view_ref: a string specifying which axis should be treated as 'up' in the viewing frame, defaults to 'z'
    :type view_ref: str, optional
    :param parallax: if True, particles further away appear smaller. If False, particles closer are scaled to take up a smaller portion of the proected surface, defaults to False
    :type parallax: bool, optional
    :param centered: if True, recenters all particles to have median position (0,0) in the projected plane, defaults to False
    :type centered: bool, optional
    :return: a list of matplotlib patches correctly rotated into place.
    :rtype: list[Patch]
    """

    # set up reference direction
    ref = np.array([0,-1,0])
    if view_ref == 'z':
        pass
    elif 'x' in view_ref:
        ref = np.array([0,0,-1])
    elif 'y' in view_ref:
        ref = np.array([-1,0,0])
    else:
        raise Exception(f"unrecognized reference view: {view_ref}");
    # optional negation to flip axes
    if '-' in view_ref: ref*=-1

    # set up projected coorinate system, xp,yp,zp are positions, ip,jp,kp are normal vectors to a constraing surface
    e3 = view_dir/np.linalg.norm(view_dir)
    e1 = np.cross(e3,ref)
    e1 = e1/np.linalg.norm(e1,axis=-1)
    e2 = np.cross(e3,e1)
    e2 = e2/np.linalg.norm(e2,axis=-1)
    xp, yp, zp = np.array([pts @ e1, pts @ e2, pts @ e3])
    if centered:
        xp = xp - np.median(xp)
        yp = yp - np.median(yp)
    ip, jp, kp = np.array([grads @ e1, grads @ e2, grads @ e3])

    # calculate the factors needed to scale and rotate each particle to look as if it's sitting on the projected surface
    if parallax:
        parallax_factor = 1.0 + zp/view_dist
    else:
        parallax_factor = 1.0 - zp/view_dist
    rotate_angle = np.atan2(ip,-jp)
    zorder = np.array(10*(zp-zp.min()) + 1, dtype=int)

    # keep track of which particles are actually visible
    to_render = (kp > 0.1) & (zp > np.median(zp)-zp.std())

    # Each particle is a polygon rotated and translated to match simulation state, first we set up the matplotlib transforms:
    trans_list = [transforms.Affine2D() for _ in pts]
    trans_list = [t.scale(sx = p, sy = p*s) for t, p, s in zip(trans_list, parallax_factor, kp)]
    trans_list = [t.rotate(a) for t, a in zip(trans_list, rotate_angle)]
    trans_list = [t.translate(x, y) for t, x, y in zip(trans_list, xp, yp)]

    # Then we create a collection of particle patches using those transforms
    peri = shape.surface(np.linspace(0,2*np.pi,n_resolve,endpoint=False))
    all_patches = np.array([patches.Polygon(peri, transform = t, zorder = int(zo)) for t, zo in zip(trans_list, zorder)])

    return all_patches, to_render


def plot_principal_axes(com:np.ndarray, gyr:np.ndarray, ax=None, view_dir = np.array([0,0,1]), **plt_kwargs):
    """
    Plot the principal axes of the gyration tensor using quiver on the provided matplotlib ``Axes``.

    :param com: A :math:`[3,]` center of mass position a in 3D.
    :type com: ndarray
    :param gyr: The :math:`[3,3]` gyration tensor in 3D.
    :type gyr: ndarray
    :param ax: Axis to draw the contours on. If None, a new figure and axis will be created.
    :type ax: matplotlib.axes.Axes, optional
    :param view_dir: a :math:`[3,]` vector specifying the viewing direction in 3D, defaults to looking down the z-axis
    :type view_dir: ndarray, optional
    :param plt_kwargs: Additional keyword arguments forwarded to :meth:`matplotlib.axes.Axes.plot` (color, scale, etc)
    :type plt_kwargs: dict
    :return: The Axes object after plotting
    :rtype: matplotlib.pyplot.Axes
    """

    # create axis if none provided
    if ax is None:
        fig, ax = plt.subplots()

    gyr_eigvals, gyr_eigvecs = np.linalg.eigh(gyr) # ascending order

    e3 = view_dir/np.linalg.norm(view_dir)
    e1 = np.cross(e3,np.array([0,-1,0]))
    e1 = e1/np.linalg.norm(e1,axis=-1)
    e2 = np.cross(e3,e1)
    e2 = e2/np.linalg.norm(e2,axis=-1)

    if 'lw' not in plt_kwargs:    plt_kwargs['lw'] = 2.0
    if 'color' not in plt_kwargs: plt_kwargs['color'] = 'orange'
    if 'alpha' not in plt_kwargs: plt_kwargs['alpha'] = 0.75

    proj_x0, proj_y0 = com @ e1, com @ e2
    for li, vi in zip(np.sqrt(gyr_eigvals[1:]), gyr_eigvecs.T[1:]):
        proj_x = vi @ e1
        proj_y = vi @ e2
        ax.plot([proj_x0 - proj_x * li, proj_x0 + proj_x * li],
                [proj_y0 - proj_y * li, proj_y0 + proj_y * li],
                **plt_kwargs)

    return ax


if __name__ == "__main__":

    # for testing

    pass