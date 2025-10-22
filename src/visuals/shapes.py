# -*- coding: utf-8 -*-
"""
Contains a few helper methods representing colloidal shapes in matplotlib. Includes a class to represent superelliptical shapes. Includes methods to plot them on matplotlib axes using real spatial metrics.
"""

import numpy as np
import math
from matplotlib import collections, transforms, patches
from calc import quat_to_angle

class SuperEllipse():
    """A class to contain helpful methods for characterizing the superellipses used in this module. Superellipses follow the equation:
    
    .. math::

        \\bigg|\\frac{x}{a_x}\\bigg|^n + \\bigg|\\frac{y}{a_y}\\bigg|^n = 1
    
    :param ax: one of the radii of the superellipse, defaults to 1.0
    :type ax: scalar, optional
    :param ay: the other radius of the superellipse, defaults to 1.0
    :type ay: scalar, optional
    :param n: the 'superellipse parameter' defines how sharp the corners are. :math:`n\\to\\infty` produces rectangles, :math:`n=2` gives ellipses, :math:`n\\to1` produces rhombuses, defaults to 2.0:
    :type n: scalar, optional

    """
    def __init__(self,ax:float=1.0,ay:float=1.0,n:float=2.0):
        """
        Constructor
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
def flat_patches(pts, orient, shape=_default_sphere,n_resolve=251):
    """Generates a PatchCollection of SuperEllipse shapes in 2D at specified positions and orientations.

    :param pts: an [N x 2+] array of points where each shape should be centered. Only the x- and y-coordinates are used.
    :type pts: ndarray
    :param orient: an [N x 4] array of quaternions specifying the orientation of each shape
    :type orient: ndarray
    :param shape: the shape to be drawn at each point, defaults to a circle of radius 0.5
    :type shape: SuperEllipse, optional
    :param n_resolve: the number of points to use to resolve the perimeter of each shape, defaults to 251
    :type n_resolve: int, optional
    :return: a PatchCollection of the specified shapes at the specified points and orientations
    :rtype: PatchCollection
    """

    angles = quat_to_angle(orient)
    peri = shape.surface(np.linspace(0,2*np.pi,n_resolve,endpoint=False))
    
    all_patches = [patches.Polygon(peri, transform=transforms.Affine2D().rotate(a).translate(r[0], r[1]), zorder=1) 
                   for a, r in zip(angles, pts)]
    
    return collections.PatchCollection(all_patches)
    


def projected_patches(pts, orient, shape=_default_sphere, n_resolve=251, view_dir = np.array([0,0,1]), view_dist=100.0):
    """Generates a PatchCollection of SuperEllipse shapes in 3D projected onto 2D at specified positions and orientations.

    _WIP: only works for spherical particles on spherical surfaces right now._

    :param pts: an [N x 3] array of points of each shape in real space
    :type pts: ndarray
    :param orient: an [N x 4] array of quaternions specifying the orientation of each shape
    :type orient: ndarray
    :param shape: the shape to be drawn at each point, defaults to a sphere of radius 0.5
    :type shape: SuperEllipse, optional
    :param n_resolve: the number of points to use to resolve the perimeter of each shape, defaults to 251
    :type n_resolve: int, optional
    :param view_dir: a 3-element array specifying the viewing direction, defaults to looking down the z-axis
    :type view_dir: ndarray, optional
    :param view_dist: the distance from the viewer to the origin, defaults to 100.0
    :type view_dist: float, optional
    :return: a PatchCollection of the specified shapes at the specified points and orientations, projected onto 2D, as well as a boolean array indicating which particles are in front of the viewer
    :rtype: Tuple[PatchCollection, ndarray]
    """

    e3 = view_dir/np.linalg.norm(view_dir)
    e1 = np.cross(e3,np.array([0,-1,0]))
    e1 = e1/np.linalg.norm(e1,axis=-1)
    e2 = np.cross(e3,e1)
    e2 = e2/np.linalg.norm(e2,axis=-1)
    proj_pts = np.array([pts @ e1, pts @ e2, pts @ e3]).T

    # Create a collection of particle patches with proper orientation and position
    # Each particle is a polygon rotated and translated to match simulation state
    peri = shape.surface(np.linspace(0,2*np.pi,n_resolve,endpoint=False))
    to_render = proj_pts[:, 2] >= 0
    
    # # CODE TO CALCULATE PROJECTED QUATERNION ORIENTATIONS
    # angles = quat_to_angle(orient)
    # proj_orients = np.array([orient @ np.array([0,0,1,0]) for orient in orient])
    # proj_orients = np.array([o/np.linalg.norm(o) for o in proj_orients])
    # proj_angles = np.arctan2(proj_orients[:,1], proj_orients[:,0])
    # proj_angles = proj_angles + np.pi/2.0 

    all_patches = [patches.Polygon(peri,
                                   transform=transforms.Affine2D().scale(sx=(1.0-r[2]/view_dist),sy=r[2]/np.linalg.norm(r)*(1.0-r[2]/view_dist)).rotate(np.atan2(r[0],-r[1])).translate(r[0],r[1]),
                                   zorder=int(r[2])+1) 
                   for r in proj_pts[to_render]]

    return collections.PatchCollection(all_patches), to_render



if __name__ == "__main__":

    # for testing

    pass