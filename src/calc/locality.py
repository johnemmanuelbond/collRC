# -*- coding: utf-8 -*-
"""
Contains many helper methods to compute geometric properties of particle configurations, including neighbor lists, handling periodic boundary conditions, and local tangent plane vectors on curved surfaces.
"""

import numpy as np
import itertools
from scipy.spatial.distance import pdist,cdist,squareform

#first coordination shell for discs at close-packing
DEFAULT_CUTOFF = 0.5*(1+np.sqrt(3))


def neighbors(pts:np.ndarray, neighbor_cutoff:float|None = None, num_closest:int|None = None) -> np.ndarray:
    """Determines neighbors in a configuration of particles based on a cutoff distance:

    .. math::

        n_{jk} = \\delta r_{jk} < r_{cut}

    :param pts: :math:`[N,d]` array of particle positions in 'd' dimensions.
    :type pts: ndarray
    :param neighbor_cutoff: specify the distance which defines neighbors. Defaults to halfway between the first coordination peaks for a perfect sixfold crystal.
    :type neighbor_cutoff: scalar, optional
    :param num_closest: specify the maximum number of neighbors (within the cutoff) per particle. a.k.a pick that many of the closest neighbors per particle.
    :type num_closest: int, optional
    :return: :math:`[N,N]` boolean array indicating which particles are neighbors
    :rtype: ndarray
    """

    dists = squareform(pdist(pts))
    #determines neighbors using a cutoff
    if neighbor_cutoff is None:
        cut = np.ones_like(dists)*DEFAULT_CUTOFF
    else:
        cut = np.ones_like(dists)*neighbor_cutoff
    
    if num_closest is not None:
        cut_i = np.sort(dists,axis=0)[num_closest+2]
        cut = np.array([cut_i,cut]).min(axis=0)

    
    nei = dists < cut
    nei[np.eye(len(pts))>0] = False
    return nei


def quat_to_angle(quat:np.ndarray) -> np.ndarray:
    """

    Finds the angle :math:`\\theta` about the z-axis from a quaternion representation in 2D: :math:`q = \\cos(\\theta/2) + \\sin(\\theta/2)\\mathbf{k}`

    :param quat: an :math:`[N,4]` array of quaterions encoding particle orientation
    :type quat: ndarray
    :return: an :math:`[N,]` array of quadrant-corrected 2d angular orientations of the particles
    :rtype: ndarray
    """    
    angles = 2.0*np.arctan2(quat[:,-1],quat[:,0])
    return angles


def stretched_neighbors(pts:np.ndarray, angles:np.ndarray, rx:float = 1.0, ry:float = 1.0, neighbor_cutoff:float = 2.6) ->np.ndarray:
    """Determines neighbors in a configuration of anisotropic particles based on a cutoff distance in the rotated/stretched frame of each particle using the equaitons defined in `(Torrez-Diaz Soft Matter, 2022) <https://doi.org/10.1039/D1SM01523K>`_:

    .. math::

        n_{jk} = \\sqrt{\\big(\\mathbf{\\delta r_jk} \\cdot \\hat{\\mathbf{x_j}}/r_x\\big)^2 + \\big(\\mathbf{\\delta r_jk} \\cdot \\hat{\\mathbf{y_j}}/r_y\\big)^2} < n_{cut}

    Where :math:`\\hat{\\mathbf{x_j}} = \\cos(\\theta_j)\\hat{\\mathbf{x}} + \\sin(\\theta_j)\\hat{\\mathbf{y}}` and :math:`\\hat{\\mathbf{y_j}} = -\\sin(\\theta_j)\\hat{\\mathbf{x}} + \\cos(\\theta_j)\\hat{\\mathbf{y}}` are the local unit vectors along the long (:math:`r_x`) and short (:math:`r_y`) axes of particle :math:`j` rotated to it's orientation :math:`\\theta_j`, respectively.

    :param pts: an :math:`[N,d]` array of the positions each anisotropic particle in the configuration
    :type pts: ndarray
    :param angles: an :math:`[N,]` array of the orientation of each anisotropic particle in the configuration
    :type angles: ndarray
    :param rx: the radius of the long axis of the particle (insphere radius times aspect ratio), defaults to 1.0
    :type rx: scalar, optional
    :param ry: the radius of the short axis of the partice (i.e. insphere radius), defaults to 1.0
    :type ry: scalar, optional
    :param neighbor_cutoff: specify the dimemsionless stretched distance which defines neighbors. Defaults to 2.6.
    :type neighbor_cutoff: scalar, optional
    :return: :math:`[N,N]` boolean array indicating which particles are neighbors
    :rtype: ndarray
    """    

    pnum = pts.shape[0]
    i,j = np.mgrid[0:pnum,0:pnum]
    dr_vec = pts[i]-pts[j]
    trig = [np.cos(angles[i]),np.sin(angles[i])]
    xs =  trig[0]*dr_vec[:,:,0]+trig[1]*dr_vec[:,:,1]
    ys = -trig[1]*dr_vec[:,:,0]+trig[0]*dr_vec[:,:,1]

    # format into NxN boolean array
    nei = np.sqrt((xs/rx)**2+(ys/ry)**2) < neighbor_cutoff
    nei[np.eye(len(pts))>0] = False

    return nei


def matrix_to_box(box:np.ndarray) -> np.ndarray:
    """
    .. math::

        \\begin{bmatrix} L_x & xy*L_y & xz*L_z \\\\ 
        0 & L_y & yz*L_z \\\\ 
        0 & 0 & L_z 
        \\end{bmatrix} \\rightarrow [L_x,L_y,L_z,xy,xz,yz]

    :param box: a matrix containing the basis vectors of a bounding box
    :type box: ndarray
    :return: a length 6 list of box parameters
    :rtype: ndarray
    """    
    hbox= np.array([box[0,0],box[1,1],box[2,2],box[0,1]/box[1,1],box[0,2]/box[2,2],box[1,2]/box[2,2]])
    if box[2,2]==0:
        hbox[4]=0
        hbox[5]=0
    return hbox

def box_to_matrix(box:list) -> np.ndarray:
    """
    .. math::

        [L_x,L_y,L_z,xy,xz,yz] \\rightarrow \\begin{bmatrix} L_x & xy*L_y & xz*L_z \\\\ 
        0 & L_y & yz*L_z \\\\ 
        0 & 0 & L_z 
        \\end{bmatrix}

    :param box: a length 6 list of box parameters
    :type box: array_like
    :return: a matrix containing the basis vectors for the equivalent hoomd box
    :rtype: ndarray
    """    
    return np.array([[box[0],box[3]*box[1],box[4]*box[2]],[0,box[1],box[5]*box[2]],[0,0,box[2]]])


def expand_around_pbc(coords:np.ndarray, basis:np.ndarray, padfrac:float = 0.8, max_dist:float = None)->tuple[np.ndarray,np.ndarray]:
    """
    Given a frame and a box basis matrix, returns a larger frame which includes surrounding particles from the nearest images, as well as the index relating padded particles back to their original image. This will enable methods like :py:meth:`scipy.voronoi` to respect periodic boundary conditions.

    :param coords: a :math:`[N,d]` array of particle coordinates in d-dimensions
    :type coords: ndarray
    :param basis: a :math:`[d,d]` matrix of basis vectors for the simulation box. A 2D box should have basis[2,2]=0, otherwise the 3D case is assumed and the function prepares 27 periodic images (rather than only 9 in 2D).
    :type basis: ndarray
    :param padfrac: the number of extra particles, as a fraction of the total number, to include in the 'pad' of surrounding particles, defaults to 0.8
    :type padfrac: float, optional
    :param max_dist: alternatively to padfrac, specify a maximum distance from original particles to include surrounding particles. defaults to None
    :type max_dist: float, optional
    :return: a :math:`[N', d]` array of particle coordinates in d-dimensions which respect periodic boundary conditions around the central N particles, as well as a :math:`[N',]` array of indices relating padded particles back to their original image
    :rtype: ndarray, ndarray
    """    

    pnum = coords.shape[0]
    e1,e2,e3 = np.eye(3)
    is_2d = basis[2,2]==0
    b = basis.copy()
    if is_2d: b[2,2]=1

    f = (np.linalg.inv(b) @ coords.T).T

    if max_dist is None:
        if is_2d:
            expanded = np.array([
                *(f+e1),*(f+e2),
                *(f-e1),*(f-e2),
                *(f+e1+e2),*(f+e1-e2),
                *(f-e1+e2),*(f-e1-e2)
                ])
            pad_idx = np.argsort(np.abs(expanded[:,:2]).max(axis=-1))[:(int(padfrac*pnum))]
        else:
            expanded = np.array([
                *(f+e1),*(f+e2),*(f+e3),
                *(f-e1),*(f-e2),*(f-e3),
                *(f+e1+e2),*(f+e1-e2),*(f-e1+e2),*(f-e1-e2),
                *(f+e1+e3),*(f+e1-e3),*(f-e1+e3),*(f-e1-e3),
                *(f+e2+e3),*(f+e2-e3),*(f-e2+e3),*(f-e2-e3),
                *(f+e1+e2+e3),*(f+e1+e2-e3),*(f+e1-e2+e3),*(f+e1-e2-e3),
                *(f-e1+e2+e3),*(f-e1+e2-e3),*(f-e1-e2+e3),*(f-e1-e2-e3)
                ])
            pad_idx = np.argsort(np.abs(expanded).max(axis=-1))[:(int(padfrac*pnum))]
        
        pad = (b @ expanded[pad_idx].T).T

    else:
        assert padfrac is not None, "Either max_dist or padfrac must be specified"
        Lx,Ly,Lz = np.linalg.norm(basis,axis=1)
        max_x,max_y,max_z = np.max(f,axis=0)
        min_x,min_y,min_z = np.min(f,axis=0)
        pad_xp = (max_x >=  (Lx/2 - max_dist)/Lx)
        pad_xm = (min_x <= -(Lx/2 - max_dist)/Lx)
        pad_yp = (max_y >=  (Ly/2 - max_dist)/Ly)
        pad_ym = (min_y <= -(Ly/2 - max_dist)/Ly)
        pad_zp = (not is_2d) and (max_z >=  (Lz/2 - max_dist)/Lz)
        pad_zm = (not is_2d) and (min_z <= -(Lz/2 - max_dist)/Lz)

        if not np.any([pad_xp,pad_xm,pad_yp,pad_ym,pad_zp,pad_zm]):
            return coords, np.arange(pnum)

        expanded = []
        if pad_xp: expanded = [*expanded, *(f + e1)]
        if pad_xm: expanded = [*expanded, *(f - e1)]
        if pad_yp: expanded = [*expanded, *(f + e2)]
        if pad_ym: expanded = [*expanded, *(f - e2)]
        if pad_xp and pad_yp: expanded = [*expanded, *(f + e1 + e2)]
        if pad_xp and pad_ym: expanded = [*expanded, *(f + e1 - e2)]
        if pad_xm and pad_yp: expanded = [*expanded, *(f - e1 + e2)]
        if pad_xm and pad_ym: expanded = [*expanded, *(f - e1 - e2)]
        if not is_2d:
            if pad_zp: expanded = [*expanded, *(f + e3)]
            if pad_zm: expanded = [*expanded, *(f - e3)]
            if pad_xp and pad_zp: expanded = [*expanded, *(f + e1 + e3)]
            if pad_xp and pad_zm: expanded = [*expanded, *(f + e1 - e3)]
            if pad_xm and pad_zp: expanded = [*expanded, *(f - e1 + e3)]
            if pad_xm and pad_zm: expanded = [*expanded, *(f - e1 - e3)]
            if pad_yp and pad_zp: expanded = [*expanded, *(f + e2 + e3)]
            if pad_yp and pad_zm: expanded = [*expanded, *(f + e2 - e3)]
            if pad_ym and pad_zp: expanded = [*expanded, *(f - e2 + e3)]
            if pad_ym and pad_zm: expanded = [*expanded, *(f - e2 - e3)]
            if pad_xp and pad_yp and pad_zp: expanded = [*expanded, *(f + e1 + e2 + e3)]
            if pad_xp and pad_yp and pad_zm: expanded = [*expanded, *(f + e1 + e2 - e3)]
            if pad_xp and pad_ym and pad_zp: expanded = [*expanded, *(f + e1 - e2 + e3)]
            if pad_xp and pad_ym and pad_zm: expanded = [*expanded, *(f + e1 - e2 - e3)]
            if pad_xm and pad_yp and pad_zp: expanded = [*expanded, *(f - e1 + e2 + e3)]
            if pad_xm and pad_yp and pad_zm: expanded = [*expanded, *(f - e1 + e2 - e3)]
            if pad_xm and pad_ym and pad_zp: expanded = [*expanded, *(f - e1 - e2 + e3)]
            if pad_xm and pad_ym and pad_zm: expanded = [*expanded, *(f - e1 - e2 - e3)]

        expanded = (b @ np.array(expanded).T).T
        dist_to_coords = cdist(expanded,coords)
        pad_idx = np.where(dist_to_coords.min(axis=-1) <= max_dist)[0]
        pad = expanded[pad_idx]

    return np.array([*coords,*pad]), np.array([*np.arange(pnum),*(pad_idx%pnum)])


def padded_neighbors(pts:np.ndarray, basis:np.ndarray, neighbor_cutoff:float = DEFAULT_CUTOFF, padfrac:float=None) -> np.ndarray:
    """Determines neighbors in a configuration of particles based on a cutoff distance while respecting the periodic boundary condition using :py:meth:`expand_around_pbc`:

    .. math::

        n_{jk} = \\delta r_{jk} < r_{cut} \\big|\\big| \\delta r_{jk'} < r_{cut}
    
    For particles :math:`k'` which are periodic images of particle :math:`k`.

    :param pts: :math:`[N,d]` array of particle positions in 'd' dimensions.
    :type pts: ndarray
    :param basis: a :math:`[d,d]` matrix of basis vectors for the simulation box
    :type basis: ndarray
    :param neighbor_cutoff: specify the distance which defines neighbors. Defaults to halfway between the first coordination peaks for a perfect sixfold crystal.
    :type neighbor_cutoff: scalar, optional
    :param padfrac: the number of extra particles, as a fraction of the total number, to include in the 'pad' of surrounding particles, This may be computatinally faster than using :py:meth:`expand_around_pbc` with :code:`max_dist`. Defaults to None
    :return:  :math:`[N,N]` boolean array indicating which particles are neighbors
    :rtype: ndarray
    """

    pnum = pts.shape[0]
    if padfrac is not None: md = None
    else: md = neighbor_cutoff*1.1
    pts_padded, idx_padded = expand_around_pbc(pts,basis,max_dist=md,padfrac=padfrac)

    nei_padded = squareform(pdist(pts_padded)) <= neighbor_cutoff
    nei_padded[np.eye(pts_padded.shape[0])==1]=False
    nei_real = nei_padded[:pnum,:pnum]
    for i,n in enumerate(nei_padded[:pnum]):
        nei_real[i][np.unique(idx_padded[n])] = True

    return nei_real


def _lvec(x,gradient,ref = np.array([0,-1,0])):
    grad = gradient(x)
    grad = grad/np.linalg.norm(grad)
    ref = ref/np.linalg.norm(ref)
    # if np.abs(grad@ref) > 0.7:
    #     ref = np.array([0,0,-np.sign(grad@ref)])
    e1 = np.cross(grad,ref)
    e1 = e1/np.linalg.norm(e1,axis=-1)
    e2 = np.cross(grad,e1)
    e2 = e2/np.linalg.norm(e2,axis=-1)

    return e1, e2

def local_vectors(pts:np.ndarray,gradient:callable,ref:np.ndarray = np.array([0,-1,0])):
    """Computes two orthogonal unit vectors tangent to the local surface at each point :math:`\\mathbf{r}_j`:

    .. math::

        \\mathbf{e}_{1,j} = \\frac{\\nabla f(\\mathbf{r}_j) \\times \\hat{\\mathbf{\\gamma}}}{|\\nabla f(\\mathbf{r}_j) \\times \\hat{\\mathbf{\\gamma}}|} \\\\
        \\mathbf{e}_{2,j} = \\frac{\\nabla f(\\mathbf{r}_j) \\times \\mathbf{e}_1}{|\\nabla f(\\mathbf{r}_j) \\times \\mathbf{e}_1|}
    
    Given the implicit function :math:`f(x,y,z)=0` defines the surface, its gradient :math:`\\nabla f` defines the normal vector, and :math:`\\hat{\\mathbf{\\gamma}}` is an arbitrary reference unit vector.

    :param pts: an :math:`[N,3]` array of positions at which to compute local tangent vectors
    :type pts: ndarray
    :param gradient: a function which computes normal vector to a surface
    :type gradient: callable
    :param ref: a reference vector to help define the local frame, defaults to :math:`(0,-1,0)` (so that :math:`e_1=\\hat{x}` usually)
    :type ref: ndarray, optional
    :return: an :math:`[N,2,3]` array of two orthogonal unit vectors tangent to the local surface defined by the gradient at each x
    :rtype: ndarray
    """
    return np.array([_lvec(x,gradient,ref=ref) for x in pts])


def tangent_connection(pts:np.ndarray,gradient:callable,ref:np.ndarray = np.array([0,-1,0])) -> np.ndarray:
    """
    Computes the complex connection between local tangent planes at each pair of points. This factor takes the form

    .. math::

        R_{jk} = e^{i \\theta_{jk}} = \\exp\\bigg[i\\tan^{-1}\\bigg(\\frac{e_{1,j} \\cdot e_{2,k}}{e_{1,j} \\cdot e_{1,k}}\\bigg)\\bigg]

    where the local tangent vectors :math:`e_{1,j}` and :math:`e_{2,j}` are computed using :py:meth:`local_vectors`. :math:`R_{jk}` describes how to rotate complex numbers defined relative to :math:`e_{1,j}` into those defined relative to :math:`e_{1,k}`. Due to parallel transport on curved surfaces, this factor is only really useful when :math:`j` and :math:`k` are nearby points on the surface, and is more and more approximate as the distance between points increases.

    :param pts: an :math:`[N,3]` array of positions at which to compute local tangent vectors
    :type pts: ndarray
    :param gradient: a function which computes normal vector to a surface
    :type gradient: callable
    :param ref: a reference vector to help define the local frame, defaults to :math:`-\\hat{\\mathbf{y}}` so that :math:`e_1=\\hat{\\mathbf{x}}` usually
    :type ref: ndarray, optional
    :return: an :math:`[N,N]` complex representation of the connection between local tangent planes at each pair of points
    :rtype: ndarray[complex]
    """

    pnum = pts.shape[0]
    ii,jj = np.mgrid[0:pnum,0:pnum]

    # collect local tangent vectors at each particle
    lvec = local_vectors(pts,gradient,ref=ref)
    e1i = lvec[:,0,:][ii]
    e2i = lvec[:,1,:][ii]
    e1j = lvec[:,0,:][jj]
    e2j = lvec[:,1,:][jj]

    xref = np.sum(e1i*e1j,axis=-1)
    yref = np.sum(e1i*e2j,axis=-1)
    thetas = np.arctan2(yref,xref)

    return np.exp(1j*thetas)