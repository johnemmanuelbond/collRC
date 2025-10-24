# -*- coding: utf-8 -*-
"""
Contains many helper methods to compute geometric properties of particle configurations, including neighbor lists, handling periodic boundary conditions, and local tangent plane vectors on curved surfaces.
"""

import numpy as np
from scipy.spatial.distance import pdist,squareform

#first coordination shell for discs at close-packing
DEFAULT_CUTOFF = 0.5*(1+np.sqrt(3))


def neighbors(pts:np.ndarray, neighbor_cutoff:float|None = None, num_closest:int|None = None) -> np.ndarray:
    """Determines neighbors in a configuration of particles based on a cutoff distance:

    .. math::

        n_{ij} = \\delta r_{ij} < r_{cut}

    :param pts: (N,d) array of particle positions in 'D' dimensions.
    :type pts: ndarray
    :param neighbor_cutoff: specify the distance which defines neighbors. Defaults to halfway between the first coordination peaks for a perfect crystal.
    :type neighbor_cutoff: scalar, optional
    :param num_closest: specify the maximum number of neighbors (within the cutoff) per particle. a.k.a pick that many of the closest neighbors per particle.
    :type num_closest: int, optional
    :return:  (N,N) boolean array indicating which particles are neighbors
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

    :param quat: an (N,4) array of quaterions encoding particle orientation
    :type quat: ndarray
    :return: an (N,) array of quadrant-corrected 2d angular orientations of the particles
    :rtype: ndarray
    """    
    angles = 2.0*np.arctan2(quat[:,-1],quat[:,0])
    return angles


def stretched_neighbors(pts:np.ndarray, angles:np.ndarray, rx:float = 1.0, ry:float = 1.0, neighbor_cutoff:float = 2.6) ->np.ndarray:
    """Determines neighbors in a configuration of anisotropic particles based on a cutoff distance in the rotated/stretched frame of each particle using the equaitons defined in `(Torrez-Diaz Soft Matter, 2022) <https://doi.org/10.1039/D1SM01523K>`_:

    .. math::

        n_{ij} = \\sqrt{\\big(\\mathbf{\\delta r_ij} \\cdot \\hat{\\mathbf{x_i}}/r_x\\big) + \\big(\\mathbf{\\delta r_ij} \\cdot \\hat{\\mathbf{y_i}}/r_y\\big)} < n_{cut}

    Where :math:`\\hat{\\mathbf{x_i}} = \\cos(\\theta_i)\\hat{\\mathbf{x}} + \\sin(\\theta_i)\\hat{\\mathbf{y}}` and :math:`\\hat{\\mathbf{y_i}} = -\\sin(\\theta_i)\\hat{\\mathbf{x}} + \\cos(\\theta_i)\\hat{\\mathbf{y}}` are the local unit vectors along the long (:math:`r_x`) and short (:math:`r_y`) axes of particle :math:`i`, respectively.

    :param pts: an (N,d) array of the positions each anisotropic particle in the configuration
    :type pts: ndarray
    :param angles: an (N,) array of the orientation of each anisotropic particle in the configuration
    :type angles: ndarray
    :param rx: the radius of the long axis of the particle (insphere radius times aspect ratio), defaults to 1.0
    :type rx: scalar, optional
    :param ry: the radius of the short axis of the partice (i.e. insphere radius), defaults to 1.0
    :type ry: scalar, optional
    :param neighbor_cutoff: specify the dimemsionless stretched distance which defines neighbors. Defaults to 2.6.
    :type neighbor_cutoff: scalar, optional
    :return: (N,N) boolean array indicating which particles are neighbors
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


def expand_around_pbc(coords:np.ndarray, basis:np.ndarray, padfrac:float = 0.8)->tuple[np.ndarray,np.ndarray]:
    """
    given a frame and a box basis matrix, returns a larger frame which includes
    surrounding particles from the nearest images, as well as the index relating padded
    particles back to their original image. This will enable methods like
    scipy.voronoi to respect periodic boundary conditions.

    :param coords: a (N,d) array of particle coordinates in d-dimensions
    :type coords: ndarray
    :param basis: a (d,d) matrix of basis vectors for the simulation box
    :type basis: ndarray
    :param padfrac: the number of extra particles, as a fraction of the total number, to include in the 'pad' of surrounding particles, defaults to 0.8
    :type padfrac: float, optional
    :return: a ((N+N*padfrac), d) array of particle coordinates in d-dimensions which respect periodic boundary conditions around the central N particles, as well as a (N+N*padfrac,) array of indices relating padded particles back to their original image
    :rtype: np.ndarray, np.ndarray
    """    

    pnum = coords.shape[0]
    if basis[2,2]==0: basis[2,2]=1

    frame_basis = (np.linalg.inv(basis) @ coords.T).T
    expanded = np.array([
        *(frame_basis+np.array([ 1, 0, 0])),*(frame_basis+np.array([ 0, 1, 0])),
        *(frame_basis+np.array([-1, 0, 0])),*(frame_basis+np.array([ 0,-1, 0])),
        *(frame_basis+np.array([ 1, 1, 0])),*(frame_basis+np.array([ 1,-1, 0])),
        *(frame_basis+np.array([-1, 1, 0])),*(frame_basis+np.array([-1,-1, 0]))
        ])

    pad_idx = np.argsort(np.max(np.abs(expanded),axis=-1))[:(int(padfrac*pnum))]
    pad = (basis @ expanded[pad_idx].T).T
    
    return np.array([*coords,*pad]), np.array([*np.arange(pnum),*(pad_idx%pnum)])


def padded_neighbors(pts:np.ndarray, basis:np.ndarray, neighbor_cutoff:float = DEFAULT_CUTOFF, padfrac:float = 0.8) -> np.ndarray:
    """Determines neighbors in a configuration of particles based on a cutoff distance while respecting the periodic boundary condition using :py:meth:`expand_around_pbc`:

    .. math::

        n_{ij} = \\delta r_{ij} < r_{cut} || \\delta r_{ij'} < r_{cut}
    
    For particles :math:`j'` which are periodic images of particle :math:`j`.

    :param pts: (N,d) array of particle positions in 'd' dimensions.
    :type pts: ndarray
    :param basis: a (d,d) matrix of basis vectors for the simulation box
    :type basis: ndarray
    :param neighbor_cutoff: specify the distance which defines neighbors. Defaults to halfway between the first coordination peaks for a perfect crystal.
    :type neighbor_cutoff: scalar, optional
    :param padfrac: the number of extra particles, as a fraction of the total number, to include in the 'pad' of surrounding particles, defaults to 0.8
    :return:  (N,N) boolean array indicating which particles are neighbors
    :rtype: ndarray
    """

    pnum = pts.shape[0]
    pts_padded, idx_padded = expand_around_pbc(pts,basis,padfrac=padfrac)

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
    """computes two orthogonal unit vectors tangent to the local surface at a point :math:`\\mathbf{r}_i`:

    .. math::

        \\mathbf{e}_{1,i} = \\frac{\\nabla f(\\mathbf{r}_i) \\times \\hat{\\mathbf{\\gamma}}}{|\\nabla f(\\mathbf{r}_i) \\times \\hat{\\mathbf{\\gamma}}|} \\\\
        \\mathbf{e}_{2,i} = \\frac{\\nabla f(\\mathbf{r}_i) \\times \\mathbf{e}_1}{|\\nabla f(\\mathbf{r}_i) \\times \\mathbf{e}_1|}
    
    Given the implicit function :math:`f(x,y,z)=0` defines the surface, its gradient :math:`\\nabla f` defines the normal vector, and :math:`\\hat{\\mathbf{\\gamma}}` is an arbitrary reference unit vector.

    :param pts: an (N,3) array of positions at which to compute local tangent vectors
    :type pts: ndarray
    :param gradient: a function which computes normal vector to a surface
    :type gradient: callable
    :param ref: a reference vector to help define the local frame, defaults to (0,-1,0) (so that :math:`e_1=\\hat{x}` usually)
    :type ref: ndarray, optional
    :return: an (N,2,3) array of two orthogonal unit vectors tangent to the local surface defined by the gradient at each x
    :rtype: ndarray
    """
    return np.array([_lvec(x,gradient,ref=ref) for x in pts])


def tangent_connection(pts:np.ndarray,gradient:callable,ref:np.ndarray = np.array([0,-1,0])) -> np.ndarray:
    """computes the complex connection between local tangent planes at each pair of points. This factor takes the form

    .. math::

        R_{ij} = e^{i \\theta_{ij}} = \\exp\\bigg[i\\tan^{-1}\\bigg(\\frac{e_{1,i} \\cdot e_{2,j}}{e_{1,i} \\cdot e_{1,j}}\\bigg)\\bigg]

    where the local tangent vectors :math:`e_{1,i}` and :math:`e_{2,i}` are computed using :py:meth:`local_vectors`. :math:`R_{ij}` describes how to rotate complex numbers defined relative to :math:`e_{1,i}` into those defined relative to :math:`e_{1,j}`.

    :param pts: an (N,3) array of positions at which to compute local tangent vectors
    :type pts: ndarray
    :param gradient: a function which computes normal vector to a surface
    :type gradient: callable
    :param ref: a reference vector to help define the local frame, defaults to np.array([0,-1,0])
    :type ref: ndarray, optional
    :return: an (N,N) complex representation of the connection between local tangent planes at each pair of points
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