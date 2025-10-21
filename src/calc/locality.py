# -*- coding: utf-8 -*-
"""
Contains many helper methods to compute geometric properties of particle configurations, including neighbor lists, handling periodic boundary conditions, and local tangent plane vectors on curved surfaces.
"""

import numpy as np
from scipy.spatial.distance import pdist,squareform

#first coordination shell for discs at close-packing
DEFAULT_CUTOFF = 0.5*(1+np.sqrt(3))


def neighbors(pts:np.ndarray, neighbor_cutoff:float|None = None, num_closest:int|None = None) -> np.ndarray:
    """Determines neighbors in a configuration of particles based on a cutoff distance.

    :param pts: [Nxd] array of particle positions in 'd' dimensions.
    :type pts: ndarray
    :param neighbor_cutoff: specify the distance which defines neighbors. Defaults to halfway between the first coordination peaks for a perfect crystal.
    :type neighbor_cutoff: scalar, optional
    :param num_closest: specify the maximum number of neighbors (within the cutoff) per particle. a.k.a pick that many of the closest neighbors per particle.
    :type num_closest: int, optional
    :return:  [NxN] boolean array indicating which particles are neighbors
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
    :param quat: a list of quaterions encoding particle orientation
    :type quat: ndarray
    :return: the quadrant-corrected 2d angular orientations of the particles
    :rtype: ndarray
    """    
    angles = 2.0*np.arctan2(quat[:,-1],quat[:,0])
    return angles


def stretched_neighbors(pts:np.ndarray, angles:np.ndarray, rx:float = 1.0, ry:float = 1.0, neighbor_cutoff:float = 2.6) ->np.ndarray:
    """Determines neighbors in a configuration of anisotropic particles based on a cutoff distance in the rotated/stretched frame of each particle.

    :param pts: the position of the centers of each anisotropic particle in the configuration
    :type pts: ndarray
    :param angles: the orientation of each anisotropic particle in the configuration
    :type angles: ndarray
    :param rx: the radius of the long axis of the particle (insphere radius times aspect ratio), defaults to 1.0
    :type rx: scalar, optional
    :param ry: the radius of the short axis of the partice (i.e. insphere radius), defaults to 1.0
    :type ry: scalar, optional
    :param neighbor_cutoff: specify the stretched distance which defines neighbors. Defaults to 2.6.
    :type neighbor_cutoff: scalar, optional
    :return: [NxN] boolean array indicating which particles are neighbors
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


def hoomd_matrix_to_box(box:np.ndarray) -> np.ndarray:
    """returns the hoomd box from a given set of basis vectors

    :param box: a matrix containing the basis vectors of a bounding box
    :type box: ndarray
    :return: a length 6 list of box paramters [Lx,Ly,Lz,xy,xz,yz]
    :rtype: ndarray
    """    
    hbox= np.array([box[0,0],box[1,1],box[2,2],box[0,1]/box[1,1],box[0,2]/box[2,2],box[1,2]/box[2,2]])
    if box[2,2]==0:
        hbox[4]=0
        hbox[5]=0
    return hbox


def expand_around_pbc(coords:np.ndarray, basis:np.ndarray, padfrac:float = 0.8)->tuple[np.ndarray,np.ndarray]:
    """
    given a frame and a box basis matrix, returns a larger frame which includes
    surrounding particles from the nearest images, as well as the index relating padded
    particles back to their original image. This will enable methods like
    scipy.voronoi to respect periodic boundary conditions.

    :param coords: a [Nxd] list of particle coordinates in d-dimensions
    :type coords: ndarray
    :param basis: a [dxd] matrix of basis vectors for the simulation box
    :type basis: ndarray
    :param padfrac: the number of extra particles, as a fraction of the total number, to include in the 'pad' of surrounding particles, defaults to 0.8
    :type padfrac: float, optional
    :return: a [(N+N*padfrac) x d ] array of particle coordinates in d-dimensions which respect periodic boundary conditions around the central N particles, as well as a [N+N*padfrac] list of indices relating padded particles back to their original image
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


def padded_neighbors(pts:np.ndarray, basis:np.ndarray, cutoff:float = DEFAULT_CUTOFF, padfrac:float = 0.8) -> np.ndarray:
    """Determines neighbors in a configuration of particles based on a cutoff distance while respecting the periodic boundary condition using :py:meth:`utils.geometry.expand_around_pbc`.

    :param pts: [Nxd] array of particle positions in 'd' dimensions.
    :type pts: ndarray
    :param basis: a [dxd] matrix of basis vectors for the simulation box
    :type basis: ndarray
    :param neighbor_cutoff: specify the distance which defines neighbors. Defaults to halfway between the first coordination peaks for a perfect crystal.
    :type neighbor_cutoff: scalar, optional
    :param padfrac: the number of extra particles, as a fraction of the total number, to include in the 'pad' of surrounding particles, defaults to 0.8
    :return:  [NxN] boolean array indicating which particles are neighbors
    :rtype: ndarray
    """

    pnum = pts.shape[0]
    pts_padded, idx_padded = expand_around_pbc(pts,basis,padfrac=padfrac)
    
    nei_padded = squareform(pdist(pts_padded)) <= cutoff    
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
    """computes two orthogonal unit vectors tangent to the local surface defined by the normal vector (gradient of implicit function):

    .. math:
        \\mathbf{e}_1 = \\frac{\\nabla f(\\mathbf{x}) \\times \\mathbf{r}}{|\\nabla f(\\mathbf{x}) \\times \\mathbf{r}|} \\\\
        \\mathbf{e}_2 = \\frac{\\nabla f(\\mathbf{x}) \\times \\mathbf{e}_1}{|\\nabla f(\\mathbf{x}) \\times \\mathbf{e}_1|}
    
    given :math:`f(x,y,z)=0` defines the surface, and :math:`\\mathbf{r}` is an arbitrary reference vector.

    :param x: an [Nx3] array of positions at which to compute local tangent vectors
    :type x: ndarray
    :param gradient: a function which computes normal vector to a surface
    :type gradient: callable
    :param ref: a reference vector to help define the local frame, defaults to np.array([0,-1,0])
    :type ref: ndarray, optional
    :return: two orthogonal unit vectors tangent to the local surface defined by the gradient at each x
    :rtype: tuple(ndarray, ndarray)
    """
    return np.array([_lvec(x,gradient,ref=ref) for x in pts])


def tangent_connection(pts:np.ndarray,gradient:callable,ref:np.ndarray = np.array([0,-1,0])) -> np.ndarray:
    """computes the complex connection between local tangent planes at each pair of points. This factor takes the form

    .. math::
        R_{ij} = e^{i \\theta_{ij}} \\\\
        \\theta_{ij} = \\arctan\\left(\\frac{\\mathbf{e}_1(\\mathbf{x}_i) \\cdot \\mathbf{e}_2(\\mathbf{x}_j)}{\\mathbf{e}_1(\\mathbf{x}_i) \\cdot \\mathbf{e}_1(\\mathbf{x}_j)}\\right)

    :param pts: an [Nx3] array of positions at which to compute local tangent vectors
    :type pts: ndarray
    :param gradient: a function which computes normal vector to a surface
    :type gradient: callable
    :param ref: a reference vector to help define the local frame, defaults to np.array([0,-1,0])
    :type ref: ndarray, optional
    :return: an [NxN] complex representation of the connection between local tangent planes at each pair of points
    :rtype: ndarray
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

from visuals import SuperEllipse
_default_sphere = SuperEllipse(ax=0.5,ay=0.5,n=2.0)

def central_eta(pts:np.ndarray, box:list, shape:SuperEllipse=_default_sphere, nbins:int=3, bin_width:float=4.0, jac:str='x'):
    """Computes the average area fraction in the central region of a configuration of particles, accounting for the jacobian of the coordinate system:

    .. math::
        \\eta = \\langle\\frac{N\\cdot A_p}{A_{bin}}\\rangle_{{bins}}

    :param pts: an [Nxd] array of particle positions in any dimensions (though only the first two are used)
    :type pts: ndarray
    :param box: a list defining the simulation box in the `gsd <https://gsd.readthedocs.io/en/stable/schema-hoomd.html#chunk-configuration-box>`_ convention
    :type box: array-like
    :param shape: a :py::class:`visuals.shapes.SuperEllipse` instance defining the shape of the particles, defaults to a circle of diameter 1.0
    :type shape: SuperEllipse, optional
    :param nbins: the number of histogram bins to average over in the center of the configuration, defaults to 3
    :type nbins: int, optional
    :param bin_width: the width of each histogram bin, defaults to 4
    :type bin_width: float, optional
    :param jac: the type of jacobian to account for when computing area fraction. Options are 'x' (linear in x), 'y' (linear in y), and 'r' (radial). Defaults to 'x'.
    :type jac: str, optional
    :return: the average area fraction in the central region of the configuration
    :rtype: float
    """

    # determine appropriate coorinate to count, as well as the bin edges and areas based on jacobian type
    match jac:
        case 'x':
            to_bin = pts[:,0]
            x_max = np.abs(to_bin).max()
            xbin = np.arange(0,x_max + bin_width,bin_width)
            bin_edges = np.sort(np.unique(np.concatenate([-xbin,xbin])))
            bin_areas = (bin_edges[1:] - bin_edges[:-1]) * box[1]

        case 'y':
            to_bin = pts[:,1]
            y_max = np.abs(to_bin).max()
            ybin = np.arange(0,y_max + bin_width,bin_width)
            bin_edges = np.sort(np.unique(np.concatenate([-ybin,ybin])))
            bin_areas = (bin_edges[1:] - bin_edges[:-1]) * box[0]

        case 'r':
            to_bin = np.sqrt((pts**2).sum(axis=-1))
            r_max = to_bin.max()
            bin_edges = np.arange(0,r_max + bin_width,bin_width)
            bin_areas = np.pi*(bin_edges[1:]**2 - bin_edges[:-1]**2)
        
        case _:
            raise NotImplementedError(f'jacobian type {jac} not implemented')

    # Compute the histogram of the points
    counts, edges = np.histogram(to_bin, bins=bin_edges, density=False)
    mids = 0.5 * (edges[:-1] + edges[1:])
    eta = counts / bin_areas * shape.area

    # average over the 'nbins' most central histrogram bins
    central_eta = np.mean(eta[np.argsort(np.abs(mids))[:nbins]])

    return central_eta

# def gyration_tensor(pts:np.ndarray, ref:np.ndarray|None = None) -> np.ndarray:
#     """returns the gyration tensor (for principal moments analysis) of an ensemble of particles according to the formula

#     .. math::

#         S_{mn} = \\frac{1}{N}\\sum_{j}r^{(j)}_m r^{(j)}_n

#     where the positions, :math:`r`, are defined in their center of mass reference frame

#     :param pts: [Nxd] array of particle positions in 'd' dimensions,
#     :type pts: ndarray
#     :param ref: point in d-dimensional space from which to reference particle positions, defaults to the mean position of the points. Use this for constraining the center of mass to the surface of a manifold, for instance.
#     :type ref: ndarray , optional
#     :return: the [dxd] gyration tensor of the ensemble
#     :rtype: ndarray
#     """    

#     if ref is None:
#         ref = pts.mean(axis=0)
#     assert (pts.shape[-1],) == ref.shape, 'reference must have same dimesionality as the points'
#     centered = pts - ref
#     gyrate = centered.T @ centered
#     return gyrate/len(pts)


# def gyration_radius(pts:np.ndarray) -> float:
#     """returns the radius of gyration according to the formula

#     .. math::

#         R_g^2 \\equiv \\frac{1}{N}\\sum_k|\\mathbf{r}_k-\\bar{\\mathbf{r}}|^2 = \\frac{1}{N^2}\\sum_{j>i}|\\mathbf{r}_i - \\mathbf{r}_j|^2 

#     where the :math:`j>i` in the summation index indicates that repeated pairs are not summed over

#     :param pts: [Nxd] array of particle positions in 'd' dimensions
#     :type pts: ndarray
#     :return: the radius of gyration of the particles about their center of mass
#     :rtype: scalar
#     """    
#     N = len(pts)
#     dists = pdist(pts)
#     Rg2 = np.sum(dists**2) / (N**2) # pdist accounts for the factor of 2 in the denominator
#     return np.sqrt(Rg2)