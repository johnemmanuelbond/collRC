# -*- coding: utf-8 -*-
"""
Contains many order parameters for characterizing colloidal ensembles.
"""

import numpy as np
from scipy.spatial.distance import pdist,squareform
from utils.geometry import expand_around_pbc

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


def padded_neighbors(pts, basis, cutoff=DEFAULT_CUTOFF, padfrac=0.8):
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


def bond_order(pts:np.ndarray, nei_bool:np.ndarray|None = None, order:int = 6) -> tuple[np.ndarray,float]:
    """Calculates the local and global bond orientational order parameter of each particle in a 2D configuration with respect to the y axis. The local n-fold bond orientaitonal order for a particle :math:`j` is:

    .. math::

        \\psi_j = \\frac{1}{N_j}\\sum_k\\psi_{jk}=\\frac{1}{N_j}\\sum_ke^{in\\theta_{jk}}

    Where the sum is over all :math:`N_j` neighboring particles :math:`k` to particle :math:`j` and :math:`\\theta_{jk}` is the angle between particles :math:`j` and :math:`k`. Similarly, the global n-fold bond orientational order is:

    .. math::

        \\psi_g = \\frac{1}{N}\\frac{1}{N_j}\\sum_{jk}\\psi_{jk}=\\frac{1}{N}\\frac{1}{N_j}\\sum_{jk}e^{in\\theta_{jk}}

    Where the sum is over all unique bonds between particles :math:`j` and :math:`k`.

    :param pts: [Nxd] array of particle positions in 'd' dimensions, though the calculation only access the first two dimensions.
    :type pts: ndarray
    :param nei_bool: a [NxN] boolean array indicating neighboring particles, will calculate neighbors using the default cutoff value if none is given.
    :type nei_bool: ndarray, optional
    :param order: n-fold order defines the argument of the complex number used to calculate psi_n, defaults to 6
    :type order: int, optional
    :return: [N] array of complex bond orientational order parameters, and the norm of their mean.
    :rtype: tuple(ndarray, scalar)
    """

    pnum = pts.shape[0]
    i,j = np.mgrid[0:pnum,0:pnum]
    dr_vec = pts[i]-pts[j]
    if nei_bool is None:
        nei_bool = squareform(pdist(pts))<=DEFAULT_CUTOFF
        nei_bool[i==j]=False

    # double-check for degenerate neighborless states
    if not np.any(nei_bool): return np.zeros(pnum),0
    # get neighbor count per particle, and cumulative
    sizes = np.sum(nei_bool,axis=-1)
    csizes = np.cumsum(sizes)

    #assemble lists of vectors to evaluate angles between (us and vs)
    bonds = dr_vec[nei_bool]
    xs = bonds[:,0]
    ys = bonds[:,1]
    angles = np.arctan2(ys,xs)
    
    # now we compute psi_ij for each of the bonds
    psi_ij = np.exp(1j*order*angles)
    # pick out only the last summed psi for each particle
    psi_csum = np.array([0,*np.cumsum(psi_ij)])
    # subtract off the previous particles' summed psi values
    c_idx = np.array([0,*csizes])
    psi = psi_csum[c_idx[1:]]-psi_csum[c_idx[:-1]]

    #return the neighbor-averaged psi
    psi[sizes>0]*=1/sizes[sizes>0]
    psi[sizes==0]=0
    
    return psi, np.abs(np.mean(psi_ij))





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


def stretched_bond_order(pts:np.ndarray, angles:np.ndarray, nei_bool:np.ndarray|None = None, rx:float = 1.0, ry:float = 1.0, order:int = 6) -> tuple[np.ndarray,float]:
    """Computes the local and global stretched bond orientational order parameter. This calculation rotates coordinates into a frame of reference stretched according to the long and short axes of each particle according to equations given in `(Torrez-Diaz Soft Matter, 2022) <https://doi.org/10.1039/D1SM01523K>`_.

    :param pts: [Nxd] array of particle positions in 'd' dimensions, though the calculation only access the first two dimensions.
    :type pts: ndarray
    :param angles: the orientation of each anisotropic particle in the configuration
    :type angles: ndarray
    :param nei_bool: a [NxN] boolean array indicating neighboring particles, will calculate stretched neighbors using the default cutoff value if none is given.
    :type nei_bool: ndarray, optional
    :param rx: the radius of the long axis of the particle (insphere radius times aspect ratio), defaults to 1.0
    :type rx: scalar, optional
    :param ry: the radius of the short axis of the partice (i.e. insphere radius), defaults to 1.0
    :type ry: scalar, optional
    :param order: n-fold order defines the argument of the complex number used to calculate psi_n, defaults to 6
    :type order: int, optional
    :return: [N] array of complex bond orientational order parameters, and the norm of their mean.
    :rtype: tuple(ndarray, scalar)
    """    
    pnum = pts.shape[0]
    i,j = np.mgrid[0:pnum,0:pnum]
    dr_vec = pts[i]-pts[j]
    if nei_bool is None:
        nei_bool = stretched_neighbors(pts,angles,rx=rx,ry=ry)

    # double-check for degenerate neighborless states
    if not np.any(nei_bool): return np.zeros(pnum),0
    # get neighbor count per particle, and cumulative
    sizes = np.sum(nei_bool,axis=-1)
    csizes = np.cumsum(sizes)

    #assemble lists of vectors to evaluate angles between (us and vs)
    bonds = dr_vec[nei_bool]
    orient = angles[i[nei_bool]]
    trig = [np.cos(orient), np.sin(orient)]

    xs =  trig[0]*bonds[:,0]+trig[1]*bonds[:,1]
    ys = -trig[1]*bonds[:,0]+trig[0]*bonds[:,1]
    angles = np.arctan2(ys/ry,xs/rx) + orient

    # now we compute psi_ij for each of the bonds
    psi_ij = np.exp(1j*order*angles)
    # pick out only the last summed psi for each particle
    psi_csum = np.array([0,*np.cumsum(psi_ij)])
    # subtract off the previous particles' summed psi values
    c_idx = np.array([0,*csizes])
    psi = psi_csum[c_idx[1:]]-psi_csum[c_idx[:-1]]

    #return the neighbor-averaged psi
    psi[sizes>0]*=1/sizes[sizes>0]
    psi[sizes==0]=0
    
    return psi, np.abs(np.mean(psi_ij))





def local_vectors(x,gradient,ref = np.array([0,-1,0])):
    """computes two orthogonal unit vectors tangent to the local surface defined by the normal vector (gradient of implicit function):

    .. math:
        \\mathbf{e}_1 = \\frac{\\nabla f(\\mathbf{x}) \\times \\mathbf{r}}{|\\nabla f(\\mathbf{x}) \\times \\mathbf{r}|} \\\\
        \\mathbf{e}_2 = \\frac{\\nabla f(\\mathbf{x}) \\times \\mathbf{e}_1}{|\\nabla f(\\mathbf{x}) \\times \\mathbf{e}_1|}
    
    given :math:`f(x,y,z)=0` defines the surface, and :math:`\\mathbf{r}` is an arbitrary reference vector.

    :param x: the positions at which to compute local tangent vectors
    :type x: ndarray
    :param gradient: a function which computes the gradient at position x
    :type gradient: callable
    :param ref: a reference vector to help define the local frame, defaults to np.array([0,-1,0])
    :type ref: ndarray, optional
    :return: two orthogonal unit vectors tangent to the local surface defined by the gradient at each x
    :rtype: tuple(ndarray, ndarray)
    """

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


def tangent_connection(pts,gradient,ref=np.array([0,-1,0]), nei_bool:np.ndarray|None = None):
    """Computes the connection between local tangent planes at each pair of particles in a 2D configuration on a surface defined by an implicit function. The connection is given by the complex phase factor:

    .. math:
        R_{ij} = e^{i\\theta_{ij}} \\\\
        \\theta_{ij} = \\arctan\\bigg( \\frac{\\mathbf{e}_1^i \\cdot \\mathbf{e}_2^j}{\\mathbf{e}_1^i \\cdot \\mathbf{e}_1^j} \\bigg)

    where :math:`\\mathbf{e}_1` and :math:`\\mathbf{e}_2` are the two orthogonal unit vectors tangent to the local surface at each particle position.

    :param pts: an [N x 3] array of points of each shape in real space
    :type pts: ndarray
    :param gradient: a function which computes the gradient at position x
    :type gradient: callable
    :param ref: a reference vector to help define the local frame, defaults to np.array([0,-1,0])
    :type ref: ndarray, optional
    :param nei_bool: a boolean array indicating neighboring particles, defaults to None
    :type nei_bool: ndarray|None
    :return: a tuple of local tangent vectors and complex phase factors
    :rtype: tuple(ndarray, ndarray)
    """

    pnum = pts.shape[0]
    ii,jj = np.mgrid[0:pnum,0:pnum]

    # collect local tangent vectors at each particle
    lvec = np.array([local_vectors(x,gradient,ref=ref) for x in pts])
    e1i = lvec[:,0,:][ii]
    e2i = lvec[:,1,:][ii]
    e1j = lvec[:,0,:][jj]
    e2j = lvec[:,1,:][jj]

    xref = np.sum(e1i*e1j,axis=-1)
    yref = np.sum(e1i*e2j,axis=-1)
    thetas = np.arctan2(yref,xref)

    if nei_bool is None:
        nei_bool = squareform(pdist(pts))<=DEFAULT_CUTOFF
        nei_bool[ii==jj]=False

    return lvec, np.exp(1j*thetas)*nei_bool


def projected_bond_order(pts, gradient, nei_bool=None, order=6, ref=np.array([0,-1,0])):
    pnum = pts.shape[0]
    ii,jj = np.mgrid[0:pnum,0:pnum]
    dr_vec = pts[jj]-pts[ii]
    if nei_bool is None:
        nei_bool = squareform(pdist(pts))<=DEFAULT_CUTOFF
        nei_bool[ii==jj]=False
    
    lvec, rot = tangent_connection(pts,gradient,ref=ref, nei_bool=nei_bool)
    phase_correct = rot**order

    psi = np.zeros_like(nei_bool)
    # double-check for degenerate neighborless states
    if not np.any(nei_bool): return psi
    # get neighbor count per particle, and cumulative
    sizes = np.sum(nei_bool,axis=-1)
    csizes = np.cumsum(sizes)
 
    #assemble lists of vectors to evaluate angles between (us and vs)
    bonds = dr_vec[nei_bool]
    e1_orig = lvec[:,0,:][ii][nei_bool]
    e2_orig = lvec[:,1,:][ii][nei_bool]
    # compute coordiates on local tangent plane
    xs = np.sum(bonds*e1_orig,axis=-1)
    ys = np.sum(bonds*e2_orig,axis=-1)
    angles = np.arctan2(ys,xs)
    
    # now we compute psi_ij for each of the bonds
    psi_ij = np.exp(1j*order*angles)
    # pick out only the last summed psi for each particle
    psi_csum = np.array([0,*np.cumsum(psi_ij)])
    # subtract off the previous particles' summed psi values
    c_idx = np.array([0,*csizes])
    psi_i = psi_csum[c_idx[1:]]-psi_csum[c_idx[:-1]]

    #return the neighbor-averaged psi for each particle in it's own local tangent plane
    psi_i[sizes>0]*=1/sizes[sizes>0]
    psi_i[sizes==0]=0

    return psi_i, phase_correct





def crystal_connectivity(psis:np.ndarray, nei_rotate:np.ndarray, crystallinity_threshold:float = 0.32, norm:float|None=6) -> np.ndarray:
    """Computes the crystal connectivity of each particle in a 2D configuration. The crystal connectivity measures the similarity of bond-orientational order parameter over all pairs of neighboring particles in order to determine which particles are part of a definite crystalline domain. The crystal connectivity of particle :math:`j` is given by:

    .. math::
        
        C_n^j = \\frac{1}{n}\\sum_k^{\\text{nei}}\\bigg[ \\frac{\\text{Re}\\big[\\psi_j\\psi_k^*\\big]}{|\\psi_j\\psi_k^*|} \\geq \\Theta_C \\bigg]

    Where the :math:`\\psi`\'s are any-fold bond orientational order parameters for each particle, :math:`\\Theta_{C}` is a \'crystallinity threshold\' used to determine whether two neighboring particles are part of the same crystalline domain, and :math:`n` is a factor used to simply normalize :math:`C_6^j` between zero and one.


    :param psis: [N] array of complex bond orientational order parameters
    :type psis: ndarray
    :param nei_bool: a [NxN] array indicating neighboring particles. If
    :type nei_bool: ndarray,
    :param crystallinity_threshold: the minimum innier product of adjacent complex bond-OPs needed in order to consider adjacent particles 'connected', defaults to 0.32
    :type crystallinity_threshold: float, optional
    :param norm: an optional factor to normalize the result, defaults to the connectivity value for a perfectly crystalline hexagon (i.e. equations 8 and 10 in the SI from `(Juarez, Lab on a Chip 2012) <https://doi.org/10.1039/C2LC40692F>`_)
    :type norm: float | None, optional
    :return: [N] array of real crystal connectivities
    :rtype: ndarray
    """    

    # double-check for degenerate neighborless states
    if not np.any(nei_rotate): return np.zeros_like(psis)
    
    if norm is None:
        # computing the reference c6 for a perfect lattice
        # equation 8 from SI of https://doi.org/10.1039/C2LC40692F
        shells = -1/2 + np.sqrt((len(psis)-1)/3 + 1/4)
        # equation 10 from SI of https://doi.org/10.1039/C2LC40692F
        c6_hex = 6*(3*shells**2 + shells)/len(psis)
        norm = c6_hex

    pnum = len(psis)
    psi_i = np.array([psis]*pnum)
    psi_j = np.conjugate(nei_rotate*psi_i.T)

    psi_prod = np.outer(psis,np.conjugate(psis))

    chi_ij = np.abs(np.real(psi_i*psi_j))/np.abs(psi_i*psi_j)
    chi_ij[np.abs(psi_i*psi_j)==0]=0

    return np.sum(chi_ij>crystal_connectivity,axis=-1)/norm
