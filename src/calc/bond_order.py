# -*- coding: utf-8 -*-
"""
Contains methods to calculate bond orientational order. First in flat space, then in stretched space for anisotropic particles, and finally projected onto curved surfaces.
"""

import numpy as np


def flat_bond_order(pts:np.ndarray, nei_bool:np.ndarray, order:int = 6) -> tuple[np.ndarray,float]:
    """Calculates the local and global bond orientational order parameter of each particle in a 2D configuration with respect to the y axis. The local n-fold bond orientaitonal order for a particle :math:`j` is:

    .. math::

        \\psi_j = \\frac{1}{N_j}\\sum_k\\psi_{jk}=\\frac{1}{N_j}\\sum_ke^{in\\theta_{jk}}

    Where the sum is over all :math:`N_j` neighboring particles :math:`k` to particle :math:`j` and :math:`\\theta_{jk}` is the angle between particles :math:`j` and :math:`k`. Similarly, the global n-fold bond orientational order is:

    .. math::

        \\psi_g = \\frac{1}{N}\\frac{1}{N_j}\\sum_{jk}\\psi_{jk}=\\frac{1}{N}\\frac{1}{N_j}\\sum_{jk}e^{in\\theta_{jk}}

    Where the sum is over all unique bonds between particles :math:`j` and :math:`k`.

    :param pts: [Nxd] array of particle positions in 'd' dimensions, though the calculation only access the first two dimensions.
    :type pts: ndarray
    :param nei_bool: a [NxN] boolean array indicating neighboring particles.
    :type nei_bool: ndarray
    :param order: n-fold order defines the argument of the complex number used to calculate psi_n, defaults to 6
    :type order: int, optional
    :return: [N] array of complex bond orientational order parameters, and the norm of their mean.
    :rtype: tuple(ndarray, scalar)
    """

    pnum = pts.shape[0]
    i,j = np.mgrid[0:pnum,0:pnum]
    dr_vec = pts[i]-pts[j]

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


def stretched_bond_order(pts:np.ndarray, angles:np.ndarray, nei_bool:np.ndarray, rx:float = 1.0, ry:float = 1.0, order:int = 6) -> tuple[np.ndarray,float]:
    """Computes the local and global stretched bond orientational order parameter. This calculation rotates coordinates into a frame of reference stretched according to the long and short axes of each particle according to equations given in `(Torrez-Diaz Soft Matter, 2022) <https://doi.org/10.1039/D1SM01523K>`_.

    :param pts: [Nxd] array of particle positions in 'd' dimensions, though the calculation only access the first two dimensions.
    :type pts: ndarray
    :param angles: the orientation of each anisotropic particle in the configuration
    :type angles: ndarray
    :param nei_bool: a [NxN] boolean array indicating neighboring particles.
    :type nei_bool: ndarray
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

from .locality import local_vectors

def projected_bond_order(pts:np.ndarray, gradient:callable, nei_bool:np.ndarray, order:int = 6, ref:np.ndarray = np.array([0,-1,0])) -> tuple[np.ndarray,float]:
    """Computes the local and global bond orientational order parameter of each particle in a 2D configuration projected onto the local tangent plane of a curved surface. The local n-fold bond orientaitonal order for a particle :math:`j` is:

    .. math::

        \\psi_j = \\frac{1}{N_j}\\sum_k\\psi_{jk}=\\frac{1}{N_j}\\sum_ke^{in\\theta_{jk}}

    Where the sum is over all :math:`N_j` neighboring particles :math:`k` to particle :math:`j` and :math:`\\theta_{jk}` is the angle between particles :math:`j` and :math:`k` projected onto the local tangent plane of particle :math:`j`. 

    :param pts: [Nxd] array of particle positions in 'd' dimensions.
    :type pts: ndarray
    :param gradient: a callable function that computes the local gradient at a given point
    :type gradient: callable
    :param nei_bool: a [NxN] boolean array indicating neighboring particles.
    :type nei_bool: ndarray
    :param order: n-fold order defines the argument of the complex number used to calculate psi_n, defaults to 6
    :type order: int, optional
    :param ref: a reference vector to help define the local frame, defaults to np.array([0,-1,0])
    :type ref: ndarray, optional
    :return: [N] array of complex bond orientational order parameters, and the norm of their mean.
    :rtype: tuple(ndarray, scalar)
    """    

    pnum = pts.shape[0]
    ii,jj = np.mgrid[0:pnum,0:pnum]
    dr_vec = pts[jj]-pts[ii]
    if nei_bool is None:
        nei_bool = squareform(pdist(pts))<=DEFAULT_CUTOFF
        nei_bool[ii==jj]=False
    
    lvec = local_vectors(pts,gradient,ref=ref)

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

    return psi_i


def crystal_connectivity(psis:np.ndarray, nei_bool:np.ndarray, crystallinity_threshold:float = 0.32, norm:float|None=6, phase_rotate:np.ndarray|None=None) -> np.ndarray:
    """Computes the crystal connectivity of each particle in a 2D configuration. The crystal connectivity measures the similarity of bond-orientational order parameter over all pairs of neighboring particles in order to determine which particles are part of a definite crystalline domain. The crystal connectivity of particle :math:`j` is given by:

    .. math::
        
        C_n^j = \\frac{1}{n}\\sum_k^{\\text{nei}}\\bigg[ \\frac{\\text{Re}\\big[\\psi_j\\psi_k^*\\big]}{|\\psi_j\\psi_k^*|} \\geq \\Theta_C \\bigg]

    Where the :math:`\\psi`\'s are any-fold bond orientational order parameters for each particle, :math:`\\Theta_{C}` is a \'crystallinity threshold\' used to determine whether two neighboring particles are part of the same crystalline domain, and :math:`n` is a factor used to simply normalize :math:`C_6^j` between zero and one.


    :param psis: [N] array of complex bond orientational order parameters
    :type psis: ndarray
    :param nei_bool: a [NxN] array indicating neighboring particles.
    :type nei_bool: ndarray
    :param crystallinity_threshold: the minimum inner product of adjacent complex bond-OPs needed in order to consider adjacent particles 'connected', defaults to 0.32
    :type crystallinity_threshold: float, optional
    :param norm: an optional factor to normalize the result, defaults to 6, but passing ``None`` will reference the connectivity value for a perfectly crystalline hexagon (i.e. equations 8 and 10 in the SI from `(Juarez, Lab on a Chip 2012) <https://doi.org/10.1039/C2LC40692F>`_)
    :type norm: float | None, optional
    :param phase_rotate: an optional [N] array of complex phase factors to include a rotation between neighboring particles' bond-OPs (i.e. the output of :py:meth:`calc.locality.tangent_connection`) defaults to None
    :type phase_rotate: ndarray | None, optional
    :return: [N] array of real crystal connectivities
    :rtype: ndarray
    """    

    # double-check for degenerate neighborless states
    if not np.any(nei_bool): return np.zeros_like(psis)
    
    if norm is None:
        # computing the reference c6 for a perfect lattice
        # equation 8 from SI of https://doi.org/10.1039/C2LC40692F
        shells = -1/2 + np.sqrt((len(psis)-1)/3 + 1/4)
        # equation 10 from SI of https://doi.org/10.1039/C2LC40692F
        c6_hex = 6*(3*shells**2 + shells)/len(psis)
        norm = c6_hex

    # include optional phase rotation between neighbors
    if phase_rotate is not None:
        nei_rotate = nei_bool*phase_rotate
    else:
        nei_rotate = nei_bool*1.0

    pnum = len(psis)
    psi_i = np.array([psis]*pnum)
    psi_j = np.conjugate(nei_rotate*psi_i.T)

    chi_ij = np.abs(np.real(psi_i*psi_j))/np.abs(psi_i*psi_j)
    chi_ij[np.abs(psi_i*psi_j)==0]=0

    return np.sum(chi_ij>crystallinity_threshold,axis=-1)/norm
