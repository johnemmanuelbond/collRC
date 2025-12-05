# -*- coding: utf-8 -*-
"""
Contains methods to calculate bond orientational order. First in 2D cartesian space, then in 2D stretched space for anisotropic particles, then in 2D spaces projected onto curved surfaces, and finally 3D cartesian spaces using Steinhardt calculations.
"""

import numpy as np
from scipy.special import sph_harm_y


def flat_bond_order(pts:np.ndarray, nei_bool:np.ndarray, order:int = 6, ret_global=False) -> tuple[np.ndarray,np.complexfloating]:
    """Calculates the local and global bond orientational order parameter of each particle in a 2D configuration with respect to the y axis. The local n-fold bond orientaitonal order for a particle :math:`j` is:

    .. math::

        \\psi_{n,j} = \\frac{1}{N_j}\\sum_k\\psi_{n,jk}=\\frac{1}{N_j}\\sum_ke^{in\\theta_{jk}}

    Where the sum is over all :math:`N_j` neighboring particles and :math:`\\theta_{jk}` is the angle between particles :math:`j` and :math:`k`. The global n-fold bond orientational order is:

    .. math::

        \\psi_{n,g} = \\langle e^{in\\theta_{jk}} \\rangle_{jk} \\simeq \\langle \\psi_{n,j} \\rangle_{j}

    Where the mean is over all unique bonds between particle pairs :math:`j` and :math:`k`, or approximately the mean over all particles' local bond order parameters.

    :param pts: :math:`[N,d]` array of particle positions in 'd' dimensions, though the calculation only accesses the first two dimensions.
    :type pts: ndarray
    :param nei_bool: a :math:`[N,N]` boolean array indicating neighboring particles.
    :type nei_bool: ndarray
    :param order: n-fold order defines the argument of the complex number used to calculate psi_n, defaults to 6
    :type order: int, optional
    :param ret_global: whether to return the global bond order parameter, defaults to False
    :type ret_global: bool, optional
    :return: :math:`[N,]` array of complex bond orientational order parameters, and (if ret_global) the global bond orientational order.
    :rtype: ndarray[complex] `(, complex)`
    """

    pnum = pts.shape[0]
    # double-check for degenerate neighborless states
    if not np.any(nei_bool):
        if ret_global: return np.zeros(pnum),0
        else: return np.zeros(pnum)

    i,j = np.mgrid[0:pnum,0:pnum]
    dr_vec = pts[i]-pts[j]
    
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

    if ret_global:
        return psi, np.abs(psi_ij.mean())
    else:
        return psi


def stretched_bond_order(pts:np.ndarray, angles:np.ndarray, nei_bool:np.ndarray, rx:float = 1.0, ry:float = 1.0, order:int = 6, ret_global=False) -> tuple[np.ndarray,np.complexfloating]:
    """Computes the local and global stretched bond orientational order parameter. This calculation rotates coordinates into a frame of reference stretched according to the long and short axes of each particle according to equations given in `(Torrez-Diaz Soft Matter, 2022) <https://doi.org/10.1039/D1SM01523K>`_:

    .. math::

        \\psi_{n,j} = \\frac{1}{N_j}\\sum_k\\psi_{n,jk}=\\frac{1}{N_j}\\sum_ke^{in\\theta^s_{jk}}e^{in\\theta_j}

    Where the sum is over all :math:`N_j` neighboring particles, :math:`\\theta_j` is the orientaion of particle :math:`j`, and :math:`\\theta^s_{jk}` is the angle between particles :math:`j` and :math:`k` in the stretched coordiante system of particle :math:`j`:

    .. math::

        \\theta^s_{jk} = \\tan^{-1}\\bigg[\\frac{\\mathbf{\\delta r_{jk}} \\cdot \\hat{\\mathbf{y_j}}/r_y}{\\mathbf{\\delta r_{jk}} \\cdot \\hat{\\mathbf{x_j}}/r_x}\\bigg]

    Where :math:`\\hat{\\mathbf{x_j}} = \\cos(\\theta_j)\\hat{\\mathbf{x}} + \\sin(\\theta_j)\\hat{\\mathbf{y}}` and :math:`\\hat{\\mathbf{y_j}} = -\\sin(\\theta_j)\\hat{\\mathbf{x}} + \\cos(\\theta_j)\\hat{\\mathbf{y}}` are the local unit vectors along the long (:math:`r_x`) and short (:math:`r_y`) axes of particle :math:`j`, respectively. The global n-fold stretched bond orientational order is:

    .. math::

        \\psi_{n,g} = \\langle e^{in\\theta^s_{jk}}e^{in\\theta_j} \\rangle_{jk} \\simeq \\langle \\psi_{n,j} \\rangle_j

    Where the mean is over all unique bonds between particles :math:`j` and :math:`k`, or approximately the mean over all particles' local bond order parameters.

    :param pts: :math:`[N,d]` array of particle positions in 'd' dimensions, though the calculation only access the first two dimensions.
    :type pts: ndarray
    :param angles: the orientation of each anisotropic particle in the configuration
    :type angles: ndarray
    :param nei_bool: a :math:`[N,N]` boolean array indicating neighboring particles.
    :type nei_bool: ndarray
    :param rx: the radius of the long axis of the particle (insphere radius times aspect ratio), defaults to 1.0
    :type rx: scalar, optional
    :param ry: the radius of the short axis of the partice (i.e. insphere radius), defaults to 1.0
    :type ry: scalar, optional
    :param order: n-fold order defines the argument of the complex number used to calculate psi_n, defaults to 6
    :type order: int, optional
    :param ret_global: whether to return the global bond order parameter, defaults to False
    :type ret_global: bool, optional
    :return: :math:`[N,]` array of complex bond orientational order parameters, and (if ret_global) the global bond orientational order.
    :rtype: ndarray[complex] `(, complex)`
    """    
    pnum = pts.shape[0]
    # double-check for degenerate neighborless states
    if not np.any(nei_bool):
        if ret_global: return np.zeros(pnum),0
        else: return np.zeros(pnum)

    i,j = np.mgrid[0:pnum,0:pnum]
    dr_vec = pts[i]-pts[j]

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
    
    if ret_global:
        return psi, np.abs(psi_ij.mean())
    else:
        return psi

from .locality import local_vectors

def projected_bond_order(pts:np.ndarray, gradient:callable, nei_bool:np.ndarray, order:int = 6, ref:np.ndarray = np.array([0,-1,0])) -> tuple[np.ndarray,float]:
    """
    Computes the local bond orientational order parameter of each particle in a 2D configuration projected onto the local tangent plane of a curved surface. The local n-fold bond orientaitonal order for a particle :math:`j` is:

    .. math::

        \\psi_{n,j} = \\frac{1}{N_j}\\sum_k\\psi_{n,jk}=\\frac{1}{N_j}\\sum_ke^{in\\theta_{jk}}

    Where the sum is over all :math:`N_j` neighboring particles and :math:`\\theta_{jk}` is the angle between particles :math:`j` and :math:`k` projected onto the local tangent plane of particle :math:`j`, defined by a basis :math:`\\{\\hat{\\mathbf{e}}_{1,j},\\hat{\\mathbf{e}}_{2,j}\\}` computed using the provided gradient function and the :py:meth:`local_vectors() <calc.locality.local_vectors>` method.

    .. math::

        \\theta_{jk} = \\tan^{-1}\\big[ (\\mathbf{r}_{jk}\\cdot\\hat{\\mathbf{e}}_{2,j})\\big/(\\mathbf{r}_{jk}\\cdot\\hat{\\mathbf{e}}_{1,j}) \\big]
    
    Due to parallel transport on curved surfaces, there is no well-defined global bond orientational order because of the lack of a common reference vector.

    :param pts: :math:`[N,d]` array of particle positions in 'd' dimensions.
    :type pts: ndarray
    :param gradient: a callable function that computes the local gradient at a given point
    :type gradient: callable
    :param nei_bool: a :math:`[N,N]` boolean array indicating neighboring particles.
    :type nei_bool: ndarray
    :param order: n-fold order defines the argument of the complex number used to calculate psi_n, defaults to 6
    :type order: int, optional
    :param ref: a reference vector to help define the local frame, defaults to np.array([0,-1,0])
    :type ref: ndarray, optional
    :return: :math:`[N,]` array of complex bond orientational order parameters, and the norm of their mean.
    :rtype: ndarray[complex]
    """    

    pnum = pts.shape[0]
    # double-check for degenerate neighborless states
    if not np.any(nei_bool): return np.zeros(pnum)

    ii,jj = np.mgrid[0:pnum,0:pnum]
    dr_vec = pts[jj]-pts[ii]
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


def steinhardt_bond_order(pts:np.ndarray, nei_bool:np.ndarray, l:int = 6, ret_global:bool=False) -> tuple[np.ndarray,np.complexfloating]:
    """Calculates the local and global bond orientational Steinhardt order parameter of each particle in a 3D configuration using a superposition of complex spherical harmonics. The local l-fold bond orientaitonal order for a particle :math:`j` is a list of :math:`2l+1` complex-valued spherical harmonic coefficients:

    .. math::

        \\big\\{q_{lm,j}\\big\\} = \\big\\{\\frac{1}{N_j}\\sum_kY_{lm}(\\theta_{jk},\\phi_{jk})\\big\\}

    Where the sum is over all :math:`N_j` neighboring particles and :math:`\\theta_{jk}, \\phi_{jk}` are the angles (in spherical coordinates) between particles :math:`j` and :math:`k`. Each particle has a rotationally invariant local bond order magnitude given by:

    .. math::

        q_{l,j} = \\sqrt{\\frac{4\\pi}{2l+1} \\sum_m |q_{lm,j}|^2}

    Which allows us to similarly define a rotationally-invariant global l-fold bond orientational order:

    .. math::

        q_{l} = \\sqrt{\\frac{4\\pi}{2l+1} \\sum_m |\\langle q_{lm,j}\\rangle_j|^2}

    Where we average each particle's :math:`q_{lm}` `before` summing their magnitudes over :math:`m` to make it rotationally invariant.

    :param pts: :math:`[N,d]` array of particle positions in 'd' dimensions, though the calculation only access the first three dimensions.
    :type pts: ndarray
    :param nei_bool: a :math:`[N,N]` boolean array indicating neighboring particles.
    :type nei_bool: ndarray
    :param l: The degree of the spherical harmonics used to calculate :math:`l`-fold bond order parameter, defaults to 6
    :type l: int, optional
    :param ret_global: whether to return the global bond order parameter, defaults to False
    :type ret_global: bool, optional
    :return: :math:`[N,2l+1]` array of complex bond orientational order parameters, and (if ret_global) the global bond orientational order.
    :rtype: ndarray[complex] `(, complex)`
    """
    pnum = pts.shape[0]
    # double-check for degenerate neighborless states
    if not np.any(nei_bool):
        if ret_global: return np.zeros(pnum),0
        else: return np.zeros(pnum)

    i,j = np.mgrid[0:pnum,0:pnum]
    dr_vec = pts[i]-pts[j]

    # get neighbor count per particle, and cumulative
    sizes = np.sum(nei_bool,axis=-1)
    csizes = np.cumsum(sizes)

    #assemble lists of vectors to evaluate angles between (us and vs)
    bonds = dr_vec[nei_bool]
    rs = np.linalg.norm(bonds, axis=1)
    xs = bonds[:,0]
    ys = bonds[:,1]
    zs = bonds[:,2]
    phis = np.arctan2(ys,xs)
    thetas = np.arccos(np.clip(zs/rs, -1, 1))

    # set up matrices of spherical harmonic orders, degrees, and eval points for each bond.
    m = np.arange(-l,l+1)
    mm, pphi  = np.meshgrid(m,phis)
    _, ttheta = np.meshgrid(m,thetas)
    _, ssizes = np.meshgrid(m,sizes)
    ll = l*np.ones_like(mm)

    # calculate spherical harmonics for each bond given order l and degree m.
    qlm_ij = sph_harm_y(ll,mm,ttheta,pphi)
    # pick out only the last summed psi for each particle
    qlm_csum = np.array([0*m,*np.cumsum(qlm_ij,axis=0)])
    # subtract off the previous particles' summed psi values
    c_idx = np.array([0,*csizes])
    qlm = qlm_csum[c_idx[1:]]-qlm_csum[c_idx[:-1]]

    #return the neighbor-averaged qlm
    qlm[ssizes>0]*=1/ssizes[ssizes>0]
    qlm[ssizes==0]=0

    if ret_global:
        Q6 = np.sqrt(4*np.pi/(2*6+1)*np.sum(np.abs(qlm.mean(axis=0))**2))
        return qlm, Q6
    else:
        return qlm



def crystal_connectivity(psis:np.ndarray, nei_bool:np.ndarray, crystallinity_threshold:float = 0.32, norm:float|None=6, phase_rotate:np.ndarray|None=None, calc_3d = False) -> np.ndarray:
    """
    Computes the crystal connectivity of each particle in a 2D cartesian, 2D stretched, 2D projected, or 3D system. The crystal connectivity measures the similarity of bond-orientational order parameter over all pairs of neighboring particles in order to determine which particles are part of a definitive crystalline domain. In flat 2D the crystal connectivity of particle :math:`j` is given by:

    .. math::

        C_{n,j} = \\frac{1}{N_C}\\sum_k^{\\text{nei}}\\bigg[ \\frac{\\text{Re}\\big[\\psi_{n,j}\\psi_{n,k}^*\\big]}{|\\psi_{n,j}\\psi_{n,k}^*|} \\geq \\Theta_C \\bigg]

    Where the :math:`\\psi`\'s are n-fold bond orientational order parameters for each particle, :math:`\\Theta_{C}` is a 'crystallinity threshold' used to determine whether two neighboring particles are part of the same crystalline domain, and :math:`N_C` is a factor used to simply normalize :math:`C_{n,j}` between zero and one.

    On curved surfaces neighboring particles may have bond orientational order parameters defined in different local tangent planes. In this case the crystalline connectivity becomes:

    .. math::

        C_{n,j} = \\frac{1}{N_C}\\sum_k^{\\text{nei}}\\bigg[ \\frac{\\text{Re}\\big[\\psi_{n,j}((R_{jk})^n\\psi_{n,k})^*\\big]}{|\\psi_{n,j}((R_{jk})^n\\psi_{n,k})^*|} \\geq \\Theta_C \\bigg]

    Where :math:`R_{jk}` encodes the rotation between neighboring tangent planes, i.e. the output of :py:meth:`tangent_connection() <calc.locality.tangent_connection>`. :math:`(R_{jk})^n` is then used to rotate the n-fold bond orientational order of neighboring particles.

    Optionally, users may pass in an :math:`[N,2l+1]` array of complex spherical harmonic Steinhardt order parameters (and pass ``calc_3d=True``) for use in 3D systems. In this case, the inner product is computed over all :math:`2l+1` components of the Steinhardt order parameter rather than a single complex bond-OP:

    .. math::

        C_{l,j} = \\frac{1}{N_C}\\sum_k^{\\text{nei}}\\bigg[ \\frac{\\text{Re}\\big[\\sum_{m=-l}^{l}q_{lm,j}q_{lm,k}^*\\big]}{\\sqrt{\\sum_m|q_{lm,j}|^2} \\sqrt{\\sum_m|q_{lm,k}|^2}} \\geq \\Theta_C \\bigg]

    :param psis: :math:`[N,]` array of complex bond orientational order parameters. Optionally, an :math:`[N,2l+1]` array of complex spherical harmonic Steinhardt order parameters for use in 3D systems.
    :type psis: ndarray
    :param nei_bool: a :math:`[N,N]` array indicating neighboring particles.
    :type nei_bool: ndarray
    :param crystallinity_threshold: the minimum inner product of adjacent complex bond-OPs needed in order to consider adjacent particles 'connected', defaults to 0.32
    :type crystallinity_threshold: float, optional
    :param norm: an optional factor to normalize the result, defaults to 6, but passing ``None`` will reference the connectivity value for a perfectly crystalline hexagon (i.e. equations 8 and 10 in the SI from `(Juarez, Lab on a Chip 2012) <https://doi.org/10.1039/C2LC40692F>`_)
    :type norm: float | None, optional
    :param phase_rotate: an optional :math:`[N,N]` array of complex phase factors (:math:`(R_{jk})^n`) to include a rotation between neighboring particles' :math:`\\psi_{n,j}`, defaults to None
    :type phase_rotate: ndarray | None, optional
    :param calc_3d: whether to compute the 3D version of the crystal connectivity using spherical harmonic Steinhardt order parameters, defaults to False
    :type calc_3d: bool, optional
    :return: :math:`[N,]` array of real crystal connectivities
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
    if calc_3d:
        nei_rotate = np.array([nei_bool.T] * psis.shape[1]).T
    elif phase_rotate is not None:
        nei_rotate = nei_bool*phase_rotate
    else:
        nei_rotate = nei_bool*1.0

    psi_i = np.array([psis]*psis.shape[0])
    psi_j = np.conjugate(nei_rotate*np.swapaxes(psi_i,0,1))

    if not calc_3d:
        chi_top = np.abs(np.real(psi_i*psi_j))
        chi_bot = np.abs(psi_i*psi_j)
    else:
        chi_top = np.sum(np.real(psi_i*psi_j),axis=-1)
        qi = np.sum(np.abs(psi_i)**2,axis=-1)**0.5
        qj = np.sum(np.abs(psi_j)**2,axis=-1)**0.5
        chi_bot = qi*qj

    chi_top[chi_bot==0]=0
    chi_bot[chi_bot==0]=1  # prevent div by zero
    chi_ij = chi_top/chi_bot

    return np.sum(chi_ij>crystallinity_threshold,axis=-1)/norm
