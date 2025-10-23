# -*- coding: utf-8 -*-
"""
Contains methods to calculate morphological properties of particle ensembles and subsets of ensembles, including local area fraction, gyration tensors, and functions of their eigenvalues like gyration radius, acylindricity, asphericity and anisotropy.

"""

import numpy as np
from scipy.spatial.distance import pdist,squareform


def central_eta(pts:np.ndarray, box:list, ptcl_area:float = np.pi/4, nbins:int=3, bin_width:float=4.0, jac:str='x'):
    """Computes the average area fraction in the central region of a configuration of particles, accounting for the jacobian of the coordinate system:

    .. math::

        \\eta = \\langle\\frac{N\\cdot A_p}{A_{bin}}\\rangle_{{bins}}

    :param pts: an [Nxd] array of particle positions in any dimensions (though only the first two are used)
    :type pts: ndarray
    :param box: a list defining the simulation box in the `gsd <https://gsd.readthedocs.io/en/stable/schema-hoomd.html#chunk-configuration-box>`_ convention
    :type box: array-like
    :param ptcl_area: the area of a single particle, defaults to pi/4 (for a circle of diameter 1.0)
    :type ptcl_area: float, optional
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
    eta = counts / bin_areas * ptcl_area

    # average over the 'nbins' most central histrogram bins
    central_eta = np.mean(eta[np.argsort(np.abs(mids))[:nbins]])

    return central_eta


def gyration_radius(pts:np.ndarray) -> float:
    """returns the radius of gyration according to the formula

    .. math::

        R_g^2 \\equiv \\frac{1}{N}\\sum_k|\\mathbf{r}_k-\\bar{\\mathbf{r}}|^2 = \\frac{1}{N^2}\\sum_{j>i}|\\mathbf{r}_i - \\mathbf{r}_j|^2 

    where the :math:`j>i` in the summation index indicates that repeated pairs are not summed over

    :param pts: [Nxd] array of particle positions in 'd' dimensions
    :type pts: ndarray
    :return: the radius of gyration of the particles about their center of mass
    :rtype: scalar
    """    
    N = len(pts)
    dists = pdist(pts)
    Rg2 = np.sum(dists**2) / (N**2) # pdist accounts for the factor of 2 in the denominator
    return np.sqrt(Rg2)


def gyration_tensor(pts:np.ndarray, ref:np.ndarray|None = None) -> np.ndarray:
    """returns the gyration tensor (for principal moments analysis) of an ensemble of particles according to the formula

    .. math::

        S_{mn} = \\frac{1}{N}\\sum_{j}r^{(j)}_m r^{(j)}_n

    where the positions, :math:`r`, are defined in their center of mass reference frame

    :param pts: [Nxd] array of particle positions in 'd' dimensions,
    :type pts: ndarray
    :param ref: point in d-dimensional space from which to reference particle positions, defaults to the mean position of the points. Use this for constraining the center of mass to the surface of a manifold, for instance.
    :type ref: ndarray , optional
    :return: the [dxd] gyration tensor of the ensemble
    :rtype: ndarray
    """    

    if ref is None:
        ref = pts.mean(axis=0)
    assert (pts.shape[-1],) == ref.shape, 'reference must have same dimesionality as the points'
    centered = pts - ref
    gyrate = centered.T @ centered
    return gyrate/len(pts)

def acylindricity(pts:np.ndarray=None, gyr:np.ndarray=None) -> float:
    """returns the acylindricity of a set of points, defined as a function of the gyration tensor eigenvalues:

    .. math::

        c \\equiv \\lambda_y^2 - \\lambda_x^2

    Where :math:`\\lambda_x^2\\leq\\lambda_y^2\\leq\\lambda_z^2` are the eigenvalues of the gyration tensor.

    :param pts: [Nxd] array of particle positions in 'd' dimensions
    :type pts: ndarray
    :param gyr: [dxd] gyration tensor of the ensemble, optional
    :type gyr: ndarray
    :return: the acylindricity of the particles
    :rtype: float
    """    
    if gyr is None:
        assert pts is not None, 'must provide either pts or gyr'
        gyr = gyration_tensor(pts)

    lz, ly, lx = np.linalg.eigvalsh(gyr) # ascending order
    return ly**2 - lx**2

def asphericity(pts:np.ndarray=None, gyr:np.ndarray=None) -> float:
    """returns the asphericity of a set of points, defined as a function of the gyration tensor eigenvalues:

    .. math::

        b \\equiv \\\\lambda_z^2 - \\frac{{1}}{{2}}\\big(\\lambda_y^2 + \\lambda_x^2\\big)

    Where :math:`\\lambda_x^2\\leq\\lambda_y^2\\leq\\lambda_z^2` are the eigenvalues of the gyration tensor.

    :param pts: [Nxd] array of particle positions in 'd' dimensions
    :type pts: ndarray
    :param gyr: [dxd] gyration tensor of the ensemble, optional
    :type gyr: ndarray
    :return: the asphericity of the particles
    :rtype: float
    """    
    if gyr is None:
        assert pts is not None, 'must provide either pts or gyr'
        gyr = gyration_tensor(pts)

    gyr_evals = np.linalg.eigvalsh(gyr)
    return (gyr_evals[1] - gyr_evals[0]) / np.mean(gyr_evals[:2])

def shape_anisotropy(pts:np.ndarray=None, gyr:np.ndarray=None) -> float:
    """returns the anisotropy of a set of points, defined as a function of the gyration tensor eigenvalues:

    .. math::

        \\kappa^2 \\equiv \\frac{{3}}{{2}}\\frac{{\\lambda_z^4 + \\lambda_y^4 + \\lambda_x^4}}{{(\\lambda_x^2 + \\lambda_y^2 + \\lambda_z^2)^2}} - \\frac{{1}}{{2}} = \\frac{{b^2 + (3/4)c^2}}{{R_g^4}}

    Where :math:`\\lambda_x^2\\leq\\lambda_y^2\\leq\\lambda_z^2` are the eigenvalues of the gyration tensor.

    :param pts: [Nxd] array of particle positions in 'd' dimensions
    :type pts: ndarray
    :param gyr: [dxd] gyration tensor of the ensemble, optional
    :type gyr: ndarray
    :return: the anisotropy of the particles
    :rtype: float
    """    
    if gyr is None:
        assert pts is not None, 'must provide either pts or gyr'
        gyr = gyration_tensor(pts)

    lz, ly, lx = np.linalg.eigvalsh(gyr) # ascending order
    return (3/2 * (lz**4 + ly**4 + lx**4) / (lx**2 + ly**2 + lz**2)**2) - 1/2

def circularity(pts=None, gyr=None, ref:np.ndarray=np.array([0,1,0])) -> float:
    """
    the 'circularity' of a colloidal cluster as used in `Zhang, Sci. Adv. 2020 <https://doi.org/10.1126/sciadv.abd6716>`_. This metric is calcuated using the principal moments of the :py:meth:`gyration_tensor`. For 2d ensembles, after diagonalization:

    .. math::

        S_{mn} = \\begin{pmatrix}
        \\lambda_1^2 & 0 & 0 \\\\
        0 & \\lambda_2^2 & 0 \\\\
        0 & 0 & \\lambda_3^2=0
        \\end{pmatrix}

    With the prinicipal moments defined as :math:`\\lambda_1>\\lambda_2`. Under these definitions, we can compute the radius of gyration :math:`R_g=\\sqrt{{\\lambda_1^2 + \\lambda_2^2}}` and the acylindricity :math:`a = \\lambda_1-\\lambda_2`, then combine them to define the circularity:

    .. math::

        c = 1 - \\frac{a}{R_g}
    
    When the cluster is circular the two principal moments are equal and so :math:`a=0 \\to c=1`. When the cluster is a linear chain, the smaller principal moment approaches zero, and so :math:`a=R_g=\\lambda_1 \\to c=0`.

    :param gyr_tensor: [dxd] array , defaults to None
    :type gyr_tensor: ndarray, optional
    :param pts: an ensemble of colloidal positions, defaults to None
    :type pts: ndarray, optional
    :return: _description_
    :rtype: float
    """
    if gyr is None:
        assert pts is not None, 'must provide either pts or gyr'
        gyr = gyration_tensor(pts)

    mom, evecs = np.linalg.eig(gyr)
    lx = np.sort(mom)[-1]
    ly = np.sort(mom)[-2]

    a = lx**0.5 - ly**0.5
    rg = (lx+ly)**0.5
    c = 1-a/rg

    ax = evecs[np.sort(mom)][:, -1]
    theta = (np.arccos(ax @ ref)+np.pi/2)%np.pi - np.pi/2

    return c*np.exp(1j*theta)