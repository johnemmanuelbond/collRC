# -*- coding: utf-8 -*-
"""
Contains methods to calculate p-atic orientational order. These methods generally cover nematic, tetract and hexatic order parameters.

.. math::

    S_2 \\equiv P_2 \\qquad T_4 \\equiv P_4 \\qquad H_6 \\equiv P_6
    
"""

import numpy as np

def global_patic(angles:np.ndarray, p:int=2) -> tuple[float,float]:
    """
    Computes the global p-atic order of a particle ensemble.

    .. math::

        P_p = max_{\\theta_p}\\langle\\cos(p(\\theta_j-\\theta_p))\\rangle = |\\langle e^{ip\\theta_j}\\rangle_j|

    :param angles: the orientation of each particle in the frame
    :type angles: ndarray
    :param p: the p-atic order to compute (e.g. 2 for nematic, 4 for tetratic, 6 for hexatic)
    :type p: int
    :return: the value of the global p-atic order and the angle which defines it
    :rtype: scalar, scalar
    """

    pp = np.exp(1j*p*angles).mean()
    return np.abs(pp), np.angle(pp)/p

def local_patic(angles:np.ndarray, nei_bool:np.ndarray, p:int =2) -> tuple[np.ndarray,np.ndarray]:
    """
    Computes the local p-atic order, per particle, of an ensemble. Quantifies the local orientational order of a system by calculating a director for particle i via its neighboring particle(s) j, as defined by `(Baron J. Chem. Phys. 2023) <https://doi.org/10.1063/5.0169659>`_.

    .. math::

        P_{p,i} = max_{\\theta_{p,i}}\\langle\\cos(p(\\theta_j-\\theta_{p,i}))\\rangle_{j(r_{ij}<6a_x)} = |\\langle e^{ip\\theta_j}\\rangle_{j(r_{ij}<6a_x})|

    :param angles: the orientation of each particle in the frame
    :type angles: ndarray
    :param nei_bool: [NxN] boolean array defining particle neighbors
    :type nei_bool: ndarray
    :param p: the p-atic order to compute (e.g. 2 for nematic, 4 for tetratic, 6 for hexatic), defaults to 2
    :type p: int, optional
    :return: [N] array of the local p-atic parameter and the associated angle of the director
    :rtype: ndarray, ndarray
    """

    nei_angles = np.full(nei_bool.shape,angles)*nei_bool
    nb = nei_bool.sum(axis=-1)
    pp = (np.exp(1j*p*nei_angles)*nei_bool).sum(axis=-1)/nb
    pp[nb==0]=0+0j

    return np.abs(pp), np.angle(pp)/p


# # WIP code for projected p-atic