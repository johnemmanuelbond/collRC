# -*- coding: utf-8 -*-
"""
Contains methods to calculate orientational order, meaning the order and structure inherent to to the orientations of the particles themselves. These methods generally cover nematic, tetract and hexatic order parameters, which can be generalized as 'p-atic' order.

.. math::

    S_2 \\equiv P_2 \\qquad T_4 \\equiv P_4 \\qquad H_6 \\equiv P_6
    
"""

import numpy as np

def global_patic(angles:np.ndarray, p:int=2) -> tuple[float,float]:
    """
    Computes the global p-atic order of a particle ensemble. Traditionally this is a scalar-valued order parameter which characterizes how aligned particles are with a *director* orientation:

    .. math::
        |P_p| = \\max_{\\theta_p}\\langle\\cos(p(\\theta_j-\\theta_p))\\rangle = |\\langle e^{ip\\theta_j}\\rangle_j|

    But, it is convienient to represent this as a complex-valued quantity which also encodes the director angle:

     .. math:
        `P_p = \\langle e^{i p \\theta_p}\\rangle` = |P_p|e^{i p \\theta_p}.

    :param angles: the orientation of each particle in the frame
    :type angles: ndarray
    :param p: the p-atic order to compute (e.g. 2 for nematic, 4 for tetratic, 6 for hexatic)
    :type p: int
    :return: the complex value of the global p-atic order and the angle which defines it
    :rtype: scalar
    """
    return np.exp(1j*p*angles).mean()

def local_patic(angles:np.ndarray, nei_bool:np.ndarray, p:int =2) -> tuple[np.ndarray,np.ndarray]:
    """
    Computes the local p-atic order, per particle, of an ensemble. Quantifies the local orientational order of a system by calculating a director for particle :math:`j` via its neighboring particle(s) :math:`k`, as defined by `(Baron J. Chem. Phys. 2023) <https://doi.org/10.1063/5.0169659>`_.

    .. math::

        |P_{p,j}| = \\max_{\\theta_{p,j}}\\langle\\cos(p(\\theta_k-\\theta_{p,j}))\\rangle_{r_{jk}<6a_x}

    Or, more succinctly:

    .. math::
        P_{p,j} = \\langle e^{ip\\theta_k}\\rangle_{j(r_{jk}<6a_x)} = |P_{p,j}|e^{i p \\theta_{p,j}}

    :param angles: the orientation of each particle in the frame
    :type angles: ndarray
    :param nei_bool: (N,N) boolean array defining particle neighbors
    :type nei_bool: ndarray
    :param p: the p-atic order to compute (e.g. 2 for nematic, 4 for tetratic, 6 for hexatic), defaults to 2
    :type p: int, optional
    :return: (N,) array of the local p-atic parameter encoded with the associated angle of the director
    :rtype: ndarray, ndarray
    """

    nei_angles = np.full(nei_bool.shape,angles)*nei_bool
    nb = nei_bool.sum(axis=-1)
    pp = (np.exp(1j*p*nei_angles)*nei_bool).sum(axis=-1)
    pp[nb!=0]/=nb[nb!=0]
    pp[nb==0]*=0

    return pp


# # WIP code for projected p-atic