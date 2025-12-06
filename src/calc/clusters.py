# -*- coding: utf-8 -*-
"""
Contains methods to calculate cluster morphological properties, including cluster-averaged bond order parameters, centers of-mass, gyration tensors/radii, and anisotropies.
"""

import numpy as np
from scipy.sparse import lil_matrix
from graph_tool.all import Graph
from graph_tool import topology as gtop

from .morphology import gyration_tensor
from .morphology import gyration_radius
from .morphology import shape_anisotropy


def c6_clusters(c6:np.ndarray, nei:np.ndarray[bool]) -> np.ndarray[int]:
    """
    Uses graph theory to identify clusters of particles based on their defect status (as defined by c6) and neighbor connectivity (as defined by nei).
    
    :param: c6: an :math:`[N,]` array of complex bond order parameters for each particle
    :type c6: ndarray
    :param: nei: a :math:`[N,N]` boolean array defining particle neighbors
    :type nei: ndarray
    :return: an :math:`[N,]` array of cluster indices for each particle
    :rtype: ndarray[int]
    """

    pnum = len(c6)
    is_defect = (np.abs(1-c6)>0.05)
    ii,jj = np.mgrid[0:pnum, 0:pnum]

    match = ((is_defect[ii])==(is_defect[jj]))
    bonds = np.logical_and(nei,match)

    g = Graph(lil_matrix(bonds),directed=False)

    cidx = gtop.label_components(g)[0]
    return cidx.a


def cluster_averager(cidx:np.ndarray[int]) -> tuple[np.ndarray[bool], np.ndarray[int]]:
    """
    Computes cluster membership map and cluster sizes from cluster indices. These arrays can be used to compute cluster-averaged properties. :code:`cluster_averager` is used internally by other cluster analysis functions to compute cluster-averaged quantities efficiently, any cluster-averaged quantity can be computed array-qise ysing :code:`c_map @ quantity / c_sizes`.

    :param: cidx: an :math:`[N,]` array of cluster indices for each particle
    :type cidx: ndarray[int]
    :return: a tuple containing:
        - **c_map**: a :math:`[N,C]` boolean array mapping particles to clusters
        - **c_sizes**: a :math:`[C,]` array of cluster sizes
    :rtype: tuple[np.ndarray[bool], np.ndarray[int]]
    """

    p_iden, c_iden = np.meshgrid(cidx,np.unique(cidx))
    c_map = (p_iden==c_iden)
    c_sizes = c_map.sum(axis=-1)

    return c_map, c_sizes


def cluster_com(pts:np.ndarray,cidx:np.ndarray[int]) -> np.ndarray:
    """
    Computes the center of mass for each cluster based on particle positions and their cluster indices.

    :param: pts: an :math:`[N,d]` array of particle positions in :math:`d` dimensions
    :type pts: ndarray
    :param: cidx: an :math:`[N,]` array of cluster indices for each particle
    :type cidx: ndarray[int]
    :return: an :math:`[C,d]` array of cluster centers of mass
    :rtype: ndarray
    """

    c_map, c_sizes = cluster_averager(cidx)
    return np.array([(c_map @ dim)/c_sizes for dim in pts.T ]).T


def cluster_rg(pts:np.ndarray,cidx:np.ndarray[int]) -> np.ndarray:
    """
    Computes the radius of gyration using the :py:meth:`gyration_radius() <calc.morphology.gyration_radius>` method for each cluster based on particle positions and their cluster indices.

    :param: pts: an :math:`[N,d]` array of particle positions in :math:`d` dimensions
    :type pts: ndarray
    :param: cidx: an :math:`[N,]` array of cluster indices for each particle
    :type cidx: ndarray[int]
    :return: an :math:`[C,]` array of cluster radii of gyration
    :rtype: ndarray
    """

    c_map, _ = cluster_averager(cidx)
    clust_rg = np.array([gyration_radius(pts[c_map[i]]) for i in range(c_map.shape[0])])
    return clust_rg


def cluster_gyrate(pts:np.ndarray, cidx:np.ndarray[int]) -> np.ndarray:   
    """
    Computes the gyration tensor, using the :py:meth:`gyration_tensor() <calc.morphology.gyration_tensor>` method, for each cluster based on particle positions and their cluster indices.

    :param: pts: an :math:`[N,d]` array of particle positions in :math:`d` dimensions
    :type pts: ndarray
    :param: cidx: an :math:`[N,]` array of cluster indices for each particle
    :type cidx: ndarray[int]
    :return: an :math:`[C,d,d]` array of cluster gyration tensors
    :rtype: ndarray
    """

    c_map, _ = cluster_averager(cidx)
    clust_gyr = np.array([gyration_tensor(pts[c_map[i]]) for i in range(c_map.shape[0])])

    return clust_gyr


def cluster_shape(pts:np.ndarray, cidx:np.ndarray[int]) -> np.ndarray:
    """
    Computes the shape anisotropy (:math:`\\kappa^2`) using the :py:meth:`shape_anisotropy() <calc.morphology.shape_anisotropy>` method for each cluster based on particle positions and their cluster indices.

    :param: pts: an :math:`[N,d]` array of particle positions in :math:`d` dimensions
    :type pts: ndarray
    :param: cidx: an :math:`[N,]` array of cluster indices for each particle
    :type cidx: ndarray[int]
    :return: an :math:`[C,]` array of cluster shape anisotropies :math:`\\kappa^2`
    :rtype: ndarray
    """

    clust_gyr = cluster_gyrate(pts,cidx)
    kaps = np.array([shape_anisotropy(pts=None, gyr = g) for g in clust_gyr])
    return kaps

# def defect_ani(pts,c6,cidx):
#     clust_c6 = cluster_c6(pts,c6,cidx)
#     clust_ani = cluster_ani(pts,c6,cidx)
#     csizes = np.bincount(cidx)

#     is_defect = (np.abs(1-clust_c6)>0.05)
#     is_cluster = (csizes>1)
#     include = np.logical_and(is_defect,is_cluster)

#     ave_ani = np.sum(clust_ani[include]*csizes[include])/np.sum(csizes[include])

#     return ave_ani



if __name__ == "__main__":
    pass