# -*- coding: utf-8 -*-
"""
Contains methods to calculate cluster morphological properties, including cluster-averaged bond order parameters, centers of-mass, gyration tensors/radii, and anisotropies.
"""

import numpy as np
from scipy.sparse import lil_matrix
from graph_tool.all import Graph
from graph_tool import topology as gtop
# from .locality import gyration_tensor


def graph_clusters(c6, nei):

    pnum = len(c6)
    is_defect = (np.abs(1-c6)>0.05)
    ii,jj = np.mgrid[0:pnum, 0:pnum]

    match = ((is_defect[ii])==(is_defect[jj]))
    bonds = np.logical_and(nei,match)

    g = Graph(lil_matrix(bonds),directed=False)

    cidx = gtop.label_components(g)[0]
    return cidx.a

def cluster_c6(pts,c6,cidx):

    p_iden, c_iden = np.meshgrid(cidx,np.unique(cidx))
    c_map = (p_iden==c_iden)
    clust_sizes = c_map.sum(axis=-1)

    c6_clust = (c_map @ c6) / clust_sizes

    return c6_clust

def cluster_com(pts,c6,cidx):

    p_iden, c_iden = np.meshgrid(cidx,np.unique(cidx))
    c_map = (p_iden==c_iden)
    clust_sizes = c_map.sum(axis=-1)

    x_clust = (c_map @ pts[:,0]) / clust_sizes
    y_clust = (c_map @ pts[:,1]) / clust_sizes
    z_clust = (c_map @ pts[:,2]) / clust_sizes

    cluster_com = np.array([x_clust,y_clust,z_clust]).T

    return cluster_com

# def cluster_gyrate(pts, c6, cidx):   

#     p_iden, c_iden = np.meshgrid(cidx,np.unique(cidx))
#     c_map = (p_iden==c_iden)
#     clust_gyr = np.array([gyration_tensor(pts[c_map[i]]) for i in range(c_map.shape[0])])

#     return clust_gyr

# def cluster_rg(pts,c6,cidx):
#     p_iden, c_iden = np.meshgrid(cidx,np.unique(cidx))
#     c_map = (p_iden==c_iden)
#     clust_rg = np.array([gyration_radius(pts[c_map[i]]) for i in range(c_map.shape[0])])
#     return clust_rg

# def cluster_ani(pts,c6,cidx):
#     p_iden, c_iden = np.meshgrid(cidx,np.unique(cidx))
#     c_map = (p_iden==c_iden)
#     clust_gyr = np.array([gyration_tensor(pts[c_map[i]]) for i in range(c_map.shape[0])])
#     clust_evals = np.array([np.sort(np.linalg.eig(g)[0]) for g in clust_gyr])
#     clust_rg = np.sqrt(clust_evals.sum(axis=-1))
#     clust_acyl = clust_evals[:,1] - clust_evals[:,0]
#     clust_asph = 1.5*clust_evals[:,2] - 0.5*clust_rg**2
#     clust_ani = (clust_asph**2 + 0.75*clust_acyl**2)/(clust_rg**4)
#     return clust_ani

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