# -*- coding: utf-8 -*-
"""
Docstring for chains
"""

import numpy as np
from matplotlib.lines import Line2D

def chain_patches(pts, zs):
    """
    Docstring for chain_patches
    
    :param pts: [Nx2x2] array containing N sets of [[x1, y1], [x2, y2]] of projected coordinates representing the termini of bonds between subsequent particles in 2D space
    :param zs: [Nx2] array containing N sets of [zorder1, zorder2] to determine zorder of bonds between subsequent particles for plotting
    """
    
    all_lines = np.array([Line2D(p[:,0], p[:,1], zorder=(np.min([zo[0],zo[1]]) - 0.01)) for p, zo in zip(pts, zs)])
    all_outlines = np.array([Line2D(p[:,0], p[:,1], zorder=(np.min([zo[0],zo[1]]) - 0.02)) for p, zo in zip(pts, zs)])
    
    return all_lines, all_outlines

if __name__ == "__main__":

    # for testing

    pass