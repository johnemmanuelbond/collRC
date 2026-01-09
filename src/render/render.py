# -*- coding: utf-8 -*-
"""
Contains methods for rendering particle configurations on matplotlib figures. Contains methods to animate rendered matplotlib figures into moviepy movies.
"""

import numpy as np
import gsd.hoomd
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection, LineCollection

from moviepy.video.VideoClip import VideoClip

from visuals import Field
from visuals import contour_PEL, spectral_PEL, flat_patches, projected_patches, chain_patches
from coloring import ColorBase

def render_npole(snap:gsd.hoomd.Frame, style:ColorBase,
                 PEL='contour', show_text=True,
                 dpi=600, figsize=3.5, dark=True,
                 **kwargs):
    """
    Create a visualization of a single GSD frame with superellipse particles and optional field overlays.

    :param snap: GSD frame object containing particle data and simulation state
    :type snap: gsd.hoomd.Frame
    :param style: :py:class:`ColorBase` object for coloring scheme
    :type style: ColorBase
    :param PEL: Type of potential energy landscape (PEL) overlay to render; options are 'contour' or 'spectral' (default: 'contour')
    :type PEL: str, optional
    :param show_text: Whether to display text annotations (default: True)
    :type show_text: bool, optional
    :param dpi: Resolution for output image (default: 600)
    :type dpi: int, optional
    :param figsize: Base figure size in inches (default: 3.5)
    :type figsize: float, optional
    :param dark: Whether to use dark background theme (default: True)
    :type dark: bool, optional
    :param kwargs: Additional options for customizating action strings and other overlays
    :type kwargs: dict
    :return: Matplotlib figure and axis objects
    :rtype: Tuple[plt.Figure, plt.Axes]
    """
    
    # Set matplotlib style based on background preference
    if dark: plt.style.use('dark_background')

    pts = snap.particles.position
    qts = snap.particles.orientation

    # Update reaction coordinates inside coloring style
    style.snap = snap
    # style.calc_state()

    # Setup figure with aspect ratio matching simulation box
    Lx, Ly, _, _, _, _ = snap.configuration.box
    fig, ax = plt.subplots(figsize=(figsize, figsize*Ly/Lx), dpi=dpi)
    ax.set_xlim([-Lx/2, Lx/2])
    ax.set_ylim([-Ly/2, Ly/2])
    ax.set_aspect('equal')

    # Apply coloring and styling
    local_colors = style.local_colors()
    patches = flat_patches(pts, qts, shape = style.shape)
    ptcls = ax.add_collection(PatchCollection(patches), autolim=True)
    ptcls.set_fc(local_colors)  # Face colors from coloring scheme
    if dark:
        ptcls.set_ec('grey')    # Edge color for dark background
    else:
        ptcls.set_ec('black')   # Edge color for light background
    ptcls.set_lw(0.5)          # Edge line width

    # Extract electrode/field parameters from simulation log
    try:
        field = Field.create_from_gsd(snap)
    except KeyError:
        field = None
        PEL = None
        
    match PEL:
        case 'spectral':
            spec = spectral_PEL(ax=ax, field=field, dark=dark)
        case 'contour':
            colors = 'white' if dark else 'black'
            cont = contour_PEL(ax=ax, field=field, colors=colors)

    if 'act_string' in kwargs:
            act_string = kwargs['act_string'](snap)
    else:
        try:
            # Generate text annotation for field strength
            dg = field.electrode_gap
            kt = field.k_trans
            act_string = "".join([f"$k_{{{i+1}}}/d_g^2$ = {k/dg**2:.2f}, " for i, k in enumerate(kt)])
        except AttributeError:
            act_string = ""

    # Add text annotations in top-left corner
    state_string = style.state_string()
    if (state_string == "") and (act_string == ""):
        show_text = False
    if show_text:
        if state_string == "":
            textbox = act_string
        elif act_string == "":
            textbox = state_string
        else:
            textbox = f"{act_string}\n{state_string}"
        ax.text(-Lx/2, Ly/2, textbox,
                backgroundcolor='white', zorder=2, color='k',
                horizontalalignment='left', verticalalignment='top')

    # Clean up plot appearance
    ax.axis('off')
    fig.tight_layout()

    return fig, ax


def render_3d(snap:gsd.hoomd.Frame, style:ColorBase,
              view_dir = np.array([0,0,1]), view_dist=50, show_text=True,
              dpi=600, figsize=3.5, dark=True, **kwargs):
    """
    Create a visualization of a single GSD frame with particles in a 3d space.

    :param snap: GSD frame object containing particle data and simulation state
    :type snap: gsd.hoomd.Frame
    :param style: :py:class:`ColorBase` object for coloring scheme
    :type style: ColorBase
    :param view_dir: Direction vector for viewing the sphere (default: [0,0,1])
    :type view_dir: ndarray, optional
    :param view_dist: Distance from the sphere center to the viewpoint (default: 100)
    :type view_dist: float, optional
    :param show_text: Whether to display text annotations (default: True)
    :type show_text: bool, optional
    :param dpi: Resolution for output image (default: 600)
    :type dpi: int, optional
    :param figsize: Base figure size in inches (default: 3.5)
    :type figsize: float, optional
    :param dark: Whether to use dark background theme (default: True)
    :type dark: bool, optional
    :param kwargs: Additional options for customizating action strings and other overlays. Can include 'Lx', 'Ly' or 'L' to specify static box dimensions. Defaults to box dimensions from GSD frame, but since these may change over the course of a trajectory, specifying fixed values can help maintain consistent aspect ratios across frames. 
    :type kwargs: dict
    :return: Matplotlib figure and axis objects
    :rtype: Tuple[plt.Figure, plt.Axes]
    """
    
    # Set matplotlib style based on background preference
    if dark: plt.style.use('dark_background')
    
    pts = snap.particles.position
    # qts = snap.particles.orientation
    qts = np.array([view_dir]*pts.shape[0])

    # Update reaction coordinates inside coloring style
    style.snap = snap
    # style.calc_state()
    
    # Setup figure with aspect ratio matching simulation box
    if 'Lx' in kwargs and 'Ly' in kwargs:
        Lx = kwargs['Lx']
        Ly = kwargs['Ly']
    elif 'L' in kwargs:
        Lx = kwargs['L']
        Ly = kwargs['L']
    else:
        Lx, Ly, _, _, _, _ = snap.configuration.box
    
    if 'view_ref' in kwargs:
        view_ref = kwargs['view_ref']
    else:
        view_ref = 'z'

    if 'parallax' in kwargs:
        parallax = kwargs['parallax']
    else:
        parallax = True

    fig, ax = plt.subplots(figsize=(figsize, figsize*Ly/Lx), dpi=dpi)
    ax.set_xlim([-Lx/2, Lx/2])
    ax.set_ylim([-Ly/2, Ly/2])
    ax.set_aspect('equal')

    # Apply coloring and styling
    local_colors = style.local_colors()
    patches, _ = projected_patches(pts, qts, shape = style.shape,
                                           view_dir=view_dir, view_dist=view_dist, view_ref=view_ref,
                                           parallax = parallax)
    
    # we can make collecions of both patches AND line, so we should shorten this code!

    # Add and layer bonds if specified in kwargs
    if 'chain' in kwargs and kwargs['chain']:       # start with boolean, assume subsequent particles are connected
        centers = np.array([patch.get_transform().transform(patch.get_xy())[:-1].mean(axis=0)
                    for patch in patches]) #[to_render].tolist()
        zos = np.array([patch.get_zorder() for patch in patches]) #[to_render].tolist()
        bonded_ptcls = np.stack([centers[:-1], centers[1:]], axis=1)
        bonded_zs = np.stack([zos[:-1], zos[1:]], axis=1)
        lines, outlines = chain_patches(pts=bonded_ptcls, zs=bonded_zs)
        if dark:
            for i in range(len(patches)):
                ptcl = ax.add_collection(PatchCollection([patches[i]]), autolim=True) # patches[to_render].tolist()
                ptcl.set_fc(local_colors[i])  # Face colors from coloring scheme, local_colors[to_render]
                ptcl.set_ec('grey')    # Edge color for dark background
                ptcl.set_lw(0.5)       # Edge line width
                ptcl.set_zorder(zos[i])
            for i in range(len(lines)):
                bond_outline = ax.add_line(outlines[i])
                bond = ax.add_line(lines[i])
                bond_outline.set_color('grey')
                bond_outline.set_linewidth(6)
                bond.set_color('white')
                bond.set_linewidth(5)
        else:
            for i in range(len(patches)):
                ptcl = ax.add_collection(PatchCollection([patches[i]]), autolim=True) # patches[to_render].tolist()
                ptcl.set_fc(local_colors[i])  # Face colors from coloring scheme, local_colors[to_render]
                ptcl.set_ec('black')    # Edge color for dark background
                ptcl.set_lw(0.5)        # Edge line width
                ptcl.set_zorder(zos[i])
            for i in range(len(lines)):
                bond_outline = ax.add_line(outlines[i])
                bond_outline.set_color('black')
                bond_outline.set_linewidth(6)
                bond = ax.add_line(lines[i])
                bond.set_color('lightgray')
                bond.set_linewidth(5)
    else:
        zos = np.array([patch.get_zorder() for patch in patches])
        order = np.argsort(zos)
        patches_sorted = patches[order]
        colors_sorted = local_colors[order]
        ptcls = ax.add_collection(PatchCollection(patches_sorted.tolist()), autolim=True) # patches[to_render].tolist()
        ptcls.set_fc(colors_sorted) # Face colors from coloring scheme, local_colors[to_render]
        if dark:
            ptcls.set_ec('grey')    # Edge color for dark background
        else:
            ptcls.set_ec('black')   # Edge color for light background
        ptcls.set_lw(0.5)           # Edge line width

    # Add text annotations in top-left corner
    state_string = style.state_string()
    if show_text:
        ax.text(-1.0*Lx/2, 0.9*Ly/2, state_string,
                backgroundcolor='white', fontsize='large', zorder=2, color='k')

    # Clean up plot appearance
    ax.axis('off')
    fig.tight_layout()

    return fig, ax


def render_sphere(snap:gsd.hoomd.Frame, style:ColorBase,
                  view_dir = np.array([0,0,1]), view_dist=100, show_text=True,
                  dpi=600, figsize=3.5, dark=True, **kwargs):
    """
    Create a visualization of a single GSD frame with particles on a spherical surface.

    :param snap: GSD frame object containing particle data and simulation state
    :type snap: gsd.hoomd.Frame
    :param style: :py:class:`ColorBase` object for coloring scheme
    :type style: ColorBase
    :param view_dir: Direction vector for viewing the sphere (default: [0,0,1])
    :type view_dir: ndarray, optional
    :param view_dist: Distance from the sphere center to the viewpoint (default: 100)
    :type view_dist: float, optional
    :param show_text: Whether to display text annotations (default: True)
    :type show_text: bool, optional
    :param dpi: Resolution for output image (default: 600)
    :type dpi: int, optional
    :param figsize: Base figure size in inches (default: 3.5)
    :type figsize: float, optional
    :param dark: Whether to use dark background theme (default: True)
    :type dark: bool, optional
    :param kwargs: Additional options for customizating action strings and other overlays. Can include 'Lx', 'Ly' or 'L' to specify static box dimensions. Defaults to box dimensions from GSD frame, but since these may change over the course of a trajectory, specifying fixed values can help maintain consistent aspect ratios across frames. 
    :type kwargs: dict
    :return: Matplotlib figure and axis objects
    :rtype: Tuple[plt.Figure, plt.Axes]
    """
    
    # Set matplotlib style based on background preference
    if dark: plt.style.use('dark_background')
    
    pts = snap.particles.position
    R = np.linalg.norm(pts, axis=-1).mean()
    qts = pts/R

    # Update reaction coordinates inside coloring style
    style.snap = snap
    # style.calc_state()
    
    # Setup figure with aspect ratio matching simulation box
    if 'Lx' in kwargs and 'Ly' in kwargs:
        Lx = kwargs['Lx']
        Ly = kwargs['Ly']
    elif 'L' in kwargs:
        Lx = kwargs['L']
        Ly = kwargs['L']
    else:
        Lx, Ly, _, _, _, _ = snap.configuration.box
    
    if 'view_ref' in kwargs:
        view_ref = kwargs['view_ref']
    else:
        view_ref = 'z'

    fig, ax = plt.subplots(figsize=(figsize, figsize*Ly/Lx), dpi=dpi)
    ax.set_xlim([-Lx/2, Lx/2])
    ax.set_ylim([-Ly/2, Ly/2])
    ax.set_aspect('equal')

    dr = np.linspace(-R,R,1000)
    xx,yy = np.meshgrid(dr, dr)
    rads = np.sqrt(xx**2+yy**2)
    rads[rads>R]=np.nan
    C = (rads/R)**2/2
    ax.pcolormesh(dr,dr,C,cmap='binary',vmin=0,vmax=1)

    # Apply coloring and styling
    local_colors = style.local_colors()
    patches, to_render = projected_patches(pts, qts, shape = style.shape,
                                           view_dir=view_dir, view_dist=view_dist, view_ref=view_ref,
                                           parallax = False, centered=False)
    ptcls = ax.add_collection(PatchCollection(patches[to_render].tolist()), autolim=True)
    ptcls.set_fc(local_colors[to_render])  # Face colors from coloring scheme
    if dark:
        ptcls.set_ec('grey')    # Edge color for dark background
    else:
        ptcls.set_ec('black')   # Edge color for light background
    ptcls.set_lw(0.5)          # Edge line width

    # Add text annotations in top-left corner
    state_string = style.state_string()
    r_string = f"$R/2a = {R/style.shape.ay/2:.2f}$"
    if show_text:
        if state_string == "":
            textbox = r_string
        else:
            textbox = f"{r_string}\n{state_string}"
        ax.text(-1.0*Lx/2, 0.9*Ly/2, textbox,
                backgroundcolor='white', fontsize='large', zorder=2, color='k')

    # Clean up plot appearance
    ax.axis('off')
    fig.tight_layout()

    return fig, ax


def render_surf(snap:gsd.hoomd.Frame, style:ColorBase, gradient:callable,
                  view_dir = np.array([0,0,1]), view_dist=100, show_text=True,
                  dpi=600, figsize=3.5, dark=True, **kwargs):
    """
    Create a visualization of a single GSD frame with particles on an arbitrary surface defined by a gradient function which computes the surface normal at each point.

    :param snap: GSD frame object containing particle data and simulation state
    :type snap: gsd.hoomd.Frame
    :param style: :py:class:`ColorBase` object for coloring scheme
    :type style: ColorBase
    :param gradient: Function that computes surface normal vectors at given positions
    :type gradient: callable
    :param view_dir: Direction vector for viewing the sphere (default: [0,0,1])
    :type view_dir: ndarray, optional
    :param view_dist: Distance from the sphere center to the viewpoint (default: 100)
    :type view_dist: float, optional
    :param show_text: Whether to display text annotations (default: True)
    :type show_text: bool, optional
    :param dpi: Resolution for output image (default: 600)
    :type dpi: int, optional
    :param figsize: Base figure size in inches (default: 3.5)
    :type figsize: float, optional
    :param dark: Whether to use dark background theme (default: True)
    :type dark: bool, optional
    :param kwargs: Additional options for customizating action strings and other overlays. Can include 'Lx', 'Ly' or 'L' to specify static box dimensions. Defaults to box dimensions from GSD frame, but since these may change over the course of a trajectory, specifying fixed values can help maintain consistent aspect ratios across frames. 
    :type kwargs: dict
    :return: Matplotlib figure and axis objects
    :rtype: Tuple[plt.Figure, plt.Axes]
    """
    
    # Set matplotlib style based on background preference
    if dark: plt.style.use('dark_background')
    
    pts = snap.particles.position
    grads = gradient(pts.T)
    grads = grads/np.linalg.norm(grads, axis=-1, keepdims=True)
    qts = np.array([np.zeros(pts.shape[0]), *grads.T]).T

    # Update reaction coordinates inside coloring style
    style.snap = snap
    # style.calc_state()
    
    # Setup figure with aspect ratio matching simulation box
    if 'Lx' in kwargs and 'Ly' in kwargs:
        Lx = kwargs['Lx']
        Ly = kwargs['Ly']
    elif 'L' in kwargs:
        Lx = kwargs['L']
        Ly = kwargs['L']
    else:
        Lx, Ly, _, _, _, _ = snap.configuration.box
    
    if 'view_ref' in kwargs:
        view_ref = kwargs['view_ref']
    else:
        view_ref = 'z'

    fig, ax = plt.subplots(figsize=(figsize, figsize*Ly/Lx), dpi=dpi)
    ax.set_xlim([-Lx/2, Lx/2])
    ax.set_ylim([-Ly/2, Ly/2])
    ax.set_aspect('equal')

    # dr = np.linspace(-R,R,1000)
    # xx,yy = np.meshgrid(dr, dr)
    # rads = np.sqrt(xx**2+yy**2)
    # rads[rads>R]=np.nan
    # C = (rads/R)**2/2
    # ax.pcolormesh(dr,dr,C,cmap='binary',vmin=0,vmax=1)

    # Apply coloring and styling
    local_colors = style.local_colors()
    patches, to_render = projected_patches(pts, qts[:,1:], shape = style.shape,
                                           view_dir=view_dir, view_dist=view_dist, view_ref=view_ref,
                                           parallax = False, centered=True)
    ptcls = ax.add_collection(PatchCollection(patches[to_render].tolist()), autolim=True)
    ptcls.set_fc(local_colors[to_render])  # Face colors from coloring scheme
    if dark:
        ptcls.set_ec('grey')    # Edge color for dark background
    else:
        ptcls.set_ec('black')   # Edge color for light background
    ptcls.set_lw(0.5)          # Edge line width

    # Add text annotations in top-left corner
    state_string = style.state_string()
    if show_text:
        ax.text(-1.0*Lx/2, 0.9*Ly/2, state_string,
                backgroundcolor='white', fontsize='large', zorder=2, color='k')

    # Clean up plot appearance
    ax.axis('off')
    fig.tight_layout()

    return fig, ax


def animate(render_frames:gsd.hoomd.HOOMDTrajectory, outpath:str, figure_maker:callable, fps:int=20, codec='mpeg4'):
    """
    Render a movie from the given frames using the specified figure maker function.
    
    The :code:`figure maker` function should take a GSD frame and return a matplotlib figure.

    :param render_frames: GSD trajectory containing frames to render
    :type render_frames: gsd.hoomd.HOOMDTrajectory
    :param outpath: Output file path for the rendered movie
    :type outpath: str
    :param figure_maker: Function that generates a matplotlib figure from a GSD frame
    :type figure_maker: callable
    :param fps: Frames per second for the output movie, defaults to 20
    :type fps: int, optional
    :param codec: Codec to use for movie encoding, defaults to 'mpeg4'
    :type codec: str, optional
    """

    def _render(t):
        # Convert movie time to frame index
        fig,ax = figure_maker(render_frames[int(t*fps)])
        fig.canvas.draw()

        # Extract RGB data from matplotlib figure
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf.shape = (h, w, 4)  # Height, width, RGBA channels

        plt.close(fig)  # Clean up memory
        return buf[:, :, :-1]  # Return RGB only (drop alpha channel)

    clip = VideoClip(_render, duration=len(render_frames)/fps)
    clip.write_videofile(outpath, fps=fps, codec=codec)
    clip.close()

if __name__ == "__main__":
    
    from visuals import SuperEllipse


    # from coloring import QpoleSuite
    # sphere = SuperEllipse(ax=0.5,ay=0.5,n=2)
    # style = QpoleSuite(dark=True)

    # figure_maker = lambda snap: render_npole(snap, style=style, PEL='contour', dark=True, figsize=4, dpi=600)
    # frames = gsd.hoomd.open('../tests/test-opole1.gsd',mode='r')
    # f_select = frames[::25]
    # animate(f_select, outpath='../tests/test-opole1.mp4', figure_maker=figure_maker, fps=10, codec='mpeg4')


    # from coloring import ColorByConn, ColorC4Defects
    # rect = SuperEllipse(ax=1.0,ay=0.5,n=20)
    # # bg_style = ColorByConn(shape=rect, order=4, dark=True)
    # # style = ColorC4Defects(shape=rect, dark=True, bgColor=bg_style)

    # from coloring import ColorByEta0, ColorS2Defects
    # bg_style = ColorByEta0(shape=rect, dark=True)
    # style = ColorS2Defects(shape=rect, dark=True, bgColor=bg_style)

    # figure_maker = lambda snap: render_npole(snap, style=style, PEL='contour', dark=True, figsize=5, dpi=500)
    # frames = gsd.hoomd.open('../tests/test-rect.gsd',mode='r')
    # f_select = frames[::10]
    # animate(f_select, outpath='../tests/test-rect.mp4', figure_maker=figure_maker, fps=10, codec='mpeg4')

    sphere_grad = lambda r: r/np.linalg.norm(r, axis=-1, keepdims=True)

    from coloring import ColorByPsi
    style = ColorByPsi(dark=True, surface_normal=sphere_grad)

    from coloring import ColorByConn, ColorC6Defects
    bg_style = ColorByConn(dark=True, order=6, surface_normal=sphere_grad)
    style = ColorC6Defects(dark=True, bgColor=bg_style, surface_normal=sphere_grad)

    frames = gsd.hoomd.open('../tests/test-sphere.gsd',mode='r')
    f_select = frames[0:600:3]
    R0 = np.linalg.norm(frames[0].particles.position, axis=-1).mean()
    L0 = 1.05*R0*2

    figure_maker = lambda snap: render_sphere(snap, style=style, dark=True, figsize=5, dpi=500, L=L0)
    # animate(f_select, outpath='../tests/test-sphere.mp4', figure_maker=figure_maker, fps=10, codec='mpeg4')

    f_select = frames[0:100:2]
    animate(f_select, outpath='../tests/test-sphere-short.webm', figure_maker=figure_maker, fps=5, codec='libvpx')