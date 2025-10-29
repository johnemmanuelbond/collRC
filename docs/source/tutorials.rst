Tutorials
=========

Basic collRC functionality
--------------------------

Calculating reaction coordinates
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^



Rendering a system
^^^^^^^^^^^^^^^^^^



Animating simulations
^^^^^^^^^^^^^^^^^^^^^

Once you've made a style using one of the built-in color schemes, rendering and animating a trajectory is a straightforward as calling :py:meth:`animate <render.render.animate>`:

.. code-block:: python

   import gsd.hoomd

   from coloring import ColorBase
   from render import render_npole, animate

   style = ColorBase(dark=True)
   figure_maker = lambda snap: render_npole(snap, style=style, PEL='contour', dark=True, figsize=5, dpi=500)

   fps = 10
   with  gsd.hoomd.open('./qpole1.gsd', 'r') as traj:
      # here you might index the trajectory to select fewer frames, i.e. traj[::10].
      print(f'Animating trajectory with {len(traj)} frames will have duration {len(traj)/fps} seconds at {fps} fps.')
      animate(traj, figure_maker=figure_maker, outpath='./base-qpole1.mp4', fps=fps)


.. video:: _static/base-qpole1.webm
   :width: 300
   :autoplay:
   :loop:
   :nocontrols:
   :muted:


Custom reaction coordinate renders
----------------------------------

Inheriting ColorBase
^^^^^^^^^^^^^^^^^^^^

You may want to define your own color schemes by inheriting from :py:class:`ColorBase <coloring.base.ColorBase>`. The basic items to be aware of are the :py:meth:`calc_state <coloring.base.ColorBase.calc_state>`, :py:meth:`local_colors <coloring.base.ColorBase.local_colors>`, and :py:meth:`color_mapper <coloring.base.ColorBase.state_string>` methods which get called in the :py:mod:`render <render.render>` module.

Below we demonstrate how to create a custom color scheme which colors particles based on their nth nearest neighbor distance. We overwrite ``calc_state`` (using a ``@classmethod``) to perform the necessary neighbor calculations. We overwrite local_colors to explicitly map those distances to colors using a colormap. In this example we ignore ``state_string``.

.. literalinclude:: ../../tutorials/tut_nnfigs.py
   :language: python

At the end of this example we demonstrate a little bit of what goes on in the :py:mod:`render <render.render>` module by grabbing the output image from matplotlib as an RBGA array and sticking it on one of several subplot axes:

.. img:: _static/nth-nearest-neighbors.png
   :width: 600


Inheriting ColorBase subclasses
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

collRC already has a lot of reaction coordinates built-in, so often it's more useful to inherit from an existing subclass of :py:class:`ColorBase <coloring.base.ColorBase>` so you can focus on adding in your own functionality rather than rewriting existing physics. For example, we can inherit from :py:class:`ColorConn <coloring.bondcolor.ColorConn>` to create a color scheme based on the existing :math:`(\psi_6,C_6)` fomalism which highlights particles in only in the most prevalent crystalline domains, which helps for exerting control over the morphology and overall crystallinity of a colloidal system.

.. literalinclude:: ../../tutorials/tut_bicrystal.py
   :language: python

At the end of the file we demonstrate how to write an extended figure making method, using tools in this module, to animate the principal moments of that most prevalent crystal cluster:

.. container:: row-assets

   .. container:: asset

      .. video:: _static/xtal-domains-qpole2.webm
         :width: 300
         :autoplay:
         :loop:
         :nocontrols:
         :muted:

   .. container:: asset

      .. video:: _static/xtal-domains-opole1.webm
         :width: 300
         :autoplay:
         :loop:
         :nocontrols:
         :muted:

Examples
========

Rectangles in a coplanar electrode
----------------------------------

The following code demonstrates how to use the defect-driven color schemes to render movies which highlight defects in systems of self-assembling rectangles. Both :py:class:`ColorS2Defects <coloring.defectcolor.ColorS2Defects>` and :py:class:`ColorC4Defects <coloring.defectcolor.ColorC4Defects>` have functionality which lets us set a background color scheme for the nondefective particles (:py:class:`ColorEta0 <coloring.morphcolor.ColorEta0>` and :py:class:`ColorConn <coloring.bondcolor.ColorConn>` respectively).

.. literalinclude:: ../../tutorials/ex_rect.py
   :language: python

Below we've included the movies for ``rect2.gsd`` in the ``tutorials/`` folder on the github repository, rendered using both color schemes.

.. container:: row-assets

   .. container:: asset

      .. video:: _static/s2d-rect2.webm
         :width: 300
         :autoplay:
         :loop:
         :nocontrols:
         :muted:

   .. container:: asset

      .. video:: _static/c4d-rect2.webm
         :width: 300
         :autoplay:
         :loop:
         :nocontrols:
         :muted:


Discs on a spherical surface
----------------------------


We can use :py:class:`ColorC6Defects <coloring.defectcolor.ColorC6Defects>` to highlight defects in crystalline domains on curved surfaces. 

.. literalinclude:: ../../tutorials/ex_sphere.py
   :language: python

Correctly comparing bond orientational order on curved surfaces requires using the gradient of the implicit function to compute local tangent planes and the rotations between them. Without providing this function (`left`), we see many artificial c6 defects whereas including this correction (`right`) reveals a more coherent defect structure:

.. container:: row-assets

   .. container:: asset
      
      .. video:: _static/c6d-sphere-incorrect.webm
         :width: 300
         :autoplay:
         :loop:
         :nocontrols:
         :muted:
      
   .. container:: asset

      .. video:: _static/c6d-sphere-correct.webm
         :width: 300
         :autoplay:
         :loop:
         :nocontrols:
         :muted:

Small clusters crystallizing in 3D
----------------------------------

The following code demonstrates how to blend two defect-driven color schemes, using :py:class:`ColorBlender <coloring.base.ColorBlender>`, to show the emergence of local, then global, bond orientational order in small clusters which crystallize at a specific osmotic pressure of deplentants. By instantiating a :py:class:`ColorQG <coloring.bondcolor.ColorQG>` style and a :py:class:`ColorConn <coloring.bondcolor.ColorConn>` style, we can blend them together with a custom colormap to render both at once.

.. literalinclude:: ../../tutorials/ex_clust.py
   :language: python

Which produces this movie (and accompanying colormap):

.. container:: row-assets

   .. container:: asset

      .. video:: _static/Q6C6-clust.webm
         :width: 300
         :autoplay:
         :loop:
         :nocontrols:
         :muted:

   .. container:: asset
      
      .. image:: _static/white-purp.png
         :height: 300
         :alt: white-purp.png

