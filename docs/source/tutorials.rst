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
   with  gsd.hoomd.open('./qpole.gsd', 'r') as traj:
      # here you might index the trajectory to select fewer frames, i.e. traj[::10].
      print(f'Animating trajectory with {len(traj)} frames will have duration {len(traj)/fps} seconds at {fps} fps.')
      animate(traj, figure_maker=figure_maker, outpath='./base-qpole.mp4', fps=fps)


.. video:: _static/base-qpole.webm
   :width: 300
   :autoplay:
   :loop:
   :nocontrols:
   :muted:


Custom reaction coordinate renders
----------------------------------

Inheriting ColorBase
^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import numpy as np
   from matplotlib.cm import hsv as hsv_map
   rainbow = lambda a: hsv_map(a).clip(0, 1)

   from visuals import SuperEllipse
   from calc import stretched_neighbors

   default_sphere = SuperEllipse(ax=0.5, ay=0.5, n=2.0)

   class neicolor(ColorBase):
      def __init__(self, shape = default_sphere, dark = True, ptcl = 10):
         super().__init__(shape, dark)
         self._c = lambda n: rainbow(n)
         self._i = ptcl

      def calc_state(self):
         super().calc_state()
         pts = self.snap.particles.position
         ang = quat_to_angle(self.snap.particles.orientation)
         nei = stretched_neighbors(pts, ang, rx=self._shape.ax, ry=self._shape.ay, neighbor_cutoff=2.8)
         self.nei = nei

      def local_colors(self, snap: gsd.hoomd.Frame = None):
         if snap is not None: self.snap = snap
         col = np.array([white]*self.snap.particles.N)
         col[self._i] = self._c(np.zeros(1))
         for n in range(1, 6):
               nn = self.nth_neighbors(self.nei, n=n)
               col[nn[self._i]] = self._c(np.array([(n+1)/10]))
         return col

      @classmethod
      def nth_neighbors(cls, nei, n=1):
         
         n_nei = nei
         old_nei = np.logical_or(nei, np.eye(nei.shape[0], dtype=bool))
         for _ in range(n-1):
               new_nei = (nei@n_nei)

               new_nei[old_nei] = False
               old_nei[new_nei] = True                

               n_nei = new_nei

         return n_nei


Inheriting ColorBase subclasses
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^





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

.. row-assets::

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

