Gallery
=======

Spheres in a quadrupolar electrode
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use :py:class:`ColorPsiG <coloring.bondcolor.ColorPsiG>`, :py:class:`ColorConn <coloring.bondcolor.ColorConn>`, and :py:class:`ColorBlender <coloring.base.ColorBlender>` to reproduce the Ψ6 and C6 reaction coordinates from \_\_\_. Then pass them to :py:meth:`render_npole <render.render.render_npole>` to render particles in a quadrupolar electrode:

.. container:: twocol

   .. container:: leftcol

      .. video:: _static/base-qpole.webm
         :width: 300
         :autoplay:
         :loop:
         :nocontrols:
         :muted:
      
      .. video:: _static/psig-qpole.webm
         :width: 300
         :autoplay:
         :loop:
         :nocontrols:
         :muted:
      
   .. container:: rightcol

      .. video:: _static/c6-qpole.webm
         :width: 300
         :autoplay:
         :loop:
         :nocontrols:
         :muted:
       
      .. video:: _static/psi6c6-qpole.webm
         :width: 300
         :autoplay:
         :loop:
         :nocontrols:
         :muted:


Use :py:class:`ColorPsiPhase <coloring.bondcolor.ColorPsiPhase>` to highlight misoriented crystalline domains:

.. video:: _static/phase-qpole.webm
   :width: 400
   :autoplay:
   :loop:
   :nocontrols:
   :muted:


Use :py:class:`ColorC6Defects <coloring.defectcolor.ColorC6Defects>` to highlight particles on the boundaries of crystalline domains:

.. video:: _static/c6d-qpole.webm
   :width: 400
   :autoplay:
   :loop:
   :nocontrols:
   :muted:


Spheres in an octopolar electrode
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^



Ellipses in a coplanar electrode
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^



Rectangles in a coplanar electrode
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use :py:class:`ColorBase <coloring.base.ColorBase>` and :py:meth:`render_npole <render.render.render_npole>` to render rectanglular particles in a coplanar electrode:

.. container:: twocol

   .. container:: leftcol
      
      .. video:: _static/base-rect1.webm
         :width: 300
         :autoplay:
         :loop:
         :nocontrols:
         :muted:
      
   .. container:: rightcol

      .. video:: _static/base-rect2.webm
         :width: 300
         :autoplay:
         :loop:
         :nocontrols:
         :muted:


Use :py:class:`ColorS2 <coloring.paticcolor.ColorS2>`, :py:class:`ColorS2G <coloring.paticcolor.ColorS2G>`, and :py:class:`ColorS2Defects <coloring.defectcolor.ColorS2Defects>` to showcase nematic order and misorientation defects:

.. container:: twocol

   .. container:: leftcol
      
      .. video:: _static/s2-rect1.webm
         :width: 300
         :autoplay:
         :loop:
         :nocontrols:
         :muted:
      
      .. video:: _static/s2g-rect1.webm
         :width: 300
         :autoplay:
         :loop:
         :nocontrols:
         :muted:
      
      .. video:: _static/s2d-rect1.webm
         :width: 300
         :autoplay:
         :loop:
         :nocontrols:
         :muted:
      
   .. container:: rightcol

      .. video:: _static/s2-rect2.webm
         :width: 300
         :autoplay:
         :loop:
         :nocontrols:
         :muted:

      .. video:: _static/s2g-rect2.webm
         :width: 300
         :autoplay:
         :loop:
         :nocontrols:
         :muted:
      
      .. video:: _static/s2d-rect2.webm
         :width: 300
         :autoplay:
         :loop:
         :nocontrols:
         :muted:


Use :py:class:`ColorT4 <coloring.paticcolor.ColorT4>`, :py:class:`ColorConn <coloring.bondcolor.ColorConn>` and :py:class:`ColorC4Defects <coloring.defectcolor.ColorC4Defects>` to highlight 4-fold order and packing defects:

.. container:: twocol

   .. container:: leftcol
      
      .. video:: _static/t4g-rect1.webm
         :width: 300
         :autoplay:
         :loop:
         :nocontrols:
         :muted:
      
      .. video:: _static/c4-rect1.webm
         :width: 300
         :autoplay:
         :loop:
         :nocontrols:
         :muted:
      
      .. video:: _static/c4d-rect1.webm
         :width: 300
         :autoplay:
         :loop:
         :nocontrols:
         :muted:
    
   .. container:: rightcol

      .. video:: _static/t4g-rect2.webm
         :width: 300
         :autoplay:
         :loop:
         :nocontrols:
         :muted:

      .. video:: _static/c4-rect2.webm
         :width: 300
         :autoplay:
         :loop:
         :nocontrols:
         :muted:
      
      .. video:: _static/c4d-rect2.webm
         :width: 300
         :autoplay:
         :loop:
         :nocontrols:
         :muted:


Spheres on a spherical surface
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use :py:class:`ColorBase <coloring.base.ColorBase>` and :py:meth:`render_sphere <render.render.render_sphere>` to render particles on a spherical surface.

.. video:: _static/base-sphere.webm
   :width: 300
   :autoplay:
   :loop:
   :nocontrols:
   :muted:


Use :py:class:`ColorPsiPhase <coloring.bondcolor.ColorPsiPhase>` to highlight parallel transport problems on curved surfaces

.. video:: _static/phase-sphere.webm
   :width: 300
   :autoplay:
   :loop:
   :nocontrols:
   :muted:


Use :py:class:`ColorConn <coloring.bondcolor.ColorConn>` and :py:class:`ColorC6Defects <coloring.defectcolor.ColorC6Defects>` to highlight defects in crystalline domains on curved surfaces.

.. container:: twocol

   .. container:: leftcol
      
      .. video:: _static/c6-sphere.webm
         :width: 300
         :autoplay:
         :loop:
         :nocontrols:
         :muted:
      
   .. container:: rightcol

      .. video:: _static/c6d-sphere.webm
         :width: 300
         :autoplay:
         :loop:
         :nocontrols:
         :muted:


Small clusters crystallizing in 3D
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use :py:class:`ColorPsiG <coloring.bondcolor.ColorPsiG>`, :py:class:`ColorConn <coloring.bondcolor.ColorConn>`, and :py:class:`ColorBlender<coloring.base.ColorBlender>` to calculate steinhardt parameters. Then pass them to :py:meth:`render_3d <render.render.render_3d>` to render particles in a quadrupolar electrode:

.. container:: twocol

   .. container:: leftcol

      .. video:: _static/base-clust.webm
         :width: 300
         :autoplay:
         :loop:
         :nocontrols:
         :muted:
      
      .. video:: _static/Q6-clust.webm
         :width: 300
         :autoplay:
         :loop:
         :nocontrols:
         :muted:
      
   .. container:: rightcol

      .. video:: _static/C6-clust.webm
         :width: 300
         :autoplay:
         :loop:
         :nocontrols:
         :muted:
       
      .. video:: _static/Q6C6-clust.webm
         :width: 300
         :autoplay:
         :loop:
         :nocontrols:
         :muted: