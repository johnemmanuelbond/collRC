calc module
===========

collRC is all about calculating and representing reaction coordinates from simulated colloidal ensembles. This module contains all of the framework to build up the various calculations needed to compute robust reaction coordinates:

The :py:mod:`locality <calc.locality>` module processes particle positions to compute local environment metrics like nearest neighbors in flat, stretched, and projected 2D spaces.

The :py:mod:`morphology <calc.morphology>` module computes information like local density and gyration tensors of ensembles and ensemble subsets.

The :py:mod:`orient_order <calc.orient_order>` module computes complex-valued local and global p-atic orientational order.

The :py:mod:`bond_order <calc.bond_order>` module computes complex-valued bond orientational order parameters in flat, stretched, and projected 2D spaces.

The :py:mod:`clusters <calc.clusters>` module computes identifies spatially correlated groups of particles and computes statistics on their sizes and shapes.

locality submodule
------------------

.. automodule:: calc.locality
    :members:
    :undoc-members:
    :show-inheritance:

morphology submodule
--------------------

.. automodule:: calc.morphology
    :members:
    :undoc-members:
    :show-inheritance:

orient_order submodule
----------------------

.. automodule:: calc.orient_order
    :members:
    :undoc-members:
    :show-inheritance:

bond_order submodule
--------------------

.. automodule:: calc.bond_order
    :members:
    :undoc-members:
    :show-inheritance:

clusters submodule
------------------

.. automodule:: calc.clusters
    :members:
    :undoc-members:
    :show-inheritance:
