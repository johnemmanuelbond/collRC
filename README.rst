collRC
======

.. description.rst

A library for calculating *reaction coordinates* which describe local and global dynamics within self-assembling colloidal ensembles. Reaction coordinates are instantaneous measures that capture structural features of particle configurations, so that they may be characterized over time to measure and represent collective dynamic phenomema.

Based out of the `bevan lab`_, collRC provides functionality for calculating reaction coordinates and using them to render colored visualizations of the colloidal ensemble. collRC is particularly built to process and visualize the output of simulations run with `hoomd-blue`_ which are commonly saved to the `gsd`_ format.

.. _bevan lab: https://bevan.jh.edu/
.. _hoomd-blue: https://hoomd-blue.readthedocs.io/en/latest/
.. _gsd: https://gsd.readthedocs.io/en/latest/

.. intro.rst

Getting Started
===============

Prerequisites
*************

| color-scheming is easily set up using anaconda. Create a conda environment with the following packages installed.
| \- `numpy`_, `scipy`_, `graph-tool`_: for calculating reaction coordinates
| \- `gsd`_: to store simulated particle trajectories
| \- `matplotlib`_, `moviepy`_: to render movies of particle trajectories
| \- `ffmpeg`_: is automatically installed as a dependency of `moviepy`_

.. code-block:: bash

   $ conda create -n ENV_NAME
   $ conda activate ENV_NAME
   $ conda install -c conda-forge numpy scipy graph-tool gsd matplotlib moviepy

Installation
************

After installing prerequesites, clone the git repository:

.. code-block:: bash

   $ git clone https://github.com/johnemmanuelbond/collRC path_to_repo

Users may install the repository into a conda environment using :code:`conda develop`

.. code-block:: bash

   $ conda develop path_to_repo/src

Uninstall with

.. code-block:: bash

   $ conda develop -u path_to_repo/src


.. _numpy: https://numpy.org/doc/stable/
.. _scipy: https://docs.scipy.org/doc/scipy/
.. _graph-tool: https://graph-tool.skewed.de/static/docs/stable/
.. _gsd: https://gsd.readthedocs.io/en/latest/
.. _matplotlib: https://matplotlib.org/stable/contents.html
.. _ffmpeg: https://www.ffmpeg.org/documentation.html
.. _moviepy: https://zulko.github.io/moviepy/

Documentation
=============

The full documentation is available at https://collRC.readthedocs.io/en/latest/