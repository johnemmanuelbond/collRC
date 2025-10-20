color-scheming
==============

.. description.rst

A framework for rendering movies of colloidal particles based on local and global reaction coordinates (low-dimensional representations of high-dimensional structures). color-scheming is particularly built to process and visualize the output of simulations run with `hoomd-blue`_ which are commonly saved to the `gsd`_ format.

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
| \- `matplotlib_`, `moviepy_`: to render movies of particle trajectories
| \- `ffmpeg`_: is automatically installed as a dependency of `moviepy`_

.. code-block:: bash

   $ conda create -n ENV_NAME
   $ conda activate ENV_NAME
   $ conda install -c conda-forge numpy scipy graph-tool gsd matplotlib moviepy

Installation
************

After installing prerequesites, clone the git repository:

.. code-block:: bash

   $ git clone https://github.com/johnemmanuelbond/color-scheming path_to_repo

Users may install the repository into a conda environment using :code:`conda develop`

.. code-block:: bash

   $ conda develop path_to_repo/src

Uninstall with

.. code-block:: bash

   $ conda develop -u path_to_repo/src


.. _numpy: https://numpy.org/doc/stable/
.. _scipy: https://docs.scipy.org/doc/scipy/
.. _graph-tool: https://graph-tool.skewed.de/static/docs/latest/
.. _gsd: https://gsd.readthedocs.io/en/latest/
.. _matplotlib: https://matplotlib.org/stable/contents.html
.. _ffmpeg: https://www.ffmpeg.org/documentation.html
.. _moviepy: https://zulko.github.io/moviepy/