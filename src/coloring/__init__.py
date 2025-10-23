"""Module src/coloring/__init__.py."""


from .base import base_colors, color_gradient, color_blender
from .base import ColorBase

from .psicolor import ColorByPsi, ColorByPhase, ColorByGlobalPsi, ColorByConn, ColorByGlobalConn

from .paticcolor import ColorByS2, ColorByS2g, ColorS2Phase, ColorByT4, ColorByT4g

from .defectcolor import ColorS2Defects, ColorC4Defects, ColorC6Defects

from .morphcolor import ColorByEta0

from .psicolor import QpoleSuite