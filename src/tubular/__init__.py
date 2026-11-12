"""TubULAR's experimental Python numerical core.

This package does not yet replace the MATLAB analysis pipeline.
"""

from .mesh import Mesh

__version__ = "0.2.0a1"

from .experiment import MeshSource, TubULAR
from ._method_catalog import MethodNotPortedError

__all__ = ["Mesh", "MeshSource", "TubULAR", "MethodNotPortedError", "__version__"]
