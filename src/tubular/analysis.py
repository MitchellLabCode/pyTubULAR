"""Small application service shared by command-line and future GUI clients."""

from dataclasses import asdict

import numpy as np

from . import __version__
from .geometry import induced_metric, signed_volume, surface_area
from .mesh import Mesh, topology


def analyze_mesh(mesh: Mesh) -> dict:
    info = topology(mesh)
    result = {
        "schema_version": 1,
        "software_version": __version__,
        "length_unit": mesh.length_unit,
        "area": surface_area(mesh),
        "signed_volume": None,
        "topology": {**asdict(info), "euler_characteristic": info.euler_characteristic,
                     "closed": info.closed, "oriented": info.oriented},
        "warnings": [],
    }
    if info.closed and info.oriented:
        result["signed_volume"] = signed_volume(mesh)
        result["warnings"].append("Volume is algebraic; self-intersections and shell nesting are not checked.")
    else:
        result["warnings"].append("Volume unavailable: requires a closed, consistently oriented manifold.")
    if mesh.length_unit == "unknown":
        result["warnings"].append("Physical units were not supplied; area/volume use coordinate units.")
    if mesh.uv is not None:
        eigenvalues = np.linalg.eigvalsh(induced_metric(mesh))
        result["metric_eigenvalue_range"] = [float(eigenvalues.min()), float(eigenvalues.max())]
    return result
