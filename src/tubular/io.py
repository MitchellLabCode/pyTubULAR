"""Explicit MATLAB interchange and versioned, pickle-free NPZ snapshots."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat, savemat

from .mesh import Mesh


def load_mat_mesh(path, *, variable=None, length_unit="unknown") -> Mesh:
    """Read MATLAB v5-v7.2 fields v/f/u, optionally from a named scalar struct.

    MATLAB f is always interpreted as one-based; no indexing heuristics.
    v and u must be flat N-by-D arrays. Gridded arrays, transposed arrays,
    v7.3/HDF5, and arbitrary experiment objects need explicit future adapters.
    """
    data = loadmat(path, struct_as_record=False, squeeze_me=False,
                   variable_names=[variable] if variable else None)
    if variable is not None:
        if variable not in data:
            raise ValueError(f"MATLAB variable {variable!r} was not found")
        value = data[variable]
        if value.shape != (1, 1) or not hasattr(value[0, 0], "_fieldnames"):
            raise ValueError("the selected MATLAB variable must be a scalar struct")
        struct = value[0, 0]
        data = {k: getattr(struct, k) for k in struct._fieldnames}
    if "v" not in data or "f" not in data:
        raise ValueError("expected MATLAB fields v and f; select the mesh struct with variable=")
    f = np.asarray(data["f"])
    v = np.asarray(data["v"])
    if (f.dtype.kind not in "iuf" or not np.isfinite(f).all()
            or np.any(f < 1) or np.any(f > len(v)) or np.any(f != np.floor(f))):
        raise ValueError("MATLAB faces must contain valid one-based integer indices")
    return Mesh(v, f.astype(np.int64) - 1, data.get("u"), length_unit)


def save_mat_mesh(path, mesh: Mesh, *, variable="mesh") -> None:
    """Export a mesh struct using v/f/u, converting indices to MATLAB doubles."""
    if (not isinstance(variable, str) or not variable.isascii()
            or not variable.isidentifier() or not variable[0].isalpha() or len(variable) > 63):
        raise ValueError("variable must be an ASCII MATLAB identifier of at most 63 characters")
    data = {"v": mesh.vertices, "f": (mesh.faces + 1).astype(np.float64)}
    if mesh.uv is not None:
        data["u"] = mesh.uv
    with Path(path).open("xb") as stream:
        savemat(stream, {variable: data}, do_compression=True)


def save_mesh(path, mesh: Mesh) -> None:
    """Save a new NPZ snapshot. Refuses to overwrite an existing file."""
    metadata = json.dumps({"schema_version": 1, "index_base": 0,
                           "coordinate_order": "xyz", "length_unit": mesh.length_unit})
    arrays = {"vertices": mesh.vertices, "faces": mesh.faces, "metadata": metadata}
    if mesh.uv is not None:
        arrays["uv"] = mesh.uv
    with Path(path).open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def load_mesh(path) -> Mesh:
    """Load only the documented NPZ schema; pickle payloads are never allowed."""
    with np.load(path, allow_pickle=False) as archive:
        if not {"vertices", "faces", "metadata"}.issubset(archive.files):
            raise ValueError("not a TubULAR mesh snapshot")
        meta = json.loads(str(archive["metadata"].item()))
        if not isinstance(meta, dict) or meta.get("schema_version") != 1:
            raise ValueError("unsupported mesh schema version")
        if meta.get("index_base") != 0 or meta.get("coordinate_order") != "xyz":
            raise ValueError("unsupported index or coordinate convention")
        if "length_unit" not in meta:
            raise ValueError("mesh metadata must specify length_unit")
        return Mesh(archive["vertices"], archive["faces"],
                    archive["uv"] if "uv" in archive else None, meta["length_unit"])
