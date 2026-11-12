"""Headless demo: known tube chart -> volume sampling -> saved pullback.

Run: python examples/synthetic_pullback.py /tmp/pytubular-demo
Creates a new directory; never overwrites previous results.
"""

import json
from pathlib import Path
import sys

import numpy as np

from tubular.analysis import analyze_mesh
from tubular.grid import cylinder_mesh
from tubular.imaging import pullback_image
from tubular.io import save_mesh


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    mesh = cylinder_mesh(17, 65, radius=6, length=24, length_unit="um")
    mesh = mesh.with_vertices(mesh.vertices + [4, 12, 12])
    z, y, x = np.indices((25, 25, 33))
    volume = 100 + 40 * np.sin(x * np.pi / 6) + 2 * y + z
    result = pullback_image(mesh, volume, np.linspace(0, 24, 97), np.linspace(0, 2*np.pi, 129))
    save_mesh(output / "mesh.npz", mesh)
    np.savez_compressed(output / "pullback.npz", image=result.image, xyz=result.xyz,
                        valid=result.valid, face_indices=result.face_indices)
    (output / "report.json").write_text(json.dumps(analyze_mesh(mesh), indent=2) + "\n")
    print(f"Saved mesh, pullback, and report to {output}")


if __name__ == "__main__":
    run(sys.argv[1])
