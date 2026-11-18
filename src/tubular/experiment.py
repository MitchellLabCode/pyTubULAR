"""Stateful TubULAR experiment interface over the independently tested core."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import re

import numpy as np

from ._coordinates import CoordinateMethods, boolean, points, rotation, scalar
from .geometry import fundamental_forms, signed_volume, surface_area
from .grid import _size
from .imaging import pullback_image
from .io import load_mat_mesh, load_mesh
from .mesh import Mesh, real_array, topology


MESH_KINDS = (
    "rawMesh", "alignedMesh", "cylinderMesh", "cylinderMeshClean", "cutMesh",
    "uvcutMesh", "spcutMesh", "spcutMeshSm", "spcutMeshSmRS", "spcutMeshSmRSC", "ricciMesh",
)
_BASE_KEYS = {"rawMesh": "mesh", **{kind: kind for kind in MESH_KINDS if kind != "rawMesh"}}


@dataclass(frozen=True)
class MeshSource:
    """A file-backed mesh with explicit MATLAB variable and coordinate unit.

    variable=None means top-level v/f/u in a MAT file. NPZ carries its own
    units; length_unit must stay 'unknown' or agree with those stored units.
    """
    path: object
    variable: str = None
    length_unit: str = "unknown"


def _namespace(mapping):
    return SimpleNamespace(**{k: _namespace(v) if isinstance(v, Mapping) else deepcopy(v)
                              for k, v in mapping.items()})


def _options(options, allowed, method):
    if options is None:
        return {}
    if not isinstance(options, Mapping):
        raise TypeError(f"{method} options must be a mapping")
    unknown = set(options) - set(allowed)
    if unknown:
        raise NotImplementedError(f"{method} does not yet support options: {', '.join(sorted(unknown))}")
    return dict(options)


class TubULAR(CoordinateMethods):
    """A Python TubULAR object with MATLAB-style method names.

    Accepts explicit acquisition time_points and an options mapping. No
    ImSAnE experiment object or simulated xp metadata is used. No filesystem
    writes occur on construction. Register mesh/data sources explicitly.
    Use method_status() to distinguish implemented methods from pending ports.
    """

    def __init__(self, *, time_points=None, opts=None, mesh_sources=None, data_sources=None):
        times = [0] if time_points is None else time_points
        times = real_array(times, "timePoints", ndim=1)
        if len(times) == 0 or np.any(np.diff(times) <= 0):
            raise ValueError("timePoints must be nonempty, unique, and strictly increasing")
        times.setflags(write=False)
        self._time_points = times
        options = _options(opts, {"meshDir", "timeUnits", "spaceUnits", "nU", "nV", "flipy",
                                "timeInterval", "timeinterval", "t0", "resolution", "rot", "trans",
                                "fullFileBase", "fileName", "makeDirs", "normalShift"}, "TubULAR")
        if options.get("makeDirs", False):
            raise NotImplementedError("automatic MATLAB directory-tree creation is not implemented")
        if "timeInterval" in options and "timeinterval" in options:
            raise ValueError("supply only one spelling of timeInterval")
        self.timeInterval = options.get("timeInterval", options.get("timeinterval", 1))
        self.timeUnits = options.get("timeUnits", "min")
        self.spaceUnits = options.get("spaceUnits", "um")
        for name in ("timeUnits", "spaceUnits"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a nonempty string")
        self.nU = _size(options.get("nU", 100), "nU")
        self.nV = _size(options.get("nV", 100), "nV")
        self.flipy = boolean(options.get("flipy", False), "flipy")
        self.normalShift = scalar(options.get("normalShift", 0), "normalShift")
        self.plotting = SimpleNamespace(xyzlim_raw=None, xyzlim_pix=None,
                                        xyzlim_um=None, xyzlim_um_buff=None)
        self.piv = SimpleNamespace(imCoords="sp_sme")
        self.uvexten = f"_nU{self.nU:04d}_nV{self.nV:04d}"
        self.dir = SimpleNamespace(mesh=Path(options.get("meshDir", ".")))
        for name in ("fileName", "fullFileBase"):
            if not isinstance(options.get(name, {}), Mapping):
                raise TypeError(f"{name} must be a mapping")
        self.fileName = _namespace(options.get("fileName", {}))
        self.fullFileBase = _namespace(options.get("fullFileBase", {}))
        if not hasattr(self.fileName, "t0"):
            self.fileName.t0 = self.dir.mesh / "t0_referenceTime.txt"
        resolution = scalar(options.get("resolution", 1), "resolution", positive=True)
        self.APDV = SimpleNamespace(resolution=resolution, rot=None, trans=None)
        if "rot" in options or "trans" in options:
            if "rot" not in options or "trans" not in options:
                raise ValueError("provide both rot and trans")
            self.set_apdv_frame(options["rot"], options["trans"], resolution=resolution)
        if "t0" in options:
            self.t0 = options["t0"]
        elif Path(self.fileName.t0).is_file():
            self.t0 = np.loadtxt(self.fileName.t0, comments="#", delimiter=",")
        else:
            self.t0 = times[0]
        self._mesh_sources = {kind: {} for kind in MESH_KINDS}
        self._data_sources = {}
        self._centerline_sources = {}
        self.data = SimpleNamespace(adjustlow=0, adjusthigh=100, axisOrder="zyx")
        self.apdvOptions = {}
        self.endcapOptions = {}
        self.clearTime()
        if mesh_sources is not None:
            if not isinstance(mesh_sources, Mapping):
                raise TypeError("mesh_sources must map mesh kinds to time/source mappings")
            for kind, series in mesh_sources.items():
                if not isinstance(series, Mapping):
                    raise TypeError("each mesh series must be a time/source mapping")
                for time, source in series.items():
                    self.register_mesh(time, source, kind=kind)
        if data_sources is not None:
            if not isinstance(data_sources, Mapping):
                raise TypeError("data_sources must map timepoints to arrays or .npy paths")
            for time, source in data_sources.items():
                self.register_data(time, source)

    @property
    def timePoints(self):
        return self._time_points

    @property
    def dynamic(self):
        return len(self._time_points) > 1

    @property
    def timeInterval(self):
        return self._time_interval

    @timeInterval.setter
    def timeInterval(self, value):
        self._time_interval = scalar(value, "timeInterval", positive=True)

    @property
    def t0(self):
        return self._t0

    @t0.setter
    def t0(self, value):
        self._t0 = scalar(value, "t0")

    @property
    def currentTime(self):
        return self._current_time

    @currentTime.setter
    def currentTime(self, value):
        if value is None:
            self.clearTime()
        else:
            self.setTime(value)

    def _time(self, value):
        value = scalar(value, "timepoint")
        if value not in self._time_points:
            raise ValueError(f"timepoint {value:g} is not in timePoints")
        return value

    def _require_time(self):
        if self.currentTime is None:
            raise ValueError("first select a timepoint with setTime(timepoint)")
        return self.currentTime

    def setTime(self, tt):
        """Select an acquired timepoint; a change invalidates every current cache."""
        tt = self._time(tt)
        if tt != self.currentTime:
            self.clearTime()
            self._current_time = tt

    def clearTime(self):
        self._current_time = None
        self.currentMesh = SimpleNamespace(**{kind: None for kind in MESH_KINDS}, cutPath=None)
        self.currentData = SimpleNamespace(IV=None, adjustlow=0, adjusthigh=100)
        self.currentVelocity = SimpleNamespace(piv3d={}, average={})
        self.currentSegmentation = SimpleNamespace(seg2d={}, seg2dCorrected={}, seg3d={}, seg3dCorrected={})
        self.currentStrain = SimpleNamespace(pathline={})
        self.currentCline = SimpleNamespace(mss=None, mcline=None, avgpts=None)

    def t0set(self, t0=None):
        """Read/set the reference time; a getter never resets an explicit t0."""
        if t0 is not None:
            self.t0 = t0
        return self.t0

    def trueTime(self, tp, div60=False, *, return_label=False):
        """Physical time relative to t0: (tp-t0)*timeInterval, not frame position."""
        a = np.asarray(tp)
        time = (real_array(a, "tp", ndim=a.ndim) - self.t0) * self.timeInterval
        if boolean(div60, "div60"):
            time = time / 60
        result = float(time) if time.ndim == 0 else time
        if not boolean(return_label, "return_label"):
            return result
        if time.ndim != 0:
            raise ValueError("return_label is supported for a scalar timepoint")
        unit = self.timeUnits
        if div60:
            if unit in ("min", "minute", "minutes"):
                unit = "hr"
            elif unit in ("s", "sec", "second", "seconds"):
                unit = "min"
            else:
                raise ValueError("cannot label div60 for this time unit")
        return result, f"{result:g} {unit}"

    def set_apdv_frame(self, rot, trans, *, resolution=None, flipy=None):
        """Configure an explicit frame; does not estimate anatomical landmarks."""
        r = rotation(rot)
        t = real_array(trans, "trans", ndim=1)
        if t.shape != (3,):
            raise ValueError("trans must contain three coordinates")
        scale = scalar(self.APDV.resolution if resolution is None else resolution, "resolution", positive=True)
        flip = self.flipy if flipy is None else boolean(flipy, "flipy")
        self.APDV = SimpleNamespace(rot=r, trans=t, resolution=scale)
        self.flipy = flip

    def getRotTrans(self):
        """Return validated rotation/translation, loading explicit text paths if set."""
        rot, trans = self.APDV.rot, self.APDV.trans
        if rot is None and hasattr(self.fileName, "rot"):
            rot = np.loadtxt(self.fileName.rot)
        if trans is None and hasattr(self.fileName, "trans"):
            trans = np.loadtxt(self.fileName.trans)
        if rot is None or trans is None:
            raise ValueError("APDV frame is unset; call set_apdv_frame(rot, trans, resolution=...)")
        self.set_apdv_frame(rot, trans)
        return self.APDV.rot.copy(), self.APDV.trans.copy()

    def computeAPDVCoords(self, anterior, posterior, dorsal, *, origin=None,
                          resolution=None, flipy=None):
        """Set an APDV frame from supplied XYZ landmarks; no image processing.

        All points use the raw mesh coordinates. Origin defaults to anterior;
        supplying it corresponds to an explicitly selected start point. Returns
        (anterior, posterior, dorsal, rotation, translation), all copied. Scale
        and reflection keep their configured values unless explicitly supplied.
        """
        from .alignment import landmark_frame
        rot, trans = landmark_frame(anterior, posterior, dorsal, origin=origin)
        self.set_apdv_frame(rot, trans, resolution=resolution, flipy=flipy)
        return (np.array(anterior, dtype=float, copy=True),
                np.array(posterior, dtype=float, copy=True),
                np.array(dorsal, dtype=float, copy=True), rot.copy(), trans.copy())

    def alignMeshesAPDV(self, *, time_points=None, register=False, preserve_orientation=True):
        """Align registered raw meshes using one explicitly configured APDV frame.

        Returns meshes and transform metadata in a dictionary. By default this
        is a preview: registration is explicit and happens only after the entire
        series succeeds. register=True replaces alignedMesh sources for those
        times and invalidates the selected aligned cache. No disk writes,
        endpoint adjustment, smoothing, segmentation, or rendering occurs.
        """
        from .alignment import align_mesh
        times = self._series_times(self.timePoints if time_points is None else time_points)
        register = boolean(register, "register")
        preserve = boolean(preserve_orientation, "preserve_orientation")
        rot, trans = self.getRotTrans()
        scale, flip = self.APDV.resolution, self.flipy
        meshes, units = [], set()
        for time in times:
            mesh = self._read_mesh("rawMesh", time)
            units.add(mesh.length_unit)
            meshes.append(align_mesh(mesh, rot, trans, resolution=scale, flipy=flip,
                                     preserve_orientation=preserve, length_unit=self.spaceUnits))
        if len(units) != 1:
            raise ValueError("raw mesh units differ across timepoints")
        result = {"timepoints": times, "times": self.trueTime(times), "time_unit": self.timeUnits,
                  "meshes": tuple(meshes), "rot": rot, "trans": trans, "resolution": scale,
                  "flipy": flip, "preserve_orientation": preserve,
                  "face_corner_order": np.array([0, 2, 1] if flip and preserve else [0, 1, 2]),
                  "source_length_unit": next(iter(units)), "length_unit": self.spaceUnits}
        if register:
            for time, mesh in zip(times, meshes):
                self.register_mesh(time, mesh, kind="alignedMesh")
        return result

    def xyz2APDV(self, a):
        rot, trans = self.getRotTrans()
        result = (points(a, "a", 3) @ rot.T + trans) * self.APDV.resolution
        if self.flipy:
            result[:, 1] *= -1
        return result

    def APDV2xyz(self, a):
        rot, trans = self.getRotTrans()
        a = points(a, "a", 3)
        if self.flipy:
            a[:, 1] *= -1
        return (a / self.APDV.resolution - trans) @ rot

    def dx2APDV(self, da):
        rot, _ = self.getRotTrans()
        result = (points(da, "da", 3) @ rot.T) * self.APDV.resolution
        if self.flipy:
            result[:, 1] *= -1
        return result

    def APDV2dxyz(self, a):
        rot, _ = self.getRotTrans()
        a = points(a, "a", 3)
        if self.flipy:
            a[:, 1] *= -1
        return (a / self.APDV.resolution) @ rot

    def setAPDVOptions(self, apdvOpts):
        if not isinstance(apdvOpts, Mapping):
            raise TypeError("APDV options must be a mapping")
        self.apdvOptions = deepcopy(dict(apdvOpts))

    def setEndcapOptions(self, endcapOpts):
        if not isinstance(endcapOpts, Mapping):
            raise TypeError("endcap options must be a mapping")
        self.endcapOptions = deepcopy(dict(endcapOpts))

    def register_mesh(self, timepoint, source, *, kind="rawMesh"):
        """Register a Mesh, MeshSource, or .npz/.mat path; no generation occurs."""
        timepoint = self._time(timepoint)
        if kind not in MESH_KINDS:
            raise ValueError(f"unknown mesh kind {kind!r}")
        if not isinstance(source, (Mesh, MeshSource, str, Path)):
            raise TypeError("source must be a Mesh, MeshSource, or path")
        self._mesh_sources[kind][timepoint] = source
        if timepoint == self.currentTime:
            setattr(self.currentMesh, kind, None)

    @staticmethod
    def _format_path(pattern, timepoint):
        pattern = str(pattern)
        if "{" in pattern:
            t = int(timepoint) if timepoint.is_integer() else timepoint
            return Path(pattern.format(time=t, timepoint=t))
        if re.search(r"%(?:0\d+)?d", pattern):
            if not timepoint.is_integer():
                raise ValueError("an integer filename pattern cannot represent this timepoint")
            return Path(pattern % int(timepoint))
        return Path(pattern)

    def _read_mesh(self, kind, timepoint):
        source = self._mesh_sources[kind].get(timepoint)
        if source is None:
            base = _BASE_KEYS[kind]
            if not hasattr(self.fullFileBase, base):
                raise FileNotFoundError(f"no {kind} source for time {timepoint:g}; use register_mesh")
            path = self._format_path(getattr(self.fullFileBase, base), timepoint)
            # These variable names follow the legacy named-struct loaders.
            variable = kind if path.suffix.lower() == ".mat" and kind in MESH_KINDS[4:] else None
            source = MeshSource(path, variable=variable)
        if isinstance(source, Mesh):
            return source
        if not isinstance(source, MeshSource):
            source = MeshSource(source)
        path = Path(source.path)
        if path.suffix.lower() == ".mat":
            return load_mat_mesh(path, variable=source.variable, length_unit=source.length_unit)
        if path.suffix.lower() == ".npz":
            result = load_mesh(path)
            if source.variable is not None:
                raise ValueError("MATLAB variable selection does not apply to NPZ")
            if source.length_unit != "unknown" and source.length_unit != result.length_unit:
                raise ValueError("MeshSource units disagree with the NPZ mesh metadata")
            return result
        raise NotImplementedError("mesh sources currently support .mat and .npz; PLY/OBJ are not ported")

    def get_mesh(self, kind="rawMesh", *, reload=False):
        timepoint = self._require_time()
        if kind not in MESH_KINDS:
            raise ValueError(f"unknown mesh kind {kind!r}")
        result = getattr(self.currentMesh, kind)
        if result is None or reload:
            result = self._read_mesh(kind, timepoint)
            setattr(self.currentMesh, kind, result)
        if not isinstance(result, Mesh):
            raise TypeError("currentMesh entries must be validated Mesh objects")
        return result

    def register_data(self, timepoint, source):
        """Register one ZYX array/.npy file, or a sequence of channels; no resampling."""
        timepoint = self._time(timepoint)
        channels = source if isinstance(source, (tuple, list)) else (source,)
        if len(channels) == 0:
            raise ValueError("data needs at least one channel")
        checked = []
        for channel in channels:
            if isinstance(channel, (str, Path)):
                if Path(channel).suffix.lower() != ".npy":
                    raise NotImplementedError("file-backed volume data currently requires .npy")
                checked.append(Path(channel))
            else:
                checked.append(self._volume(channel))
        self._data_sources[timepoint] = tuple(checked)
        if timepoint == self.currentTime:
            self.currentData = SimpleNamespace(IV=None, adjustlow=0, adjusthigh=100)

    @staticmethod
    def _volume(value):
        a = np.asarray(value)
        if a.ndim != 3 or a.dtype.kind not in "iuf" or min(a.shape) < 1:
            raise ValueError("each channel must be a nonempty real (z, y, x) array")
        a = np.array(a, copy=True)
        a.setflags(write=False)
        return a

    def getCurrentData(self, adjustIV=True):
        """Return a tuple of unresampled ZYX channels; nontrivial LUTs are pending."""
        timepoint = self._require_time()
        if boolean(adjustIV, "adjustIV") and (not np.array_equal(self.data.adjustlow, 0)
                                               or not np.array_equal(self.data.adjusthigh, 100)):
            raise NotImplementedError("intensity adjustment has not been validated; use adjustIV=False")
        if not isinstance(self.data.axisOrder, str) or self.data.axisOrder != "zyx":
            raise ValueError("registered volumes must use explicit ZYX order")
        if self.currentData.IV is None:
            if timepoint not in self._data_sources:
                raise FileNotFoundError(f"no volume data for time {timepoint:g}; use register_data")
            result = tuple(self._volume(np.load(s, allow_pickle=False)) if isinstance(s, Path)
                           else s for s in self._data_sources[timepoint])
            if any(a.shape != result[0].shape for a in result):
                raise ValueError("all channels at a timepoint must have the same shape")
            self.currentData.IV = result
        return self.currentData.IV

    def loadCurrentData(self, adjustIV=True):
        self._require_time()
        self.currentData.IV = None
        return self.getCurrentData(adjustIV)

    def mip(self, tp, dim, pages=None, adjustIV=False, *, channel=0):
        """Maximum projection; MATLAB dim/pages are one-based, channel is zero-based.

        dim=1 projects Z, dim=2 Y, dim=3 X because registered volumes are ZYX.
        """
        axis = _size(dim, "dim", 1) - 1
        if axis > 2:
            raise ValueError("dim must be 1, 2, or 3")
        channel = _size(channel, "channel", 0)
        self.setTime(tp)
        data = self.getCurrentData(adjustIV)
        if channel >= len(data):
            raise ValueError("channel is out of range")
        a = data[channel]
        if pages is not None:
            ids = np.asarray(pages)
            if (ids.ndim != 1 or ids.dtype.kind not in "iu" or len(ids) == 0
                    or np.any(ids < 1) or np.any(ids > a.shape[axis])):
                raise ValueError("pages must be valid one-based integer slice indices")
            a = np.take(a, ids - 1, axis=axis)
        return a.max(axis=axis)

    def measureSurfaceAreaVolume(self, options=None):
        """Return unsmoothed area/volume series; no legacy filtering, plotting, or saves.

        Supported options: mesh_kind (default alignedMesh), timePoints,
        allow_open (default False; True reports NaN volume for open meshes).
        This is an adapted numerical entry point, not the full MATLAB workflow.
        """
        opts = _options(options, {"mesh_kind", "timePoints", "allow_open"}, "measureSurfaceAreaVolume")
        kind = opts.get("mesh_kind", "alignedMesh")
        if kind not in MESH_KINDS:
            raise ValueError(f"unknown mesh kind {kind!r}")
        times = real_array(opts.get("timePoints", self.timePoints), "timePoints", ndim=1)
        if len(times) == 0 or np.any(np.diff(times) <= 0):
            raise ValueError("measurement timePoints must be nonempty and strictly increasing")
        times = np.array([self._time(t) for t in times])
        allow_open = boolean(opts.get("allow_open", False), "allow_open")
        areas, volumes, units = [], [], set()
        for timepoint in times:
            mesh = self._read_mesh(kind, timepoint)
            units.add(mesh.length_unit)
            info = topology(mesh)
            if not info.manifold or not info.oriented:
                raise ValueError("area/volume series requires consistently oriented manifold meshes")
            areas.append(surface_area(mesh))
            if allow_open and not info.closed:
                volumes.append(np.nan)
            else:
                volumes.append(signed_volume(mesh))
        if len(units) != 1:
            raise ValueError("mesh units differ across timepoints")
        signed = np.asarray(volumes)
        return {"timepoints": times, "times": self.trueTime(times), "time_unit": self.timeUnits,
                "area": np.asarray(areas), "volume": np.abs(signed), "signed_volume": signed,
                "length_unit": next(iter(units)), "mesh_kind": kind}

    def measure_current_geometry(self, *, mesh_kind="spcutMeshSm"):
        """Native Python access to first/second forms of the current chart."""
        return fundamental_forms(self.get_mesh(mesh_kind))

    def measureCurvatures(self, *, mesh_kind="spcutMeshSmRSC", time_points=None):
        """Return vertex curvature estimates for a supplied mesh series.

        Uses the same estimator and mean-curvature sign as MATLAB's
        GetCurvatures-based measurement. Open boundaries are flagged, not
        glued or capped. No smoothing, plotting, saves, or selection changes.
        """
        from .curvature import estimate_curvature
        if mesh_kind not in MESH_KINDS:
            raise ValueError(f"unknown mesh kind {mesh_kind!r}")
        times = self._series_times(self.timePoints if time_points is None else time_points)
        results, units = [], set()
        for time in times:
            mesh = self._read_mesh(mesh_kind, time)
            units.add(mesh.length_unit)
            results.append(estimate_curvature(mesh))
        if len(units) != 1:
            raise ValueError("mesh units differ across timepoints")
        return {"timepoints": times, "times": self.trueTime(times), "time_unit": self.timeUnits,
                "curvatures": tuple(results), "mesh_kind": mesh_kind,
                "length_unit": next(iter(units))}

    def _beltrami_series(self, reference, mapped_coordinates, time_points, dimension):
        from .quasiconformal import beltrami_coefficient
        if not isinstance(mapped_coordinates, Mapping):
            raise TypeError("supply a timepoint-to-coordinate mapping in reference vertex order")
        if not isinstance(reference, Mesh):
            raise TypeError("reference must be a Mesh")
        times = self._series_times(self.timePoints if time_points is None else time_points)
        baseline = beltrami_coefficient(reference, reference.uv if dimension == 2 else None)
        results = []
        for time in times:
            if time not in mapped_coordinates:
                raise KeyError(f"no mapped coordinates for time {time:g}")
            target = real_array(mapped_coordinates[time], "mapped_coordinates", ndim=2)
            if target.shape != (len(reference.v), dimension):
                raise ValueError(f"coordinates must have shape (N, {dimension}) in reference vertex order")
            results.append(beltrami_coefficient(reference, target))
        return {"timepoints": times, "times": self.trueTime(times), "time_unit": self.timeUnits,
                "mu": np.stack([r.mu for r in results]),
                "dilatation": np.stack([r.dilatation for r in results]),
                "reference_mu": baseline.mu, "target_dimension": dimension}

    def measureBeltramiCoefficient(self, reference, vertex_positions, *, time_points=None):
        """Measure supplied 3D vertex trajectories against a fixed reference UV chart.

        vertex_positions maps acquired times to (N, 3) coordinates with explicit
        reference vertex correspondence. Returns raw per-face mu/dilatation and
        reference_mu separately. No reference offset, tracking, Ricci flow,
        filtering, plots, file writes, or current-state changes are performed.
        """
        return self._beltrami_series(reference, vertex_positions, time_points, 3)

    def measureBeltramiCoefficientPullbackToPullback(self, reference, target_charts, *, time_points=None):
        """Measure supplied chart-to-chart maps on fixed vertex correspondence.

        target_charts maps acquired times to (N, 2) UV arrays in reference vertex
        order. Every target triangle must preserve orientation relative to the
        reference chart. Charts are supplied, not constructed or made conformal.
        """
        return self._beltrami_series(reference, target_charts, time_points, 2)

    def uv2APDV(self, uv, coordSys="spsmrs", umax=1, vmax=1, *, return_details=False):
        """Map chart points to the supplied APDV mesh using its original triangles.

        Only spsmrs is supported: register an already aligned spcutMeshSmRS.
        U is normalized by max(U) as in MATLAB. V is not rescaled; vmax must
        be 1 because the original ignores that argument. No periodic wrapping.
        With return_details, return (xyz, face_indices, chart_mesh, barycentric).
        Outside points have NaN coordinates/weights and face index -1.
        """
        from .surface import locate_chart
        if coordSys is None or coordSys == "":
            coordSys = "spsmrs"
        if not isinstance(coordSys, str) or coordSys.lower() != "spsmrs":
            raise NotImplementedError("uv2APDV currently supports only spsmrs; sprs requires a separate sphi field")
        umax = scalar(umax, "umax", positive=True)
        if scalar(vmax, "vmax", positive=True) != 1:
            raise NotImplementedError("MATLAB ignores vmax here; only vmax=1 is supported")
        details = boolean(return_details, "return_details")
        mesh = self.getCurrentSPCutMeshSmRS()
        if mesh.u is None or mesh.u[:, 0].max() <= 0:
            raise ValueError("uv2APDV requires a UV chart with positive max(U)")
        chart = mesh.u.copy()
        chart[:, 0] *= umax / chart[:, 0].max()
        chart_mesh = Mesh(mesh.v, mesh.f, chart, mesh.length_unit)
        located = locate_chart(chart_mesh, uv)
        xyz = located.interpolate(mesh.v)
        if details:
            return xyz, located.face_indices, chart_mesh, located.barycentric
        return xyz

    def interpolateOntoPullbackXY(self, XY, scalar_field, options=None):
        """Interpolate a rectilinear sp_sme vertex field on the legacy tiled grid.

        Requires explicit Lx (width), Ly (height), and nU*nV values, U fastest.
        Linear interpolation is bilinear, as in the original griddedInterpolant;
        nearest extrapolation applies outside the entire three-tile grid.
        Exact midpoint ties select the upper grid node, matching MATLAB.
        This grid operation is distinct from triangle-based surface.locate_chart.
        """
        from scipy.interpolate import RegularGridInterpolator
        opts = _options(options, {"imCoords", "sfLocation", "iMethod", "Lx", "Ly"},
                        "interpolateOntoPullbackXY")
        if opts.get("imCoords", self.piv.imCoords) != "sp_sme":
            raise NotImplementedError("only the sp_sme double-cover grid is supported")
        if opts.get("sfLocation", "vertices") != "vertices":
            raise NotImplementedError("the legacy face-field interpolation branch is incomplete")
        method = opts.get("iMethod", "linear")
        if method not in ("linear", "nearest"):
            raise NotImplementedError("only linear and nearest interpolation are supported")
        if "Lx" not in opts or "Ly" not in opts:
            raise ValueError("supply Lx (width) and Ly (height) explicitly")
        Lx, Ly = scalar(opts["Lx"], "Lx", positive=True), scalar(opts["Ly"], "Ly", positive=True)
        if Lx <= 1 or Ly <= 1:
            raise ValueError("Lx and Ly must exceed one")
        xy = points(XY, "XY", 2)
        field = np.asarray(scalar_field)
        if field.ndim not in (1, 2):
            raise ValueError("scalar_field must be a vector or (nU, nV) grid")
        field = real_array(field, "scalar_field", ndim=field.ndim)
        if field.ndim == 2 and field.shape not in ((self.nU, self.nV), (self.nU*self.nV, 1)):
            raise ValueError("scalar_field grid must have shape (nU, nV), or be a column vector")
        if field.size != self.nU*self.nV:
            raise ValueError("scalar_field must contain nU*nV values")
        field = field.reshape((self.nU, self.nV), order="F")
        tiled = np.concatenate((field[:, :-1], field, field[:, 1:]), axis=1)
        x = np.linspace(1, Lx, self.nU)
        y = np.linspace(1, 1.5*Ly, 3*self.nV-2) - 0.25*Ly
        def nearest_values(query):
            # SciPy's nearest mode chooses the lower node at ties, whereas
            # MATLAB chooses the upper node. Use explicit midpoint bins.
            i = np.searchsorted(x[:-1] + np.diff(x)/2, query[:, 0], side="right")
            j = np.searchsorted(y[:-1] + np.diff(y)/2, query[:, 1], side="right")
            return tiled[i, j]

        if method == "nearest":
            return nearest_values(xy)
        interpolator = RegularGridInterpolator((x, y), tiled, method="linear",
                                               bounds_error=False, fill_value=np.nan)
        result = interpolator(xy)
        outside = ((xy[:, 0] < x[0]) | (xy[:, 0] > x[-1]) |
                   (xy[:, 1] < y[0]) | (xy[:, 1] > y[-1]))
        if outside.any():
            result[outside] = nearest_values(xy[outside])
        return result

    def measureXYZLims(self, options=None):
        """Return (raw, aligned_pixels, physical, buffered) 3-by-2 bounds.

        Reads registered raw meshes across time; no writes or time selection
        changes. Physical padding is 2*abs(normalShift)*APDV.resolution.
        Supported options: timePoints, normalShift. Legacy output files/plots
        are not produced. The physical unit is spaceUnits, not necessarily um.
        """
        opts = _options(options, {"timePoints", "normalShift"}, "measureXYZLims")
        times = real_array(opts.get("timePoints", self.timePoints), "timePoints", ndim=1)
        if len(times) == 0 or np.any(np.diff(times) <= 0):
            raise ValueError("timePoints must be nonempty and strictly increasing")
        times = [self._time(t) for t in times]
        shift = scalar(opts.get("normalShift", self.normalShift), "normalShift")
        self.getRotTrans()  # Validate the frame before reading mesh sources.
        lower, upper = np.full(3, np.inf), np.full(3, -np.inf)
        apdv_lower, apdv_upper = lower.copy(), upper.copy()
        units = set()
        for time in times:
            mesh = self._read_mesh("rawMesh", time)
            units.add(mesh.length_unit)
            lower, upper = np.minimum(lower, mesh.v.min(axis=0)), np.maximum(upper, mesh.v.max(axis=0))
            aligned = self.xyz2APDV(mesh.v)
            apdv_lower = np.minimum(apdv_lower, aligned.min(axis=0))
            apdv_upper = np.maximum(apdv_upper, aligned.max(axis=0))
        if len(units) != 1:
            raise ValueError("raw mesh units differ across timepoints")
        raw = np.column_stack((lower, upper))
        physical = np.column_stack((apdv_lower, apdv_upper))
        pixels = physical / self.APDV.resolution
        buffered = physical + 2*abs(shift)*self.APDV.resolution*np.array([-1, 1])
        result = (raw, pixels, physical, buffered)
        for key, array in zip(("xyzlim_raw", "xyzlim_pix", "xyzlim_um", "xyzlim_um_buff"), result):
            setattr(self.plotting, key, array.copy())
        return result

    def getXYZLims(self):
        """Recompute bounds from current sources/frame, avoiding stale file caches."""
        return self.measureXYZLims()

    def getCurrentClineDVhoop(self):
        """Load mss, mcline, avgpts from an explicit MAT filename pattern.

        The stored curve is in physical coordinates before optional Y reflection.
        Reloads each call as MATLAB does; no additional rotation/scale or fitting.
        """
        time = self._require_time()
        result = self._read_centerline(time, flipy=self.flipy)
        for key, array in zip(("mss", "mcline", "avgpts"), result):
            cached = array.copy()
            cached.setflags(write=False)
            setattr(self.currentCline, key, cached)
        return result

    @staticmethod
    def _validate_centerline(mss, cline, average):
        mss = real_array(mss, "mss", ndim=1)
        cline, average = points(cline, "mcline", 3), points(average, "avgpts", 3)
        if len(mss) < 2 or len(cline) != len(mss) or len(average) < 2 or np.any(np.diff(mss) <= 0):
            raise ValueError("centerline needs matching mss/mcline lengths, increasing mss, and at least two points")
        return mss, cline, average

    def register_centerline(self, timepoint, mcline, *, mss=None, avgpts=None):
        """Register physical centerline coordinates before optional Y reflection.

        Omitted mss is computed as polygonal arc length; avgpts defaults to
        mcline. Copies inputs; registered data take precedence over MAT files.
        """
        from .curves import curve_arclength
        timepoint = self._time(timepoint)
        if mss is None:
            mss = curve_arclength(mcline)
        if avgpts is None:
            avgpts = mcline
        result = self._validate_centerline(mss, mcline, avgpts)
        for array in result:
            array.setflags(write=False)
        self._centerline_sources[timepoint] = result
        if timepoint == self.currentTime:
            self.currentCline = SimpleNamespace(mss=None, mcline=None, avgpts=None)

    def _read_centerline(self, time, *, flipy=False):
        from scipy.io import loadmat
        if time in self._centerline_sources:
            result = tuple(a.copy() for a in self._centerline_sources[time])
            if flipy:
                result[1][:, 1] *= -1
                result[2][:, 1] *= -1
            return result
        if not hasattr(self.fullFileBase, "clineDVhoop"):
            raise FileNotFoundError("register a centerline or supply fullFileBase.clineDVhoop for existing MAT files")
        path = self._format_path(self.fullFileBase.clineDVhoop, time)
        data = loadmat(path, variable_names=["mss", "mcline", "avgpts"])
        if not {"mss", "mcline", "avgpts"}.issubset(data):
            raise ValueError("centerline MAT file must contain mss, mcline, and avgpts")
        mss = np.asarray(data["mss"])
        if mss.ndim != 2 or 1 not in mss.shape:
            raise ValueError("mss must be a MATLAB row or column vector")
        mss, cline, average = self._validate_centerline(mss.ravel(), data["mcline"], data["avgpts"])
        if flipy:
            cline[:, 1] *= -1
            average[:, 1] *= -1
        return mss, cline, average

    def _series_times(self, values):
        times = real_array(values, "timePoints", ndim=1)
        if len(times) == 0 or np.any(np.diff(times) <= 0):
            raise ValueError("timePoints must be nonempty and strictly increasing")
        return np.array([self._time(t) for t in times])

    @staticmethod
    def _time_derivative(values, times):
        return np.gradient(values, times) if len(times) > 1 else np.full_like(values, np.nan)

    def _length_result(self, times, lengths, geometric):
        lengths, physical = np.asarray(lengths), self.trueTime(times)
        return {"lengths": lengths, "polygonal_length": np.asarray(geometric),
                "dl": self._time_derivative(lengths, physical), "timepoints": times,
                "times": physical, "length_unit": self.spaceUnits, "time_unit": self.timeUnits}

    def measureLength(self, options=None):
        """Return stored centerline span, polygonal length, and physical-time rate.

        No writhe, filtering, trimming, plots, or output writes. length uses
        mss[-1]-mss[0]; polygonal_length independently sums geometric segments.
        Supported option: timePoints. The selected time and caches are preserved.
        """
        from .curves import curve_arclength
        opts = _options(options, {"timePoints"}, "measureLength")
        times = self._series_times(opts.get("timePoints", self.timePoints))
        lengths, geometric = [], []
        for time in times:
            mss, curve, _ = self._read_centerline(time)
            lengths.append(mss[-1]-mss[0])
            geometric.append(curve_arclength(curve)[-1])
        return self._length_result(times, lengths, geometric)

    def measureWrithe(self, options=None):
        """Return (Wr, Wr_density, dWr, Length_t, curves) for existing open curves.

        Levitt polygonal writhe and Gauss quadrature are supported. Polar writhe,
        curve/temporal smoothing, plots, and output writes remain unsupported.
        Options: timePoints, omit_endpts (default 4 at EACH end), filter_curve
        (only 0, the Python default), flipy_centerline (default False), and
        gauss_legacy (default False: correct the original normalization defect).
        dWr/dl are unsmoothed physical-time derivatives; a single time gives NaN.
        """
        from .curves import curve_arclength, writhe_levitt, writhe_gauss
        opts = _options(options, {"timePoints", "omit_endpts", "filter_curve", "flipy_centerline",
                                 "gauss_legacy"}, "measureWrithe")
        if scalar(opts.get("filter_curve", 0), "filter_curve") != 0:
            raise NotImplementedError("curve smoothing is not yet validated; use filter_curve=0")
        omit = np.asarray(opts.get("omit_endpts", 4))
        if omit.dtype.kind not in "iu" or omit.ndim > 1 or omit.size not in (1, 2) or np.any(omit < 0):
            raise ValueError("omit_endpts must be one or two nonnegative integers")
        omit = np.repeat(omit, 2) if omit.size == 1 else omit
        front, back = int(omit[0]), int(omit[1])
        flip = boolean(opts.get("flipy_centerline", False), "flipy_centerline")
        legacy = boolean(opts.get("gauss_legacy", False), "gauss_legacy")
        times = self._series_times(opts.get("timePoints", self.timePoints))
        wr, density = {"Levitt": [], "Gauss": []}, {"Levitt": [], "Gauss": []}
        curves = []
        lengths, geometric = [], []
        for time in times:
            mss, curve, _ = self._read_centerline(time, flipy=flip)
            lengths.append(mss[-1]-mss[0])
            geometric.append(curve_arclength(curve)[-1])
            if len(curve)-front-back < 3:
                raise ValueError("endpoint omission must leave at least three curve points")
            kept = np.arange(front, len(curve)-back)
            for name, calculation in (("Levitt", lambda: writhe_levitt(curve[kept])),
                                      ("Gauss", lambda: writhe_gauss(curve[kept], mss[kept], legacy=legacy))):
                total, local = calculation()
                wr[name].append(total)
                density[name].append(local)
            curves.append({"ssx": mss, "xyzp": curve, "kept_indices": kept})
        wr = {name: np.asarray(values) for name, values in wr.items()}
        density = {name: tuple(values) for name, values in density.items()}
        physical = self.trueTime(times)
        derivative = {name: self._time_derivative(values, physical) for name, values in wr.items()}
        length = self._length_result(times, lengths, geometric)
        length["gauss_convention"] = "matlab_legacy" if legacy else "standard_1_over_4pi"
        return wr, density, derivative, length, tuple(curves)

    def sample_current_pullback(self, u, v, *, mesh_kind="spcutMeshSm", channel=0,
                                spacing_xyz=(1, 1, 1), origin_xyz=(0, 0, 0),
                                progress=None, cancelled=None):
        """Sample a supplied chart and volume in the SAME coordinate frame.

        No implicit raw/APDV conversion, chart fitting, or normal-stack projection.
        """
        channel = _size(channel, "channel", 0)
        data = self.getCurrentData(False)
        if channel >= len(data):
            raise ValueError("channel is out of range")
        return pullback_image(self.get_mesh(mesh_kind), data[channel], u, v,
                              spacing_xyz=spacing_xyz, origin_xyz=origin_xyz,
                              progress=progress, cancelled=cancelled)


# Each getter/loader is an actual class method discoverable by dir/help. The
# factories keep their cache behavior identical across all eleven mesh stages.
def _mesh_method(name, kind, reload):
    def method(self):
        return self.get_mesh(kind, reload=reload)
    method.__name__ = name
    method.__qualname__ = f"TubULAR.{name}"
    method.__doc__ = f"{'Reload' if reload else 'Return/cache'} the current {kind} from registered sources."
    return method


for _kind, _stem in (("rawMesh", "RawMesh"), ("alignedMesh", "AlignedMesh"),
                     ("cylinderMesh", "CylinderMesh"), ("cylinderMeshClean", "CylinderMeshClean"),
                     ("cutMesh", "CutMesh"), ("uvcutMesh", "UVCutMesh"),
                     ("spcutMesh", "SPCutMesh"), ("spcutMeshSm", "SPCutMeshSm"),
                     ("spcutMeshSmRS", "SPCutMeshSmRS"), ("spcutMeshSmRSC", "SPCutMeshSmRSC"),
                     ("ricciMesh", "RicciMesh")):
    for _prefix, _reload in (("get", False), ("load", True)):
        _name = f"{_prefix}Current{_stem}"
        setattr(TubULAR, _name, _mesh_method(_name, _kind, _reload))

from ._method_catalog import install_matlab_interface  # noqa: E402

install_matlab_interface(TubULAR)
