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

    Accepts a metadata mapping xp with fileMeta.timePoints, plus an options
    mapping; native callers may instead supply time_points. No filesystem
    writes occur on construction. Register mesh/data sources explicitly.
    Use method_status() to distinguish implemented methods from pending ports.
    """

    def __init__(self, xp=None, opts=None, *, time_points=None, mesh_sources=None, data_sources=None):
        if xp is not None and not isinstance(xp, Mapping):
            raise TypeError("xp must be a metadata mapping; ImSAnE objects are not supported")
        metadata = deepcopy(dict(xp or {}))
        file_meta = metadata.setdefault("fileMeta", {})
        if not isinstance(file_meta, Mapping):
            raise TypeError("xp.fileMeta must be a mapping")
        file_meta = metadata["fileMeta"] = dict(file_meta)
        if time_points is not None and "timePoints" in file_meta:
            raise ValueError("specify time_points or xp.fileMeta.timePoints, not both")
        times = time_points if time_points is not None else file_meta.get("timePoints", [0])
        times = real_array(times, "timePoints", ndim=1)
        if len(times) == 0 or np.any(np.diff(times) <= 0):
            raise ValueError("timePoints must be nonempty, unique, and strictly increasing")
        times.setflags(write=False)
        file_meta["timePoints"] = times
        metadata.setdefault("expMeta", {})
        self.xp = _namespace(metadata)
        self.xp.fileMeta.timePoints = times
        self.xp.tIdx = {float(tp): i for i, tp in enumerate(times)}
        self._time_points = times
        options = _options(opts, {"meshDir", "timeUnits", "spaceUnits", "nU", "nV", "flipy",
                                "timeInterval", "timeinterval", "t0", "resolution", "rot", "trans",
                                "fullFileBase", "fileName", "makeDirs"}, "TubULAR")
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
        self.uvexten = f"_nU{self.nU:04d}_nV{self.nV:04d}"
        self.dir = SimpleNamespace(mesh=Path(options.get("meshDir", ".")),
                                   data=Path(file_meta.get("dataDir", ".")))
        for name in ("fileName", "fullFileBase"):
            if not isinstance(options.get(name, {}), Mapping):
                raise TypeError(f"{name} must be a mapping")
        self.fileName = _namespace(options.get("fileName", {}))
        self.fullFileBase = _namespace(options.get("fullFileBase", {}))
        if not hasattr(self.fileName, "t0"):
            self.fileName.t0 = self.dir.mesh / "t0_referenceTime.txt"
        if "resolution" in options:
            resolution = scalar(options["resolution"], "resolution", positive=True)
        else:
            spacing = real_array(file_meta.get("stackResolution", [1, 1, 1]), "stackResolution", ndim=1)
            if (spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0)
                    or not np.allclose(spacing, spacing[0], rtol=1e-12, atol=0)):
                raise ValueError("anisotropic stackResolution requires an explicit APDV resolution; "
                                 "Python does not implicitly resample image data")
            resolution = float(spacing[0])
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
            raise ValueError(f"timepoint {value:g} is not in xp.fileMeta.timePoints")
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
