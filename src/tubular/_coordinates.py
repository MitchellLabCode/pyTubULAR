"""MATLAB-named coordinate helpers with explicit legacy arithmetic options."""

import numpy as np

from .mesh import real_array


def scalar(value, name, *, positive=False):
    a = real_array(value, name, ndim=0)
    result = float(a)
    if positive and result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def points(value, name, width):
    a = real_array(value, name, ndim=2)
    if a.shape[1] != width:
        raise ValueError(f"{name} must have shape (N, {width})")
    return a


def rotation(value):
    r = real_array(value, "rotation", ndim=2)
    if (r.shape != (3, 3) or not np.allclose(r @ r.T, np.eye(3), rtol=0, atol=1e-10)
            or not np.isclose(np.linalg.det(r), 1, rtol=0, atol=1e-10)):
        raise ValueError("rotation must be a proper orthonormal 3-by-3 matrix")
    return r


def image_size(im):
    a = np.asarray(im)
    if a.ndim == 1 and a.size == 2:
        size = real_array(a, "image size", ndim=1)
        if np.any(size != np.floor(size)):
            raise ValueError("image size must contain integer (width, height)")
        width, height = size
    elif a.ndim in (2, 3):
        height, width = a.shape[:2]
    else:
        raise ValueError("supply an image or a (width, height) pair")
    if width <= 1 or height <= 1:
        raise ValueError("both image dimensions must exceed one")
    return float(width), float(height)


def boolean(value, name):
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be boolean")
    return bool(value)


class CoordinateMethods:
    """One-based XY pixel locations; mesh connectivity remains zero-based."""

    @staticmethod
    def XY2uv(im, XY, doubleCovered=True, umax=1.0, vmax=1.0, *, legacy=False):
        """Map one-based pixels to UV; legacy reproduces MATLAB's vmax offset bug."""
        width, height = image_size(im)
        xy = points(XY, "XY", 2)
        umax, vmax = scalar(umax, "umax", positive=True), scalar(vmax, "vmax", positive=True)
        double = boolean(doubleCovered, "doubleCovered")
        boolean(legacy, "legacy")
        uv = (xy - 1) * [umax / (width - 1), vmax * (2 if double else 1) / (height - 1)]
        if double:
            uv[:, 1] -= 0.5 if legacy else 0.5 * vmax
        return uv

    @staticmethod
    def dXY2duv(im, dXY, doubleCovered=True, umax=1.0, vmax=1.0, *, legacy=False):
        """Map pixel displacements linearly. legacy adds MATLAB's spurious -0.5."""
        width, height = image_size(im)
        dxy = points(dXY, "dXY", 2)
        umax, vmax = scalar(umax, "umax", positive=True), scalar(vmax, "vmax", positive=True)
        double = boolean(doubleCovered, "doubleCovered")
        boolean(legacy, "legacy")
        duv = dxy * [umax / (width - 1), vmax * (2 if double else 1) / (height - 1)]
        if double and legacy:
            duv[:, 1] -= 0.5
        return duv

    @staticmethod
    def uv2XY(im, uv, doubleCovered=True, umax=1.0, vmax=1.0):
        """Inverse of corrected XY2uv, including non-unit vmax."""
        width, height = image_size(im)
        uv = points(uv, "uv", 2)
        umax, vmax = scalar(umax, "umax", positive=True), scalar(vmax, "vmax", positive=True)
        double = boolean(doubleCovered, "doubleCovered")
        xy = uv * [(width - 1) / umax, (height - 1) / ((2 if double else 1) * vmax)] + 1
        if double:
            xy[:, 1] += (height - 1) / 4
        return xy

    @staticmethod
    def clipXY(xx, yy, Lx, Ly, *, legacy=False):
        """Clip X and wrap Y with period Ly-1 (endpoints 1 and Ly coincide).

        legacy=True reproduces MATLAB's single, asymmetric wrapping step.
        """
        x, y = np.asarray(xx), np.asarray(yy)
        x, y = real_array(x, "xx", ndim=x.ndim), real_array(y, "yy", ndim=y.ndim)
        if x.shape != y.shape:
            raise ValueError("xx and yy must have the same shape")
        Lx, Ly = scalar(Lx, "Lx", positive=True), scalar(Ly, "Ly", positive=True)
        if Lx < 1 or Ly <= 1:
            raise ValueError("Lx must be >= 1 and Ly > 1")
        if boolean(legacy, "legacy"):
            y = np.where(y > Ly, y - Ly + 1, y)
            y = np.where(y < 1, y + Ly, y)
        else:
            y = np.where((y >= 1) & (y <= Ly), y, (y - 1) % (Ly - 1) + 1)
        return np.clip(x, 1, Lx), y

    @staticmethod
    def extendImage2DoubleCover(im, histeq=False):
        """Tile rows using MATLAB half-up rounding, preserving the input dtype.

        Unlike MATLAB, intensities are never silently cast to uint8.
        Histogram equalization is not yet supported.
        """
        if boolean(histeq, "histeq"):
            raise NotImplementedError("adaptive histogram equalization has not been ported")
        a = np.asarray(im)
        if (a.dtype.kind not in "biuf" or a.ndim not in (2, 3) or min(a.shape) < 1
                or (a.ndim == 3 and a.shape[2] not in (1, 3))):
            raise ValueError("im must be a nonempty grayscale or RGB image")
        if a.ndim == 3 and a.shape[2] == 1:
            a = a[..., 0]
        half = (a.shape[0] + 1) // 2
        return np.concatenate((a[-half:], a, a[:half]), axis=0)

    @staticmethod
    def singleToDoubleCover(XY, Ly):
        """Shift Y by round(Ly/2); Ly is the single-cover image height."""
        xy = points(XY, "XY", 2)
        xy[:, 1] += np.floor(scalar(Ly, "Ly", positive=True) / 2 + 0.5)
        return xy

    @staticmethod
    def doubleToSingleCover(XY, Ly):
        """Legacy recentering into the middle of a double-cover image.

        Ly is the double-cover height. This preserves the MATLAB behavior:
        coordinates stay in the double-cover frame; it is not the inverse of
        singleToDoubleCover. The algorithm wraps at most once each direction.
        """
        a = np.asarray(XY)
        if a.ndim not in (2, 3) or a.shape[-1] != 2:
            raise ValueError("XY must have shape (N, 2) or (Q, R, 2)")
        xy = real_array(a, "XY", ndim=a.ndim)
        Ly = scalar(Ly, "Ly", positive=True)
        y = xy[..., 1]
        y[:] = np.where(y < Ly * 0.25, y + Ly * 0.5, y)
        y[:] = np.where(y > Ly * 0.75, y - Ly * 0.5, y)
        return xy

    @staticmethod
    def dvAverageNematic(magnitude, theta):
        """Average a nematic across axis 1; return magnitudes and angles mod pi."""
        magnitude, theta = real_array(magnitude, "magnitude", ndim=2), real_array(theta, "theta", ndim=2)
        if magnitude.shape != theta.shape or 0 in magnitude.shape or np.any(magnitude < 0):
            raise ValueError("magnitude and theta need matching nonempty grids; magnitudes >= 0")
        x = np.mean(magnitude * np.cos(2 * theta), axis=1)
        y = np.mean(magnitude * np.sin(2 * theta), axis=1)
        return np.hypot(x, y), 0.5 * (np.arctan2(y, x) % (2 * np.pi))

    @staticmethod
    def invertRotation(rot):
        return rotation(rot).T

    @staticmethod
    def quarterIndicesDV(nV):
        """Exact legacy inclusive groups, converted to zero-based indices.

        These groups overlap, and the legacy dorsal set extends through q1
        rather than q0. They are not a disjoint partition into four quarters.
        """
        from .grid import _size
        n = _size(nV, "nV", 4)
        q0, q1, q2, q3 = np.floor(n * np.array([0.125, 0.375, 0.625, 0.875]) + 0.5).astype(int)
        return (np.r_[np.arange(q3, n + 1), np.arange(1, q1 + 1)] - 1,
                np.arange(q1, q2 + 1) - 1, np.arange(q0, q1 + 1) - 1,
                np.arange(q2, q3 + 1) - 1)
