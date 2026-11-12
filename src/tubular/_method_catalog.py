"""Honest, inspectable coverage of the original MATLAB class interface."""

from copy import deepcopy
from importlib.resources import files
import json


class MethodNotPortedError(NotImplementedError):
    """A MATLAB method exists in the interface but has no Python implementation."""


def install_matlab_interface(cls):
    catalog = json.loads(files("tubular").joinpath("matlab_methods.json").read_text())
    records = {}
    for entry in catalog["methods"]:
        record = dict(entry)
        name = record["name"]
        implemented = callable(getattr(cls, name, None))
        record["status"] = "adapted" if implemented else "not_ported"
        records[name] = record
        if implemented:
            continue

        def make_pending(name, source, is_static):
            def fail(*args, **kwargs):
                raise MethodNotPortedError(
                    f"TubULAR.{name} is not yet ported to Python. MATLAB source: {source}. "
                    "See method_status() and docs/class_api.md; no calculation or file write occurred."
                )
            fail.__name__ = name
            fail.__qualname__ = f"TubULAR.{name}"
            fail.__doc__ = f"NOT PORTED. MATLAB source: {source}. Raises MethodNotPortedError."
            return staticmethod(fail) if is_static else fail

        setattr(cls, name, make_pending(name, record["source"], record["static"]))

    @classmethod
    def method_status(klass, name=None):
        """Return coverage records. 'adapted' means working within documented scope.

        The term does not promise every MATLAB option or side effect. Native
        Python convenience methods are not counted as original MATLAB methods.
        """
        if name is None:
            return deepcopy(records)
        if name not in records:
            raise KeyError(f"{name!r} is not in the original MATLAB method catalog")
        return deepcopy(records[name])

    @classmethod
    def available_methods(klass):
        """Names of MATLAB methods with working Python implementations."""
        return tuple(sorted(name for name, entry in records.items() if entry["status"] == "adapted"))

    cls.method_status = method_status
    cls.available_methods = available_methods
