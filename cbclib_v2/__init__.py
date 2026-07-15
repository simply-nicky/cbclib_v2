"""`cbclib`_ is a Python library for data processing of convergent beam crystallography datasets.

.. _cbclib: https://github.com/simply-nicky/cbclib

(c) Nikolay Ivanov, 2024
"""
from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from . import annotations, cuda, indexer, label, ndimage, scripts, slurm, streak_finder
    from ._src.array_api import (add_at, array_namespace, ascupy, asjax, asnumpy, default_api,
                                 default_rng, min_at, set_at)
    from ._src.config import (CPUConfig, get_cpu_config, reset_cpu_config, set_cpu_config,
                              set_cpu_pool_worker)
    from ._src.crystfel import Detector, Panel, read_crystfel
    from ._src.cxi_protocol import (H5Files, H5Handler, H5Protocol, Kinds, LoadIndices, read_hdf,
                                    write_hdf)
    from ._src.data_container import (ArrayContainer, Container, DataContainer, IndexedContainer,
                                      split, to_list)
    from ._src.data_processing import (CrystData, CrystMetadata, LSQData, RegionDetector,
                                       StreakDetector)
    from ._src.run import (BaseRun, RunConfig, RunLocator, LCLSConfig, LCLSRun, SwissFELConfig,
                           SwissFELRun, XFELRun, XFELConfig, open_run)
    from ._src.state import DynamicField, State, dynamic_fields, field, static_fields
    from ._src.streaks import Lines, StackedStreaks, Streaks

_EXPORTS = {
    "add_at": ("._src.array_api", "add_at"),
    "array_namespace": ("._src.array_api", "array_namespace"),
    "ascupy": ("._src.array_api", "ascupy"),
    "asjax": ("._src.array_api", "asjax"),
    "asnumpy": ("._src.array_api", "asnumpy"),
    "default_api": ("._src.array_api", "default_api"),
    "default_rng": ("._src.array_api", "default_rng"),
    "min_at": ("._src.array_api", "min_at"),
    "set_at": ("._src.array_api", "set_at"),
    "CPUConfig": ("._src.config", "CPUConfig"),
    "get_cpu_config": ("._src.config", "get_cpu_config"),
    "reset_cpu_config": ("._src.config", "reset_cpu_config"),
    "set_cpu_config": ("._src.config", "set_cpu_config"),
    "set_cpu_pool_worker": ("._src.config", "set_cpu_pool_worker"),
    "Detector": ("._src.crystfel", "Detector"),
    "Panel": ("._src.crystfel", "Panel"),
    "read_crystfel": ("._src.crystfel", "read_crystfel"),
    "H5Files": ("._src.cxi_protocol", "H5Files"),
    "H5Protocol": ("._src.cxi_protocol", "H5Protocol"),
    "H5Handler": ("._src.cxi_protocol", "H5Handler"),
    "Kinds": ("._src.cxi_protocol", "Kinds"),
    "LoadIndices": ("._src.cxi_protocol", "LoadIndices"),
    "read_hdf": ("._src.cxi_protocol", "read_hdf"),
    "write_hdf": ("._src.cxi_protocol", "write_hdf"),
    "Container": ("._src.data_container", "Container"),
    "DataContainer": ("._src.data_container", "DataContainer"),
    "ArrayContainer": ("._src.data_container", "ArrayContainer"),
    "IndexedContainer": ("._src.data_container", "IndexedContainer"),
    "split": ("._src.data_container", "split"),
    "to_list": ("._src.data_container", "to_list"),
    "CrystData": ("._src.data_processing", "CrystData"),
    "CrystMetadata": ("._src.data_processing", "CrystMetadata"),
    "LSQData": ("._src.data_processing", "LSQData"),
    "OnlineDetector": ("._src.data_processing", "OnlineDetector"),
    "StreakDetector": ("._src.data_processing", "StreakDetector"),
    "RegionDetector": ("._src.data_processing", "RegionDetector"),
    "RunConfig": ("._src.run", "RunConfig"),
    "RunLocator": ("._src.run", "RunLocator"),
    "BaseRun": ("._src.run", "BaseRun"),
    "LCLSConfig": ("._src.run", "LCLSConfig"),
    "LCLSRun": ("._src.run", "LCLSRun"),
    "XFELConfig": ("._src.run", "XFELConfig"),
    "XFELRun": ("._src.run", "XFELRun"),
    "SwissFELConfig": ("._src.run", "SwissFELConfig"),
    "SwissFELRun": ("._src.run", "SwissFELRun"),
    "open_run": ("._src.run", "open_run"),
    "DynamicField": ("._src.state", "DynamicField"),
    "State": ("._src.state", "State"),
    "dynamic_fields": ("._src.state", "dynamic_fields"),
    "field": ("._src.state", "field"),
    "static_fields": ("._src.state", "static_fields"),
    "Lines": ("._src.streaks", "Lines"),
    "StackedStreaks": ("._src.streaks", "StackedStreaks"),
    "Streaks": ("._src.streaks", "Streaks"),
}

_SUBMODULES = {
    "annotations",
    "cuda",
    "indexer",
    "label",
    "ndimage",
    "scripts",
    "slurm",
    "streak_finder",
}

__all__ = sorted((*_EXPORTS, *_SUBMODULES))

def __getattr__(name: str) -> Any:
    if name in _EXPORTS:
        module_name, attr_name = _EXPORTS[name]
        value = getattr(import_module(module_name, __name__), attr_name)
        globals()[name] = value
        return value
    if name in _SUBMODULES:
        value = import_module(f".{name}", __name__)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__() -> list[str]:
    return sorted((*globals(), *__all__))
