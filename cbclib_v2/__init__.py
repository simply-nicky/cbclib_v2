"""`cbclib`_ is a Python library for data processing of convergent beam crystallography datasets.

.. _cbclib: https://github.com/simply-nicky/cbclib

(c) Nikolay Ivanov, 2024
"""
from ._src.array_api import (add_at, argmin_at, array_namespace, ascupy, asjax, asnumpy,
                             default_rng, min_at, set_at)
from ._src.config import CPUConfig, get_cpu_config, reset_cpu_config, set_cpu_config
from ._src.crystfel import Detector, Panel, read_crystfel
from ._src.cxi_protocol import H5Protocol, H5Handler, Kinds, read_hdf, write_hdf
from ._src.data_container import (Container, DataContainer, ArrayContainer, IndexArray, split,
                                  to_list)
from ._src.data_processing import CrystData, CrystMetadata, StreakDetector, RegionDetector
from ._src.run import (RunConfig, BaseRun, XFELRunConfig, XFELRun, SwissFELConfig, SwissFELRun,
                       open_run)
from ._src.state import DynamicField, State, dynamic_fields, field, static_fields
from ._src.streaks import Lines, StackedStreaks, Streaks
from . import indexer
from . import label
from . import ndimage
from . import scripts
from . import slurm
from . import streak_finder
