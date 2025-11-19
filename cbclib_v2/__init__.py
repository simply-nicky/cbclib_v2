"""`cbclib`_ is a Python library for data processing of convergent beam crystallography datasets.

.. _cbclib: https://github.com/simply-nicky/cbclib

(c) Nikolay Ivanov, 2024
"""
from ._src.crystfel import Detector, Panel, read_crystfel
from ._src.cxi_protocol import (CXIFiles, CXIIndices, CXIProtocol, CXIStore, Kinds, read_hdf,
                                write_hdf)
from ._src.data_container import (Container, DataContainer, ArrayContainer, IndexArray, add_at,
                                  argmin_at, array_namespace, min_at, set_at, split, to_list)
from ._src.data_processing import CrystData, CrystMetadata, StreakDetector, RegionDetector
from ._src.state import DynamicField, State, dynamic_fields, field, static_fields
from ._src.streaks import Lines, StackedStreaks, Streaks
from . import fft
from . import indexer
from . import label
from . import ndimage
from . import scripts
from . import slurm
from . import streak_finder
