from __future__ import annotations
from typing import Callable, ClassVar, Iterable, List, Optional, cast
from dataclasses import dataclass
import numpy as np
import pandas as pd
from .annotations import IntSequence, NDArray, ROI

@dataclass
class CBCTable():
    columns     : ClassVar[List[str]] = ['frames', 'index', 'x', 'y', 'I_raw', 'bgd', 'snr']
    table       : pd.DataFrame

    def __post_init__(self):
        if np.any([column not in self.columns for column in self.table.columns]):
            self.table = self.table[self.columns]

    @classmethod
    def read_hdf(cls, path: str, key: str) -> CBCTable:
        """Initialize a CBC table with data saved in a HDF5 file ``path`` at a ``key`` key inside
        the file and an experimental geometry object ``setup``.

        Args:
            path : Path to the HDF5 file.
            key : The group identifier in the HDF5 file.
            setup : Experimental geometry.

        Returns:
            A new CBC table object.
        """
        return cls(pd.DataFrame(pd.read_hdf(path, key)))

    def _repr_html_(self) -> Optional[str]:
        return cast(Callable[[], Optional[str]], self.table._repr_html_)()

    def get_frames(self) -> List[int]:
        return list(self.table['frames'].unique())

    def get_roi(self) -> ROI:
        return (self.table['y'].min(), self.table['y'].max() + 1,
                self.table['x'].min(), self.table['x'].max() + 1)

    def patterns_dataframe(self, roi: Optional[ROI]=None, frames: Optional[IntSequence]=None,
                           indices: Optional[Iterable[int]]=None) -> pd.DataFrame:
        """Return a single pattern table. The `x`, `y` coordinates are transformed by the ``crop``
        attribute.

        Args:
            frame : Frame index.

        Returns:
            A :class:`pandas.DataFrame` table.
        """
        if roi is None:
            roi = self.get_roi()
        if frames is None:
            frames = self.get_frames()
        frames = np.atleast_1d(frames)

        patterns = pd.concat([self.table[self.table['frames'] == frame] for frame in frames])
        if indices is not None:
            patterns['frames'] = patterns['frames'].replace(list(frames), list(indices))

        mask = (roi[0] <= patterns['y']) & (patterns['y'] < roi[1]) & \
               (roi[2] <= patterns['x']) & (patterns['x'] < roi[3])
        patterns = patterns[mask]
        patterns['y'], patterns['x'] = patterns['y'] - roi[0], patterns['x'] - roi[2]
        return patterns

    def patterns_image(self, roi: Optional[ROI]=None, frames: Optional[IntSequence]=None,
                       key: str='snr') -> NDArray:
        """Return a CBC pattern image array of the given attribute `key`. The `x`, `y` coordinates
        are transformed by the ``crop`` attribute.

        Args:
            frame : Frame index.
            key : Attribute's name.

        Returns:
            A pattern image array.
        """
        if roi is None:
            roi = self.get_roi()
        if frames is None:
            frames = self.get_frames()
        frames = np.atleast_1d(frames)

        patterns = self.patterns_dataframe(roi, frames, indices=np.arange(frames.size))
        image = np.zeros((frames.size, roi[1] - roi[0], roi[3] - roi[2]),
                         dtype=self.table.dtypes[key])
        image[patterns['frames'], patterns['y'], patterns['x']] = patterns[key]
        return image.squeeze()
