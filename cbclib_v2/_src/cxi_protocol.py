"""CXI protocol (:class:`cbclib_v2.CXIProtocol`) is a helper class for a
:class:`cbclib_v2.CrystData` data container, which tells it where to look for the necessary data
fields in a CXI file. The class is fully customizable so you can tailor it to your particular data
structure of CXI file.

Examples:
    Generate the default built-in CXI protocol as follows:

    >>> import cbclib as cbc
    >>> cbc.CXIProtocol.import_default()
    CXIProtocol(load_paths={...})
"""
from dataclasses import dataclass, field
from functools import partial
from enum import auto, Enum
from glob import iglob
from multiprocessing import Pool
import os
import fnmatch
import re
from types import TracebackType
from typing import (Any, Callable, ClassVar, Dict, List, Literal, Optional, Protocol, Sequence,
                    Tuple, cast, get_type_hints)
import h5py
import numpy as np
from tqdm.auto import tqdm
from extra_data import open_run, stack_detector_data, DataCollection
from extra_geom import JUNGFRAUGeometry
from .data_container import DataContainer, Parser, INIParser, JSONParser, to_list
from .annotations import (Attribute, Array, Indices, IntSequence, IntTuple, NDArray, NDIntArray,
                          Processor, Shape)

EXP_ROOT_DIR = '/gpfs/exfel/exp'
CXI_PROTOCOL = os.path.join(os.path.dirname(__file__), 'config/cxi_protocol.ini')

cxi_worker : Callable[[str], Tuple[Any, ...]]
read_worker : Callable[[Array], NDArray]

class Kinds(Enum):
    SCALAR = auto()
    SEQUENCE = auto()
    FRAME = auto()
    STACK = auto()
    NO_KIND = auto()

class CrystProtocol():
    kinds : ClassVar[Dict[str, Kinds]] = {'data': Kinds.STACK, 'frames': Kinds.SEQUENCE,
                                          'good_frames': Kinds.SEQUENCE, 'mask': Kinds.FRAME,
                                          'scales': Kinds.SEQUENCE, 'snr': Kinds.STACK,
                                          'std': Kinds.FRAME, 'whitefield': Kinds.FRAME}

    @classmethod
    def get_kind(cls, attr: str) -> Kinds:
        return cls.kinds.get(attr, Kinds.NO_KIND)

    @classmethod
    def has_kind(cls, *attributes: str, kind: Kinds=Kinds.STACK) -> bool:
        for attr in attributes:
            if cls.get_kind(attr) is kind:
                return True
        return False

@dataclass
class CXIProtocol(DataContainer):
    """CXI protocol class. Contains a CXI file tree path with the paths written to all the data
    attributes necessary for the :class:`cbclib_v2.CrystData` detector data container, their
    corresponding attributes' data types, and data structure.

    Args:
        load_paths : Dictionary with attributes' CXI default file paths.
    """
    load_paths  : Dict[str, List[str]]
    known_ndims : ClassVar[Dict[Kinds, IntTuple]] = {Kinds.STACK: (3,), Kinds.FRAME: (2, 3),
                                                     Kinds.SEQUENCE: (1, 2, 3),
                                                     Kinds.SCALAR: (0, 1, 2)}

    def __post_init__(self):
        super().__post_init__()
        self.load_paths = {attr: paths for attr, paths in self.load_paths.items()
                           if CrystProtocol.get_kind(attr) != Kinds.NO_KIND}

    def read_worker(self, fname: str, attr: str) -> Tuple[Any, Any, Any]:
        kind = CrystProtocol.get_kind(attr)
        with h5py.File(fname) as cxi_file:
            shapes = self.read_file_shapes(attr, cxi_file)
            for cxi_path, shape in shapes.items():
                if len(shape) not in self.get_ndim(attr):
                    err_txt = f'Dataset at {cxi_file.filename}:'\
                            f' {cxi_path} has invalid shape: {str(shape)}'
                    raise ValueError(err_txt)

                if kind in [Kinds.STACK, Kinds.SEQUENCE]:
                    return (shape[0] * [cxi_file.filename,],
                            shape[0] * [cxi_path,], range(shape[0]))
                if kind in [Kinds.FRAME, Kinds.SCALAR]:
                    return ([cxi_file.filename,], [cxi_path,], [tuple(),])

            return ([], [], [])

    @classmethod
    def parser(cls, ext: str='ini') -> Parser:
        if ext == 'ini':
            return INIParser({'load_paths': 'load_paths'},
                             types=get_type_hints(cls))
        if ext == 'json':
            return JSONParser({'load_paths': 'load_paths'})

        raise ValueError(f"Invalid format: {ext}")

    @classmethod
    def read(cls, file: Optional[str]=None, ext: str='ini') -> 'CXIProtocol':
        """Return the default :class:`CXIProtocol` object.

        Returns:
            A :class:`CXIProtocol` object with the default parameters.
        """
        if file is None:
            file = CXI_PROTOCOL
        return cls(**cls.parser(ext).read(file))

    def add_attribute(self, attr: str, load_paths: List[str]) -> 'CXIProtocol':
        """Add a data attribute to the protocol.

        Args:
            attr : Attribute's name.
            load_paths : List of attribute's CXI paths.

        Returns:
            A new protocol with the new attribute included.
        """
        return self.replace(load_paths=self.load_paths | {attr: load_paths})

    def find_path(self, attr: str, cxi_file: h5py.File) -> str:
        """Find attribute's path in a CXI file `cxi_file`.

        Args:
            attr : Data attribute.
            cxi_file : :class:`h5py.File` object of the CXI file.

        Returns:
            Attribute's path in the CXI file, returns an empty string if the attribute is not
            found.
        """
        paths = self.get_load_paths(attr)
        for path in paths:
            if path in cxi_file:
                return path
        return str()

    def get_load_paths(self, attr: str, value: List[str]=[]) -> List[str]:
        """Return the attribute's default path in the CXI file. Return ``value`` if ``attr`` is not
        found.

        Args:
            attr : The attribute to look for.
            value : Value which is returned if the `attr` is not found.

        Returns:
            Attribute's cxi file path.
        """
        return self.load_paths.get(attr, value)

    def get_ndim(self, attr: str, value: IntTuple=(0, 1, 2, 3)) -> IntTuple:
        """Return the acceptable number of dimensions that the attribute's data may have.

        Args:
            attr : The data attribute.
            value : value which is returned if the ``attr`` is not found.

        Returns:
            Number of dimensions acceptable for the attribute.
        """
        return self.known_ndims.get(CrystProtocol.get_kind(attr), value)

    def read_frame_shape(self, cxi_file: h5py.File) -> Tuple[int, int]:
        for attr in self.load_paths:
            if CrystProtocol.get_kind(attr) in (Kinds.STACK, Kinds.FRAME):
                for shape in self.read_file_shapes(attr, cxi_file).values():
                    return cast(Tuple[int, int], shape[-2:])

        return (0, 0)

    def read_file_shapes(self, attr: str, cxi_file: h5py.File) -> Dict[str, IntTuple]:
        """Return a shape of the dataset containing the attribute's data inside a file.

        Args:
            attr : Attribute's name.
            cxi_file : HDF5 file object.

        Returns:
            List of all the datasets and their shapes inside ``cxi_file``.
        """
        cxi_path = self.find_path(attr, cxi_file)

        shapes = {}

        def caller(sub_path, obj):
            if isinstance(obj, h5py.Dataset):
                shapes[os.path.join(cxi_path, sub_path)] = obj.shape

        if cxi_path in cxi_file:
            cxi_obj = cxi_file[cxi_path]
            if isinstance(cxi_obj, h5py.Dataset):
                shapes[cxi_path] = cxi_obj.shape
            elif isinstance(cxi_obj, h5py.Group):
                cxi_obj.visititems(caller)
            else:
                raise ValueError(f"Invalid CXI object at '{cxi_path:s}'")

        return shapes

    def read_indices(self, attr: str, fnames: List[str]) -> NDArray:
        """Return a set of indices of the dataset containing the attribute's data inside a set
        of files.

        Args:
            attr : Attribute's name.
            cxi_files : A list of HDF5 file objects.

        Returns:
            Dataset indices of the data pertined to the attribute ``attr``.
        """
        files, cxi_paths, fidxs = [], [], []
        kind = CrystProtocol.get_kind(attr)

        if kind != Kinds.NO_KIND:
            for fname in fnames:
                new_files, new_cxi_paths, new_fidxs = self.read_worker(fname, attr)
                files.extend(new_files)
                cxi_paths.extend(new_cxi_paths)
                fidxs.extend(new_fidxs)

            return np.array([files, cxi_paths, fidxs], dtype=object).T

        raise ValueError(f"Invalid attribute {attr}")

class ReadWorker(Protocol):
    def __call__(self, index: Array, ss_idxs: Indices, fs_idxs: Indices) -> NDArray:
        ...

@dataclass
class CXIReader():
    protocol : CXIProtocol

    @staticmethod
    def initializer(reader: ReadWorker, ss_idxs: NDIntArray, fs_idxs: NDIntArray,
                    proc: Optional[Processor]):
        global read_worker
        read_worker = partial(reader, ss_idxs=ss_idxs, fs_idxs=fs_idxs, proc=proc)

    @staticmethod
    def read_item(index: Array) -> NDArray:
        dset : h5py.Dataset = cast(h5py.Dataset, h5py.File(index[0])[index[1]])
        return dset[index[2]]

    @staticmethod
    def read_frame(index: Array, ss_idxs: Indices, fs_idxs: Indices, proc: Optional[Processor]
                   ) -> NDArray | int | float | Sequence[int] | Sequence[float]:
        dset : h5py.Dataset = cast(h5py.Dataset, h5py.File(index[0])[index[1]])
        data = dset[index[2]][..., ss_idxs, fs_idxs]
        if proc is not None:
            data = proc(data)
        return data

    @staticmethod
    def read_worker(index: Array) -> NDArray:
        return read_worker(index)

    def load_stack(self, attr: Attribute, idxs: Array, ss_idxs: Indices, fs_idxs: Indices,
                   proc: Optional[Processor], processes: int, verbose: bool) -> NDArray:
        stack = []

        with Pool(processes=processes, initializer=type(self).initializer,
                  initargs=(type(self).read_frame, ss_idxs, fs_idxs, proc)) as pool:
            for frame in tqdm(pool.imap(CXIReader.read_worker, idxs), total=idxs.shape[0],
                              disable=not verbose, desc=f'Loading {attr:s}'):
                stack.append(frame)

        return np.stack(stack, axis=0)

    def load_frame(self, index: Array, ss_idxs: Indices, fs_idxs: Indices, proc: Optional[Processor]
                   ) -> NDArray | int | float | Sequence[int] | Sequence[float]:
        return self.read_frame(index, ss_idxs, fs_idxs, proc)

    def load_sequence(self, idxs: Array) -> NDArray:
        return np.array([self.read_item(index) for index in idxs])

@dataclass
class CXIWriter():
    files : List[h5py.File]
    protocol : CXIProtocol

    def find_dataset(self, attr: str) -> Tuple[Optional[h5py.File], str]:
        """Return the path to the attribute from the first file where the attribute is found. Return
        the default path if the attribute is not found inside the first file.

        Args:
            attr : Attribute's name.

        Returns:
            A file where the attribute is found and a path to the attribute inside the file.
        """
        for file in self.files:
            cxi_path = self.protocol.find_path(attr, file)

            if cxi_path:
                return file, cxi_path

        return None, self.protocol.get_load_paths(attr)[0]

    def save_stack(self, attr: str, data: Array, mode: str='overwrite',
                    idxs: Optional[Indices]=None) -> None:
        file, cxi_path = self.find_dataset(attr)

        if file is not None and cxi_path in file:
            dset : h5py.Dataset = cast(h5py.Dataset, file[cxi_path])
            if dset.shape[1:] == data.shape[1:]:
                if mode == 'append':
                    dset.resize(dset.shape[0] + data.shape[0],
                                                axis=0)
                    dset[-data.shape[0]:] = data
                elif mode == 'overwrite':
                    dset.resize(data.shape[0], axis=0)
                    dset[...] = data
                elif mode == 'insert':
                    if idxs is None:
                        raise ValueError('Incompatible indices')
                    if isinstance(idxs, slice):
                        idxs = cast(NDIntArray, np.arange(dset.shape[0])[idxs])
                    if isinstance(idxs, int):
                        idxs = [idxs,]
                    if len(idxs) != data.shape[0]:
                        raise ValueError('Incompatible indices')
                    dset.resize(max(dset.shape[0], max(idxs) + 1), axis=0)
                    dset[idxs] = data

        else:
            if file is None:
                file = self.files[0]
            if cxi_path in file:
                del file[cxi_path]
            file.create_dataset(cxi_path, data=data, shape=data.shape,
                                chunks=(1,) + data.shape[1:],
                                maxshape=(None,) + data.shape[1:])

    def save_data(self, attr: str, data: Array) -> None:
        file, cxi_path = self.find_dataset(attr)

        if file is not None and cxi_path in file:
            dset : h5py.Dataset = cast(h5py.Dataset, file[cxi_path])
            if dset.shape == data.shape:
                dset[...] = data

        else:
            if file is None:
                file = self.files[0]
            if cxi_path in file:
                del file[cxi_path]
            file.create_dataset(cxi_path, data=data, shape=data.shape)

class FileStore():
    size : int

    def attributes(self) -> List[Attribute]:
        raise NotImplementedError

    def load(self, attr: Attribute, idxs: Optional[Indices]=None, ss_idxs: Indices=slice(None),
             fs_idxs: Indices=slice(None), proc: Optional[Processor]=None, processes: int=1,
             verbose: bool=True) -> NDArray:
        raise NotImplementedError

    def read_frame_shape(self) -> Tuple[int, int]:
        raise NotImplementedError

    def save(self, attr: str, data: Array, mode: str='overwrite',
             idxs: Optional[Indices]=None):
        raise NotImplementedError

    def is_empty(self) -> bool:
        return self.size == 0

    def update(self):
        raise NotImplementedError

@dataclass
class CXIStore(FileStore):
    """File handler class for HDF5 and CXI files. Provides an interface to save and load data
    attributes to a file. Support multiple files. The handler saves data to the first file.

    Args:
        names : Paths to the files.
        mode : Mode in which to open file; one of ('w', 'r', 'r+', 'a', 'w-').
        protocol : CXI protocol. Uses the default protocol if not provided.

    Attributes:
        files : Dictionary of paths to the files and their file
            objects.
        protocol : :class:`cbclib_v2.CXIProtocol` protocol object.
        mode : File mode. Valid modes are:

            * 'r' : Readonly, file must exist (default).
            * 'r+' : Read/write, file must exist.
            * 'w' : Create file, truncate if exists.
            * 'w-' or 'x' : Create file, fail if exists.
            * 'a' : Read/write if exists, create otherwise.
    """
    Mode = Literal['r', 'r+', 'w', 'w-', 'x', 'a']

    names       : str | List[str]
    mode        : Mode = 'r'
    protocol    : CXIProtocol = CXIProtocol.read()
    files       : Dict[str, Optional[h5py.File]] = field(default_factory=dict)
    indices     : Dict[str, NDArray] = field(default_factory=dict)

    def __post_init__(self):
        if self.mode not in ['r', 'r+', 'w', 'w-', 'x', 'a']:
            raise ValueError(f'Wrong file mode: {self.mode}')
        if len(self.files) != len(to_list(self.names)):
            self.files = {fname: h5py.File(fname, mode=self.mode)
                          for fname in to_list(self.names)}

    @property
    def size(self) -> int:
        for attr in self.protocol.load_paths:
            if CrystProtocol.get_kind(attr) == Kinds.STACK:
                if attr in self.indices:
                    return self.indices[attr].shape[0]
        return 0

    def __bool__(self) -> bool:
        isopen = True
        for cxi_file in self.files.values():
            isopen &= bool(cxi_file)
        return isopen

    def __enter__(self) -> 'CXIStore':
        return self

    def __exit__(self, exc_type: Optional[BaseException], exc: Optional[BaseException],
                 traceback: Optional[TracebackType]):
        self.close()

    def attributes(self) -> List[str]:
        if self.indices is not None:
            return list(self.indices)
        return []

    def close(self):
        """Close the files."""
        if self:
            for fname, cxi_file in self.files.items():
                if cxi_file is not None:
                    cxi_file.close()
                self.files[fname] = None

    def load(self, attr: Attribute, idxs: Optional[Indices]=None, ss_idxs: Indices=slice(None),
             fs_idxs: Indices=slice(None), proc: Optional[Processor]=None, processes: int=1,
             verbose: bool=True) -> NDArray:
        """Load a data attribute from the files.

        Args:
            attr : Attribute's name to load.
            idxs : A list of frames' indices to load.
            verbose : Set the verbosity of the loading process.

        Raises:
            ValueError : If the attribute's kind is invalid.
            RuntimeError : If the files are not opened.

        Returns:
            Attribute's data array.
        """
        if not self:
            raise KeyError("Unable to load data (file is closed)")

        kind = CrystProtocol.get_kind(attr)

        if kind == Kinds.NO_KIND:
            raise ValueError(f'Invalid attribute: {attr:s}')

        if idxs is None:
            idxs = np.arange(self.size)

        reader = CXIReader(self.protocol)

        if kind == Kinds.STACK:
            return reader.load_stack(attr=attr, idxs=self.indices[attr][idxs],
                                     processes=processes, ss_idxs=ss_idxs,
                                     fs_idxs=fs_idxs, proc=proc, verbose=verbose)
        if kind == Kinds.FRAME:
            return np.asarray(reader.load_frame(index=self.indices[attr][0],
                                                ss_idxs=ss_idxs, fs_idxs=fs_idxs,
                                                proc=proc))
        if kind == Kinds.SCALAR:
            return reader.load_sequence(idxs=self.indices[attr][0])
        if kind == Kinds.SEQUENCE:
            return reader.load_sequence(idxs=self.indices[attr][idxs])

        raise ValueError("Wrong kind: " + str(kind))

    def read_frame_shape(self) -> Tuple[int, int]:
        """Read the input files and return a shape of the `frame` type data attribute.

        Raises:
            RuntimeError : If the files are not opened.

        Returns:
            The shape of the 2D `frame`-like data attribute.
        """
        for cxi_file in self.files.values():
            return self.protocol.read_frame_shape(cast(h5py.File, cxi_file))
        return (0, 0)

    def save(self, attr: str, data: Array, mode: str='overwrite',
             idxs: Optional[Indices]=None):
        """Save a data array pertained to the data attribute into the first file.

        Args:
            attr : Attribute's name.
            data : Data array.
            mode : Writing mode:

                * `append` : Append the data array to already existing dataset.
                * `insert` : Insert the data under the given indices ``idxs``.
                * `overwrite` : Overwrite the existing dataset.

            idxs : Indices where the data is saved. Used only if ``mode`` is set to 'insert'.

        Raises:
            ValueError : If the attribute's kind is invalid.
            ValueError : If the file is opened in read-only mode.
            RuntimeError : If the file is not opened.
        """
        if not self:
            raise KeyError("Unable to save data (file is closed)")
        if self.mode == 'r':
            raise ValueError('File is open in read-only mode')
        kind = CrystProtocol.get_kind(attr)

        writer = CXIWriter(cast(List[h5py.File], list(self.files.values())),
                            self.protocol)

        if kind in (Kinds.STACK, Kinds.SEQUENCE):
            writer.save_stack(attr=attr, data=data, mode=mode, idxs=idxs)

        if kind in (Kinds.FRAME, Kinds.SCALAR):
            writer.save_data(attr=attr, data=data)

    def update(self):
        """Read the files for the data attributes contained in the protocol."""
        self.indices = {}
        for attr in self.protocol.load_paths:
            idxs = self.protocol.read_indices(attr, list(self.files))
            if idxs.size:
                self.indices[attr] = idxs

@dataclass
class ExtraProtocol(DataContainer):
    data_paths  : Dict[str, str]
    folder      : str
    geom_path   : str
    modules     : int
    pattern     : str
    proposal    : int
    source      : str
    starts_at   : int

    @classmethod
    def parser(cls, ext: str='ini') -> Parser:
        if ext == 'ini':
            return INIParser({'experiment': ('folder', 'geom_path', 'modules', 'pattern',
                                             'proposal', 'source', 'starts_at'),
                              'data_paths': 'data_paths'}, types=get_type_hints(cls))
        if ext == 'json':
            return JSONParser({'experiment': ('folder', 'geom_path', 'modules', 'pattern',
                                              'proposal', 'source', 'starts_at'),
                               'data_paths': 'data_paths'})

        raise ValueError(f"Invalid format: {ext}")

    @classmethod
    def read(cls, file: str, ext: str='ini') -> 'ExtraProtocol':
        return cls(**cls.parser(ext).read(file))

    @classmethod
    def get_index(cls, fname: str) -> NDIntArray:
        with h5py.File(fname, 'r') as file:
            index: NDIntArray = cast(h5py.Dataset, file['INDEX/trainId'])[()]

        return index

    @classmethod
    def get_index_and_run(cls, fname: str) -> NDIntArray:
        match = re.search(r'/r\d{4}/', fname)
        if match is None:
            raise ValueError("Invalid fname: " + fname)

        run = int(match[0][2:-1])
        index = cls.get_index(fname)
        return np.stack((np.full(index.size, run), index), axis=-1, dtype=int)

    def detector_geometry(self) -> JUNGFRAUGeometry:
        return JUNGFRAUGeometry.from_crystfel_geom(self.geom_path)

    def detector_data(self, attr: str, train_data: Dict) -> NDArray:
        data = stack_detector_data(train_data, self.data_paths[attr], modules=self.modules,
                                   starts_at=self.starts_at, pattern=self.pattern)
        return np.asarray(data)

    def find_proposal(self) -> str:
        """Find the proposal directory for a given proposal on Maxwell"""
        for d in iglob(os.path.join(EXP_ROOT_DIR, f'*/*/p{self.proposal:06d}')):
            return d

        raise ValueError(f"Couldn't find proposal dir for {self.proposal:!r}")

    def find_files(self, prop_dir: str, run: IntSequence, include: str='*') -> List[str]:
        files = []
        for current in to_list(run):
            path = os.path.join(prop_dir, self.folder, f'r{current:04d}')
            new_files = [f for f in os.listdir(path)
                         if f.endswith('.h5') and (f.lower() != 'overview.h5')]
            new_files = [os.path.join(path, f) for f in fnmatch.filter(new_files, include)]
            files.extend(new_files)
        return files

    def train_ids(self, run: int | List[int], include: str='*',
                  processes: int=1) -> NDIntArray:
        files = self.find_files(self.find_proposal(), run, include)

        tids = []
        with Pool(processes=processes) as pool:
            for tid in pool.imap_unordered(ExtraProtocol.get_index, files):
                tids.append(tid)

        return np.unique(np.concatenate(tids))

    def open_run(self, run: int) -> DataCollection:
        run_data: DataCollection = open_run(proposal=self.proposal, run=run, data=self.folder,
                                            parallelize=False)
        return run_data.select(self.source)

    def read_frame_shape(self) -> Shape:
        return self.detector_geometry().output_array_for_position().shape

    def read_indices(self, run: int | List[int], include: str='*') -> NDIntArray:
        files = self.find_files(self.find_proposal(), run, include)

        tids = [self.get_index_and_run(file) for file in files]
        return np.unique(np.concatenate(tids), axis=0)

@dataclass
class ExtraReader():
    protocol : ExtraProtocol
    geom : JUNGFRAUGeometry

    @staticmethod
    def initializer(protocol: ExtraProtocol, attr: Attribute, ss_idxs: NDIntArray, fs_idxs: NDIntArray,
                    proc: Optional[Processor]):
        global worker
        worker = partial(ExtraReader(protocol, protocol.detector_geometry()).read_frame,
                         attr=attr, ss_idxs=ss_idxs, fs_idxs=fs_idxs, proc=proc)

    @staticmethod
    def read_worker(index: Array) -> NDArray | int | float | Sequence[int] | Sequence[float]:
        return worker(index)

    def read_frame(self, index: Array, attr: Attribute, ss_idxs: Indices,
                   fs_idxs: Indices, proc: Optional[Processor]
                   ) -> NDArray | int | float | Sequence[int] | Sequence[float]:
        run = self.protocol.open_run(int(index[0]))
        _, train_data = run.train_from_id(index[1])
        data = self.protocol.detector_data(attr, train_data)
        data = np.nan_to_num(data, nan=0, posinf=0, neginf=0)
        data = np.nan_to_num(np.squeeze(self.geom.position_modules(data)[0]))[..., ss_idxs, fs_idxs]
        if proc is not None:
            data = proc(data)
        return data

    def load(self, attr: Attribute, idxs: Array, ss_idxs: Indices, fs_idxs: Indices,
             processes: int, proc: Optional[Processor], verbose: bool) -> NDArray:
        stack = []
        with Pool(processes=processes, initializer=type(self).initializer,
                  initargs=(self.protocol, attr, ss_idxs, fs_idxs, proc)) as pool:
            for frame in tqdm(pool.imap(ExtraReader.read_worker, idxs), total=idxs.shape[0],
                              disable=not verbose, desc=f'Loading {attr:s}'):
                stack.append(frame)

        return np.stack(stack, axis=0)

@dataclass
class ExtraStore(FileStore):
    runs : int | List[int]
    protocol : ExtraProtocol
    indices : NDIntArray = field(default_factory=lambda: np.array([], dtype=int))

    @property
    def size(self) -> int:
        return self.indices.shape[0]

    def attributes(self) -> List[str]:
        return list(self.protocol.data_paths)

    def load(self, attr: Attribute, idxs: Optional[Indices]=None, ss_idxs: Indices=slice(None),
             fs_idxs: Indices=slice(None), proc: Optional[Processor]=None, processes: int=1,
             verbose: bool=True) -> NDArray:
        if idxs is None:
            idxs = np.arange(self.size)

        reader = ExtraReader(self.protocol, self.protocol.detector_geometry())
        return reader.load(attr, self.indices[idxs], ss_idxs, fs_idxs, processes,
                           proc, verbose)

    def read_frame_shape(self) -> Shape:
        return self.protocol.read_frame_shape()

    def update(self):
        self.indices = self.protocol.read_indices(self.runs)
