"""CXI protocol (:class:`cbclib_v2.CXIProtocol`) is a helper class for a
:class:`cbclib_v2.CrystData` data container, which tells it where to look for the necessary data
fields in a CXI file. The class is fully customizable so you can tailor it to your particular data
structure of CXI file.

Examples:
    Generate the default built-in CXI protocol as follows:

    >>> import cbclib as cbc
    >>> cbc.CXIProtocol.import_default()
    CXIProtocol(paths={...})
"""
from dataclasses import dataclass, field
from functools import partial
from enum import Enum
from glob import iglob
from multiprocessing import Pool
import os
import fnmatch
import re
from types import TracebackType
from typing import (Callable, ClassVar, Dict, Iterator, List, Literal, NamedTuple, Optional,
                    Tuple, cast, get_type_hints)
import h5py
import numpy as np
from tqdm.auto import tqdm
from extra_data import open_run, stack_detector_data, DataCollection
from extra_geom import JUNGFRAUGeometry
from .data_container import Container, array_namespace, to_list
from .parser import Parser, INIParser, JSONParser
from .annotations import (Array, ArrayNamespace, Indices, IntArray, IntSequence, IntTuple, NumPy,
                          ReadOut, Shape)

EXP_ROOT_DIR = '/gpfs/exfel/exp'
CXI_PROTOCOL = os.path.join(os.path.dirname(__file__), 'config/cxi_protocol.ini')

class DataIndices(NamedTuple):
    files       : Array
    cxi_paths   : Array
    indices     : Array

    def __iter__(self) -> Iterator['DataIndices']:
        for index in range(len(self)):
            yield self[index]

    def __len__(self) -> int:
        return min(self.files.size, self.cxi_paths.size)

    def __getitem__(self, indices: Indices) -> 'DataIndices':
        file, cxi_path, index = self.files[indices], self.cxi_paths[indices], self.indices[indices]
        return DataIndices(file.reshape(file.shape if file.shape else 1),
                           cxi_path.reshape(cxi_path.shape if cxi_path.shape else 1),
                           index.reshape(index.shape if index.shape else 1))

    def __reduce__(self) -> Tuple:
        return (self.__class__, (self.files, self.cxi_paths, self.indices))

Processor = Callable[[Array], ReadOut]
cxi_worker : Callable[[DataIndices,], ReadOut]
extra_worker : Callable[[Array,], ReadOut]

class Kinds(str, Enum):
    scalar = 'scalar'
    sequence = 'sequence'
    frame = 'frame'
    stack = 'stack'
    no_kind = 'none'

Kind = Literal['scalar', 'sequence', 'frame', 'stack', 'none']

class BaseProtocol(Container):
    paths       : Dict[str, str | List[str]]
    kinds       : Dict[str, Kind]

    def __post_init__(self):
        self.kinds = {attr: self.kinds[attr] for attr in self.paths}

    def get_kind(self, attr: str) -> Kinds:
        return Kinds(self.kinds.get(attr, 'none'))

    def has_kind(self, *attributes: str, kind: Kinds=Kinds.stack) -> bool:
        for attr in attributes:
            if self.get_kind(attr) is kind:
                return True
        return False

@dataclass
class CXIProtocol(BaseProtocol):
    """CXI protocol class. Contains a CXI file tree path with the paths written to all the data
    attributes necessary for the :class:`cbclib_v2.CrystData` detector data container, their
    corresponding attributes' data types, and data structure.

    Args:
        paths : Dictionary with attributes' CXI default file paths.
    """
    paths       : Dict[str, List[str]]
    kinds       : Dict[str, Kind]
    known_ndims : ClassVar[Dict[Kinds, IntTuple]] = {Kinds.stack: (3,), Kinds.frame: (2, 3),
                                                     Kinds.sequence: (1, 2, 3),
                                                     Kinds.scalar: (0, 1, 2)}

    @classmethod
    def parser(cls, ext: str='ini') -> Parser:
        if ext == 'ini':
            return INIParser({'paths': 'paths', 'kinds': 'kinds'},
                             types=get_type_hints(cls))
        if ext == 'json':
            return JSONParser({'paths': 'paths', 'kinds': 'kinds'})

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

    def add_attribute(self, attr: str, paths: List[str]) -> 'CXIProtocol':
        """Add a data attribute to the protocol.

        Args:
            attr : Attribute's name.
            paths : List of attribute's CXI paths.

        Returns:
            A new protocol with the new attribute included.
        """
        return self.replace(paths=self.paths | {attr: paths})

    def find_path(self, attr: str, cxi_file: h5py.File) -> str:
        """Find attribute's path in a CXI file `cxi_file`.

        Args:
            attr : Data attribute.
            cxi_file : :class:`h5py.File` object of the CXI file.

        Returns:
            Attribute's path in the CXI file, returns an empty string if the attribute is not
            found.
        """
        paths = self.get_paths(attr)
        for path in paths:
            if path in cxi_file:
                return path
        return str()

    def get_paths(self, attr: str, value: List[str]=[]) -> List[str]:
        """Return the attribute's default path in the CXI file. Return ``value`` if ``attr`` is not
        found.

        Args:
            attr : The attribute to look for.
            value : Value which is returned if the `attr` is not found.

        Returns:
            Attribute's cxi file path.
        """
        return self.paths.get(attr, value)

    def get_ndim(self, attr: str, value: IntTuple=(0, 1, 2, 3)) -> IntTuple:
        """Return the acceptable number of dimensions that the attribute's data may have.

        Args:
            attr : The data attribute.
            value : value which is returned if the ``attr`` is not found.

        Returns:
            Number of dimensions acceptable for the attribute.
        """
        return self.known_ndims.get(self.get_kind(attr), value)

    def read_frame_shape(self, cxi_file: h5py.File) -> Tuple[int, int]:
        for attr in self.paths:
            if self.get_kind(attr) in (Kinds.stack, Kinds.frame):
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

    def read_indices(self, attr: str, fnames: List[str], xp: ArrayNamespace=NumPy) -> DataIndices:
        """Return a set of indices of the dataset containing the attribute's data inside a set
        of files.

        Args:
            attr : Attribute's name.
            cxi_files : A list of HDF5 file objects.

        Returns:
            Dataset indices of the data pertined to the attribute ``attr``.
        """
        files, cxi_paths, indices = [], [], []
        kind = self.get_kind(attr)

        for fname in fnames:
            with h5py.File(fname) as cxi_file:
                shapes = self.read_file_shapes(attr, cxi_file)
                for cxi_path, shape in shapes.items():
                    if len(shape) not in self.get_ndim(attr):
                        err_txt = f'Dataset at {cxi_file.filename}:' \
                                  f' {cxi_path} has invalid shape: {str(shape)}'
                        raise ValueError(err_txt)

                    if kind in [Kinds.stack, Kinds.sequence]:
                        files.extend(shape[0] * [cxi_file.filename,])
                        cxi_paths.extend(shape[0] * [cxi_path,])
                        indices.extend(range(shape[0]))
                    if kind in [Kinds.frame, Kinds.scalar]:
                        files.append(cxi_file.filename)
                        cxi_paths.append(cxi_path)
                        indices.append([])

        return DataIndices(xp.array(files), xp.array(cxi_paths), xp.array(indices))

@dataclass
class ReadWorker():
    ss_indices  : Indices
    fs_indices  : Indices
    proc        : Optional[Processor] = None

    def __call__(self, index: DataIndices) -> ReadOut:
        with h5py.File(index.files[0]) as cxi_file:
            dset = cast(h5py.Dataset, cxi_file[index.cxi_paths[0]])
            if index.indices.size:
                data = dset[index.indices[0], self.ss_indices, self.fs_indices]
            else:
                data = dset[..., self.ss_indices, self.fs_indices]
        if self.proc is not None:
            data = self.proc(data)
        return data

    @classmethod
    def initializer(cls, ss_indices: Indices, fs_indices: Indices, proc: Optional[None]=None):
        global cxi_worker
        cxi_worker = cls(ss_indices, fs_indices, proc)

    @classmethod
    def read(cls, index: DataIndices) -> Array:
        with h5py.File(index.files[0]) as cxi_file:
            dset = cast(h5py.Dataset, cxi_file[index.cxi_paths[0]])
            if index.indices.size:
                data = dset[index.indices[0]]
            else:
                data = dset[()]
        return data

    @staticmethod
    def run(index: DataIndices) -> ReadOut:
        return cxi_worker(index)

@dataclass
class CXIReader():
    protocol : CXIProtocol

    def load_stack(self, attr: str, indices: DataIndices, ss_idxs: Indices, fs_idxs: Indices,
                   proc: Optional[Processor], processes: int, verbose: bool,
                   xp: ArrayNamespace) -> Array:
        stack = []

        with Pool(processes=processes, initializer=ReadWorker.initializer,
                  initargs=(ss_idxs, fs_idxs, proc)) as pool:
            for frame in tqdm(pool.imap(ReadWorker.run, iter(indices)), total=len(indices),
                              disable=not verbose, desc=f'Loading {attr:s}'):
                stack.append(frame)

        return xp.stack(stack, axis=0)

    def load_frame(self, index: DataIndices, ss_idxs: Indices, fs_idxs: Indices,
                   proc: Optional[Processor]) -> ReadOut:
        return ReadWorker(ss_idxs, fs_idxs, proc)(index)

    def load_sequence(self, indices: DataIndices, xp: ArrayNamespace) -> Array:
        return xp.array([ReadWorker.read(index) for index in indices])

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

        return None, self.protocol.get_paths(attr)[0]

    def save_stack(self, attr: str, data: Array, mode: str='overwrite',
                    idxs: Optional[Indices]=None) -> None:
        xp = array_namespace(data)
        file, cxi_path = self.find_dataset(attr)

        if file is not None and cxi_path in file:
            dset : h5py.Dataset = cast(h5py.Dataset, file[cxi_path])
            if dset.shape[1:] == data.shape[1:]:
                if mode == 'append':
                    dset.resize(dset.shape[0] + data.shape[0], axis=0)
                    dset[-data.shape[0]:] = data
                elif mode == 'overwrite':
                    dset.resize(data.shape[0], axis=0)
                    dset[...] = data
                elif mode == 'insert':
                    if idxs is None:
                        raise ValueError('Incompatible indices')
                    if isinstance(idxs, slice):
                        idxs = xp.arange(dset.shape[0])[idxs]
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

class FileStore(Container):
    protocol    : BaseProtocol
    size        : int

    def __post_init__(self):
        if self.is_empty():
            self.update()

    def attributes(self) -> List[str]:
        raise NotImplementedError

    def load(self, attr: str, idxs: Optional[Indices]=None, ss_idxs: Indices=slice(None),
             fs_idxs: Indices=slice(None), proc: Optional[Processor]=None, processes: int=1,
             verbose: bool=True, xp: ArrayNamespace=NumPy) -> Array:
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
    indices     : Dict[str, DataIndices] = field(default_factory=dict)

    def __post_init__(self):
        if self.mode not in ['r', 'r+', 'w', 'w-', 'x', 'a']:
            raise ValueError(f'Wrong file mode: {self.mode}')
        if len(self.files) != len(to_list(self.names)):
            self.files = {fname: h5py.File(fname, mode=self.mode)
                          for fname in to_list(self.names)}
        super().__post_init__()

    @property
    def size(self) -> int:
        for attr in self.protocol.paths:
            if self.protocol.get_kind(attr) in [Kinds.stack, Kinds.sequence]:
                if attr in self.indices:
                    return len(self.indices[attr])
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

    def load(self, attr: str, idxs: Indices=slice(None), ss_idxs: Indices=slice(None),
             fs_idxs: Indices=slice(None), proc: Optional[Processor]=None, processes: int=1,
             verbose: bool=True, xp: ArrayNamespace=NumPy) -> Array:
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

        kind = self.protocol.get_kind(attr)

        if kind == Kinds.no_kind:
            raise ValueError(f'Invalid attribute: {attr:s}')

        reader = CXIReader(self.protocol)

        if kind == Kinds.stack:
            return reader.load_stack(attr=attr, indices=self.indices[attr][idxs],
                                     processes=processes, ss_idxs=ss_idxs, fs_idxs=fs_idxs,
                                     proc=proc, verbose=verbose, xp=xp)
        if kind == Kinds.frame:
            return xp.asarray(reader.load_frame(index=self.indices[attr],
                                                ss_idxs=ss_idxs, fs_idxs=fs_idxs,
                                                proc=proc))
        if kind == Kinds.scalar:
            return reader.load_sequence(self.indices[attr], xp)
        if kind == Kinds.sequence:
            return reader.load_sequence(self.indices[attr][idxs], xp)

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
        kind = self.protocol.get_kind(attr)

        writer = CXIWriter(cast(List[h5py.File], list(self.files.values())),
                            self.protocol)

        if kind in (Kinds.stack, Kinds.sequence):
            writer.save_stack(attr=attr, data=data, mode=mode, idxs=idxs)

        if kind in (Kinds.frame, Kinds.scalar):
            writer.save_data(attr=attr, data=data)

    def update(self):
        """Read the files for the data attributes contained in the protocol."""
        self.indices = {}
        for attr in self.protocol.paths:
            idxs = self.protocol.read_indices(attr, list(self.files))
            if len(idxs) != 0:
                self.indices[attr] = idxs

@dataclass
class ExtraProtocol(BaseProtocol):
    paths       : Dict[str, str]
    kinds       : Dict[str, Kind]
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
                              'paths': 'paths', 'kinds': 'kinds'},
                              types=get_type_hints(cls))
        if ext == 'json':
            return JSONParser({'experiment': ('folder', 'geom_path', 'modules', 'pattern',
                                              'proposal', 'source', 'starts_at'),
                               'paths': 'paths', 'kinds': 'kinds'})

        raise ValueError(f"Invalid format: {ext}")

    @classmethod
    def read(cls, file: str, ext: str='ini') -> 'ExtraProtocol':
        return cls(**cls.parser(ext).read(file))

    @classmethod
    def get_index(cls, fname: str, xp: ArrayNamespace=NumPy) -> IntArray:
        with h5py.File(fname, 'r') as file:
            index = cast(h5py.Dataset, file['INDEX/trainId'])[()]

        return xp.asarray(index)

    @classmethod
    def get_index_and_run(cls, fname: str, xp: ArrayNamespace=NumPy) -> IntArray:
        match = re.search(r'/r\d{4}/', fname)
        if match is None:
            raise ValueError("Invalid fname: " + fname)

        run = int(match[0][2:-1])
        index = cls.get_index(fname, xp)
        return xp.stack((xp.full(index.size, run), index), axis=-1, dtype=int)

    def detector_geometry(self) -> JUNGFRAUGeometry:
        return JUNGFRAUGeometry.from_crystfel_geom(self.geom_path)

    def detector_data(self, attr: str, train_data: Dict, xp: ArrayNamespace=NumPy) -> Array:
        data = stack_detector_data(train_data, self.paths[attr], modules=self.modules,
                                   starts_at=self.starts_at, pattern=self.pattern)
        return xp.asarray(data)

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

    def train_ids(self, run: int | List[int], include: str='*', processes: int=1,
                  xp: ArrayNamespace=NumPy) -> IntArray:
        files = self.find_files(self.find_proposal(), run, include)

        tids = []
        with Pool(processes=processes) as pool:
            for tid in pool.imap_unordered(ExtraProtocol.get_index, files):
                tids.append(tid)

        return xp.unique(xp.concatenate(tids))

    def open_run(self, run: int) -> DataCollection:
        run_data: DataCollection = open_run(proposal=self.proposal, run=run, data=self.folder,
                                            parallelize=False)
        return run_data.select(self.source)

    def read_frame_shape(self) -> Shape:
        return self.detector_geometry().output_array_for_position().shape

    def read_indices(self, run: int | List[int], include: str='*', xp: ArrayNamespace=NumPy
                     ) -> IntArray:
        files = self.find_files(self.find_proposal(), run, include)

        tids = [self.get_index_and_run(file, xp) for file in files]
        return xp.unique(xp.concatenate(tids), axis=0)

@dataclass
class ExtraReader():
    protocol : ExtraProtocol
    geom : JUNGFRAUGeometry

    @staticmethod
    def initializer(protocol: ExtraProtocol, attr: str, ss_idxs: Indices, fs_idxs: Indices,
                    proc: Optional[Processor], xp: ArrayNamespace):
        global extra_worker
        extra_worker = partial(ExtraReader(protocol, protocol.detector_geometry()).read_frame,
                               attr=attr, ss_idxs=ss_idxs, fs_idxs=fs_idxs, proc=proc, xp=xp)

    @staticmethod
    def read_worker(index: Array) -> ReadOut:
        return extra_worker(index)

    def read_frame(self, index: Array, attr: str, ss_idxs: Indices, fs_idxs: Indices,
                   proc: Optional[Processor], xp: ArrayNamespace) -> ReadOut:
        run = self.protocol.open_run(int(index[0]))
        _, train_data = run.train_from_id(index[1])
        data = self.protocol.detector_data(attr, train_data, xp)
        data = xp.nan_to_num(data, nan=0, posinf=0, neginf=0)
        data = xp.nan_to_num(xp.squeeze(self.geom.position_modules(data)[0]))[..., ss_idxs, fs_idxs]
        if proc is not None:
            data = proc(data)
        return data

    def load(self, attr: str, idxs: Array, ss_idxs: Indices, fs_idxs: Indices, processes: int,
             proc: Optional[Processor], verbose: bool, xp: ArrayNamespace) -> Array:
        stack = []
        with Pool(processes=processes, initializer=type(self).initializer,
                  initargs=(self.protocol, attr, ss_idxs, fs_idxs, proc, xp)) as pool:
            for frame in tqdm(pool.imap(ExtraReader.read_worker, idxs), total=idxs.shape[0],
                              disable=not verbose, desc=f'Loading {attr:s}'):
                stack.append(frame)

        return xp.stack(stack, axis=0)

@dataclass
class ExtraStore(FileStore):
    runs : int | List[int]
    protocol : ExtraProtocol
    indices : IntArray = field(default_factory=lambda: np.array([], dtype=int))

    @property
    def size(self) -> int:
        return self.indices.shape[0]

    def attributes(self) -> List[str]:
        return list(self.protocol.paths)

    def load(self, attr: str, idxs: Optional[Indices]=None, ss_idxs: Indices=slice(None),
             fs_idxs: Indices=slice(None), proc: Optional[Processor]=None, processes: int=1,
             verbose: bool=True, xp: ArrayNamespace=NumPy) -> Array:
        if idxs is None:
            idxs = xp.arange(self.size)

        reader = ExtraReader(self.protocol, self.protocol.detector_geometry())
        return reader.load(attr, self.indices[idxs], ss_idxs, fs_idxs, processes,
                           proc, verbose, xp)

    def read_frame_shape(self) -> Shape:
        return self.protocol.read_frame_shape()

    def update(self):
        self.indices = self.protocol.read_indices(self.runs)
