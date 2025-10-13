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
from enum import Enum
from glob import iglob
from math import prod
from multiprocessing import Pool
import os
import fnmatch
import re
from types import TracebackType
from typing import Any, Callable, ClassVar, Dict, Iterable, Iterator, List, Literal, Sequence, Tuple, TypeVar, cast
import h5py, hdf5plugin
import numpy as np
from tqdm.auto import tqdm
from extra_data import open_run, stack_detector_data, DataCollection
from extra_geom import JUNGFRAUGeometry
from .data_container import ArrayContainer, Container, DataContainer, array_namespace, to_list
from .parser import Parser, INIParser, JSONParser
from .annotations import (Array, ArrayNamespace, FileMode, Indices, IntArray, IntSequence, IntTuple,
                          NumPy, ReadOut, Shape)

EXP_ROOT_DIR = '/gpfs/exfel/exp'

I = TypeVar("I", bound='DataIndices')

class DataIndices(ArrayContainer):
    def __iter__(self: I) -> Iterator[I]:
        for index in range(len(self)):
            yield self[index]

    def __len__(self) -> int:
        lengths = [len(val) for val in self.to_dict().values() if len(val)]
        return min(lengths) if len(lengths) else 0

    def __getitem__(self: I, indices: Indices) -> I:
        xp = self.__array_namespace__()
        data = {attr: xp.atleast_1d(val[indices])
                for attr, val in self.to_dict().items() if xp.size(val)}
        return self.replace(**data)

@dataclass
class CXIIndices(DataIndices):
    files       : Array
    cxi_paths   : Array
    indices     : Array

Processor = Callable[[Array], ReadOut]

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
            return INIParser.from_container(cls)
        if ext == 'json':
            return JSONParser.from_container(cls)

        raise ValueError(f"Invalid format: {ext}")

    @classmethod
    def read(cls, file: str, ext: str='ini') -> 'CXIProtocol':
        """Return the default :class:`CXIProtocol` object.

        Returns:
            A :class:`CXIProtocol` object with the default parameters.
        """
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

    def read_indices(self, attr: str, cxi_files: Iterable[h5py.File], xp: ArrayNamespace=NumPy
                     ) -> CXIIndices:
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

        for cxi_file in cxi_files:
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

        return CXIIndices(xp.array(files), xp.array(cxi_paths), xp.array(indices))

cxi_worker : Callable[[CXIIndices,], ReadOut]

@dataclass
class CXIReadWorker():
    ss_indices  : Indices
    fs_indices  : Indices
    proc        : Processor | None = None

    def __call__(self, index: CXIIndices) -> ReadOut:
        with h5py.File(index.files[0], 'r') as cxi_file:
            dset = cast(h5py.Dataset, cxi_file[index.cxi_paths[0]])
            if index.indices.size:
                data = dset[index.indices[0], self.ss_indices, self.fs_indices]
            else:
                data = dset[..., self.ss_indices, self.fs_indices]
        if self.proc is not None:
            data = self.proc(data)
        return data

    @classmethod
    def initializer(cls, ss_indices: Indices, fs_indices: Indices, proc: Processor | None=None):
        global cxi_worker
        cxi_worker = cls(ss_indices, fs_indices, proc)

    @classmethod
    def read(cls, index: CXIIndices) -> Array:
        with h5py.File(index.files[0]) as cxi_file:
            dset = cast(h5py.Dataset, cxi_file[index.cxi_paths[0]])
            if index.indices.size:
                data = dset[index.indices[0]]
            else:
                data = dset[()]
        return data

    @staticmethod
    def run(index: CXIIndices) -> ReadOut:
        return cxi_worker(index)

@dataclass
class CXIReader():
    protocol : CXIProtocol

    def load_stack(self, attr: str, indices: CXIIndices, ss_idxs: Indices, fs_idxs: Indices,
                   proc: Processor | None, processes: int, verbose: bool,
                   xp: ArrayNamespace) -> Array:
        stack = []

        if processes > 1:
            with Pool(processes=processes, initializer=CXIReadWorker.initializer,
                    initargs=(ss_idxs, fs_idxs, proc)) as pool:
                for frame in tqdm(pool.imap(CXIReadWorker.run, iter(indices)), total=len(indices),
                                  disable=not verbose, desc=f'Loading {attr:s}'):
                    stack.append(frame)
        else:
            worker = CXIReadWorker(ss_idxs, fs_idxs, proc)
            for index in tqdm(indices, total=len(indices), disable=not verbose,
                              desc=f'Loading {attr:s}'):
                stack.append(worker(index))

        return xp.stack(stack, axis=0)

    def load_frame(self, index: CXIIndices, ss_idxs: Indices, fs_idxs: Indices,
                   proc: Processor | None) -> ReadOut:
        return CXIReadWorker(ss_idxs, fs_idxs, proc)(index)

    def load_sequence(self, attr, indices: CXIIndices, verbose: bool, xp: ArrayNamespace) -> Array:
        sequence = []
        for index in tqdm(indices, total=len(indices), disable=not verbose,
                          desc=f'Loading {attr:s}'):
            sequence.append(CXIReadWorker.read(index))
        return xp.array(sequence)

@dataclass
class CXIWriter():
    files : List[h5py.File]
    protocol : CXIProtocol

    def find_dataset(self, attr: str) -> Tuple[h5py.File | None, str]:
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

    def save_stack(self, attr: str, data: Array, mode: str, chunk: Shape,
                   idxs: Sequence[int] | IntArray | None=None):
        xp = array_namespace(data)
        file, cxi_path = self.find_dataset(attr)

        if file is not None and cxi_path in file:
            dset : h5py.Dataset = cast(h5py.Dataset, file[cxi_path])
            if dset.chunks is None:
                raise ValueError(f"Dataset {cxi_path} doesn't support chunks")
            if chunk != tuple(dset.chunks)[1:]:
                raise ValueError(f'Incompatible chunk size: {chunk} and {dset.chunks}')

            num_chunks = data.size // prod(chunk)
            data = xp.reshape(data, (num_chunks,) + chunk)

            if dset.shape[-len(chunk):] == chunk:
                if mode == 'append':
                    dset.resize(dset.shape[0] + num_chunks, axis=0)
                    dset[-num_chunks:] = data
                elif mode == 'overwrite':
                    dset.resize(num_chunks, axis=0)
                    dset[...] = data
                elif mode == 'insert':
                    if idxs is None:
                        raise ValueError('idxs is required for insert mode')
                    dset.resize(max(dset.shape[0], max(idxs) + 1), axis=0)
                    dset[idxs] = data

        else:
            if file is None:
                file = self.files[0]
            if cxi_path in file:
                del file[cxi_path]

            num_chunks = data.size // prod(chunk)
            data = xp.reshape(data, (num_chunks,) + chunk)

            file.create_dataset(cxi_path, data=data, shape=(num_chunks,) + chunk,
                                chunks=(1,) + chunk, maxshape=(None,) + chunk)

    def save_data(self, attr: str, data: Array):
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

    def attributes(self) -> List[str]:
        raise NotImplementedError

    def indices(self, attr: str) -> DataIndices:
        raise NotImplementedError

    def load(self, attr: str, idxs: DataIndices | None=None, ss_idxs: Indices=slice(None),
             fs_idxs: Indices=slice(None), proc: Processor | None=None, processes: int=1,
             verbose: bool=True, xp: ArrayNamespace=NumPy) -> Array:
        raise NotImplementedError

    def read_frame_shape(self) -> Tuple[int, int]:
        raise NotImplementedError

@dataclass
class CXIFiles():
    names   : str | List[str]
    mode    : FileMode = 'r'
    files   : Dict[str, h5py.File] | None = None

    def __post_init__(self):
        if self.mode not in ['r', 'r+', 'w', 'w-', 'x', 'a']:
            raise ValueError(f'Wrong file mode: {self.mode}')
        self.open()

    def __bool__(self) -> bool:
        return self.files is not None

    def __enter__(self) -> 'CXIFiles':
        self.open()
        return self

    def __exit__(self, exc_type: BaseException | None, exc: BaseException | None,
                 traceback: TracebackType | None):
        self.close()

    def __len__(self) -> int:
        return len(to_list(self.names))

    def __iter__(self) -> Iterator[h5py.File]:
        if self.files is not None:
            yield from self.files.values()
        else:
            raise ValueError('The hdf5 files are close')

    def open(self):
        if not self:
            self.files = {fname: h5py.File(fname, self.mode) for fname in to_list(self.names)}

    def close(self):
        """Close the files."""
        if self:
            self.files = None

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
    names       : str | List[str]
    protocol    : CXIProtocol

    def attributes(self) -> List[str]:
        return list(self.protocol.paths)

    def files(self, mode: FileMode='r') -> CXIFiles:
        return CXIFiles(self.names, mode)

    def indices(self, attr: str) -> CXIIndices:
        with self.files() as cxi_files:
            idxs = self.protocol.read_indices(attr, cxi_files)
            if len(idxs) != 0:
                return idxs
        raise ValueError(f"The file doesn't contain the '{attr}' attribute")

    def load(self, attr: str, idxs: CXIIndices | None=None, ss_idxs: Indices=slice(None),
             fs_idxs: Indices=slice(None), proc: Processor | None=None, processes: int=1,
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
        kind = self.protocol.get_kind(attr)

        if kind == Kinds.no_kind:
            raise ValueError(f'Invalid attribute: {attr:s}')

        reader = CXIReader(self.protocol)
        if idxs is None:
            idxs = self.indices(attr)

        if kind == Kinds.stack:
            return reader.load_stack(attr=attr, indices=idxs, processes=processes,
                                     ss_idxs=ss_idxs, fs_idxs=fs_idxs,
                                     proc=proc, verbose=verbose, xp=xp)
        if kind == Kinds.frame:
            return xp.asarray(reader.load_frame(index=idxs,
                                                ss_idxs=ss_idxs, fs_idxs=fs_idxs,
                                                proc=proc))
        if kind == Kinds.scalar:
            return reader.load_sequence(attr, idxs, False, xp)
        if kind == Kinds.sequence:
            return reader.load_sequence(attr, idxs, verbose, xp)

        raise ValueError("Wrong kind: " + str(kind))

    def read_frame_shape(self) -> Tuple[int, int]:
        """Read the input files and return a shape of the `frame` type data attribute.

        Raises:
            RuntimeError : If the files are not opened.

        Returns:
            The shape of the 2D `frame`-like data attribute.
        """
        with self.files() as cxi_files:
            for cxi_file in cxi_files:
                return self.protocol.read_frame_shape(cast(h5py.File, cxi_file))
        return (0, 0)

    def save(self, attr: str, data: Array, files: CXIFiles, mode: str='overwrite',
             chunk: Shape | None=None, idxs: int | slice | IntArray | Sequence[int] | None=None):
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
        if not files:
            raise ValueError('Files are closed')
        if files.mode == 'r':
            raise ValueError('Files are open in read-only mode')
        kind = self.protocol.get_kind(attr)

        writer = CXIWriter(list(files), self.protocol)

        if kind in (Kinds.sequence, Kinds.stack):
            if mode == 'append':
                if chunk is None:
                    chunk = data.shape
                idxs = None
            elif mode == 'insert':
                if idxs is None:
                    raise ValueError('idxs is required for insert mode')
                if isinstance(idxs, slice):
                    idxs = list(range(idxs.start, idxs.stop, idxs.step))
                if isinstance(idxs, int):
                    idxs = [idxs,]
                if len(idxs) != data.shape[0]:
                    if len(idxs) == 1:
                        data = data[None, ...]
                    else:
                        raise ValueError('Incompatible indices')

                if chunk is None:
                    chunk = data.shape[1:]
            elif mode == 'overwrite':
                if chunk is None:
                    chunk = data.shape[1:]
                idxs = None
            else:
                raise ValueError(f'Invalid mode: {mode}')

            writer.save_stack(attr=attr, data=data, mode=mode, chunk=chunk, idxs=idxs)

        if kind in (Kinds.frame, Kinds.scalar):
            writer.save_data(attr=attr, data=data)

@dataclass
class ExtraIndices(DataIndices):
    run         : IntArray = field(default_factory=lambda: np.array([], dtype=int))
    train_id    : IntArray = field(default_factory=lambda: np.array([], dtype=int))

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
            return INIParser.from_container(cls, 'experiment')
        if ext == 'json':
            return JSONParser.from_container(cls, 'experiment')

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
        return xp.array(data)

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

    def open_run(self, run: int) -> DataCollection:
        run_data: DataCollection = open_run(proposal=self.proposal, run=run, data=self.folder,
                                            parallelize=False)
        return run_data.select(self.source)

    def read_frame_shape(self) -> Shape:
        return self.detector_geometry().output_array_for_position().shape

    def read_indices(self, run: int | List[int], include: str='*', xp: ArrayNamespace=NumPy
                     ) -> ExtraIndices:
        files = self.find_files(self.find_proposal(), run, include)

        tids = [self.get_index_and_run(file, xp) for file in files]
        indices = xp.unique(xp.concatenate(tids), axis=0)
        return ExtraIndices(indices[..., 0], indices[..., 1])

extra_worker : Callable[[ExtraIndices,], ReadOut]

@dataclass
class ExtraReadWorker():
    protocol    : ExtraProtocol
    geom        : JUNGFRAUGeometry
    attr        : str
    ss_indices  : Indices
    fs_indices  : Indices
    proc        : Processor | None = None

    def __call__(self, index: ExtraIndices) -> ReadOut:
        run = self.protocol.open_run(int(index.run[0]))
        _, train_data = run.train_from_id(index.train_id[0])
        data = self.protocol.detector_data(self.attr, train_data)
        xp = array_namespace(data)
        data = xp.nan_to_num(data, nan=0, posinf=0, neginf=0)
        data = xp.nan_to_num(xp.squeeze(self.geom.position_modules(data)[0]))
        data = data[..., self.ss_indices, self.fs_indices]
        if self.proc is not None:
            data = self.proc(data)
        return data

    @classmethod
    def initializer(cls, protocol: ExtraProtocol, geom: JUNGFRAUGeometry, attr: str, ss_indices,
                    fs_indices, proc: Processor | None=None):
        global extra_worker
        extra_worker = cls(protocol, geom, attr, ss_indices, fs_indices, proc)

    @staticmethod
    def run(index: ExtraIndices) -> ReadOut:
        return extra_worker(index)

@dataclass
class ExtraReader():
    protocol    : ExtraProtocol
    geom        : JUNGFRAUGeometry

    def load(self, attr: str, idxs: ExtraIndices, ss_idxs: Indices, fs_idxs: Indices,
             processes: int, proc: Processor | None, verbose: bool, xp: ArrayNamespace
             ) -> Array:
        stack = []
        if processes > 1:
            with Pool(processes=processes, initializer=ExtraReadWorker.initializer,
                    initargs=(self.protocol, self.geom, attr, ss_idxs, fs_idxs, proc)) as pool:
                for frame in tqdm(pool.imap(ExtraReadWorker.run, iter(idxs)), total=len(idxs),
                                  disable=not verbose, desc=f'Loading {attr:s}'):
                    stack.append(frame)
        else:
            worker = ExtraReadWorker(self.protocol, self.geom, attr, ss_idxs, fs_idxs, proc)
            for index in tqdm(iter(idxs), total=len(idxs),
                              disable=not verbose, desc=f'Loading {attr:s}'):
                stack.append(worker(index))

        return xp.stack(stack, axis=0)

@dataclass
class ExtraStore(FileStore):
    runs : int | List[int]
    protocol : ExtraProtocol

    def attributes(self) -> List[str]:
        return list(self.protocol.paths)

    def indices(self, attr: str) -> ExtraIndices:
        if attr in self.attributes():
            return self.protocol.read_indices(self.runs)
        raise ValueError(f"The file doesn't contain the '{attr}' attribute")

    def load(self, attr: str, idxs: ExtraIndices | None=None, ss_idxs: Indices=slice(None),
             fs_idxs: Indices=slice(None), proc: Processor | None=None, processes: int=1,
             verbose: bool=True, xp: ArrayNamespace=NumPy) -> Array:
        if idxs is None:
            idxs = self.indices(attr)

        reader = ExtraReader(self.protocol, self.protocol.detector_geometry())
        return reader.load(attr, idxs, ss_idxs, fs_idxs, processes, proc, verbose, xp)

    def read_frame_shape(self) -> Shape:
        return self.protocol.read_frame_shape()

OptIntSequence = IntSequence | None

def read_hdf(input_file: FileStore, *attributes: str,
             indices: OptIntSequence | Tuple[OptIntSequence, Indices, Indices]=None,
             processes: int=1, verbose: bool=True) -> Dict[str, Any]:
    """Load data attributes from the input files in `files` file handler object.

    Args:
        attributes : List of attributes to load. Loads all the data attributes contained in
            the file(s) by default.
        idxs : List of frame indices to load.
        processes : Number of parallel workers used during the loading.
        verbose : Set the verbosity of the loading process.

    Raises:
        ValueError : If attribute is not existing in the input file(s).
        ValueError : If attribute is invalid.

    Returns:
        New :class:`CrystData` object with the attributes loaded.
    """
    if not attributes:
        attributes = tuple(input_file.attributes())

    if indices is None:
        frames, ss_idxs, fs_idxs = slice(None), slice(None), slice(None)
    elif isinstance(indices, (tuple, list)):
        frames, ss_idxs, fs_idxs = indices
        if frames is None:
            frames = slice(None)
        else:
            frames = to_list(frames)
    else:
        frames = to_list(indices)
        ss_idxs, fs_idxs = slice(None), slice(None)

    data_dict: Dict[str, Any] = {}
    size = 1

    for attr in attributes:
        if attr not in input_file.attributes():
            raise ValueError(f"No '{attr}' attribute in the input files")

        size = max(size, len(input_file.indices(attr)))
        idxs = input_file.indices(attr)
        if input_file.protocol.get_kind(attr) in [Kinds.stack, Kinds.sequence]:
            idxs = idxs[frames]

        data = input_file.load(attr, idxs=idxs, ss_idxs=ss_idxs, fs_idxs=fs_idxs,
                               processes=processes, verbose=verbose)

        data_dict[attr] = data

    if 'frames' not in data_dict:
        if hasattr(frames, '__len__'):
            data_dict['frames'] = np.asarray(frames)
        elif size > 1:
            data_dict['frames'] = np.arange(size)

    return data_dict

def write_hdf(container: DataContainer, output_file: CXIStore, *attributes: str,
              mode: str='overwrite', file_mode: FileMode='w', indices: Indices | None=None):
    """Save data arrays of the data attributes contained in the container to an output file.

    Args:
        attributes : List of attributes to save. Saves all the data attributes contained in
            the container by default.
        apply_transform : Apply `transform` to the data arrays if True.
        mode : Writing modes. The following keyword values are allowed:

            * `append` : Append the data array to already existing dataset.
            * `insert` : Insert the data under the given indices `idxs`.
            * `overwrite` : Overwrite the existing dataset.

        idxs : Indices where the data is saved. Used only if ``mode`` is set to 'insert'.

    Raises:
        ValueError : If the ``output_file`` is not defined inside the container.
    """
    xp = container.__array_namespace__()
    if not attributes:
        attributes = tuple(container.contents())

    with output_file.files(file_mode) as files:
        for attr in attributes:
            data = xp.asarray(getattr(container, attr))
            if not container.is_empty(data):
                output_file.save(attr, data, files, mode=mode, idxs=indices)
