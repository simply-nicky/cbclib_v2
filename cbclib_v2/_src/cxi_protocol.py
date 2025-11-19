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
from math import prod
from multiprocessing import Pool
import os
import re
from typing import Any, Callable, Dict, Iterable, Iterator, List, Literal, Sequence, Tuple, TypeVar, cast
import h5py, hdf5plugin
import numpy as np
from tqdm.auto import tqdm
from .data_container import Container, DataContainer, IndexedContainer, array_namespace, list_indices, to_list
from .parser import Parser, get_parser
from .annotations import (Array, ArrayNamespace, BoolArray, FileMode, Indices, IntArray, IntSequence,
                          IntTuple, MultiIndices, NumPy, ReadOut, Shape)

I = TypeVar("I", bound='DataIndices')

class DataIndices(IndexedContainer):
    index : IntArray

    def __iter__(self: I) -> Iterator[I]:
        for index in range(len(self)):
            yield self[index]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self: I, indices: MultiIndices | BoolArray) -> I:
        if isinstance(indices, int):
            indices = [indices,]

        return super().__getitem__(indices)

    def zip(self) -> Iterator[Tuple]:
        xp = self.__array_namespace__()
        yield from zip(*(xp.ravel(val).tolist() for val in self.contents().values()))

@dataclass
class CXIIndices(DataIndices):
    index       : IntArray = field(default_factory=lambda: np.array([], dtype=int))
    files       : Array = field(default_factory=lambda: np.array([]))
    cxi_paths   : Array = field(default_factory=lambda: np.array([]))
    indices     : Array = field(default_factory=lambda: np.array([]))

Processor = Callable[[Array], ReadOut]

class Kinds(str, Enum):
    scalar = 'scalar'
    sequence = 'sequence'
    frame = 'frame'
    stack = 'stack'
    no_kind = 'none'

Kind = Literal['scalar', 'sequence', 'frame', 'stack', 'none']

def remove_slash(paths: List[str]):
    return [path[1:] if path.startswith('/') else path for path in paths]

class BaseProtocol(Container):
    paths       : Dict[str, List[str]]
    kinds       : Dict[str, Kind]

    def __post_init__(self):
        self.kinds = {attr: self.kinds[attr] for attr in self.paths}
        self.paths = {attr: remove_slash(paths) for attr, paths in self.paths.items()}

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
    kinds       : Dict[str, str]

    @classmethod
    def parser(cls, file_or_extension: str='ini') -> Parser:
        return get_parser(file_or_extension, cls)

    @classmethod
    def read(cls, file: str) -> 'CXIProtocol':
        """Return the default :class:`CXIProtocol` object.

        Returns:
            A :class:`CXIProtocol` object with the default parameters.
        """
        return cls(**cls.parser(file).read(file))

    def add_attribute(self, attr: str, paths: List[str]) -> 'CXIProtocol':
        """Add a data attribute to the protocol.

        Args:
            attr : Attribute's name.
            paths : List of attribute's CXI paths.

        Returns:
            A new protocol with the new attribute included.
        """
        return self.replace(paths=self.paths | {attr: paths})

    def find_path(self, attr: str, cxi_file: h5py.File, default: str='') -> str:
        """Find attribute's path in a CXI file `cxi_file`.

        Args:
            attr : Data attribute.
            cxi_file : :class:`h5py.File` object of the CXI file.

        Returns:
            Attribute's path in the CXI file, returns an empty string if the attribute is not
            found.
        """
        paths = self.get_paths(attr)
        matched = []

        def find_match(path: str):
            def caller(cxi_name: str, cxi_obj: h5py.Dataset | h5py.Group):
                if isinstance(cxi_obj, h5py.Dataset):
                    if re.match(path, cxi_name):
                        matched.append(cxi_name)

            return caller

        for path in paths:
            cxi_file.visititems(find_match(path))
            if matched:
                return matched[0]

        return default

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
        indices = []
        kind = self.get_kind(attr)

        for cxi_file in cxi_files:
            shapes = self.read_file_shapes(attr, cxi_file)
            for cxi_path, shape in shapes.items():
                if kind in [Kinds.stack, Kinds.sequence]:
                    index = CXIIndices(xp.arange(shape[0]), xp.full(shape[0], cxi_file.filename),
                                       xp.full(shape[0], cxi_path), xp.arange(shape[0]))
                elif kind in [Kinds.frame, Kinds.scalar]:
                    index = CXIIndices(xp.zeros(1), xp.array([cxi_file.filename,]),
                                       xp.array([cxi_path,]), xp.array([slice(None),]))
                else:
                    raise ValueError("Invalid kind: {kind}")

                indices.append(index)

        if indices:
            return CXIIndices.concatenate(indices)
        return CXIIndices()

cxi_worker : Callable[[CXIIndices,], ReadOut]

@dataclass
class CXIReadWorker():
    ss_indices  : Indices | None = None
    fs_indices  : Indices | None = None
    proc        : Processor | None = None

    def __call__(self, index: CXIIndices) -> ReadOut:
        loaded = []
        for file, cxi_path, idx in index.zip():
            with h5py.File(file, 'r') as cxi_file:
                dset = cast(h5py.Dataset, cxi_file[cxi_path])
                if self.ss_indices is not None and self.fs_indices is not None:
                    if idx != slice(None):
                        chunk = dset[idx, ..., self.ss_indices, self.fs_indices]
                    else:
                        chunk = dset[..., self.ss_indices, self.fs_indices]
                    xp = array_namespace(chunk)
                    chunk[xp.where(xp.isnan(chunk))] = 0
                    chunk = xp.reshape(chunk, (-1,) + chunk.shape[-2:])
                    if chunk.shape[0] == 1:
                        chunk = chunk.squeeze(axis=0)
                    loaded.append(chunk)
                else:
                    loaded.append(dset[idx])

        if len(loaded) > 1:
            xp = array_namespace(*loaded)
            data = xp.stack(loaded)
        else:
            data = loaded[0]

        if self.proc is not None:
            data = self.proc(data)
        return data

    @classmethod
    def initializer(cls, ss_indices: Indices, fs_indices: Indices, proc: Processor | None=None):
        global cxi_worker
        cxi_worker = cls(ss_indices, fs_indices, proc)

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
        worker = CXIReadWorker()
        for index in tqdm(indices, total=len(indices), disable=not verbose,
                          desc=f'Loading {attr:s}'):
            sequence.append(worker(index))
        return xp.array(sequence)

@dataclass
class CXIWriter():
    file : h5py.File
    protocol : CXIProtocol

    def save_stack(self, attr: str, data: Array, mode: str, chunk: Shape,
                   idxs: Sequence[int] | IntArray | None=None):
        xp = array_namespace(data)
        cxi_path = self.protocol.find_path(attr, self.file, self.protocol.get_paths(attr)[0])

        if cxi_path in self.file:
            dset : h5py.Dataset = cast(h5py.Dataset, self.file[cxi_path])
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
            if cxi_path in self.file:
                del self.file[cxi_path]

            num_chunks = data.size // prod(chunk)
            data = xp.reshape(data, (num_chunks,) + chunk)

            self.file.create_dataset(cxi_path, data=data, shape=(num_chunks,) + chunk,
                                     chunks=(1,) + chunk, maxshape=(None,) + chunk)

    def save_data(self, attr: str, data: Array):
        cxi_path = self.protocol.find_path(attr, self.file, self.protocol.get_paths(attr)[0])

        if cxi_path in self.file:
            dset : h5py.Dataset = cast(h5py.Dataset, self.file[cxi_path])
            if dset.shape == data.shape:
                dset[...] = data

        else:
            if cxi_path in self.file:
                del self.file[cxi_path]
            self.file.create_dataset(cxi_path, data=data, shape=data.shape)

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
    names   : List[str] | List[List[str]]
    mode    : FileMode = 'r'

    def __post_init__(self):
        if self.mode not in ['r', 'r+', 'w', 'w-', 'x', 'a']:
            raise ValueError(f'Wrong file mode: {self.mode}')

    def __len__(self) -> int:
        return len(self.names)

    def __iter__(self) -> 'Iterator[CXIFiles]':
        rest: List[str] = []
        for name in self.names:
            if isinstance(name, str):
                rest.append(name)
            else:
                yield CXIFiles(name, self.mode)

        if rest:
            for name in [rest,]:
                yield CXIFiles(name, self.mode)

    def visit_files(self) -> Iterator[h5py.File]:
        for name in self.names:
            if isinstance(name, str):
                with h5py.File(name, self.mode) as file:
                    yield file
            else:
                raise ValueError('names is a nested list, use ravel() to iterate over all files')

    def ravel(self) -> 'CXIFiles':
        return CXIFiles(to_list(self.names), self.mode)

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
    names       : str | List[str] | List[List[str]]
    protocol    : CXIProtocol

    def attributes(self) -> List[str]:
        return list(self.protocol.paths)

    def files(self, mode: FileMode='r') -> CXIFiles:
        if isinstance(self.names, str):
            return CXIFiles([self.names,], mode)
        return CXIFiles(self.names, mode)

    def file(self, mode: FileMode='r', index: int=0) -> h5py.File:
        if isinstance(self.names, str):
            return h5py.File(self.names, mode)
        return h5py.File(self.names[index], mode)

    def indices(self, attr: str) -> CXIIndices:
        indices = []
        for files in self.files():
            idxs = self.protocol.read_indices(attr, files.visit_files())
            if len(idxs) != 0:
                indices.append(idxs)
        if not indices:
            return CXIIndices()
        return CXIIndices.stack(indices, axis=-1)

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

        if len(idxs) == 0:
            return xp.array([])

        if kind == Kinds.stack:
            return reader.load_stack(attr=attr, indices=idxs, processes=processes,
                                     ss_idxs=ss_idxs, fs_idxs=fs_idxs,
                                     proc=proc, verbose=verbose, xp=xp)
        if kind == Kinds.frame:
            return xp.asarray(reader.load_frame(index=idxs, ss_idxs=ss_idxs,
                                                fs_idxs=fs_idxs, proc=proc))
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
        for cxi_file in self.files().ravel().visit_files():
            return self.protocol.read_frame_shape(cxi_file)
        return (0, 0)

    def save(self, attr: str, data: Array, file: h5py.File, mode: str='overwrite',
             chunk: Shape | None=None, idxs: Indices | None=None):
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
        if not file:
            raise ValueError(f'File {file.filename} are closed')
        if file.mode == 'r':
            raise ValueError('Files are open in read-only mode')
        kind = self.protocol.get_kind(attr)

        writer = CXIWriter(file, self.protocol)

        if kind in (Kinds.sequence, Kinds.stack):
            if mode == 'append':
                if chunk is None:
                    chunk = data.shape
                idxs = None
            elif mode == 'insert':
                if idxs is None:
                    raise ValueError('idxs is required for insert mode')
                idxs = list_indices(idxs, data.shape[0])
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

OptIntSequence = IntSequence | None

def read_hdf(data_file: FileStore, *attributes: str,
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
        attributes = tuple(data_file.attributes())

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
        if attr not in data_file.attributes():
            raise ValueError(f"No '{attr}' attribute in the input files")

        size = max(size, len(data_file.indices(attr)))
        idxs = data_file.indices(attr)
        if data_file.protocol.get_kind(attr) in [Kinds.stack, Kinds.sequence]:
            idxs = idxs[frames]

        data = data_file.load(attr, idxs=idxs, ss_idxs=ss_idxs, fs_idxs=fs_idxs,
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
    if not attributes:
        attributes = tuple(container.contents())

    with output_file.file(file_mode) as file:
        for attr in attributes:
            data = getattr(container, attr)
            if not container.is_empty(data):
                output_file.save(attr, data, file, mode=mode, idxs=indices)
