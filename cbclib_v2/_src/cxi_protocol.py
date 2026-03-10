"""H5 protocol (:class:`cbclib_v2.H5Protocol`) is a helper class for a
:class:`cbclib_v2.CrystData` data container, which tells it where to look for the necessary data
fields in a H5 file. The class is fully customizable so you can tailor it to your particular data
structure of H5 file.

Examples:
    Generate the default built-in H5 protocol as follows:

    >>> import cbclib as cbc
    >>> cbc.H5Protocol.import_default()
    H5Protocol(paths={...})
"""
from dataclasses import InitVar, dataclass
from enum import Enum
from math import prod
from multiprocessing import Pool
import re
from typing import (Any, Callable, Dict, Generic, Iterable, Iterator, List, Literal, Sequence,
                    Tuple, TypeVar, cast, overload)
from typing_extensions import Self
import h5py, hdf5plugin
from tqdm.auto import tqdm
from .array_api import array_namespace, asnumpy
from .data_container import Container, DataContainer, list_indices, split, to_list
from .parser import from_container, from_file
from .annotations import (AnyNamespace, Array, ArrayNamespace, Attribute, CPArray, FileMode,
                          Indices, IntArray, IntSequence, JaxArray, NDArray, NumPy, NumPyNamespace,
                          Shape)

class DataIndices:
    def __iter__(self) -> Iterator[Any]:
        raise NotImplementedError

    def __getitem__(self: Self, key: Indices) -> Self:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

class TrainIndices(DataIndices):
    def __iter__(self) -> Iterator:
        raise NotImplementedError

    def index(self) -> Iterator[int]:
        raise NotImplementedError

    def split(self: Self, num_chunks: int) -> Self:
        raise NotImplementedError

Output = TypeVar('Output')

class LoadWorker(Generic[Output]):
    def __call__(self, index: Any) -> Output:
        raise NotImplementedError

    def initializer(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def run(index: Any) -> Output:
        raise NotImplementedError

@dataclass
class StackIndex(DataIndices):
    filename    : str
    n_frames    : int
    indices     : List[int] | None = None

    def __iter__(self) -> Iterator[Tuple[str, int]]:
        if self.indices is None:
            for index in range(self.n_frames):
                yield self.filename, index
        else:
            for index in self.indices:
                if index < 0:
                    index += self.n_frames
                if index < 0 or index >= self.n_frames:
                    raise IndexError(f"Index {index} is out of bounds for length {self.n_frames}")
                yield self.filename, index

    def __getitem__(self, key: Indices) -> "StackIndex":
        if isinstance(key, slice):
            key_list = list_indices(key, len(self))
        else:
            key_list = to_list(key)
        if self.indices is not None:
            key_list = [self.indices[index] for index in key_list]
        return StackIndex(filename=self.filename, n_frames=self.n_frames, indices=key_list)

    def __len__(self) -> int:
        return self.n_frames if self.indices is None else len(self.indices)

@dataclass
class FrameIndex(DataIndices):
    filename    : str

    def __iter__(self) -> Iterator[Tuple[str, slice]]:
        yield self.filename, slice(None)

    def __getitem__(self, key: Indices) -> "FrameIndex":
        if isinstance(key, slice):
            key_list = list_indices(key, len(self))
        else:
            key_list = to_list(key)

        if len(key_list) > 0:
            raise IndexError("FrameIndex only supports a single index or slice(None)")

        return self

    def __len__(self) -> int:
        return 1

@dataclass
class FrameIndices(DataIndices):
    file_indices    : List[FrameIndex]
    indices         : List[int] | None = None

    def __iter__(self) -> Iterator[Tuple[str, slice]]:
        if self.indices is None:
            for data_file in self.file_indices:
                yield from data_file
        else:
            for index in self.indices:
                if index < 0:
                    index += len(self.file_indices)
                if index < 0 or index >= len(self.file_indices):
                    raise IndexError(f"Index {index} is out of bounds for length "\
                                     f"{len(self.file_indices)}")
                yield from self.file_indices[index]

    def __getitem__(self, key: Indices) -> "FrameIndices":
        if isinstance(key, slice):
            key_list = list_indices(key, len(self))
        else:
            key_list = to_list(key)
        if self.indices is not None:
            key_list = [self.indices[index] for index in key_list]
        return FrameIndices(file_indices=self.file_indices, indices=key_list)

    def __len__(self) -> int:
        return len(self.file_indices) if self.indices is None else len(self.indices)

@dataclass
class StackIndices(TrainIndices):
    file_indices    : List[StackIndex]
    indices         : List[int] | None = None

    def __post_init__(self):
        self.total = sum(len(data_file) for data_file in self.file_indices)
        self.offsets = []
        running = 0
        for data_file in self.file_indices:
            self.offsets.append(running)
            running += len(data_file)

    def __iter__(self) -> Iterator[Tuple[str, int]]:
        if self.indices is None:
            for data_file in self.file_indices:
                yield from data_file
        else:
            for index in self.indices:
                if index < 0:
                    index += self.total
                if index < 0 or index >= self.total:
                    raise IndexError(f"Index {index} is out of bounds for length {self.total}")

                running = int(NumPy.searchsorted(self.offsets, index, side='right') - 1)
                yield from self.file_indices[running][index - self.offsets[running]]

    def __getitem__(self, key: Indices) -> "StackIndices":
        if isinstance(key, slice):
            key_list = list_indices(key, len(self))
        else:
            key_list = to_list(key)
        if self.indices is not None:
            key_list = [self.indices[index] for index in key_list]
        return StackIndices(file_indices=self.file_indices, indices=key_list)

    def __len__(self) -> int:
        return self.total if self.indices is None else len(self.indices)

    def index(self) -> Iterator[int]:
        if self.indices is None:
            yield from range(len(self))
        else:
            yield from self.indices

    def split(self, num_chunks: int) -> Iterator["StackIndices"]:
        if self.indices is None:
            indices = list(range(self.total))
        else:
            indices = self.indices

        if num_chunks <= 0:
            raise ValueError("num_chunks must be greater than 0")

        if num_chunks > 1:
            for chunk in split(indices, num_chunks):
                yield StackIndices(file_indices=self.file_indices, indices=chunk)
        else:
            yield self

T_Indices = TypeVar('T_Indices', bound=DataIndices)

@dataclass
class LoadIndices(Generic[T_Indices]):
    data_path   : str
    indices     : T_Indices

    def __getitem__(self, key: Indices) -> "LoadIndices[T_Indices]":
        return LoadIndices(data_path=self.data_path, indices=self.indices[key])

    @overload
    def __iter__(self: 'LoadIndices[StackIndices]') -> Iterator[Tuple[str, int]]: ...

    @overload
    def __iter__(self: 'LoadIndices[FrameIndices]') -> Iterator[Tuple[str, slice]]: ...

    def __iter__(self) -> Iterator[Tuple[str, int | slice]]:
        yield from self.indices

    def __len__(self) -> int:
        return len(self.indices)

AnyLoadIndices = LoadIndices[StackIndices] | LoadIndices[FrameIndices]

class Kinds(str, Enum):
    scalar = 'scalar'
    sequence = 'sequence'
    frame = 'frame'
    stack = 'stack'
    no_kind = 'none'

Kind = Literal['scalar', 'sequence', 'frame', 'stack', 'none']

def remove_slash(paths: List[str]):
    return [path[1:] if path.startswith('/') else path for path in paths]

StackAttributes = Literal['data', 'snr', 'eigen_field', 'eigen_value', 'whitefields']
FrameAttributes = Literal['flatfield', 'mask', 'std', 'whitefield']

@dataclass
class H5Protocol(Container):
    """H5 protocol class. Contains a H5 file tree path with the paths written to all the data
    attributes necessary for the :class:`cbclib_v2.CrystData` detector data container, their
    corresponding attributes' data types, and data structure.

    Args:
        paths : Dictionary with attributes' H5 default file paths.
    """
    paths       : Dict[str, List[str]]
    kinds       : Dict[str, str]

    def __post_init__(self):
        self.kinds = {attr: self.kinds[attr] for attr in self.paths}
        self.paths = {attr: remove_slash(paths) for attr, paths in self.paths.items()}

    def get_kind(self, attr: Attribute) -> Kinds:
        return Kinds(self.kinds.get(attr, 'none'))

    def has_kind(self, *attributes: str, kind: Kinds=Kinds.stack) -> bool:
        for attr in attributes:
            if self.get_kind(attr) is kind:
                return True
        return False

    @classmethod
    def read(cls, file: str) -> 'H5Protocol':
        """Return the default :class:`H5Protocol` object.

        Returns:
            A :class:`H5Protocol` object with the default parameters.
        """
        parser = from_file(file, cls)
        return cls.from_dict(**parser.read(file))

    def write(self, file: str):
        """Write the protocol to a file.

        Args:
            file : Path to the output file.
        """
        parser = from_container(file, self)
        parser.write(file, self)

    def find_path(self, attr: Attribute, cxi_file: h5py.File, default: str='') -> str:
        """Find attribute's path in a H5 file `cxi_file`.

        Args:
            attr : Data attribute.
            cxi_file : :class:`h5py.File` object of the H5 file.

        Returns:
            Attribute's path in the H5 file, returns an empty string if the attribute is not
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

    def get_paths(self, attr: Attribute, value: List[str]=[]) -> List[str]:
        """Return the attribute's default path in the H5 file. Return ``value`` if ``attr`` is not
        found.

        Args:
            attr : The attribute to look for.
            value : Value which is returned if the `attr` is not found.

        Returns:
            Attribute's cxi file path.
        """
        return self.paths.get(attr, value)

    @overload
    def read_indices(self, attr: StackAttributes, cxi_files: Iterable[h5py.File]
                     ) -> LoadIndices[StackIndices]: ...

    @overload
    def read_indices(self, attr: FrameAttributes, cxi_files: Iterable[h5py.File]
                     ) -> LoadIndices[FrameIndices]: ...

    @overload
    def read_indices(self, attr: str, cxi_files: Iterable[h5py.File]) -> AnyLoadIndices: ...

    def read_indices(self, attr: Attribute, cxi_files: Iterable[h5py.File]) -> AnyLoadIndices:
        """Return a set of indices of the dataset containing the attribute's data inside a set
        of files.

        Args:
            attr : Attribute's name.
            cxi_files : A list of H5 file objects.

        Returns:
            Dataset indices of the data pertined to the attribute ``attr``.
        """
        indices = []
        kind = self.get_kind(attr)

        data_path = ''
        for cxi_file in cxi_files:
            new_path = self.find_path(attr, cxi_file)

            if data_path and new_path != data_path:
                raise ValueError(f"Attribute '{attr}' is located at different paths in the files: "
                                 f"'{data_path}' and '{new_path}'")

            if new_path in cxi_file and isinstance(cxi_file[new_path], h5py.Dataset):
                if kind in [Kinds.stack, Kinds.sequence]:
                    index = StackIndex(cxi_file.filename, cxi_file[new_path].shape[0])
                elif kind in [Kinds.frame, Kinds.scalar]:
                    index = FrameIndex(cxi_file.filename)
                else:
                    raise ValueError(f"Invalid kind: {kind}")

                indices.append(index)
                data_path = new_path
            else:
                raise ValueError(f"Attribute '{attr}' not found in file '{cxi_file.filename}'")

        if kind in [Kinds.stack, Kinds.sequence]:
            return LoadIndices(data_path, StackIndices(file_indices=indices))
        return LoadIndices(data_path, FrameIndices(file_indices=indices))

cxi_worker : Callable[[Tuple[str, Indices]], NDArray]

@dataclass
class H5ReadWorker(LoadWorker[NDArray]):
    data_path   : str
    ss_indices  : Indices | None
    fs_indices  : Indices | None

    def __call__(self, index: Tuple[str, Indices]) -> NDArray:
        return self.load(self.data_path, index)

    def load(self, data_path: str, index: Tuple[str, Indices]) -> NDArray:
        xp : NumPyNamespace = NumPy

        file, idx = index
        with h5py.File(file, 'r') as cxi_file:
            dset = cast(h5py.Dataset, cxi_file[data_path])
            if self.ss_indices is not None and self.fs_indices is not None:
                if idx != slice(None):
                    chunk = xp.asarray(dset[idx, ..., self.ss_indices, self.fs_indices])
                else:
                    chunk = xp.asarray(dset[..., self.ss_indices, self.fs_indices])
            else:
                chunk = xp.asarray(dset[idx])

            # Replace NaNs with zeros
            chunk[xp.where(xp.isnan(chunk))] = 0

            # Reshape the chunk to remove the leading dimension if it's 1
            chunk = xp.reshape(chunk, (-1,) + chunk.shape[-2:])
            if chunk.shape[0] == 1:
                chunk = chunk.squeeze(axis=0)

        return chunk

    @classmethod
    def initializer(cls, data_path: str, ss_indices: Indices, fs_indices: Indices):
        global cxi_worker
        cxi_worker = cls(data_path, ss_indices, fs_indices)

    @staticmethod
    def run(index: Tuple[str, Indices]) -> NDArray:
        return cxi_worker(index)

@dataclass
class H5Reader():
    protocol : H5Protocol

    @overload
    def load_stack(self, attr: Attribute, indices: AnyLoadIndices, ss_idxs: Indices,
                   fs_idxs: Indices, processes: int, verbose: bool, xp: ArrayNamespace[CPArray]
                   ) -> CPArray: ...

    @overload
    def load_stack(self, attr: Attribute, indices: AnyLoadIndices, ss_idxs: Indices,
                   fs_idxs: Indices, processes: int, verbose: bool, xp: ArrayNamespace[JaxArray]
                   ) -> JaxArray: ...

    @overload
    def load_stack(self, attr: Attribute, indices: AnyLoadIndices, ss_idxs: Indices,
                   fs_idxs: Indices, processes: int, verbose: bool, xp: ArrayNamespace[NDArray]
                   ) -> NDArray: ...

    @overload
    def load_stack(self, attr: Attribute, indices: AnyLoadIndices, ss_idxs: Indices,
                   fs_idxs: Indices, processes: int, verbose: bool, xp: AnyNamespace
                   ) -> Array: ...

    def load_stack(self, attr: Attribute, indices: AnyLoadIndices, ss_idxs: Indices,
                   fs_idxs: Indices, processes: int, verbose: bool, xp: AnyNamespace) -> Array:
        stack = []

        if processes > 1:
            with Pool(processes=processes, initializer=H5ReadWorker.initializer,
                    initargs=(indices.data_path, ss_idxs, fs_idxs)) as pool:
                for frame in tqdm(pool.imap(H5ReadWorker.run, iter(indices)), total=len(indices),
                                  disable=not verbose, desc=f'Loading {attr:s}'):
                    stack.append(frame)
        else:
            worker = H5ReadWorker(indices.data_path, ss_idxs, fs_idxs)
            for index in tqdm(indices, total=len(indices), disable=not verbose,
                              desc=f'Loading {attr:s}'):
                stack.append(worker(index))

        if len(stack) == 1:
            return xp.asarray(stack[0])
        return xp.asarray(NumPy.stack(stack, axis=0))

    @overload
    def load_sequence(self, attr: Attribute, indices: AnyLoadIndices, verbose: bool,
                      xp: ArrayNamespace[CPArray]) -> CPArray: ...

    @overload
    def load_sequence(self, attr: Attribute, indices: AnyLoadIndices, verbose: bool,
                      xp: ArrayNamespace[JaxArray]) -> JaxArray: ...

    @overload
    def load_sequence(self, attr: Attribute, indices: AnyLoadIndices, verbose: bool,
                      xp: ArrayNamespace[NDArray]) -> NDArray: ...

    @overload
    def load_sequence(self, attr: Attribute, indices: AnyLoadIndices, verbose: bool,
                      xp: AnyNamespace) -> Array: ...

    def load_sequence(self, attr: Attribute, indices: AnyLoadIndices, verbose: bool,
                      xp: AnyNamespace) -> Array:
        sequence = []
        worker = H5ReadWorker(indices.data_path, None, None)
        for index in tqdm(indices, total=len(indices), disable=not verbose,
                          desc=f'Loading {attr:s}'):
            sequence.append(worker(index))
        return xp.array(sequence)

@dataclass
class H5Writer():
    file : h5py.File
    protocol : H5Protocol

    def save_stack(self, attr: Attribute, data: Array, mode: str, chunk: Shape,
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

    def save_data(self, attr: Attribute, data: Array):
        cxi_path = self.protocol.find_path(attr, self.file, self.protocol.get_paths(attr)[0])

        if cxi_path in self.file:
            dset : h5py.Dataset = cast(h5py.Dataset, self.file[cxi_path])
            if dset.shape == data.shape:
                dset[...] = data

        else:
            if cxi_path in self.file:
                del self.file[cxi_path]
            self.file.create_dataset(cxi_path, data=data, shape=data.shape)

@dataclass
class H5Files():
    names   : InitVar[str | List[str]]
    mode    : FileMode = 'r'

    def __post_init__(self, names):
        if isinstance(names, str):
            names = [names]
        self.files = sorted(names)

    def visit_files(self) -> Iterator[h5py.File]:
        for name in self.files:
            with h5py.File(name, self.mode) as file:
                yield file

@dataclass
class H5Handler:
    """File handler class for H5 and H5 files. Provides an interface to save and load data
    attributes to a file. Support multiple files. The handler saves data to the first file.

    Args:
        names : Paths to the files.
        mode : Mode in which to open file; one of ('w', 'r', 'r+', 'a', 'w-').
        protocol : H5 protocol. Uses the default protocol if not provided.

    Attributes:
        files : Dictionary of paths to the files and their file
            objects.
        protocol : :class:`cbclib_v2.H5Protocol` protocol object.
        mode : File mode. Valid modes are:

            * 'r' : Readonly, file must exist (default).
            * 'r+' : Read/write, file must exist.
            * 'w' : Create file, truncate if exists.
            * 'w-' or 'x' : Create file, fail if exists.
            * 'a' : Read/write if exists, create otherwise.
    """
    protocol    : H5Protocol

    def attributes(self) -> List[str]:
        return list(self.protocol.paths)

    @overload
    def indices(self, files: str | List[str] | H5Files, attr: StackAttributes
                ) -> LoadIndices[StackIndices]: ...

    @overload
    def indices(self, files: str | List[str] | H5Files, attr: FrameAttributes
                ) -> LoadIndices[FrameIndices]: ...

    @overload
    def indices(self, files: str | List[str] | H5Files, attr: str) -> AnyLoadIndices: ...

    def indices(self, files: str | List[str] | H5Files, attr: Attribute) -> AnyLoadIndices:
        if isinstance(files, (str, list)):
            files = H5Files(files)
        return self.protocol.read_indices(attr, files.visit_files())

    @overload
    def load(self, attr: Attribute, idxs: AnyLoadIndices, *, ss_idxs: Indices=slice(None),
             fs_idxs: Indices=slice(None), processes: int=1, verbose: bool=True,
             xp: ArrayNamespace[CPArray]) -> CPArray: ...

    @overload
    def load(self, attr: Attribute, idxs: AnyLoadIndices, *, ss_idxs: Indices=slice(None),
             fs_idxs: Indices=slice(None), processes: int=1, verbose: bool=True,
             xp: ArrayNamespace[JaxArray]) -> JaxArray: ...

    @overload
    def load(self, attr: Attribute, idxs: AnyLoadIndices, *, ss_idxs: Indices=slice(None),
             fs_idxs: Indices=slice(None), processes: int=1, verbose: bool=True,
             xp: ArrayNamespace[NDArray]) -> NDArray: ...

    @overload
    def load(self, attr: Attribute, idxs: AnyLoadIndices, *, ss_idxs: Indices=slice(None),
             fs_idxs: Indices=slice(None), processes: int=1, verbose: bool=True) -> NDArray: ...

    def load(self, attr: Attribute, idxs: AnyLoadIndices, *, ss_idxs: Indices=slice(None),
             fs_idxs: Indices=slice(None), processes: int=1, verbose: bool=True,
             xp: AnyNamespace=NumPy) -> Array:
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

        reader = H5Reader(self.protocol)

        if len(idxs) == 0:
            return xp.array([])

        if kind in (Kinds.stack, Kinds.frame):
            return reader.load_stack(attr=attr, indices=idxs, processes=processes,
                                     ss_idxs=ss_idxs, fs_idxs=fs_idxs, verbose=verbose,
                                     xp=xp)
        if kind == Kinds.scalar:
            return reader.load_sequence(attr, idxs, False, xp)
        if kind == Kinds.sequence:
            return reader.load_sequence(attr, idxs, verbose, xp)

        raise ValueError("Wrong kind: " + str(kind))

    def save(self, attr: Attribute, data: Array, file: h5py.File, mode: str='overwrite',
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

        writer = H5Writer(file, self.protocol)

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

            writer.save_stack(attr=attr, data=asnumpy(data), mode=mode, chunk=chunk, idxs=idxs)

        if kind in (Kinds.frame, Kinds.scalar):
            writer.save_data(attr=attr, data=asnumpy(data))

OptIntSequence = IntSequence | None

def read_hdf(filenames: str | List[str], handler: H5Handler, *attributes: str,
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
    xp = NumPy

    if not attributes:
        attributes = tuple(handler.attributes())

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
        if attr not in handler.attributes():
            raise ValueError(f"No '{attr}' attribute in the input files")

        idxs = handler.indices(filenames, attr)
        size = max(size, len(idxs))
        if handler.protocol.get_kind(attr) in [Kinds.stack, Kinds.sequence]:
            idxs = idxs[frames]

        data = handler.load(attr, idxs=idxs, ss_idxs=ss_idxs, fs_idxs=fs_idxs,
                            processes=processes, verbose=verbose)

        data_dict[attr] = data

    if 'frames' not in data_dict:
        if hasattr(frames, '__len__'):
            data_dict['frames'] = xp.asarray(frames)
        elif size > 1:
            data_dict['frames'] = xp.arange(size)

    return data_dict

def write_hdf(container: DataContainer, filename: str, handler: H5Handler, *attributes: str,
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

    with h5py.File(filename, file_mode) as file:
        for attr in attributes:
            data = getattr(container, attr)
            if not container.is_empty(data):
                handler.save(attr, data, file, mode=mode, idxs=indices)
