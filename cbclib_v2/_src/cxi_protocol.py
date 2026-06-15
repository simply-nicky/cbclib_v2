"""H5 protocol (:class:`cbclib_v2.H5Protocol`) is a helper class for a
:class:`cbclib_v2.CrystData` data container, which tells it where to look for the necessary data
fields in a H5 file. The class is fully customizable so you can tailor it to your particular data
structure of H5 file.

Example:
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

    def initializer(self, *args: Any, is_pool: bool=False, **kwargs: Any) -> None:
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
    """Index table for a single HDF5 attribute across one or more files.

    Returned by :meth:`H5Handler.indices`; passed to :meth:`H5Handler.load`.
    Supports subscripting and iteration so a subset of frames can be selected
    before loading.

    Attributes:
        data_path: HDF5 dataset path resolved for this attribute.
        indices: Per-file frame or slice indices.
        attr: Attribute name; carried from :meth:`H5Handler.indices` so
            :meth:`H5Handler.load` does not need it as a separate argument.

    Example:
        Read the first 100 frames of the 'data' attribute from 'run_001.h5':

        >>> indices = handler.indices('run_001.h5', 'data')
        >>> first_100 = indices[:100]
        >>> frames = handler.load(first_100)
    """
    data_path   : str
    indices     : T_Indices
    attr        : str = ''

    def __getitem__(self, key: Indices) -> "LoadIndices[T_Indices]":
        return LoadIndices(data_path=self.data_path, indices=self.indices[key], attr=self.attr)

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
    """Data dimensionality kind for HDF5 dataset attributes.

    Controls which index dimensions :meth:`H5Handler.load` supports when
    reading a dataset.

    Attributes:
        scalar: Single value per file; loaded whole, no index subsetting.
        sequence: 1-D array per file; supports selecting elements by frame
            index.
        frame: Single 2-D image; supports pixel ROI selection via slow-scan
            and fast-scan indices.
        stack: 3-D stack of frames; supports both frame-index subsetting and
            pixel ROI.
        no_kind: Attribute not defined in the protocol; loading raises
            :exc:`ValueError`.
    """
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
    """HDF5 file-tree protocol for a :class:`CrystData` detector dataset.

    Maps each data attribute to one or more candidate HDF5 paths and to its
    :class:`Kinds` dimensionality.  Pass the protocol to :class:`H5Handler`
    to load and save those attributes from HDF5 files.

    The protocol is normally loaded from a JSON or INI file with :meth:`H5Protocol.read`;
    it can also be constructed directly by supplying ``paths`` and ``kinds``
    dictionaries.

    Attributes:
        paths: Mapping from attribute name to a list of candidate HDF5
            dataset paths searched in order when locating the attribute in a
            file.
        kinds: Mapping from attribute name to its :class:`Kinds` value
            (stored as a string and resolved via :meth:`get_kind`).

    Example:
        Read a protocol from a JSON file and use it to create an H5Handler for loading data:

        >>> protocol = H5Protocol.read('cbc_protocol.json')
        >>> protocol.get_kind('data')
        <Kinds.stack: 'stack'>
        >>> handler = H5Handler(protocol)
    """
    paths       : Dict[str, List[str]]
    kinds       : Dict[str, str]

    def __post_init__(self):
        self.kinds = {attr: self.kinds[attr] for attr in self.paths}
        self.paths = {attr: remove_slash(paths) for attr, paths in self.paths.items()}

    def get_kind(self, attr: Attribute) -> Kinds:
        """Get the kind of a given attribute.

        Args:
            attr: The attribute name.

        Returns:
            The :class:`Kinds` value for the attribute. If the attribute is not found,
            returns :attr:`Kinds.no_kind`.
        """
        return Kinds(self.kinds.get(attr, 'none'))

    def has_kind(self, *attributes: str, kind: Kinds=Kinds.stack) -> bool:
        """Check if any of the given attributes has the specified kind.

        Args:
            *attributes: The attribute names to check.
            kind: The kind to check for.

        Returns:
            True if any of the attributes has the specified kind, False otherwise.
        """
        for attr in attributes:
            if self.get_kind(attr) is kind:
                return True
        return False

    @classmethod
    def read(cls, file: str) -> 'H5Protocol':
        """Load a protocol from a JSON or INI file.

        Args:
            file: Path to the protocol file.

        Returns:
            :class:`H5Protocol` populated with the paths and kinds read from
            ``file``.

        Example:
            Read a protocol from a JSON file:

            >>> protocol = H5Protocol.read('cbc_protocol.json')
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
            return LoadIndices(data_path, StackIndices(file_indices=indices), attr)
        return LoadIndices(data_path, FrameIndices(file_indices=indices), attr)

WorkerType = Callable[[Tuple[str, Indices]], NDArray]
cxi_worker : WorkerType

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

        # Reshape the chunk to remove the leading dimension if it's 1
        chunk = xp.reshape(chunk, (-1,) + chunk.shape[-2:])

        # Replace NaNs with zeros
        chunk[xp.where(xp.isnan(chunk))] = 0

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
    def load_stack(self, indices: AnyLoadIndices, ss_idxs: Indices,
                   fs_idxs: Indices, processes: int, verbose: bool, xp: ArrayNamespace[CPArray]
                   ) -> CPArray: ...

    @overload
    def load_stack(self, indices: AnyLoadIndices, ss_idxs: Indices,
                   fs_idxs: Indices, processes: int, verbose: bool, xp: ArrayNamespace[JaxArray]
                   ) -> JaxArray: ...

    @overload
    def load_stack(self, indices: AnyLoadIndices, ss_idxs: Indices,
                   fs_idxs: Indices, processes: int, verbose: bool, xp: ArrayNamespace[NDArray]
                   ) -> NDArray: ...

    @overload
    def load_stack(self, indices: AnyLoadIndices, ss_idxs: Indices,
                   fs_idxs: Indices, processes: int, verbose: bool, xp: AnyNamespace
                   ) -> Array: ...

    def load_stack(self, indices: AnyLoadIndices, ss_idxs: Indices,
                   fs_idxs: Indices, processes: int, verbose: bool, xp: AnyNamespace) -> Array:
        stack = []

        if processes > 1:
            with Pool(processes=processes, initializer=H5ReadWorker.initializer,
                    initargs=(indices.data_path, ss_idxs, fs_idxs)) as pool:
                for frame in tqdm(pool.imap(H5ReadWorker.run, iter(indices)), total=len(indices),
                                  disable=not verbose, desc=f'Loading {indices.attr:s}'):
                    stack.append(frame)
        else:
            worker = H5ReadWorker(indices.data_path, ss_idxs, fs_idxs)
            for index in tqdm(indices, total=len(indices), disable=not verbose,
                              desc=f'Loading {indices.attr:s}'):
                stack.append(worker(index))

        if len(stack) == 1:
            return xp.asarray(stack[0])
        return xp.asarray(NumPy.stack(stack, axis=0))

    @overload
    def load_sequence(self, indices: AnyLoadIndices, verbose: bool,
                      xp: ArrayNamespace[CPArray]) -> CPArray: ...

    @overload
    def load_sequence(self, indices: AnyLoadIndices, verbose: bool,
                      xp: ArrayNamespace[JaxArray]) -> JaxArray: ...

    @overload
    def load_sequence(self, indices: AnyLoadIndices, verbose: bool,
                      xp: ArrayNamespace[NDArray]) -> NDArray: ...

    @overload
    def load_sequence(self, indices: AnyLoadIndices, verbose: bool,
                      xp: AnyNamespace) -> Array: ...

    def load_sequence(self, indices: AnyLoadIndices, verbose: bool,
                      xp: AnyNamespace) -> Array:
        sequence = []
        worker = H5ReadWorker(indices.data_path, None, None)
        for index in tqdm(indices, total=len(indices), disable=not verbose,
                          desc=f'Loading {indices.attr:s}'):
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
    """Ordered collection of HDF5 file paths opened on demand.

    Pass a single path or a list of paths; the files are sorted and opened
    one at a time by :meth:`visit_files`.  Used by :meth:`H5Handler.indices`
    and :meth:`H5Handler.load` to iterate over multi-file runs without
    keeping all files open simultaneously.

    Attributes:
        names: Path or list of paths to the HDF5 files.
        mode: h5py file open mode (default ``'r'``).

    Example:
        Visit the files 'run_001.h5' and 'run_002.h5' and print their filenames and keys:

        >>> files = H5Files(['run_001.h5', 'run_002.h5'])
        >>> for hf in files.visit_files():
        ...     print(hf.filename, list(hf.keys()))
    """
    names   : InitVar[str | List[str]]
    mode    : FileMode = 'r'

    def __post_init__(self, names):
        if isinstance(names, str):
            names = [names]
        self.files = sorted(names)

    def visit_files(self) -> Iterator[h5py.File]:
        """Generator that yields h5py.File objects for each file in the collection,
        opened in the specified mode.

        Yields:
            h5py.File objects for each file in the collection, opened in the specified
            mode.
        """
        for name in self.files:
            with h5py.File(name, self.mode) as file:
                yield file

@dataclass
class H5Handler:
    """Protocol-aware interface for reading and writing HDF5 datasets.

    Wraps an :class:`H5Protocol` and exposes :meth:`load` and :meth:`save`
    methods that dispatch on the attribute's :class:`Kinds`, so callers never
    need to manage dataset paths or array layout directly.

    The typical workflow is to first call :meth:`indices` to build an index
    table of all data elements stored across one or more files.  The returned
    :class:`LoadIndices` object supports indexing and iteration, so a subset
    of frames can be selected before passing it to :meth:`load`.

    Attributes:
        protocol: The :class:`H5Protocol` that describes paths and
            dimensionality for each attribute.

    Example:
        Load the first 100 frames of the 'data' attribute from 'run_001.h5' using a
        protocol read from a JSON file:

        >>> protocol = H5Protocol.read('cbc_protocol.json')
        >>> handler = H5Handler(protocol)
        >>> indices = handler.indices('run_001.h5', 'data')
        >>> frames = handler.load(indices[:100])
    """
    protocol    : H5Protocol

    def attributes(self) -> List[str]:
        """Return the list of attribute names defined in the protocol."""
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
        """Build an index table for an attribute across one or more HDF5 files.

        Scans ``files`` to locate the dataset path for ``attr`` and records
        the per-file frame counts.  The returned :class:`LoadIndices` object
        supports indexing and iteration, allowing a subset of frames to be
        selected before passing the result to :meth:`load`.

        Args:
            files: Path, list of paths, or :class:`H5Files` object.
            attr: Attribute name as defined in the protocol.

        Returns:
            :class:`LoadIndices` carrying file locations, frame counts, and
            the attribute name.

        Example:
            Index the 'data' attribute across 'run_001.h5' and 'run_002.h5', then load
            frames 10 to 19:

            >>> indices = handler.indices(['run_001.h5', 'run_002.h5'], 'data')
            >>> frames = handler.load(indices[10:20])
        """
        if isinstance(files, (str, list)):
            files = H5Files(files)
        return self.protocol.read_indices(attr, files.visit_files())

    @overload
    def load(self, idxs: AnyLoadIndices, *, ss_idxs: Indices=slice(None),
             fs_idxs: Indices=slice(None), processes: int=1, verbose: bool=True,
             xp: ArrayNamespace[CPArray]) -> CPArray: ...

    @overload
    def load(self, idxs: AnyLoadIndices, *, ss_idxs: Indices=slice(None),
             fs_idxs: Indices=slice(None), processes: int=1, verbose: bool=True,
             xp: ArrayNamespace[JaxArray]) -> JaxArray: ...

    @overload
    def load(self, idxs: AnyLoadIndices, *, ss_idxs: Indices=slice(None),
             fs_idxs: Indices=slice(None), processes: int=1, verbose: bool=True,
             xp: ArrayNamespace[NDArray]) -> NDArray: ...

    @overload
    def load(self, idxs: AnyLoadIndices, *, ss_idxs: Indices=slice(None),
             fs_idxs: Indices=slice(None), processes: int=1, verbose: bool=True) -> NDArray: ...

    def load(self, idxs: AnyLoadIndices, *, ss_idxs: Indices=slice(None),
             fs_idxs: Indices=slice(None), processes: int=1, verbose: bool=True,
             xp: AnyNamespace=NumPy) -> Array:
        """Load a data attribute from the files.

        Args:
            idxs: Index table returned by :meth:`indices`; carries both the
                file locations and the attribute name.
            ss_idxs: Slow-scan (row) pixel indices for ROI selection.
            fs_idxs: Fast-scan (column) pixel indices for ROI selection.
            processes: Number of parallel worker processes.
            verbose: Show a progress bar when ``True``.

        Raises:
            ValueError: If the attribute kind is not defined in the protocol.

        Returns:
            Array of loaded data with shape ``(n_frames, *frame_shape)``.

        Example:
            Load the first 100 frames of the 'data' attribute from 'run_001.h5':

            >>> indices = handler.indices('run_001.h5', 'data')
            >>> frames = handler.load(indices[:100])
        """
        kind = self.protocol.get_kind(idxs.attr)

        if kind == Kinds.no_kind:
            raise ValueError(f'Invalid attribute: {idxs.attr:s}')

        reader = H5Reader(self.protocol)

        if len(idxs) == 0:
            return xp.array([])

        if kind in (Kinds.stack, Kinds.frame):
            return reader.load_stack(indices=idxs, processes=processes, ss_idxs=ss_idxs,
                                     fs_idxs=fs_idxs, verbose=verbose, xp=xp)
        if kind == Kinds.scalar:
            return reader.load_sequence(idxs, False, xp)
        if kind == Kinds.sequence:
            return reader.load_sequence(idxs, verbose, xp)

        raise ValueError("Wrong kind: " + str(kind))

    def save(self, attr: Attribute, data: Array, file: h5py.File, mode: str='overwrite',
             chunk: Shape | None=None, idxs: Indices | None=None):
        """Save a data array for an attribute into an open HDF5 file.

        Args:
            attr: Attribute name as defined in the protocol.
            data: Array to write.
            file: Open :class:`h5py.File` object to write into.
            mode: Dataset write mode:

                * ``'overwrite'`` — replace any existing dataset (default).
                * ``'append'`` — extend an existing dataset along axis 0.
                * ``'insert'`` — write into specific frame positions given by
                  ``idxs``.

            chunk: HDF5 chunk shape.  Inferred from ``data`` when ``None``.
            idxs: Frame positions to write into; required when
                ``mode='insert'``.

        Raises:
            ValueError: If the attribute kind is not defined in the protocol,
                the file is read-only, or ``idxs`` is missing for
                ``mode='insert'``.

        Example:
            Write a stack of frames to the 'data' attribute in 'output.h5' using a
            protocol read from a JSON file:

            >>> with h5py.File('output.h5', 'w') as f:
            ...     handler.save('data', frames, f)
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
    """Load data attributes from HDF5 files.

    Args:
        filenames: Path or list of paths to the HDF5 files.
        handler: :class:`H5Handler` that resolves attribute paths and loading
            behaviour.
        *attributes: Attribute names to load.  All attributes defined in the
            protocol are loaded when none are supplied.
        indices: Frame or pixel indices to select.  Pass a bare integer
            sequence to select frames; pass a 3-tuple
            ``(frames, ss_idxs, fs_idxs)`` to additionally restrict the
            slow-scan and fast-scan pixel ROI.
        processes: Number of parallel worker processes.
        verbose: Show a progress bar when ``True``.

    Returns:
        Dictionary mapping each attribute name to its loaded array.

    Raises:
        ValueError: If an attribute is not defined in the protocol or not
            found in the files.

    Example:
        Read the first 100 frames of the 'data' and 'mask' attributes from
        'run_001.h5':

        >>> arrays = read_hdf('run_001.h5', handler, 'data', 'mask',
        ...                   indices=slice(0, 100))
        >>> arrays['data'].shape
        (100, 512, 512)
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

        data = handler.load(idxs, ss_idxs=ss_idxs, fs_idxs=fs_idxs,
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
    """Save data attributes from a container to an HDF5 file.

    Args:
        container: Data container whose attributes are written.
        filename: Path to the output HDF5 file.
        handler: :class:`H5Handler` that resolves attribute paths and writing
            behaviour.
        *attributes: Attribute names to save.  All non-empty attributes in
            the container are saved when none are supplied.
        mode: Dataset write mode:

            * ``'overwrite'`` — replace any existing dataset (default).
            * ``'append'`` — extend an existing dataset along axis 0.
            * ``'insert'`` — write into specific frame positions given by
              ``indices``.

        file_mode: h5py file open mode (``'w'``, ``'r+'``, ``'a'``, …).
        indices: Frame positions to write into; required when
            ``mode='insert'``.

    Example:
        Save the 'data' and 'mask' attributes from a container to 'output.h5':

        >>> write_hdf(cryst_data, 'output.h5', handler, 'data', 'mask')
    """
    if not attributes:
        attributes = tuple(container.contents())

    with h5py.File(filename, file_mode) as file:
        for attr in attributes:
            data = getattr(container, attr)
            if not container.is_empty(data):
                handler.save(attr, data, file, mode=mode, idxs=indices)
