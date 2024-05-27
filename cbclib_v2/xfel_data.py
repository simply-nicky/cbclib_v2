from collections import OrderedDict, defaultdict
import fnmatch
from glob import iglob
import json
from multiprocessing import Pool
import signal
import sys
from operator import index
import os
import re
import resource
from tempfile import mkstemp
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple, Union, cast
from warnings import warn
from weakref import WeakValueDictionary
import h5py, h5py.h5o

import numpy as np

DATA_ROOT_DIR = os.environ.get('EXTRA_DATA_DATA_ROOT', '/gpfs/exfel/exp')
SCRATCH_ROOT_DIR = "/gpfs/exfel/exp/"

class FileStructureError(Exception):
    pass

class SourceNameError(KeyError):
    def __init__(self, source):
        self.source = source

    def __str__(self):
        return (
            f"This data has no source named {self.source:!r}.\n" \
            "See data.all_sources for available sources."
        )

class PropertyNameError(KeyError):
    def __init__(self, prop, source):
        self.prop = prop
        self.source = source

    def __str__(self):
        return f"No property {self.prop:!r} for source {self.source:!r}"

class TrainIDError(KeyError):
    def __init__(self, train_id):
        self.train_id = train_id

    def __str__(self):
        return f"Train ID {self.train_id:!r} not found in this data"

def available_cpu_cores():
    # This process may be restricted to a subset of the cores on the machine;
    # sched_getaffinity() tells us which on some Unix flavours (inc Linux)
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    else:
        # Fallback, inc on Windows
        ncpu = os.cpu_count() or 2
        return min(ncpu, 8)

def ignore_sigint():
    # Used in child processes to prevent them from receiving KeyboardInterrupt
    signal.signal(signal.SIGINT, signal.SIG_IGN)

def find_proposal(propno):
    """Find the proposal directory for a given proposal on Maxwell"""
    if '/' in propno:
        # Already passed a proposal directory
        return propno

    for d in iglob(os.path.join(DATA_ROOT_DIR, f'*/*/{propno}')):
        return d

    raise ValueError(f"Couldn't find proposal dir for {propno:!r}")

def same_run(*args) -> bool:
    """return True if arguments objects contain data from the same RUN

    arguments can be of type *DataCollection* or *SourceData*
    """
    # DataCollection union of format version = 0.5 (no run/proposal # in
    # files) is not considered a single run.
    proposal_nos = set()
    run_nos = set()
    for dc in args:
        md = dc.run_metadata() if dc.is_single_run else {}
        proposal_nos.add(md.get("proposalNumber", -1))
        run_nos.add(md.get("runNumber", -1))

    return (len(proposal_nos) == 1 and (-1 not in proposal_nos)
            and len(run_nos) == 1 and (-1 not in run_nos))

def atomic_dump(obj, path, **kwargs):
    """Write JSON to a file atomically

    This aims to avoid garbled files from multiple processes writing the same
    cache. It doesn't try to protect against e.g. sudden power failures, as
    forcing the OS to flush changes to disk may hurt performance.
    """
    dirname, basename = os.path.split(path)
    fd, tmp_filename = mkstemp(dir=dirname, prefix=basename)
    try:
        with open(fd, 'w') as f:
            json.dump(obj, f, **kwargs)
    except:
        os.unlink(tmp_filename)
        raise

    os.replace(tmp_filename, path)

class RunFilesMap:
    """Cached data about HDF5 files in a run directory

    Stores the train IDs and source names in each file, along with some
    metadata to check that the cache is still valid. The cached information
    can be stored in:

    - (run dir)/karabo_data_map.json
    - (proposal dir)/scratch/.karabo_data_maps/raw_r0032.json
    """
    cache_file = None

    def __init__(self, directory):
        self.directory = os.path.abspath(directory)
        self.dir_stat = os.stat(self.directory)
        self.files_data = {}

        self.candidate_paths = self.map_paths_for_run(directory)

        self.load()

    def map_paths_for_run(self, directory):
        paths = [os.path.join(directory, 'karabo_data_map.json')]
        # After resolving symlinks, data on Maxwell is stored in either
        # GPFS, e.g. /gpfs/exfel/d/proc/SCS/201901/p002212  or
        # dCache, e.g. /pnfs/xfel.eu/exfel/archive/XFEL/raw/SCS/201901/p002212
        # On the online cluster the resolved path stay:
        #   /gpfs/exfel/exp/inst/cycle/prop/(raw|proc)/run
        maxwell_match = re.match(
            #     raw/proc  instr  cycle prop   run
            r'.+/(raw|proc)/(\w+)/(\w+)/(p\d+)/(r\d+)/?$',
            os.path.realpath(directory)
        )
        online_match = re.match(
            #     instr cycle prop   raw/proc   run
            r'^.+/(\w+)/(\w+)/(p\d+)/(raw|proc)/(r\d+)/?$',
            os.path.realpath(directory)
        )

        if maxwell_match or online_match:
            if maxwell_match:
                raw_proc, instr, cycle, prop, run_nr = maxwell_match.groups()
            if online_match:
                instr, cycle, prop, raw_proc, run_nr = online_match.groups()

            fname = f'{raw_proc}_{run_nr}.json'
            prop_scratch = os.path.join(
                SCRATCH_ROOT_DIR, instr, cycle, prop, 'scratch'
            )
            if os.path.isdir(prop_scratch):
                paths.append(
                    os.path.join(prop_scratch, '.karabo_data_maps', fname)
                )

        return paths

    def load(self):
        """Load the cached data

        This skips over invalid cache entries(based on the file's size & mtime).
        """
        loaded_data = []

        for path in self.candidate_paths:
            try:
                with open(path) as f:
                    loaded_data = json.load(f)

                self.cache_file = path
                break
            except (FileNotFoundError, PermissionError, json.JSONDecodeError,):
                pass

        for info in loaded_data:
            filename = info['filename']
            try:
                st = os.stat(os.path.join(self.directory, filename))
            except OSError:
                continue
            if (st.st_mtime == info['mtime']) and (st.st_size == info['size']):
                self.files_data[filename] = info

    def is_my_directory(self, dir_path):
        return os.path.samestat(os.stat(dir_path), self.dir_stat)

    def _cache_valid(self, fname):
        # The cache is invalid (needs to be written out) if the file is not in
        # files_data (which it won't be if the size or mtime don't match - see
        # load()), or if suspect_train_indices is missing. This was added after
        # we started making cache files, so we want to add it to existing caches.
        return 'suspect_train_indices' in self.files_data.get(fname, {})

    def save(self, files):
        """Save the cache if needed

        This skips writing the cache out if all the data files already have
        valid cache entries. It also silences permission errors from writing
        the cache file.
        """
        need_save = False

        for file_access in files:
            dirname, fname = os.path.split(os.path.abspath(file_access.filename))
            if self.is_my_directory(dirname) and not self._cache_valid(fname):
                need_save = True

                # It's possible that the file we opened has been replaced by a
                # new one before this runs. If possible, use the stat FileAccess got
                # from the file descriptor, which will always be accurate.
                # Stat-ing the filename will almost always work as a fallback.
                if isinstance(file_access.metadata_fstat, os.stat_result):
                    st = file_access.metadata_fstat
                else:
                    st = os.stat(file_access.filename)

                self.files_data[fname] = {
                    'filename': fname,
                    'mtime': st.st_mtime,
                    'size': st.st_size,
                    'train_ids': [int(t) for t in file_access.train_ids],
                    'control_sources': sorted(file_access.control_sources),
                    'instrument_sources': sorted(file_access.instrument_sources),
                    'suspect_train_indices': [
                        int(i) for i in (~file_access.validity_flag).nonzero()[0]
                    ],
                }

        if need_save:
            save_data = [info for (_, info) in sorted(self.files_data.items())]
            for path in self.candidate_paths:
                try:
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    atomic_dump(save_data, path, indent=2)
                except PermissionError:
                    continue
                else:
                    return

file_access_registry = WeakValueDictionary()

class OpenFilesLimiter(object):
    """
    Working with FileAccess, keep the number of opened HDF5 files
    under the given limit by closing files accessed longest time ago.
    """
    def __init__(self, maxfiles=128):
        self._maxfiles = maxfiles
        # We don't use the values, but OrderedDict is a handy as a queue
        # with efficient removal of entries by key.
        self._cache = OrderedDict()

    @property
    def maxfiles(self):
        return self._maxfiles

    @maxfiles.setter
    def maxfiles(self, maxfiles):
        """Set the new file limit and closes files over the limit"""
        self._maxfiles = maxfiles
        self.close_old_files()

    def _check_files(self):
        # Discard entries from self._cache if their FileAccess no longer exists
        self._cache = OrderedDict.fromkeys(
            path for path in self._cache if path in file_access_registry
        )

    def n_open_files(self):
        self._check_files()
        return len(self._cache)

    def close_old_files(self):
        if len(self._cache) <= self.maxfiles:
            return

        # Now check how many paths still have an existing FileAccess object
        n = self.n_open_files()
        while n > self.maxfiles:
            path, _ = self._cache.popitem(last=False)
            file_access = file_access_registry.get(path, None)
            if file_access is not None:
                file_access.close()
            n -= 1

    def touch(self, filename):
        """
        Add/move the touched file to the end of the `cache`.

        If adding a new file takes it over the limit of open files, another file
        will be closed.

        For use of the file cache, FileAccess should use `touch(filename)` every time
        it provides the underlying instance of `h5py.File` for reading.
        """
        if filename in self._cache:
            self._cache.move_to_end(filename)
        else:
            self._cache[filename] = None
            self.close_old_files()

    def closed(self, filename):
        """Discard a closed file from the cache"""
        self._cache.pop(filename, None)

def init_open_files_limiter():
    # Raise the limit for open files (1024 -> 4096 on Maxwell)
    nofile = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (nofile[1], nofile[1]))
    maxfiles = nofile[1] // 2
    return OpenFilesLimiter(maxfiles)

open_files_limiter = init_open_files_limiter()

class MetaFileAccess(type):
    # Override regular instance creation to check in the registry first.
    # Defining __new__ on the class is not enough, because an object retrieved
    # from the registry that way will have its __init__ run again.
    def __call__(cls, filename, _cache_info=None):
        filename = os.path.abspath(filename)
        inst = file_access_registry.get(filename, None)
        if inst is None:
            inst = file_access_registry[filename] = type.__call__(
                cls, filename, _cache_info
            )

        return inst

class FileAccess(metaclass=MetaFileAccess):
    """Access an EuXFEL HDF5 file.

    This does not necessarily keep the real file open, but opens it on demand.
    It assumes that the file is not changing on disk while this object exists.

    Parameters
    ----------
    filename: str
        A path to an HDF5 file
    """
    _file = None
    _format_version = None
    _path_infos = None
    _filename_infos = None
    metadata_fstat = None

    # Regular expressions to extract path and filename information for HDF5
    # files saved on the EuXFEL computing infrastructure.
    euxfel_path_pattern = re.compile(
        # A path may have three different prefixes depending on the storage location.
        r'\/(gpfs\/exfel\/exp|gpfs\/exfel\/d|pnfs\/xfel.eu\/exfel\/archive\/XFEL)'

        # The first prefix above uses the pattern <instrument>/<cycle>/<proposal>/<class>/<run>,
        # the other two use <class>/<instrument>/<cycle>/<proposal>
        r'\/(\w+)\/(\d{6}|\w+)\/(\d{6}|p\d{6})\/(p\d{6}|[a-z]+)\/r\d{4}')

    euxfel_filename_pattern = re.compile(r'([A-Z]+)-R\d{4}-(\w+)-S(\d{5}).h5')

    def __init__(self, filename: str, _cache_info=None):
        self.filename = os.path.abspath(filename)

        if _cache_info:
            self.train_ids = _cache_info['train_ids']
            self.control_sources = _cache_info['control_sources']
            self.instrument_sources = _cache_info['instrument_sources']
            self.validity_flag = _cache_info.get('flag', None)
        else:
            try:
                tid_data = cast(h5py.Dataset, self.file['INDEX/trainId'])[:]
            except KeyError as exc:
                raise FileStructureError('INDEX/trainId dataset not found') from exc

            self.train_ids = tid_data[tid_data != 0]

            self.control_sources, self.instrument_sources = self._read_data_sources()

            self.validity_flag = None

        if self.validity_flag is None:
            if self.format_version == '0.5':
                self.validity_flag = self._guess_valid_trains()
            else:
                dset = cast(h5py.Dataset, self.file['INDEX/flag'])
                self.validity_flag = np.asarray(dset[:len(self.train_ids)], dtype=bool)

                if self.format_version == '1.1':
                    # File format version 1.1 changed the semantics of
                    # INDEX/flag from a boolean flag to an index, with
                    # the time server device being hardcoded to occur
                    # at index 0. Inverting the flag after the boolean
                    # cast above restores compatibility with format
                    # version 1.0, with any "invalid" train having an
                    # index >= 1, thus being casted to True and inverted
                    # to False. Format 1.2 restored the 1.0 semantics.
                    np.logical_not(self.validity_flag, out=self.validity_flag)

                    warn(
                        'Train validation is not fully supported for data '
                        'format version 1.1. If you have issues accessing '
                        'these files, please contact da-support@xfel.eu.',
                        stacklevel=2
                    )

        if self._file is not None:
            # Store the stat of the file as it was when we read the metadata.
            # This is used by the run files map.
            self.metadata_fstat = os.stat(self.file.id.get_vfd_handle())

        # {(file, source, group): (firsts, counts)}
        self._index_cache = {}
        # {source: set(keys)}
        self._keys_cache = {}
        self._run_keys_cache = {}
        # {source: set(keys)} - including incomplete sets
        self._known_keys = defaultdict(set)

    @property
    def file(self) -> h5py.File:
        open_files_limiter.touch(self.filename)
        # Local var to avoid a race condition when another thread calls .close()
        file = self._file
        if file is None:
            file = self._file = h5py.File(self.filename, 'r')

        return file

    @property
    def format_version(self) -> str:
        if self._format_version is None:
            version_ds: Any = self.file.get('METADATA/dataFormatVersion')
            if version_ds is not None:
                self._format_version = version_ds[0].decode('ascii')
            else:
                # The first version of the file format had no version number.
                # Numbering started at 1.0, so we call the first version 0.5.
                self._format_version = '0.5'

        return self._format_version

    def _read_data_sources(self) -> Tuple[FrozenSet, FrozenSet]:
        control_sources, instrument_sources = set(), set()

        # The list of data sources moved in file format 1.0
        if self.format_version == '0.5':
            data_sources_path = 'METADATA/dataSourceId'
        else:
            data_sources_path = 'METADATA/dataSources/dataSourceId'

        try:
            data_sources_group = cast(h5py.Group, self.file[data_sources_path])
        except KeyError as exc:
            raise FileStructureError(f'{data_sources_path} not found') from exc

        for source in cast(h5py.Group, data_sources_group[:]):
            if not source:
                continue
            source = source.decode()
            category, _, h5_source = source.partition('/')
            if category == 'INSTRUMENT':
                device, _, chan_grp = h5_source.partition(':')
                chan, _, _ = chan_grp.partition('/')
                source = device + ':' + chan
                instrument_sources.add(source)
                # TODO: Do something with groups?
            elif category == 'CONTROL':
                control_sources.add(h5_source)
            elif category == 'Karabo_TimerServer':
                # Ignore virtual data source used only in file format
                # version 1.1 / pclayer-1.10.3-2.10.5.
                pass
            else:
                raise ValueError(f"Unknown data category {category}")

        return frozenset(control_sources), frozenset(instrument_sources)

    def _guess_valid_trains(self) -> np.ndarray:
        # File format version 1.0 includes a flag which is 0 if a train ID
        # didn't come from the time server. We use this to skip bad trains,
        # especially for AGIPD.
        # Older files don't have this flag, so this tries to estimate validity.
        # The goal is to have a monotonic sequence within the file with the
        # fewest trains skipped.
        train_ids = self.train_ids
        flag = np.ones_like(train_ids, dtype=bool)

        for ix in np.nonzero(train_ids[1:] <= train_ids[:-1])[0]:
            # train_ids[ix] >= train_ids[ix + 1]
            invalid_before = train_ids[:ix+1] >= train_ids[ix+1]
            invalid_after = train_ids[ix+1:] <= train_ids[ix]
            # Which side of the downward jump in train IDs would need fewer
            # train IDs invalidated?
            if np.count_nonzero(invalid_before) < np.count_nonzero(invalid_after):
                flag[:ix+1] &= ~invalid_before
            else:
                flag[ix+1:] &= ~invalid_after

        return flag

    def _read_index(self, source: str, group: str) -> Tuple[np.ndarray, np.ndarray]:
        """Get first index & count for a source.

        This is 'real' reading when the requested index is not in the cache.
        """
        ntrains = len(self.train_ids)
        ix_group = cast(h5py.Group, self.file[f'/INDEX/{source}/{group}'])
        firsts = cast(h5py.Dataset, ix_group['first'])[:ntrains]
        if 'count' in ix_group:
            counts = cast(h5py.Dataset, ix_group['count'])[:ntrains]
        else:
            status = cast(h5py.Dataset, ix_group['status'])[:ntrains]
            dset = cast(h5py.Dataset, ix_group['last'])
            counts = np.asarray((dset[:ntrains] - firsts + 1) * status,
                                dtype=np.uint64)
        return firsts, counts

    def get_keys(self, source: str) -> Set[str]:
        """Get keys for a given source name

        Keys are found by walking the HDF5 file, and cached for reuse.
        """
        try:
            return self._keys_cache[source]
        except KeyError:
            pass

        if source in self.control_sources:
            group = '/CONTROL/' + source
        elif source in self.instrument_sources:
            group = '/INSTRUMENT/' + source
        else:
            raise SourceNameError(source)

        res = set()

        def add_key(key, value):
            if isinstance(value, h5py.Dataset):
                res.add(key.replace('/', '.'))

        cast(h5py.Group, self.file[group]).visititems(add_key)
        self._keys_cache[source] = res
        return res

    def get_index(self, source: str, group: str) -> Tuple[np.ndarray, np.ndarray]:
        """Get first index & count for a source and for a specific train ID.

        Indices are cached; this appears to provide some performance benefit.
        """
        try:
            return self._index_cache[(source, group)]
        except KeyError:
            ix = self._read_index(source, group)
            self._index_cache[(source, group)] = ix
            return ix

    def has_source_key(self, source: str, key: str) -> bool:
        """Check if the given source and key exist in this file

        This doesn't scan for all the keys in the source, as .get_keys() does.
        """
        try:
            return key in self._keys_cache[source]
        except KeyError:
            pass

        if key in self._known_keys[source]:
            return True

        if source in self.control_sources:
            path = f"/CONTROL/{source}/{key.replace('.', '/')}"
        elif source in self.instrument_sources:
            path = f"/INSTRUMENT/{source}/{key.replace('.', '/')}"
        else:
            raise SourceNameError(source)

        # self.file.get(path, getclass=True) works, but is weirdly slow.
        # Checking like this is much faster.
        if (path in self.file) and isinstance(
                h5py.h5o.open(self.file.id, path.encode()), h5py.h5d.DatasetID
        ):
            self._known_keys[source].add(key)
            return True
        return False

class SourceData:
    """Data for one source

    Don't create this directly; get it from ``run[source]``.
    """
    _device_class = ...
    _first_source_file = ...

    def __init__(
            self, source, *, sel_keys, train_ids, files, section,
            is_single_run, inc_suspect_trains=True
    ):
        self.source = source
        self.sel_keys = sel_keys
        self.train_ids = train_ids
        self.files: List[FileAccess] = files
        self.section = section
        self.is_single_run = is_single_run
        self.inc_suspect_trains = inc_suspect_trains

    def __repr__(self):
        return f"<extra_data.SourceData source={self.source!r} " \
               f"for {len(self.train_ids)} trains>"

    @property
    def is_control(self) -> bool:
        """Whether this source is a control source."""
        return self.section == 'CONTROL'

    def _has_exact_key(self, key: str) -> bool:
        if self.sel_keys is not None:
            return key in self.sel_keys

        for f in self.files:
            return f.has_source_key(self.source, key)

        return False

    def __contains__(self, key: str) -> bool:
        res = self._has_exact_key(key)
        if (not res) and self.is_control:
            res = self._has_exact_key(key + '.value')
        return res

    def _glob_keys(self, pattern: str) -> Optional[set]:
        if self.is_control and not pattern.endswith(('.value', '*')):
            pattern += '.value'

        if pattern == '*':
            # When the selection refers to all keys, make sure this
            # is restricted to the current selection of keys for
            # this source.
            matched = self.sel_keys
        elif re.compile(r'([*?[])').search(pattern) is None:
            # Selecting a single key (no wildcards in pattern)
            # This check should be faster than scanning all keys:
            matched = {pattern} if pattern in self else set()
        else:
            key_re = re.compile(fnmatch.translate(pattern))
            matched = set(filter(key_re.match, self.keys()))

        if matched == set():
            raise PropertyNameError(pattern, self.source)

        return matched

    def keys(self, inc_timestamps: bool=True) -> Set[str]:
        """Get a set of key names for this source

        If you have used :meth:`select` to filter keys, only selected keys
        are returned.

        For control sources, each Karabo property is stored in the file as two
        keys, with '.value' and '.timestamp' suffixes. By default, these are
        given separately. Pass ``inc_timestamps=False`` to ignore timestamps and
        drop the '.value' suffix, giving names as used in Karabo.

        Only one file is used to find the keys. Within a run, all files should
        have the same keys for a given source, but if you use :meth:`union` to
        combine two runs where the source was configured differently, the
        result can be unpredictable.
        """
        if (not inc_timestamps) and self.is_control:
            return {k[:-6] for k in self.keys() if k.endswith('.value')}

        if self.sel_keys is not None:
            return self.sel_keys

        # The same source may be in multiple files, but this assumes it has
        # the same keys in all files that it appears in.
        for f in self.files:
            return f.get_keys(self.source)

        return set()

    def select_keys(self, keys) -> 'SourceData':
        """Select a subset of the keys in this source

        *keys* is either a single key name, a set of names, or a glob pattern
        (e.g. ``beamPosition.*``) matching a subset of keys. PropertyNameError
        is matched if a specified key does not exist.

        Returns a new :class:`SourceData` object.
        """
        if isinstance(keys, str):
            keys = self._glob_keys(keys)
        elif keys:
            # If a specific set of keys is selected, make sure
            # they are all valid, adding .value as needed for CONTROl keys.
            normed_keys = set()
            for key in keys:
                if self._has_exact_key(key):
                    normed_keys.add(key)
                elif self.is_control and self._has_exact_key(key + '.value'):
                    normed_keys.add(key + '.value')
                else:
                    raise PropertyNameError(key, self.source)
                keys = normed_keys
        else:
            # Catches both an empty set and None.
            # While the public API describes an empty set to
            # refer to all keys, the internal API actually uses
            # None for this case. This method is supposed to
            # accept both cases in order to natively support
            # passing a DataCollection as the selector. To keep
            # the conditions below clearer, any non-True value
            # is converted to None.
            keys = None

        if self.sel_keys is None:
            # Current keys are unspecific - use the specified keys
            new_keys = keys
        elif keys is None:
            # Current keys are specific but new selection is not - use current
            new_keys = self.sel_keys
        else:
            # Both the new and current keys are specific: take the intersection.
            # The check above should ensure this never results in an empty set,
            # but
            new_keys = self.sel_keys & keys
            assert new_keys

        return SourceData(
            self.source,
            sel_keys=new_keys,
            train_ids=self.train_ids,
            files=self.files,
            section=self.section,
            is_single_run=self.is_single_run,
            inc_suspect_trains=self.inc_suspect_trains
        )

class DataCollection:
    """An assemblage of data generated at European XFEL

    Data consists of *sources* which each have *keys*. It is further
    organised by *trains*, which are identified by train IDs.

    You normally get an instance of this class by calling :func:`H5File`
    for a single file or :func:`run_directory` for a directory.
    """
    def __init__(
            self, files, sources_data=None, train_ids=None,
            ctx_closes: bool=False, *, inc_suspect_trains: bool=True, is_single_run: bool=False,
    ):
        self.files = list(files)
        self.ctx_closes = ctx_closes
        self.inc_suspect_trains = inc_suspect_trains
        self.is_single_run = is_single_run

        if train_ids is None:
            if inc_suspect_trains:
                tid_sets = [f.train_ids for f in files]
            else:
                tid_sets = [f.valid_train_ids for f in files]
            train_ids = sorted(set().union(*tid_sets))
        self.train_ids = train_ids

        if sources_data is None:
            files_by_sources = defaultdict(list)
            for f in self.files:
                for source in f.control_sources:
                    files_by_sources[source, 'CONTROL'].append(f)
                for source in f.instrument_sources:
                    files_by_sources[source, 'INSTRUMENT'].append(f)
            sources_data = {
                src: SourceData(src,
                    sel_keys=None,
                    train_ids=train_ids,
                    files=files,
                    section=section,
                    is_single_run=self.is_single_run,
                    inc_suspect_trains=self.inc_suspect_trains,
                )
                for ((src, section), files) in files_by_sources.items()
            }
        self._sources_data = sources_data

        # Throw an error if we have conflicting data for the same source
        self._check_source_conflicts()

        self.control_sources = frozenset({
            name for (name, sd) in self._sources_data.items()
            if sd.section == 'CONTROL'
        })
        self.instrument_sources = frozenset({
            name for (name, sd) in self._sources_data.items()
            if sd.section == 'INSTRUMENT'
        })

    @staticmethod
    def _open_file(path, cache_info=None):
        try:
            fa = FileAccess(path, _cache_info=cache_info)
        except Exception as e:
            return os.path.basename(path), str(e)
        return os.path.basename(path), fa

    @classmethod
    def from_paths(
            cls, paths: Sequence[str], _files_map=None, *, inc_suspect_trains: bool=True,
            is_single_run: bool=False, parallelize: bool=True
    ) -> 'DataCollection':
        files = []
        uncached = []

        def handle_open_file_attempt(fname, fa):
            if isinstance(fa, FileAccess):
                files.append(fa)
            else:
                print(f"Skipping file {fname}", file=sys.stderr)
                print(f"  (error was: {fa})", file=sys.stderr)

        for path in paths:
            cache_info = _files_map and _files_map.get(path)
            if cache_info and ('flag' in cache_info):
                filename, fa = cls._open_file(path, cache_info=cache_info)
                handle_open_file_attempt(filename, fa)
            else:
                uncached.append(path)

        if uncached:
            # Open the files either in parallel or serially
            if parallelize:
                nproc = min(available_cpu_cores(), len(uncached))
                with Pool(processes=nproc, initializer=ignore_sigint) as pool:
                    for fname, fa in pool.imap_unordered(cls._open_file, uncached):
                        handle_open_file_attempt(fname, fa)
            else:
                for path in uncached:
                    handle_open_file_attempt(*cls._open_file(path))

        if not files:
            raise RuntimeError("All HDF5 files specified are unusable")

        return cls(
            files, ctx_closes=True, inc_suspect_trains=inc_suspect_trains,
            is_single_run=is_single_run,
        )

    @property
    def all_sources(self):
        return self.control_sources | self.instrument_sources

    def _check_source_conflicts(self):
        """Check for data with the same source and train ID in different files.
        """
        sources_with_conflicts = set()
        files_conflict_cache = {}

        def files_have_conflict(files):
            fset = frozenset({f.filename for f in files})
            if fset not in files_conflict_cache:
                if self.inc_suspect_trains:
                    tids = np.concatenate([f.train_ids for f in files])
                else:
                    tids = np.concatenate([f.valid_train_ids for f in files])
                files_conflict_cache[fset] = len(np.unique(tids)) != len(tids)
            return files_conflict_cache[fset]

        for source, srcdata in self._sources_data.items():
            if files_have_conflict(srcdata.files):
                sources_with_conflicts.add(source)

        if sources_with_conflicts:
            raise ValueError(
                f"{len(sources_with_conflicts)} sources have conflicting data " \
                f"(same train ID in different files): {', '.join(sources_with_conflicts)}"
            )

    def _expand_selection(self, selection: Any) -> Dict[str, Any]:
        if isinstance(selection, dict):
            # {source: {key1, key2}}
            # {source: set()} or {source: None} -> all keys for this source

            res = {}
            for source, in_keys in selection.items():
                if source not in self.all_sources:
                    raise SourceNameError(source)

                # Empty dict was accidentally allowed and tested; keep it
                # working just in case.
                if in_keys == {}:
                    in_keys = set()

                if in_keys is not None and not isinstance(in_keys, set):
                    raise TypeError(
                        f"keys in selection dict should be a set or None (got "
                        f"{in_keys!r})"
                    )

                res[source] = self._sources_data[source].select_keys(in_keys)

            return res

        return {}

    def _find_data(self, source, train_id) -> Tuple[Optional[FileAccess], Optional[int]]:
        for f in self._sources_data[source].files:
            ixs = (f.train_ids == train_id).nonzero()[0]
            if self.inc_suspect_trains and ixs.size > 0:
                return f, ixs[0]

            for ix in ixs:
                if cast(np.ndarray, f.validity_flag)[ix]:
                    return f, ix

        return None, None

    def _get_source_data(self, source: str) -> SourceData:
        if source not in self._sources_data:
            raise SourceNameError(source)

        return self._sources_data[source]

    def keys_for_source(self, source: str) -> Set[str]:
        """Get a set of key names for the given source

        If you have used :meth:`select` to filter keys, only selected keys
        are returned.

        Only one file is used to find the keys. Within a run, all files should
        have the same keys for a given source, but if you use :meth:`union` to
        combine two runs where the source was configured differently, the
        result can be unpredictable.
        """
        return self._get_source_data(source).keys()

    def select(self, seln_or_source_glob, key_glob: str='*', require_all: bool=False,
               require_any: bool=False, *, warn_drop_trains_frac: float=1.) -> 'DataCollection':
        """Select a subset of sources and keys from this data.

        There are four possible ways to select data:

        1. With two glob patterns (see below) for source and key names::

            # Select data in the image group for any detector sources
            sel = run.select('*/DET/*', 'image.*')

        2. With an iterable of source glob patterns, or (source, key) patterns::

            # Select image.data and image.mask for any detector sources
            sel = run.select([('*/DET/*', 'image.data'), ('*/DET/*', 'image.mask')])

            # Select & align undulator & XGM devices
            sel = run.select(['*XGM/*', 'MID_XTD1_UND/DOOCS/ENERGY'], require_all=True)

           Data is included if it matches any of the pattern pairs.

        3. With a dict of source names mapped to sets of key names
           (or empty sets to get all keys)::

            # Select image.data from one detector source, and all data from one XGM
            sel = run.select({'SPB_DET_AGIPD1M-1/DET/0CH0:xtdf': {'image.data'},
                              'SA1_XTD2_XGM/XGM/DOOCS': set()})

           Unlike the others, this option *doesn't* allow glob patterns.
           It's a more precise but less convenient option for code that knows
           exactly what sources and keys it needs.

        4. With an existing DataCollection, SourceData or KeyData object::

             # Select the same data contained in another DataCollection
             prev_run.select(sel)

        The optional `require_all` and `require_any` arguments restrict the
        trains to those for which all or at least one selected sources and
        keys have at least one data entry. By default, all trains remain selected.

        With `require_all=True`, a warning will be shown if there are no trains
        with all the required data. Setting `warn_drop_trains_frac` can show the
        same warning if there are a few remaining trains. This is a number 0-1
        representing the fraction of trains dropped for one source (default 1).

        Returns a new :class:`DataCollection` object for the selected data.

        .. note::
           'Glob' patterns may be familiar from selecting files in a Unix shell.
           ``*`` matches anything, so ``*/DET/*`` selects sources with "/DET/"
           anywhere in the name. There are several kinds of wildcard:

           - ``*``: anything
           - ``?``: any single character
           - ``[xyz]``: one character, "x", "y" or "z"
           - ``[0-9]``: one digit character
           - ``[!xyz]``: one character, *not* x, y or z

           Anything else in the pattern must match exactly. It's case-sensitive,
           so "x" does not match "X".
        """
        if isinstance(seln_or_source_glob, str):
            seln_or_source_glob = [(seln_or_source_glob, key_glob)]
        sources_data = self._expand_selection(seln_or_source_glob)

        if require_all or require_any:
            # Select only those trains for which all (require_all) or at
            # least one (require_any) selected sources and keys have
            # data, i.e. have a count > 0 in their respective INDEX
            # section.

            if require_all:
                train_ids = self.train_ids
            else:  # require_any
                # Empty list would be converted to np.float64 array.
                train_ids = np.empty(0, dtype=np.uint64)

            for source, srcdata in sources_data.items():
                n_trains_prev = len(train_ids)
                for group in srcdata.index_groups:
                    source_tids = np.empty(0, dtype=np.uint64)

                    for f in self._sources_data[source].files:
                        valid = True if self.inc_suspect_trains else np.asarray(f.validity_flag)
                        # Add the trains with data in each file.
                        _, counts = f.get_index(source, group)
                        source_tids = np.union1d(
                            f.train_ids[valid & (counts > 0)], source_tids
                        )

                    # Remove any trains previously selected, for which this
                    # selected source and key group has no data.

                    if require_all:
                        train_ids = np.intersect1d(train_ids, source_tids)
                    else:  # require_any
                        train_ids = np.union1d(train_ids, source_tids)

                n_drop = n_trains_prev - len(train_ids)
                if n_trains_prev and (n_drop / n_trains_prev) >= warn_drop_trains_frac:
                    warn(f"{n_drop}/{n_trains_prev} ({n_drop / n_trains_prev :.0%})"
                         f" trains dropped when filtering by {source}")

            train_ids = list(train_ids)  # Convert back to a list.
            sources_data = {
                src: srcdata._only_tids(train_ids)
                for src, srcdata in sources_data.items()
            }

        else:
            train_ids = self.train_ids

        files = set().union(*[sd.files for sd in sources_data.values()])

        return DataCollection(
            files, sources_data, train_ids=train_ids,
            inc_suspect_trains=self.inc_suspect_trains,
            is_single_run=self.is_single_run
        )

    def deselect(self, seln_or_source_glob, key_glob: str='*') -> 'DataCollection':
        """Select everything except the specified sources and keys.

        This takes the same arguments as :meth:`select`, but the sources and
        keys you specify are dropped from the selection.

        Returns a new :class:`DataCollection` object for the remaining data.
        """

        if isinstance(seln_or_source_glob, str):
            seln_or_source_glob = [(seln_or_source_glob, key_glob)]
        deselection = self._expand_selection(seln_or_source_glob)

        # Subtract deselection from selection on self
        sources_data = {}
        for source, srcdata in self._sources_data.items():
            if source not in deselection:
                sources_data[source] = srcdata
                continue

            desel_keys = deselection[source].sel_keys
            if desel_keys is None:
                continue  # Drop the entire source

            remaining_keys = srcdata.keys() - desel_keys

            if remaining_keys:
                sources_data[source] = srcdata.select_keys(remaining_keys)

        files = set().union(*[sd.files for sd in sources_data.values()])

        return DataCollection(
            files, sources_data=sources_data, train_ids=self.train_ids,
            inc_suspect_trains=self.inc_suspect_trains,
            is_single_run=self.is_single_run,
        )

    def train_from_id(
            self, train_id: int, devices: Optional[Union[Dict, List]]=None, *,
            flat_keys: bool=False, keep_dims: bool=False) -> Tuple[int, Dict]:
        """Get train data for specified train ID.

        Parameters
        ----------

        train_id: int
            The train ID
        devices: dict or list, optional
            Filter data by sources and keys.
            Refer to :meth:`select` for how to use this.
        flat_keys: bool
            False (default) returns a nested dict indexed by source and then key.
            True returns a flat dictionary indexed by (source, key) tuples.
        keep_dims: bool
            False (default) drops the first dimension when there is
            a single entry. True preserves this dimension.

        Returns
        -------

        tid : int
            The train ID of the returned train
        data : dict
            The data for this train, keyed by device name

        Raises
        ------
        KeyError
            if `train_id` is not found in the run.
        """
        if train_id not in self.train_ids:
            raise TrainIDError(train_id)

        if devices is not None:
            return self.select(devices).train_from_id(train_id)

        res = {}
        for source in self.control_sources:
            source_data = res[source] = {
                'metadata': {'source': source, 'timestamp.tid': train_id}
            }
            file, pos = self._find_data(source, train_id)
            if file is None:
                continue

            firsts, counts = file.get_index(source, '')
            first, count = firsts[pos], counts[pos]
            if not count:
                continue

            for key in self.keys_for_source(source):
                path = f"/CONTROL/{source}/{key.replace('.', '/')}"
                source_data[key] = cast(h5py.Dataset, file.file[path])[first]

        for source in self.instrument_sources:
            source_data = res[source] = {
                'metadata': {'source': source, 'timestamp.tid': train_id}
            }
            file, pos = self._find_data(source, train_id)
            if file is None:
                continue

            for key in self.keys_for_source(source):
                group = key.partition('.')[0]
                firsts, counts = file.get_index(source, group)
                first, count = firsts[pos], counts[pos]
                if not count:
                    continue

                path = f"/INSTRUMENT/{source}/{key.replace('.', '/')}"
                if count == 1 and not keep_dims:
                    source_data[key] = cast(h5py.Dataset, file.file[path])[first]
                else:
                    source_data[key] = cast(h5py.Dataset, file.file[path])[first : first + count]

        if flat_keys:
            # {src: {key: data}} -> {(src, key): data}
            res = {(src, key): v for src, source_data in res.items()
                   for (key, v) in source_data.items()}

        return train_id, res

    def union(self, *others) -> 'DataCollection':
        """Join the data in this collection with one or more others.

        This can be used to join multiple sources for the same trains,
        or to extend the same sources with data for further trains.
        The order of the datasets doesn't matter. Any aliases defined on
        the collections are combined as well unless their values conflict.

        Note that the trains for each source are unioned as well, such that
        ``run.train_ids == run[src].train_ids``.

        Returns a new :class:`DataCollection` object.
        """

        sources_data_multi = defaultdict(list)
        for dc in (self,) + others:
            for source, srcdata in dc._sources_data.items():
                sources_data_multi[source].append(srcdata)

        sources_data = {src: src_datas[0].union(*src_datas[1:])
                        for src, src_datas in sources_data_multi.items()}

        train_ids = sorted(set().union(*[sd.train_ids for sd in sources_data.values()]))
        # Update the internal list of train IDs for the sources
        for sd in sources_data.values():
            sd.train_ids = train_ids

        files = set().union(*[sd.files for sd in sources_data.values()])

        return DataCollection(
            files, sources_data=sources_data, train_ids=train_ids,
            inc_suspect_trains=self.inc_suspect_trains,
            is_single_run=same_run(self, *others),
        )

def run_directory(
        path: str, include: str='*', *, inc_suspect_trains: bool=True,
        parallelize: bool=True
    ) -> DataCollection:
    """Open data files from a 'run' at European XFEL.

    ::

        run = run_directory("/gpfs/exfel/exp/XMPL/201750/p700000/raw/r0001")

    A 'run' is a directory containing a number of HDF5 files with data from the
    same time period.

    Returns a :class:`DataCollection` object.

    Parameters
    ----------
    path: str
        Path to the run directory containing HDF5 files.
    include: str
        Wildcard string to filter data files.
    inc_suspect_trains: bool
        If False, suspect train IDs within a file are skipped.
        In newer files, trains where INDEX/flag are 0 are suspect. For older
        files which don't have this flag, out-of-sequence train IDs are suspect.
        If True (default), it tries to include these trains.
    parallelize: bool
        Enable or disable opening files in parallel. Particularly useful if
        creating child processes is not allowed (e.g. in a daemonized
        :class:`multiprocessing.Process`).
    """
    files = [f for f in os.listdir(path)
             if f.endswith('.h5') and (f.lower() != 'overview.h5')]
    files = [os.path.join(path, f) for f in fnmatch.filter(files, include)]
    sel_files = files
    if not sel_files:
        raise FileNotFoundError(
            f"No HDF5 files found in {path} with glob pattern {include}")

    files_map = RunFilesMap(path)
    d = DataCollection.from_paths(
        files, files_map, inc_suspect_trains=inc_suspect_trains,
        is_single_run=True, parallelize=parallelize
    )
    files_map.save(d.files)

    return d

def open_run(
    proposal: Union[str, int], run: Union[str, int], data: Union[str, Sequence[str]]='raw',
    include: str='*', inc_suspect_trains: bool=True, parallelize: bool=True,
) -> DataCollection:
    """Access EuXFEL data on the Maxwell cluster by proposal and run number.

    ::

        run = open_run(proposal=700000, run=1)

    Returns a :class:`DataCollection` object.

    Parameters
    ----------
    proposal: str, int
        A proposal number, such as 2012, '2012', 'p002012', or a path such as
        '/gpfs/exfel/exp/SPB/201701/p002012'.
    run: str, int
        A run number such as 243, '243' or 'r0243'.
    data: str or Sequence of str
        'raw', 'proc' (processed), or any other location relative to the
        proposal path with data per run to access. May also be 'all'
        (both 'raw' and 'proc') or a sequence of strings to load data from
        several locations, with later locations overwriting sources present
        in earlier ones.
    include: str
        Wildcard string to filter data files.
    inc_suspect_trains: bool
        If False, suspect train IDs within a file are skipped.
        In newer files, trains where INDEX/flag are 0 are suspect. For older
        files which don't have this flag, out-of-sequence train IDs are suspect.
        If True (default), it tries to include these trains.
    parallelize: bool
        Enable or disable opening files in parallel. Particularly useful if
        creating child processes is not allowed (e.g. in a daemonized
        :class:`multiprocessing.Process`).
    """
    if data == 'all':
        data = ['raw', 'proc']

    if isinstance(data, Sequence) and not isinstance(data, str):
        common_args = {'proposal': proposal, 'run': run, 'include': include,
                       'inc_suspect_trains': inc_suspect_trains,
                       'parallelize': parallelize}

        # Open the raw data
        base_dc = open_run(**common_args, data=data[0])

        for origin in data[1:]:
            try:
                # Attempt to open data at this origin, but this may not
                # exist.
                origin_dc = open_run(**common_args, data=origin)
            except FileNotFoundError:
                warn(f'No data available for this run at origin {origin}')
            else:
                # Deselect to those sources in the base not present in
                # this origin.
                base_extra = base_dc.deselect(
                    [(src, '*') for src
                    in base_dc.all_sources & origin_dc.all_sources])

                if base_extra.files:
                    # If base is not a subset of this origin, merge the
                    # "extra" base sources into the origin sources and
                    # re-enable is_single_run flag.
                    base_dc = origin_dc.union(base_extra)
                    base_dc.is_single_run = True
                else:
                    # If the sources we previously found are a subset of those
                    # in the latest origin, discard the previous data.
                    base_dc = origin_dc

        return base_dc

    if isinstance(proposal, str):
        if ('/' not in proposal) and not proposal.startswith('p'):
            proposal = 'p' + proposal.rjust(6, '0')
    else:
        # Allow integers, including numpy integers
        proposal = f'p{index(proposal):06d}'

    prop_dir = find_proposal(proposal)

    if isinstance(run, str):
        if run.startswith('r'):
            run = run[1:]
    else:
        run = index(run)  # Allow integers, including numpy integers
    run = 'r' + str(run).zfill(4)

    dc = run_directory(
        os.path.join(prop_dir, data, run), include=include,
        inc_suspect_trains=inc_suspect_trains, parallelize=parallelize)

    return dc

class StackView:
    """Limited array-like object holding detector data from several modules.

    Access is limited to either a single module at a time or all modules
    together, but this is enough to assemble detector images.
    """
    def __init__(self, data, nmodules, mod_shape, dtype, fillvalue,
                 stack_axis=-3):
        self._nmodules = nmodules
        self._data = data  # {modno: array}
        self.dtype = dtype
        self._fillvalue = fillvalue
        self._mod_shape = mod_shape
        self.ndim = len(mod_shape) + 1
        self._stack_axis = stack_axis
        if self._stack_axis < 0:
            self._stack_axis += self.ndim
        sax = self._stack_axis
        self.shape = mod_shape[:sax] + (nmodules,) + mod_shape[sax:]

    def asarray(self):
        """Copy this data into a real numpy array

        Don't do this until necessary - the point of using VirtualStack is to
        avoid copying the data unnecessarily.
        """
        start_shape = (self._nmodules,) + self._mod_shape
        arr = np.full(start_shape, self._fillvalue, dtype=self.dtype)
        for modno, data in self._data.items():
            arr[modno] = data
        return np.moveaxis(arr, 0, self._stack_axis)

def stack_detector_data(
    train: Dict, data: str, axis: int=-3, modules: int=16, fillvalue: Optional[float]=None,
    real_array: bool=True, *, pattern: str=r'/DET/(\d+)CH', starts_at: int=0,
):
    """Stack data from detector modules in a train.

    Parameters
    ----------
    train: dict
        Train data.
    data: str
        The path to the device parameter of the data you want to stack, e.g. 'image.data'.
    axis: int
        Array axis on which you wish to stack (default is -3).
    modules: int
        Number of modules composing a detector (default is 16).
    fillvalue: number
        Value to use in place of data for missing modules. The default is nan
        (not a number) for floating-point data, and 0 for integers.
    real_array: bool
        If True (default), copy the data together into a real numpy array.
        If False, avoid copying the data and return a limited array-like wrapper
        around the existing arrays. This is sufficient for assembling images
        using detector geometry, and allows better performance.
    pattern: str
        Regex to find the module number in source names. Should contain a group
        which can be converted to an integer. E.g. ``r'/DET/JNGFR(\\d+)'`` for
        one JUNGFRAU naming convention.
    starts_at: int
        By default, uses module numbers starting at 0 (e.g. 0-15 inclusive).
        If the numbering is e.g. 1-16 instead, pass starts_at=1. This is not
        automatic because the first or last module may be missing from the data.

    Returns
    -------
    combined: numpy.array
        Stacked data for requested data path.
    """

    if not train:
        raise ValueError("No data")

    dtypes, shapes, empty_mods = set(), set(), set()
    modno_arrays = {}
    for src in train:
        det_mod_match = re.search(pattern, src)
        if not det_mod_match:
            raise ValueError(f"Source {src!r} doesn't match pattern {pattern!r}")
        modno = int(det_mod_match.group(1)) - starts_at

        try:
            array = train[src][data]
        except KeyError:
            continue
        dtypes.add(array.dtype)
        shapes.add(array.shape)
        modno_arrays[modno] = array

    if len(dtypes) > 1:
        raise ValueError(f"Arrays have mismatched dtypes: {dtypes}")
    if len(shapes) > 1:
        s1, s2, *_ = sorted(shapes)
        if len(shapes) > 2 or (s1[0] != 0) or (s1[1:] != s2[1:]):
            raise ValueError(f"Arrays have mismatched shapes: {shapes}")
        empty_mods = {n for n, a in modno_arrays.items() if a.shape == s1}
        for modno in empty_mods:
            del modno_arrays[modno]
        shapes.remove(s1)
    if max(modno_arrays) >= modules:
        raise IndexError(
            f"Module {max(modno_arrays)} is out of range for a detector " \
            f"with {modules} modules"
        )

    dtype = dtypes.pop()
    shape = shapes.pop()

    if fillvalue is None:
        fillvalue = np.nan if dtype.kind == 'f' else 0
    fillvalue = dtype.type(fillvalue)  # check value compatibility with dtype

    stack = StackView(
        modno_arrays, modules, shape, dtype, fillvalue, stack_axis=axis
    )
    if real_array:
        return stack.asarray()

    return stack
