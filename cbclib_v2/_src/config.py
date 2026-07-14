import threading
from threading import local

class CPUConfig:
    """Thread-local configuration for the C++ OpenMP extensions.

    Controls how many OpenMP threads the native image-processing kernels
    (median filter, streak finder, etc.) may use when operating on CPU
    (NumPy) arrays. JAX and CuPy manage their own parallelism and are
    unaffected by this setting.

    Can be used as a context manager to temporarily override the active
    thread count and restore the previous value on exit.

    Attributes:
        num_threads: Number of OpenMP threads to use. Must be >= 1.

    Example:
        Perform a median through the stack of frames using 4 threads,
        without affecting the global default:

        >>> from cbclib_v2 import CPUConfig
        >>> import cbclib_v2.ndimage as ndimage
        >>> with CPUConfig(num_threads=4):
        ...     whitefield = ndimage.median(frames, axis=0)
    """

    def __init__(self, num_threads: int = 1):
        self.num_threads = num_threads
        self._old : int | None = None

    @staticmethod
    def _validate_num_threads(n: int) -> int:
        if not isinstance(n, int) or n < 1:
            raise ValueError(f"num_threads must be an integer >= 1, got {n}")
        return n

    def __setattr__(self, name, value):
        if name == "num_threads":
            value = self._validate_num_threads(value)
        super().__setattr__(name, value)

    def __enter__(self):
        self._old = get_cpu_config().num_threads
        get_cpu_config().num_threads = self.num_threads
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._old is not None:
            get_cpu_config().num_threads = self._old
            self._old = None

    def effective_num_threads(self) -> int:
        """Return the thread count that will actually be used.

        Returns 1 when called from a multiprocessing pool worker or from a
        non-main thread to prevent nested parallelism.

        Returns:
            Effective number of OpenMP threads.
        """
        if in_cpu_pool_worker():
            return 1
        if threading.current_thread() is not threading.main_thread():
            return 1
        return self.num_threads

_default_num_threads = 1
_thread_local = local()

def set_cpu_pool_worker(is_pool: bool=True) -> None:
    """Mark whether the current process is running as a pool worker.

    Args:
        is_pool: ``True`` when the current process belongs to an outer
            multiprocessing pool.
    """
    _thread_local.is_pool = is_pool

def in_cpu_pool_worker() -> bool:
    """Return whether the current process belongs to an outer pool."""
    return bool(getattr(_thread_local, "is_pool", False))

def get_cpu_config() -> CPUConfig:
    """Return the active :class:`CPUConfig` for the current thread.

    Creates a default configuration with ``num_threads=1`` if none has been
    set yet for this thread.

    Returns:
        The thread-local :class:`CPUConfig` instance.
    """
    if not hasattr(_thread_local, "config"):
        _thread_local.config = CPUConfig(_default_num_threads)
    return getattr(_thread_local, "config",)

def set_cpu_config(num_threads: int) -> None:
    """Set the number of OpenMP threads for the current thread.

    Equivalent to ``get_cpu_config().num_threads = num_threads``.  Use
    :class:`CPUConfig` as a context manager instead when you need a temporary
    override.

    Args:
        num_threads: Number of threads to use. Must be an integer >= 1.
    """
    config = get_cpu_config()
    config.num_threads = num_threads

def reset_cpu_config() -> None:
    """Reset the CPU configuration for the current thread to its default.

    The next call to :func:`get_cpu_config` will create a fresh
    :class:`CPUConfig` with ``num_threads=1``.
    """
    if hasattr(_thread_local, "config"):
        delattr(_thread_local, "config")
