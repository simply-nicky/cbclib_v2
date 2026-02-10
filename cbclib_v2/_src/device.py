"""Device context management for CPU/CUDA backend dispatch.

This module implements Solution 6: Hybrid from cuda-solutions.md, providing a JAX-like
device context API with internal type-safe backend dispatch.

Usage:
    >>> from cbclib_v2 import device
    >>>
    >>> # Configure CPU device
    >>> cpu_dev = device.cpu(num_threads=8)
    >>> device.set_device(cpu_dev)
    >>>
    >>> # Or use context manager
    >>> with device.context(device.cuda()):
    ...     result = some_function(data)

Environment Variables:
    CBCLIB_DEVICE: Set default device at import time ('cpu' or 'cuda'). Default: 'cpu'.
    CBCLIB_CPU_NUM_THREADS: Set default CPU thread count. Default: 1.
"""
from contextlib import contextmanager
from threading import local
from typing import List, Literal, overload
import os

from .annotations import CuPy, CuPyNamespace, NumPy, NumPyNamespace
from .src import CUDA_AVAILABLE

DeviceType = Literal['cpu', 'gpu']

class CPUDevice:
    """CPU device with OpenMP parallelization.

    Attributes:
        num_threads: Number of OpenMP threads for parallel execution.

    Examples:
        >>> from cbclib_v2 import device
        >>> cpu = device.cpu(num_threads=8)
        >>> device.set_device(cpu)
    """

    def __init__(self, num_threads: int = 1):
        if not isinstance(num_threads, int) or num_threads < 1:
            raise ValueError(f"num_threads must be an integer >= 1, got {num_threads}")
        self.num_threads = num_threads

    @property
    def platform(self) -> DeviceType:
        return 'cpu'

    def __repr__(self) -> str:
        return f"CPUDevice(num_threads={self.num_threads})"

class CUDADevice:
    """CUDA GPU device.

    Examples:
        >>> from cbclib_v2 import device
        >>> gpu = device.cuda()
        >>> device.set_device(gpu)
    """

    @property
    def platform(self) -> DeviceType:
        return 'gpu'

    def __repr__(self) -> str:
        return "CUDADevice()"

Device = CPUDevice | CUDADevice

# Thread-local storage for current device
_thread_local = local()

# Default devices
_default_cpu = CPUDevice(num_threads=int(os.environ.get('CBCLIB_CPU_NUM_THREADS', '1')))
_default_cuda = CUDADevice()

def cpu(num_threads: int=1) -> CPUDevice:
    """Create a CPU device with specified thread count.

    Args:
        num_threads: Number of OpenMP threads (must be >= 1).

    Returns:
        CPUDevice instance.

    Examples:
        >>> from cbclib_v2 import device
        >>> cpu_dev = device.cpu(num_threads=8)
        >>> device.set_device(cpu_dev)
    """
    return CPUDevice(num_threads=num_threads)

def gpu() -> CUDADevice:
    """Create a CUDA device.

    Returns:
        CUDADevice instance.

    Raises:
        RuntimeError: If CUDA not available.

    Examples:
        >>> from cbclib_v2 import device
        >>> gpu_dev = device.cuda()
        >>> device.set_device(gpu_dev)
    """
    if not CUDA_AVAILABLE:
        raise RuntimeError("CUDA not available. Rebuild with CUDA support.")
    return CUDADevice()

def _get_default_device() -> Device:
    """Get default device from environment variable."""
    env_device = os.environ.get('CBCLIB_DEVICE', 'cpu').lower()
    if env_device == 'cuda':
        return _default_cuda if CUDA_AVAILABLE else _default_cpu
    return _default_cpu

def _init_device() -> None:
    """Initialize thread-local device state if not already set."""
    if not hasattr(_thread_local, 'device'):
        _thread_local.device = _get_default_device()

def get_device() -> Device:
    """Get the current device for the calling thread.

    Returns:
        Current device (CPUDevice or CUDADevice).

    Examples:
        >>> device.get_device()
        CPUDevice(num_threads=1)
        >>> device.set_device(device.cuda())
        >>> device.get_device()
        CUDADevice()
    """
    _init_device()
    return _thread_local.device

AnyDevice = Device | DeviceType

def to_device(device: AnyDevice) -> Device:
    """Convert input to a Device instance.

    Args:
        device: Device instance or string ('cpu' or 'cuda').

    Returns:
        Device instance (CPUDevice or CUDADevice).

    Raises:
        RuntimeError: If CUDA device requested but CUDA not available.

    Examples:
        >>> device.to_device('cpu')
        CPUDevice(num_threads=1)
        >>> device.to_device(device.cuda())
        CUDADevice()
    """
    if device == 'cpu':
        device = CPUDevice()
    if device == 'gpu':
        device = CUDADevice()

    if not isinstance(device, (CPUDevice, CUDADevice)):
        raise TypeError(f"Invalid device type: {type(device)}")

    if isinstance(device, CUDADevice) and not CUDA_AVAILABLE:
        raise RuntimeError("CUDA device requested but CUDA extension not available.")

    return device

def set_device(device: AnyDevice) -> None:
    """Set the current device for the calling thread.

    Args:
        device: Device instance (CPUDevice or CUDADevice).

    Raises:
        RuntimeError: If CUDA device requested but CUDA not available.

    Examples:
        >>> device.set_device(device.cpu(num_threads=8))
        >>> device.set_device(device.gpu())
    """
    _init_device()
    _thread_local.device = to_device(device)

@contextmanager
def context(device: AnyDevice):
    """Context manager for temporary device switching.

    Args:
        device: Device instance for the context.

    Yields:
        None

    Examples:
        >>> with device.context(device.cuda()):
        ...     result = compute_intensive_function(data)
        >>> # Back to previous device
    """
    old_device = get_device()
    set_device(device)
    try:
        yield
    finally:
        _thread_local.device = old_device

def devices() -> List[DeviceType]:
    """List available device types.

    Returns:
        List of available device types ('cpu' and optionally 'cuda').

    Examples:
        >>> device.devices()
        ['cpu', 'cuda']
    """
    if CUDA_AVAILABLE:
        return ['cpu', 'gpu']
    return ['cpu']

@overload
def default_array_api(device: Literal['cpu'] | CPUDevice) -> NumPyNamespace: ...

@overload
def default_array_api(device: Literal['gpu'] | CUDADevice) -> CuPyNamespace: ...

def default_array_api(device: AnyDevice) -> NumPyNamespace | CuPyNamespace:
    """Get default array API for the specified device.

    Args:
        device: Device instance or string ('cpu' or 'cuda').

    Returns:
        Array API corresponding to the device.
    """
    device = to_device(device)
    if device.platform == 'cpu':
        return NumPy
    if device.platform == 'gpu':
        if CuPy is None:
            raise RuntimeError("CuPy is not available for CUDA device." \
                               "Please install CuPy.")
        return CuPy

    raise ValueError(f"Unsupported device platform: {device.platform}")
