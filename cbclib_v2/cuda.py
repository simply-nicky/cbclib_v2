"""Configure CUDA allocator policy for cbclib, CuPy, and JAX.

This module is intentionally import-light so it can be used at the top of a
notebook before JAX or CuPy initialize their GPU runtimes.
"""
from __future__ import annotations

import importlib
import os
import re
import sys
import warnings
from typing import Any, Literal, TypedDict

Allocator = Literal["default", "cuda_malloc_async"]

__all__ = [
    "Allocator",
    "AllocatorConfig",
    "get_allocator_config",
    "set_allocator",
    "set_cuda_allocator",
    "set_cupy_allocator",
    "set_cupy_limit",
    "set_jax_allocator",
    "set_jax_limit",
]

_CUDA_ALLOCATOR_ENV = "CBCLIB_CUDA_ALLOCATOR"
_JAX_ALLOCATOR_ENV = "TF_GPU_ALLOCATOR"
_JAX_ASYNC_PREALLOC_ENV = "TF_CUDA_MALLOC_ASYNC_SUPPORTED_PREALLOC"
_JAX_FRACTION_ENV = "XLA_PYTHON_CLIENT_MEM_FRACTION"

_ALLOCATORS = {"default", "cuda_malloc_async"}
_MISSING = object()
_UNSET = object()

_cuda_allocator: Allocator = "default"
_cupy_allocator: Allocator = "default"
_jax_allocator: Allocator = "default"
_cupy_async_pool: Any | None = None
_managed_env: dict[str, str | object] = {}
_cupy_limit: int | float | None = None
_jax_limit: float | None = None

class AllocatorConfig(TypedDict):
    cuda_allocator: Allocator
    cupy_allocator: Allocator
    jax_allocator: Allocator
    cupy_limit: int | float | None
    jax_limit: float | None
    jax_initialized: bool

def set_allocator(allocator: Allocator, *, strict: bool = False) -> None:
    """Set cbclib C++, CuPy, and JAX allocator modes together.

    Args:
        allocator: Allocator mode to apply to all supported GPU backends.
        strict: If ``True``, raise when a backend cannot be configured.
            Otherwise issue a warning and keep the closest safe fallback.
    """
    set_cuda_allocator(allocator, strict=strict)
    set_cupy_allocator(allocator, strict=strict)
    set_jax_allocator(allocator, strict=strict)

def set_cuda_allocator(allocator: Allocator, *, strict: bool = False) -> None:
    """Select the allocator used by cbclib CUDA C++ temporary buffers.

    The C++ CUDA helpers read the selected mode at allocation time through
    ``CBCLIB_CUDA_ALLOCATOR``.

    Args:
        allocator: ``"default"`` for ``cudaMalloc`` / ``cudaFree`` or
            ``"cuda_malloc_async"`` for stream-ordered allocation when
            supported by the CUDA runtime.
        strict: Accepted for API symmetry; currently unused by this backend.
    """
    global _cuda_allocator
    allocator = _validate_allocator(allocator)
    _cuda_allocator = allocator
    if allocator == "cuda_malloc_async":
        _set_managed_env(_CUDA_ALLOCATOR_ENV, allocator)
    elif _CUDA_ALLOCATOR_ENV in _managed_env:
        _restore_managed_env(_CUDA_ALLOCATOR_ENV)
    elif os.environ.get(_CUDA_ALLOCATOR_ENV) == "cuda_malloc_async":
        os.environ[_CUDA_ALLOCATOR_ENV] = "default"

def set_cupy_allocator(allocator: Allocator, *, strict: bool = False) -> None:
    """Select CuPy's GPU allocator.

    CuPy is imported lazily so this function can be called from a notebook
    setup cell before the rest of cbclib is imported.

    Args:
        allocator: ``"default"`` for CuPy's default pool or
            ``"cuda_malloc_async"`` for ``cupy.cuda.MemoryAsyncPool``.
        strict: If ``True``, raise when CuPy is unavailable or async setup
            fails. Otherwise warn and fall back to the default allocator.
    """
    global _cupy_allocator, _cupy_async_pool, _cupy_limit
    allocator = _validate_allocator(allocator)

    try:
        import cupy as cp
    except ImportError as exc:
        if allocator == "cuda_malloc_async":
            _handle_failure("CuPy is not available; cannot enable cuda_malloc_async for CuPy.",
                            strict, exc)
        _cupy_allocator = allocator
        return

    if allocator == "cuda_malloc_async":
        if _cupy_limit is not None:
            _handle_failure("Clearing CuPy memory limit because cuda_malloc_async uses "
                            "CUDA driver-managed pools.", strict)
            _cupy_limit = None
        try:
            _cupy_async_pool = cp.cuda.MemoryAsyncPool()
            cp.cuda.set_allocator(_cupy_async_pool.malloc)
        except Exception as exc:  # pragma: no cover - depends on CUDA runtime support
            _cupy_async_pool = None
            _handle_failure("CuPy cuda_malloc_async allocator setup failed; using default "
                            "CuPy allocator.", strict, exc)
            _set_cupy_default_allocator(cp)
            _cupy_allocator = "default"
            return
    else:
        _cupy_async_pool = None
        _set_cupy_default_allocator(cp)

    _cupy_allocator = allocator

def set_jax_allocator(allocator: Allocator, *, strict: bool = False) -> None:
    """Select JAX/XLA's GPU allocator.

    JAX reads these settings while initializing XLA, so call this before
    JAX backend initialization.

    Args:
        allocator: ``"default"`` for JAX's normal allocator or
            ``"cuda_malloc_async"`` for XLA's experimental CUDA async
            allocator.
        strict: If ``True``, raise when JAX backend initialization has already
            happened. Otherwise issue a warning.
    """
    global _jax_allocator, _jax_limit
    allocator = _validate_allocator(allocator)
    _check_jax_not_initialized("set JAX allocator", strict)

    if allocator == "cuda_malloc_async":
        if _jax_limit is not None:
            _handle_failure("Clearing JAX memory limit because cuda_malloc_async uses "
                            "CUDA driver-managed pools.", strict)
            _jax_limit = None
            _restore_managed_env(_JAX_FRACTION_ENV)
        _set_managed_env(_JAX_ALLOCATOR_ENV, "cuda_malloc_async")
    else:
        if _JAX_ALLOCATOR_ENV in _managed_env:
            _restore_managed_env(_JAX_ALLOCATOR_ENV)
        elif os.environ.get(_JAX_ALLOCATOR_ENV) == "cuda_malloc_async":
            os.environ.pop(_JAX_ALLOCATOR_ENV, None)
        _restore_managed_env(_JAX_ASYNC_PREALLOC_ENV)

    _jax_allocator = allocator

def set_cupy_limit(limit: int | float | str | None, *, device: int | None = None) -> None:
    """Set CuPy's default memory-pool limit.

    Limits are intentionally rejected for ``cuda_malloc_async`` mode because that
    mode uses CUDA driver-managed pools rather than a hard cbclib/CuPy budget.

    Args:
        limit: Byte count, byte string such as ``"8GB"``, fraction such as
            ``0.5`` or ``"50%"``, or ``None`` to clear the pool limit.
        device: Optional CUDA device index whose pool limit should be changed.

    Raises:
        RuntimeError: If CuPy is currently configured for ``cuda_malloc_async``.
    """
    global _cupy_limit
    if _cupy_allocator == "cuda_malloc_async":
        raise RuntimeError("CuPy memory limits are only supported with the default allocator. "
                           "The cuda_malloc_async allocator uses CUDA driver-managed memory pools.")

    import cupy as cp

    size, fraction = _parse_limit(limit, allow_bytes=True)
    pool = cp.get_default_memory_pool()

    if device is None:
        pool.set_limit(size=size, fraction=fraction)
    else:
        with cp.cuda.Device(device):
            pool.set_limit(size=size, fraction=fraction)

    _cupy_limit = fraction if fraction is not None else size

def set_jax_limit(limit: float | str | None, *, strict: bool = False) -> None:
    """Set JAX's default allocator memory fraction before JAX initializes.

    Args:
        limit: Fraction such as ``0.5`` or ``"50%"``, or ``None`` to clear the
            cbclib-managed limit.
        strict: If ``True``, raise when JAX backend initialization has already
            happened. Otherwise issue a warning.

    Raises:
        RuntimeError: If JAX is currently configured for ``cuda_malloc_async``.
    """
    global _jax_limit
    if _jax_allocator == "cuda_malloc_async":
        raise RuntimeError("JAX memory limits are only supported with the default allocator. "
                           "The cuda_malloc_async allocator uses CUDA driver-managed memory pools.")

    _check_jax_not_initialized("set JAX memory limit", strict)
    _, fraction = _parse_limit(limit, allow_bytes=False)

    if fraction is None:
        _restore_managed_env(_JAX_FRACTION_ENV)
        _jax_limit = None
    else:
        _set_managed_env(_JAX_FRACTION_ENV, str(fraction))
        _jax_limit = fraction

def get_allocator_config() -> AllocatorConfig:
    """Return cbclib's requested allocator configuration.

    Returns:
        Current allocator modes, configured memory limits, and whether JAX has
        already initialized its backend client in this Python process.
    """
    return {
        "cuda_allocator": _cuda_allocator,
        "cupy_allocator": _cupy_allocator,
        "jax_allocator": _jax_allocator,
        "cupy_limit": _cupy_limit,
        "jax_limit": _jax_limit,
        "jax_initialized": _is_jax_backend_initialized(),
    }

def _validate_allocator(allocator: str) -> Allocator:
    if allocator not in _ALLOCATORS:
        raise ValueError("allocator must be 'default' or 'cuda_malloc_async'")
    return allocator  # type: ignore[return-value]

def _set_cupy_default_allocator(cp: Any) -> None:
    cp.cuda.set_allocator(cp.get_default_memory_pool().malloc)

def _check_jax_not_initialized(action: str, strict: bool) -> None:
    if _is_jax_backend_initialized():
        _handle_failure(f"Cannot reliably {action} after JAX backend initialization. "
                        "Restart the Python process or Jupyter kernel and configure "
                        "the allocator before JAX initializes its backend.", strict)

def _is_jax_backend_initialized() -> bool:
    if "jax" not in sys.modules:
        return False

    xla_bridge = sys.modules.get("jax._src.xla_bridge")
    try:
        if xla_bridge is None:
            xla_bridge = importlib.import_module("jax._src.xla_bridge")
    except ModuleNotFoundError:
        return False

    try:
        return bool(xla_bridge.backends_are_initialized())
    except Exception:
        # Conservative fallback: if JAX internals are importable but their
        # backend-init status cannot be queried reliably, assume initialization
        # may already have happened.
        return True

def _handle_failure(message: str, strict: bool, exc: BaseException | None = None) -> None:
    if strict:
        if exc is None:
            raise RuntimeError(message)
        raise RuntimeError(message) from exc
    warnings.warn(message, RuntimeWarning, stacklevel=3)

def _set_managed_env(name: str, value: str) -> None:
    if name not in _managed_env:
        _managed_env[name] = os.environ.get(name, _UNSET)
    os.environ[name] = value

def _restore_managed_env(name: str) -> None:
    previous = _managed_env.pop(name, _MISSING)
    if previous is _MISSING:
        return
    if previous is _UNSET:
        os.environ.pop(name, None)
    else:
        os.environ[name] = str(previous)

def _parse_limit(limit: int | float | str | None, *,
                 allow_bytes: bool) -> tuple[int | None, float | None]:
    if limit is None:
        return None, None

    if isinstance(limit, float):
        return None, _validate_fraction(limit)

    if isinstance(limit, int):
        if allow_bytes:
            if limit < 0:
                raise ValueError("limit must be non-negative")
            return limit, None
        return None, _validate_fraction(float(limit))

    if isinstance(limit, str):
        text = limit.strip()
        if text.endswith("%"):
            return None, _validate_fraction(float(text[:-1]) / 100.0)
        if allow_bytes:
            return _parse_bytes(text), None
        raise ValueError("JAX limits must be a fraction, for example 0.4 or '40%'.")

    raise TypeError("limit must be an int, float, str, or None")

def _validate_fraction(value: float) -> float:
    if not 0.0 <= value <= 1.0:
        raise ValueError("fractional limits must be in the range [0, 1]")
    return value

def _parse_bytes(value: str) -> int:
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*([kmgt]?i?b?)?", value, re.IGNORECASE)
    if match is None:
        raise ValueError("byte limits must look like '1024', '8GB', or '512MiB'")

    number = float(match.group(1))
    unit = (match.group(2) or "b").lower()
    factors = {
        "": 1,
        "b": 1,
        "k": 1000,
        "kb": 1000,
        "m": 1000 ** 2,
        "mb": 1000 ** 2,
        "g": 1000 ** 3,
        "gb": 1000 ** 3,
        "t": 1000 ** 4,
        "tb": 1000 ** 4,
        "ki": 1024,
        "kib": 1024,
        "mi": 1024 ** 2,
        "mib": 1024 ** 2,
        "gi": 1024 ** 3,
        "gib": 1024 ** 3,
        "ti": 1024 ** 4,
        "tib": 1024 ** 4,
    }
    return int(number * factors[unit])
