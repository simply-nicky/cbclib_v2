from . import bresenham, fft_functions, index, label, median, signal_proc, streak_finder, test

try:
    from . import cuda_functions
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    cuda_functions = None

# Expose CUDA availability at the package level
__all__ = ['CUDA_AVAILABLE']
