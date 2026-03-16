from . import bresenham, index, label, median, streak_finder, test

try:
    from . import cuda_draw_lines, cuda_label, cuda_median
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    cuda_draw_lines = None
    cuda_label = None
    cuda_median = None

# Expose CUDA availability at the package level
__all__ = ['CUDA_AVAILABLE']
