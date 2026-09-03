from . import bresenham, label, median, online_detector, streak_finder, test

try:
    from . import cuda_draw_lines, cuda_label, cuda_median, cuda_online_detector, cuda_streak_finder
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    cuda_draw_lines = None
    cuda_label = None
    cuda_median = None
    cuda_online_detector = None
    cuda_streak_finder = None

# Expose CUDA availability at the package level
__all__ = ['CUDA_AVAILABLE']
