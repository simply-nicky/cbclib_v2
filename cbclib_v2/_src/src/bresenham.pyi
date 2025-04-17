from typing import Optional, Tuple
from ..annotations import IntArray, NDIntArray, NDRealArray, RealArray, Shape

def accumulate_lines(lines: RealArray, shape: Shape, in_idxs: IntArray, out_idxs: IntArray,
                     max_val: float=1.0, kernel: str='rectangular', in_overlap: str='sum',
                     out_overlap: str='sum', num_threads: int=1) -> NDRealArray:
    ...

def draw_lines(lines: RealArray, shape: Shape, idxs: Optional[IntArray]=None,
               max_val: float=1.0, kernel: str='rectangular', overlap: str='sum',
               num_threads: int=1) -> NDRealArray:
    """Draw thick lines with variable thickness and the antialiasing applied on a single frame.

    Args:
        lines : A dictionary of the detected lines. Each array of lines must have a shape of
            (`N`, 5), where `N` is the number of lines. Each line is comprised of 5 parameters
            as follows:

            * `[x0, y0]`, `[x1, y1]` : The coordinates of the line's ends.
            * `width` : Line's width.

        shape : Shape of the output array. All the lines outside the shape will be discarded.
        idxs : An array of indices that specify to what frame each of the lines belong.
        max_val : Maximum pixel value of a drawn line.
        kernel : Choose one of the supported kernel functions [Krn]_. The following kernels
            are available:

            * 'biweigth' : Quartic (biweight) kernel.
            * 'gaussian' : Gaussian kernel.
            * 'parabolic' : Epanechnikov (parabolic) kernel.
            * 'rectangular' : Uniform (rectangular) kernel.
            * 'triangular' : Triangular kernel.

        num_threads : Number of threads used in the calculations.

    Raises:
        ValueError : If any of `lines` dictionary values have an incompatible shape.

    Returns:
        Output array with the lines drawn.
    """
    ...

def write_lines(lines: RealArray, shape: Shape, idxs: Optional[IntArray]=None,
                max_val: float=1.0, kernel: str='rectangular', num_threads: int=1
                ) -> Tuple[NDIntArray, NDIntArray, NDRealArray]:
    """Return an array of rasterized thick lines indices and their corresponding pixel values.
    The lines are drawn with variable thickness and the antialiasing applied.

    Args:
        lines : A dictionary of the detected lines. Each array of lines must have a shape of
            (`N`, 5), where `N` is the number of lines. Each line is comprised of 5 parameters
            as follows:

            * `[x0, y0]`, `[x1, y1]` : The coordinates of the line's ends.
            * `width` : Line's width.

        shape : Shape of the image. All the lines outside the shape will be discarded.
        idxs : An array of indices that specify to what frame each of the lines belong.
        max_val : Maximum pixel value of a drawn line.
        kernel : Choose one of the supported kernel functions [Krn]_. The following kernels
            are available:

            * 'biweigth' : Quartic (biweight) kernel.
            * 'gaussian' : Gaussian kernel.
            * 'parabolic' : Epanechnikov (parabolic) kernel.
            * 'rectangular' : Uniform (rectangular) kernel.
            * 'triangular' : Triangular kernel.

        num_threads : Number of threads used in the calculations.

    Raises:
        ValueError : If `lines` has an incompatible shape.
        RuntimeError : If C backend exited with error.

    Returns:
        Output line indices.
    """
    ...
