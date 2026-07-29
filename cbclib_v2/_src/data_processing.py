""":class:`cbclib_v2.CrystData` and :class:`cbclib_v2.CrystMetadata` implement the
core data processing pipeline for convergent beam crystallography detector data,
covering bad-pixel masking, background subtraction, SNR computation, and diffraction
streak detection.

Raw detector frames are wrapped in :class:`cbclib_v2.CrystData`, which exposes methods
for bad-pixel masking, whitefield estimation, SNR computation, and launching streak
detectors. :class:`cbclib_v2.CrystMetadata` stores a reusable background model —
optionally decomposed into PCA principal components — that can be applied to new frame
batches.
"""
from __future__ import annotations
from math import prod
import os
from typing import Literal, Sequence, Tuple, cast
from dataclasses import dataclass, field
from weakref import ref
from typing_extensions import Self
import numpy as np
from .array_api import array_namespace, default_rng
from .crystfel import Assembler
from .cxi_protocol import H5Protocol, Kinds
from .data_container import DataContainer, list_indices
from .streak_finder import PatternStreakFinder, PeakLabels, Streaks as StreakResult
from .streaks import StackedStreaks, Streaks
from .annotations import (Array, ArrayLike, BoolArray, Indices, IntArray, MultiIndices, NumPy,
                          RealArray, ReferenceType, ROI, Shape)
from .functions import (LabelResult, RadialProfiles, Structure, center_of_mass, covariance_matrix,
                        ellipse_fit, label, line_fit, median, radial_profiles, robust_mean)

MaskMethod = Literal['all-bad', 'no-bad', 'range', 'snr']
MDMethod = Literal['median-poisson', 'robust-mean-scale', 'robust-mean-poisson']
STDMethod = Literal['poisson', 'robust-scale']
WFMethod = Literal['median', 'robust-mean', 'robust-mean-scale']

DATA_PROTOCOL = os.path.join(os.path.dirname(__file__), 'config/cryst_data.ini')
METADATA_PROTOCOL = os.path.join(os.path.dirname(__file__), 'config/cryst_metadata.ini')

class CrystBase(DataContainer):
    protocol    : H5Protocol

    @property
    def frame_shape(self) -> Shape:
        current, old = tuple(), tuple()
        for attr, data in self.contents().items():
            kind = self.protocol.get_kind(attr)
            if isinstance(data, Array):
                if kind == Kinds.frame:
                    current = data.shape
                elif kind == Kinds.stack:
                    current = data.shape[1:]
                else:
                    continue

                if old and current != old:
                    raise ValueError(f"Attribute {attr} has an incompatible shape: {current}")

            old = current

        if current:
            return current
        return (0,)

    @property
    def num_modules(self) -> int:
        return prod(self.frame_shape) // prod(self.frame_shape[-2:])

    def assemble(self: Self, assembler: Assembler) -> Self:
        """Assemble the detector modules data onto a single lab-frame images.

        Args:
            assembler: A :class:`~cbclib_v2.Assembler` object that knows how to
                assemble the stacked module data onto a single lab-frame image.

        Returns:
            A new container with all frame-like arrays assembled into a single
            frame.
        """
        assembled = {}
        for attr, data in self.contents().items():
            if self.protocol.get_kind(attr) in (Kinds.frame, Kinds.stack):
                assembled[attr] = assembler(data)
        return self.replace(**assembled)

    def crop(self: Self, roi: ROI) -> Self:
        cropped = {}
        for attr, data in self.contents().items():
            if self.protocol.get_kind(attr) in (Kinds.frame, Kinds.stack):
                cropped[attr] = data[..., roi[0]:roi[1], roi[2]:roi[3]]
        return self.replace(**cropped)

@dataclass
class LSQData(DataContainer):
    """Linear least-squares target and design matrix.

    Attributes:
        y: Per-frame target values, shape ``(n_frames, *frame_shape)``.
        W: Design matrix, shape ``(n_frames or 1, n_fields, *frame_shape)``.
    """
    y : RealArray
    W : RealArray

    def __post_init__(self):

        if (self.W.ndim != self.y.ndim + 1 or
            self.W.shape[0] not in (1, self.y.shape[0]) or
            self.W.shape[2:] != self.y.shape[1:]):
            raise ValueError('W must have a shape (n_frames or 1, n_fields, *frame_shape)')

        xp = self.__array_namespace__()
        self.y = xp.reshape(self.y, (self.y.shape[0], -1))
        self.W = xp.reshape(self.W, self.W.shape[:2] + (-1,))

    def apply_mask(self, mask: BoolArray) -> 'LSQData':
        """Return least-squares data with rejected entries zeroed.

        Args:
            mask: Per-frame acceptance mask, shape ``(n_frames, *frame_shape)``.

        Returns:
            Masked least-squares data with a frame-specific design matrix.
        """
        xp = self.__array_namespace__()
        mask = xp.reshape(mask, (mask.shape[0], -1))
        return self.replace(y=xp.where(mask, self.y, 0),
                            W=xp.where(mask[:, None, :], self.W, 0))

    def solve(self) -> RealArray:
        """Solve the masked joint least-squares systems.

        Returns:
            Per-frame field coefficients, shape ``(n_frames, n_fields)``.
        """
        xp = self.__array_namespace__()
        gram = xp.linalg.matmul(self.W, xp.permute_dims(self.W, (0, 2, 1)))
        rhs = xp.sum(self.W * self.y[:, None, :], axis=-1)
        return xp.linalg.matmul(xp.linalg.pinv(gram), rhs[..., None])[..., 0]

@dataclass
class PCAProjection(DataContainer):
    good_fields : Sequence[int] | IntArray
    projection  : RealArray

    def apply(self, metadata: CrystMetadata) -> RealArray:
        """Reconstruct per-frame whitefields from PCA projection coefficients.

        Computes :math:`\\bar{W} + \\sum_k c_{ik} e_k` for each frame
        :math:`i`, where :math:`c_{ik}` are the projection coefficients and
        :math:`e_k` are the eigen fields.

        Args:
            metadata: A :class:`CrystMetadata` containing the PCA decomposition
                to apply.

        Returns:
            Per-frame whitefield array, shape ``(N, *frame_shape)``.

        Example:
            Reconstruct per-frame backgrounds and attach them to new data:

            >>> proj = metadata.project(frames, n_iter=1)
            >>> whitefields = proj.apply(metadata)
            >>> data = metadata.to_data(frames, projection=proj)
        """
        xp = self.__array_namespace__()

        if metadata.is_empty(metadata.eigen_field):
            return xp.tensordot(self.projection, metadata.flatfield[None],
                                axes=((-1,), (0,)))

        fields = xp.tensordot(self.projection, metadata.eigen_field[self.good_fields],
                              axes=((-1,), (0,)))
        return metadata.flatfield + fields

@dataclass
class CrystMetadata(CrystBase):
    """Background model for CBC detector data.

    Stores the pixel mask, noise standard deviation, and one or more whitefield
    images estimated from background frames. Optionally holds a PCA decomposition
    of whitefield variability for per-frame dynamic background subtraction.

    The container can be saved to and loaded from HDF5 files using
    :func:`~cbclib_v2.write_hdf` and :func:`~cbclib_v2.read_hdf` with the
    default protocol returned by :meth:`default_protocol`.

    Attributes:
        eigen_field: Principal-component whitefield images,
            shape ``(N, *frame_shape)``. Populated by :meth:`pca`.
        eigen_value: Normalised eigenvalues (sum to 1) for each principal
            component. Populated by :meth:`pca`.
        flatfield: Mean background image, shape ``frame_shape``. Initialised
            as the mean of ``whitefields`` when not supplied explicitly.
        frames: Frame indices associated with the stored whitefields.
        mask: Bad-pixel mask, shape ``frame_shape``. ``True`` = good pixel.
        std: Per-pixel noise standard deviation, shape ``frame_shape``.
        whitefields: Stack of individual background estimates,
            shape ``(N, *frame_shape)``.

    Example:
        Merge background estimates, decompose with PCA, and apply per-frame
        dynamic background subtraction:

        >>> metadata = cbc.CrystMetadata.stack(meta_a, meta_b, meta_c)
        >>> metadata = metadata.pca()
        >>> proj = metadata.project(frames, n_iter=1)
        >>> data = metadata.to_data(frames, projection=proj)
        >>> data = data.update_snr(std_min=0.5)
    """
    eigen_field : RealArray = field(default_factory=lambda: np.array([]))
    eigen_value : RealArray = field(default_factory=lambda: np.array([]))
    flatfield   : RealArray = field(default_factory=lambda: np.array([]))
    frames      : IntArray  = field(default_factory=lambda: np.array([], dtype=int))
    mask        : BoolArray = field(default_factory=lambda: np.array([], dtype=bool))
    std         : RealArray = field(default_factory=lambda: np.array([]))
    whitefields : RealArray = field(default_factory=lambda: np.array([]))

    protocol    : H5Protocol = field(default_factory=lambda: H5Protocol.read(METADATA_PROTOCOL))

    def __post_init__(self) -> None:
        if not self.is_empty(self.mask):
            if not self.is_empty(self.std):
                self.std *= self.mask
            if not self.is_empty(self.whitefields):
                self.whitefields *= self.mask

        if self.is_empty(self.flatfield) and not self.is_empty(self.whitefields):
            self.flatfield = self.whitefields.mean(axis=0)

    def apply_mask(self, indices: MultiIndices) -> 'CrystMetadata':
        """Select detector pixels from every frame-like array.

        Args:
            indices: Integer indices, boolean mask, or slice selecting detector
                pixels.

        Returns:
            Metadata restricted to the selected detector pixels.
        """
        attributes = {}
        for attr, data in self.contents().items():
            if not isinstance(data, Array) or self.is_empty(data):
                continue

            kind = self.protocol.get_kind(attr)
            if kind == Kinds.frame:
                attributes[attr] = data[indices]
            elif kind == Kinds.stack:
                if isinstance(indices, tuple):
                    attributes[attr] = data[(...,) + indices]
                else:
                    attributes[attr] = data[..., indices]
        return self.replace(**attributes)

    @classmethod
    def default_protocol(cls) -> H5Protocol:
        """Return the built-in :class:`~cbclib_v2.H5Protocol` for this container.

        Used by :func:`~cbclib_v2.write_hdf` and :func:`~cbclib_v2.read_hdf`
        to resolve attribute names to HDF5 dataset paths.  The default
        protocol maps attributes to the following dataset paths:

        .. code-block:: text

            /entry/crystallography/mask          ← mask
            /entry/crystallography/std           ← std
            /entry/metadata/flatfield            ← flatfield
            /entry/metadata/whitefields          ← whitefields
            /entry/metadata/eigen_fields         ← eigen_field
            /entry/metadata/eigen_values         ← eigen_value

        Returns:
            The default :class:`~cbclib_v2.H5Protocol` read from the built-in
            INI configuration file.
        """
        return H5Protocol.read(METADATA_PROTOCOL)

    @classmethod
    def stack(cls, *metadata_containers: 'CrystMetadata') -> CrystMetadata:
        """Stack multiple :class:`CrystMetadata` objects into a single container.

        The mask is the element-wise AND of all individual masks. The standard
        deviation is the quadratic mean across containers. The individual
        whitefields are stacked along a new leading axis.

        Args:
            *metadata_containers: One or more :class:`CrystMetadata` objects that
                each contain a ``flatfield`` and ``std``.

        Raises:
            ValueError: If no containers are supplied.

        Returns:
            A new :class:`CrystMetadata` with combined ``mask``, ``std``, and
            stacked ``whitefields``.
        """
        if not metadata_containers:
            raise ValueError('At least one CrystMetadata container is required to stack')
        xp = array_namespace(*metadata_containers)
        mask, var = xp.ones(1, dtype=bool), xp.zeros(1)
        whitefields = []

        protocol = cls.default_protocol()
        for metadata in metadata_containers:
            mask = mask & metadata.mask
            var = var + metadata.std ** 2
            whitefields.append(metadata.flatfield)

        return cls(mask=mask, std=xp.sqrt(var / len(metadata_containers)),
                   whitefields=xp.stack(whitefields, axis=0), protocol=protocol)

    def to_data(self, data: RealArray, frames: IntArray | int | None=None,
                projection: PCAProjection | None=None) -> 'CrystData':
        """Attach this background model to a new array of detector frames.

        Creates a :class:`CrystData` container populated with ``mask`` and
        ``std`` from this model. When ``whitefield`` is omitted, ``flatfield``
        is used as a static background for every frame. When a per-frame
        ``whitefield`` stack is supplied (e.g. from :meth:`project`), it is
        stored as a frame-wise stack inside the returned container.

        Args:
            data: Raw detector data, shape ``(N, *frame_shape)`` or
                ``(*batch_shape, *frame_shape)``.
            frames: Integer frame indices. Inferred from the leading dimensions
                of ``data`` when ``None``.
            projection: A :class:`PCAProjection` containing the PCA decomposition
                to apply.

        Raises:
            ValueError: If ``projection`` is None and ``flatfield`` is absent.
            ValueError: If ``projection`` is supplied but the resulting ``whitefield``
                has a wrong size.

        Returns:
            A new :class:`CrystData` with ``data``, ``frames``, ``mask``, ``std``,
            and ``whitefield`` set.

        Example:
            Apply static and dynamic background subtraction:

            >>> data = metadata.to_data(frames)                       # static
            >>> proj = metadata.project(frames, n_iter=1)
            >>> data = metadata.to_data(frames, projection=proj)      # dynamic
        """
        xp = self.__array_namespace__()
        if frames is None:
            frames = xp.arange(prod(data.shape[:-len(self.frame_shape)]), dtype=int)
        elif not isinstance(frames, Array):
            frames = xp.array([frames,], dtype=int)
        data = xp.reshape(data, (frames.size,) + self.frame_shape)

        if projection is None:
            if self.is_empty(self.flatfield):
                raise ValueError('no flatfield in the container')
            return CrystData(data=data, frames=frames, mask=self.mask, std=self.std,
                             whitefield=self.flatfield)

        whitefield = projection.apply(self)
        if whitefield.size != data.size:
            raise ValueError(f'whitefield size {whitefield.size} must be equal to data size '
                             f'{data.size}')

        protocol = CrystData.default_protocol()
        protocol.kinds['whitefield'] = 'stack'
        result = CrystData(data=data, frames=frames, mask=self.mask, std=self.std,
                           whitefield=xp.reshape(whitefield, data.shape), protocol=protocol)
        result.mask &= xp.all(result.whitefield >= 0, axis=0)
        return result.apply_mask()

    def pca(self) -> 'CrystMetadata':
        """Decompose whitefield variability into principal components.

        Computes the eigendecomposition of the covariance matrix of zero-mean
        whitefield fluctuations :math:`\\Delta W_k = W_k - \\bar{W}` and stores
        the resulting eigen fields and normalised eigenvalues in the container.

        Raises:
            ValueError: If ``whitefields`` is absent.
            ValueError: If fewer than two whitefields are present.

        Returns:
            A new :class:`CrystMetadata` with ``eigen_field`` and
            ``eigen_value`` populated.

        Example:
            Decompose a set of background estimates and inspect the explained
            variance:

            >>> metadata = metadata.pca()
            >>> print(metadata.eigen_value)
        """
        if self.is_empty(self.whitefields):
            raise ValueError('no whitefield in the container')
        if self.whitefields.shape[0] == 1:
            raise ValueError('A stack of several whitefields is needed to perform PCA')

        xp = self.__array_namespace__()
        fields = self.whitefields - self.flatfield
        axes = tuple(range(1, len(self.frame_shape) + 1))
        mat_svd = xp.tensordot(fields, fields, axes=(axes, axes))
        eig_vals, eig_vecs = xp.linalg.eigh(mat_svd)
        effs = xp.tensordot(eig_vecs, fields, axes=((0,), (0,)))
        return self.replace(eigen_field=effs, eigen_value=eig_vals / eig_vals.sum())

    def project(self, data: RealArray, good_fields: Indices=slice(None),
                clip_snr: float=3.0, n_iter: int=3, std_min: float=0.0,
                n_pixels: int | None=None) -> PCAProjection:
        """Project detector frames onto the PCA basis.

        Fits the residual :math:`D - \\bar{W}` for each frame to a linear
        combination of the stored eigen fields and returns the projection
        coefficients. Pass the result to :meth:`PCAProjection.apply` to
        reconstruct a per-frame background.

        Args:
            data: Raw detector data, shape ``(N, *frame_shape)``.
            good_fields: Indices of eigen fields to include in the fit.
                All fields are used by default.
            clip_snr: SNR threshold for rejecting bright diffraction signal.
            n_iter: Total number of least-squares fits. A value of one performs
                ordinary masked least squares; later fits reject signal using
                the preceding background estimate.
            std_min: Lower bound for the per-pixel standard deviation used in
                signal rejection.
            n_pixels: Number of detector pixels used for fitting. A deterministic
                random subset is selected without replacement. By default, the
                complete frame is used.

        Raises:
            ValueError: If ``flatfield`` is absent, ``n_iter`` is less than one,
                ``n_pixels`` is invalid, or iterative rejection is requested
                without ``std``.

        Returns:
            A :class:`PCAProjection` with fields ``good_fields`` (selected
            component indices) and ``projection`` (per-frame coefficient array,
            shape ``(N, n_fields)``).

        Example:
            Project frames onto the two dominant PCA components:

            >>> proj = metadata.project(frames, good_fields=[0, 1], n_iter=3)
            >>> whitefields = proj.apply(metadata)
        """
        if self.is_empty(self.flatfield):
            raise ValueError('No flatfield in the container')
        if n_iter < 1:
            raise ValueError('n_iter must be at least one')
        if n_iter > 1 and self.is_empty(self.std):
            raise ValueError('No std in the container for iterative signal rejection')

        xp = self.__array_namespace__()
        frame_size = prod(self.frame_shape)
        if n_pixels is not None and (n_pixels < 1 or n_pixels > frame_size):
            raise ValueError(f'n_pixels must be between one and the frame size {frame_size}')

        indices = None
        if n_pixels is not None and n_pixels < frame_size:
            indices = default_rng(0, NumPy).choice(frame_size, (n_pixels,), replace=False)
            indices = xp.asarray(indices)
            indices = xp.unravel_index(indices, self.frame_shape)

        if indices is not None:
            data = xp.reshape(data[(...,) + indices], (-1, n_pixels))
            metadata = self.apply_mask(indices)
        else:
            data = xp.reshape(data, (-1,) + self.frame_shape)
            metadata = self

        if self.is_empty(metadata.eigen_field):
            good_fields = xp.array([], dtype=int)
            lsq_data = LSQData(y=data, W=metadata.flatfield[None, None, ...])

        else:
            good_fields = list_indices(good_fields, metadata.eigen_field.shape[0])
            good_fields = xp.asarray(good_fields, dtype=int)
            fields = metadata.eigen_field[good_fields]
            lsq_data = LSQData(y=data - metadata.flatfield, W=fields[None, ...])

        if self.is_empty(metadata.mask):
            mask = xp.ones(data.shape, dtype=bool)
        else:
            mask = xp.broadcast_to(metadata.mask, data.shape)

        projection = lsq_data.apply_mask(mask).solve()
        result = PCAProjection(good_fields=good_fields, projection=projection)

        if n_iter == 1:
            return result

        std = xp.clip(metadata.std, std_min, xp.inf)
        for _ in range(1, n_iter):
            background = result.apply(metadata)
            n_mask = mask & (data <= background + clip_snr * std)
            projection = lsq_data.apply_mask(n_mask).solve()
            result = result.replace(projection=projection)

        return result

@dataclass
class CrystData(CrystBase):
    """Detector data container for a single CBC frame stack.

    Holds raw detector frames together with the bad-pixel mask, whitefield,
    noise standard deviation, and the resulting SNR frames. All mutating
    operations return a new :class:`CrystData` rather than modifying in place.
    The container can be saved to and loaded from HDF5 files using
    :func:`~cbclib_v2.write_hdf` and :func:`~cbclib_v2.read_hdf` with the
    default protocol returned by :meth:`default_protocol`.

    Attributes:
        data: Raw detector frames, shape ``(n_frames, *frame_shape)``.
        whitefield: Background model, shape ``frame_shape`` (static) or
            ``(n_frames, *frame_shape)`` (per-frame).
        std: Per-pixel noise standard deviation, shape ``frame_shape``.
        snr: Background-corrected signal-to-noise ratio, shape
            ``(n_frames, *frame_shape)``. Computed by :meth:`update_snr`.
        frames: Integer indices of the frames in this container.
        mask: Bad-pixel mask, shape ``frame_shape``. ``True`` = good pixel.

    Example:
        Load frames, estimate the background, compute SNR, and launch streak
        detection:

        >>> frames = run.data(indices[:20])
        >>> data = cbc.CrystData(frames)
        >>> data = data.update_mask(method='range', vmin=0, vmax=10_000_000)
        >>> data = data.update_metadata(method='robust-mean-scale',
        ...                             r0=0.5, r1=0.95, n_iter=2, lm=9.0)
        >>> data = data.update_snr(std_min=0.5)
    """
    data        : IntArray | RealArray = field(default_factory=lambda: np.array([]))

    whitefield  : RealArray = field(default_factory=lambda: np.array([]))
    std         : RealArray = field(default_factory=lambda: np.array([]))
    snr         : RealArray = field(default_factory=lambda: np.array([]))

    frames      : IntArray = field(default_factory=lambda: np.array([], dtype=int))
    mask        : BoolArray = field(default_factory=lambda: np.array([], dtype=bool))

    protocol    : H5Protocol = field(default_factory=lambda: H5Protocol.read(DATA_PROTOCOL))

    def __post_init__(self):
        xp = self.__array_namespace__()
        if self.frames.size != self.num_frames:
            self.frames = xp.arange(self.num_frames)
        if self.mask.shape != self.frame_shape:
            self.mask = xp.ones(self.frame_shape, dtype=bool)

    @property
    def num_frames(self) -> int:
        """Number of frames in this container."""
        current, old = 0, 0
        for attr, data in self.contents().items():
            kind = self.protocol.get_kind(attr)
            if kind == Kinds.stack and isinstance(data, Array):
                current = data.shape[0]

            if old and current != old:
                raise ValueError(f"Attribute {attr} has an incompatible shape: {data.shape}")

            old = current

        return current

    @property
    def num_whitefields(self) -> int:
        """Number of whitefield images stored (1 for static, N for per-frame)."""
        return self.whitefield.size // prod(self.frame_shape)

    @property
    def shape(self) -> Shape:
        """Full shape of the data array: ``(n_frames, *frame_shape)``."""
        return (self.num_frames,) + self.frame_shape

    @classmethod
    def default_protocol(cls) -> H5Protocol:
        """Return the built-in :class:`~cbclib_v2.H5Protocol` for this container.

        Used by :func:`~cbclib_v2.write_hdf` and :func:`~cbclib_v2.read_hdf`
        to resolve attribute names to HDF5 dataset paths.  The default
        protocol maps attributes to the following dataset paths:

        .. code-block:: text

            /entry/data/data                     ← data
            /entry/crystallography/frames        ← frames
            /entry/crystallography/mask          ← mask
            /entry/crystallography/whitefield    ← whitefield
            /entry/crystallography/std           ← std
            /entry/crystallography/snr           ← snr

        Returns:
            The default :class:`~cbclib_v2.H5Protocol` read from the built-in
            INI configuration file.
        """
        return H5Protocol.read(DATA_PROTOCOL)

    def apply_mask(self) -> 'CrystData':
        """Return a new :class:`CrystData` with ``whitefield``, ``std``, and
        ``snr`` zeroed at bad pixels.

        Returns:
            New :class:`CrystData` with masked arrays.
        """
        attributes = {}
        if not self.is_empty(self.whitefield):
            attributes['whitefield'] = self.whitefield * self.mask
        if not self.is_empty(self.std):
            attributes['std'] = self.std * self.mask
        if not self.is_empty(self.snr):
            attributes['snr'] = self.snr * self.mask
        return self.replace(**attributes)

    def import_mask(self, mask: BoolArray, update: str='reset') -> 'CrystData':
        """Return a new :class:`CrystData` object with the new mask.

        Args:
            mask : New mask array.
            update : Multiply the new mask and the old one if 'multiply', use the
                new one if 'reset'.

        Raises:
            ValueError : If the mask shape is incompatible with the data.
            ValueError : If there is no ``data`` inside the container.

        Returns:
            New :class:`CrystData` object with the updated ``mask``.
        """
        if self.is_empty(self.mask):
            raise ValueError('no mask in the container')
        if mask.shape != self.frame_shape:
            raise ValueError('mask and data have incompatible shapes: '\
                             f'{mask.shape} != {self.frame_shape}')

        if update == 'reset':
            return self.replace(mask=mask)
        if update == 'multiply':
            return self.replace(mask=mask * self.mask)
        raise ValueError(f'Invalid update keyword: {update:s}')

    def mask_region(self, roi: ROI) -> 'CrystData':
        """Return a new :class:`CrystData` object with the updated mask. The region
        defined by the `[y_min, y_max, x_min, x_max]` will be masked out.

        Args:
            roi : Bad region of interest in the detector plane. A set of four
                coordinates `[y_min, y_max, x_min, x_max]`.

        Raises:
            ValueError : If there is no ``data`` inside the container.

        Returns:
            New :class:`CrystData` object with the updated ``mask``.
        """
        if self.is_empty(self.mask):
            raise ValueError('no mask in the container')

        xp = self.__array_namespace__()
        mask = xp.copy(self.mask)
        mask[roi[0]:roi[1], roi[2]:roi[3]] = False
        return self.replace(mask=mask).apply_mask()

    def metadata(self) -> CrystMetadata:
        """Extract a :class:`CrystMetadata` from this container.

        Creates a single-whitefield metadata object from the current ``mask``,
        ``std``, and ``whitefield`` (stored as ``flatfield``).

        Raises:
            ValueError: If ``whitefield`` is absent.
            ValueError: If ``std`` is absent.

        Returns:
            A :class:`CrystMetadata` with ``mask``, ``std``, and ``flatfield``
            set.
        """
        if self.is_empty(self.whitefield):
            raise ValueError('no whitefield in the container')
        if self.is_empty(self.std):
            raise ValueError('no std in the container')

        return CrystMetadata(mask=self.mask, std=self.std, flatfield=self.whitefield)

    def region_detector(self, structure: Structure) -> 'RegionDetector':
        """Return a :class:`RegionDetector` for connected-region streak detection.

        Args:
            structure: 2-D connectivity structure for region growing and line
                fitting (see :mod:`cbclib_v2.label`).

        Raises:
            ValueError: If ``snr`` is absent (call :meth:`update_snr` first).

        Returns:
            A :class:`RegionDetector` operating on the current SNR frames.
        """
        if self.is_empty(self.snr):
            raise ValueError('no snr in the container')

        parent = cast(ReferenceType[CrystData], ref(self))
        return RegionDetector(data=self.snr, structure=structure, parent=parent)

    def online_detector(self, structure: Structure, radial_index: IntArray, n_bins: int
                        ) -> 'OnlineDetector':
        """Return an online radial-background region detector.

        The online detector estimates a compact radial background profile for
        each frame and labels connected pixels whose residual exceeds a radial
        SNR threshold. It is intended for fast hit finding or coarse region
        detection directly on raw detector counts, before constructing a
        persistent :class:`CrystMetadata` background model.

        Args:
            structure: Connectivity structure used to group signal pixels into
                labeled regions.
            radial_index: Integer radial-bin map with the same detector shape
                as one frame. Usually created with
                :meth:`~cbclib_v2.Detector.radial_index`.
            n_bins: Number of radial bins represented in ``radial_index``.

        Raises:
            ValueError: If ``data`` is absent.

        Returns:
            An :class:`OnlineDetector` operating on the current raw frames.
        """
        if self.is_empty(self.data):
            raise ValueError('no data in the container')

        parent = cast(ReferenceType[CrystData], ref(self))
        return OnlineDetector(data=self.data, structure=structure, radial_index=radial_index,
                              n_bins=n_bins, parent=parent)

    def reset_mask(self) -> 'CrystData':
        """Reset bad pixel mask. Every pixel is assumed to be good by default.

        Raises:
            ValueError : If there is no ``data`` inside the container.

        Returns:
            New :class:`CrystData` object with the default ``mask``.
        """
        xp = self.__array_namespace__()
        return self.replace(mask=xp.array([], dtype=bool))

    def select(self, idxs: Indices | None=None):
        """Return a new :class:`CrystData` containing a subset of frames.

        Indexes all ``stack``- and ``sequence``-kind attributes along the
        leading axis; frame-level attributes (``mask``, ``whitefield``,
        ``std``) are carried over unchanged.

        Args:
            idxs: Frame indices. Accepts integer, slice, or array-like index.

        Returns:
            New :class:`CrystData` with the selected frames.

        Example:
            Select three specific frames from the container:

            >>> hits = data.select([0, 5, 12])
        """
        data_dict = {}
        for attr in self.contents():
            if self.protocol.get_kind(attr) in (Kinds.sequence, Kinds.stack):
                data_dict[attr] = getattr(self, attr)[idxs]
            else:
                data_dict[attr] = getattr(self, attr)
        return self.replace(**data_dict)

    def streak_detector(self, structure: Structure, vmin: float) -> 'StreakDetector':
        """Return a :class:`StreakDetector` for diffraction streak detection.

        Args:
            structure: Connectivity structure used for peak detection and
                linelet fitting (see :mod:`cbclib_v2.label`).
            vmin: SNR threshold used to assess the statistical significance of
                detected streaks.

        Raises:
            ValueError: If ``snr`` is absent (call :meth:`update_snr` first).

        Returns:
            A :class:`StreakDetector` operating on the current SNR frames.
        """
        if self.is_empty(self.snr):
            raise ValueError('no snr in the container')

        parent = cast(ReferenceType[CrystData], ref(self))
        return StreakDetector(data=self.snr, structure=structure, vmin=vmin, parent=parent)

    def update_mask(self, method: MaskMethod='no-bad', vmin: int=0, vmax: int=65535,
                    snr_max: float=3.0, roi: ROI | None=None) -> 'CrystData':
        """Return a new :class:`CrystData` object with the updated bad pixels mask.

        Args:
            method : Bad pixels masking methods. The following keyword values are
                allowed:

                * 'all-bad' : Mask out all pixels.
                * 'no-bad' (default) : No bad pixels.
                * 'range' : Mask the pixels which values lie outside of (`vmin`,
                  `vmax`) range.
                * 'snr' : Mask the pixels which SNR values lie exceed the SNR
                  threshold `snr_max`. The snr is given by
                  :code:`abs(data - whitefield) / sqrt(whitefield)`.

            vmin : Lower intensity bound of 'range-bad' masking method.
            vmax : Upper intensity bound of 'range-bad' masking method.
            snr_max : SNR threshold.
            roi : Region of the frame undertaking the update. The whole frame is updated
                by default.

        Raises:
            ValueError : If there is no ``data`` inside the container.
            ValueError : If there is no ``snr`` inside the container.
            ValueError : If ``method`` keyword is invalid.
            ValueError : If ``vmin`` is larger than ``vmax``.

        Returns:
            New :class:`CrystData` object with the updated ``mask``.
        """
        if self.is_empty(self.data):
            raise ValueError('no data in the container')
        if self.is_empty(self.mask):
            raise ValueError('no mask in the container')

        xp = self.__array_namespace__()
        if vmin >= vmax:
            raise ValueError('vmin must be less than vmax')
        if roi is None:
            roi = (0, self.shape[-2], 0, self.shape[-1])

        data = (self.data * self.mask)[..., roi[0]:roi[1], roi[2]:roi[3]]

        if method == 'all-bad':
            mask = xp.zeros(self.frame_shape, dtype=bool)
        elif method == 'no-bad':
            mask = xp.ones(self.frame_shape, dtype=bool)
        elif method == 'range':
            mask = xp.all((data >= vmin) & (data < vmax), axis=0)
        elif method == 'snr':
            if self.snr is None:
                raise ValueError('No snr in the container')

            snr = self.snr[..., roi[0]:roi[1], roi[2]:roi[3]]
            mask = xp.mean(xp.abs(snr), axis=0) < snr_max
        else:
            raise ValueError(f'Invalid method argument: {method:s}')

        new_mask = xp.copy(self.mask)
        new_mask[..., roi[0]:roi[1], roi[2]:roi[3]] &= mask
        return self.replace(mask=new_mask)

    def update_snr(self, std_min: float=0.0) -> 'CrystData':
        """Return a new :class:`CrystData` with background-corrected SNR frames.

        Computes ``SNR = (data * mask - whitefield) / max(std, std_min)`` for
        each frame.

        Args:
            std_min: Noise floor applied before division to prevent near-zero
                denominators. A value of ``0.5`` is typical for photon-counting
                detectors.

        Raises:
            ValueError: If ``mask`` is absent.
            ValueError: If ``std`` is absent.
            ValueError: If ``whitefield`` is absent.

        Returns:
            New :class:`CrystData` with ``snr`` populated.
        """
        if self.is_empty(self.mask):
            raise ValueError('no mask in the container')
        if self.is_empty(self.std):
            raise ValueError('no std in the container')
        if self.is_empty(self.whitefield):
            raise ValueError('no whitefield in the container')

        xp = self.__array_namespace__()
        std = xp.clip(self.std, std_min, xp.inf)
        snr = xp.where(std, (self.data * self.mask - self.whitefield) / std, 0.0)
        return self.replace(snr=snr)

    def update_metadata(self, method: MDMethod='robust-mean-scale', frames: Indices | None=None,
                        r0: float=0.0, r1: float=0.5, n_iter: int=12, lm: float=9.0) -> 'CrystData':
        """Estimate whitefield and noise std jointly from the frame stack.

        A convenience wrapper that calls :meth:`update_whitefield` and
        :meth:`update_std` together using a consistent estimation strategy.

        Args:
            method: Joint estimation method:

                * ``'robust-mean-scale'`` (default) — robust mean for both
                  whitefield and std using the FLkOS estimator.
                * ``'median-poisson'`` — pixel-wise median whitefield with
                  Poisson noise (``std = √whitefield``).
                * ``'robust-mean-poisson'`` — robust mean whitefield with
                  Poisson noise.

            frames: Frame indices to use. All frames are used when ``None``.
            r0: Lower bound on the expected inlier fraction (FLkOS).
            r1: Upper bound on the expected inlier fraction (FLkOS).
            n_iter: Number of Gaussian-fitting iterations (FLkOS).
            lm: Outlier threshold in units of the estimated standard deviation
                (FLkOS).

        Raises:
            ValueError: If ``method`` is not one of the allowed values.

        Returns:
            New :class:`CrystData` with ``whitefield`` and ``std`` updated.

        Example:
            Estimate background from 20 frames using the robust mean:

            >>> data = data.update_metadata(method='robust-mean-scale',
            ...                             r0=0.5, r1=0.95, n_iter=2, lm=9.0)
        """
        if method == 'median-poisson':
            data = self.update_whitefield('median', frames)
            return data.update_std('poisson', frames)

        if method == 'robust-mean-scale':
            xp = self.__array_namespace__()
            if frames is None:
                frames = xp.arange(self.num_frames)

            whitefield, std = robust_mean(inp=self.data[frames] * self.mask, axis=0, r0=r0, r1=r1,
                                          n_iter=n_iter, lm=lm, return_std=True)
            return self.replace(whitefield=xp.asarray(whitefield), std=xp.asarray(std))

        if method == 'robust-mean-poisson':
            data = self.update_whitefield('robust-mean', frames, r0, r1, n_iter, lm)
            return data.update_std('poisson')

        raise ValueError(f"Invalid method argument: {method}")

    def update_std(self, method: STDMethod='robust-scale', frames: Indices | None=None,
                   r0: float=0.0, r1: float=0.5, n_iter: int=12, lm: float=9.0
                   ) -> 'CrystData':
        """Estimate the per-pixel noise standard deviation.

        Args:
            method: Noise estimation method:

                * ``'robust-scale'`` (default) — derives std from the robust
                  spread of the frame stack using the FLkOS estimator.
                * ``'poisson'`` — assumes Poisson statistics:
                  ``std = √mean(whitefield)``.

            frames: Frame indices to use. All frames are used when ``None``.
            r0: Lower bound on the expected inlier fraction (FLkOS).
            r1: Upper bound on the expected inlier fraction (FLkOS).
            n_iter: Number of Gaussian-fitting iterations (FLkOS).
            lm: Outlier threshold in units of the estimated standard deviation
                (FLkOS).

        Raises:
            ValueError: If ``data`` or ``mask`` are absent (``'robust-scale'``).
            ValueError: If ``whitefield`` is absent (``'poisson'``).
            ValueError: If ``method`` is not one of the allowed values.

        Returns:
            New :class:`CrystData` with ``std`` updated.
        """
        xp = self.__array_namespace__()
        if frames is None:
            frames = xp.arange(self.num_frames)

        if method == 'robust-scale':
            if self.is_empty(self.data):
                raise ValueError('no data in the container')
            if self.is_empty(self.mask):
                raise ValueError('no mask in the container')

            _, std = robust_mean(inp=self.data[frames] * self.mask, axis=0, r0=r0, r1=r1,
                                 n_iter=n_iter, lm=lm, return_std=True)
            std = xp.asarray(std)
        elif method == 'poisson':
            if self.is_empty(self.whitefield):
                raise ValueError('no whitefield in the container')

            whitefields = xp.reshape(self.whitefield, (self.num_whitefields,) + self.frame_shape)
            std = xp.sqrt(xp.mean(whitefields, axis=0))
        else:
            raise ValueError(f"Invalid method argument: {method}")

        return self.replace(std=std)

    def update_whitefield(self, method: WFMethod='median', frames: Indices | None=None,
                          r0: float=0.0, r1: float=0.5, n_iter: int=12, lm: float=9.0
                          ) -> 'CrystData':
        """Return a new :class:`CrystData` object with new whitefield.

        Args:
            method : Choose method for white-field generation. The following keyword
                values are allowed:

                * 'median' : Taking a median through the stack of frames.
                * 'robust-mean' : Finding a robust mean through the stack of frames.

            frames : List of frames to use for the white-field estimation.
            r0 : A lower bound guess of ratio of inliers. We'd like to make a sample
                out of worst inliers from data points that are between `r0` and `r1`
                of sorted residuals.
            r1 : An upper bound guess of ratio of inliers. Choose the `r0` to be as
                high as you are sure the ratio of data is inlier.
            n_iter : Number of iterations of fitting a gaussian with the FLkOS
                algorithm.
            lm : How far (normalized by STD of the Gaussian) from the mean of the
                Gaussian, data is considered inlier.

        Raises:
            ValueError : If there is no ``data`` inside the container.
            ValueError : If ``method`` keyword is invalid.

        Returns:
            New :class:`CrystData` object with the updated ``whitefield``.
        """
        if self.is_empty(self.data):
            raise ValueError('no data in the container')
        if self.is_empty(self.mask):
            raise ValueError('no mask in the container')

        if frames is None:
            xp = self.__array_namespace__()
            frames = xp.arange(self.num_frames)

        if method == 'median':
            whitefield = median(inp=self.data[frames] * self.mask, axis=0)
        elif method == 'robust-mean':
            whitefield = robust_mean(inp=self.data[frames] * self.mask, axis=0, r0=r0,
                                     r1=r1, n_iter=n_iter, lm=lm)
        else:
            raise ValueError('Invalid method argument')

        return self.replace(whitefield=whitefield, protocol=self.default_protocol())

class DetectorBase(DataContainer):
    """Base class for SNR-frame detectors returned by :class:`CrystData`."""

    data            : RealArray
    parent          : ReferenceType[CrystData]

    @property
    def shape(self) -> Shape:
        """Shape of the SNR data array."""
        return self.data.shape

    def __getitem__(self: Self, idxs: Indices) -> Self:
        return self.replace(data=self.data[idxs])

    def clip(self: Self, vmin: ArrayLike, vmax: ArrayLike) -> Self:
        """Return a new detector with SNR values clipped to ``[vmin, vmax]``."""
        xp = self.__array_namespace__()
        return self.replace(data=xp.clip(self.data, vmin, vmax))

@dataclass
class OnlineDetector(DataContainer):
    """Radial-background detector for online hit and region detection.

    Online detection is a fast, permissive save/reject step used during data
    acquisition or early inspection, when high-rate FEL detector streams are
    too large to keep in full. ``OnlineDetector`` estimates a compact radial
    background from the frames being searched and labels connected pixels whose
    residual radial SNR exceeds a threshold.

    The result is suitable for hit finding and candidate-region discovery. It
    is not a final background-subtraction method for intensity scaling, and it
    does not enforce the streak-like morphology expected from CBC diffraction.

    Attributes:
        data: Raw detector frame stack, shape ``(n_frames, *frame_shape)`` or a
            compatible leading batch shape.
        structure: Connectivity structure used by :func:`~cbclib_v2.label` to
            group signal pixels into regions.
        radial_index: Integer radial-bin map with shape ``frame_shape``.
            Pixels outside the detector panels may be marked ``-1`` by native
            geometry helpers and are ignored by the radial-profile kernels.
        n_bins: Number of radial bins in the compact profiles.
        parent: Weak reference to the source :class:`CrystData` container.

    Example:
        Estimate radial profiles and label online signal regions:

        >>> detector = read_crystfel('detector.geom')
        >>> radial = np.empty(detector.shape[-2:], dtype=np.int32)
        >>> detector.radial_index(radial, center=(512, 512), n_bins=1024)
        >>> online = data.online_detector(Structure([1, 1], 1), radial, 1024)
        >>> profiles = online.profiles(clip_snr=4.0, n_iter=5)
        >>> regions = online.detect_regions(profiles, min_snr=5.0, npts=3)
    """
    data            : IntArray | RealArray
    structure       : Structure
    radial_index    : IntArray
    n_bins          : int
    parent          : ReferenceType[CrystData]

    def profiles(self, interval: int=1, clip_snr: float=3.0, n_iter: int=3,
                 std_min: float=0.0) -> RadialProfiles:
        """Compute compact per-frame radial whitefield and noise profiles.

        The profile estimator groups pixels by ``radial_index`` and computes
        one mean and standard deviation per radial bin. The fit is repeated
        ``n_iter`` times; after each pass, pixels above
        ``mean + clip_snr * std`` are excluded so sparse diffraction peaks do
        not bias the background estimate.

        Args:
            interval: Process every ``interval``-th radial bin together in the
                native kernel. Larger values can improve throughput for many
                narrow bins at the cost of coarser temporary grouping.
            clip_snr: SNR threshold used to reject bright outliers during
                iterative profile estimation.
            n_iter: Number of outlier-rejection passes.
            std_min: Lower bound for per-bin standard deviation.

        Returns:
            Compact :class:`~cbclib_v2.RadialProfiles` containing per-frame
            radial ``whitefield``, ``std``, and pixel ``counts`` arrays of
            shape ``(n_frames, n_bins)``.
        """
        return radial_profiles(self.data, self.radial_index, self.n_bins, interval,
                               clip_snr=clip_snr, n_iter=n_iter, std_min=std_min)

    def detect_regions(self, profiles: RadialProfiles, min_snr: float,
                       npts: int=1, std_min: float=0.0) -> LabelResult:
        """Label connected pixels above the radial residual-SNR threshold.

        Each pixel is compared with the whitefield/std value of its radial bin:
        ``(data - whitefield[r]) / max(std[r], std_min)``. Pixels with residual
        SNR at least ``min_snr`` are foreground and are grouped with
        :func:`~cbclib_v2.label` using :attr:`structure`.

        Args:
            profiles: Compact radial background model returned by
                :meth:`profiles`.
            min_snr: Minimum radial residual SNR for a pixel to be considered
                signal.
            npts: Minimum connected-region size in pixels. Smaller regions are
                discarded.
            std_min: Lower bound for the radial standard deviation used in the
                SNR denominator.

        Returns:
            Labeled online signal regions as a :class:`~cbclib_v2.LabelResult`.
        """
        signal = profiles.is_signal(self.data, self.radial_index, min_snr, std_min)
        return label(signal, structure=self.structure, npts=npts)

@dataclass
class StreakDetector(DetectorBase):
    """Streak detector operating on SNR frames from a :class:`CrystData` container.

    Thin wrapper around :class:`~cbclib_v2.streak_finder.PatternStreakFinder`
    that exposes a step-by-step pipeline: connected-region detection, peak
    finding, linelet fitting, and streak assembly.  Obtain an instance from
    :meth:`CrystData.streak_detector`.

    See :class:`~cbclib_v2.streak_finder.PatternStreakFinder` for the full
    algorithmic description and parameter details.

    Attributes:
        data: SNR frame stack, shape ``(n_frames, *frame_shape)``.
        structure: Connectivity structure for peak detection and linelet
            fitting.
        vmin: SNR threshold for streak significance testing.
    """

    data            : RealArray
    structure       : Structure
    vmin            : float
    parent          : ReferenceType[CrystData]

    def __post_init__(self):
        self.finder = PatternStreakFinder(self.data, self.structure, self.vmin)

    def detect_regions(self, npts: int, connectivity: Structure | None=None) -> LabelResult:
        return self.finder.detect_regions(npts, connectivity)

    def detect_peaks(self, regions: LabelResult) -> Tuple[PeakLabels, IntArray]:
        return self.finder.detect_peaks(regions)

    def fit_linelets(self, regions: PeakLabels, peaks: IntArray) -> Tuple[RealArray, PeakLabels]:
        return self.finder.fit_linelets(regions, peaks)

    def detect_streaks(self, labels: PeakLabels, peaks: IntArray, linelets: RealArray,
                       xtol: float, nfa: int = 0) -> StreakResult:
        return self.finder.detect_streaks(labels, peaks, linelets, xtol, nfa)

    def streak_labels(self, streaks: StreakResult, labels: PeakLabels, peaks: IntArray
                      ) -> LabelResult:
        ranks = self.finder.ranking(streaks, labels, peaks)
        return self.finder.streak_labels(streaks, ranks, labels, peaks)

    def line_fit(self, labeled: LabelResult) -> RealArray:
        return self.finder.line_fit(labeled)

    def min_support(self, labeled: LabelResult, lines: RealArray, xtol: float) -> RealArray:
        return self.finder.min_support(labeled, lines, xtol)

    def to_streaks(self, lines: RealArray) -> StackedStreaks | Streaks:
        xp = self.__array_namespace__()
        points = lines.reshape((-1, 2, self.data.ndim))
        indices = xp.round(xp.mean(points[:, :, 2:], axis=-2)).astype(int)
        lines = points[:, :, :2].reshape((-1, 4))

        num_modules = self.parent().num_modules
        if num_modules > 1:
            return StackedStreaks(indices[..., -1], indices[..., -2], lines, num_modules)
        return Streaks(indices[..., -1], lines)

@dataclass
class RegionDetector(DetectorBase):
    """Region-based detector operating on SNR frames from a :class:`CrystData` container.

    Detects connected regions of elevated SNR and fits geometric primitives
    (lines, ellipses) to their pixel distributions. A simpler alternative to
    :class:`StreakDetector` when streaks are broad or arc-shaped.  Obtain an
    instance from :meth:`CrystData.region_detector`.

    Attributes:
        data: SNR frame stack, shape ``(n_frames, *frame_shape)``.
        structure: 2-D connectivity structure, automatically expanded to match
            the data dimensionality.
    """

    data            : RealArray
    structure       : Structure
    parent          : ReferenceType[CrystData]

    def __post_init__(self):
        if self.structure.rank != 2:
            raise ValueError('Only 2D connectivity structure is supported for streak detection')
        self.structure = self.structure.expand_dims(list(range(self.data.ndim - 2)))

    def detect_regions(self, vmin: float, npts: int) -> LabelResult:
        """Label connected regions in the SNR frame stack.

        This function is similar to :func:`scipy.ndimage.label`: all pixels
        whose SNR exceeds *vmin* are treated as foreground.  Connected
        foreground pixels are assigned the same integer label; background
        pixels are labeled 0.  Connectivity is determined by
        :attr:`structure`.  Regions with fewer than *npts* pixels are
        discarded (their pixels reset to 0).

        Args:
            vmin: SNR threshold.  Pixels with ``data > vmin`` are foreground.
            npts: Minimum region size.  Regions with fewer pixels are removed.

        Returns:
            Labeled regions.
        """
        return label(self.data > vmin, structure=self.structure, npts=npts)

    def detect_streaks(self, regions: LabelResult) -> StackedStreaks | Streaks:
        """Fit lines to labeled regions and return a streak collection.

        Each region is represented by the major axis of its intensity
        distribution, computed via :meth:`line_fit`.  The frame and module
        coordinates of each streak are taken from the region's center of mass
        along the non-spatial axes.

        Args:
            regions: Labeled regions returned by :meth:`detect_regions`.

        Returns:
            :class:`~cbclib_v2.Streaks` for single-module data or
            :class:`~cbclib_v2.StackedStreaks` for stacked multi-module data.
        """
        xp = self.__array_namespace__()
        points = self.line_fit(regions).reshape((-1, 2, self.data.ndim))

        indices = xp.round(xp.mean(points[:, :, 2:], axis=-2)).astype(int)
        lines = points[:, :, :2].reshape((-1, 4))

        num_modules = self.parent().num_modules
        if num_modules > 1:
            return StackedStreaks(indices[..., -1], indices[..., -2], lines, num_modules)
        return Streaks(indices[..., -1], lines)

    def ellipse_fit(self, regions: LabelResult) -> RealArray:
        """Fit an ellipse to each labeled region using image moments.

        The ellipse parameters are derived from the second-order central
        `image moments <https://en.wikipedia.org/wiki/Image_moment>`_ of
        each region, weighted by the SNR values in :attr:`data`.  The
        covariance matrix of the spatial coordinates is decomposed into its
        eigenvectors; the semi-axes of the ellipse correspond to the FWHM
        along those eigenvectors.

        Args:
            regions: Labeled regions returned by :meth:`detect_regions`.

        Returns:
            Array of shape ``(N, 3)`` where each row is ``(a, b, theta)``:
            *a* and *b* are the FWHM of the major and minor axes, and
            *theta* is the orientation angle in radians.
        """
        return ellipse_fit(regions, self.data)

    def line_fit(self, regions: LabelResult) -> RealArray:
        """Fit a line to each labeled region using image moments.

        The line is the major axis of the intensity-weighted covariance
        ellipse computed from second-order central
        `image moments <https://en.wikipedia.org/wiki/Image_moment>`_.  The
        eigenvector corresponding to the largest eigenvalue of the covariance
        matrix gives the orientation; the endpoints are placed at the
        extremes of the region along that axis.

        Args:
            regions: Labeled regions returned by :meth:`detect_regions`.

        Returns:
            Array of shape ``(N, 4)`` where each row is
            ``(x1, y1, x2, y2)``, the pixel coordinates of the two endpoints
            of the fitted line segment.
        """
        return line_fit(regions, self.data)

    def center_of_mass(self, regions: LabelResult) -> RealArray:
        """Compute the intensity-weighted center of mass for each labeled region.

        The center of mass is the first-order
        `image moment <https://en.wikipedia.org/wiki/Image_moment>`_
        normalised by the total intensity (zeroth-order moment):
        ``c = M_10 / M_00`` along each axis, where
        ``M_ij = Σ x^i y^j · w(x, y)`` and *w* are the SNR values from
        :attr:`data`.

        Args:
            regions: Labeled regions returned by :meth:`detect_regions`.

        Returns:
            Array of shape ``(N, ndim)`` with the center-of-mass coordinates
            for each region.
        """
        return center_of_mass(regions, self.data)

    def covariance_matrix(self, regions: LabelResult) -> RealArray:
        """Compute the intensity-weighted covariance matrix for each labeled region.

        The covariance matrix is built from second-order central
        `image moments <https://en.wikipedia.org/wiki/Image_moment>`_
        weighted by the SNR values in :attr:`data`:
        ``Cov[i, j] = μ_ij / M_00``, where
        ``μ_ij = Σ (x - x̄)^i (y - ȳ)^j · w(x, y)``.

        Args:
            regions: Labeled regions returned by :meth:`detect_regions`.

        Returns:
            Array of shape ``(N, ndim, ndim)`` with the covariance matrix for
            each region.
        """
        return covariance_matrix(regions, self.data)
