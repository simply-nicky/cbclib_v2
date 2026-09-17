from typing import Iterable, Type
from typing_extensions import Self
import pandas as pd

from .._src.annotations import AnyNamespace, BoolArray, IntArray, IntSequence, RealArray, Shape
from .._src.array_api import add_at, asnumpy, safe_log, safe_divide
from .._src.crystfel import Detector
from .._src.data_container import ArrayContainer, DataContainer, IndexedContainer
from .._src.data_processing import CrystData
from .._src.state import field, State
from ..indexer.cbc_data import AnyPoints, IndexLookup, Miller
from ..indexer.cbc_pupil import BasePupil, SourcePlane
from ..indexer.cbc_setup import BaseSetup, ResolvedSetup
from .cbc_symmetry import PointGroup

class IntensityModel(State, DataContainer):
    kout        : RealArray     # (n_points, 3) endpoint kout vectors of the streak line
    source      : SourcePlane   # (n_points,) source-plane support
    pupil       : BasePupil     # (n_points,) or (1,) pupil support

    @property
    def kin(self) -> RealArray:
        """Return incident-vector candidates associated with the observed rays."""
        return self.kout - self.source.q

    @property
    def distance_sq(self) -> RealArray:
        """Return the summed squared distances to the source plane and pupil support."""
        xp = self.__array_namespace__()
        dist_src = xp.sum((self.kin - self.source.project(self.kin))**2, axis=-1)
        dist_ppl = xp.sum((self.kin - self.pupil.project(self.kin))**2, axis=-1)
        return xp.asarray(dist_src + dist_ppl)

    def log_radial(self, log_sigma: RealArray, xp: AnyNamespace) -> RealArray:
        """Return Gaussian attenuation by source-plane and pupil-support distance."""
        inv_sigma_sq = xp.exp(-2.0 * log_sigma)
        return -0.5 * self.distance_sq * inv_sigma_sq

    def log_profile(self, log_sigma: RealArray, xp: AnyNamespace) -> RealArray:
        """Return the continuous source-plane and pupil-support log profile."""
        return self.log_radial(log_sigma, xp)

class StreakPoints(State, IndexedContainer):
    index       : IntArray  # (n_points,) frame index
    streak_id   : IntArray  # (n_points,) streak index
    points      : RealArray # (n_points, 2) (x, y) coordinates in meters

    @property
    def shape(self) -> Shape:
        return self.points.shape[:-1]

    @property
    def x(self) -> RealArray:
        return self.points[..., 0]

    @property
    def y(self) -> RealArray:
        return self.points[..., 1]

class StreakIndices(State, IndexedContainer):
    index       : IntArray  # (n_points,) frame index
    streak_id   : IntArray  # (n_points,) streak index
    y           : IntArray  # (n_points,) y pixel coordinates
    x           : IntArray  # (n_points,) x pixel coordinates

    @classmethod
    def import_dataframe(cls, df: pd.DataFrame, xp: AnyNamespace) -> 'StreakIndices':
        """Import pattern points from :method:`~cbclib_v2.Patterns.pattern_dataframe`."""
        df = df.drop_duplicates(subset=['index', 'y', 'x'], keep=False)
        index = xp.asarray(df['index'].to_numpy())
        streak_id = xp.asarray(df['streak_id'].to_numpy())
        y = xp.asarray(df['y'].to_numpy())
        x = xp.asarray(df['x'].to_numpy())
        return cls(index=index, y=y, x=x, streak_id=streak_id)

    @property
    def shape(self) -> Shape:
        return self.y.shape

    @property
    def xy(self) -> IntArray:
        xp = self.__array_namespace__()
        return xp.stack((self.x, self.y), axis=-1)

    def mask(self, is_good: IntArray) -> 'StreakIndices':
        return self[is_good[self.y, self.x] > 0]

    def to_points(self, detector: Detector) -> StreakPoints:
        return StreakPoints(index=self.index, streak_id=self.streak_id,
                            points=detector.pixel_size * self.xy)

class PhotonCounts(State, ArrayContainer):
    I0          : IntArray
    background  : RealArray

    @classmethod
    def import_data(cls, data: CrystData, indices: StreakIndices, xp: AnyNamespace
                    ) -> 'PhotonCounts':
        if data.is_empty(data.whitefield):
            raise ValueError("Data does not contain whitefield counts.")
        I0 = xp.asarray(data.data[(indices.index, indices.y, indices.x)])
        background = xp.asarray(data.whitefield[(indices.index, indices.y, indices.x)])
        return cls(I0=I0, background=background)

    @property
    def shape(self) -> Shape:
        return self.I0.shape

    @property
    def signal(self) -> RealArray:
        xp = self.__array_namespace__()
        return self.I0 - self.background

class WeightedLinearEstimator(State, DataContainer):
    """Estimate grouped weighted linear coefficients through the origin.

    Attributes:
        group_id: Group index for every observation, with shape ``(n_observations,)``.
        weights: Non-negative observation weights with the same shape.
    """
    group_id: IntArray
    weights: RealArray

    def fit(self, predictor: RealArray, response: RealArray,
            previous: RealArray) -> RealArray:
        """Fit one coefficient per group, retaining unsupported previous estimates."""
        xp = self.__array_namespace__()
        numerator = add_at(xp.zeros_like(previous), self.group_id,
                           self.weights * predictor * response)
        denominator = add_at(xp.zeros_like(previous), self.group_id,
                             self.weights * predictor**2)
        estimate = safe_divide(numerator, denominator, xp)
        return xp.where(denominator > 0.0, estimate, previous)

class Reflections(State, DataContainer):
    hkl             : IntArray  # (n_reflections, 3) canonical Miller indices
    reflection_id   : IntArray  # (n_observations,) compact reflection index

    def __post_init__(self):
        self._indexer = None

    @classmethod
    def from_miller(cls, index: IntArray, hkl: IntArray, xp: AnyNamespace) -> 'Reflections':
        keys = xp.concat((index[:, None], hkl), axis=-1)
        keys, reflection_id = xp.unique(keys, return_inverse=True, axis=0)
        return cls(hkl=keys[:, 1:], reflection_id=reflection_id)

    @classmethod
    def from_hkl(cls, hkl: IntArray, xp: AnyNamespace) -> 'Reflections':
        keys, reflection_id = xp.unique(hkl, return_inverse=True, axis=0)
        return cls(hkl=keys, reflection_id=reflection_id)

    @property
    def indexer(self) -> IndexLookup:
        """Return a lookup for the reflection ID of each observation."""
        if self._indexer is None:
            self._indexer = IndexLookup.build(self.reflection_id)
        return self._indexer

    @property
    def n_observations(self) -> IntArray:
        xp = self.__array_namespace__()
        return add_at(xp.zeros((self.n_reflections,), dtype=int), self.reflection_id,
                      xp.ones_like(self.reflection_id))

    @property
    def n_reflections(self) -> int:
        return self.hkl.shape[0]

    def estimator(self, weights: RealArray) -> WeightedLinearEstimator:
        """Bind observation weights to the canonical-reflection grouping."""
        return WeightedLinearEstimator(group_id=self.reflection_id, weights=weights)

    def merge(self, data: 'ReflectionList', weights: RealArray, state: 'MergeState') -> RealArray:
        """Return merged intensities in reflection order."""
        return self.estimator(weights).fit(state.scale_at(data), data.I_hkl, state.I_hkl)

    def where(self, reflection_ids: IntSequence) -> IntArray:
        return self.indexer.get_index(reflection_ids)[0]

class ReflectionsMap(State, DataContainer):
    """Map observed streaks to pattern-local symmetry-equivalent reflections.

    Attributes:
        reflection_id: Contiguous merged-reflection index for every input streak,
            with shape ``(n_streaks,)``.
        n_reflections: Number of pattern-local merged reflections.
    """
    reflection_id : IntArray
    n_reflections : int = field(static=True)
    point_group   : str = field(static=True)

    @classmethod
    def from_miller(cls, miller: Miller, point_group: PointGroup,
                    xp: AnyNamespace) -> 'ReflectionsMap':
        """Map Miller indices to pattern-local intensity-equivalence orbits.

        Args:
            miller: Indexed Miller indices for the observed streaks.
            point_group: Crystallographic point group used for merging.
            xp: Array namespace used for the calculation.

        Returns:
            Streak-to-reflection mapping and the number of merged reflections.
        """
        canonical = point_group.canonical(miller.hkl_indices, xp)
        reflections = Reflections.from_miller(miller.index, canonical, xp)
        return cls(reflection_id=xp.asarray(reflections.reflection_id),
                   n_reflections=reflections.n_reflections,
                   point_group=point_group.symbol)

    def __len__(self) -> int:
        return self.n_reflections

    def at(self, points: StreakPoints) -> IntArray:
        """Return the merged-reflection index associated with each measured point."""
        return self.reflection_id[points.streak_id]

    def canonical(self, miller: Miller, xp: AnyNamespace) -> Miller:
        """Return one canonical Miller index for each mapped reflection.

        The output rows follow ``reflection_id`` order, so row ``r`` is aligned with scaler
        state and result arrays at row ``r``.

        Args:
            miller: Unmerged Miller indices used to construct this reflection map.
            point_group: Crystallographic point group used to construct this map.
            xp: Array namespace used for canonicalization.

        Returns:
            Pattern-local canonical Miller indices with shape ``(n_reflections,)``.
        """
        canonical = PointGroup(self.point_group).canonical(miller.hkl_indices, xp)
        _, representatives = xp.unique(self.reflection_id, return_index=True)
        return Miller(index=miller.index[representatives],
                      hkl=canonical[representatives])

class ScalerData(State, DataContainer):
    """Hold observed streak data and its shared-intensity mapping.
    """
    points      : StreakPoints
    counts      : PhotonCounts
    miller      : Miller
    reflections : ReflectionsMap

    @classmethod
    def import_data(cls, data: CrystData, streak_ids: StreakIndices, miller: Miller,
                    point_group: PointGroup, detector: Detector, xp: AnyNamespace
                    ) -> 'ScalerData':
        points = streak_ids.to_points(detector)
        counts = PhotonCounts.import_data(data, streak_ids, xp)
        reflections = ReflectionsMap.from_miller(miller, point_group, xp)
        return cls(points=points, counts=counts, miller=miller, reflections=reflections)

class ScalerState(State, DataContainer):
    log_hkl        : RealArray    # (n_reflections,)
    log_sigma_kin  : RealArray    # (n_frames,)

    @classmethod
    def default(cls, n_reflections: int, n_frames: int, sigma: float, xp: AnyNamespace
                ) -> 'ScalerState':
        log_hkl = xp.zeros((n_reflections,))
        log_sigma_kin = xp.full((n_frames,), xp.log(sigma))
        return cls(log_hkl=log_hkl, log_sigma_kin=log_sigma_kin)

    @classmethod
    def from_data(cls, data: ScalerData, setup: ResolvedSetup, sigma: float
                  ) -> 'ScalerState':
        xp = setup.__array_namespace__()
        reflection_id = data.reflections.at(data.points)
        sums = xp.zeros((len(data.reflections),))
        counts = xp.zeros((len(data.reflections),))
        sums = add_at(sums, reflection_id, data.counts.signal)
        counts = add_at(counts, reflection_id, xp.ones(data.points.shape[0]))
        log_hkl = safe_log(safe_divide(sums, counts, xp), xp)

        log_sigma_kin = xp.full((len(setup.xtal),), xp.log(sigma))
        return cls(log_hkl=log_hkl, log_sigma_kin=log_sigma_kin)

    @property
    def n_reflections(self) -> int:
        return len(self.log_hkl)

    @property
    def n_frames(self) -> int:
        return len(self.log_sigma_kin)

    def log_sigma_at(self, points: AnyPoints) -> RealArray:
        return self.log_sigma_kin[points.index]

    def log_at(self, points: StreakPoints, reflections: ReflectionsMap) -> RealArray:
        return self.log_hkl[reflections.at(points)]

class FullState(State, DataContainer):
    scaling : ScalerState
    setup   : BaseSetup

class ReflectionList(State, IndexedContainer):
    """Hold fitted reflection intensities and their conditional uncertainties.

    Attributes:
        index: Nonnegative pattern index per observation, with shape ``(n_observations,)``.
        hkl: Canonical HKL per observation, with shape ``(n_observations, 3)``.
            The same HKL may occur in several patterns.
        I_hkl: Fitted intensities, with shape ``(n_observations,)``.
        sigma_hkl: Conditional Poisson standard errors of ``I_hkl``, with shape
            ``(n_observations,)``. Positive infinity denotes zero conditional information.
    """
    index           : IntArray    # (n_observations,) compact index
    hkl             : IntArray    # (n_observations, 3) canonical hkl indices
    I_hkl           : RealArray   # (n_observations,) fitted intensities
    sigma_hkl       : RealArray   # (n_observations,) conditional uncertainties

    @classmethod
    def concat(cls: Type[Self], containers: Iterable[Self],
               monotonic_index: bool=True) -> Self:
        """Combine observations, treating input pattern ranges as distinct by default.

        Set ``monotonic_index=False`` when indices already identify patterns globally.
        Canonical HKLs and observation order are preserved; derive reflection mappings
        from the combined list before scaling.
        """
        return super().concat(containers, monotonic_index=monotonic_index)

    @classmethod
    def from_data(cls, data: ScalerData, I_hkl: RealArray, sigma_hkl: RealArray,
                  xp: AnyNamespace) -> 'ReflectionList':
        """Construct a result with canonical Miller indices from scaling data.

        Canonical indices are merged independently within each pattern in the same order as
        :meth:`ReflectionsMap.from_miller`.

        Args:
            data: Scaling observations containing the unmerged Miller indices.
            I_hkl: Fitted intensities with shape ``(n_reflections,)``.
            sigma_hkl: Conditional standard errors with shape ``(n_reflections,)``.
            xp: Array namespace used for canonicalization.

        Returns:
            Self-contained scaling result aligned with the merged reflection order.
        """
        miller = data.reflections.canonical(data.miller, xp)
        return cls(index=miller.index, hkl=miller.hkl_indices,
                   I_hkl=I_hkl, sigma_hkl=sigma_hkl)

    @classmethod
    def import_dataframe(cls, df: pd.DataFrame | pd.Series, frames: IntArray | None,
                         xp: AnyNamespace) -> 'ReflectionList':
        """Import canonical Miller indices, intensities, and uncertainties from a dataframe."""
        miller = Miller.import_dataframe(df, frames, xp)
        return cls(index=miller.index, hkl=miller.hkl_indices, I_hkl=xp.asarray(df['I_hkl']),
                   sigma_hkl=xp.asarray(df['sigma_hkl']))

    @property
    def shape(self) -> Shape:
        return self.hkl.shape[:-1]

    @property
    def h(self) -> IntArray:
        return self.hkl[..., 0]

    @property
    def k(self) -> IntArray:
        return self.hkl[..., 1]

    @property
    def l(self) -> IntArray:
        return self.hkl[..., 2]

    @property
    def n_patterns(self) -> int:
        """Return the pattern lookup size, including index gaps; zero for an empty list."""
        xp = self.__array_namespace__()
        return int(xp.max(self.index)) + 1 if self.index.size else 0

    def reflections(self) -> Reflections:
        """Return unique canonical HKLs and the reflection ID of each observation.

        Returns:
            HKLs with shape ``(n_unique, 3)`` and integer IDs with the observation shape.
            Indexing the HKL table by these IDs reconstructs :attr:`hkl`. Inputs must
            already use the same symmetry and indexing convention. No symmetry is applied.
            Results stay in the input array namespace. This discrete operation is not JIT-safe.
        """
        xp = self.__array_namespace__()
        return Reflections.from_hkl(self.hkl, xp)

    @property
    def is_informative(self) -> BoolArray:
        """Return observations carrying finite conditional information."""
        xp = self.__array_namespace__()
        return xp.isfinite(self.sigma_hkl)

    def n_informative(self, reflections: Reflections) -> IntArray:
        """Return the number of informative observations per reflection.

        Args:
            reflections: Reflection mapping for the observations in this list.

        Returns:
            Counts with shape ``(n_reflections,)``.
        """
        xp = self.__array_namespace__()
        return add_at(xp.zeros((reflections.n_reflections,), dtype=int),
                      reflections.reflection_id, self.is_informative)

    def estimator(self, weights: RealArray) -> WeightedLinearEstimator:
        """Bind observation weights and a fixed output domain to the pattern grouping."""
        return WeightedLinearEstimator(group_id=self.index, weights=weights)

    def fit_scales(self, intensity: RealArray, weights: RealArray, previous: RealArray
                   ) -> RealArray:
        """Fit one scale per pattern, retaining unsupported previous estimates."""
        return self.estimator(weights).fit(intensity, self.I_hkl, previous)

    def to_dataframe(self, frames: IntArray) -> pd.DataFrame:
        """Export canonical Miller indices, intensities, and uncertainties to a dataframe."""
        return pd.DataFrame({'index': asnumpy(frames[self.index]),
                             'h': asnumpy(self.h), 'k': asnumpy(self.k), 'l': asnumpy(self.l),
                             'I_hkl': asnumpy(self.I_hkl), 'sigma_hkl': asnumpy(self.sigma_hkl)})

class MergeState(State, DataContainer):
    """Shared intensities and positive illumination scales for one connected component.

    Attributes:
        log_scale: Natural log scales, shape ``(n_patterns,)``, including index gaps.
        I_hkl: Non-negative intensities, shape ``(n_reflections,)``, ordered by Reflections.
            Initial values must be finite; initialise supported HKLs above zero.
    """
    log_scale: RealArray
    I_hkl: RealArray

    @classmethod
    def from_data(cls, data: ReflectionList, reflections: Reflections,
                  log_scale: RealArray | None=None) -> 'MergeState':
        xp = data.__array_namespace__()
        if log_scale is None:
            log_scale = xp.zeros((data.n_patterns,))

        # Return an unweighted upper median for each reflection.
        reflection_id = reflections.reflection_id
        intensity = data.I_hkl / xp.exp(log_scale[data.index])
        indices = xp.lexsort((intensity, reflection_id))
        counts = add_at(xp.zeros((reflections.n_reflections,), dtype=reflection_id.dtype),
                        reflection_id, xp.ones_like(reflection_id))
        offsets = xp.cumsum(counts) - counts
        medians = intensity[indices][offsets + counts // 2]
        return cls(log_scale=log_scale, I_hkl=medians).normalise()

    def scale_at(self, data: ReflectionList) -> RealArray:
        """Return positive scales in observation order."""
        xp = self.__array_namespace__()
        return xp.exp(self.log_scale[data.index])

    def intensity_at(self, reflections: Reflections) -> RealArray:
        """Return shared intensities in observation order."""
        return self.I_hkl[reflections.reflection_id]

    def normalise(self) -> 'MergeState':
        """Set the geometric mean scale to one while preserving every prediction.

        This fixes one gauge only; disconnected components must be fitted independently.
        Empty states are unchanged. Index gaps participate in the gauge convention.
        """
        if not self.log_scale.size:
            return self
        xp = self.__array_namespace__()
        shift = xp.mean(self.log_scale)
        return self.replace(log_scale=self.log_scale - shift, I_hkl=self.I_hkl * xp.exp(shift))
