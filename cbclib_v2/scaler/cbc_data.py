from typing import Iterable, Type
from typing_extensions import Self
import pandas as pd
from .._src.annotations import AnyNamespace, BoolArray, IntArray, IntSequence, RealArray, Shape
from .._src.array_api import add_at, asnumpy, det_to_k, safe_divide
from .._src.crystfel import Detector
from .._src.data_container import ArrayContainer, DataContainer, IndexedContainer, IndexLookup
from .._src.data_processing import CrystData
from .._src.state import State, field
from ..indexer.cbc_data import Miller
from ..indexer.cbc_pupil import BasePupil, SourcePlane
from ..indexer.cbc_setup import BaseSetup
from .cbc_symmetry import PointGroup

class IntensityModel(State, DataContainer):
    kout        : RealArray     # (n_points, 3) endpoint kout vectors of the streak line
    source      : SourcePlane   # (n_points,) source-plane support
    pupil       : BasePupil     # (n_points,) or (1,) pupil support
    smp_pos     : RealArray     # (n_points, 3) sample position in metres

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

    def displace(self, points: RealArray, xp: AnyNamespace) -> 'IntensityModel':
        """Return a copy of the model with displaced streak endpoints."""
        return self.replace(kout=det_to_k(points, self.smp_pos, xp))

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
    """Hold detector counts, background estimates, and measured noise.

    Attributes:
        I0: Raw detected counts at each streak point.
        background: Estimated background counts at each streak point.
        std: Measured detector-noise standard deviation at each streak point.
    """
    I0          : IntArray
    background  : RealArray
    std         : RealArray

    @classmethod
    def import_data(cls, data: CrystData, indices: StreakIndices, xp: AnyNamespace, *,
                    I_min: float=float('-inf')) -> 'PhotonCounts':
        if data.is_empty(data.whitefield):
            raise ValueError("Data does not contain whitefield counts.")
        if data.is_empty(data.std):
            raise ValueError("Data does not contain noise standard deviations.")
        I0 = xp.asarray(data.values_at('data', indices.index, indices.y, indices.x))
        background = xp.asarray(
            data.values_at('whitefield', indices.index, indices.y, indices.x))
        std = xp.asarray(data.values_at('std', indices.index, indices.y, indices.x))
        return cls(I0=xp.clip(I0, I_min, xp.inf), background=background, std=std)

    @property
    def shape(self) -> Shape:
        return self.I0.shape

    @property
    def signal(self) -> RealArray:
        xp = self.__array_namespace__()
        return xp.asarray(self.I0 - self.background)

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

    def fit_median(self, predictor: RealArray, response: RealArray,
                   previous: RealArray) -> RealArray:
        """Fit grouped coefficients by their information-weighted median."""
        xp = self.__array_namespace__()
        if not self.group_id.size:
            return previous
        coefficients = safe_divide(response, predictor, xp)
        weights = self.weights * predictor**2
        order = xp.lexsort((coefficients, self.group_id))

        group_id = self.group_id[order]
        coefficients = coefficients[order]
        weights = weights[order]

        total = add_at(xp.zeros_like(previous), group_id, weights)
        counts = add_at(xp.zeros(previous.shape, dtype=group_id.dtype), group_id,
                        xp.ones_like(group_id))
        offsets = xp.cumsum(counts) - counts
        cumulative = xp.cumsum(weights)
        baseline = xp.where(offsets > 0, cumulative[xp.maximum(offsets - 1, 0)], 0.0)
        cumulative = cumulative - baseline[group_id]
        half_total = 0.5 * total[group_id]
        is_median = ((cumulative >= half_total)
                     & (cumulative - weights < half_total)
                     & (total[group_id] > 0.0))
        estimate = add_at(xp.zeros_like(previous), group_id,
                          xp.where(is_median, coefficients, 0.0))
        return xp.where(total > 0.0, estimate, previous)

class BaseData(DataContainer):
    @property
    def points(self) -> StreakPoints:
        raise NotImplementedError("Concrete data classes must implement points.")

    @property
    def counts(self) -> PhotonCounts:
        raise NotImplementedError("Concrete data classes must implement counts.")

    @property
    def miller(self) -> Miller:
        raise NotImplementedError("Concrete data classes must implement miller.")

    @property
    def n_patterns(self) -> int:
        """Return the number of patterns in the indexed data."""
        if self.points.size:
            return int(self.points.index[-1] + 1)
        return 0

    @property
    def n_streaks(self) -> int:
        """Return the number of predicted streaks."""
        return self.miller.shape[0]

    def sum_by_streak(self, values: RealArray, xp: AnyNamespace) -> RealArray:
        """Sum point values over their predicted streaks."""
        return add_at(xp.zeros((self.n_streaks,), dtype=values.dtype),
                      self.points.streak_id, values)

    def mean_by_profile(self, values: RealArray, profile: RealArray,
                        xp: AnyNamespace) -> RealArray:
        """Return profile-weighted means for each streak."""
        return safe_divide(self.sum_by_streak(profile * values, xp),
                           self.sum_by_streak(profile, xp), xp)

    def std_by_streak(self, profile: RealArray, xp: AnyNamespace) -> RealArray:
        """Return profile-coefficient errors from measured detector noise."""
        profile = safe_divide(profile, self.counts.std, xp)
        information = self.sum_by_streak(profile**2, xp)
        positive = information > 0.0
        denominator = xp.sqrt(xp.where(positive, information, 1.0))
        return xp.where(positive, 1.0 / denominator, xp.inf)

    def mean_by_pattern(self, values: RealArray, xp: AnyNamespace) -> RealArray:
        """Return mean point values for each pattern."""
        sums = add_at(xp.zeros((self.n_patterns,), dtype=values.dtype),
                      self.points.index, values)
        counts = add_at(xp.zeros((self.n_patterns,), dtype=values.dtype),
                        self.points.index, xp.ones_like(values))
        return safe_divide(sums, counts, xp)

class PedestalData(State, BaseData):
    """Hold observed photon counts and geometry for independently fitted streaks."""
    points : StreakPoints
    counts : PhotonCounts
    miller : Miller

    @classmethod
    def import_data(cls, data: CrystData, streak_ids: StreakIndices, miller: Miller,
                    detector: Detector, xp: AnyNamespace, *, I_min: float=float('-inf')
                    ) -> 'PedestalData':
        points = streak_ids.to_points(detector)
        counts = PhotonCounts.import_data(data, streak_ids, xp, I_min=I_min)
        return cls(points=points, counts=counts, miller=miller)

class StreakData(State, BaseData):
    scaling  : PedestalData
    modelled : IntensityModel

    @property
    def points(self) -> StreakPoints:
        return self.scaling.points

    @property
    def counts(self) -> PhotonCounts:
        return self.scaling.counts

    @property
    def miller(self) -> Miller:
        return self.scaling.miller

class PedestalState(State, DataContainer):
    """Hold additive background corrections for predicted streaks.

    Attributes:
        pedestal: Additive photon-count pedestal for each streak, with shape
            ``(n_streaks,)``.
    """
    pedestal : RealArray # (n_streaks,)

    @classmethod
    def default(cls, n_streaks: int, xp: AnyNamespace) -> 'PedestalState':
        """Construct a zero pedestal for each streak."""
        return cls(pedestal=xp.zeros((n_streaks,)))

    @property
    def n_streaks(self) -> int:
        """Return the number of independently fitted predicted streaks."""
        return len(self.pedestal)

    @property
    def shape(self) -> Shape:
        return self.pedestal.shape

    def pedestal_at(self, points: StreakPoints) -> RealArray:
        """Return the pedestal associated with each measured point."""
        return self.pedestal[points.streak_id]

class StreakState(State, DataContainer):
    """Hold local photometric parameters for predicted streaks.

    Attributes:
        asinh_hkl: Inverse hyperbolic sine of the streak intensity coefficient for each
            unnormalised profile, with shape ``(n_streaks,)``.
        log_sigma: Natural logarithm of the profile width for each pattern, with shape
            ``(n_patterns,)``.
    """
    asinh_hkl   : RealArray # (n_streaks,)
    log_sigma   : RealArray # (n_patterns,)

    @classmethod
    def default(cls, n_streaks: int, n_patterns: int, sigma: float, xp: AnyNamespace
                ) -> 'StreakState':
        """Construct zero intensity values at a fixed initial width."""
        return cls(asinh_hkl=xp.zeros((n_streaks,)),
                   log_sigma=xp.full((n_patterns,), xp.log(sigma)))

    @classmethod
    def from_data(cls: Type[Self], data: StreakData, sigma: float, weights: RealArray,
                  xp: AnyNamespace) -> Self:
        """Initialise signed profile coefficients by their information-weighted median."""
        previous = xp.zeros((data.n_streaks,))
        log_sigma = xp.full((data.n_patterns,), xp.log(sigma))
        profile = xp.exp(data.modelled.log_profile(log_sigma[data.points.index], xp))
        estimator = WeightedLinearEstimator(group_id=data.points.streak_id, weights=weights)
        intensity = estimator.fit_median(profile, data.counts.signal, previous)
        return cls(asinh_hkl=xp.arcsinh(intensity), log_sigma=log_sigma)

    @property
    def n_streaks(self) -> int:
        """Return the number of independently fitted predicted streaks."""
        return self.asinh_hkl.size

    @property
    def n_patterns(self) -> int:
        """Return the number of patterns in the indexed data."""
        return self.log_sigma.size

    def intensity_at(self, points: StreakPoints) -> RealArray:
        """Return the signed profile coefficient associated with each measured point."""
        xp = self.__array_namespace__()
        return xp.sinh(self.asinh_hkl[points.streak_id])

    def log_sigma_at(self, points: StreakPoints) -> RealArray:
        """Return the log profile width associated with each measured point."""
        return self.log_sigma[points.index]

class RefineStreakState(StreakState):
    """Hold refined position and broadening parameters for predicted streaks.

    Attributes:
        asinh_hkl: Inverse hyperbolic sine of the streak intensity coefficient for each
            unnormalised profile, with shape ``(n_streaks,)``.
        log_sigma: Natural logarithm of the profile width for each pattern,
            with shape ``(n_patterns,)``.
        displacement: Detector-plane ``(x, y)`` displacement in pixels for each streak,
            with shape ``(n_streaks, 2)``.
        pixel_size: Pixel size in metres.
    """
    displacement : RealArray # (n_streaks, 2), detector-plane in pixels
    pixel_size   : float = field(static=True)

    @classmethod
    def default(cls, n_streaks: int, n_patterns: int, sigma: float, pixel_size: float,
                xp: AnyNamespace) -> 'RefineStreakState':
        """Construct zero intensity and displacement values at a fixed initial width."""
        state = StreakState.default(n_streaks, n_patterns, sigma, xp)
        return cls.from_streaks(state, pixel_size)

    @classmethod
    def from_streaks(cls, streaks: StreakState, pixel_size: float) -> 'RefineStreakState':
        """Construct a refined state with zero displacement."""
        xp = streaks.__array_namespace__()
        return cls(asinh_hkl=streaks.asinh_hkl, log_sigma=streaks.log_sigma,
                   displacement=xp.zeros((streaks.n_streaks, 2)), pixel_size=pixel_size)

    def displacement_at(self, points: StreakPoints) -> RealArray:
        """Return the detector-plane displacement associated with each measured point."""
        return self.pixel_size * self.displacement[points.streak_id]

    def points_at(self, points: StreakPoints) -> StreakPoints:
        """Return detector points translated by their per-streak displacement."""
        return points.replace(points=points.points + self.displacement_at(points))

class FullState(State, DataContainer):
    streaks : StreakState
    setup   : BaseSetup

class ReflectionList(State, IndexedContainer):
    """Hold fitted reflection intensities and their conditional uncertainties.

    Attributes:
        index: Nonnegative pattern index per observation, with shape ``(n_observations,)``.
        hkl: Indexed HKL per observation, with shape ``(n_observations, 3)``. Symmetry
            canonicalisation belongs to the merging stage.
        I_hkl: Profile-normalised fitted rate contrasts, with shape ``(n_observations,)``.
        sigma_hkl: Detector-noise standard errors of ``I_hkl``, with shape
            ``(n_observations,)``. Positive infinity denotes zero measurement information.
    """
    index           : IntArray    # (n_observations,) compact index
    hkl             : IntArray    # (n_observations, 3) indexed hkl indices
    I_hkl           : RealArray   # (n_observations,) fitted rate contrasts
    sigma_hkl       : RealArray   # (n_observations,) conditional uncertainties

    @classmethod
    def concat(cls: Type[Self], containers: Iterable[Self],
               monotonic_index: bool=True) -> Self:
        """Combine observations, treating input pattern ranges as distinct by default.

        Set ``monotonic_index=False`` when indices already identify patterns globally.
        Miller indices and observation order are preserved; derive the symmetry-aware
        reflection mapping from the combined list during merging.
        """
        return super().concat(containers, monotonic_index=monotonic_index)

    @classmethod
    def import_miller(cls, miller: Miller, I_hkl: RealArray, sigma_hkl: RealArray
                      ) -> 'ReflectionList':
        """Construct per-streak results with their indexed Miller indices.

        Args:
            miller: Indexed Miller indices.
            I_hkl: Profile-normalised fitted rate contrasts with shape ``(n_streaks,)``.
            sigma_hkl: Conditional standard errors with shape ``(n_streaks,)``.
            xp: Array namespace used for the result.

        Returns:
            Self-contained scaling result aligned with the input streak order.
        """
        return cls(index=miller.index, hkl=miller.hkl_indices, I_hkl=I_hkl,
                   sigma_hkl=sigma_hkl)

    @classmethod
    def import_dataframe(cls, df: pd.DataFrame | pd.Series, frames: IntArray | None,
                         xp: AnyNamespace) -> 'ReflectionList':
        """Import Miller indices, intensities, and uncertainties from a dataframe."""
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

    def to_dataframe(self, frames: IntArray) -> pd.DataFrame:
        """Export Miller indices, intensities, and uncertainties to a dataframe."""
        return pd.DataFrame({'index': asnumpy(frames[self.index]),
                             'h': asnumpy(self.h), 'k': asnumpy(self.k), 'l': asnumpy(self.l),
                             'I_hkl': asnumpy(self.I_hkl), 'sigma_hkl': asnumpy(self.sigma_hkl)})

class MergeData(State, DataContainer):
    """Bind local intensity estimates to patterns and canonical reflections.

    Attributes:
        index: Nonnegative pattern index per observation, with shape ``(n_observations,)``.
        reflection_id: Compact canonical-reflection index per observation, with shape
            ``(n_observations,)``.
        hkl: Canonical Miller indices with shape ``(n_reflections, 3)``.
        I_hkl: Profile-normalised fitted rate contrasts, with shape ``(n_observations,)``.
        sigma_hkl: Detector-noise standard errors of ``I_hkl``, with shape
            ``(n_observations,)``. Positive infinity denotes zero measurement information.
    """
    index           : IntArray    # (n_observations,) compact index
    reflection_id   : IntArray    # (n_observations,) compact reflection ID
    hkl             : IntArray    # (n_reflections, 3) canonical Miller indices
    I_hkl           : RealArray   # (n_observations,) fitted rate contrasts
    sigma_hkl       : RealArray   # (n_observations,) conditional uncertainties

    def __post_init__(self):
        self._reflection_indexer = None
        self._pattern_indexer = None

    @classmethod
    def from_reflections(cls, reflections: ReflectionList,
                         point_group: PointGroup) -> 'MergeData':
        """Bind a reflection list to its symmetry-aware merging domain."""
        xp = reflections.__array_namespace__()
        canonical = point_group.canonical(reflections.hkl, xp)
        hkl, reflection_id = xp.unique(canonical, return_inverse=True, axis=0)
        return cls(index=reflections.index, reflection_id=reflection_id, hkl=hkl,
                   I_hkl=reflections.I_hkl, sigma_hkl=reflections.sigma_hkl)

    @property
    def n_patterns(self) -> int:
        """Return the pattern lookup size, including index gaps; zero when empty."""
        if not self.index.size:
            return 0
        return int(self.index[-1] + 1)

    @property
    def n_reflections(self) -> int:
        """Return the number of canonical reflections."""
        return self.hkl.shape[0]

    @property
    def reflection_indexer(self) -> IndexLookup:
        """Return the observation lookup for canonical reflection IDs."""
        if self._reflection_indexer is None:
            self._reflection_indexer = IndexLookup.build(self.reflection_id)
        return self._reflection_indexer

    @property
    def pattern_indexer(self) -> IndexLookup:
        """Return the observation lookup for pattern indices."""
        if self._pattern_indexer is None:
            self._pattern_indexer = IndexLookup.build(self.index)
        return self._pattern_indexer

    def pattern_at(self, frame: IntSequence) -> 'MergeData':
        """Return observations belonging to the requested patterns.

        The returned data preserves the original reflection IDs and remaps the selected
        pattern indices to a compact range in the requested order. This discrete lookup is
        not JIT-safe.
        """
        positions, new_ids = self.pattern_indexer.get_index(frame)
        return self.replace(index=new_ids, reflection_id=self.reflection_id[positions],
                            hkl=self.hkl, I_hkl=self.I_hkl[positions],
                            sigma_hkl=self.sigma_hkl[positions])

    def reflection_at(self, reflection_id: IntSequence) -> 'MergeData':
        """Return observations belonging to the requested canonical reflections.

        The returned data preserves the original pattern indices and remaps the selected
        reflection IDs to a compact range in the requested order. This discrete lookup is
        not JIT-safe.
        """
        positions, new_ids = self.reflection_indexer.get_index(reflection_id)
        return self.replace(index=self.index[positions], reflection_id=new_ids,
                            hkl=self.hkl[reflection_id], I_hkl=self.I_hkl[positions],
                            sigma_hkl=self.sigma_hkl[positions])

    @property
    def is_informative(self) -> BoolArray:
        """Return observations carrying finite conditional information."""
        xp = self.__array_namespace__()
        return xp.isfinite(self.sigma_hkl)

    @property
    def n_informative(self) -> IntArray:
        """Return the number of informative observations per canonical reflection."""
        xp = self.__array_namespace__()
        return add_at(xp.zeros((self.n_reflections,), dtype=int), self.reflection_id,
                      self.is_informative)

    def median_hkl(self, scale: RealArray) -> RealArray:
        """Return an unweighted upper-median intensity for each reflection."""
        xp = self.__array_namespace__()
        intensity = safe_divide(self.I_hkl, scale[self.index], xp)
        indices = xp.lexsort((intensity, self.reflection_id))
        counts = add_at(xp.zeros((self.n_reflections,), dtype=self.reflection_id.dtype),
                        self.reflection_id, xp.ones_like(self.reflection_id))
        offsets = xp.cumsum(counts) - counts
        return intensity[indices][offsets + counts // 2]

    def fit_intensities(self, scale: RealArray, weights: RealArray,
                        previous: RealArray) -> RealArray:
        """Fit unrestricted merged intensities at fixed pattern scales."""
        estimator = WeightedLinearEstimator(group_id=self.reflection_id, weights=weights)
        return estimator.fit(scale[self.index], self.I_hkl, previous)

    def fit_scales(self, intensity: RealArray, weights: RealArray,
                   previous: RealArray) -> RealArray:
        """Fit unrestricted pattern scales at fixed merged intensities."""
        estimator = WeightedLinearEstimator(group_id=self.index, weights=weights)
        return estimator.fit(intensity[self.reflection_id], self.I_hkl, previous)

    def mean_by_reflection(self, values: RealArray, xp: AnyNamespace) -> RealArray:
        """Return mean point values for each canonical reflection."""
        sums = add_at(xp.zeros((self.n_reflections,), dtype=values.dtype),
                      self.reflection_id, values)
        counts = add_at(xp.zeros((self.n_reflections,), dtype=values.dtype),
                        self.reflection_id, xp.ones_like(values))
        return safe_divide(sums, counts, xp)

    def mean_by_pattern(self, values: RealArray, xp: AnyNamespace) -> RealArray:
        """Return mean point values for each pattern."""
        sums = add_at(xp.zeros((self.n_patterns,), dtype=values.dtype),
                      self.index, values)
        counts = add_at(xp.zeros((self.n_patterns,), dtype=values.dtype),
                        self.index, xp.ones_like(values))
        return safe_divide(sums, counts, xp)

class MergeState(DataContainer, State):
    """Shared intensities and positive illumination scales for one connected component.

    Attributes:
        log_scale: Natural log scales, shape ``(n_patterns,)``, including index gaps.
        I_hkl: Signed intensities, shape ``(n_reflections,)``, ordered by
            :attr:`MergeData.hkl`.
    """
    log_scale: RealArray
    I_hkl: RealArray

    @classmethod
    def from_data(cls, data: MergeData,
                  log_scale: RealArray | None=None) -> 'MergeState':
        xp = data.__array_namespace__()
        if log_scale is None:
            log_scale = xp.zeros((data.n_patterns,))

        scale = xp.exp(log_scale)
        return cls(log_scale=log_scale, I_hkl=data.median_hkl(scale)).normalise()

    def intensity_at(self, data: MergeData) -> RealArray:
        """Return shared intensities in observation order."""
        return self.I_hkl[data.reflection_id]

    def scale_at(self, data: MergeData) -> RealArray:
        """Return positive scales in observation order."""
        xp = self.__array_namespace__()
        return xp.exp(self.log_scale[data.index])

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
