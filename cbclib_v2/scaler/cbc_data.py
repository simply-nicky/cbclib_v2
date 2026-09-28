from typing import Iterable, Type
from typing_extensions import Self
import pandas as pd
from .._src.annotations import AnyNamespace, BoolArray, IntArray, IntSequence, RealArray, Shape
from .._src.array_api import add_at, asnumpy, det_to_k, safe_divide
from .._src.crystfel import Detector
from .._src.data_container import ArrayContainer, DataContainer, IndexedContainer
from .._src.data_processing import CrystData
from .._src.state import State, field
from ..indexer.cbc_data import IndexLookup, Miller
from ..indexer.cbc_pupil import BasePupil, SourcePlane
from ..indexer.cbc_setup import BaseSetup

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

    def profiled_information(self, profile: RealArray, point_information: RealArray,
                             xp: AnyNamespace) -> RealArray:
        """Return profile-coefficient information after profiling out a constant pedestal."""
        pedestal = self.sum_by_streak(point_information, xp)
        covariance = self.sum_by_streak(point_information * profile, xp)
        intensity = self.sum_by_streak(point_information * profile**2, xp)
        return intensity - safe_divide(covariance**2, pedestal, xp)

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
                    detector: Detector, xp: AnyNamespace) -> 'PedestalData':
        points = streak_ids.to_points(detector)
        counts = PhotonCounts.import_data(data, streak_ids, xp)
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

class StreakState(PedestalState):
    """Hold local photometric parameters for predicted streaks.

    Attributes:
        pedestal: Additive photon-count pedestal for each streak, with shape
            ``(n_streaks,)``.
        intensity: Signed streak intensity coefficient for each unnormalised profile, with
            shape ``(n_streaks,)``.
        log_sigma: Natural logarithm of the profile width for each pattern, with shape
            ``(n_patterns,)``.
    """
    intensity   : RealArray # (n_streaks,)
    log_sigma   : RealArray # (n_patterns,)

    @classmethod
    def default(cls, n_streaks: int, n_patterns: int, sigma: float, xp: AnyNamespace
                ) -> 'StreakState':
        """Construct zero pedestal and intensity values at a fixed initial width."""
        return cls(pedestal=xp.zeros((n_streaks,)), intensity=xp.zeros((n_streaks,)),
                   log_sigma=xp.full((n_patterns,), xp.log(sigma)))

    @classmethod
    def from_data(cls: Type[Self], data: StreakData, sigma: float, weights: RealArray,
                  xp: AnyNamespace) -> Self:
        """Initialise signed profile coefficients by weighted linear regression."""
        pedestal = xp.zeros((data.n_streaks,))
        previous = xp.zeros((data.n_streaks,))
        log_sigma = xp.full((data.n_patterns,), xp.log(sigma))
        point_log_sigma = log_sigma[data.points.index]
        profile = xp.exp(data.modelled.log_profile(point_log_sigma, xp))
        estimator = WeightedLinearEstimator(group_id=data.points.streak_id, weights=weights)
        intensity = estimator.fit(profile, data.counts.signal, previous)
        return cls(pedestal=pedestal, intensity=intensity, log_sigma=log_sigma)

    @property
    def n_streaks(self) -> int:
        """Return the number of independently fitted predicted streaks."""
        return self.intensity.size

    @property
    def n_patterns(self) -> int:
        """Return the number of patterns in the indexed data."""
        return self.log_sigma.size

    def intensity_at(self, points: StreakPoints) -> RealArray:
        """Return the signed profile coefficient associated with each measured point."""
        return self.intensity[points.streak_id]

    def log_sigma_at(self, points: StreakPoints) -> RealArray:
        """Return the log profile width associated with each measured point."""
        return self.log_sigma[points.index]

class RefineStreakState(StreakState):
    """Hold refined position and broadening parameters for predicted streaks.

    Attributes:
        pedestal: Additive photon-count pedestal for each streak, with shape
            ``(n_streaks,)``.
        intensity: Signed streak intensity coefficient for each unnormalised profile, with
            shape ``(n_streaks,)``.
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
        """Construct a zero displacement and variance for each streak."""
        state = StreakState.default(n_streaks, n_patterns, sigma, xp)
        return cls.from_streaks(state, pixel_size)

    @classmethod
    def from_streaks(cls, streaks: StreakState, pixel_size: float) -> 'RefineStreakState':
        """Construct a refined state with zero displacement."""
        xp = streaks.__array_namespace__()
        return cls(pedestal=streaks.pedestal, intensity=streaks.intensity,
                   log_sigma=streaks.log_sigma,
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
        I_hkl: Fitted intensities, with shape ``(n_observations,)``.
        sigma_hkl: Conditional Poisson standard errors of ``I_hkl``, with shape
            ``(n_observations,)``. Positive infinity denotes zero conditional information.
    """
    index           : IntArray    # (n_observations,) compact index
    hkl             : IntArray    # (n_observations, 3) indexed hkl indices
    I_hkl           : RealArray   # (n_observations,) fitted intensities
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
            I_hkl: Fitted integrated intensities with shape ``(n_streaks,)``.
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

    def reflections(self) -> Reflections:
        """Return unique HKLs and the reflection ID of each observation.

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
        """Export Miller indices, intensities, and uncertainties to a dataframe."""
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
