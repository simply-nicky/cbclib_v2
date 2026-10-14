from dataclasses import dataclass
from typing import ClassVar, Iterable, List, Literal, overload, Tuple, Type
from typing_extensions import Self
import pandas as pd
from .._src.annotations import AnyNamespace, BoolArray, IntArray, IntSequence, RealArray, Shape
from .._src.array_api import add_at, asnumpy, det_to_k, safe_divide
from .._src.crystfel import Detector
from .._src.data_container import ArrayContainer, DataContainer, IndexedContainer, IndexLookup
from .._src.data_processing import CrystData
from .._src.state import State, field
from ..indexer.cbc_data import Miller
from ..indexer.cbc_pupil import RectangularPupil, SourcePlane
from ..indexer.cbc_setup import BaseSetup
from .cbc_symmetry import PointGroup

@dataclass(frozen=True)
class LegendreFactory:
    """Build pupil bases from globally cached exact shifted Legendre coefficients.

    The process-wide cache contains unnormalised Python-integer rows in ascending
    powers of the unit coordinate. Degrees zero through four are predefined;
    higher degrees are generated once by the shifted Legendre recurrence.
    """
    cached: ClassVar[List[Tuple[int, ...]]] = [
        (1,),
        (-1, 2),
        (1, -6, 6),
        (-1, 12, -30, 20),
        (1, -20, 90, -140, 70),
    ]

    @classmethod
    def coefficients(cls, degree: int) -> Tuple[int, ...]:
        """Return the cached exact coefficient row for a non-negative degree."""
        if degree < 0:
            raise ValueError("Legendre degree must be non-negative")
        while len(cls.cached) <= degree:
            n = len(cls.cached) - 1
            current = cls.cached[n] + (0,)
            previous = cls.cached[n - 1] + (0, 0)
            shifted = (0,) + cls.cached[n]
            # Each recurrence numerator is exactly divisible by n + 1.
            row = tuple(((2 * n + 1) * (2 * shifted[k] - current[k])
                         - n * previous[k]) // (n + 1) for k in range(n + 2))
            cls.cached.append(row)
        return cls.cached[degree]

    @classmethod
    def create(cls, degree: int, xp: AnyNamespace) -> 'LegendreBasis':
        """Create a normalised backend basis from the shared integer cache.

        Args:
            degree: Maximum non-negative total degree ``m + n``.
            xp: Array namespace in which the basis will be evaluated.
        """
        cls.coefficients(degree)
        coefficients = xp.asarray([
            [float(value) for value in cls.cached[n]] + [0.0] * (degree - n)
            for n in range(degree + 1)])
        normalisation = xp.sqrt(2.0 * xp.arange(degree + 1) + 1.0)
        total, n = xp.meshgrid(xp.arange(degree + 1, dtype=int),
                              xp.arange(degree + 1, dtype=int), indexing='ij')
        modes = xp.stack((total - n, n), axis=-1)[n <= total]
        return LegendreBasis(coefficients=xp.asarray(coefficients * normalisation[:, None]),
                             modes=modes)

class LegendreBasis(State, DataContainer):
    """Reusable orthonormal shifted Legendre products on the unit-square pupil.

    Modes use uniform area measure on ``[0, 1] ** 2`` and include the constant.
    Each product is ``sqrt((2*m + 1)*(2*n + 1))*P_m(2*u - 1)*P_n(2*v - 1)``.
    Construct once with :meth:`LegendreFactory.create` and reuse at different coordinates.

    Attributes:
        coefficients: Normalised shifted Legendre coefficients, shape
            ``(degree + 1, degree + 1)``; rows index polynomial degree and columns
            index ascending powers of the unit coordinate.
        modes: Degree pairs ``(m, n)``, shape ``(n_modes, 2)``, ordered by total
            degree, then descending u degree.
    """
    coefficients: RealArray
    modes: IntArray

    @property
    def degree(self) -> int:
        """Return the maximum polynomial degree represented by the coefficients."""
        return self.coefficients.shape[0] - 1

    def evaluate(self, points: RealArray) -> RealArray:
        """Evaluate every mode using the stored coefficients without Python loops.

        Args:
            points: Coordinates ``(u, v)`` with shape ``(..., 2)``, in the same
                array namespace as the basis.

        Returns:
            Values with shape ``(..., n_modes)``, ordered as :attr:`modes`.
            No clipping or aperture mask is applied to the coordinates.
        """
        xp = self.__array_namespace__()
        powers = points[..., None]**xp.arange(self.degree + 1, dtype=int)
        polynomials = powers @ self.coefficients.T
        return xp.asarray(polynomials[..., 0, self.modes[:, 0]]
                          * polynomials[..., 1, self.modes[:, 1]])

class IntensityModel(State, DataContainer):
    kout        : RealArray         # (n_points, 3) endpoint kout vectors of the streak line
    source      : SourcePlane       # (n_points,) source-plane support
    pupil       : RectangularPupil  # (n_points or 1,) selected pupil support
    smp_pos     : RealArray         # (n_points, 3) sample position in metres, used for displacement

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

    def profile(self, log_sigma: RealArray, xp: AnyNamespace) -> RealArray:
        """Return Gaussian attenuation by source-plane and pupil-support distance."""
        inv_sigma_sq = xp.exp(-2.0 * log_sigma)
        return xp.exp(-0.5 * self.distance_sq * inv_sigma_sq)

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
        return cls(index=xp.asarray(df['index']), y=xp.asarray(df['y']),
                   x=xp.asarray(df['x']), streak_id=xp.asarray(df['streak_id']))

    @property
    def shape(self) -> Shape:
        return self.y.shape

    @property
    def xy(self) -> IntArray:
        xp = self.__array_namespace__()
        return xp.stack((self.x, self.y), axis=-1)

    def unique(self) -> Tuple['StreakIndices', IntArray]:
        """Return unique frame-pixels and a pixel ID for each streak contribution.

        Pixel identity includes the frame. This discrete construction is eager, while
        the returned mapping can be used inside compiled losses.
        """
        xp = self.__array_namespace__()
        pixels = xp.stack((self.index, self.y, self.x), axis=-1)
        _, positions, inverse = xp.unique(pixels, axis=0, return_index=True,
                                          return_inverse=True)
        return self[positions], inverse.reshape(self.shape)

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
    def background(self) -> RealArray:
        """Return background counts in streak-point order."""
        raise NotImplementedError

    @property
    def signal(self) -> RealArray:
        """Return measured signal in streak-point order for initialisation."""
        raise NotImplementedError

    @property
    def std(self) -> RealArray:
        """Return detector noise in streak-point order."""
        raise NotImplementedError

    def predictor(self, signal: RealArray, xp: AnyNamespace) -> RealArray:
        """Combine per-streak signals with background in measured-pixel order."""
        raise NotImplementedError("Concrete data classes must implement predictor.")

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
        profile = safe_divide(profile, self.std, xp)
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

class ScalerData(State, BaseData):
    """Hold matching geometry and count rows for non-overlapping streak pixels."""
    points : StreakPoints
    counts : PhotonCounts
    miller : Miller

    @property
    def background(self) -> RealArray:
        """Return background counts in streak-point order."""
        return self.counts.background

    @property
    def signal(self) -> RealArray:
        """Return measured signal in streak-point order."""
        return self.counts.signal

    @property
    def std(self) -> RealArray:
        """Return detector noise in streak-point order."""
        return self.counts.std

    def predictor(self, signal: RealArray, xp: AnyNamespace) -> RealArray:
        """Add the streak signal to the background at each measured pixel."""
        return self.counts.background + signal

    @classmethod
    def import_data(cls, data: CrystData, streak_ids: StreakIndices, miller: Miller,
                    detector: Detector, xp: AnyNamespace, *, I_min: float=float('-inf')
                    ) -> 'ScalerData':
        points = streak_ids.to_points(detector)
        counts = PhotonCounts.import_data(data, streak_ids, xp, I_min=I_min)
        return cls(points=points, counts=counts, miller=miller)

class OverlapData(State, BaseData):
    """Hold unique pixel counts and geometry for overlapping streak contributions.

    Attributes:
        points: Geometry for each streak-pixel contribution, shape ``(n_points,)``.
        counts: Counts, background and noise for unique frame-pixels, shape ``(n_pixels,)``.
        miller: Miller indices for the predicted streaks.
        pixel_id: Unique pixel ID for each contribution, shape ``(n_points,)``.
    """
    points   : StreakPoints
    counts   : PhotonCounts
    miller   : Miller
    pixel_id : IntArray

    @property
    def background(self) -> RealArray:
        """Return background counts in streak-point order."""
        return self.counts.background[self.pixel_id]

    @property
    def signal(self) -> RealArray:
        """Return measured signal in streak-point order."""
        return self.counts.signal[self.pixel_id]

    @property
    def std(self) -> RealArray:
        """Return detector noise in streak-point order."""
        return self.counts.std[self.pixel_id]

    def predictor(self, signal: RealArray, xp: AnyNamespace) -> RealArray:
        """Sum streak contributions at each pixel and add its background once."""
        signal = add_at(xp.zeros(self.counts.shape, dtype=signal.dtype), self.pixel_id, signal)
        return self.counts.background + signal

    @classmethod
    def import_data(cls, data: CrystData, streak_ids: StreakIndices, miller: Miller,
                    detector: Detector, xp: AnyNamespace, *, I_min: float=float('-inf')
                    ) -> 'OverlapData':
        points = streak_ids.to_points(detector)
        pixels, pixel_id = streak_ids.unique()
        counts = PhotonCounts.import_data(data, pixels, xp, I_min=I_min)
        return cls(points=points, counts=counts, miller=miller, pixel_id=pixel_id)

class StreakData(State, BaseData):
    scaling  : ScalerData | OverlapData
    modelled : IntensityModel

    @property
    def points(self) -> StreakPoints:
        return self.scaling.points

    @property
    def counts(self) -> PhotonCounts:
        return self.scaling.counts

    @property
    def background(self) -> RealArray:
        """Return background counts in streak-point order."""
        return self.scaling.background

    @property
    def signal(self) -> RealArray:
        """Return measured signal in streak-point order."""
        return self.scaling.signal

    @property
    def std(self) -> RealArray:
        """Return detector noise in streak-point order."""
        return self.scaling.std

    def predictor(self, signal: RealArray, xp: AnyNamespace) -> RealArray:
        """Combine signals according to the contained scaling data's pixel representation."""
        return self.scaling.predictor(signal, xp)

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
        profile = data.modelled.profile(log_sigma[data.points.index], xp)
        estimator = WeightedLinearEstimator(group_id=data.points.streak_id, weights=weights)
        intensity = estimator.fit_median(profile, data.signal, previous)
        return cls(asinh_hkl=xp.arcsinh(intensity), log_sigma=log_sigma)

    @overload
    @classmethod
    def import_dataframe(cls, df: pd.DataFrame | pd.Series, xp: AnyNamespace, *,
                         index: Literal[False]=False) -> 'StreakState': ...

    @overload
    @classmethod
    def import_dataframe(cls, df: pd.DataFrame | pd.Series, xp: AnyNamespace, *,
                         index: Literal[True]) -> Tuple[IntArray, 'StreakState']: ...

    @classmethod
    def import_dataframe(cls, df: pd.DataFrame | pd.Series, xp: AnyNamespace, *,
                         index: bool=False
                         ) -> 'StreakState' | Tuple[IntArray, 'StreakState']:
        lookup = IndexLookup.build(xp.asarray(df['index']))
        state = cls(asinh_hkl=xp.asarray(df['asinh_hkl']),
                    log_sigma=xp.asarray(df['log_sigma'])[lookup.indices()])
        if index:
            return lookup.unique, state
        return state

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

    def to_dataframe(self, index: IntArray, frames: IntArray) -> pd.DataFrame:
        """Export photometric parameters for each predicted streak."""
        if index.size != self.n_streaks or frames.size != self.n_patterns:
            raise ValueError("Index and frame arrays must match the streak and pattern sizes.")
        return pd.DataFrame({'index': asnumpy(frames[index]),
                             'asinh_hkl': asnumpy(self.asinh_hkl),
                             'log_sigma': asnumpy(self.log_sigma[index])})

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

    @overload
    @classmethod
    def import_dataframe(cls, df: pd.DataFrame | pd.Series, pixel_size: float,
                         xp: AnyNamespace, *, index: Literal[False]=False,
                         ) -> 'RefineStreakState': ...

    @overload
    @classmethod
    def import_dataframe(cls, df: pd.DataFrame | pd.Series, pixel_size: float,
                         xp: AnyNamespace, *, index: Literal[True]
                         ) -> Tuple[IntArray, 'RefineStreakState']: ...

    @classmethod
    def import_dataframe(cls, df: pd.DataFrame | pd.Series, pixel_size: float,
                         xp: AnyNamespace, *, index: bool=False
                         ) -> 'RefineStreakState' | Tuple[IntArray, 'RefineStreakState']:
        frames, streaks = StreakState.import_dataframe(df, index=True, xp=xp)
        displacement = xp.stack((xp.asarray(df['displacement_x']),
                                 xp.asarray(df['displacement_y'])), axis=-1)
        state = cls(asinh_hkl=streaks.asinh_hkl, log_sigma=streaks.log_sigma,
                    displacement=displacement, pixel_size=pixel_size)
        if index:
            return frames, state
        return state

    @classmethod
    def has_displacements(cls, df: pd.DataFrame | pd.Series) -> bool:
        """Return whether the dataframe contains per-streak displacements."""
        return 'displacement_x' in df and 'displacement_y' in df

    def displacement_at(self, points: StreakPoints) -> RealArray:
        """Return the detector-plane displacement associated with each measured point."""
        return self.pixel_size * self.displacement[points.streak_id]

    def points_at(self, points: StreakPoints) -> StreakPoints:
        """Return detector points translated by their per-streak displacement."""
        return points.replace(points=points.points + self.displacement_at(points))

    def to_dataframe(self, index: IntArray, frames: IntArray) -> pd.DataFrame:
        """Export photometric and displacement parameters for each predicted streak."""
        df = super().to_dataframe(index, frames)
        df['displacement_x'] = asnumpy(self.displacement[..., 0])
        df['displacement_y'] = asnumpy(self.displacement[..., 1])
        return df

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
