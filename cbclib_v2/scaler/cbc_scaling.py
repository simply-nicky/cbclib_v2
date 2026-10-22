from dataclasses import dataclass
from math import isfinite
from typing import Generic, TypeVar
from .._src.annotations import AnyNamespace, RealArray
from .._src.array_api import broadcast_to, det_to_k, safe_divide
from .._src.data_container import DataContainer
from ..indexer.cbc_data import Miller
from ..indexer.cbc_indexing import CBDSetup
from ..indexer.cbc_pupil import RectangularPupil, SourcePlane
from ..indexer.cbc_setup import ResolvedGeometry, ResolvedSetup
from .cbc_data import (BaseData, IntensityModel, MergedList, OverlapData,
                       PhotonCounts, ReflectionList, ScalerData, PoissonData, StreakData, StreakPoints)
from .cbc_setup import (FullState, MergeState, PedestalState, RefineStreakState, PoissonState,
                        StreakState)
from .cbc_symmetry import PointGroup

@dataclass(frozen=True, unsafe_hash=True)
class ScalerModel(CBDSetup):
    rate_floor: float = 1e-6

    def __post_init__(self):
        if self.rate_floor <= 0.0:
            raise ValueError("rate_floor must be positive")

    def pupil_at(self, points: StreakPoints, setup: ResolvedSetup, xp: AnyNamespace
                 ) -> RectangularPupil:
        kmin, kmax = self.lens.kin_min(setup.geometry, xp), self.lens.kin_max(setup.geometry, xp)
        roi = xp.stack((kmin[..., 1], kmax[..., 1], kmin[..., 0], kmax[..., 0]), axis=-1)
        return RectangularPupil(broadcast_to(roi, points.index, (4,), xp))

    def model_at(self, points: StreakPoints, miller: Miller, setup: ResolvedSetup,
                 xp: AnyNamespace) -> IntensityModel:
        """Construct the physical intensity model at detector points."""
        smp_pos = self.smp_center(points, setup.geometry, xp)
        miller = self.xtal.hkl_to_q(miller, setup.xtal, xp)
        kout = det_to_k(points.points, smp_pos, xp)
        q = broadcast_to(miller.q, points.streak_id, (3,), xp)

        if isinstance(setup.geometry, ResolvedGeometry):
            kin = kout - q
            smp_pos = self.kin_to_sample(points.index, kin, setup.geometry, xp)
            kout = det_to_k(points.points, smp_pos, xp)

        pupil = self.pupil_at(points, setup, xp)
        return IntensityModel(kout=kout, source=SourcePlane.from_q(q=q), pupil=pupil,
                              smp_pos=smp_pos)

    def init_model(self, data: BaseData, setup: ResolvedSetup, xp: AnyNamespace
                   ) -> IntensityModel:
        """Construct the physical intensity model at the observed detector points."""
        return self.model_at(data.points, data.miller, setup, xp)

    def log_rate(self, values: RealArray, xp: AnyNamespace) -> RealArray:
        """Map unconstrained additive predictions to smooth positive log rates."""
        return xp.log(self.rate(values, xp))

    def rate(self, values: RealArray, xp: AnyNamespace) -> RealArray:
        """Map unconstrained additive predictions to smooth positive rates."""
        return self.rate_floor + xp.logaddexp(values, 0.0)

    def rate_slope(self, values: RealArray, xp: AnyNamespace) -> RealArray:
        """Return the derivative of the positive rate with respect to its predictor."""
        return xp.exp(-xp.logaddexp(-values, 0.0))

    def rate_contrast(self, baseline: RealArray, signal: RealArray,
                      xp: AnyNamespace) -> RealArray:
        """Return the signed rate change caused by an additive predictor signal."""
        return self.rate(baseline + signal, xp) - self.rate(baseline, xp)

    def combine_rates(self, log_background: RealArray, log_signal: RealArray, xp: AnyNamespace
                      ) -> RealArray:
        """Return the log rate of additive background and diffraction signal."""
        return xp.logaddexp(log_background, log_signal)

    def poisson_loss(self, counts: PhotonCounts, log_expected: RealArray, xp: AnyNamespace
                     ) -> RealArray:
        return -xp.asarray(counts.I0 * log_expected - xp.exp(log_expected))

    def poisson_deviance(self, counts: PhotonCounts, log_expected: RealArray, xp: AnyNamespace
                         ) -> RealArray:
        """Return pointwise Poisson deviance from the saturated count model."""
        positive = counts.I0 > 0
        log_observed = xp.log(xp.where(positive, counts.I0, 1))
        log_ratio = xp.where(positive, counts.I0 * (log_observed - log_expected), 0.0)
        return 2.0 * (log_ratio - counts.I0 + xp.exp(log_expected))

T_Data = TypeVar('T_Data', bound=BaseData)
T_State = TypeVar('T_State', bound=DataContainer)

class BaseLoss(Generic[T_Data, T_State]):
    """Evaluate a softplus Poisson likelihood from a concrete count predictor."""
    model : ScalerModel

    def predictor(self, data: T_Data, state: T_State, xp: AnyNamespace) -> RealArray:
        """Return the unconstrained additive predictor at each detector point."""
        raise NotImplementedError

    def log_expected(self, data: T_Data, state: T_State, xp: AnyNamespace) -> RealArray:
        """Return expected log counts under the background-plus-signal model."""
        return self.model.log_rate(self.predictor(data, state, xp), xp)

    def poisson_loss(self, data: T_Data, state: T_State, xp: AnyNamespace) -> RealArray:
        """Return pointwise Poisson loss for the concrete count predictor."""
        log_expected = self.log_expected(data, state, xp)
        return self.model.poisson_loss(data.counts, log_expected, xp)

    def poisson_deviance(self, data: T_Data, state: T_State, xp: AnyNamespace) -> RealArray:
        """Return pointwise deviance for the concrete fitted model."""
        log_expected = self.log_expected(data, state, xp)
        return self.model.poisson_deviance(data.counts, log_expected, xp)

    def __call__(self, data: T_Data, state: T_State) -> RealArray:
        xp = state.__array_namespace__()
        return xp.mean(self.poisson_loss(data, state, xp))

    def profile(self, data: T_Data, state: T_State, xp: AnyNamespace) -> RealArray:
        """Return the unnormalised linear profile at each detector point."""
        raise NotImplementedError

    def I_hkl(self, data: T_Data, state: T_State, xp: AnyNamespace) -> RealArray:
        """Return the profile-normalised fitted rate contrast for each reflection."""
        raise NotImplementedError

    def std_hkl(self, data: T_Data, state: T_State, xp: AnyNamespace) -> RealArray:
        """Propagate independent detector noise through the fixed profile-weighted mean."""
        profile = self.profile(data, state, xp)
        variance = data.streak_estimator.sum((profile * data.std)**2, xp)
        return safe_divide(xp.sqrt(variance), data.streak_estimator.sum(profile, xp), xp)

    def to_list(self, data: T_Data, state: T_State, xp: AnyNamespace) -> ReflectionList:
        """Return fitted reflection signals and their conditional standard errors."""
        return ReflectionList.import_miller(miller=data.miller,
                                            I_hkl=self.I_hkl(data, state, xp),
                                            sigma_hkl=self.std_hkl(data, state, xp))

@dataclass(frozen=True, unsafe_hash=True)
class PedestalLoss(BaseLoss[ScalerData | OverlapData, PedestalState]):
    """Fit local pedestal corrections without a diffraction signal."""
    model : ScalerModel

    def profile(self, data: ScalerData | OverlapData, state: PedestalState,
                xp: AnyNamespace) -> RealArray:
        """Return the unit profile of the spatially constant pedestal."""
        return xp.ones(data.points.shape, dtype=state.pedestal.dtype)

    def predictor(self, data: ScalerData | OverlapData, state: PedestalState,
                  xp: AnyNamespace) -> RealArray:
        """Return the background-plus-pedestal predictor."""
        return data.predictor(state.pedestal_at(data.points), xp)

    def I_hkl(self, data: ScalerData | OverlapData, state: PedestalState,
              xp: AnyNamespace) -> RealArray:
        return xp.zeros(state.pedestal.shape, dtype=state.pedestal.dtype)

@dataclass(frozen=True, unsafe_hash=True)
class StreakLoss(BaseLoss[StreakData, StreakState]):
    """Fit a signed profile coefficient at fixed streak geometry."""
    model : ScalerModel

    def profile(self, data: StreakData, state: StreakState,
                xp: AnyNamespace) -> RealArray:
        """Return the fixed-geometry streak profile at each detector point."""
        log_sigma = state.log_sigma_at(data.points)
        return data.modelled.profile(log_sigma, xp)

    def predictor(self, data: StreakData, state: StreakState,
                  xp: AnyNamespace) -> RealArray:
        """Return the background-plus-profile-intensity predictor."""
        signal = state.intensity_at(data.points) * self.profile(data, state, xp)
        return data.predictor(signal, xp)

    def I_hkl(self, data: StreakData, state: StreakState,
              xp: AnyNamespace) -> RealArray:
        """Return the profile-weighted mean of each streak's own fitted rate contrast."""
        profile = self.profile(data, state, xp)
        signal = state.intensity_at(data.points)
        contrast = self.model.rate_contrast(data.background, signal, xp)
        return data.streak_estimator.weighted_mean(contrast, profile, xp)

@dataclass(frozen=True, unsafe_hash=True)
class RefineStreakLoss(StreakLoss):
    """Fit local photometry with per-streak detector-plane displacement."""

    def model_at(self, data: StreakData, state: RefineStreakState, xp: AnyNamespace
                 ) -> IntensityModel:
        """Construct the physical intensity model at detector points."""
        return data.modelled.displace(state.points_at(data.points), xp)

    def profile(self, data: StreakData, state: RefineStreakState,
                xp: AnyNamespace) -> RealArray:
        """Return the displaced and per-streak-broadened profile."""
        modelled = self.model_at(data, state, xp)
        log_sigma = state.log_sigma_at(data.points)
        return modelled.profile(log_sigma, xp)

@dataclass(frozen=True, unsafe_hash=True)
class FullLoss(BaseLoss[ScalerData | OverlapData, FullState]):
    model      : ScalerModel

    def model_at(self, data: ScalerData | OverlapData, state: FullState,
                 xp: AnyNamespace) -> IntensityModel:
        resolved = state.setup.resolve(xp)
        return self.model.model_at(data.points, data.miller, resolved, xp)

    def profile(self, data: ScalerData | OverlapData, state: FullState,
                xp: AnyNamespace) -> RealArray:
        """Return the streak profile under the jointly refined setup."""
        modelled = self.model_at(data, state, xp)
        log_sigma = state.streaks.log_sigma_at(data.points)
        return modelled.profile(log_sigma, xp)

    def predictor(self, data: ScalerData | OverlapData, state: FullState,
                  xp: AnyNamespace) -> RealArray:
        """Return the background-plus-profile-intensity predictor."""
        signal = state.streaks.intensity_at(data.points) * self.profile(data, state, xp)
        return data.predictor(signal, xp)

    def I_hkl(self, data: ScalerData | OverlapData, state: FullState,
              xp: AnyNamespace) -> RealArray:
        profile = self.profile(data, state, xp)
        signal = state.streaks.intensity_at(data.points)
        contrast = self.model.rate_contrast(data.background, signal, xp)
        return data.streak_estimator.weighted_mean(contrast, profile, xp)

@dataclass(frozen=True, eq=False)
class PoissonMerger(BaseLoss[PoissonData, PoissonState]):
    """Merge a scan's detector counts by maximising their Poisson likelihood.

    The signal predictor is pattern scale times shared signed reflection
    coefficient times analytically normalised pupil illumination times the
    geometric profile. Overlapping contributions are added before the existing
    softplus count transformation, and each measured pixel enters the likelihood
    once. Calling the object returns the mean negative log likelihood without
    parameter-independent factorial terms.

    Geometry and background remain fixed; profile widths belong to the fitted state.
    Reflection coefficients are
    signed predictor amplitudes, not profile-normalised count-rate contrasts.
    Numerical evaluation supports JAX compilation; the fixed Legendre basis belongs
    to the scan data and remains outside the differentiated scan state.

    Attributes:
        model: Scaling model owning the softplus rate and Poisson likelihood.
    """
    model : ScalerModel = ScalerModel()

    def profile(self, data: PoissonData, state: PoissonState, xp: AnyNamespace) -> RealArray:
        """Return pupil-weighted geometric profiles in streak-point order."""
        return data.modelled.profile(state.log_sigma_at(data), xp) * state.pupil_at(data)

    def predictor(self, data: PoissonData, state: PoissonState, xp: AnyNamespace) -> RealArray:
        """Return the aggregated background-plus-diffraction predictor per measured pixel."""
        signal = state.scale_at(data) * state.intensity_at(data) * self.profile(data, state, xp)
        return data.predictor(signal, xp)

    def I_hkl(self, data: PoissonData, state: PoissonState, xp: AnyNamespace) -> RealArray:
        profile = self.profile(data, state, xp)
        signal = state.intensity_at(data)
        contrast = self.model.rate_contrast(data.background, signal, xp)
        return data.streak_estimator.weighted_mean(contrast, profile, xp)

@dataclass(frozen=True)
class StudentMerger:
    """Student-t loss for jointly optimised pattern scales and merged intensities.

    Gradient optimisation uses :class:`MergeState`, whose exponential scale parameterisation
    stays positive and whose inverse-sinh intensities allow signed coefficients.
    Conditional uncertainties stay fixed and the
    Student-t tails suppress individual inconsistent observations, not indexing alternatives
    or systematic whole-pattern errors. Mapping construction and initialisation are eager
    caller responsibilities; numerical methods support JAX compilation with fixed shapes.

    Attributes:
        nu: Finite positive Student-t degrees of freedom.
    """
    nu: float = 4.0

    def __post_init__(self):
        if not isfinite(self.nu) or self.nu <= 0.0:
            raise ValueError('nu must be finite and positive')

    def expected(self, data: MergedList, state: MergeState) -> RealArray:
        """Predict fitted intensities in observation order."""
        return state.scale_at(data) * state.intensity_at(data)

    def residuals(self, data: MergedList, state: MergeState) -> RealArray:
        """Return standardised residuals, with zero at unsupported observations."""
        return (data.I_hkl - self.expected(data, state)) / data.sigma_hkl

    def __call__(self, data: MergedList, state: MergeState) -> RealArray:
        """Return the mean Student-t negative log likelihood without fixed constants."""
        xp = data.__array_namespace__()
        residual = self.residuals(data, state)
        return 0.5 * (self.nu + 1.0) * xp.mean(xp.log1p(residual**2 / self.nu))
