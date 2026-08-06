from dataclasses import dataclass
from .._src.annotations import AnyNamespace, RealArray
from .._src.array_api import add_at, broadcast_to, det_to_k
from ..indexer.cbc_indexing import CBDSetup
from ..indexer.cbc_pupil import Rectangle, SourcePlane
from ..indexer.cbc_setup import ResolvedGeometry, ResolvedSetup
from .cbc_data import IntensityModel, PhotonCounts, ScalerData, ScalerResult, ScalerState
from .cbc_symmetry import PointGroup

@dataclass(frozen=True, unsafe_hash=True)
class ScalerModel(CBDSetup):
    rate_floor: float = 1e-6

    def __post_init__(self) -> None:
        if self.rate_floor <= 0.0:
            raise ValueError("rate_floor must be positive")

    def init_model(self, data: ScalerData, setup: ResolvedSetup,
                   xp: AnyNamespace) -> IntensityModel:
        smp_pos = self.smp_center(data.points.index, setup.geometry, xp)
        miller = self.xtal.hkl_to_q(data.miller, setup.xtal, xp)
        kout = det_to_k(data.points.points, smp_pos, xp)
        q = broadcast_to(miller.q, data.points.streak_id, (3,), xp)

        if isinstance(setup.geometry, ResolvedGeometry):
            kin = kout - q
            smp_pos = self.kin_to_sample(kin, data.points.index, setup.geometry, xp)
            kout = det_to_k(data.points.points, smp_pos, xp)

        pupil = self.lens.pupil(setup.geometry, xp)
        roi = broadcast_to(pupil.roi, data.points.index, (4,), xp)
        return IntensityModel(kout=kout, source=SourcePlane.from_q(q=q),
                              pupil=Rectangle(roi=roi))

    def log_signal(self, modelled: IntensityModel, data: ScalerData, state: ScalerState,
                   xp: AnyNamespace) -> RealArray:
        """Return the log of the reflection intensity multiplied by its profile."""
        log_hkl = state.log_at(data.points, data.reflections)
        log_profile = modelled.log_profile(state.log_sigma_at(data.points), xp)
        return log_hkl + log_profile

    def log_expected(self, counts: PhotonCounts, log_signal: RealArray,
                     xp: AnyNamespace) -> RealArray:
        """Return the log of the total signal, background, and floor count rate."""
        rate_floor = xp.asarray(self.rate_floor, dtype=log_signal.dtype)
        background = xp.clip(counts.background, 0.0, xp.inf)
        log_background = xp.log(background + rate_floor)
        return xp.logaddexp(log_signal, log_background)

    def poisson_loss(self, counts: PhotonCounts, log_expected: RealArray,
                     xp: AnyNamespace) -> RealArray:
        y_hat = xp.exp(log_expected)
        log_likelihood = counts.I0 * log_expected - y_hat
        return -xp.asarray(log_likelihood)

    def std_hkl(self, modelled: IntensityModel, data: ScalerData, state: ScalerState,
                xp: AnyNamespace) -> RealArray:
        """Return conditional Poisson standard errors of fitted intensities.

        Geometry and profile-width parameters are held fixed. The uncertainty is the inverse
        square root of the observed likelihood curvature with respect to each linear
        reflection intensity.

        Returns:
            Standard errors with shape ``(n_reflections,)``. Reflections with zero observed
            information have infinite uncertainty.
        """
        log_profile = modelled.log_profile(state.log_sigma_at(data.points), xp)
        log_signal = self.log_signal(modelled, data, state, xp)
        log_expected = self.log_expected(data.counts, log_signal, xp)
        point_information = data.counts.I0 * xp.exp(2.0 * (log_profile - log_expected))
        information = xp.zeros((len(data.reflections),), dtype=log_profile.dtype)
        information = add_at(information, data.reflections.at(data.points), point_information)
        positive = information > 0.0
        denominator = xp.sqrt(xp.where(positive, information, 1.0))
        return xp.where(positive, 1.0 / denominator, xp.inf)

    def init_result(self, modelled: IntensityModel, data: ScalerData, state: ScalerState,
                    point_group: PointGroup, xp: AnyNamespace) -> ScalerResult:
        """Return fitted reflection intensities and their conditional standard errors."""
        return ScalerResult.from_data(data=data, point_group=point_group,
                                      I_hkl=xp.exp(state.log_hkl),
                                      sigma_hkl=self.std_hkl(modelled, data, state, xp), xp=xp)

@dataclass(frozen=True, unsafe_hash=True)
class ScalerLoss:
    model      : ScalerModel

    def __call__(self, modelled: IntensityModel, data: ScalerData,
                 state: ScalerState) -> RealArray:
        xp = state.__array_namespace__()
        log_signal = self.model.log_signal(modelled, data, state, xp)
        log_expected = self.model.log_expected(data.counts, log_signal, xp)
        loss = self.model.poisson_loss(data.counts, log_expected, xp)
        return xp.mean(loss)

@dataclass(frozen=True, unsafe_hash=True)
class SetupLoss:
    model      : ScalerModel

    def __call__(self, data: ScalerData, state: ScalerState,
                 setup: ResolvedSetup) -> RealArray:
        xp = setup.__array_namespace__()
        modelled = self.model.init_model(data, setup, xp)
        log_signal = self.model.log_signal(modelled, data, state, xp)
        log_expected = self.model.log_expected(data.counts, log_signal, xp)
        loss = self.model.poisson_loss(data.counts, log_expected, xp)
        return xp.mean(loss)
