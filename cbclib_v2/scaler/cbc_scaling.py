from dataclasses import dataclass
from .._src.annotations import AnyNamespace, RealArray
from .._src.array_api import broadcast_to
from ..indexer.cbc_indexing import CBDSetup
from ..indexer.cbc_pupil import Rectangle, SourcePlane
from ..indexer.cbc_setup import ResolvedSetup
from .cbc_data import (IntensityModel, PatternData, PhotonCounts, ScalerState, ScalerData,
                       StreakPoints)

@dataclass(frozen=True, unsafe_hash=True)
class ScalerModel(CBDSetup):
    rate_floor: float = 1e-6

    def __post_init__(self) -> None:
        if self.rate_floor <= 0.0:
            raise ValueError("rate_floor must be positive")

    def init_model(self, data: PatternData, setup: ResolvedSetup,
                   xp: AnyNamespace) -> IntensityModel:
        miller = self.xtal.hkl_to_q(data.miller, setup.xtal, xp)
        kout = self.points_to_kout(data.points, setup.geometry, xp)
        q = broadcast_to(miller.q, data.points.streak_id, (3,), xp)
        pupil = self.lens.pupil(setup.geometry.lens, xp)
        roi = broadcast_to(pupil.roi, data.points.index, (4,), xp)
        return IntensityModel(kout=kout, source=SourcePlane.from_q(q=q),
                              pupil=Rectangle(roi=roi))

    def log_expected(self, modelled: IntensityModel, points: StreakPoints,
                     state: ScalerState, xp: AnyNamespace) -> RealArray:
        return state.log_at(points) + modelled.log_profile(state.log_sigma_at(points), xp)

    def poisson_loss(self, counts: PhotonCounts, log_expected: RealArray,
                     xp: AnyNamespace) -> RealArray:
        rate_floor = xp.asarray(self.rate_floor, dtype=log_expected.dtype)
        background = xp.clip(counts.background, 0.0, xp.inf)
        log_background = xp.log(background + rate_floor)
        log_y_hat = xp.logaddexp(log_expected, log_background)
        y_hat = xp.exp(log_y_hat)
        log_likelihood = counts.I0 * log_y_hat - y_hat
        return -xp.asarray(log_likelihood)

    def scaler_data(self, data: PatternData, setup: ResolvedSetup,
                     xp: AnyNamespace) -> ScalerData:
        modelled = self.init_model(data, setup, xp)
        return ScalerData(points=data.points, counts=data.counts, modelled=modelled)

@dataclass(frozen=True, unsafe_hash=True)
class ScalerLoss:
    model      : ScalerModel

    def __call__(self, data: ScalerData, state: ScalerState) -> RealArray:
        xp = state.__array_namespace__()
        log_expected = self.model.log_expected(data.modelled, data.points, state, xp)
        loss = self.model.poisson_loss(data.counts, log_expected, xp)
        return xp.mean(loss)

@dataclass(frozen=True, unsafe_hash=True)
class SetupLoss:
    model      : ScalerModel

    def __call__(self, data: PatternData, state: ScalerState, setup: ResolvedSetup) -> RealArray:
        xp = setup.__array_namespace__()
        modelled = self.model.init_model(data, setup, xp)
        log_expected = self.model.log_expected(modelled, data.points, state, xp)
        loss = self.model.poisson_loss(data.counts, log_expected, xp)
        return xp.mean(loss)
