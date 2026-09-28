from typing import Callable, Tuple, cast
import pytest
from jax import jit, tree, value_and_grad
from cbclib_v2.annotations import IntArray, JaxNamespace, JaxNumPy, NumPy, NumPyNamespace, RealArray
from cbclib_v2.indexer import (FixedPupilSetup, Miller, Rectangle, ResolvedSetup, SourcePlane)
from cbclib_v2.scaler import (FullLoss, FullState, IntensityModel, PhotonCounts, ReflectionList,
                              PedestalData, ScalerModel, StreakState, StreakData, StreakLoss,
                              StreakPoints)
from cbclib_v2.test_util import check_close, TestSetup

StreakLossGradFn = Callable[[StreakData, StreakState], Tuple[RealArray, StreakState]]
FullLossGradFn = Callable[[PedestalData, FullState], Tuple[RealArray, FullState]]

class TestIntensityModel:
    @pytest.fixture
    def xp(self) -> NumPyNamespace:
        return NumPy

    def model(self, kin: RealArray, q: RealArray, pupil: Rectangle) -> IntensityModel:
        return IntensityModel(kout=kin + q, source=SourcePlane.from_q(q=q), pupil=pupil,
                              smp_pos=kin * 0.0)

    def test_source_distance(self, xp: NumPyNamespace):
        kin = xp.asarray([1.0, 0.0, 0.0])
        q = xp.asarray([2.0, 0.0, 0.0])
        pupil = Rectangle(roi=xp.asarray([-1.0, 1.0, -1.0, 1.0]))
        model = self.model(kin, q, pupil)

        # Unit width reduces radial attenuation to half the negative squared distance.
        expected = -0.5 * model.distance_sq
        check_close(model.log_radial(xp.asarray(0.0), xp), expected)

    def test_outside_pupil(self, xp: NumPyNamespace):
        kin = xp.asarray([0.2, 0.0, xp.sqrt(0.96)])
        q = xp.zeros(3)
        pupil = Rectangle(roi=xp.asarray([-0.1, 0.1, -0.1, 0.1]))
        model = self.model(kin, q, pupil)

        profile = xp.exp(model.log_profile(xp.asarray(0.0), xp))

        assert xp.all(profile > 0.0)
        assert xp.all(profile < 1.0)

    def test_log_profile(self, xp: NumPyNamespace):
        kin = xp.asarray([0.2, 0.0, xp.sqrt(0.96)])
        q = xp.asarray([0.1, 0.0, 0.0])
        pupil = Rectangle(roi=xp.asarray([-0.1, 0.1, -0.1, 0.1]))
        model = self.model(kin, q, pupil)
        expected = xp.exp(-0.5 * model.distance_sq)

        check_close(xp.exp(model.log_profile(xp.asarray(0.0), xp)), expected)

class TestPoissonLoss:
    @pytest.fixture
    def xp(self) -> NumPyNamespace:
        return NumPy

    def test_signal_and_background(self, xp: NumPyNamespace):
        model = ScalerModel()
        counts = PhotonCounts(I0=xp.asarray([4]), background=xp.asarray([3.0]))
        log_expected = model.log_rate(counts.background + xp.asarray([2.0]), xp)
        expected_rate = xp.exp(log_expected)

        loss = model.poisson_loss(counts, log_expected, xp)

        # The Poisson negative log-likelihood combines the full expected rate and counts.
        check_close(loss, expected_rate - counts.I0 * log_expected)

    def test_signal_underflow(self, xp: NumPyNamespace):
        model = ScalerModel()
        counts = PhotonCounts(I0=xp.asarray([1]), background=xp.asarray([0.0]))
        log_expected = model.log_rate(xp.asarray([-1000.0]), xp)

        loss = model.poisson_loss(counts, log_expected, xp)

        assert xp.all(xp.isfinite(loss))

    def test_rate_floor(self, xp: NumPyNamespace):
        model = ScalerModel()
        counts = PhotonCounts(I0=xp.asarray([0]), background=xp.asarray([0.0]))
        log_expected = model.log_rate(counts.background, xp)

        loss = model.poisson_loss(counts, log_expected, xp)

        # Softplus rounds the positivity boundary instead of creating a clipped corner.
        expected_rate = model.rate_floor + xp.logaddexp(counts.background, 0.0)
        check_close(loss, expected_rate)

    def test_custom_rate_floor(self, xp: NumPyNamespace):
        model = ScalerModel(rate_floor=0.25)
        counts = PhotonCounts(I0=xp.asarray([0]), background=xp.asarray([0.0]))
        log_expected = model.log_rate(counts.background, xp)

        loss = model.poisson_loss(counts, log_expected, xp)

        # The floor remains an additive lower bound on the smoothly transformed rate.
        expected_rate = model.rate_floor + xp.logaddexp(counts.background, 0.0)
        check_close(loss, expected_rate)

    def test_invalid_rate_floor(self):
        with pytest.raises(ValueError, match="rate_floor must be positive"):
            ScalerModel(rate_floor=0.0)

class TestLossGradients:
    @pytest.fixture
    def xp(self) -> JaxNamespace:
        return JaxNumPy

    @pytest.fixture
    def model(self) -> ScalerModel:
        return ScalerModel()

    @pytest.fixture
    def setup(self, xp: JaxNamespace) -> FixedPupilSetup:
        return FixedPupilSetup(TestSetup.xtal(xp), TestSetup.fixed_pupil_geometry(xp))

    @pytest.fixture
    def resolved(self, setup: FixedPupilSetup, xp: JaxNamespace) -> ResolvedSetup:
        return setup.resolve(xp)

    @pytest.fixture
    def index(self, xp: JaxNamespace) -> IntArray:
        return xp.asarray([0, 0, 1, 1])

    @pytest.fixture
    def streak_points(self, index: IntArray, xp: JaxNamespace) -> StreakPoints:
        return StreakPoints(index=index, streak_id=xp.asarray([0, 0, 1, 1]),
                            points=xp.asarray([[0.12, 0.14], [0.13, 0.15],
                                               [0.18, 0.16], [0.19, 0.17]]))

    @pytest.fixture
    def miller(self, index: IntArray, xp: JaxNamespace) -> Miller:
        return Miller(index=xp.unique(index), hkl=xp.asarray([[1, 0, 0], [0, 1, 0]]))

    @pytest.fixture
    def counts(self, xp: JaxNamespace) -> PhotonCounts:
        return PhotonCounts(I0=xp.asarray([11, 13, 17, 19]),
                            background=xp.asarray([2.0, 3.0, 2.0, 4.0]))

    @pytest.fixture
    def data(self, streak_points: StreakPoints, counts: PhotonCounts,
             miller: Miller) -> PedestalData:
        return PedestalData(points=streak_points, counts=counts, miller=miller)

    @pytest.fixture
    def modelled(self, model: ScalerModel, data: PedestalData,
                 resolved: ResolvedSetup, xp: JaxNamespace) -> IntensityModel:
        return model.init_model(data, resolved, xp)

    @pytest.fixture
    def state(self, data: PedestalData, xp: JaxNamespace) -> StreakState:
        return StreakState.default(n_streaks=data.n_streaks, n_patterns=data.n_patterns,
                                   sigma=0.02, xp=xp)

    def check_gradient(self, value: RealArray, gradient: StreakState | FullState,
                       xp: JaxNamespace):
        leaves = tree.leaves(gradient)

        assert xp.isfinite(value)
        assert all(xp.all(xp.isfinite(leaf)) for leaf in leaves)
        assert any(xp.any(leaf != 0.0) for leaf in leaves)

    def test_scaler(self, model: ScalerModel, modelled: IntensityModel, data: PedestalData,
                    state: StreakState, xp: JaxNamespace):
        loss = StreakLoss(model)
        loss_grad_fn = cast(StreakLossGradFn, jit(value_and_grad(loss, argnums=1)))
        value, gradient = loss_grad_fn(StreakData(data, modelled), state)

        self.check_gradient(value, gradient, xp)

    def test_setup(self, model: ScalerModel, data: PedestalData, state: StreakState,
                   setup: FixedPupilSetup, xp: JaxNamespace):
        loss = FullLoss(model)
        loss_grad_fn = cast(FullLossGradFn, jit(value_and_grad(loss, argnums=1)))
        value, gradient = loss_grad_fn(data, FullState(state, setup))

        self.check_gradient(value, gradient, xp)

class TestStreakScaling:
    @pytest.fixture
    def xp(self) -> JaxNamespace:
        return JaxNumPy

    @pytest.fixture
    def model(self) -> ScalerModel:
        return ScalerModel()

    @pytest.fixture
    def setup(self, xp: JaxNamespace) -> ResolvedSetup:
        initial = FixedPupilSetup(TestSetup.xtal(xp), TestSetup.fixed_pupil_geometry(xp))
        return initial.resolve(xp)

    @pytest.fixture
    def streak_points(self, xp: JaxNamespace) -> StreakPoints:
        return StreakPoints(index=xp.asarray([0, 0, 0, 0, 1, 1, 1, 1]),
                            streak_id=xp.asarray([0, 0, 1, 1, 2, 2, 3, 3]),
                            points=xp.asarray([[0.12, 0.14], [0.13, 0.15],
                                               [0.14, 0.12], [0.15, 0.13],
                                               [0.18, 0.16], [0.19, 0.17],
                                               [0.16, 0.18], [0.17, 0.19]]))

    @pytest.fixture
    def miller(self, xp: JaxNamespace) -> Miller:
        return Miller(index=xp.asarray([0, 0, 1, 1]),
                      hkl=xp.asarray([[1, 0, 0], [0, 1, 0],
                                      [1, 0, 0], [0, 1, 0]]))

    @pytest.fixture
    def counts(self, xp: JaxNamespace) -> PhotonCounts:
        return PhotonCounts(I0=xp.asarray([1, 1, 20, 24, 30, 36, 50, 58]),
                            background=xp.full((8,), 2.0))

    @pytest.fixture
    def data(self, streak_points: StreakPoints, counts: PhotonCounts,
             miller: Miller) -> PedestalData:
        return PedestalData(points=streak_points, counts=counts, miller=miller)

    @pytest.fixture
    def modelled(self, model: ScalerModel, data: PedestalData,
                 setup: ResolvedSetup, xp: JaxNamespace) -> IntensityModel:
        return model.init_model(data, setup, xp)

    @pytest.fixture
    def streak_data(self, data: PedestalData, modelled: IntensityModel) -> StreakData:
        return StreakData(data, modelled)

    @pytest.fixture
    def sigma(self) -> float:
        return 0.02

    @pytest.fixture
    def weights(self, xp: JaxNamespace) -> RealArray:
        return xp.asarray([1.0, 2.0, 1.5, 0.5, 3.0, 2.5, 1.0, 4.0])

    @pytest.fixture
    def state(self, streak_data: StreakData, sigma: float, weights: RealArray,
              xp: JaxNamespace) -> StreakState:
        return StreakState.from_data(streak_data, sigma, weights, xp)

    def test_state_initialization(self, streak_data: StreakData, state: StreakState,
                                  weights: RealArray, xp: JaxNamespace):
        profile = xp.exp(streak_data.modelled.log_profile(
            state.log_sigma_at(streak_data.points), xp))
        numerator = xp.asarray([
            xp.sum(weights[streak_data.points.streak_id == index]
                   * profile[streak_data.points.streak_id == index]
                   * streak_data.counts.signal[streak_data.points.streak_id == index])
            for index in range(streak_data.n_streaks)
        ])
        denominator = xp.asarray([
            xp.sum(weights[streak_data.points.streak_id == index]
                   * profile[streak_data.points.streak_id == index]**2)
            for index in range(streak_data.n_streaks)
        ])
        supported = denominator > 0.0
        expected = xp.where(supported,
                            numerator / xp.where(supported, denominator, 1.0), 0.0)

        # Weighted regression retains zero for streaks without supported profile points.
        check_close(state.intensity, expected)

    def test_original_hkl_geometry(self, model: ScalerModel, data: PedestalData,
                                   modelled: IntensityModel, setup: ResolvedSetup,
                                   xp: JaxNamespace):
        miller = model.xtal.hkl_to_q(data.miller, setup.xtal, xp)
        q = miller.q[data.points.streak_id]

        # Local fitting uses each streak's original hkl without applying symmetry.
        check_close(modelled.source.q, q)

    def test_streak_intensity_gradient(self, model: ScalerModel, streak_data: StreakData,
                                       state: StreakState, xp: JaxNamespace):
        loss = StreakLoss(model)
        loss_grad_fn = cast(StreakLossGradFn, jit(value_and_grad(loss, argnums=1)))
        value, gradient = loss_grad_fn(streak_data, state)

        # Every predicted streak retains one independently differentiable profile coefficient.
        assert xp.isfinite(value)
        assert gradient.intensity.shape == (streak_data.n_streaks,)
        assert xp.all(xp.isfinite(gradient.intensity))
        assert xp.all(xp.isfinite(gradient.pedestal))

    def test_result(self, model: ScalerModel, streak_data: StreakData,
                    state: StreakState, xp: JaxNamespace):
        loss = StreakLoss(model)
        result = loss.to_list(streak_data, state, xp)
        profile = loss.profile(streak_data, state, xp)
        point_information = model.poisson_information(loss.predictor(streak_data, state, xp), xp)
        information = streak_data.profiled_information(profile, point_information, xp)
        positive = information > 0.0
        sigma = xp.where(positive, 1.0 / xp.sqrt(xp.where(positive, information, 1.0)), xp.inf)

        # Results retain one indexed row per fitted streak in input order.
        assert xp.all(result.index == streak_data.miller.index)
        assert xp.all(result.hkl == streak_data.miller.hkl)
        # Streak length changes information, not the fitted intensity coefficient itself.
        check_close(result.I_hkl, state.intensity)
        # Intensity uncertainty excludes information confounded with the pedestal.
        check_close(result.sigma_hkl, sigma)

    def test_result_dataframe(self, model: ScalerModel, streak_data: StreakData,
                              state: StreakState, xp: JaxNamespace):
        result = StreakLoss(model).to_list(streak_data, state, xp)
        frames = xp.unique(streak_data.points.index)

        imported = ReflectionList.import_dataframe(result.to_dataframe(frames), frames, xp)

        # Dataframe export and import preserve every fitted reflection field.
        assert xp.all(imported.index == result.index)
        assert xp.all(imported.hkl == result.hkl)
        check_close(imported.I_hkl, result.I_hkl)
        check_close(imported.sigma_hkl, result.sigma_hkl)

    def test_zero_information(self, model: ScalerModel, streak_data: StreakData,
                              state: StreakState, xp: JaxNamespace):
        saturated = state.replace(pedestal=xp.full_like(state.pedestal, -1e6),
                                  intensity=xp.zeros_like(state.intensity))
        std_hkl = StreakLoss(model).std_hkl(streak_data, saturated, xp)

        # A saturated softplus predictor has no information about its fitted coefficients.
        assert xp.all(xp.isinf(std_hkl))
