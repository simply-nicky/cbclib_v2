from typing import Callable, Tuple, cast
import pytest
from jax import jit, tree, value_and_grad
from cbclib_v2 import CrystData
from cbclib_v2.annotations import IntArray, JaxNamespace, JaxNumPy, NumPy, NumPyNamespace, RealArray
from cbclib_v2.indexer import (FixedPupilSetup, Miller, Rectangle, ResolvedSetup, SourcePlane)
from cbclib_v2.scaler import (FullLoss, FullState, IntensityModel, PhotonCounts, ReflectionList,
                              PedestalData, ScalerModel, StreakState, StreakData, StreakLoss,
                              StreakIndices, StreakPoints)
from cbclib_v2.test_util import check_close, TestSetup

StreakLossGradFn = Callable[[StreakData, StreakState], Tuple[RealArray, StreakState]]
FullLossGradFn = Callable[[PedestalData, FullState], Tuple[RealArray, FullState]]

class TestPhotonCounts:
    @pytest.fixture
    def xp(self) -> NumPyNamespace:
        return NumPy

    @pytest.fixture
    def indices(self, xp: NumPyNamespace) -> StreakIndices:
        return StreakIndices(index=xp.asarray([0, 1]), streak_id=xp.asarray([0, 1]),
                             y=xp.asarray([0, 1]), x=xp.asarray([1, 0]))

    @pytest.fixture
    def frames(self, xp: NumPyNamespace) -> RealArray:
        return xp.asarray([[[1.0, 2.0], [3.0, 4.0]],
                           [[5.0, 6.0], [7.0, 8.0]]])

    def test_frame_noise(self, indices: StreakIndices, frames: RealArray,
                         xp: NumPyNamespace):
        background = xp.asarray([[10.0, 20.0], [30.0, 40.0]])
        std = xp.asarray([[1.0, 2.0], [3.0, 4.0]])
        data = CrystData(data=frames, whitefield=background, std=std)

        counts = PhotonCounts.import_data(data, indices, xp)

        # A frame noise model is shared while raw counts remain pattern-specific.
        check_close(counts.I0, xp.asarray([2.0, 7.0]))
        check_close(counts.background, xp.asarray([20.0, 30.0]))
        check_close(counts.std, xp.asarray([2.0, 3.0]))
        assert counts.std.shape == counts.I0.shape

    def test_stack_noise(self, indices: StreakIndices, frames: RealArray,
                         xp: NumPyNamespace):
        background = frames + 10.0
        std = frames + 20.0
        protocol = CrystData.default_protocol()
        protocol.kinds['whitefield'] = 'stack'
        protocol.kinds['std'] = 'stack'
        data = CrystData(data=frames, whitefield=background, std=std, protocol=protocol)

        counts = PhotonCounts.import_data(data, indices, xp)

        # A stack noise model follows the compact pattern index of every point.
        check_close(counts.I0, xp.asarray([2.0, 7.0]))
        check_close(counts.background, xp.asarray([12.0, 17.0]))
        check_close(counts.std, xp.asarray([22.0, 27.0]))
        assert counts.std.shape == counts.I0.shape

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

class TestPoissonLoss:
    @pytest.fixture
    def xp(self) -> NumPyNamespace:
        return NumPy

    def test_signal_and_background(self, xp: NumPyNamespace):
        model = ScalerModel()
        counts = PhotonCounts(I0=xp.asarray([4]), background=xp.asarray([3.0]),
                              std=xp.asarray([2.0]))
        log_expected = model.log_rate(counts.background + xp.asarray([2.0]), xp)
        expected_rate = xp.exp(log_expected)

        loss = model.poisson_loss(counts, log_expected, xp)

        # The Poisson negative log-likelihood combines the full expected rate and counts.
        check_close(loss, expected_rate - counts.I0 * log_expected)

    def test_signal_underflow(self, xp: NumPyNamespace):
        model = ScalerModel()
        counts = PhotonCounts(I0=xp.asarray([1]), background=xp.asarray([0.0]),
                              std=xp.asarray([1.0]))
        log_expected = model.log_rate(xp.asarray([-1000.0]), xp)

        loss = model.poisson_loss(counts, log_expected, xp)

        assert xp.all(xp.isfinite(loss))

    def test_rate_floor(self, xp: NumPyNamespace):
        model = ScalerModel()
        counts = PhotonCounts(I0=xp.asarray([0]), background=xp.asarray([0.0]),
                              std=xp.asarray([1.0]))
        log_expected = model.log_rate(counts.background, xp)

        loss = model.poisson_loss(counts, log_expected, xp)

        # Softplus rounds the positivity boundary instead of creating a clipped corner.
        expected_rate = model.rate_floor + xp.logaddexp(counts.background, 0.0)
        check_close(loss, expected_rate)

    def test_custom_rate_floor(self, xp: NumPyNamespace):
        model = ScalerModel(rate_floor=0.25)
        counts = PhotonCounts(I0=xp.asarray([0]), background=xp.asarray([0.0]),
                              std=xp.asarray([1.0]))
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
                            background=xp.asarray([2.0, 3.0, 2.0, 4.0]),
                            std=xp.asarray([1.0, 2.0, 1.5, 0.5]))

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
                            background=xp.asarray([0.25, 1.0, 2.0, 4.0,
                                                   0.5, 2.0, 5.0, 8.0]),
                            std=xp.asarray([0.5, 1.0, 1.5, 2.0,
                                            2.5, 3.0, 3.5, 4.0]))

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

    @pytest.fixture
    def loss(self, model: ScalerModel) -> StreakLoss:
        return StreakLoss(model)

    @pytest.fixture
    def result_state(self, state: StreakState, xp: JaxNamespace) -> StreakState:
        intensity = xp.asarray([-1.0, 0.25, 1.0, 3.0])
        return state.replace(asinh_hkl=xp.arcsinh(intensity))

    @pytest.fixture
    def result_profile(self, loss: StreakLoss, streak_data: StreakData,
                       result_state: StreakState, xp: JaxNamespace) -> RealArray:
        return loss.profile(streak_data, result_state, xp)

    @pytest.fixture
    def result_std(self, streak_data: StreakData, result_profile: RealArray,
                   xp: JaxNamespace) -> RealArray:
        information = streak_data.sum_by_streak(
            (result_profile / streak_data.counts.std)**2, xp)
        return 1.0 / xp.sqrt(information)

    @pytest.fixture
    def reflection_list(self, loss: StreakLoss, streak_data: StreakData,
                        result_state: StreakState, xp: JaxNamespace) -> ReflectionList:
        return loss.to_list(streak_data, result_state, xp)

    def profile_mean(self, values: RealArray, profile: RealArray,
                     streak_data: StreakData, xp: JaxNamespace) -> RealArray:
        length = streak_data.sum_by_streak(profile, xp)
        supported = length > 0.0
        denominator = xp.where(supported, length, 1.0)
        mean = streak_data.sum_by_streak(profile * values, xp) / denominator
        return xp.where(supported, mean, 0.0)

    def test_state_initialization(self, streak_data: StreakData, state: StreakState,
                                  sigma: float, weights: RealArray, xp: JaxNamespace):
        profile = xp.exp(streak_data.modelled.log_profile(
            state.log_sigma_at(streak_data.points), xp))
        supported = profile > 0.0
        coefficient = xp.where(
            supported, streak_data.counts.signal / xp.where(supported, profile, 1.0), 0.0)
        information = weights * profile**2
        expected = []
        for index in range(streak_data.n_streaks):
            mask = streak_data.points.streak_id == index
            order = xp.argsort(coefficient[mask])
            cumulative = xp.cumsum(information[mask][order])
            median_index = xp.argmax(cumulative >= 0.5 * cumulative[-1])
            expected.append(xp.where(cumulative[-1] > 0.0,
                                     coefficient[mask][order][median_index], 0.0))

        # Initial intensities are information-weighted medians of the point estimates.
        check_close(xp.sinh(state.asinh_hkl), xp.asarray(expected))

        total_information = xp.asarray([
            xp.sum(information[streak_data.points.streak_id == index])
            for index in range(streak_data.n_streaks)
        ])
        minority = ((information > 0.0)
                    & (information < 0.5 * total_information[
                        streak_data.points.streak_id]))
        assert xp.any(minority)
        outlier_index = xp.argmax(minority)
        I0 = xp.where(xp.arange(streak_data.counts.I0.size) == outlier_index,
                      1_000_000, streak_data.counts.I0)
        counts = streak_data.counts.replace(I0=I0)
        corrupted = streak_data.replace(scaling=streak_data.scaling.replace(counts=counts))
        robust = StreakState.from_data(corrupted, sigma, weights, xp)

        # An arbitrarily bright point below half the information weight does not move the median.
        check_close(xp.sinh(robust.asinh_hkl), xp.sinh(state.asinh_hkl))

    def test_modelled_q(self, model: ScalerModel, data: PedestalData, modelled: IntensityModel,
                        setup: ResolvedSetup, xp: JaxNamespace):
        miller = model.xtal.hkl_to_q(data.miller, setup.xtal, xp)
        q = miller.q[data.points.streak_id]

        # Local fitting uses each streak's original hkl without applying symmetry.
        check_close(modelled.source.q, q)

    def test_result(self, streak_data: StreakData,  result_profile: RealArray,
                    reflection_list: ReflectionList, xp: JaxNamespace):
        expected = self.profile_mean(
            streak_data.counts.signal, result_profile, streak_data, xp)

        # Results retain input order and average the raw signal over profile support.
        assert xp.all(reflection_list.index == streak_data.miller.index)
        assert xp.all(reflection_list.hkl == streak_data.miller.hkl)
        check_close(reflection_list.I_hkl, expected)

    def test_std_by_streak(self, loss: StreakLoss, streak_data: StreakData,
                           result_state: StreakState, result_std: RealArray,
                           xp: JaxNamespace):
        profile = loss.profile(streak_data, result_state, xp)

        # Detector-noise errors follow weighted profile regression.
        check_close(streak_data.std_by_streak(profile, xp), result_std)

    def test_std_hkl(self, loss: StreakLoss, streak_data: StreakData,
                     state: StreakState, xp: JaxNamespace):
        magnitude = xp.full_like(state.asinh_hkl, 10.0)
        negative = state.replace(asinh_hkl=xp.arcsinh(-magnitude))
        positive = state.replace(asinh_hkl=xp.arcsinh(magnitude))

        # Detector-noise propagation does not collapse for negative fitted intensities.
        check_close(loss.std_hkl(streak_data, negative, xp),
                    loss.std_hkl(streak_data, positive, xp))

    def test_zero_information(self, model: ScalerModel, streak_data: StreakData,
                              state: StreakState, xp: JaxNamespace):
        counts = streak_data.counts.replace(std=xp.zeros_like(streak_data.counts.std))
        unsupported = streak_data.replace(
            scaling=streak_data.scaling.replace(counts=counts))
        std_hkl = StreakLoss(model).std_hkl(unsupported, state, xp)

        # Points without a detector-noise estimate provide no uncertainty information.
        assert xp.all(xp.isinf(std_hkl))
