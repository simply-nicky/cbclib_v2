from typing import Callable, Tuple, cast
import pytest
import pandas as pd
from jax import jit, tree, value_and_grad
from cbclib_v2 import CrystData
from cbclib_v2.annotations import (IntArray, JaxNamespace, JaxNumPy, NumPy,
                                   NumPyNamespace, RealArray)
from cbclib_v2.indexer import (FixedPupilGeometry, FixedPupilSetup, FixedRectangle, Focus, Miller,
                              ResolvedGeometry,
                              ResolvedLens, ResolvedSetup, SourcePlane, XtalState)
from cbclib_v2.indexer.cbc_pupil import RectangularPupil
from cbclib_v2.indexer.cbc_setup import Lens
from cbclib_v2.scaler import (FullLoss, FullState, IntensityModel, OverlapData, PhotonCounts,
                              RefineStreakState, ReflectionList, ScalerData, ScalerModel,
                              StreakState, StreakData, StreakLoss, StreakIndices, StreakPoints)
from cbclib_v2.test_util import check_close, TestSetup

StreakLossGradFn = Callable[[StreakData, StreakState], Tuple[RealArray, StreakState]]
FullLossGradFn = Callable[[ScalerData | OverlapData, FullState], Tuple[RealArray, FullState]]

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
                         xp: NumPyNamespace) -> None:
        background = xp.asarray([[10.0, 20.0], [30.0, 40.0]])
        std = xp.asarray([[1.0, 2.0], [3.0, 4.0]])
        data = CrystData(data=frames, whitefield=background, std=std)

        counts = PhotonCounts.import_data(data, indices, xp)

        # A frame noise model is shared while raw counts remain pattern-specific.
        check_close(counts.I0, frames[indices.index, indices.y, indices.x])
        check_close(counts.background, background[indices.y, indices.x])
        check_close(counts.std, std[indices.y, indices.x])

    def test_stack_noise(self, indices: StreakIndices, frames: RealArray,
                         xp: NumPyNamespace) -> None:
        background = frames + 10.0
        std = frames + 20.0
        protocol = CrystData.default_protocol()
        protocol.kinds['whitefield'] = 'stack'
        protocol.kinds['std'] = 'stack'
        data = CrystData(data=frames, whitefield=background, std=std, protocol=protocol)

        counts = PhotonCounts.import_data(data, indices, xp)

        # A stack noise model follows the compact pattern index of every point.
        check_close(counts.I0, frames[indices.index, indices.y, indices.x])
        check_close(counts.background, background[indices.index, indices.y, indices.x])
        check_close(counts.std, std[indices.index, indices.y, indices.x])

class TestIntensityModel:
    @pytest.fixture
    def xp(self) -> NumPyNamespace:
        return NumPy

    def model(self, kin: RealArray, q: RealArray, pupil: RectangularPupil) -> IntensityModel:
        return IntensityModel(kout=kin + q, source=SourcePlane.from_q(q=q), pupil=pupil,
                              smp_pos=kin * 0.0)

    def test_source_distance(self, xp: NumPyNamespace) -> None:
        kin = xp.asarray([1.0, 0.0, 0.0])
        q = xp.asarray([2.0, 0.0, 0.0])
        pupil = RectangularPupil(roi=xp.asarray([[-1.0, 1.0, -1.0, 1.0]]))
        model = self.model(kin, q, pupil)

        # Unit width gives Gaussian attenuation by the squared support distance.
        expected = xp.exp(-0.5 * model.distance_sq)
        check_close(model.profile(xp.asarray(0.0), xp), expected)

    def test_outside_pupil(self, xp: NumPyNamespace) -> None:
        kin = xp.asarray([0.2, 0.0, xp.sqrt(0.96)])
        q = xp.zeros(3)
        pupil = RectangularPupil(roi=xp.asarray([[-0.1, 0.1, -0.1, 0.1]]))
        model = self.model(kin, q, pupil)

        profile = model.profile(xp.asarray(0.0), xp)

        assert xp.all(profile > 0.0)
        assert xp.all(profile < 1.0)

class TestPoissonLoss:
    @pytest.fixture
    def xp(self) -> NumPyNamespace:
        return NumPy

    def test_signal_and_background(self, xp: NumPyNamespace) -> None:
        model = ScalerModel()
        counts = PhotonCounts(I0=xp.asarray([4]), background=xp.asarray([3.0]),
                              std=xp.asarray([2.0]))
        log_expected = model.log_rate(counts.background + xp.asarray([2.0]), xp)
        expected_rate = xp.exp(log_expected)

        loss = model.poisson_loss(counts, log_expected, xp)

        # The Poisson negative log-likelihood combines the full expected rate and counts.
        check_close(loss, expected_rate - counts.I0 * log_expected)

    def test_negative_rate_input(self, xp: NumPyNamespace) -> None:
        model = ScalerModel()
        counts = PhotonCounts(I0=xp.asarray([1]), background=xp.asarray([0.0]),
                              std=xp.asarray([1.0]))
        log_expected = model.log_rate(xp.asarray([-1000.0]), xp)

        loss = model.poisson_loss(counts, log_expected, xp)

        assert xp.all(xp.isfinite(loss))

    @pytest.mark.parametrize("rate_floor", [1e-6, 0.25])
    def test_rate_floor(self, rate_floor: float, xp: NumPyNamespace) -> None:
        model = ScalerModel(rate_floor=rate_floor)
        counts = PhotonCounts(I0=xp.asarray([0]), background=xp.asarray([0.0]),
                              std=xp.asarray([1.0]))
        log_expected = model.log_rate(counts.background, xp)

        loss = model.poisson_loss(counts, log_expected, xp)

        # Softplus rounds the positivity boundary instead of creating a clipped corner.
        expected_rate = model.rate_floor + xp.logaddexp(counts.background, 0.0)
        check_close(loss, expected_rate)

    def test_invalid_rate_floor(self) -> None:
        with pytest.raises(ValueError, match="rate_floor must be positive"):
            ScalerModel(rate_floor=0.0)

class ScalingCase:
    @pytest.fixture
    def xp(self) -> JaxNamespace:
        return JaxNumPy

    @pytest.fixture
    def model(self) -> ScalerModel:
        return ScalerModel()

    @pytest.fixture
    def initial_setup(self, xp: JaxNamespace) -> FixedPupilSetup:
        geometry = FixedPupilGeometry(
            focus=Focus(xp.asarray(TestSetup.foc_pos)),
            pupil=FixedRectangle(TestSetup.pupil_roi),
            defocus=xp.asarray([TestSetup.defocus])).broadcast(2)
        return FixedPupilSetup(TestSetup.xtal(xp), geometry)

    @pytest.fixture
    def setup(self, initial_setup: FixedPupilSetup, xp: JaxNamespace) -> ResolvedSetup:
        return initial_setup.resolve(xp)

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
             miller: Miller, overlap: bool, xp: JaxNamespace) -> ScalerData | OverlapData:
        if overlap:
            pixel_id = xp.asarray([0, 1, 0, 2, 3, 4, 3, 5])
            positions = xp.asarray([0, 1, 3, 4, 5, 7])
            points = streak_points.replace(points=streak_points.points[positions][pixel_id])
            return OverlapData(points=points, counts=counts[positions], miller=miller,
                                pixel_id=pixel_id)
        return ScalerData(points=streak_points, counts=counts, miller=miller)

    @pytest.fixture
    def modelled(self, model: ScalerModel, data: ScalerData | OverlapData,
                 setup: ResolvedSetup, xp: JaxNamespace) -> IntensityModel:
        return model.init_model(data, setup, xp)

    @pytest.fixture
    def streak_data(self, data: ScalerData | OverlapData, modelled: IntensityModel) -> StreakData:
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
            (result_profile / streak_data.std)**2, xp)
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

    @pytest.fixture(params=[False, True], ids=["ordinary", "overlap"])
    def overlap(self, request: pytest.FixtureRequest) -> bool:
        return request.param

class TestLossGradients(ScalingCase):
    def check_gradient(self, value: RealArray, gradient: StreakState | FullState,
                       xp: JaxNamespace) -> None:
        leaves = tree.leaves(gradient)

        assert xp.isfinite(value)
        assert all(xp.all(xp.isfinite(leaf)) for leaf in leaves)
        assert any(xp.any(leaf != 0.0) for leaf in leaves)

    def test_scaler(self, model: ScalerModel, modelled: IntensityModel,
                    data: ScalerData | OverlapData,
                    state: StreakState, xp: JaxNamespace) -> None:
        loss = StreakLoss(model)
        loss_grad_fn = cast(StreakLossGradFn, jit(value_and_grad(loss, argnums=1)))
        streak_data = StreakData(data, modelled)
        value, gradient = loss_grad_fn(streak_data, state)
        check_close(value, loss(streak_data, state))

        # Compilation preserves the likelihood and yields finite, nonzero derivatives.
        self.check_gradient(value, gradient, xp)

    def test_setup(self, model: ScalerModel, data: ScalerData | OverlapData, state: StreakState,
                   initial_setup: FixedPupilSetup, xp: JaxNamespace) -> None:
        loss = FullLoss(model)
        loss_grad_fn = cast(FullLossGradFn, jit(value_and_grad(loss, argnums=1)))
        full_state = FullState(state, initial_setup)
        value, gradient = loss_grad_fn(data, full_state)
        check_close(value, loss(data, full_state))

        # Compilation preserves the likelihood and yields finite, nonzero derivatives.
        self.check_gradient(value, gradient, xp)

class TestStreakInitialisation(ScalingCase):
    @pytest.fixture
    def overlap(self) -> bool:
        return False

    def test_weighted_median(self, streak_data: StreakData, state: StreakState,
                             weights: RealArray, xp: JaxNamespace) -> None:
        profile = streak_data.modelled.profile(state.log_sigma_at(streak_data.points), xp)
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

    def test_minority_outlier(self, streak_data: StreakData, state: StreakState,
                              sigma: float, weights: RealArray, xp: JaxNamespace) -> None:
        profile = streak_data.modelled.profile(state.log_sigma_at(streak_data.points), xp)
        information = weights * profile**2
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

class TestStreakStateSerialisation(ScalingCase):
    @pytest.fixture
    def frames(self, xp: JaxNamespace) -> IntArray:
        return xp.asarray([10, 20])

    @pytest.fixture
    def refined(self, result_state: StreakState, xp: JaxNamespace) -> RefineStreakState:
        refined = RefineStreakState.from_streaks(result_state, pixel_size=0.1)
        displacement = xp.arange(2 * refined.n_streaks, dtype=float).reshape((-1, 2))
        return refined.replace(displacement=displacement)

    def test_streak_round_trip(self, streak_data: StreakData,
                               result_state: StreakState, frames: IntArray,
                               xp: JaxNamespace) -> None:
        index = streak_data.miller.index
        saved_frames, restored = StreakState.import_dataframe(
            result_state.to_dataframe(index, frames), index=True, xp=xp)
        assert xp.all(saved_frames == frames)
        check_close(restored.asinh_hkl, result_state.asinh_hkl)
        check_close(restored.log_sigma, result_state.log_sigma)

    def test_refined_round_trip(self, streak_data: StreakData,
                                refined: RefineStreakState, frames: IntArray,
                                xp: JaxNamespace) -> None:
        index = streak_data.miller.index
        saved_frames, restored_refined = RefineStreakState.import_dataframe(
            refined.to_dataframe(index, frames), index=True, xp=xp,
            pixel_size=refined.pixel_size)
        assert xp.all(saved_frames == frames)
        check_close(restored_refined.asinh_hkl, refined.asinh_hkl)
        check_close(restored_refined.log_sigma, refined.log_sigma)
        check_close(restored_refined.displacement, refined.displacement)
        assert restored_refined.pixel_size == refined.pixel_size

class TestReflectionResults(ScalingCase):
    def test_result(self, streak_data: StreakData, result_state: StreakState,
                    result_profile: RealArray, reflection_list: ReflectionList,
                    xp: JaxNamespace) -> None:
        model = ScalerModel()
        signal = result_state.intensity_at(streak_data.points)
        baseline = streak_data.background
        contrast = model.rate(baseline + signal, xp) - model.rate(baseline, xp)
        expected = self.profile_mean(contrast, result_profile, streak_data, xp)

        # Results retain input order and report the fitted rate contrast.
        assert xp.all(reflection_list.index == streak_data.miller.index)
        assert xp.all(reflection_list.hkl == streak_data.miller.hkl)
        check_close(reflection_list.I_hkl, expected)

    def test_std_by_streak(self, loss: StreakLoss, streak_data: StreakData,
                           result_state: StreakState, result_std: RealArray,
                           xp: JaxNamespace) -> None:
        profile = loss.profile(streak_data, result_state, xp)

        # Detector-noise errors follow weighted profile regression.
        check_close(streak_data.std_by_streak(profile, xp), result_std)

    def test_std_hkl(self, loss: StreakLoss, streak_data: StreakData,
                     state: StreakState, xp: JaxNamespace) -> None:
        magnitude = xp.full_like(state.asinh_hkl, 10.0)
        negative = state.replace(asinh_hkl=xp.arcsinh(-magnitude))
        positive = state.replace(asinh_hkl=xp.arcsinh(magnitude))

        # Detector-noise propagation does not collapse for negative fitted intensities.
        check_close(loss.std_hkl(streak_data, negative, xp),
                    loss.std_hkl(streak_data, positive, xp))

    def test_zero_information(self, model: ScalerModel, streak_data: StreakData,
                              state: StreakState, xp: JaxNamespace) -> None:
        counts = streak_data.counts.replace(std=xp.zeros_like(streak_data.counts.std))
        unsupported = streak_data.replace(
            scaling=streak_data.scaling.replace(counts=counts))
        std_hkl = StreakLoss(model).std_hkl(unsupported, state, xp)

        # Points without a detector-noise estimate provide no uncertainty information.
        assert xp.all(xp.isinf(std_hkl))

class TestScalingGeometry(ScalingCase):
    def test_modelled_q(self, model: ScalerModel, data: ScalerData | OverlapData,
                        modelled: IntensityModel,
                        setup: ResolvedSetup, xp: JaxNamespace) -> None:
        miller = model.xtal.hkl_to_q(data.miller, setup.xtal, xp)
        q = miller.q[data.points.streak_id]

        # Local fitting uses each streak's original hkl without applying symmetry.
        check_close(modelled.source.q, q)

class TestOverlapInitialisation(ScalingCase):
    @pytest.fixture
    def overlap(self) -> bool:
        return True

    def test_observation_gathering(self, streak_data: StreakData, state: StreakState,
                                  sigma: float, weights: RealArray, xp: JaxNamespace) -> None:
        # Initialisation gathers unique observations into contribution order.
        expanded_counts = PhotonCounts(
            I0=xp.asarray(streak_data.signal + streak_data.background, dtype=int),
            background=streak_data.background, std=streak_data.std)
        expanded = streak_data.replace(scaling=ScalerData(
            points=streak_data.points, counts=expanded_counts, miller=streak_data.miller))
        expected = StreakState.from_data(expanded, sigma, weights, xp)
        check_close(state.asinh_hkl, expected.asinh_hkl)

class TestOverlapReflectionResults(ScalingCase):
    @pytest.fixture
    def overlap(self) -> bool:
        return True

    def test_crossing_independence(self, loss: StreakLoss, streak_data: StreakData,
                    result_state: StreakState,
                    reflection_list: ReflectionList, xp: JaxNamespace) -> None:
        changed = result_state.replace(
            asinh_hkl=result_state.asinh_hkl.at[1].set(xp.arcsinh(100.0)))
        # Changing a crossing streak does not change this streak's reported intensity.
        check_close(loss.I_hkl(streak_data, changed, xp)[0], reflection_list.I_hkl[0])

class TestOverlapLoss(ScalingCase):
    @pytest.fixture
    def overlap(self) -> bool:
        return True

    def test_overlap_loss(self, loss: StreakLoss, streak_data: StreakData,
                          result_state: StreakState, xp: JaxNamespace) -> None:
        profile = loss.profile(streak_data, result_state, xp)
        contributions = result_state.intensity_at(streak_data.points) * profile
        signal = xp.stack((contributions[0] + contributions[2], contributions[1],
                           contributions[3], contributions[4] + contributions[6],
                           contributions[5], contributions[7]))
        expected = loss.model.log_rate(streak_data.counts.background + signal, xp)
        # Shared pixels contain the sum of both signals and enter the likelihood once.
        check_close(loss.log_expected(streak_data, result_state, xp), expected)
        check_close(loss(streak_data, result_state),
                    xp.mean(loss.model.poisson_loss(streak_data.counts, expected, xp)))

    def test_full_overlap(self, model: ScalerModel, streak_data: StreakData,
                          result_state: StreakState, initial_setup: FixedPupilSetup,
                          xp: JaxNamespace) -> None:
        loss = FullLoss(model)
        state = FullState(setup=initial_setup, streaks=result_state)
        fixed = StreakLoss(model)
        # Fixed and jointly refined geometry produce the same summed pixel predictions.
        check_close(loss.log_expected(streak_data.scaling, state, xp),
                    fixed.log_expected(streak_data, result_state, xp))
        check_close(loss.I_hkl(streak_data.scaling, state, xp),
                    fixed.I_hkl(streak_data, result_state, xp))

class PupilCase:
    @pytest.fixture
    def xp(self) -> NumPyNamespace:
        return NumPy

    @pytest.fixture(params=[1, 2])
    def n_frames(self, request: pytest.FixtureRequest) -> int:
        return request.param

    @pytest.fixture
    def indices(self, n_frames: int, xp: NumPyNamespace) -> StreakIndices:
        # Each frame has two streaks with unequal pixel counts.
        streak_id = xp.repeat(xp.arange(2 * n_frames), xp.arange(2 * n_frames) + 1)
        return StreakIndices(index=streak_id // 2, streak_id=streak_id,
                             x=xp.arange(streak_id.size), y=xp.zeros_like(streak_id))

    @pytest.fixture
    def points(self, indices: StreakIndices) -> StreakPoints:
        return StreakPoints(index=indices.index, streak_id=indices.streak_id,
                            points=0.02 * indices.xy)

    @pytest.fixture(params=[False, True], ids=['focused', 'defocused'])
    def geometry(self, request: pytest.FixtureRequest, n_frames: int,
                  xp: NumPyNamespace) -> ResolvedLens:
        focus = xp.broadcast_to(xp.asarray([[0.0, 0.0, -1.0]]), (n_frames, 3))
        roi = xp.asarray([[-0.2, 0.1, -0.3, -0.1]])
        roi = xp.asarray(roi + 0.03 * xp.arange(n_frames)[:, None])
        if request.param:
            return ResolvedGeometry(focus, roi, xp.full((n_frames,), 0.1))
        return ResolvedLens(focus, roi)

    @pytest.fixture
    def lens(self, geometry: ResolvedLens) -> Lens:
        return Lens.from_resolved(geometry)

    @pytest.fixture
    def fixed_geometry(self, xp: NumPyNamespace) -> FixedPupilGeometry:
        return FixedPupilGeometry(
            focus=Focus(xp.array([0.0, 0.0, -1.0])),
            pupil=FixedRectangle((-0.2, 0.2, -0.3, -0.1)),
            defocus=xp.array([0.1]))

    @pytest.fixture
    def frame_geometry(self, fixed_geometry: FixedPupilGeometry,
                        n_frames: int) -> FixedPupilGeometry:
        return fixed_geometry.broadcast(n_frames)

    @pytest.fixture
    def miller(self, n_frames: int, xp: NumPyNamespace) -> Miller:
        return Miller(index=xp.repeat(xp.arange(n_frames), 2),
                      hkl=xp.tile(xp.asarray([[0, 1, 0], [1, 0, 0]]), (n_frames, 1)))

    @pytest.fixture
    def setup(self, geometry: ResolvedLens, n_frames: int, xp: NumPyNamespace) -> ResolvedSetup:
        xtal = XtalState(xp.broadcast_to(0.1 * xp.eye(3), (n_frames, 3, 3)))
        return ResolvedSetup(xtal, geometry)

    @pytest.fixture
    def model(self) -> ScalerModel:
        return ScalerModel()

    @pytest.fixture
    def modelled(self, model: ScalerModel, points: StreakPoints, miller: Miller,
                  setup: ResolvedSetup, xp: NumPyNamespace) -> IntensityModel:
        return model.model_at(points, miller, setup, xp)

class TestPupilGeometry(PupilCase):
    def test_resolve_preserves_pupil(self, lens: Lens, geometry: ResolvedLens,
                                     n_frames: int, xp: NumPyNamespace) -> None:
        resolved = lens.resolve(xp)

        # Resolution preserves the pupil ROI of each frame, including one frame.
        assert resolved.foc_pos.shape == (n_frames, 3)
        assert resolved.pupil_roi.shape == (n_frames, 4)
        check_close(resolved.foc_pos, geometry.foc_pos)
        check_close(resolved.pupil_roi, geometry.pupil_roi)

    def test_resolve_preserves_frames(self, fixed_geometry: FixedPupilGeometry,
                                      frame_geometry: FixedPupilGeometry, n_frames: int,
                                      xp: NumPyNamespace) -> None:
        resolved = frame_geometry.resolve(xp)

        # Each frame retains the fixed pupil, with its own focus and defocus entry.
        assert resolved.foc_pos.shape == (n_frames, 3)
        assert resolved.pupil_roi.shape == (n_frames, 4)
        assert resolved.defocus.shape == (n_frames,)
        expected_roi = xp.broadcast_to(xp.asarray(fixed_geometry.pupil.roi),
                                       (n_frames, 4))
        check_close(resolved.pupil_roi, expected_roi)
        check_close(resolved.foc_pos, xp.broadcast_to(fixed_geometry.focus.pos, (n_frames, 3)))
        check_close(resolved.defocus, xp.broadcast_to(fixed_geometry.defocus, (n_frames,)))

class TestPupilIntensityModel(PupilCase):
    def test_pupil_selection(self, model: ScalerModel, modelled: IntensityModel,
                             points: StreakPoints, setup: ResolvedSetup,
                             xp: NumPyNamespace) -> None:
        supports = model.lens.pupil(setup.geometry, xp)
        expected = supports.roi[points.index]
        # Every streak contribution uses the pupil support of its frame.
        assert isinstance(modelled.pupil, RectangularPupil)
        check_close(modelled.pupil.roi, expected)

    def test_pointwise_profile(self, modelled: IntensityModel, points: StreakPoints,
                               xp: NumPyNamespace) -> None:
        sigma = 0.2
        profile = modelled.profile(xp.asarray(xp.log(sigma)), xp)
        assert profile.shape == points.shape
        assert isinstance(modelled.pupil, RectangularPupil)
        expected = []
        vectors = xp.broadcast_to(modelled.source.q, modelled.kin.shape)
        rois = xp.broadcast_to(modelled.pupil.roi, points.shape + (4,))
        for i in range(points.size):
            kin = modelled.kin[i]
            q = vectors[i]
            source_distance = (xp.sum(kin * q) + 0.5 * xp.sum(q**2))**2 / xp.sum(q**2)
            roi = rois[i]
            xy = xp.clip(kin[:2], roi[xp.asarray([2, 0])], roi[xp.asarray([3, 1])])
            projected = xp.concat((xy, xp.sqrt(1.0 - xp.sum(xy**2))[None]))
            pupil_distance = xp.sum((kin - projected)**2)
            expected.append(xp.exp(-0.5 * (source_distance + pupil_distance) / sigma**2))
        # Gaussian attenuation is local to each ray, not a cross-product of point supports.
        check_close(profile, xp.stack(expected))

class TestPupilReflectionLookup(PupilCase):
    def test_reflection_lookup(self, points: StreakPoints, n_frames: int,
                                xp: NumPyNamespace) -> None:
        intensity = xp.arange(2 * n_frames, dtype=float) + 1.0
        state = StreakState(asinh_hkl=xp.arcsinh(intensity),
                             log_sigma=xp.zeros((n_frames,)))
        # Each pixel uses the coefficient of its streak, even within a shared frame.
        check_close(state.intensity_at(points), intensity[points.streak_id])

class TestStreakIndices(PupilCase):
    def test_import_overlap(self, xp: NumPyNamespace) -> None:
        df = pd.DataFrame({'index': [0, 0, 1], 'streak_id': [0, 1, 2],
                           'y': [2, 2, 2], 'x': [3, 3, 3]})
        indices = StreakIndices.import_dataframe(df, xp)
        observations, pixel_id = indices.unique()
        # Shared pixels survive import, and identical pixels in different frames stay distinct.
        assert indices.size == 3
        assert observations.size == 2
        assert xp.all(pixel_id == xp.asarray([0, 0, 1]))
        assert xp.all(observations.index[pixel_id] == indices.index)
        assert xp.all(observations.xy[pixel_id] == indices.xy)

    def test_import_identity(self, indices: StreakIndices, xp: NumPyNamespace) -> None:
        df = pd.DataFrame({name: getattr(indices, name)
                           for name in ('index', 'streak_id', 'x', 'y')})
        imported = StreakIndices.import_dataframe(df, xp)
        selected = imported[xp.arange(0, imported.size, 2)]
        # Filtering preserves the original frame and streak identities.
        assert xp.all(selected.index == indices.index[::2])
        assert xp.all(selected.streak_id == indices.streak_id[::2])
        assert xp.all(selected.xy == indices.xy[::2])
