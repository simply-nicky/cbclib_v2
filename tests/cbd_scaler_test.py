from typing import Callable, Tuple, cast
import pytest
from jax import jit, tree, value_and_grad
from cbclib_v2.annotations import JaxNamespace, JaxNumPy, NumPy, NumPyNamespace, RealArray
from cbclib_v2.indexer import (FixedPupilSetup, Miller, Rectangle, ResolvedSetup, SourcePlane)
from cbclib_v2.scaler import (IntensityModel, PhotonCounts, PointGroup, ReflectionsMap, ScalerData,
                              ScalerLoss, ScalerModel, ScalerResult, ScalerState, SetupLoss,
                              StreakPoints)
from cbclib_v2.test_util import check_close, TestSetup

ScalerLossGradFn = Callable[[IntensityModel, ScalerData, ScalerState],
                            Tuple[RealArray, ScalerState]]
SetupLossGradFn = Callable[[ScalerData, ScalerState, ResolvedSetup],
                           Tuple[RealArray, ResolvedSetup]]

class TestIntensityModel:
    @pytest.fixture
    def xp(self) -> NumPyNamespace:
        return NumPy

    def model(self, kin: RealArray, q: RealArray, pupil: Rectangle) -> IntensityModel:
        return IntensityModel(kout=kin + q, source=SourcePlane.from_q(q=q), pupil=pupil)

    def test_source_distance(self, xp: NumPyNamespace) -> None:
        kin = xp.asarray([1.0, 0.0, 0.0])
        q = xp.asarray([2.0, 0.0, 0.0])
        pupil = Rectangle(roi=xp.asarray([-1.0, 1.0, -1.0, 1.0]))
        model = self.model(kin, q, pupil)

        check_close(model.log_radial(xp.asarray(0.0), xp), xp.asarray(-2.0))

    def test_outside_pupil(self, xp: NumPyNamespace) -> None:
        kin = xp.asarray([0.2, 0.0, xp.sqrt(0.96)])
        q = xp.zeros(3)
        pupil = Rectangle(roi=xp.asarray([-0.1, 0.1, -0.1, 0.1]))
        model = self.model(kin, q, pupil)

        profile = xp.exp(model.log_profile(xp.asarray(0.0), xp))

        assert xp.all(profile > 0.0)
        assert xp.all(profile < 1.0)

    def test_log_profile(self, xp: NumPyNamespace) -> None:
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

    def test_signal_and_background(self, xp: NumPyNamespace) -> None:
        model = ScalerModel()
        counts = PhotonCounts(I0=xp.asarray([4]), background=xp.asarray([3.0]))
        log_signal = xp.log(xp.asarray([2.0]))
        log_expected = model.log_expected(counts, log_signal, xp)
        y_hat = xp.asarray([5.000001])

        loss = model.poisson_loss(counts, log_expected, xp)

        check_close(loss, y_hat - counts.I0 * xp.log(y_hat))

    def test_signal_underflow(self, xp: NumPyNamespace) -> None:
        model = ScalerModel()
        counts = PhotonCounts(I0=xp.asarray([1]), background=xp.asarray([0.0]))
        log_expected = model.log_expected(counts, xp.asarray([-1000.0]), xp)

        loss = model.poisson_loss(counts, log_expected, xp)

        assert xp.all(xp.isfinite(loss))

    def test_rate_floor(self, xp: NumPyNamespace) -> None:
        model = ScalerModel()
        counts = PhotonCounts(I0=xp.asarray([0]), background=xp.asarray([0.0]))
        log_expected = model.log_expected(counts, xp.asarray([-xp.inf]), xp)

        loss = model.poisson_loss(counts, log_expected, xp)

        check_close(loss, xp.asarray([1e-6]))

    def test_custom_rate_floor(self, xp: NumPyNamespace) -> None:
        model = ScalerModel(rate_floor=0.25)
        counts = PhotonCounts(I0=xp.asarray([0]), background=xp.asarray([0.0]))
        log_expected = model.log_expected(counts, xp.asarray([-xp.inf]), xp)

        loss = model.poisson_loss(counts, log_expected, xp)

        check_close(loss, xp.asarray([0.25]))

    def test_invalid_rate_floor(self) -> None:
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
    def setup(self, xp: JaxNamespace) -> ResolvedSetup:
        initial = FixedPupilSetup(TestSetup.xtal(xp), TestSetup.fixed_pupil_geometry(xp))
        return initial.resolve(xp)

    @pytest.fixture
    def data(self, xp: JaxNamespace) -> ScalerData:
        points = StreakPoints(index=xp.asarray([0, 0, 1, 1]),
                              streak_id=xp.asarray([0, 0, 1, 1]),
                              points=xp.asarray([[0.12, 0.14], [0.13, 0.15],
                                                 [0.18, 0.16], [0.19, 0.17]]))
        counts = PhotonCounts(I0=xp.asarray([11, 13, 17, 19]),
                              background=xp.asarray([2.0, 3.0, 2.0, 4.0]))
        miller = Miller(index=xp.asarray([0, 1]),
                        hkl=xp.asarray([[1, 0, 0], [0, 1, 0]]))
        reflections = ReflectionsMap.from_miller(miller, PointGroup("1"), xp)
        return ScalerData(points=points, counts=counts, miller=miller,
                          reflections=reflections)

    @pytest.fixture
    def modelled(self, model: ScalerModel, data: ScalerData,
                 setup: ResolvedSetup, xp: JaxNamespace) -> IntensityModel:
        return model.init_model(data, setup, xp)

    @pytest.fixture
    def state(self, data: ScalerData, xp: JaxNamespace) -> ScalerState:
        return ScalerState.default(n_reflections=len(data.reflections), n_frames=2,
                                   sigma=0.02, xp=xp)

    def check_gradient(self, value: RealArray, gradient: ScalerState | ResolvedSetup,
                       xp: JaxNamespace) -> None:
        leaves = tree.leaves(gradient)

        assert xp.isfinite(value)
        assert all(xp.all(xp.isfinite(leaf)) for leaf in leaves)
        assert any(xp.any(leaf != 0.0) for leaf in leaves)

    def test_scaler(self, model: ScalerModel, modelled: IntensityModel, data: ScalerData,
                    state: ScalerState, xp: JaxNamespace) -> None:
        loss = ScalerLoss(model)
        loss_grad_fn = cast(ScalerLossGradFn, jit(value_and_grad(loss, argnums=2)))
        value, gradient = loss_grad_fn(modelled, data, state)

        self.check_gradient(value, gradient, xp)

    def test_setup(self, model: ScalerModel, data: ScalerData, state: ScalerState,
                   setup: ResolvedSetup, xp: JaxNamespace) -> None:
        loss = SetupLoss(model)
        loss_grad_fn = cast(SetupLossGradFn, jit(value_and_grad(loss, argnums=2)))
        value, gradient = loss_grad_fn(data, state, setup)

        self.check_gradient(value, gradient, xp)

class TestSymmetryScaling:
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
    def data(self, xp: JaxNamespace) -> ScalerData:
        points = StreakPoints(index=xp.asarray([0, 0, 1, 1]),
                              streak_id=xp.asarray([0, 1, 2, 3]),
                              points=xp.asarray([[0.12, 0.14], [0.13, 0.15],
                                                 [0.18, 0.16], [0.19, 0.17]]))
        counts = PhotonCounts(I0=xp.asarray([10, 20, 30, 50]),
                              background=xp.zeros(4))
        miller = Miller(index=xp.asarray([0, 0, 1, 1]),
                        hkl=xp.asarray([[1, 0, 0], [0, 1, 0],
                                        [1, 0, 0], [0, 1, 0]]))
        reflections = ReflectionsMap.from_miller(miller, PointGroup("4mm"), xp)
        return ScalerData(points=points, counts=counts, miller=miller,
                          reflections=reflections)

    @pytest.fixture
    def modelled(self, model: ScalerModel, data: ScalerData,
                 setup: ResolvedSetup, xp: JaxNamespace) -> IntensityModel:
        return model.init_model(data, setup, xp)

    def test_pattern_local_reflections(self, data: ScalerData,
                                       xp: JaxNamespace) -> None:
        assert len(data.reflections) == 2
        assert xp.all(data.reflections.reflection_id == xp.asarray([0, 0, 1, 1]))

    def test_state_initialization(self, data: ScalerData, setup: ResolvedSetup,
                                  xp: JaxNamespace) -> None:
        state = ScalerState.from_data(data, setup, sigma=0.02)

        check_close(xp.exp(state.log_hkl), xp.asarray([15.0, 40.0]))

    def test_original_hkl_geometry(self, model: ScalerModel, data: ScalerData,
                                   modelled: IntensityModel, setup: ResolvedSetup,
                                   xp: JaxNamespace) -> None:
        miller = model.xtal.hkl_to_q(data.miller, setup.xtal, xp)
        q = miller.q[data.points.streak_id]

        check_close(modelled.source.q, q)

    def test_shared_intensity_gradient(self, model: ScalerModel, modelled: IntensityModel,
                                       data: ScalerData, setup: ResolvedSetup,
                                       xp: JaxNamespace) -> None:
        state = ScalerState.from_data(data, setup, sigma=0.02)

        loss = ScalerLoss(model)
        loss_grad_fn = cast(ScalerLossGradFn, jit(value_and_grad(loss, argnums=2)))
        value, gradient = loss_grad_fn(modelled, data, state)

        assert xp.isfinite(value)
        assert gradient.log_hkl.shape == (2,)
        assert xp.all(xp.isfinite(gradient.log_hkl))

    def test_result(self, model: ScalerModel, modelled: IntensityModel,
                        data: ScalerData, setup: ResolvedSetup,
                        xp: JaxNamespace) -> None:
        state = ScalerState.from_data(data, setup, sigma=0.02)
        result = model.init_result(modelled, data, state, PointGroup("4mm"), xp)
        log_profile = modelled.log_profile(state.log_sigma_at(data.points), xp)
        log_signal = model.log_signal(modelled, data, state, xp)
        log_expected = model.log_expected(data.counts, log_signal, xp)
        point_information = data.counts.I0 * xp.exp(2.0 * (log_profile - log_expected))
        information = xp.asarray([xp.sum(point_information[:2]),
                                  xp.sum(point_information[2:])])

        assert xp.all(result.miller.index == xp.asarray([0, 1]))
        assert xp.all(result.miller.hkl == xp.asarray([[1, 0, 0], [1, 0, 0]]))
        check_close(result.I_hkl, xp.exp(state.log_hkl))
        check_close(result.sigma_hkl, 1.0 / xp.sqrt(information))
        assert set(result.to_dict()) == {'miller', 'I_hkl', 'sigma_hkl'}

    def test_result_dataframe(self, model: ScalerModel, modelled: IntensityModel,
                              data: ScalerData, setup: ResolvedSetup,
                              xp: JaxNamespace) -> None:
        state = ScalerState.from_data(data, setup, sigma=0.02)
        result = model.init_result(modelled, data, state, PointGroup("4mm"), xp)

        imported = ScalerResult.import_dataframe(result.to_dataframe(), xp)

        assert xp.all(imported.miller.index == result.miller.index)
        assert xp.all(imported.miller.hkl == result.miller.hkl)
        check_close(imported.I_hkl, result.I_hkl)
        check_close(imported.sigma_hkl, result.sigma_hkl)

    def test_zero_information(self, model: ScalerModel, modelled: IntensityModel,
                              data: ScalerData, setup: ResolvedSetup,
                              xp: JaxNamespace) -> None:
        state = ScalerState.from_data(data, setup, sigma=0.02)
        counts = data.counts.replace(I0=xp.zeros_like(data.counts.I0))
        result = model.init_result(modelled, data.replace(counts=counts), state,
                                   PointGroup("4mm"), xp)

        assert xp.all(xp.isinf(result.sigma_hkl))
