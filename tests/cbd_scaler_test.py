import pytest
from jax import jit, tree, value_and_grad
from cbclib_v2.annotations import JaxNamespace, JaxNumPy, NumPy, NumPyNamespace, RealArray
from cbclib_v2.indexer import (FixedPupilSetup, Miller, Rectangle, ResolvedSetup, SourcePlane)
from cbclib_v2.scaler import (IntensityModel, PatternData, PhotonCounts, ScalerData, ScalerLoss,
                              ScalerModel, ScalerState, SetupLoss, StreakPoints)
from cbclib_v2.test_util import check_close, TestSetup

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
        log_expected = xp.log(xp.asarray([2.0]))
        y_hat = xp.asarray([5.000001])

        loss = model.poisson_loss(counts, log_expected, xp)

        check_close(loss, y_hat - counts.I0 * xp.log(y_hat))

    def test_signal_underflow(self, xp: NumPyNamespace) -> None:
        model = ScalerModel()
        counts = PhotonCounts(I0=xp.asarray([1]), background=xp.asarray([0.0]))

        loss = model.poisson_loss(counts, xp.asarray([-1000.0]), xp)

        assert xp.all(xp.isfinite(loss))

    def test_rate_floor(self, xp: NumPyNamespace) -> None:
        model = ScalerModel()
        counts = PhotonCounts(I0=xp.asarray([0]), background=xp.asarray([0.0]))

        loss = model.poisson_loss(counts, xp.asarray([-xp.inf]), xp)

        check_close(loss, xp.asarray([1e-6]))

    def test_custom_rate_floor(self, xp: NumPyNamespace) -> None:
        model = ScalerModel(rate_floor=0.25)
        counts = PhotonCounts(I0=xp.asarray([0]), background=xp.asarray([0.0]))

        loss = model.poisson_loss(counts, xp.asarray([-xp.inf]), xp)

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
    def data(self, xp: JaxNamespace) -> PatternData:
        points = StreakPoints(index=xp.asarray([0, 0, 1, 1]),
                              streak_id=xp.asarray([0, 0, 1, 1]),
                              points=xp.asarray([[0.12, 0.14], [0.13, 0.15],
                                                 [0.18, 0.16], [0.19, 0.17]]))
        counts = PhotonCounts(I0=xp.asarray([11, 13, 17, 19]),
                              background=xp.asarray([2.0, 3.0, 2.0, 4.0]))
        miller = Miller(index=xp.asarray([0, 1]),
                        hkl=xp.asarray([[1, 0, 0], [0, 1, 0]]))
        return PatternData(points=points, counts=counts, miller=miller)

    @pytest.fixture
    def state(self, xp: JaxNamespace) -> ScalerState:
        return ScalerState.default(n_streaks=2, n_frames=2, sigma=0.02, xp=xp)

    @pytest.fixture
    def scaler_data(self, model: ScalerModel, data: PatternData,
                    setup: ResolvedSetup, xp: JaxNamespace) -> ScalerData:
        return model.scaler_data(data, setup, xp)

    def check_gradient(self, value: RealArray, gradient: ScalerState | ResolvedSetup,
                       xp: JaxNamespace) -> None:
        leaves = tree.leaves(gradient)

        assert xp.isfinite(value)
        assert all(xp.all(xp.isfinite(leaf)) for leaf in leaves)
        assert any(xp.any(leaf != 0.0) for leaf in leaves)

    def test_scaler(self, model: ScalerModel, scaler_data: ScalerData,
                    state: ScalerState, xp: JaxNamespace) -> None:
        loss = ScalerLoss(model)
        value, gradient = jit(value_and_grad(loss, argnums=1))(scaler_data, state)

        self.check_gradient(value, gradient, xp)

    def test_setup(self, model: ScalerModel, data: PatternData, state: ScalerState,
                   setup: ResolvedSetup, xp: JaxNamespace) -> None:
        loss = SetupLoss(model)
        value, gradient = jit(value_and_grad(loss, argnums=2))(data, state, setup)

        self.check_gradient(value, gradient, xp)
