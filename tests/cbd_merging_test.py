from typing import Callable, Tuple, cast
import pytest
from jax import jacfwd, jit, value_and_grad
from optax import Params, Updates, adam, apply_updates, cosine_decay_schedule
from cbclib_v2.annotations import AnyNamespace, IntArray, JaxNumPy, NumPy, RealArray
from cbclib_v2.indexer import Miller, SourcePlane
from cbclib_v2.indexer.cbc_pupil import RectangularPupil
from cbclib_v2.scaler import (IntensityModel, LegendreBasis, LegendreFactory, MergedList,
                              MergeState, OverlapData, PhotonCounts, PointGroup, PoissonData,
                              PoissonMerger, PoissonState, PupilState, ReflectionList, ScalerData,
                              ScalerModel, StreakData, StreakPoints, StudentMerger)
from cbclib_v2.scripts import optimisation_loop
from cbclib_v2.test_util import check_close, check_gradient

class TestPupilState:
    @pytest.fixture(params=[NumPy, JaxNumPy], ids=['numpy', 'jax'])
    def xp(self, request: pytest.FixtureRequest) -> AnyNamespace:
        return request.param

    @pytest.fixture(params=[0, 2])
    def basis(self, request: pytest.FixtureRequest, xp: AnyNamespace) -> LegendreBasis:
        return LegendreFactory.create(request.param, xp)

    @pytest.fixture(params=[1, 2], ids=['shared', 'per-pattern'])
    def initial(self, request: pytest.FixtureRequest, basis: LegendreBasis,
                xp: AnyNamespace) -> PupilState:
        return PupilState.default(basis, request.param, xp)

    @pytest.fixture
    def state(self, initial: PupilState, xp: AnyNamespace) -> PupilState:
        coefficients = xp.arange(initial.coefficients.size, dtype=float) + 1.0
        return initial.replace(coefficients=0.1 * coefficients.reshape(initial.coefficients.shape))

    @pytest.fixture
    def points(self, xp: AnyNamespace) -> RealArray:
        return xp.asarray([[0.23, 0.61], [0.0, 1.0], [-0.2, 1.3], [0.7, 0.4]])

    @pytest.fixture
    def index(self, xp: AnyNamespace) -> IntArray:
        return xp.asarray([0, 1, 1, 0])

    @pytest.fixture
    def quadrature_axis(self, xp: AnyNamespace) -> RealArray:
        return 0.5 * (1.0 + xp.asarray([-1.0, 0.0, 1.0]) * (3.0 / 5.0)**0.5)

    @pytest.fixture
    def quadrature_points(self, quadrature_axis: RealArray, xp: AnyNamespace) -> RealArray:
        u, v = xp.meshgrid(quadrature_axis, quadrature_axis, indexing='ij')
        return xp.stack((u, v), axis=-1).reshape((-1, 2))

    @pytest.fixture
    def quadrature_weights(self, xp: AnyNamespace) -> RealArray:
        weights = xp.asarray([5.0 / 18.0, 4.0 / 9.0, 5.0 / 18.0])
        return (weights[:, None] * weights[None, :]).reshape(-1)

    def test_unit_integral(self, state: PupilState, basis: LegendreBasis,
                           quadrature_points: RealArray, quadrature_weights: RealArray,
                           xp: AnyNamespace) -> None:
        for pattern in range(2):
            index = xp.full((quadrature_points.shape[0],), pattern, dtype=int)
            intensity = state.at(quadrature_points, index, basis)
            # Three-point Gauss integration is exact for squared degree-two pupil amplitudes.
            check_close(xp.sum(quadrature_weights * intensity), xp.asarray(1.0))
            assert xp.all(intensity >= 0.0)

    def test_polynomial_continuation(self, state: PupilState, basis: LegendreBasis,
                                     points: RealArray, index: IntArray,
                                     xp: AnyNamespace) -> None:
        u, v = points[:, 0], points[:, 1]
        u_modes = xp.stack((xp.ones_like(u), 3.0**0.5 * (2.0 * u - 1.0),
                            5.0**0.5 * (6.0 * u**2 - 6.0 * u + 1.0)), axis=-1)
        v_modes = xp.stack((xp.ones_like(v), 3.0**0.5 * (2.0 * v - 1.0),
                            5.0**0.5 * (6.0 * v**2 - 6.0 * v + 1.0)), axis=-1)
        rows = xp.zeros_like(index) if state.coefficients.shape[0] == 1 else index
        coefficients = state.coefficients[rows]
        products = u_modes[:, basis.modes[1:, 0]] * v_modes[:, basis.modes[1:, 1]]
        amplitude = 1.0 + xp.sum(products * coefficients, axis=-1)
        # Orthonormal shifted Legendre amplitudes continue without clipping for each pattern.
        expected = amplitude**2 / (1.0 + xp.sum(coefficients**2, axis=-1))
        check_close(state.at(points, index, basis), expected)

class TestPupilGradients:
    @pytest.fixture
    def xp(self) -> AnyNamespace:
        return JaxNumPy

    @pytest.fixture
    def basis(self, xp: AnyNamespace) -> LegendreBasis:
        return LegendreFactory.create(2, xp)

    @pytest.fixture
    def state(self, basis: LegendreBasis, xp: AnyNamespace) -> PupilState:
        initial = PupilState.default(basis, 1, xp)
        return initial.replace(coefficients=xp.full(initial.coefficients.shape, 0.1))

    @pytest.fixture(params=[(0.0, 1.0), (-0.2, 1.3)], ids=['boundary', 'outside'])
    def point(self, request: pytest.FixtureRequest, xp: AnyNamespace) -> RealArray:
        return xp.asarray(request.param)

    @pytest.fixture
    def node_basis(self, xp: AnyNamespace) -> LegendreBasis:
        return LegendreFactory.create(1, xp)

    @pytest.fixture
    def node_state(self, xp: AnyNamespace) -> PupilState:
        return PupilState(xp.asarray([[1.0 / 3.0**0.5, 0.0]]))

    @pytest.fixture
    def node(self, xp: AnyNamespace) -> RealArray:
        return xp.asarray([0.0, 0.5])

    def test_derivatives(self, state: PupilState, basis: LegendreBasis,
                         point: RealArray, xp: AnyNamespace) -> None:
        def intensity(pupil: PupilState, coordinates: RealArray) -> RealArray:
            return pupil.at(coordinates, xp.asarray(0), basis)

        # Shape and coordinate derivatives agree with finite differences beyond the aperture.
        check_gradient(jit(intensity), (state, point))

    def test_zero_amplitude(self, node_basis: LegendreBasis, node_state: PupilState,
                            node: RealArray, xp: AnyNamespace) -> None:
        def intensity(point: RealArray) -> RealArray:
            return node_state.at(point, xp.asarray(0), node_basis)

        derivative = jit(jacfwd(intensity))(node)
        # Squared amplitude has a smooth intensity zero and zero slope at an amplitude node.
        check_close(intensity(node), xp.asarray(0.0))
        check_close(derivative, xp.zeros(2))

class TestPupilCoordinates:
    @pytest.fixture(params=[NumPy, JaxNumPy], ids=['numpy', 'jax'])
    def xp(self, request: pytest.FixtureRequest) -> AnyNamespace:
        return request.param

    @pytest.fixture
    def roi(self, xp: AnyNamespace) -> RealArray:
        return xp.asarray([[-0.3, 0.1, -0.2, 0.4], [-0.1, 0.2, -0.4, 0.1]])

    @pytest.fixture
    def pupil(self, roi: RealArray) -> RectangularPupil:
        return RectangularPupil(roi)

    @pytest.fixture
    def unit(self, xp: AnyNamespace) -> RealArray:
        return xp.asarray([[0.0, 1.0], [-0.2, 1.3]])

    @pytest.fixture
    def kin(self, roi: RealArray, unit: RealArray, xp: AnyNamespace) -> RealArray:
        x = roi[:, 2] + unit[:, 0] * (roi[:, 3] - roi[:, 2])
        y = roi[:, 0] + unit[:, 1] * (roi[:, 1] - roi[:, 0])
        return xp.stack((x, y, xp.ones(2)), axis=-1)

    @pytest.fixture(params=[(0.1, -0.1, 1.0), (-0.2, 0.1, 1.0), (0.5, -0.4, 1.0)],
                    ids=['inside', 'boundary', 'outside'])
    def point(self, request: pytest.FixtureRequest, xp: AnyNamespace) -> RealArray:
        return xp.asarray(request.param)

    def test_affine_mapping(self, pupil: RectangularPupil, kin: RealArray,
                            unit: RealArray) -> None:
        # Bounds use (ky, kx) order; unit coordinates follow (kx, ky) without clipping.
        check_close(pupil.coordinates(kin), unit)

    @pytest.mark.parametrize('xp', [JaxNumPy], indirect=True, ids=['jax'])
    def test_coordinate_gradient(self, pupil: RectangularPupil, roi: RealArray,
                                 point: RealArray, xp: AnyNamespace) -> None:
        gradient = jit(jacfwd(pupil.coordinates))(point)
        widths = xp.stack((roi[:, 3] - roi[:, 2], roi[:, 1] - roi[:, 0]), axis=-1)
        expected = xp.eye(2)[None] / widths[..., None]
        expected = xp.concat((expected, xp.zeros((2, 2, 1))), axis=-1)
        # An affine coordinate map has reciprocal-width slopes everywhere, including outside.
        check_close(gradient, expected)

class TestReflectionList:
    @pytest.fixture
    def xp(self) -> AnyNamespace:
        return NumPy

    @pytest.fixture
    def first(self, xp: AnyNamespace) -> ReflectionList:
        return ReflectionList(index=xp.asarray([0, 0]),
                              hkl=xp.asarray([[1, 0, 0], [0, 1, 0]]),
                              I_hkl=xp.asarray([2.0, 3.0]),
                              sigma_hkl=xp.asarray([1.0, 2.0]))

    @pytest.fixture
    def parts(self, first: ReflectionList, xp: AnyNamespace) -> tuple[ReflectionList, ...]:
        unsorted = first.replace(index=xp.asarray([2, 0]))
        second = first.replace(I_hkl=2.0 * first.I_hkl)
        return first[:0], unsorted, second, first, first[:0]

    @pytest.fixture(params=[False, True], ids=['global-indices', 'local-indices'])
    def monotonic(self, request: pytest.FixtureRequest) -> bool:
        return request.param

    @pytest.fixture
    def reflections(self, first: ReflectionList, xp: AnyNamespace) -> ReflectionList:
        return first.replace(index=xp.asarray([0, 1]), hkl=-first.hkl)

    @pytest.fixture
    def frames(self, xp: AnyNamespace) -> IntArray:
        return xp.asarray([10, 20])

    @pytest.fixture
    def equivalents(self, first: ReflectionList, xp: AnyNamespace) -> ReflectionList:
        return first.replace(hkl=xp.asarray([[1, 2, 3], [-2, 1, 3]]))

    @pytest.fixture
    def point_group(self) -> PointGroup:
        return PointGroup('4mm')

    @pytest.fixture
    def mapping(self, reflections: ReflectionList) -> MergedList:
        return reflections.merge(PointGroup('1'))

    @pytest.fixture
    def order(self, xp: AnyNamespace) -> IntArray:
        return xp.asarray([1, 0])

    def test_concat(self, parts: tuple[ReflectionList, ...], monotonic: bool,
                    xp: AnyNamespace) -> None:
        result = ReflectionList.concat(iter(parts), monotonic_index=monotonic)
        offset = 0
        row = 0
        for part in parts:
            size = part.I_hkl.size
            selected = result[row:row + size]
            # Concatenation preserves each observation and its pattern association.
            assert xp.all(selected.index == part.index + offset)
            for name in ('hkl', 'I_hkl', 'sigma_hkl'):
                assert xp.all(getattr(selected, name) == getattr(part, name))
            if monotonic and size:
                offset += int(xp.max(part.index)) + 1
            row += size
        # Combining lists neither loses nor duplicates integrated observations.
        check_close(xp.sum(result.I_hkl), sum(xp.sum(part.I_hkl) for part in parts))

    def test_mapping(self, equivalents: ReflectionList, point_group: PointGroup,
                     xp: AnyNamespace) -> None:
        mapping = equivalents.merge(point_group)
        canonical = point_group.canonical(equivalents.hkl, xp)
        # A quarter turn about c leaves the canonical reflection and observations unchanged.
        assert xp.all(mapping.hkl[mapping.reflection_id] == canonical)
        assert mapping.reflection_id[0] == mapping.reflection_id[1]
        check_close(mapping.I_hkl, equivalents.I_hkl)
        check_close(mapping.sigma_hkl, equivalents.sigma_hkl)

    def test_selection(self, mapping: MergedList, order: IntArray,
                       xp: AnyNamespace) -> None:
        selected = mapping.reflection_at(order)
        # Reordering reflections preserves the intensity/Miller/pattern association.
        assert xp.all(selected.hkl == mapping.hkl[order])
        original_hkl = mapping.hkl[mapping.reflection_id]
        assert xp.all(selected.hkl[selected.reflection_id] == original_hkl[selected.index])
        check_close(selected.I_hkl, mapping.I_hkl[selected.index])
        check_close(selected.sigma_hkl, mapping.sigma_hkl[selected.index])

    def test_dataframe(self, reflections: ReflectionList, frames: IntArray,
                       xp: AnyNamespace) -> None:
        result = ReflectionList.import_dataframe(reflections.to_dataframe(frames), frames, xp)
        # The round trip preserves signed Miller representatives and their measured quantities.
        for name in ('index', 'hkl', 'I_hkl', 'sigma_hkl'):
            assert xp.all(getattr(result, name) == getattr(reflections, name))

class TestStudentMerger:
    @pytest.fixture
    def xp(self) -> AnyNamespace:
        return JaxNumPy

    @pytest.fixture
    def model(self) -> StudentMerger:
        return StudentMerger()

    @pytest.fixture
    def scales(self, xp: AnyNamespace) -> RealArray:
        return xp.asarray([0.5, 1.0, 2.0, 1.5, 0.8, 1.2])

    @pytest.fixture
    def intensities(self, xp: AnyNamespace) -> RealArray:
        return xp.asarray([-10., 20., 30., 40.])

    @pytest.fixture
    def reflection_list(self, scales: RealArray, intensities: RealArray,
                        xp: AnyNamespace) -> ReflectionList:
        return ReflectionList(index=xp.repeat(xp.arange(scales.size), intensities.size),
                              hkl=xp.tile(xp.asarray([[1, 0, 0], [2, 0, 0],
                                                     [3, 0, 0], [4, 0, 0]]), (6, 1)),
                              I_hkl=cast(RealArray, (scales[:, None] * intensities).reshape(-1)),
                              sigma_hkl=xp.ones(scales.size * intensities.size))

    @pytest.fixture
    def data(self, reflection_list: ReflectionList) -> MergedList:
        return reflection_list.merge(PointGroup('1'))

    @pytest.fixture
    def initial(self, data: MergedList) -> MergeState:
        return MergeState.from_data(data)

    @pytest.fixture
    def corrupt(self, data: MergedList, xp: AnyNamespace) -> MergedList:
        return data.replace(I_hkl=data.I_hkl + xp.where(xp.arange(data.I_hkl.size) == 0,
                                                        200.0, 0.0))

    @pytest.fixture
    def unsupported(self, data: MergedList, xp: AnyNamespace) -> MergedList:
        selected = (data.index == 0) | (data.reflection_id == 0)
        return data.replace(sigma_hkl=xp.where(selected, xp.inf, data.sigma_hkl))

    @pytest.fixture(params=[0.1, 3.0])
    def unit_factor(self, request: pytest.FixtureRequest) -> float:
        return request.param

    @pytest.fixture
    def converted(self, data: MergedList, unit_factor: float) -> MergedList:
        return data.replace(I_hkl=unit_factor * data.I_hkl,
                            sigma_hkl=unit_factor * data.sigma_hkl)

    @pytest.fixture
    def converted_state(self, initial: MergeState, unit_factor: float,
                        xp: AnyNamespace) -> MergeState:
        return initial.replace(asinh_hkl=xp.arcsinh(unit_factor * xp.sinh(initial.asinh_hkl)))

    @pytest.fixture
    def shifted(self, initial: MergeState, xp: AnyNamespace) -> MergeState:
        return initial.replace(log_scale=initial.log_scale + 2.0,
                               asinh_hkl=xp.arcsinh(xp.sinh(initial.asinh_hkl)
                                                    * xp.exp(xp.asarray(-2.0))))

    @pytest.fixture(params=[False, True], ids=['finite', 'infinite-uncertainty'])
    def median_data(self, request: pytest.FixtureRequest, data: MergedList,
                    xp: AnyNamespace) -> MergedList:
        if not request.param:
            return data
        first = xp.arange(data.I_hkl.size) == 0
        return data.replace(I_hkl=xp.where(first, 1e6, data.I_hkl),
                            sigma_hkl=xp.where(first, xp.inf, data.sigma_hkl))

    def optimise(self, model: StudentMerger, data: MergedList, initial: MergeState,
                 num_steps: int=3000) -> MergeState:
        loss_grad_fn = jit(value_and_grad(model, argnums=1))
        optimiser = adam(cosine_decay_schedule(0.05, num_steps, alpha=1e-4))
        opt_state = optimiser.init(cast(Params, initial))
        state = initial
        for _ in range(num_steps):
            _, gradient = loss_grad_fn(data, state)
            updates, opt_state = optimiser.update(cast(Updates, gradient), opt_state)
            state = cast(MergeState, apply_updates(state, updates))
        return state.normalise()

    def test_initialisation(self, median_data: MergedList, xp: AnyNamespace) -> None:
        state = MergeState.from_data(median_data)
        for reflection in range(median_data.n_reflections):
            observed = xp.sort(median_data.I_hkl[median_data.reflection_id == reflection])
            # Initialisation uses the upper median of all observations regardless of uncertainty.
            check_close(xp.sinh(state.asinh_hkl[reflection]), observed[observed.size // 2])

    def test_recovery(self, model: StudentMerger, data: MergedList, initial: MergeState,
                      xp: AnyNamespace) -> None:
        state = self.optimise(model, data, initial)
        # One signed intensity per reflection and one scale per pattern explain all observations.
        check_close(model.expected(data, state), data.I_hkl, rtol=1e-5)
        check_close(xp.mean(state.log_scale), xp.asarray(0.0), atol=1e-6)

    def test_gauge(self, model: StudentMerger, data: MergedList,
                   initial: MergeState, shifted: MergeState, xp: AnyNamespace) -> None:
        # Reciprocal intensity/scale changes and gauge fixing preserve predictions and loss.
        check_close(model.expected(data, shifted), model.expected(data, initial))
        check_close(model.expected(data, shifted.normalise()), model.expected(data, initial))
        check_close(model(data, shifted.normalise()), model(data, initial))
        check_close(xp.mean(shifted.normalise().log_scale), xp.asarray(0.0), atol=1e-6)

    def test_outlier(self, model: StudentMerger, data: MergedList, corrupt: MergedList,
                     initial: MergeState, xp: AnyNamespace) -> None:
        state = self.optimise(model, corrupt, initial)
        # Student tails suppress one bad observation while recovering the common clean model.
        check_close(model.expected(data, state), data.I_hkl, atol=0.1)
        residuals = xp.abs(model.residuals(corrupt, state))
        assert residuals[0] > 100.0 * xp.max(residuals[1:])

    def test_unsupported(self, model: StudentMerger, unsupported: MergedList,
                         initial: MergeState, xp: AnyNamespace) -> None:
        _, gradient = value_and_grad(model, argnums=1)(unsupported, initial)
        # Infinite uncertainty carries no information about the affected parameters.
        assert gradient.log_scale[0] == 0.0
        assert gradient.asinh_hkl[0] == 0.0

    def test_uncertainty_units(self, model: StudentMerger, data: MergedList,
                               initial: MergeState, converted: MergedList,
                               converted_state: MergeState) -> None:
        # Changing intensity and uncertainty units together preserves dimensionless likelihood.
        check_close(model(converted, converted_state), model(data, initial))

class TestPoissonRecovery:
    @pytest.fixture
    def xp(self) -> AnyNamespace:
        return JaxNumPy

    @pytest.fixture
    def pupil_xy(self, xp: AnyNamespace) -> RealArray:
        axis = xp.asarray([0.1, 0.35, 0.65, 0.9])
        u, v = xp.meshgrid(axis, axis, indexing='ij')
        return xp.stack((u, v), axis=-1).reshape((-1, 2))

    @pytest.fixture
    def points(self, pupil_xy: RealArray, xp: AnyNamespace) -> StreakPoints:
        n_points = pupil_xy.shape[0]
        index = xp.repeat(xp.arange(3), 2 * n_points)
        streak_id = xp.repeat(xp.arange(6), n_points)
        positions = xp.tile(pupil_xy, (6, 1))
        positions = positions + xp.stack((streak_id, 0.0 * streak_id), axis=-1)
        return StreakPoints(index, streak_id, positions)

    @pytest.fixture
    def miller(self, xp: AnyNamespace) -> Miller:
        return Miller(index=xp.repeat(xp.arange(3), 2),
                      hkl=xp.tile(xp.asarray([[1, 0, 0], [2, 0, 0]]), (3, 1)))

    @pytest.fixture
    def modelled(self, pupil_xy: RealArray,
                 xp: AnyNamespace) -> IntensityModel:
        xy = xp.tile(0.2 * pupil_xy - 0.1, (6, 1))
        z = xp.sqrt(1.0 - xp.sum(xy**2, axis=-1))
        kin = xp.concat((xy, z[:, None]), axis=-1)
        q = xp.broadcast_to(xp.asarray([0.0, 0.0, -2.0]), kin.shape)
        pupil = RectangularPupil(xp.asarray([[-0.1, 0.1, -0.1, 0.1]]))
        return IntensityModel(kout=kin + q, source=SourcePlane.from_q(q), pupil=pupil,
                              smp_pos=xp.zeros_like(kin))

    @pytest.fixture
    def counts(self, points: StreakPoints, xp: AnyNamespace) -> PhotonCounts:
        return PhotonCounts(I0=xp.zeros(points.size, dtype=int),
                            background=xp.full((points.size,), 3.0),
                            std=xp.ones(points.size))

    @pytest.fixture
    def scaling(self, points: StreakPoints, counts: PhotonCounts, miller: Miller) -> ScalerData:
        return ScalerData(points, counts, miller)

    @pytest.fixture
    def streaks(self, scaling: ScalerData, modelled: IntensityModel) -> StreakData:
        return StreakData(scaling, modelled)

    @pytest.fixture
    def basis(self, xp: AnyNamespace) -> LegendreBasis:
        return LegendreFactory.create(1, xp)

    @pytest.fixture
    def unmeasured(self, streaks: StreakData, basis: LegendreBasis) -> PoissonData:
        return PoissonData.from_streaks(streaks, PointGroup('1'), basis)

    @pytest.fixture
    def merger(self) -> PoissonMerger:
        return PoissonMerger(ScalerModel())

    @pytest.fixture(params=[False, True], ids=['shared', 'per-pattern'])
    def truth(self, request: pytest.FixtureRequest, xp: AnyNamespace) -> PoissonState:
        coefficients = [[0.16, -0.12], [-0.08, 0.11], [0.05, 0.07]]
        if not request.param:
            coefficients = coefficients[:1]
        return PoissonState(log_scale=xp.log(xp.asarray([0.8, 1.25, 1.0])),
                            asinh_hkl=xp.arcsinh(xp.asarray([-1.5, 6.0])),
                            log_sigma=xp.log(xp.asarray([0.03, 0.035, 0.04])),
                            pupil=PupilState(xp.asarray(coefficients)))

    @pytest.fixture
    def data(self, unmeasured: PoissonData, truth: PoissonState, merger: PoissonMerger,
             xp: AnyNamespace) -> PoissonData:
        # Expected-count data isolate optimisation accuracy from photon sampling noise.
        counts = unmeasured.counts.replace(I0=xp.exp(merger.log_expected(unmeasured, truth, xp)))
        return unmeasured.replace(scaling=unmeasured.scaling.replace(counts=counts))

    @pytest.fixture
    def initial_pupil(self, data: PoissonData, truth: PoissonState,
                      xp: AnyNamespace) -> PupilState:
        return PupilState.default(data.basis, truth.pupil.coefficients.shape[0], xp)

    @pytest.fixture
    def initial(self, data: PoissonData, truth: PoissonState,
                 initial_pupil: PupilState) -> PoissonState:
        return PoissonState.from_data(data, truth.log_sigma, initial_pupil)

    def test_recovery(self, data: PoissonData, truth: PoissonState, initial: PoissonState,
                      merger: PoissonMerger, xp: AnyNamespace) -> None:
        # The generating expected counts attain the saturated Poisson model.
        check_close(merger.poisson_deviance(data, truth, xp),
                    xp.zeros(data.counts.shape))
        loss_grad = jit(value_and_grad(merger, argnums=1))
        gradient: Callable[[PoissonState], Tuple[RealArray, PoissonState]] = (
            lambda state: loss_grad(data, state))
        num_steps = 1500
        schedule = cosine_decay_schedule(0.04, num_steps, alpha=0.01)
        fitted, _ = optimisation_loop(gradient, initial, adam(schedule), schedule,
                                      num_steps, trace_every=100)
        fitted = fitted.normalise()

        # Unit-integral pupils and unit geometric mean scale identify the generating model.
        check_close(xp.exp(merger.log_expected(data, fitted, xp)), data.counts.I0,
                    rtol=2e-3, atol=2e-3)
        check_close(xp.sinh(fitted.asinh_hkl), xp.sinh(truth.asinh_hkl), rtol=2e-3, atol=2e-3)
        check_close(xp.exp(fitted.log_scale), xp.exp(truth.log_scale),
                    rtol=2e-3, atol=2e-3)
        check_close(fitted.pupil.coefficients, truth.pupil.coefficients, rtol=2e-3, atol=2e-3)

class ScanCase:
    @pytest.fixture
    def xp(self) -> AnyNamespace:
        return NumPy

    @pytest.fixture
    def overlap(self) -> bool:
        return True

    @pytest.fixture
    def points(self, xp: AnyNamespace) -> StreakPoints:
        return StreakPoints(index=xp.asarray([0, 0, 0, 0, 2, 2, 2, 2]),
                            streak_id=xp.repeat(xp.arange(4), 2),
                            points=xp.asarray([[0.0, 0.1], [0.1, 0.2], [0.1, 0.2], [0.2, 0.3],
                                               [0.3, 0.4], [0.4, 0.5], [0.4, 0.5], [0.5, 0.6]]))

    @pytest.fixture
    def miller(self, xp: AnyNamespace) -> Miller:
        return Miller(index=xp.asarray([0, 0, 2, 2]),
                      hkl=xp.asarray([[1, 2, 3], [-2, 1, 3], [2, -1, 3], [1, 1, 2]]))

    @pytest.fixture
    def counts(self, overlap: bool, xp: AnyNamespace) -> PhotonCounts:
        n_pixels = 6 if overlap else 8
        return PhotonCounts(I0=xp.arange(n_pixels, dtype=float) + 10,
                            background=xp.full((n_pixels,), 2.0),
                            std=xp.ones(n_pixels))

    @pytest.fixture
    def pixel_index(self, overlap: bool, xp: AnyNamespace) -> IntArray:
        if overlap:
            return xp.asarray([0, 1, 1, 2, 3, 4, 4, 5])
        return xp.arange(8)

    @pytest.fixture
    def scaling(self, points: StreakPoints, miller: Miller, counts: PhotonCounts,
                overlap: bool, pixel_index: IntArray) -> ScalerData | OverlapData:
        if overlap:
            return OverlapData(points, counts, miller, pixel_index)
        return ScalerData(points, counts, miller)

    @pytest.fixture
    def modelled(self, points: StreakPoints, xp: AnyNamespace) -> IntensityModel:
        kin = xp.concat((points.points, xp.ones((points.size, 1))), axis=-1)
        q = xp.broadcast_to(xp.asarray([0.0, 0.0, -0.2]), kin.shape)
        pupil = RectangularPupil(xp.asarray([[-0.25, 0.25, -0.25, 0.25]]))
        return IntensityModel(kout=kin + q, source=SourcePlane.from_q(q), pupil=pupil,
                              smp_pos=xp.zeros_like(kin))

    @pytest.fixture
    def streak_data(self, scaling: ScalerData | OverlapData,
                    modelled: IntensityModel) -> StreakData:
        return StreakData(scaling, modelled)

    @pytest.fixture
    def log_sigma(self, xp: AnyNamespace) -> RealArray:
        return xp.log(xp.asarray([0.1, 0.2, 0.3]))

    @pytest.fixture
    def point_group(self) -> PointGroup:
        return PointGroup('4mm')

    @pytest.fixture
    def basis(self, xp: AnyNamespace) -> LegendreBasis:
        return LegendreFactory.create(1, xp)

    @pytest.fixture
    def data(self, streak_data: StreakData,
             point_group: PointGroup, basis: LegendreBasis) -> PoissonData:
        return PoissonData.from_streaks(streak_data, point_group, basis)

    @pytest.fixture
    def pupil(self, data: PoissonData, xp: AnyNamespace) -> PupilState:
        pupil = PupilState.default(data.basis, data.n_patterns, xp)
        coefficients = xp.arange(pupil.coefficients.size, dtype=float) + 1.0
        return pupil.replace(coefficients=0.1 * coefficients.reshape(pupil.coefficients.shape))

    @pytest.fixture
    def state(self, data: PoissonData, log_sigma: RealArray, pupil: PupilState,
              xp: AnyNamespace) -> PoissonState:
        return PoissonState(log_scale=xp.log(xp.asarray([2.0, 3.0, 4.0])),
                            asinh_hkl=xp.arcsinh(
                                xp.arange(data.n_reflections, dtype=float) - 0.5),
                            log_sigma=log_sigma, pupil=pupil)

class TestPoissonData(ScanCase):
    def test_symmetry_binding(self, data: PoissonData, point_group: PointGroup,
                              xp: AnyNamespace) -> None:
        canonical = point_group.canonical(data.miller.hkl_indices, xp)
        # Equivalent streaks across patterns share a single canonical reflection.
        assert xp.all(data.hkl[data.reflection_id] == canonical)
        assert xp.all(data.reflection_id[:3] == data.reflection_id[0])
        assert data.reflection_id[3] != data.reflection_id[0]

    def test_profile(self, data: PoissonData, state: PoissonState, xp: AnyNamespace) -> None:
        sigma = xp.exp(state.log_sigma_at(data))
        # Each point's Gaussian attenuation uses its pattern's fixed width.
        expected = xp.exp(-0.5 * data.modelled.distance_sq / sigma**2)
        check_close(data.modelled.profile(state.log_sigma_at(data), xp), expected)

class TestPixelAggregation(ScanCase):
    @pytest.fixture(params=[False, True], ids=['ordinary', 'overlap'])
    def overlap(self, request: pytest.FixtureRequest) -> bool:
        return request.param

    @pytest.fixture
    def signal(self, data: PoissonData, xp: AnyNamespace) -> RealArray:
        return xp.arange(data.points.size, dtype=float) + 1.0

    def test_pixel_aggregation(self, data: PoissonData, signal: RealArray,
                               pixel_index: IntArray, xp: AnyNamespace) -> None:
        predicted = data.predictor(signal, xp)
        # Each pixel receives all its streak contributions and exactly one background term.
        expected = xp.stack([xp.sum(signal[pixel_index == pixel])
                             for pixel in range(data.counts.I0.size)])
        check_close(predicted, data.counts.background + expected)
        # Aggregation conserves total signal even when several streaks share a pixel.
        check_close(xp.sum(predicted - data.counts.background), xp.sum(signal))

class TestPoissonState(ScanCase):
    def test_point_lookup(self, data: PoissonData, state: PoissonState, xp: AnyNamespace) -> None:
        canonical = data.hkl[data.reflection_id][data.points.streak_id]
        for reflection_id, hkl in enumerate(data.hkl):
            selected = xp.all(canonical == hkl, axis=-1)
            # Symmetry-related pixels share the same signed reflection coefficient.
            check_close(state.intensity_at(data)[selected],
                        xp.full((int(xp.sum(selected)),), xp.sinh(state.asinh_hkl[reflection_id])))
        # Gaps in observed pattern indices do not shift the illumination scales.
        check_close(state.scale_at(data), xp.exp(state.log_scale[data.points.index]))

class PoissonCase(ScanCase):
    @pytest.fixture
    def log_sigma(self, xp: AnyNamespace) -> RealArray:
        return xp.log(xp.asarray([0.8, 0.9, 1.0]))

    @pytest.fixture
    def basis(self, xp: AnyNamespace) -> LegendreBasis:
        return LegendreFactory.create(2, xp)

    @pytest.fixture
    def merger(self) -> PoissonMerger:
        return PoissonMerger(ScalerModel())

    @pytest.fixture(params=[1, 3], ids=['shared', 'per-pattern'])
    def pupil(self, request: pytest.FixtureRequest, data: PoissonData,
               xp: AnyNamespace) -> PupilState:
        pupil = PupilState.default(data.basis, request.param, xp)
        coefficients = xp.arange(pupil.coefficients.size, dtype=float) + 1.0
        return pupil.replace(coefficients=0.01 * coefficients.reshape(pupil.coefficients.shape))

class TestScanInitialisation(PoissonCase):
    @pytest.fixture
    def pupil(self, data: PoissonData, xp: AnyNamespace) -> PupilState:
        return PupilState.default(data.basis, 1, xp)

    @pytest.fixture
    def initial(self, data: PoissonData, log_sigma: RealArray, pupil: PupilState) -> PoissonState:
        return PoissonState.from_data(data, log_sigma, pupil)

    @pytest.fixture(params=[0.0, -1.0])
    def measured_signal(self, request: pytest.FixtureRequest) -> float:
        return request.param

    @pytest.fixture
    def signed_data(self, data: PoissonData, measured_signal: float) -> PoissonData:
        counts = data.counts.replace(I0=data.counts.background + measured_signal)
        return data.replace(scaling=data.scaling.replace(counts=counts))

    @pytest.fixture
    def signed_initial(self, signed_data: PoissonData, log_sigma: RealArray,
                       pupil: PupilState) -> PoissonState:
        return PoissonState.from_data(signed_data, log_sigma, pupil)

    def test_profile_means(self, data: PoissonData, initial: PoissonState,
                           xp: AnyNamespace) -> None:
        profile = data.modelled.profile(initial.log_sigma_at(data), xp)
        for reflection in range(data.n_reflections):
            selected = data.reflection_id[data.points.streak_id] == reflection
            # A weighted mean conserves the profile-weighted signal within each reflection.
            expected = xp.sum(profile[selected] * data.signal[selected])
            actual = xp.sinh(initial.asinh_hkl[reflection]) * xp.sum(profile[selected])
            check_close(actual, expected)

    def test_nonpositive_signal(self, signed_initial: PoissonState,
                                measured_signal: float, xp: AnyNamespace) -> None:
        # A constant signed signal is a fixed point of profile-weighted averaging.
        check_close(xp.sinh(signed_initial.asinh_hkl),
                    xp.full(signed_initial.asinh_hkl.shape, measured_signal))

class TestPoissonMerger(PoissonCase):
    @pytest.fixture
    def zero_counts(self, data: PoissonData, xp: AnyNamespace) -> PoissonData:
        counts = data.counts.replace(I0=xp.zeros_like(data.counts.I0))
        return data.replace(scaling=data.scaling.replace(counts=counts))

    def test_likelihood(self, data: PoissonData, state: PoissonState,
                        merger: PoissonMerger, xp: AnyNamespace) -> None:
        coordinates = data.modelled.pupil_xy
        pupil = state.pupil.at(coordinates, data.points.index, data.basis)
        profile = data.modelled.profile(state.log_sigma_at(data), xp)
        signal = state.scale_at(data) * state.intensity_at(data) * pupil * profile
        predictor = data.predictor(signal, xp)
        expected = merger.model.rate_floor + xp.logaddexp(predictor, 0.0)
        # Each measured pixel contributes its softplus Poisson likelihood exactly once.
        loss = expected - data.counts.I0 * xp.log(expected)
        check_close(xp.exp(merger.log_expected(data, state, xp)), expected)
        check_close(merger.poisson_loss(data, state, xp), loss)
        check_close(merger(data, state), xp.mean(loss))

    def test_gauge_invariance(self, data: PoissonData, state: PoissonState,
                              merger: PoissonMerger, xp: AnyNamespace) -> None:
        normalised = state.normalise()
        # The gauge convention gives unit geometric mean scale, including unobserved patterns.
        check_close(xp.mean(normalised.log_scale), xp.asarray(0.0))
        # Scale normalisation leaves expected counts and the Poisson objective invariant.
        check_close(merger.log_expected(data, normalised, xp),
                    merger.log_expected(data, state, xp))
        check_close(merger(data, normalised), merger(data, state))

    def test_zero_count(self, zero_counts: PoissonData, state: PoissonState,
                        merger: PoissonMerger, xp: AnyNamespace) -> None:
        # Zero observed photons contribute only the expected count to the Poisson loss.
        check_close(merger.poisson_loss(zero_counts, state, xp),
                    xp.exp(merger.log_expected(zero_counts, state, xp)))

class TestPoissonMergerGradients(PoissonCase):
    @pytest.fixture
    def xp(self) -> AnyNamespace:
        return JaxNumPy

    def test_gradient(self, data: PoissonData, state: PoissonState, merger: PoissonMerger) -> None:
        loss = jit(merger)
        # The full softplus likelihood differentiates through every scan parameter group.
        check_gradient(lambda state: loss(data, state), (state,))
