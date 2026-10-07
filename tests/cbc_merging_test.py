from typing import cast
from jax import jit, value_and_grad
from optax import Params, Updates, adam, apply_updates, cosine_decay_schedule
import pytest
from cbclib_v2.annotations import AnyNamespace, JaxNumPy, NumPy, RealArray
from cbclib_v2.scaler import MergeData, MergeModel, MergeState, PointGroup, ReflectionList

class TestReflectionList:
    @pytest.fixture(params=[NumPy, JaxNumPy], ids=['numpy', 'jax'])
    def xp(self, request: pytest.FixtureRequest) -> AnyNamespace:
        return request.param

    @pytest.fixture
    def point_group(self) -> PointGroup:
        return PointGroup("1")

    @pytest.fixture
    def first(self, xp: AnyNamespace) -> ReflectionList:
        return ReflectionList(index=xp.asarray([0, 0]),
                              hkl=xp.asarray([[1, 0, 0], [0, 1, 0]]),
                              I_hkl=xp.asarray([2.0, 3.0]),
                              sigma_hkl=xp.asarray([1.0, 2.0]))

    @pytest.fixture
    def second(self, first: ReflectionList) -> ReflectionList:
        return first.replace(I_hkl=2.0 * first.I_hkl)

    @pytest.fixture
    def third(self, first: ReflectionList) -> ReflectionList:
        return first.replace(I_hkl=3.0 * first.I_hkl)

    @pytest.fixture
    def reflections(self, first: ReflectionList, second: ReflectionList,
                    third: ReflectionList) -> ReflectionList:
        return ReflectionList.concat(iter((first, second, third)))

    def test_concat(self, reflections: ReflectionList, first: ReflectionList,
                    xp: AnyNamespace):
        # Independent local pattern-zero groups must remain distinct patterns.
        assert xp.all(reflections.index == xp.repeat(xp.arange(3), 2))
        assert xp.all(reflections.hkl == xp.tile(first.hkl, (3, 1)))
        assert xp.all(reflections.I_hkl == xp.asarray([2., 3., 4., 6., 6., 9.]))
        assert reflections.n_patterns == 3

    def test_preserve_global_indices(self, first: ReflectionList,
                                     second: ReflectionList, xp: AnyNamespace):
        result = ReflectionList.concat((first, second), monotonic_index=False)
        assert xp.all(result.index == 0)
        assert len(result) == 1

    def test_unsorted_concat(self, first: ReflectionList, xp: AnyNamespace):
        unsorted = first.replace(index=xp.asarray([2, 0]))
        result = ReflectionList.concat((unsorted, first, first))
        # Shifts follow the full index range without rearranging observations.
        assert xp.all(result.index == xp.asarray([2, 0, 3, 3, 4, 4]))
        assert result.n_patterns == 5

    def test_mapping(self, reflections: ReflectionList, xp: AnyNamespace):
        point_group = PointGroup("4mm")
        reflections = reflections.replace(
            hkl=xp.tile(xp.asarray([[1, 2, 3], [-2, 1, 3]]), (3, 1)))
        mapping = MergeData.from_reflections(reflections, point_group)
        hkl, reflection_id = mapping.hkl, mapping.reflection_id
        canonical = point_group.canonical(reflections.hkl, xp)

        # Symmetry-equivalent observations share one canonical reflection.
        assert xp.all(hkl[reflection_id] == canonical)
        assert hkl.shape == (1, 3)
        assert xp.all(reflection_id == 0)

    def test_reference_update(self, reflections: ReflectionList, point_group: PointGroup,
                              xp: AnyNamespace):
        data = MergeData.from_reflections(reflections, point_group)
        state = MergeState.from_data(data, xp.log(xp.asarray([1., 2., 3.])))
        fitted = data.fit_scales(state.I_hkl, xp.ones_like(data.I_hkl),
                                 xp.ones(data.n_patterns))
        # Fixed shared intensities recover the exact illumination scale of every pattern.
        assert xp.allclose(fitted, xp.exp(state.log_scale))

    def test_selection(self, reflections: ReflectionList, point_group: PointGroup,
                       xp: AnyNamespace):
        selected = reflections.iloc[1]
        mapping = MergeData.from_reflections(selected, point_group)
        hkl, reflection_id = mapping.hkl, mapping.reflection_id
        reflection = mapping.reflection_at(1)

        # Symmetry-equivalent observations share one canonical reflection.
        assert xp.all(reflection.index == 1)
        assert xp.all(reflection.reflection_id == 0)
        assert xp.all(reflection.hkl == hkl[1:2])
        assert xp.all(reflection.I_hkl == selected.I_hkl[reflection_id == 1])
        reordered = mapping.reflection_at(xp.asarray([1, 0]))

        # Reordering the selection must not change the underlying mapping.
        assert xp.all(reordered.reflection_id == xp.asarray([0, 1]))
        assert xp.all(reordered.hkl == hkl[xp.asarray([1, 0])])

    def test_dataframe(self, reflections: ReflectionList, xp: AnyNamespace):
        supplied = reflections.replace(hkl=-reflections.hkl)
        frames = xp.asarray([10, 20, 30])
        result = ReflectionList.import_dataframe(supplied.to_dataframe(frames), frames, xp)
        for name in ('index', 'hkl', 'I_hkl', 'sigma_hkl'):
            assert xp.all(getattr(result, name) == getattr(supplied, name))
        # Import preserves representatives rather than imposing a symmetry policy.
        assert xp.all(result.hkl == supplied.hkl)

    def test_empty(self, first: ReflectionList, point_group: PointGroup,
                   xp: AnyNamespace):
        empty = first[:0]
        mapping = MergeData.from_reflections(empty, point_group)
        hkl, reflection_id = mapping.hkl, mapping.reflection_id
        assert hkl.shape == (0, 3)
        assert reflection_id.size == 0
        assert empty.n_patterns == 0
        result = ReflectionList.concat((empty, first, empty))
        assert xp.all(result.index == first.index)

    def test_empty_iterable(self):
        with pytest.raises(ValueError, match='must not be empty'):
            ReflectionList.concat(iter(()))

class TestMergeModel:
    @pytest.fixture(params=[NumPy, JaxNumPy], ids=['numpy', 'jax'])
    def xp(self, request: pytest.FixtureRequest) -> AnyNamespace:
        return request.param

    @pytest.fixture
    def model(self) -> MergeModel:
        return MergeModel()

    @pytest.fixture
    def point_group(self) -> PointGroup:
        return PointGroup("1")

    @pytest.fixture
    def reflection_list(self, xp: AnyNamespace) -> ReflectionList:
        scales = xp.asarray([0.5, 1.0, 2.0, 1.5, 0.8, 1.2])
        intensity = xp.asarray([-10., 20., 30., 40.])
        return ReflectionList(index=xp.repeat(xp.arange(scales.size), intensity.size),
                              hkl=xp.tile(xp.asarray([[1, 0, 0], [2, 0, 0],
                                                     [3, 0, 0], [4, 0, 0]]), (6, 1)),
                              I_hkl=cast(RealArray, (scales[:, None] * intensity).reshape(-1)),
                              sigma_hkl=xp.ones(scales.size * intensity.size))

    @pytest.fixture
    def data(self, reflection_list: ReflectionList, point_group: PointGroup) -> MergeData:
        return MergeData.from_reflections(reflection_list, point_group)

    @pytest.fixture
    def initial(self, data: MergeData) -> MergeState:
        return MergeState.from_data(data)

    def optimise(self, model: MergeModel, data: MergeData, initial: MergeState,
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

    def test_initialisation(self, data: MergeData, xp: AnyNamespace):
        log_scale = xp.log(xp.asarray([0.5, 1.0, 2.0, 1.5, 0.8, 1.2]))
        state = MergeState.from_data(data, log_scale)
        scale_shift = xp.mean(log_scale)
        expected = xp.asarray([-10., 20., 30., 40.]) * xp.exp(scale_shift)
        assert xp.allclose(state.I_hkl, expected)
        assert xp.allclose(state.log_scale, log_scale - scale_shift)

    def test_initialisation_includes_unsupported(self, data: MergeData,
                                                 xp: AnyNamespace):
        first = xp.arange(data.I_hkl.size) == 0
        altered = data.replace(I_hkl=xp.where(first, 1e6, data.I_hkl),
                               sigma_hkl=xp.where(first, xp.inf, data.sigma_hkl))
        state = MergeState.from_data(altered)

        # Initial medians include every fitted intensity; uncertainty affects later weights.
        assert xp.allclose(state.I_hkl[0], -10.0)

    def test_recovery(self, model: MergeModel, data: MergeData, initial: MergeState,
                      xp: AnyNamespace):
        if xp is not JaxNumPy:
            pytest.skip('Gradient optimisation requires JAX arrays')
        state = self.optimise(model, data, initial)
        assert xp.allclose(model.expected(data, state), data.I_hkl, rtol=1e-5)
        assert xp.allclose(xp.mean(state.log_scale), 0.0, atol=1e-6)

    def test_gauge(self, model: MergeModel, data: MergeData,
                   initial: MergeState, xp: AnyNamespace):
        shifted = initial.replace(log_scale=initial.log_scale + 2.0,
                                  I_hkl=initial.I_hkl * xp.exp(xp.asarray(-2.0)))
        assert xp.allclose(model.expected(data, shifted.normalise()),
                           model.expected(data, initial))

    def test_outlier(self, model: MergeModel, data: MergeData,
                     initial: MergeState, xp: AnyNamespace):
        if xp is not JaxNumPy:
            pytest.skip('Gradient optimisation requires JAX arrays')
        corrupt = data.replace(I_hkl=data.I_hkl + xp.where(xp.arange(data.I_hkl.size) == 0,
                                                          200.0, 0.0))
        state = self.optimise(model, corrupt, initial)
        prediction = model.expected(data, state)

        # One bad reflection must not pull the clean observations away from their common fit.
        assert xp.allclose(prediction, data.I_hkl, atol=0.1)
        residuals = xp.abs(model.residuals(corrupt, state))
        assert residuals[0] > 100.0 * xp.max(residuals[1:])

    def test_invalid(self, model: MergeModel, data: MergeData, initial: MergeState,
                     xp: AnyNamespace):
        invalid = data.replace(sigma_hkl=xp.full_like(data.sigma_hkl, xp.inf))
        assert model(invalid, initial) == 0.0

    def test_unsupported(self, model: MergeModel, data: MergeData,
                         initial: MergeState,
                         xp: AnyNamespace):
        if xp is not JaxNumPy:
            pytest.skip('Gradient calculation requires JAX arrays')
        unsupported = (data.index == 0) | (data.reflection_id == 0)
        altered = data.replace(sigma_hkl=xp.where(unsupported, xp.inf, data.sigma_hkl))
        _, gradient = value_and_grad(model, argnums=1)(altered, initial)

        # Parameters supported only by infinite-uncertainty observations have zero gradient.
        assert gradient.log_scale[0] == 0.0
        assert gradient.I_hkl[0] == 0.0
        negative = data.replace(I_hkl=-xp.abs(data.I_hkl))
        scales = negative.fit_scales(xp.ones((data.n_reflections,)),
                                     xp.ones_like(data.I_hkl), xp.ones((data.n_patterns,)))
        assert xp.all(scales < 0.0)

    def test_uncertainty_units(self, model: MergeModel, data: MergeData, initial: MergeState,
                               xp: AnyNamespace):
        factor = 3.0
        converted = data.replace(I_hkl=factor * data.I_hkl,
                                 sigma_hkl=factor * data.sigma_hkl)
        converted_state = initial.replace(I_hkl=factor * initial.I_hkl)

        # Changing intensity units cannot change the dimensionless likelihood.
        assert xp.allclose(model(converted, converted_state), model(data, initial))

    def test_empty(self, model: MergeModel, reflection_list: ReflectionList,
                   point_group: PointGroup, xp: AnyNamespace):
        empty = MergeData.from_reflections(reflection_list[:0], point_group)
        state = MergeState(log_scale=xp.zeros(0), I_hkl=xp.zeros(0))

        # The likelihood of an empty dataset is zero, regardless of the state.
        assert model(empty, state) == 0.0

    def test_jit(self, model: MergeModel, data: MergeData, initial: MergeState,
                 xp: AnyNamespace):
        if xp is not JaxNumPy:
            pytest.skip('JAX compilation requires JAX arrays')
        expected = model(data, initial)
        actual = jit(model)(data, initial)

        # JIT compilation must not change the numerical result.
        assert xp.allclose(actual, expected)

    @pytest.mark.parametrize('nu', [0.0, -1.0, float('inf'), float('nan')])
    def test_invalid_nu(self, nu: float):
        with pytest.raises(ValueError, match='nu must be finite and positive'):
            MergeModel(nu=nu)
