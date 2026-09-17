from typing import cast
import pytest
from cbclib_v2.annotations import AnyNamespace, JaxNumPy, NumPy, RealArray
from cbclib_v2._src.array_api import add_at
from cbclib_v2.scaler import MergeModel, MergeState, ReflectionList, Reflections

class TestReflectionList:
    @pytest.fixture(params=[NumPy, JaxNumPy], ids=['numpy', 'jax'])
    def xp(self, request: pytest.FixtureRequest) -> AnyNamespace:
        return request.param

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
        mapping = reflections.reflections()
        hkl, reflection_id = mapping.hkl, mapping.reflection_id
        assert xp.all(hkl[reflection_id] == reflections.hkl)
        assert hkl.shape == (2, 3)
        assert xp.all(reflection_id[:2] == reflection_id[2:4])
        assert reflections.__array_namespace__() is xp

    def test_reference_update(self, reflections: ReflectionList, xp: AnyNamespace):
        mapping = reflections.reflections()
        hkl, reflection_id = mapping.hkl, mapping.reflection_id
        scales = xp.asarray([1., 2., 3.])[reflections.index]
        weights = 1.0 / reflections.sigma_hkl**2
        numerator = add_at(xp.zeros(hkl.shape[0]), reflection_id,
                           weights * scales * reflections.I_hkl)
        denominator = add_at(xp.zeros(hkl.shape[0]), reflection_id, weights * scales**2)
        reference = numerator / denominator
        assert xp.allclose(mapping.estimator(weights).fit(
            predictor=scales, response=reflections.I_hkl, previous=xp.zeros(hkl.shape[0])),
            reference)
        # Shared intensities and illumination scales explain every observation exactly.
        assert xp.allclose(reference[reflection_id] * scales, reflections.I_hkl)
        numerator = add_at(xp.zeros(reflections.n_patterns), reflections.index,
                           weights * reference[reflection_id] * reflections.I_hkl)
        denominator = add_at(xp.zeros(reflections.n_patterns), reflections.index,
                             weights * reference[reflection_id]**2)
        assert xp.allclose(numerator / denominator, xp.asarray([1., 2., 3.]))
        assert xp.allclose(reflections.fit_scales(
            reference[reflection_id], weights, xp.ones(reflections.n_patterns)),
            numerator / denominator)

    def test_selection(self, reflections: ReflectionList, xp: AnyNamespace):
        selected = reflections.iloc[1]
        mapping = selected.reflections()
        hkl, reflection_id = mapping.hkl, mapping.reflection_id
        assert xp.all(selected.index == 1)
        assert xp.all(selected.I_hkl == xp.asarray([4., 6.]))
        assert xp.all(hkl[reflection_id] == selected.hkl)

    def test_dataframe(self, reflections: ReflectionList, xp: AnyNamespace):
        frames = xp.asarray([10, 20, 30])
        result = ReflectionList.import_dataframe(reflections.to_dataframe(frames), frames, xp)
        for name in ('index', 'hkl', 'I_hkl', 'sigma_hkl'):
            assert xp.all(getattr(result, name) == getattr(reflections, name))

    def test_import_preserves_hkl(self, first: ReflectionList, xp: AnyNamespace):
        supplied = first.replace(hkl=-first.hkl)
        frames = xp.asarray([10])
        result = ReflectionList.import_dataframe(supplied.to_dataframe(frames), frames, xp)
        # Import preserves representatives rather than imposing a symmetry policy.
        assert xp.all(result.hkl == supplied.hkl)

    def test_empty(self, first: ReflectionList, xp: AnyNamespace):
        empty = first[:0]
        mapping = empty.reflections()
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
    def data(self, xp: AnyNamespace) -> ReflectionList:
        scales = xp.asarray([0.5, 1.0, 2.0, 1.5, 0.8, 1.2])
        intensity = xp.asarray([10., 20., 30., 40.])
        return ReflectionList(index=xp.repeat(xp.arange(scales.size), intensity.size),
                              hkl=xp.tile(xp.asarray([[1, 0, 0], [2, 0, 0],
                                                     [3, 0, 0], [4, 0, 0]]), (6, 1)),
                              I_hkl=cast(RealArray, (scales[:, None] * intensity).reshape(-1)),
                              sigma_hkl=xp.ones(scales.size * intensity.size))

    @pytest.fixture
    def reflections(self, data: ReflectionList) -> Reflections:
        return data.reflections()

    @pytest.fixture
    def initial(self, data: ReflectionList, reflections: Reflections,
                xp: AnyNamespace) -> MergeState:
        intensity = reflections.estimator(xp.ones_like(data.I_hkl)).fit(
            predictor=xp.ones_like(data.I_hkl), response=data.I_hkl,
            previous=xp.zeros(reflections.n_reflections))
        return MergeState(log_scale=xp.zeros(data.n_patterns), I_hkl=intensity)

    def test_initialisation(self, data: ReflectionList, reflections: Reflections,
                            xp: AnyNamespace):
        log_scale = xp.log(xp.asarray([0.5, 1.0, 2.0, 1.5, 0.8, 1.2]))
        state = MergeState.from_data(data, reflections, log_scale)
        scale_shift = xp.mean(log_scale)
        expected = xp.asarray([10., 20., 30., 40.]) * xp.exp(scale_shift)
        assert xp.allclose(state.I_hkl, expected)
        assert xp.allclose(state.log_scale, log_scale - scale_shift)

    def test_initialisation_includes_unsupported(self, data: ReflectionList,
                                                 reflections: Reflections,
                                                 xp: AnyNamespace):
        first = xp.arange(data.I_hkl.size) == 0
        altered = data.replace(I_hkl=xp.where(first, 1e6, data.I_hkl),
                               sigma_hkl=xp.where(first, xp.inf, data.sigma_hkl))
        state = MergeState.from_data(altered, reflections)
        # Initial medians include every fitted intensity; uncertainty affects later weights.
        assert xp.allclose(state.I_hkl[0], 15.0)

    def test_recovery(self, model: MergeModel, data: ReflectionList,
                      reflections: Reflections, initial: MergeState, xp: AnyNamespace):
        state = initial
        for _ in range(40):
            state = model.step(data, reflections, state)
        assert xp.allclose(model.expected(data, reflections, state), data.I_hkl, rtol=1e-5)
        assert xp.allclose(xp.mean(state.log_scale), 0.0, atol=1e-6)

    def test_gauge(self, model: MergeModel, data: ReflectionList,
                   reflections: Reflections, initial: MergeState, xp: AnyNamespace):
        shifted = initial.replace(log_scale=initial.log_scale + 2.0,
                                  I_hkl=initial.I_hkl * xp.exp(xp.asarray(-2.0)))
        assert xp.allclose(model.expected(data, reflections, shifted.normalise()),
                           model.expected(data, reflections, initial))

    def test_outlier(self, model: MergeModel, data: ReflectionList,
                     reflections: Reflections, initial: MergeState, xp: AnyNamespace):
        corrupt = data.replace(I_hkl=data.I_hkl + xp.where(xp.arange(data.I_hkl.size) == 0,
                                                          200.0, 0.0))
        state = initial
        for _ in range(60):
            state = model.step(corrupt, reflections, state)
        prediction = model.expected(data, reflections, state)
        # One bad reflection must not pull the clean observations away from their common fit.
        assert xp.allclose(prediction, data.I_hkl, atol=0.1)
        weights = model.weights(corrupt, reflections, state)
        assert weights[0] < 0.001 * xp.min(weights[1:])

    def test_invalid(self, model: MergeModel, data: ReflectionList,
                     reflections: Reflections, initial: MergeState, xp: AnyNamespace):
        invalid = data.replace(sigma_hkl=xp.full_like(data.sigma_hkl, xp.inf))
        assert xp.all(model.weights(invalid, reflections, initial) == 0.0)
        assert model.loss(invalid, reflections, initial) == 0.0
        updated = model.step(invalid, reflections, initial)
        assert xp.allclose(updated.I_hkl, initial.I_hkl)
        assert xp.allclose(updated.log_scale, initial.log_scale)

    def test_unsupported(self, data: ReflectionList, reflections: Reflections,
                         initial: MergeState, xp: AnyNamespace):
        weights = xp.where((data.index == 0) | (reflections.reflection_id == 0), 0.0, 1.0)
        merged = reflections.estimator(weights).fit(
            predictor=initial.scale_at(data), response=data.I_hkl, previous=initial.I_hkl)
        scales = data.fit_scales(initial.intensity_at(reflections), weights,
                                 xp.exp(initial.log_scale))
        # Zero information must preserve the unsupported parameter, not replace it with zero.
        assert merged[0] == initial.I_hkl[0]
        assert scales[0] == xp.exp(initial.log_scale[0])

    def test_uncertainty_units(self, model: MergeModel, data: ReflectionList,
                               reflections: Reflections, initial: MergeState,
                               xp: AnyNamespace):
        factor = 3.0
        converted = data.replace(I_hkl=factor * data.I_hkl,
                                 sigma_hkl=factor * data.sigma_hkl)
        converted_state = initial.replace(I_hkl=factor * initial.I_hkl)
        original = model.step(data, reflections, initial)
        updated = model.step(converted, reflections, converted_state)
        # Changing intensity units cannot change inferred illumination scales or the loss.
        assert xp.allclose(updated.log_scale, original.log_scale, atol=1e-6)
        assert xp.allclose(updated.I_hkl, factor * original.I_hkl)
        assert xp.allclose(model.loss(converted, reflections, converted_state),
                           model.loss(data, reflections, initial))

    def test_empty(self, model: MergeModel, data: ReflectionList, xp: AnyNamespace):
        empty = data[:0]
        reflections = empty.reflections()
        state = MergeState(log_scale=xp.zeros(0), I_hkl=xp.zeros(0))
        updated = model.step(empty, reflections, state)
        assert updated.I_hkl.size == updated.log_scale.size == 0
        assert model.loss(empty, reflections, updated) == 0.0

    def test_jit(self, model: MergeModel, data: ReflectionList, reflections: Reflections,
                 initial: MergeState, xp: AnyNamespace):
        if xp is not JaxNumPy:
            pytest.skip('JAX compilation requires JAX arrays')
        from jax import jit
        expected = model.step(data, reflections, initial)
        actual = jit(model.step)(data, reflections, initial)
        assert xp.allclose(actual.I_hkl, expected.I_hkl)
        assert xp.allclose(actual.log_scale, expected.log_scale, atol=1e-6)

    @pytest.mark.parametrize('nu', [0.0, -1.0, float('inf'), float('nan')])
    def test_invalid_nu(self, nu: float):
        with pytest.raises(ValueError, match='nu must be finite and positive'):
            MergeModel(nu=nu)
