from math import prod
from typing import Set, Tuple
import pytest
from cbclib_v2 import default_rng, set_at
from cbclib_v2.annotations import (BoolArray, CPArray, CuPy, CuPyNamespace, Generator, IntArray,
                                   NDArray, NumPy, NumPyNamespace, RealArray, Shape)
from cbclib_v2.label import (LabelResult, Structure, binary_dilation, center_of_mass, covariance_matrix,
                             label, maximum_position)

TestNamespace = NumPyNamespace | CuPyNamespace
TestGenerator = Generator[NDArray] | Generator[CPArray]

class TestBinaryDilation:
    @pytest.fixture(params=['cpu'])
    def platform(self, request: pytest.FixtureRequest) -> str:
        return request.param

    @pytest.fixture
    def xp(self, platform: str) -> TestNamespace:
        if platform == 'cpu':
            return NumPy
        raise ValueError(f"Unknown platform: {platform}")

    @pytest.fixture
    def structure(self) -> Structure:
        return Structure([1, 1], 1)

    @pytest.fixture
    def mask(self, xp: TestNamespace) -> BoolArray:
        mask = xp.zeros((5, 5), dtype=bool)
        mask[2, 2] = True
        return mask

    def test_iterations(self, mask: BoolArray, structure: Structure, xp: TestNamespace):
        expected = xp.zeros_like(mask)
        expected[2, 2] = True
        expected[1, 2] = True
        expected[2, 1] = True
        expected[2, 3] = True
        expected[3, 2] = True

        result = binary_dilation(mask, structure=structure)
        assert xp.all(result == expected)

    def test_mask(self, mask: BoolArray, structure: Structure, xp: TestNamespace):
        limited = xp.zeros_like(mask)
        limited[2, 3] = True

        expected = xp.zeros_like(mask)
        expected[2, 2] = True
        expected[2, 3] = True

        result = binary_dilation(mask, structure=structure, iterations=2, mask=limited)
        assert xp.all(result == expected)

    def test_validation(self, structure: Structure, xp: TestNamespace):
        mask = xp.zeros((3, 3), dtype=bool)

        assert xp.all(binary_dilation(mask, structure, iterations=0) == mask)
        with pytest.raises(ValueError, match="iterations must be non-negative"):
            binary_dilation(mask, structure, iterations=-1)
        with pytest.raises(ValueError, match="does not match structure rank"):
            binary_dilation(mask, Structure([1, 1, 1], 1))

@pytest.mark.parametrize('shape,structure', [((50, 50), Structure([2, 2], 2)),
                                             ((10, 10, 10), Structure([1, 1, 1], 1))])
class TestLabel():
    def center_of_mass(self, coords: IntArray, val: RealArray, xp: TestNamespace) -> RealArray:
        return xp.sum(coords * val[..., None], axis=0) / xp.sum(val)

    def covariance_matrix(self, coords: IntArray, val: RealArray, xp: TestNamespace) -> RealArray:
        ctr = self.center_of_mass(coords, val, xp)
        return xp.sum((coords[..., None, :] * coords[..., None] - ctr[None, :] * ctr[:, None]) * \
                      val[..., None, None], axis=0) / xp.sum(val)

    def find_pixel_set(self, mask: BoolArray, seed: Tuple[int, ...], structure: Structure
                       ) -> Set[Tuple[int, ...]]:
        pixels: Set[Tuple[int, ...]] = set()
        new_pixels: Set[Tuple[int, ...]] = {seed}
        while new_pixels:
            pixels |= new_pixels
            new_pixels = set()
            for pix in pixels:
                for shift in structure:
                    new = tuple(int(x + dx) for x, dx in zip(pix, shift))
                    is_inbound = all(0 <= x < length for x, length in zip(new, mask.shape))
                    if is_inbound and mask[new] and new not in pixels:
                        new_pixels.add(new)
        return pixels

    @pytest.fixture(params=['cpu', 'gpu'])
    def platform(self, request: pytest.FixtureRequest) -> str:
        return request.param

    @pytest.fixture
    def xp(self, platform: str) -> TestNamespace:
        if platform == 'cpu':
            return NumPy
        if platform == 'gpu':
            if CuPy is None:
                pytest.skip("CuPy is not available")
            return CuPy
        raise ValueError(f"Unknown platform: {platform}")

    @pytest.fixture
    def rng(self, xp: TestNamespace) -> TestGenerator:
        return default_rng(42, xp)

    @pytest.fixture(params=[30])
    def n_good(self, request: pytest.FixtureRequest) -> int:
        return request.param

    @pytest.fixture
    def seeds(self, rng: TestGenerator, shape: Shape, n_good: int) -> IntArray:
        return rng.choice(prod(shape), size=n_good, replace=False)

    @pytest.fixture
    def mask(self, seeds: IntArray, shape: Shape, structure: Structure, xp: TestNamespace
             ) -> BoolArray:
        mask = xp.zeros(prod(shape), dtype=bool)
        mask = set_at(mask, seeds, True).reshape(shape)
        mask = binary_dilation(mask, structure=structure)
        return mask

    @pytest.fixture
    def data(self, rng: TestGenerator, shape: Shape) -> RealArray:
        return rng.random(size=shape)

    @pytest.fixture
    def labeled(self, mask: BoolArray, structure: Structure) -> LabelResult:
        return label(mask, structure=structure)

    def test_dilation(self, seeds: IntArray, shape: Shape, structure: Structure,
                      mask: BoolArray, xp: TestNamespace):
        shifts = xp.asarray(list(structure))
        seed_indices = xp.stack(xp.unravel_index(seeds, shape), axis=-1)
        dilated = seed_indices[:, None, :] + shifts[None, :, :]
        dilated = dilated.reshape(-1, len(shape))
        inbound = xp.all((dilated >= 0) &
                            (dilated < xp.asarray(shape)[None, :]), axis=-1)
        dilated = dilated[inbound]
        assert xp.all(mask[tuple(dilated.T)])

    def test_label(self, seeds: IntArray, shape: Shape, structure: Structure,
                   mask: BoolArray, labeled: LabelResult, xp: TestNamespace):
        labels = labeled.labels
        assert labels.dtype == xp.int32
        for seed in seeds:
            seed_index = tuple(int(x) for x in xp.unravel_index(seed, mask.shape))
            pixels = self.find_pixel_set(mask, seed_index, structure)
            assert all(labels[px] == labels[xp.unravel_index(seed, shape)]
                       for px in pixels)

    def test_label_moments(self, labeled: LabelResult, data: RealArray,
                           xp: TestNamespace):
        centers = center_of_mass(labeled, data)
        covmats = covariance_matrix(labeled, data)
        labels, index = labeled.labels, labeled.index
        for i, idx in enumerate(index):
            indices = xp.where(labels == idx)
            vals = data[indices]
            coords = xp.stack(indices, axis=-1)
            expected_center = self.center_of_mass(coords, vals, xp)
            expected_covmat = self.covariance_matrix(coords, vals, xp)
            assert xp.allclose(centers[i], expected_center)
            assert xp.allclose(covmats[i], expected_covmat)

class TestLabelEdgeCases:
    @pytest.fixture
    def structure(self) -> Structure:
        return Structure([1, 1], 1)

    def test_npts_filter(self, structure: Structure):
        mask = NumPy.zeros((5, 5), dtype=bool)
        mask[0, 0] = True
        mask[2, 2] = True
        mask[2, 3] = True
        mask[3, 2] = True

        labeled = label(mask, structure=structure, npts=2)

        assert labeled.labels.dtype == NumPy.int32
        assert labeled.index.dtype == NumPy.int32
        assert NumPy.all(labeled.labels[mask] == NumPy.array([0, 1, 1, 1], dtype=NumPy.int32))
        assert NumPy.all(labeled.index == NumPy.array([1], dtype=NumPy.int32))

    def test_non_contiguous_input(self, structure: Structure):
        base = NumPy.zeros((6, 6), dtype=bool)
        view = base[::2, ::2]
        view[1, 1] = True
        view[1, 2] = True

        labeled = label(view, structure=structure)

        assert labeled.labels.shape == view.shape
        assert labeled.labels.dtype == NumPy.int32
        assert labeled.labels[1, 1] == 1
        assert labeled.labels[1, 2] == 1
        assert NumPy.all(labeled.index == NumPy.array([1], dtype=NumPy.int32))

class TestMaximumPosition:
    @pytest.fixture(params=['cpu', 'gpu'])
    def platform(self, request: pytest.FixtureRequest) -> str:
        return request.param

    @pytest.fixture
    def xp(self, platform: str) -> TestNamespace:
        if platform == 'cpu':
            return NumPy
        if platform == 'gpu':
            if CuPy is None:
                pytest.skip("CuPy is not available")
            return CuPy
        raise ValueError(f"Unknown platform: {platform}")

    def label_result(self, labels: IntArray, index: IntArray) -> LabelResult:
        return LabelResult(labels=labels, index=index)

    @pytest.mark.parametrize('dtype', ['float32', 'float64', 'int32', 'int64'])
    def test_2d_first_maximum(self, xp: TestNamespace, dtype: str):
        labels = xp.asarray([[1, 1, 0], [2, 2, 2]], dtype=xp.int32)
        index = xp.asarray([1, 2], dtype=int)
        data = xp.asarray([[5, 7, 9], [4, 4, 3]], dtype=getattr(xp, dtype))
        expected = xp.asarray([[0, 1], [1, 0]], dtype=xp.int32)

        result = maximum_position(self.label_result(labels, index), data)

        assert result.shape == expected.shape
        assert result.dtype == xp.int32
        assert xp.all(result == expected)

    def test_3d_first_maximum(self, xp: TestNamespace):
        labels = xp.zeros((2, 2, 3), dtype=xp.int32)
        labels[0, 0, 1] = 1
        labels[1, 0, 2] = 1
        labels[1, 1, 0] = 2
        labels[1, 1, 2] = 2

        data = xp.zeros(labels.shape, dtype=float)
        data[0, 0, 1] = 3.0
        data[1, 0, 2] = 7.0
        data[1, 1, 0] = 5.0
        data[1, 1, 2] = 5.0

        index = xp.asarray([1, 2], dtype=int)
        expected = xp.asarray([[1, 0, 2], [1, 1, 0]], dtype=xp.int32)

        result = maximum_position(self.label_result(labels, index), data)

        assert xp.all(result == expected)

    def test_missing_label_returns_first_position(self, xp: TestNamespace):
        labels = xp.zeros((3, 4), dtype=xp.int32)
        index = xp.asarray([5], dtype=int)
        data = xp.arange(12).reshape(3, 4)

        result = maximum_position(self.label_result(labels, index), data)

        assert xp.all(result == xp.asarray([[0, 0]], dtype=xp.int32))

    def test_unindexed_label_is_ignored(self, xp: TestNamespace):
        labels = xp.asarray([[1, 99], [1, 99]], dtype=xp.int32)
        index = xp.asarray([1], dtype=int)
        data = xp.asarray([[1, 100], [5, 200]], dtype=xp.float32)

        result = maximum_position(self.label_result(labels, index), data)

        assert xp.all(result == xp.asarray([[1, 0]], dtype=xp.int32))

    def test_empty_index(self, xp: TestNamespace):
        labels = xp.zeros((3, 4), dtype=xp.int32)
        index = xp.asarray([], dtype=int)
        data = xp.arange(12).reshape(3, 4)

        result = maximum_position(self.label_result(labels, index), data)

        assert result.shape == (0, 2)
        assert result.dtype == xp.int32

    def test_shape_validation(self, xp: TestNamespace):
        labels = xp.zeros((3, 4), dtype=xp.int32)
        index = xp.asarray([1], dtype=int)
        data = xp.zeros((3, 5), dtype=float)

        with pytest.raises(ValueError, match="same shape"):
            maximum_position(self.label_result(labels, index), data)
