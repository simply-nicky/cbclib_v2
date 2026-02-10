from math import prod
from typing import Set, Tuple
import pytest
from cbclib_v2 import device, set_at
from cbclib_v2.annotations import BoolArray, Device, IntArray, RealArray, Shape
from cbclib_v2.label import (CPLabelResult, NPLabelResult, LabelResult, Structure, binary_dilation,
                             center_of_mass, covariance_matrix, label)
from cbclib_v2.test_util import TestGenerator, TestNamespace

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

    def labels_and_index(self, labeled: LabelResult, xp: TestNamespace
                         ) -> Tuple[IntArray, IntArray]:
        if isinstance(labeled, CPLabelResult):
            return labeled.labels, labeled.index
        if isinstance(labeled, NPLabelResult):
            index = xp.arange(1, len(labeled.regions) + 1)
            return labeled.to_mask(index), index
        raise TypeError("Unknown LabelResult type")

    @pytest.fixture
    def xp(self, test_xp: TestNamespace) -> TestNamespace:
        return test_xp

    @pytest.fixture
    def rng(self, test_rng: TestGenerator) -> TestGenerator:
        return test_rng

    @pytest.fixture(params=[30])
    def n_good(self, request: pytest.FixtureRequest) -> int:
        return request.param

    @pytest.fixture
    def seeds(self, rng: TestGenerator, shape: Shape, n_good: int) -> IntArray:
        return rng.choice(prod(shape), size=n_good, replace=False)

    @pytest.fixture
    def mask(self, seeds: IntArray, shape: Shape, structure: Structure, test_device: Device,
             xp: TestNamespace) -> BoolArray:
        mask = xp.zeros(prod(shape), dtype=bool)
        mask = set_at(mask, seeds, True).reshape(shape)
        with device.context(test_device):
            mask = binary_dilation(mask, structure=structure)
        return mask

    @pytest.fixture
    def data(self, rng: TestGenerator, shape: Shape) -> RealArray:
        return rng.random(size=shape)

    @pytest.fixture
    def labeled(self, mask: BoolArray, structure: Structure, test_device: Device) -> LabelResult:
        with device.context(test_device):
            labeled = label(mask, structure=structure)
        return labeled

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
        labels = self.labels_and_index(labeled, xp)[0]
        for seed in seeds:
            seed_index = tuple(int(x) for x in xp.unravel_index(seed, mask.shape))
            pixels = self.find_pixel_set(mask, seed_index, structure)
            assert all(labels[px] == labels[xp.unravel_index(seed, shape)]
                       for px in pixels)

    def test_label_moments(self, labeled: LabelResult, data: RealArray, test_device: Device,
                           xp: TestNamespace):
        with device.context(test_device):
            centers = center_of_mass(labeled, data)
            covmats = covariance_matrix(labeled, data)
        labels, index = self.labels_and_index(labeled, xp)
        for i, idx in enumerate(index):
            indices = xp.where(labels == idx)
            vals = data[indices]
            coords = xp.stack(indices, axis=-1)
            expected_center = self.center_of_mass(coords, vals, xp)
            expected_covmat = self.covariance_matrix(coords, vals, xp)
            assert xp.allclose(centers[i], expected_center)
            assert xp.allclose(covmats[i], expected_covmat)
