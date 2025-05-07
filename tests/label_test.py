from typing import Set, Tuple
import numpy as np
import pytest
from cbclib_v2.annotations import NDBoolArray, Shape
from cbclib_v2.label import Regions2D, Regions3D, Structure2D, Structure3D, label

Regions = Regions2D | Regions3D
Structure = Structure2D | Structure3D

@pytest.mark.parametrize('shape,structure', [((10, 10), Structure2D(1, 1)),
                                             ((10, 10, 10), Structure3D(1, 1))])
class TestLabel():
    def find_pixel_set(self, mask: NDBoolArray, seed: Tuple[int, ...], structure: Structure
                    ) -> Set[Tuple[int, ...]]:
        pixels: Set[Tuple[int, ...]] = set()
        new_pixels: Set[Tuple[int, ...]] = {tuple(int(x) for x in seed)}
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

    def check_regions(self, regions: Regions, seed: Tuple[int, ...], pixels: Set[Tuple[int, ...]]
                      ) -> bool:
        for region in regions:
            if seed[::-1] in region:
                return set(tuple(x for x in pix[::-1]) for pix in region) == pixels
        return False

    @pytest.fixture
    def mask(self, rng: np.random.Generator, shape: Shape) -> NDBoolArray:
        return rng.random(shape) > 0.5

    @pytest.fixture
    def regions(self, mask: NDBoolArray, structure: Structure) -> Regions:
        regions = label(mask, structure)
        if isinstance(regions, list):
            return regions[0]
        return regions

    @pytest.fixture
    def seed(self, rng: np.random.Generator, mask: NDBoolArray) -> Tuple[int, ...]:
        indices = np.where(mask.ravel())[0]
        seed = np.unravel_index(rng.choice(indices), mask.shape)
        return tuple(int(x) for x in seed)

    def test_pixel_set(self, regions: Regions, mask: NDBoolArray, seed: Tuple[int, ...],
                       structure: Structure):
        pixels = self.find_pixel_set(mask, seed, structure)
        assert self.check_regions(regions, seed, pixels)

    def test_regions(self, regions: Regions, mask: NDBoolArray):
        indices = np.where(mask.ravel())[0]
        flags = []
        for index in indices:
            flags.append(any(np.unravel_index(index, mask.shape)[::-1] in region
                             for region in regions))
        assert all(flags)
