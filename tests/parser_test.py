from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict
import pytest
from cbclib_v2 import Container, FieldLocator
from cbclib_v2._src.array_api import array_namespace
from cbclib_v2._src.run import SwissFELConfig
from cbclib_v2._src.parser import (INIParser, JSONParser, extract_fields, fields, from_container,
                                  from_file, get_type_hints, read_fields)
from cbclib_v2.annotations import AnyNamespace, Array, JaxNumPy, NumPy, NumPyNamespace, RealArray
from cbclib_v2.indexer import (FixedAperture, FixedApertureGeometry, FixedFocus, FixedGeometry,
                               FixedPupilGeometry, FixedRectangle, Focus, Geometry, Rectangle,
                               ResolvedGeometry)
from cbclib_v2.slurm import ScanArgument
from cbclib_v2.scripts import (DetectConfig, MetadataConfig, MetaListConfig, PeakParameters,
                               ScalingParameters, ScanConfig, SetupConfig, StreakFinderConfig,
                               StreakParameters, StructureParameters, SystemConfig)
from cbclib_v2.test_util import check_close

GeometryType = FixedGeometry | FixedPupilGeometry | FixedApertureGeometry | Geometry

@dataclass
class ParserChild(Container):
    first: int
    second: str

@dataclass
class ParserContainer(Container):
    child: ParserChild
    value: float

@dataclass
class ParserValues(Container):
    text: str
    enabled: bool
    count: int
    scale: float
    labels: list[str]
    bounds: tuple[tuple[float, float], ...]
    optional: float | None
    samples: RealArray

class TestFieldLocator:
    def test_getattr(self):
        container = ParserContainer(ParserChild(3, 'value'), 1.5)

        assert FieldLocator('child.second').getattr(container) == 'value'

    def test_invalid_field_name(self):
        with pytest.raises(ValueError, match='Invalid field name'):
            FieldLocator('child..first')

class TestFields:
    def test_nested_container(self):
        container = ParserContainer(ParserChild(3, 'value'), 1.5)

        assert fields(type(container), container.to_dict(), None) == {
            'child': {
                'first': FieldLocator('child.first'),
                'second': FieldLocator('child.second'),
            },
            'value': FieldLocator('value'),
        }

    def test_default_section_preserves_file_layout(self):
        container = ParserContainer(ParserChild(3, 'value'), 1.5)

        assert fields(type(container), container.to_dict(), 'general') == {
            'child': {
                'first': FieldLocator('child.first'),
                'second': FieldLocator('child.second'),
            },
            'general': {'value': FieldLocator('value')},
        }

    def test_nested_type_hints(self):
        assert get_type_hints(ParserContainer) == {
            'child': {'first': int, 'second': str},
            'value': float,
        }

class TestFieldMapping:
    def field_info(self) -> Dict[str, Any]:
        return {
            'nested': {
                'parameters': {
                    'renamed': FieldLocator('child.first'),
                },
            },
            'metadata': {
                'label': FieldLocator('child.second'),
                'value': FieldLocator('value'),
            },
        }

    def test_read_fields_constructs_object_dictionary(self):
        data = {
            'nested': {'parameters': {'renamed': 3}},
            'metadata': {'label': 'value', 'value': 1.5},
        }

        assert read_fields(self.field_info(), data) == {
            'child': {'first': 3, 'second': 'value'},
            'value': 1.5,
        }

    def test_extract_fields_constructs_file_dictionary(self):
        container = ParserContainer(ParserChild(3, 'value'), 1.5)

        assert extract_fields(self.field_info(), container) == {
            'nested': {'parameters': {'renamed': 3}},
            'metadata': {'label': 'value', 'value': 1.5},
        }

class TestParserRoundTrip:
    @pytest.fixture(params=[NumPy, JaxNumPy], ids=['numpy', 'jax'])
    def array_xp(self, request: pytest.FixtureRequest) -> AnyNamespace:
        return request.param

    @pytest.fixture(params=[None, 0.5])
    def optional(self, request: pytest.FixtureRequest) -> float | None:
        return request.param

    @pytest.fixture
    def values(self, array_xp: AnyNamespace, optional: float | None) -> ParserValues:
        return ParserValues(text='Commas, "quotes", [brackets], and\na new line.',
                            enabled=True, count=3, scale=0.125,
                            labels=['first, second', '"third"'],
                            bounds=((0.1, 0.2), (0.3, 0.4)), optional=optional,
                            samples=array_xp.arange(6, dtype=float).reshape((2, 3)))

    @pytest.fixture
    def streak_finder(self) -> StreakFinderConfig:
        structure = StructureParameters(radius=2, connectivity=1)
        return StreakFinderConfig(
            peaks=PeakParameters(npts=4, structure=structure),
            streaks=StreakParameters(
                structure=structure,
                xtol=1.5,
                vmin=3.0,
                min_size=5.0,
                nfa=2,
                keep_best=0.75,
            ),
            scaling=ScalingParameters(
                method='robust-lsq',
                good_fields=(0, 2),
                clip_snr=4.0,
                n_iter=2,
                std_min=0.1,
                n_pixels=100,
            ),
            center=(12.5, 14.5),
            std_min=0.2,
        )

    @pytest.fixture
    def scan(self) -> ScanConfig:
        return ScanConfig(
            image_kind='stacked',
            data=SwissFELConfig(
                data_dir='/data/run_{0:04d}',
                hdf5_protocol='protocol.json',
                file_pattern=r'run_\d{6}\.h5',
                geometry_file='detector.geom',
            ),
            setup=SetupConfig(
                setup_file='setup.json',
                unit_file='unit.json',
                reflections_dir='reflections',
                xtals_dir='xtals',
                solutions_dir='solutions'
            ),
            detect=DetectConfig(
                hit_threshold=10,
                streaks_dir='streaks',
                regions_dir='regions',
            ),
            metadata=MetadataConfig(n_frames=50, output_dir='metadata'),
            metalist=MetaListConfig(n_frames=20, spacing=5, output_dir='metalist'),
            system=SystemConfig(platform='cpu', num_threads=2),
        )

    def test_streak_finder_config(self, tmp_path: Path,
                                  streak_finder: StreakFinderConfig):
        path = tmp_path / 'streak_finder.json'

        streak_finder.write(str(path))
        result = StreakFinderConfig.read(str(path))

        assert result == streak_finder

    def test_scan_config(self, tmp_path: Path, scan: ScanConfig):
        path = tmp_path / 'scan.json'

        scan.write(str(path))
        result = ScanConfig.read(str(path))

        assert 'scan_num' not in json.loads(path.read_text())['parameters']
        assert result == scan

    def test_values(self, path: Path, values: ParserValues, array_xp: AnyNamespace) -> None:
        from_container(str(path), values, default='parameters').write(str(path), values)
        parser = from_file(str(path), ParserValues, default='parameters')
        assert isinstance(parser, (INIParser, JSONParser))
        restored = ParserValues.from_dict(**parser.read(str(path), array_xp))

        # JSON preserves scalar values and text; annotations restore tuples and array backend.
        assert restored.text == values.text
        assert restored.enabled is values.enabled
        assert restored.count == values.count
        assert restored.scale == values.scale
        assert restored.labels == values.labels
        assert restored.bounds == values.bounds
        assert restored.optional == values.optional
        assert restored.samples.shape == values.samples.shape
        assert array_namespace(restored.samples) is array_xp
        check_close(restored.samples, values.samples)

    @pytest.fixture
    def xp(self) -> NumPyNamespace:
        return NumPy

    @pytest.fixture
    def foc_pos(self) -> tuple[float, float, float]:
        return (0.1, 0.2, -0.4)

    @pytest.fixture
    def pupil_roi(self) -> tuple[float, float, float, float]:
        return (0.15, 0.17, 0.12, 0.16)

    @pytest.fixture
    def focus(self, foc_pos: tuple[float, float, float], xp: NumPyNamespace) -> Focus:
        return Focus(xp.asarray(foc_pos))

    @pytest.fixture
    def fixed_pupil(self, pupil_roi: tuple[float, float, float, float]
                    ) -> FixedRectangle:
        return FixedRectangle(pupil_roi)

    @pytest.fixture
    def pupil(self, fixed_pupil: FixedRectangle, xp: NumPyNamespace) -> Rectangle:
        return Rectangle(xp.asarray(fixed_pupil.roi))

    @pytest.fixture
    def aperture(self, pupil: Rectangle, xp: NumPyNamespace) -> FixedAperture:
        return FixedAperture.from_parameters(pupil.roi, xp)

    @pytest.fixture(params=['fixed', 'fixed-pupil', 'fixed-aperture', 'dynamic'])
    def geometry(self, request: pytest.FixtureRequest, foc_pos: tuple[float, float, float],
                 focus: Focus, fixed_pupil: FixedRectangle, pupil: Rectangle,
                 aperture: FixedAperture, xp: NumPyNamespace) -> GeometryType:
        if request.param == 'fixed':
            return FixedGeometry(FixedFocus(foc_pos), fixed_pupil, (0.01,))
        defocus = xp.asarray([0.01])
        if request.param == 'fixed-pupil':
            return FixedPupilGeometry(focus, fixed_pupil, defocus)
        if request.param == 'fixed-aperture':
            return FixedApertureGeometry(focus, aperture, defocus)
        return Geometry(focus, pupil, defocus)

    @pytest.fixture(params=['json', 'ini'])
    def path(self, request: pytest.FixtureRequest, tmp_path: Path) -> Path:
        return tmp_path / f'geometry.{request.param}'

    def test_file_round_trip(self, geometry: GeometryType, path: Path) -> None:
        array_xp = JaxNumPy
        geometry.write(str(path))
        if isinstance(geometry, FixedGeometry):
            restored = type(geometry).read(str(path))
        else:
            restored = type(geometry).read(str(path), array_xp)

        # Serialisation restores native static tuples and dynamic namespace arrays.
        if isinstance(geometry.focus, FixedFocus):
            assert restored.focus.pos == geometry.focus.pos
        else:
            assert isinstance(restored.focus.pos, Array)
            assert array_namespace(restored.focus.pos) is array_xp
        if isinstance(geometry.pupil, FixedRectangle):
            assert restored.pupil == geometry.pupil
        else:
            assert isinstance(restored.pupil.roi, Array)
            assert array_namespace(restored.pupil.roi) is array_xp
        if isinstance(geometry, FixedGeometry):
            assert isinstance(restored, FixedGeometry)
            assert restored.defocus == geometry.defocus
        expected = geometry.resolve(array_xp)
        actual = restored.resolve(array_xp)

        # File round trips preserve the pupil ROI, focus, and sample-plane offset.
        assert actual.foc_pos.shape == expected.foc_pos.shape
        assert actual.pupil_roi.shape == expected.pupil_roi.shape
        check_close(actual.foc_pos, expected.foc_pos)
        check_close(actual.pupil_roi, expected.pupil_roi)
        if isinstance(expected, ResolvedGeometry):
            assert isinstance(actual, ResolvedGeometry)
            assert array_namespace(actual.defocus) is array_xp
            assert actual.defocus.shape == expected.defocus.shape
            check_close(actual.defocus, expected.defocus)

class TestScanArgument:
    @pytest.fixture(params=[[], [7, 7], [-1]])
    def invalid_scan_num(self, request: pytest.FixtureRequest) -> list[int]:
        return request.param

    def test_invalid_numbers(self, invalid_scan_num: list[int]) -> None:
        # Scan selections are non-empty collections of unique, non-negative numbers.
        with pytest.raises(ValueError, match='scan_num'):
            ScanArgument(invalid_scan_num).enumerate()
