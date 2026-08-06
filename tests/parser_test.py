from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict
import pytest
from cbclib_v2 import Container, FieldLocator
from cbclib_v2._src.run import SwissFELConfig
from cbclib_v2._src.parser import extract_fields, fields, get_type_hints, read_fields
from cbclib_v2.annotations import NumPy
from cbclib_v2.indexer import FixedGeometry
from cbclib_v2.scripts import (DetectConfig, MetadataConfig, MetaListConfig, PeakParameters,
                               ScalingParameters, ScanConfig, SetupConfig, StreakFinderConfig,
                               StreakParameters, StructureParameters, SystemConfig)

@dataclass
class ParserChild(Container):
    first: int
    second: str

@dataclass
class ParserContainer(Container):
    child: ParserChild
    value: float

class TestFieldLocator:
    def test_getattr(self) -> None:
        container = ParserContainer(ParserChild(3, 'value'), 1.5)

        assert FieldLocator('child.second').getattr(container) == 'value'

    def test_invalid_field_name(self) -> None:
        with pytest.raises(ValueError, match='Invalid field name'):
            FieldLocator('child..first')

class TestFields:
    def test_nested_container(self) -> None:
        container = ParserContainer(ParserChild(3, 'value'), 1.5)

        assert fields(type(container), container.to_dict(), None) == {
            'child': {
                'first': FieldLocator('child.first'),
                'second': FieldLocator('child.second'),
            },
            'value': FieldLocator('value'),
        }

    def test_default_section_preserves_file_layout(self) -> None:
        container = ParserContainer(ParserChild(3, 'value'), 1.5)

        assert fields(type(container), container.to_dict(), 'general') == {
            'child': {
                'first': FieldLocator('child.first'),
                'second': FieldLocator('child.second'),
            },
            'general': {'value': FieldLocator('value')},
        }

    def test_nested_type_hints(self) -> None:
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

    def test_read_fields_constructs_object_dictionary(self) -> None:
        data = {
            'nested': {'parameters': {'renamed': 3}},
            'metadata': {'label': 'value', 'value': 1.5},
        }

        assert read_fields(self.field_info(), data) == {
            'child': {'first': 3, 'second': 'value'},
            'value': 1.5,
        }

    def test_extract_fields_constructs_file_dictionary(self) -> None:
        container = ParserContainer(ParserChild(3, 'value'), 1.5)

        assert extract_fields(self.field_info(), container) == {
            'nested': {'parameters': {'renamed': 3}},
            'metadata': {'label': 'value', 'value': 1.5},
        }

class TestParserRoundTrip:
    @pytest.fixture
    def geometry(self) -> FixedGeometry:
        return FixedGeometry(foc_pos=(0.1, -0.2, 0.3), pupil_roi=(-4.0, 5.0, -6.0, 7.0),
                             defocus=(-0.1676,))

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
            scan_num=17,
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
                xtals_dir='xtals',
                solutions_dir='solutions',
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

    @pytest.mark.parametrize('extension', ['ini', 'json'])
    def test_fixed_geometry(self, tmp_path: Path, geometry: FixedGeometry,
                            extension: str) -> None:
        path = tmp_path / f'geometry.{extension}'

        geometry.write(str(path))
        result = FixedGeometry.read(str(path), NumPy)

        assert result == geometry

    def test_streak_finder_config(self, tmp_path: Path,
                                  streak_finder: StreakFinderConfig) -> None:
        path = tmp_path / 'streak_finder.json'

        streak_finder.write(str(path))
        result = StreakFinderConfig.read(str(path))

        expected = json.loads(json.dumps(streak_finder.to_dict()))
        assert result.to_dict() == expected

    def test_scan_config(self, tmp_path: Path, scan: ScanConfig) -> None:
        path = tmp_path / 'scan.json'

        scan.write(str(path))
        result = ScanConfig.read(str(path))

        assert result == scan
