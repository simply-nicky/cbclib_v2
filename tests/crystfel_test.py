"""Unit tests for CrystFEL geometry file parsing.

Tests cover:
- CrystFELFile: File reading and comment/whitespace handling
- AttributeParsers: Individual field parsers (int, float, bool, string, etc.)
- ParsingContainers: Complex nested structures (panels, regions, masks)
- parse_crystfel_file: Full geometry file parsing
- Detector: Geometry calculations and transformations
"""
from pathlib import Path
import pytest
from cbclib_v2 import read_crystfel
from cbclib_v2.test_util import parse_crystfel_file

class TestFullGeometryParsing:
    """Test complete geometry file parsing."""

    def test_file_with_regions_and_masks(self, tmp_path: Path) -> None:
        """Test CrystFELFile parsing with regions and masks."""
        geom_file = tmp_path / "test_with_regions.geom"
        content = (
            "; Geometry with regions and masks\n"
            "photon_energy = 12.0keV\n"
            "clen = 0.3621m\n"
            "res = 13333.3\n"
            "data = /data/data\n"
            "\n"
            "badregionA/min_x = -10.0\n"
            "badregionA/max_x = 10.0\n"
            "badregionA/min_y = -5.0\n"
            "badregionA/max_y = 5.0\n"
            "\n"
            "badregionB/min_fs = 100\n"
            "badregionB/max_fs = 200\n"
            "badregionB/min_ss = 150\n"
            "badregionB/max_ss = 250\n"
            "badregionB/panel = panel\n"
            "\n"
            "panel/corner_x = 0.0\n"
            "panel/corner_y = 0.0\n"
            "panel/fs = +1.0x+0.0y\n"
            "panel/ss = +0.0x+1.0y\n"
            "panel/min_fs = 0\n"
            "panel/max_fs = 1023\n"
            "panel/min_ss = 0\n"
            "panel/max_ss = 511\n"
            "panel/mask_data = /data/mask\n"
            "panel/mask_file = /path/to/mask.h5\n"
            "panel/mask_goodbits = 0xFFFF\n"
            "panel/mask_badbits = 0x0000\n"
        )
        geom_file.write_text(content)

        detector = parse_crystfel_file(str(geom_file))

        assert "panel" in detector.panels
        panel = detector.panels["panel"]
        assert panel.photon_energy.value() is not None
        assert panel.clen.value() is not None
        assert panel.res.value() is not None

        assert panel.region is not None
        assert panel.region.value() == {
            "min_fs": 0,
            "max_fs": 1023,
            "min_ss": 0,
            "max_ss": 511,
        }

        assert "regionA" in detector.bad_regions
        assert detector.bad_regions["regionA"].value() == {
            "min_x": -10.0,
            "max_x": 10.0,
            "min_y": -5.0,
            "max_y": 5.0,
        }

        assert "regionB" in detector.bad_regions
        assert detector.bad_regions["regionB"].value() == {
            "min_fs": 100,
            "max_fs": 200,
            "min_ss": 150,
            "max_ss": 250,
            "panel": "panel",
        }

        assert panel.masks is not None
        assert len(panel.masks) == 1
        mask = panel.masks[0]
        assert mask.value() == {
            "mask_data": "/data/mask",
            "mask_file": "/path/to/mask.h5",
            "mask_goodbits": 0xFFFF,
            "mask_badbits": 0x0000,
        }

    def test_file_parsing_with_comments_and_whitespace(self, tmp_path: Path) -> None:
        """Test CrystFELFile parsing handles comments, whitespace, and malformed lines."""
        geom_file = tmp_path / "test.geom"
        content = (
            "; Full comment line\n"
            "; Another comment\n"
            "photon_energy = 12.0keV\n"
            "clen = 0.3621m  ; inline comment\n"
            "res = 13333.3\n"
            "data = /data/data\n"
            "malformed_line_without_equals\n"
            "\n"
            "panel/corner_x = 0.0\n"
            "panel/corner_y\t=\t0.0\n"  # Test tab handling
            "panel/fs = +1.0x+0.0y\n"
            "panel/ss = +0.0x+1.0y\n"
            "panel/min_fs   =   0  \n"  # Test extra whitespace
            "panel/max_fs = 1023\n"
            "panel/min_ss = 0\n"
            "panel/max_ss = 511\n"
        )
        geom_file.write_text(content)

        # Should parse successfully despite comments, whitespace variations, and malformed lines
        detector = parse_crystfel_file(str(geom_file))

        assert "panel" in detector.panels
        panel = detector.panels["panel"]
        assert panel.photon_energy.value() is not None
        assert panel.clen.value() is not None
        assert panel.res.value() is not None
        assert panel.region is not None
        assert panel.region.value() == {
            "min_fs": 0,
            "max_fs": 1023,
            "min_ss": 0,
            "max_ss": 511,
        }

    def test_simple_single_panel(self, tmp_path: Path) -> None:
        """Test parsing simple single-panel geometry."""
        geom_file = tmp_path / "simple.geom"
        content = (
            "; Simple single panel geometry\n"
            "photon_energy = 12.0keV\n"
            "clen = 0.3621m\n"
            "res = 13333.3\n"
            "data = /data/data\n"
            "dim0 = %\n"
            "dim2 = ss\n"
            "dim3 = fs\n"
            "\n"
            "panel/dim1 = 0\n"
            "panel/corner_x = 0.0\n"
            "panel/corner_y = 0.0\n"
            "panel/fs = +1.0x+0.0y\n"
            "panel/ss = +0.0x+1.0y\n"
            "panel/min_fs = 0\n"
            "panel/max_fs = 1023\n"
            "panel/min_ss = 0\n"
            "panel/max_ss = 511\n"
        )
        geom_file.write_text(content)

        detector = parse_crystfel_file(str(geom_file))

        assert "panel" in detector.panels
        panel = detector.panels["panel"]
        assert panel.photon_energy.value() is not None
        assert panel.clen.value() is not None

        detector = read_crystfel(str(geom_file))
        assert detector.shape == (1, 512, 1024)
        assert detector.panels['panel'].shape == (1, 512, 1024)

    def test_multi_panel_geometry(self, tmp_path: Path) -> None:
        """Test parsing multi-panel geometry."""
        geom_file = tmp_path / "multi.geom"
        content = (
            "; Multi-panel geometry\n"
            "photon_energy = 12.0keV\n"
            "clen = 0.3621m\n"
            "res = 13333.3\n"
            "data = /data/data\n"
            "dim0 = %\n"
            "dim1 = ss\n"
            "dim2 = fs\n"
            "\n"
            "panel0/corner_x = 0.0\n"
            "panel0/corner_y = 0.0\n"
            "panel0/fs = +1.0x+0.0y\n"
            "panel0/ss = +0.0x+1.0y\n"
            "panel0/min_fs = 0\n"
            "panel0/max_fs = 511\n"
            "panel0/min_ss = 0\n"
            "panel0/max_ss = 511\n"
            "\n"
            "panel1/corner_x = 550.0\n"
            "panel1/corner_y = 0.0\n"
            "panel1/fs = +1.0x+0.0y\n"
            "panel1/ss = +0.0x+1.0y\n"
            "panel1/min_fs = 512\n"
            "panel1/max_fs = 1023\n"
            "panel1/min_ss = 0\n"
            "panel1/max_ss = 511\n"
        )
        geom_file.write_text(content)

        detector = parse_crystfel_file(str(geom_file))

        assert len(detector.panels) >= 2

        detector = read_crystfel(str(geom_file))
        assert detector.shape == (512, 1024)
        assert detector.panels['panel0'].shape == (512, 512)
        assert detector.panels['panel1'].shape == (512, 512)

class TestRealGeometryFile:
    """Test with actual geometry file from experiments."""

    def test_parse_swissfel_geometry(self) -> None:
        """Test parsing actual SwissFEL geometry file if available."""
        geom_path = Path(
            "/gpfs/cfel/user/nivanov/cbclib_v2/experiments/swissfel/geometry/"
            "JF16M_360mm_xy+rot+coff.geom"
        )

        if not geom_path.exists():
            pytest.skip("SwissFEL geometry file not available")

        detector = read_crystfel(str(geom_path))

        # Check first panel
        assert len(detector.panels) == 32
        assert detector.shape == (16448, 1030)
        assert detector.indices().shape == (4433, 4218)
        for panel in detector.panels.values():
            assert panel.shape == (514, 1030)

    def test_parse_jungfrau_geometry(self) -> None:
        """Test parsing JUNGFRAU geometry file if available."""
        geom_path = Path(
            "/gpfs/cfel/user/nivanov/cbclib_v2/experiments/exfel/geometry/"
            "jungfrau_4456_v1.geom"
        )

        if not geom_path.exists():
            pytest.skip("JUNGFRAU geometry file not available")

        detector = read_crystfel(str(geom_path))

        # Should have many panels (64 for JUNGFRAU 4M)
        assert len(detector.panels) == 64
        assert detector.shape == (8, 512, 1024)  # Example expected shape
        assert detector.indices().shape == (2173, 2398)
        for panel in detector.panels.values():
            assert panel.shape == (1, 256, 256)  # Example panel shape
