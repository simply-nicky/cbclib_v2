from collections import OrderedDict
from copy import deepcopy
from itertools import chain
import math
import re
from sys import exc_info, modules
from typing import Dict, List, NamedTuple, Tuple, Type, TypedDict, Union, cast
from warnings import warn
import numpy as np
from numpy.typing import DTypeLike


TypeBeam = TypedDict(  # pylint: disable=invalid-name
    "TypeBeam",
    {
        "photon_energy": float,
        "photon_energy_from": str,
        "photon_energy_scale": float,
    },
    total=True,
)

TypePanel = TypedDict(  # pylint: disable=invalid-name
    "TypePanel",
    {
        "cnx": float,
        "cny": float,
        "coffset": float,
        "clen": float,
        "clen_from": str,
        "mask": str,
        "mask_file": str,
        "satmap": str,
        "satmap_file": str,
        "res": float,
        "badrow": str,
        "no_index": bool,
        "adu_per_photon": float,
        "max_adu": float,
        "data": str,
        "adu_per_eV": float,
        "dim_structure": List[Union[int, str, None]],
        "fsx": float,
        "fsy": float,
        "fsz": float,
        "ssx": float,
        "ssy": float,
        "ssz": float,
        "rail_x": float,
        "rail_y": float,
        "rail_z": float,
        "clen_for_centering": float,
        "xfs": float,
        "yfs": float,
        "xss": float,
        "yss": float,
        "orig_min_fs": int,
        "orig_max_fs": int,
        "orig_min_ss": int,
        "orig_max_ss": int,
        "w": int,
        "h": int,
    },
    total=True,
)

TypeBadRegion = TypedDict(  # pylint: disable=invalid-name
    "TypeBadRegion",
    {
        "panel": str,
        "min_x": float,
        "max_x": float,
        "min_y": float,
        "max_y": float,
        "min_fs": int,
        "max_fs": int,
        "min_ss": int,
        "max_ss": int,
        "is_fsss": int,
    },
    total=True,
)

TypeDetector = TypedDict(  # pylint: disable=invalid-name
    "TypeDetector",
    {
        "panels": Dict[str, TypePanel],
        "bad": Dict[str, TypeBadRegion],
        "mask_good": int,
        "mask_bad": int,
        "rigid_groups": Dict[str, List[str]],
        "rigid_group_collections": Dict[str, List[str]],
        "furthest_out_panel": str,
        "furthest_out_fs": float,
        "furthest_out_ss": float,
        "furthest_in_panel": str,
        "furthest_in_fs": float,
        "furthest_in_ss": float,
    },
    total=True,
)

def _assplode_algebraic(value):
    # type: (str) -> List[str]
    # Re-implementation of assplode_algegraic from libcrystfel/src/detector.c.
    items = [
        item for item in re.split("([+-])", string=value.strip()) if item != ""
    ]  # type: List[str]
    if items and items[0] not in ("+", "-"):
        items.insert(0, "+")
    return ["".join((items[x], items[x + 1])) for x in range(0, len(items), 2)]

def _dir_conv(direction_x, direction_y, direction_z, value):
    # type: (float, float, float, str) -> List[float]
    # Re-implementation of dir_conv from libcrystfel/src/detector.c.
    direction = [
        direction_x,
        direction_y,
        direction_z,
    ]  # type: List[float]
    items = _assplode_algebraic(value)
    if not items:
        raise RuntimeError(f"Invalid direction: {value}.")
    for item in items:
        axis = item[-1]  # type: str
        if axis not in ("x", "y", "z"):
            raise RuntimeError(f"Invalid Symbol: {axis} (must be x, y or z).")
        if item[:-1] == "+":
            value = "1.0"
        elif item[:-1] == "-":
            value = "-1.0"
        else:
            value = item[:-1]
        if axis == "x":
            direction[0] = float(value)
        elif axis == "y":
            direction[1] = float(value)
        elif axis == "z":
            direction[2] = float(value)

    return direction

def _set_dim_structure_entry(key, value, panel):
    # type: (str, str, TypePanel) -> None
    # Re-implementation of set_dim_structure_entry from libcrystfel/src/events.c.
    if panel["dim_structure"] is not None:
        dim = panel["dim_structure"]  # type: List[Union[int, str, None]]
    else:
        dim = []
    try:
        dim_index = int(key[3])  # type: int
    except IndexError as exc:
        raise RuntimeError("'dim' must be followed by a number, e.g. 'dim0')") from exc
    except ValueError as exc:
        raise RuntimeError(f"Invalid dimension number {key[3]}") from exc
    if dim_index > len(dim) - 1:
        for _ in range(len(dim), dim_index + 1):
            dim.append(None)
    if value in ("ss", "fs", "%"):
        dim[dim_index] = value
    elif value.isdigit():
        dim[dim_index] = int(value)
    else:
        raise RuntimeError(f"Invalid dim entry: {value}.")
    panel["dim_structure"] = dim

def _parse_field_for_panel(  # pylint: disable=too-many-branches, too-many-statements
    key,  # type: str
    value,  # type: str
    panel,  # type: TypePanel
    panel_name,  # type: str
    detector,  # type: TypeDetector
):
    # type: (...) -> None
    # Re-implementation of parse_field_for_panel from libcrystfel/src/detector.c.
    if key == "min_fs":
        panel["orig_min_fs"] = int(value)
    elif key == "max_fs":
        panel["orig_max_fs"] = int(value)
    elif key == "min_ss":
        panel["orig_min_ss"] = int(value)
    elif key == "max_ss":
        panel["orig_max_ss"] = int(value)
    elif key == "corner_x":
        panel["cnx"] = float(value)
    elif key == "corner_y":
        panel["cny"] = float(value)
    elif key == "rail_direction":
        try:
            panel["rail_x"], panel["rail_y"], panel["rail_z"] = _dir_conv(
                direction_x=panel["rail_x"],
                direction_y=panel["rail_y"],
                direction_z=panel["rail_z"],
                value=value,
            )
        except RuntimeError as exc:
            raise RuntimeError("Invalid rail direction. ", exc) from exc
    elif key == "clen_for_centering":
        panel["clen_for_centering"] = float(value)
    elif key == "adu_per_eV":
        panel["adu_per_eV"] = float(value)
    elif key == "adu_per_photon":
        panel["adu_per_photon"] = float(value)
    elif key == "rigid_group":
        if value in detector["rigid_groups"]:
            if panel_name not in detector["rigid_groups"][value]:
                detector["rigid_groups"][value].append(panel_name)
        else:
            detector["rigid_groups"][value] = [
                panel_name,
            ]
    elif key == "clen":
        try:
            panel["clen"] = float(value)
            panel["clen_from"] = ""
        except ValueError:
            panel["clen"] = -1
            panel["clen_from"] = value
    elif key == "data":
        if not value.startswith("/"):
            raise RuntimeError(f"Invalid data location: {value}")
        panel["data"] = value
    elif key == "mask":
        if not value.startswith("/"):
            raise RuntimeError(f"Invalid data location: {value}")
        panel["mask"] = value
    elif key == "mask_file":
        panel["mask_file"] = value
    elif key == "saturation_map":
        panel["satmap"] = value
    elif key == "saturation_map_file":
        panel["satmap_file"] = value
    elif key == "coffset":
        panel["coffset"] = float(value)
    elif key == "res":
        panel["res"] = float(value)
    elif key == "max_adu":
        panel["max_adu"] = float(value)
    elif key == "badrow_direction":
        if value == "x":
            panel["badrow"] = "f"
        elif value == "y":
            panel["badrow"] = "s"
        elif value == "f":
            panel["badrow"] = "f"
        elif value == "s":
            panel["badrow"] = "s"
        elif value == "-":
            panel["badrow"] = "-"
        else:
            print("badrow_direction must be x, t, f, s, or '-'")
            print("Assuming '-'.")
            panel["badrow"] = "-"
    elif key == "no_index":
        panel["no_index"] = bool(value)
    elif key == "fs":
        try:
            panel["fsx"], panel["fsy"], panel["fsz"] = _dir_conv(
                direction_x=panel["fsx"],
                direction_y=panel["fsy"],
                direction_z=panel["fsz"],
                value=value,
            )
        except RuntimeError as exc:
            raise RuntimeError("Invalid fast scan direction.", exc) from exc
    elif key == "ss":
        try:
            panel["ssx"], panel["ssy"], panel["ssz"] = _dir_conv(
                direction_x=panel["ssx"],
                direction_y=panel["ssy"],
                direction_z=panel["ssz"],
                value=value,
            )
        except RuntimeError as exc:
            raise RuntimeError("Invalid slow scan direction.", exc) from exc
    elif key.startswith("dim"):
        _set_dim_structure_entry(key=key, value=value, panel=panel)
    else:
        raise RuntimeError(f"Unrecognized field: {key}")

def _parse_toplevel(
    key,  # type: str
    value,  # type: str
    detector,  # type: TypeDetector
    beam,  # type: TypeBeam
    panel,  # type: TypePanel
    hdf5_peak_path,  # type: str
):  # pylint: disable=too-many-branches
    # type: (...) -> str
    # Re-implementation of parse_toplevel from libcrystfel/src/detector.c.
    if key == "mask_bad":
        try:
            detector["mask_bad"] = int(value)
        except ValueError:
            detector["mask_bad"] = int(value, base=16)
    elif key == "mask_good":
        try:
            detector["mask_good"] = int(value)
        except ValueError:
            detector["mask_good"] = int(value, base=16)
    elif key == "coffset":
        panel["coffset"] = float(value)
    elif key == "photon_energy":
        if value.startswith("/"):
            beam["photon_energy"] = 0.0
            beam["photon_energy_from"] = value
        else:
            beam["photon_energy"] = float(value)
            beam["photon_energy_from"] = ""
    elif key == "photon_energy_scale":
        beam["photon_energy_scale"] = float(value)
    elif key == "peak_info_location":
        hdf5_peak_path = value
    elif key.startswith("rigid_group") and not key.startswith("rigid_group_collection"):
        detector["rigid_groups"][key[12:]] = value.split(",")
    elif key.startswith("rigid_group_collection"):
        detector["rigid_group_collections"][key[23:]] = value.split(",")
    else:
        _parse_field_for_panel(
            key=key, value=value, panel=panel, panel_name="", detector=detector
        )

    return hdf5_peak_path

def _check_bad_fsss(bad_region, is_fsss):
    # type: (TypeBadRegion, int) -> None
    # Re-implementation of check_bad_fsss from libcrystfel/src/detector.c.
    if bad_region["is_fsss"] == 99:
        bad_region["is_fsss"] = is_fsss
        return

    if is_fsss != bad_region["is_fsss"]:
        raise RuntimeError("You can't mix x/y and fs/ss in a bad region")


def _parse_field_bad(key, value, bad):
    # type: (str, str, TypeBadRegion) -> None
    # Re-implementation of parse_field_bad from libcrystfel/src/detector.c.
    if key == "min_x":
        bad["min_x"] = float(value)
        _check_bad_fsss(bad_region=bad, is_fsss=False)
    elif key == "max_x":
        bad["max_x"] = float(value)
        _check_bad_fsss(bad_region=bad, is_fsss=False)
    elif key == "min_y":
        bad["min_y"] = float(value)
        _check_bad_fsss(bad_region=bad, is_fsss=False)
    elif key == "max_y":
        bad["max_y"] = float(value)
        _check_bad_fsss(bad_region=bad, is_fsss=False)
    elif key == "min_fs":
        bad["min_fs"] = int(value)
        _check_bad_fsss(bad_region=bad, is_fsss=True)
    elif key == "max_fs":
        bad["max_fs"] = int(value)
        _check_bad_fsss(bad_region=bad, is_fsss=True)
    elif key == "min_ss":
        bad["min_ss"] = int(value)
        _check_bad_fsss(bad_region=bad, is_fsss=True)
    elif key == "max_ss":
        bad["max_ss"] = int(value)
        _check_bad_fsss(bad_region=bad, is_fsss=True)
    elif key == "panel":
        bad["panel"] = value
    else:
        raise RuntimeError(f"Unrecognized field: {key}")


def _check_point(  # pylint: disable=too-many-arguments
    panel_name,  # type: str
    panel,  # type: TypePanel
    fs_,  # type: int
    ss_,  # type: int
    min_d,  # type: float
    max_d,  # type: float
    detector,  # type: TypeDetector
):
    # type: (...) -> Tuple[float, float]
    # Re-implementation of check_point from libcrystfel/src/detector.c.
    xs_ = fs_ * panel["fsx"] + ss_ * panel["ssx"]  # type: float
    ys_ = fs_ * panel["fsy"] + ss_ * panel["ssy"]  # type: float
    rx_ = (xs_ + panel["cnx"]) / panel["res"]  # type: float
    ry_ = (ys_ + panel["cny"]) / panel["res"]  # type: float
    dist = math.sqrt(rx_ * rx_ + ry_ * ry_)  # type: float
    if dist > max_d:
        detector["furthest_out_panel"] = panel_name
        detector["furthest_out_fs"] = fs_
        detector["furthest_out_ss"] = ss_
        max_d = dist
    elif dist < min_d:
        detector["furthest_in_panel"] = panel_name
        detector["furthest_in_fs"] = fs_
        detector["furthest_in_ss"] = ss_
        min_d = dist

    return min_d, max_d


def _find_min_max_d(detector):
    # type: (TypeDetector) -> None
    # Re-implementation of find_min_max_d from libcrystfel/src/detector.c.
    min_d = float("inf")  # type: float
    max_d = 0.0  # type: float
    for panel_name, panel in detector["panels"].items():
        min_d, max_d = _check_point(
            panel_name=panel_name,
            panel=panel,
            fs_=0,
            ss_=0,
            min_d=min_d,
            max_d=max_d,
            detector=detector,
        )
        min_d, max_d = _check_point(
            panel_name=panel_name,
            panel=panel,
            fs_=panel["w"],
            ss_=0,
            min_d=min_d,
            max_d=max_d,
            detector=detector,
        )
        min_d, max_d = _check_point(
            panel_name=panel_name,
            panel=panel,
            fs_=0,
            ss_=panel["h"],
            min_d=min_d,
            max_d=max_d,
            detector=detector,
        )
        min_d, max_d = _check_point(
            panel_name=panel_name,
            panel=panel,
            fs_=panel["w"],
            ss_=panel["h"],
            min_d=min_d,
            max_d=max_d,
            detector=detector,
        )

class CrystFELGeometry(NamedTuple):
    """Collection of objects as returned by load_crystfel_geometry"""

    detector: TypeDetector
    beam: TypeBeam
    hdf5_peak_path: Union[str, None]

def load_crystfel_geometry(filename: str) -> CrystFELGeometry:
    """
    Loads a CrystFEL geometry file.

    This function is a re-implementation of the get_detector_geometry_2 function from
    CrystFEL. It reads information from a CrystFEL geometry file, which uses a
    key/value language, fully documented in the relevant
    `man page <http://www.desy.de/~twhite/crystfel/manual-crystfel_geometry.html>`_.
    This function returns objects whose content matches CrystFEL's internal
    representation of the information in the file (see the libcrystfel/src/detector.h
    and the libcrystfel/src/image.c files from CrystFEL's source code for more
    information).

    The code of this function is currently synchronized with the code of the function
    'get_detector_geometry_2' in CrystFEL at commit cff9159b4bc6.


    Arguments:

        filename (str): the absolute or relative path to a CrystFEL geometry file.

    Returns:

        Tuple[TypeDetector, TypeBeam, Union[str, None]]: a tuple with the information
        loaded from the file.

        The first entry in the tuple is a dictionary storing information strictly
        related to the detector geometry. The following is a brief description of the
        key/value pairs in the dictionary.

        **Detector-related key/pairs**

            **panels** the panels in the detector. The value corresponding to this key
            is a dictionary containing information about the panels that make up the
            detector. In the dictionary, the keys are the panel names, and the values
            are further dictionaries storing information about the panels.

            **bad**: the bad regions in the detector. The value corresponding to this
            key is a dictionary containing information about the bad regions in the
            detector. In the dictionary, the keys are the bad region names, and the
            values are further dictionaries storing information about the bad regions.

            **mask_bad**: the value used in a mask to label a pixel as bad.

            **mask_good**: the value used in a mask to label a pixel as good.

            **rigid_groups**: the rigid groups of panels in the detector. The value
            corresponding to this key is a dictionary containing information about the
            rigid groups in the detector. In the dictionary, the keys are the names
            of the rigid groups and the values are lists storing the names of the
            panels belonging to eachgroup.

            **rigid_groups_collections**: the collections of rigid groups of panels in
            the detector. The value corresponding to this key is a dictionary
            containing information about the rigid group collections in the detector.
            In the dictionary, the keys are the names of the rigid group collections
            and the values are lists storing the names of the rigid groups belonging to
            the collections.

            **furthest_out_panel**: the name of the panel where the furthest away pixel
            from the center of the reference system can be found.

            **furthest_out_fs**: the fs coordinate, within its panel, of the furthest
            away pixel from the center of the reference system.

            **furthest_out_ss**: the ss coordinate, within its panel, of the furthest
            away pixel from the center of the reference system.

            **furthest_in_panel**: the name of the panel where the closest pixel to the
            center of the reference system can be found.

            **furthest_in_fs**: the fs coordinate, within its panel, of the closest
            pixel to the center of the reference system.

            **furthest_in_ss**: the ss coordinate, within its panel, of the closest
            pixel to the center of the reference system.

        **Panel-related key/pairs**

            **cnx**: the x location of the corner of the panel in the reference system.

            **cny**: the y location of the corner of the panel in the reference system.

            **clen**: the distance, as reported by the facility, of the sample
            interaction point from the corner of the first pixel in the panel .

            **clen_from**: the location of the clen information in a data file, in
            case the information must be extracted from it.

            **coffset**: the offset to be applied to the clen value to determine the
            real distance of the panel from the interaction point.

            **mask**: the location of the mask data for the panel in a data file.

            **mask_file**: the data file in which the mask data for the panel can be
            found.

            **satmap**: the location of the per-pixel saturation map for the panel in a
            data file.

            **satmap_file**: the data file in which the per-pixel saturation map for
            the panel can be found.

            **res**: the resolution of the panel in pixels per meter.

            **badrow**: the readout direction for the panel, for filtering out clusters
            of peaks. The value corresponding to this key is either 'x' or 'y'.

            **no_index**: wether the panel should be considered entirely bad. The panel
            will be considered bad if the value corresponding to this key is non-zero.

            **adu_per_photon**: the number of detector intensity units per photon for
            the panel.

            **max_adu**: the detector intensity unit value above which a pixel of the
            panel should be considered unreliable.

            **data**: the location, in a data file, of the data block where the panel
            data is stored.

            **adu_per_eV**: the number of detector intensity units per eV of photon
            energy for the panel.

            **dim_structure**: a description of the structure of the data block for the
            panel. The value corresponding to this key is a list of strings describing
            the meaning of each axis in the data block. See the
            `crystfel_geometry \
            <http://www.desy.de/~twhite/crystfel/manual-crystfel_geometry.html>`__ man
            page for a detailed explanation.

            **fsx**: the fs->x component of the matrix transforming pixel indexes to
            detector reference system coordinates.

            **fsy**: the fs->y component of the matrix transforming pixel indexes to
            detector reference system coordinates.

            **fsz**: the fs->z component of the matrix transforming pixel indexes to
            detector reference system coordinates.

            **ssx**: the ss->x component of the matrix transforming pixel indexes to
            detector reference system coordinates.

            **ssy**: the ss->y component of the matrix transforming pixel indexes to
            detector reference system coordinates.

            **ssz**: the ss->z component of the matrix transforming pixel indexes to
            detector reference system coordinates.

            **rail_x**: the x component, with respect to the reference system, of the
            direction of the rail along which the detector can be moved.

            **rail_y**: the y component, with respect to the reference system, of the
            direction of the rail along which the detector can be moved.

            **rail_z**: the z component, with respect to the reference system, of the
            direction of the rail along which the detector can be moved.

            **clen_for_centering**: the value of clen at which the beam hits the
            detector at the origin of the reference system.

            **xfs**: the x->fs component of the matrix transforming detector reference
            system coordinates to pixel indexes.

            **yfs**: the y->fs component of the matrix transforming detector reference
            system coordinates to pixel indexes.

            **xss**: the x->ss component of the matrix transforming detector reference
            system coordinates to pixel indexes.

            **yss**: the y->ss component of the matrix transforming detector reference
            system coordinates to pixel indexes.

            **orig_min_fs**: the initial fs index of the location of the panel data in
            the data block where it is stored.

            **orig_max_fs**: the final fs index of the location of the panel data in
            the data block where it is stored.

            **orig_min_ss**: the initial ss index of the location of the panel data in
            the data block where it is stored.

            **orig_max_ss**: the final fs index of the location of the panel data in
            the data block where it is stored.

            **w**: the width of the panel in pixels.

            **h**: the width of the panel in pixels.

        **Bad region-related key/value pairs**

            **panel**: the name of the panel in which the bad region lies.

            **min_x**: the initial x coordinate of the bad region in the detector
            reference system.

            **max_x**: the final x coordinate of the bad region in the detector
            referencesystem.

            **min_y**: the initial y coordinate of the bad region in the detector
            reference system.

            **max_y**: the final y coordinate of the bad region in the detector
            reference system.

            **min_fs**: the initial fs index of the location of the bad region in the
            block where the panel data is stored.

            **max_fs**: the final fs index of the location of the bad region in the
            block where the panel data is stored.

            **min_ss**: the initial ss index of the location of the bad region in the
            block where the panel data is stored.

            **max_ss**: the final ss index of the location of the bad region in the
            block where the panel data is stored.

            **is_fsss**: whether the fs,ss definition of the bad region is the valid
            one (as opposed to the x,y-based one). If the value corresponding to this
            key is True, the fs,ss-based definition of the bad region should be
            considered the valid one. Otherwise, the definition in x,y coordinates must
            be honored.

        The second entry in the tuple is a dictionary storing information related to
        the beam properties. The following is a brief description of the key/value
        pairs in the dictionary.

            **photon_energy**: the photon energy of the beam in eV.

            **photon_energy_from**: the location of the photon energy information in a
            data file, in case the information must be extracted from it.

            **photon_energy_scale**: the scaling factor to be applied to the photon
            energy, in case the provided energy value is not in eV.

        The third entry in the tuple is a string storing the HDF5 path where
        information about detected Bragg peaks can be found in a data file. If the
        CrystFEL geometry file does not provide this information, an empty string is
        returned.
    """
    beam = {
        "photon_energy": 0.0,
        "photon_energy_from": "",
        "photon_energy_scale": 1.0,
    }  # type: TypeBeam
    detector = {
        "panels": OrderedDict(),
        "bad": OrderedDict(),
        "mask_good": 0,
        "mask_bad": 0,
        "rigid_groups": {},
        "rigid_group_collections": {},
        "furthest_out_panel": "",
        "furthest_out_fs": float("NaN"),
        "furthest_out_ss": float("NaN"),
        "furthest_in_panel": "",
        "furthest_in_fs": float("NaN"),
        "furthest_in_ss": float("NaN"),
    }  # type: TypeDetector
    default_panel = {
        "cnx": float("NaN"),
        "cny": float("NaN"),
        "coffset": 0.0,
        "clen": float("NaN"),
        "clen_from": "",
        "mask": "",
        "mask_file": "",
        "satmap": "",
        "satmap_file": "",
        "res": -1.0,
        "badrow": "-",
        "no_index": False,
        "adu_per_photon": float("NaN"),
        "max_adu": float("inf"),
        "data": "",
        "adu_per_eV": float("NaN"),
        "dim_structure": [],
        "fsx": 1.0,
        "fsy": 0.0,
        "fsz": 0.0,
        "ssx": 0.0,
        "ssy": 1.0,
        "ssz": 0.0,
        "rail_x": float("NaN"),
        "rail_y": float("NaN"),
        "rail_z": float("NaN"),
        "clen_for_centering": float("NaN"),
        "xfs": 0.0,
        "yfs": 1.0,
        "xss": 1.0,
        "yss": 0.0,
        "orig_min_fs": -1,
        "orig_max_fs": -1,
        "orig_min_ss": -1,
        "orig_max_ss": -1,
        "w": 0,
        "h": 0,
    }  # type: TypePanel
    default_bad_region = {
        "panel": "",
        "min_x": float("NaN"),
        "max_x": float("NaN"),
        "min_y": float("NaN"),
        "max_y": float("NaN"),
        "min_fs": 0,
        "max_fs": 0,
        "min_ss": 0,
        "max_ss": 0,
        "is_fsss": 99,
    }  # type: TypeBadRegion
    default_dim = ["ss", "fs"]  # type: List[Union[int, str, None]]
    hdf5_peak_path = ""  # type: str
    try:
        with open(filename, mode="r") as file_handle:
            file_lines = file_handle.readlines()  # type: List[str]
            for line in file_lines:
                if line.startswith(";"):
                    continue
                line_without_comments = line.strip().split(";")[0]  # type: str
                line_items = re.split(
                    pattern="([ \t])", string=line_without_comments
                )  # type: List[str]
                line_items = [
                    item for item in line_items if item not in ("", " ", "\t")
                ]
                if len(line_items) < 3:
                    continue
                value = "".join(line_items[2:])  # type: str
                if line_items[1] != "=":
                    continue
                path = re.split("(/)", line_items[0])  # type: List[str]
                path = [item for item in path if item not in "/"]
                if len(path) < 2:
                    hdf5_peak_path = _parse_toplevel(
                        key=line_items[0],
                        value=value,
                        detector=detector,
                        beam=beam,
                        panel=default_panel,
                        hdf5_peak_path=hdf5_peak_path,
                    )
                    continue
                if path[0].startswith("bad"):
                    if path[0] in detector["bad"]:
                        curr_bad = detector["bad"][path[0]]
                    else:
                        curr_bad = deepcopy(default_bad_region)
                        detector["bad"][path[0]] = curr_bad
                    _parse_field_bad(key=path[1], value=value, bad=curr_bad)
                else:
                    if path[0] in detector["panels"]:
                        curr_panel = detector["panels"][path[0]]
                    else:
                        curr_panel = deepcopy(default_panel)
                        detector["panels"][path[0]] = curr_panel
                    _parse_field_for_panel(
                        key=path[1],
                        value=value,
                        panel=curr_panel,
                        panel_name=path[0],
                        detector=detector,
                    )
            if not detector["panels"]:
                raise RuntimeError("No panel descriptions in geometry file.")
            num_placeholders_in_panels = -1  # type: int
            for panel in detector["panels"].values():
                if panel["dim_structure"] is not None:
                    curr_num_placeholders = panel["dim_structure"].count(
                        "%"
                    )  # type: int
                else:
                    curr_num_placeholders = 0

                if num_placeholders_in_panels == -1:
                    num_placeholders_in_panels = curr_num_placeholders
                else:
                    if curr_num_placeholders != num_placeholders_in_panels:
                        raise RuntimeError(
                            "All panels' data and mask entries must have the same "
                            "number of placeholders."
                        )
            num_placeholders_in_masks = -1  # type: int
            for panel in detector["panels"].values():
                if panel["mask"] is not None:
                    curr_num_placeholders = panel["mask"].count("%")
                else:
                    curr_num_placeholders = 0

                if num_placeholders_in_masks == -1:
                    num_placeholders_in_masks = curr_num_placeholders
                else:
                    if curr_num_placeholders != num_placeholders_in_masks:
                        raise RuntimeError(
                            "All panels' data and mask entries must have the same "
                            "number of placeholders."
                        )
            if num_placeholders_in_masks > num_placeholders_in_panels:
                raise RuntimeError(
                    "Number of placeholders in mask cannot be larger the number than "
                    "for data."
                )
            dim_length = -1  # type: int
            for panel_name, panel in detector["panels"].items():
                if len(panel["dim_structure"]) == 0:
                    panel["dim_structure"] = deepcopy(default_dim)
                found_ss = 0  # type: int
                found_fs = 0  # type: int
                found_placeholder = 0  # type: int
                for dim_index, entry in enumerate(panel["dim_structure"]):
                    if entry is None:
                        raise RuntimeError(
                            f"Dimension {dim_index} for panel {panel_name} is undefined."
                        )
                    if entry == "ss":
                        found_ss += 1
                    elif entry == "fs":
                        found_fs += 1
                    elif entry == "%":
                        found_placeholder += 1
                if found_ss != 1:
                    raise RuntimeError(
                        f"Exactly one slow scan dim coordinate is needed (found {found_ss} for " \
                        f"panel {panel_name})."
                    )
                if found_fs != 1:
                    raise RuntimeError(
                        f"Exactly one fast scan dim coordinate is needed (found {found_fs} for "
                        f"panel {panel_name})."
                    )
                if found_placeholder > 1:
                    raise RuntimeError(
                        "Only one placeholder dim coordinate is allowed. Maximum one " \
                        "placeholder dim coordinate is allowed " \
                        f"(found {found_placeholder} for panel {panel_name})"
                    )
                if dim_length == -1:
                    dim_length = len(panel["dim_structure"])
                elif dim_length != len(panel["dim_structure"]):
                    raise RuntimeError(
                        "Number of dim coordinates must be the same for all panels."
                    )
                if dim_length == 1:
                    raise RuntimeError(
                        "Number of dim coordinates must be at least two."
                    )
            for panel_name, panel in detector["panels"].items():
                if panel["orig_min_fs"] < 0:
                    raise RuntimeError(
                        f"Please specify the minimum fs coordinate for panel {panel_name}."
                    )
                if panel["orig_max_fs"] < 0:
                    raise RuntimeError(
                        f"Please specify the maximum fs coordinate for panel {panel_name}."
                    )
                if panel["orig_min_ss"] < 0:
                    raise RuntimeError(
                        f"Please specify the minimum ss coordinate for panel {panel_name}."
                    )
                if panel["orig_max_ss"] < 0:
                    raise RuntimeError(
                        f"Please specify the maximum ss coordinate for panel {panel_name}."
                    )
                if panel["cnx"] is None:
                    raise RuntimeError(
                        f"Please specify the corner X coordinate for panel {panel_name}."
                    )
                if panel["clen"] is None and panel["clen_from"] is None:
                    raise RuntimeError(
                        f"Please specify the camera length for panel {panel_name}."
                    )
                if panel["res"] < 0:
                    raise RuntimeError(
                        f"Please specify the resolution or panel {panel_name}."
                    )
                if panel["adu_per_eV"] is None and panel["adu_per_photon"] is None:
                    raise RuntimeError(
                        "Please specify either adu_per_eV or adu_per_photon for panel " \
                        f"{panel_name}."
                    )
                if panel["clen_for_centering"] is None and panel["rail_x"] is not None:
                    raise RuntimeError(
                        "You must specify clen_for_centering if you specify the rail " \
                        f"direction (panel {panel_name})"
                    )
                if panel["rail_x"] is None:
                    panel["rail_x"] = 0.0
                    panel["rail_y"] = 0.0
                    panel["rail_z"] = 1.0
                if panel["clen_for_centering"] is None:
                    panel["clen_for_centering"] = 0.0
                panel["w"] = panel["orig_max_fs"] - panel["orig_min_fs"] + 1
                panel["h"] = panel["orig_max_ss"] - panel["orig_min_ss"] + 1
            for bad_region_name, bad_region in detector["bad"].items():
                if bad_region["is_fsss"] == 99:
                    raise RuntimeError(
                        "Please specify the coordinate ranges for bad " \
                        f"region {bad_region_name}."
                    )
            for group in detector["rigid_groups"]:
                for name in detector["rigid_groups"][group]:
                    if name not in detector["panels"]:
                        raise RuntimeError(
                            "Cannot add panel to rigid_group. Panel not " \
                            f"found: {name}"
                        )
            for group_collection in detector["rigid_group_collections"]:
                for name in detector["rigid_group_collections"][group_collection]:
                    if name not in detector["rigid_groups"]:
                        raise RuntimeError(
                            "Cannot add rigid_group to collection. Rigid group not " \
                            f"found: {name}"
                        )
            for panel in detector["panels"].values():
                d__ = (
                    panel["fsx"] * panel["ssy"] - panel["ssx"] * panel["fsy"]
                )  # type: float
                if d__ == 0.0:
                    raise RuntimeError("Panel {} transformation is singular.")
                panel["xfs"] = panel["ssy"] / d__
                panel["yfs"] = panel["ssx"] / d__
                panel["xss"] = panel["fsy"] / d__
                panel["yss"] = panel["fsx"] / d__
            _find_min_max_d(detector)
    except (IOError, OSError) as exc:
        exc_type, exc_value = exc_info()[:2]
        raise RuntimeError(
            f"The following error occurred while reading the {filename} geometry"
            f"file {cast(Type[BaseException], exc_type).__name__}: {exc_value}"
        ) from exc

    return CrystFELGeometry(detector=detector, beam=beam, hdf5_peak_path=hdf5_peak_path)

class SnappedGeometry:
    """Detector geometry approximated to align modules to a 2D grid

    The coordinates used in this class are (y, x) suitable for indexing a
    Numpy array; this does not match the (x, y, z) coordinates in the more
    precise geometry above.
    """
    def __init__(self, modules, geom, centre):
        self.modules = modules
        self.geom = geom
        self.centre = centre

        # The fragments here are already shifted so corner_idx starts from 0 in
        # each dim, so the max outer edges define the output image size.
        self.size_yx = tuple(np.max([
            np.array(frag.corner_idx) + np.array(frag.pixel_dims)
            for frag in chain(*modules)
        ], axis=0))

    def make_output_array(
            self, extra_shape: Tuple=(), dtype: DTypeLike=np.float32
        ) -> np.ndarray:
        """Make an output array for self.position_modules()
        """
        shape = extra_shape + self.size_yx
        if np.issubdtype(dtype, np.floating):
            return np.full(shape, np.nan, dtype=dtype)

        # zeros() is much faster than full() with 0 - part of the cost is just
        # deferred until we fill it, but it's probably still advantageous.
        return np.zeros(shape, dtype=dtype)

    def position_modules(self, data, out=None, threadpool=None):
        """Implementation for position_modules_fast
        """
        nmod = self.geom.expected_data_shape[0]
        if isinstance_no_import(data, 'xarray', 'DataArray'):
            # Input is an xarray labelled array
            modnos = data.coords.get('module')
            if modnos is None:
                raise ValueError(
                    "xarray arrays should have a dimension named 'module'"
                )
            modnos = modnos.values  # xarray -> numpy
            min_mod, max_mod = modnos.min(), modnos.max()
            if min_mod < 0 or max_mod >= nmod:
                raise ValueError(
                    f"module number labels should be in the range 0-{nmod-1} "
                    f"(found {min_mod}-{max_mod})"
                )
            assert data.shape[-2:] == self.geom.expected_data_shape[-2:]

            mod_dim_ix = data.dims.index('module')
            extra_shape = data.shape[:mod_dim_ix] + data.shape[mod_dim_ix+1:-2]
            get_mod_data = lambda i: data.sel(module=i).values
        else:
            # Input is a numpy array (or similar unlabelled array)
            assert data.shape[-3:] == self.geom.expected_data_shape
            modnos = range(nmod)
            extra_shape = data.shape[:-3]
            get_mod_data = lambda i: data[..., i, :, :]

        if out is None:
            out = self.make_output_array(extra_shape, data.dtype)
        else:
            assert out.shape == extra_shape + self.size_yx
            if not np.can_cast(data.dtype, out.dtype, casting='safe'):
                raise TypeError(f"{data.dtype} cannot be safely cast to {out.dtype}")

        copy_pairs = []
        for modno in modnos:
            module = self.modules[modno]
            tiles_data = self.geom.split_tiles(get_mod_data(modno))
            for tile, tile_data in zip(module, tiles_data):
                y, x = tile.corner_idx
                h, w = tile.pixel_dims

                copy_pairs.append((
                    out[..., y : y + h, x : x + w], tile.transform(tile_data)
                ))

        if threadpool is not None:
            def copy_data(pair):
                dst, src = pair
                dst[:] = src
            # concurrent.futures map() is async, so call list() to wait for it
            list(threadpool.map(copy_data, copy_pairs))
        else:
            for dst, src in copy_pairs:
                dst[:] = src

        return out, self.centre

class GeometryFragment:
    """Holds the 3D position & orientation of one detector tile

    corner_pos refers to the corner of the detector tile where the first pixel
    stored is located. The tile is assumed to be a rectangle of ss_pixels in
    the slow scan dimension and fs_pixels in the fast scan dimension.
    ss_vec and fs_vec are vectors for a step of one pixel in each dimension.

    The coordinates in this class are (x, y, z), in metres.
    """

    def __init__(self, corner_pos, ss_vec, fs_vec, ss_pixels, fs_pixels):
        self.corner_pos = corner_pos
        self.ss_vec = ss_vec
        self.fs_vec = fs_vec
        self.ss_pixels = ss_pixels
        self.fs_pixels = fs_pixels

    @classmethod
    def from_panel_dict(cls, d):
        res = d['res']
        corner_pos = np.array([d['cnx']/res, d['cny']/res, d['coffset']])
        ss_vec = np.array([d['ssx'], d['ssy'], d['ssz']]) / res
        fs_vec = np.array([d['fsx'], d['fsy'], d['fsz']]) / res
        ss_pixels = d['orig_max_ss'] - d['orig_min_ss'] + 1
        fs_pixels = d['orig_max_fs'] - d['orig_min_fs'] + 1
        return cls(corner_pos, ss_vec, fs_vec, ss_pixels, fs_pixels)

class DetectorGeometryBase:
    """Base class for detector geometry. Subclassed for specific detectors."""
    # Define in subclasses:
    detector_type_name = ''
    pixel_size = 0.0
    frag_ss_pixels = 0
    frag_fs_pixels = 0
    n_quads = 0
    n_modules = 0
    n_tiles_per_module = 0
    expected_data_shape = (0, 0, 0)
    _pixel_corners = np.array([  # pixel units; overridden for DSSC
        [0, 1, 1, 0],  # slow-scan
        [0, 0, 1, 1]   # fast-scan
    ])
    _draw_first_px_on_tile = 1  # Tile num of 1st pixel - overridden for LPD
    _pyfai_cls_name = None  # Name of class in extra_geom.pyfai

    @property
    def _pixel_shape(self):
        """Pixel (x, y) shape. Overridden for DSSC."""
        return np.array([1., 1.], dtype=np.float64) * self.pixel_size

    def __init__(self, modules, filename='No file', metadata=None):
        # List of lists (1 per module) of fragments (1 per tile)
        self.modules = modules
        # self.filename is metadata for plots, we don't read/write the file.
        # There are separate methods for reading and writing.
        self.filename = filename
        self.metadata = metadata if (metadata is not None) else {}
        self._snapped_cache = None

    @classmethod
    def _tile_slice(cls, tileno):
        """Implement in subclass: which part of module array each tile is.
        """
        raise NotImplementedError

    @staticmethod
    def split_tiles(module_data):
        """Split data from a detector module into tiles.

        Must be implemented in subclasses.
        """
        raise NotImplementedError

    @classmethod
    def _cfel_panels_by_data_coord(cls, panels: dict):
        """Arrange panel dicts from CrystFEL geometry by first data coordinate

        Index panels by which part of the data they refer to, rather than
        relying on names like p0a0.
        """
        res = {}
        for pname, info in panels.items():
            dims = info['dim_structure']
            ix_dims = [i for i in dims if isinstance(i, int)]
            if len(ix_dims) > 1:
                raise ValueError(f"Too many index dimensions for {pname}: {dims}")

            min_ss = info['orig_min_ss']
            if ix_dims:
                # Geometry for 3D data, modules stacked along separate axis
                modno = ix_dims[0]
            else:
                # Geometry for 2D data, modules concatenated along slow-scan axis
                modno, min_ss = divmod(min_ss, cls.expected_data_shape[1])

            info['panel_name'] = pname
            res[(modno, min_ss, info['orig_min_fs'])] = info

        return res

    @classmethod
    def from_crystfel_geom(cls, filename):
        """Read a CrystFEL format (.geom) geometry file.

        Returns a new geometry object.
        """
        cfel_geom = load_crystfel_geometry(filename)
        panels_by_data_coord = cls._cfel_panels_by_data_coord(
            cfel_geom.detector['panels']
        )
        n_modules = cls.n_modules
        if n_modules == 0:
            # Detector type with varying number of modules (e.g. JUNGFRAU)
            n_modules = max(c[0] for c in panels_by_data_coord) + 1

        modules = []
        panel_names_to_pNaM = {}
        for p in range(n_modules):
            tiles = []
            modules.append(tiles)
            for a in range(cls.n_tiles_per_module):
                ss_slice, fs_slice = cls._tile_slice(a)
                d = panels_by_data_coord[p, ss_slice.start, fs_slice.start]
                tiles.append(GeometryFragment.from_panel_dict(d))
                panel_names_to_pNaM[d['panel_name']] = f'p{p}a{a}'

        # Store some extra fields to write if we create another .geom file.
        # It's possible for these to have different values for different panels,
        # but it seems to be common to use them like headers, describing all
        # panels, and we're assuming that's the case here.
        cfel_md_keys = ('data', 'mask', 'adu_per_eV', 'clen')
        d1 = panels_by_data_coord[0, 0, 0]
        metadata = {'crystfel': {k: d1.get(k) for k in cfel_md_keys}}
        metadata['crystfel']['photon_energy'] = cfel_geom.beam['photon_energy']

        # Normalise description of bad regions, so we can output it correctly.
        # - Change panel names to uniform pNaM (panel N asic M) format
        # - If the file has a 2D layout (modules arranged along the slow-scan
        #   axis), convert slow-scan coordinates to a 3D layout.
        file_geom_is_2d = not any(isinstance(d, int) for d in d1['dim_structure'])
        adjusted_bad_regions = {}
        for bad_name, bad_d in cfel_geom.detector['bad'].items():
            panel_name = bad_d['panel']
            if panel_name:
                try:
                    bad_d['panel'] = panel_names_to_pNaM[panel_name]
                except KeyError:
                    warn("Discarding {bad_name}, no such panel {panel_name!r}")
                    continue
            if bad_d['is_fsss']:
                if not panel_name:
                    warn("Discarding {bad_name}, ss/fs region without panel name")
                    continue
                if file_geom_is_2d:
                    bad_d['min_ss'] %= cls.expected_data_shape[1]
                    bad_d['max_ss'] %= cls.expected_data_shape[1]
            adjusted_bad_regions[bad_name] = bad_d

        metadata['crystfel']['bad'] = adjusted_bad_regions

        return cls(modules, filename=filename, metadata=metadata)

    def _snapped(self):
        """Snap geometry to a 2D pixel grid

        This returns a new geometry object. The 'snapped' geometry is
        less accurate, but can assemble data into a 2D array more efficiently,
        because it doesn't do any interpolation.
        """
        if self._snapped_cache is None:
            modules = []
            for module in self.modules:
                tiles = [t.snap(px_shape=self._pixel_shape) for t in module]
                modules.append(tiles)
            centre = -np.min([t.corner_idx for t in chain(*modules)], axis=0)

            # Offset by centre to make all coordinates >= 0
            modules = [
                [t.offset(centre) for t in module]
                for module in modules
            ]
            self._snapped_cache = SnappedGeometry(modules, self, centre)
        return self._snapped_cache

    def position_modules(self, data, out=None, threadpool=None):
        """Assemble data from this detector according to where the pixels are.

        This approximates the geometry to align all pixels to a 2D grid.

        This method was previously called ``position_modules_fast``, and can
        still be called with that name.

        Parameters
        ----------

        data : ndarray or xarray.DataArray
          The last three dimensions should match the modules, then the
          slow scan and fast scan pixel dimensions. If an xarray labelled array
          is given, it must have a 'module' dimension.
        out : ndarray, optional
          An output array to assemble the image into. By default, a new
          array is allocated. Use :meth:`output_array_for_position` to
          create a suitable array.
          If an array is passed in, it must match the dtype of the data and the
          shape of the array that would have been allocated.
          Parts of the array not covered by detector tiles are not overwritten.
          In general, you can reuse an output array if you are assembling
          similar pulses or pulse trains with the same geometry.
        threadpool : concurrent.futures.ThreadPoolExecutor, optional
          If passed, parallelise copying data into the output image.
          By default, data for different tiles are copied serially.
          For a single 1 MPx image, the default appears to be faster, but for
          assembling a stack of several images at once, multithreading can help.

        Returns
        -------
        out : ndarray
          Array with one dimension fewer than the input.
          The last two dimensions represent pixel y and x in the detector space.
        centre : ndarray
          (y, x) pixel location of the detector centre in this geometry.
        """
        return self._snapped().position_modules(data, out=out, threadpool=threadpool)

    def output_array_for_position(
            self, extra_shape: Tuple=(), dtype: DTypeLike=np.float32
        ) -> np.ndarray:
        """Make an empty output array to use with position_modules

        You can speed up assembling images by reusing the same output array:
        call this once, and then pass the array as the ``out=`` parameter to
        :meth:`position_modules`. By default, it allocates a new array on
        each call, which can be slow.

        Parameters
        ----------

        extra_shape : tuple, optional
          By default, a 2D output array is generated, to assemble a single
          detector image. If you are assembling multiple pulses at once, pass
          ``extra_shape=(nframes,)`` to get a 3D output array.
        dtype : optional (Default: np.float32)
        """
        return self._snapped().make_output_array(extra_shape=extra_shape,
                                                 dtype=dtype)

def isinstance_no_import(obj, mod: str, cls: str):
    """Check if isinstance(obj, mod.cls) without loading mod"""
    m = modules.get(mod)
    if m is None:
        return False

    return isinstance(obj, getattr(m, cls))

class JUNGFRAUGeometry(DetectorGeometryBase):
    """Detector layout for flexible Jungfrau arrangements

     The base JUNGFRAU unit (and rigid group) in combined arrangements is the
     JF-500K module, which is an independent detector unit of 2 x 4 ASIC tiles.

     In the default orientation, the slow-scan dimension is y and the fast-scan
     dimension is x, so the data shape for one module is (y, x).
    """
    detector_type_name = 'JUNGFRAU'
    pixel_size = 7.5e-5   # 7.5e-5 metres = 75 micrometer = 0.075 mm
    frag_ss_pixels = 256  # pixels along slow scan axis within tile
    frag_fs_pixels = 256  # pixels along fast scan axis within tile
    expected_data_shape = (0, 512, 1024)  # num modules filled at instantiation
    n_tiles_per_module = 8
    _pyfai_cls_name = 'JUNGFRAU_EuXFEL'

    @staticmethod
    def split_tiles(module_data):
        # 2 rows of 4 ASICs each. This slicing is faster than np.split().
        return [
            module_data[..., :256, x:x+256] for x in range(0, 1024, 256)
        ] + [
            module_data[..., 256:, x:x+256] for x in range(0, 1024, 256)
        ]

    @classmethod
    def _tile_slice(cls, tileno):
        # Which part of the array is this tile?
        # tileno = 0 to 7
        tile_ss_offset = (tileno // 4) * cls.frag_ss_pixels
        tile_fs_offset = (tileno % 4) * cls.frag_fs_pixels
        ss_slice = slice(tile_ss_offset, tile_ss_offset + cls.frag_ss_pixels)
        fs_slice = slice(tile_fs_offset, tile_fs_offset + cls.frag_fs_pixels)
        return ss_slice, fs_slice

    def __init__(self, modules, filename='No file', metadata=None):
        super().__init__(modules, filename, metadata)
        self.expected_data_shape = (len(modules), 512, 1024)
        self.n_modules = len(modules)

    def position_modules(self, data, out=None, threadpool=None):
        if isinstance_no_import(data, 'xarray', 'DataArray'):
            # we shift module indices by one as JUNGFRAU labels modules starting from 1..
            # position_modules returns a numpy array so labels disapear anyway.
            data = data.copy(deep=False)
            data['module'] = data['module'] - 1
        return super().position_modules(data, out=out, threadpool=threadpool)
