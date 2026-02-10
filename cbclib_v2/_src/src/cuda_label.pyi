from typing import Tuple
from ..annotations import BoolArray, CPIntArray
from .label import Structure

def label(inp: BoolArray, structure: Structure, npts: int=1) -> Tuple[CPIntArray, int]: ...
