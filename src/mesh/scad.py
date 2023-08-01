from attrs import define, field
from os import PathLike


@define
class Scad:
    file: PathLike = field()
    parameters: PathLike = field()
