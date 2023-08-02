from attrs import define, field

@define
class MeshProperties:
    name: str = field()
    density: float = field(default=1080.0)
    youngs_modulus: float = field(default=None)
    poissons_ratio: float = field(default=None)
    damping_factor: float = field(default=None)
    frozen_bounding_box: list = field(default=None)