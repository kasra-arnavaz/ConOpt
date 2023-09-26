from attrs import define, field


@define
class ContactProperties:
    distance: float = field()
    ke: float = field()
    kd: float = field()
    kf: float = field()
    ground: bool = field()
