from attrs import define, field


@define
class ContactProperties:
    distance: float = field()
    ke: float = field()
    kd: float = field()
    kf: float = field()
    ground: bool = field()


    def __new__(cls, *args, **kwargs):
        if any(arg is None for arg in args) or any(value is None for value in kwargs.values()):
            return None
        return super(ContactProperties, cls).__new__(cls)