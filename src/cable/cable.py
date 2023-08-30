import torch
from cable.holes import Holes
from attrs import define, field


@define
class Cable:
    holes: Holes = field()
    _pull_ratio: torch.Tensor = field()
    stiffness: float = field(default=100.0)
    damping: float = field(default=0.01)

    @property
    def pull_ratio(self):
        return torch.clip(self._pull_ratio, min=0)

    @pull_ratio.setter
    def pull_ratio(self, value):
        self._pull_ratio = value

    @_pull_ratio.default
    def _default_pull_ratio(self):
        return torch.tensor(0.0, dtype=torch.float32)

    @_pull_ratio.validator
    def _check_type_and_shape(self, attribute, value):
        name = attribute.name
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Expected <{name}> to be of type torch.Tensor, got {type(value)}.")
        if len(value.shape) != 0:
            raise ValueError(f"Expected <{name}> to be a 0 dimensional tensor, got {len(value.shape)} dimensional.")
        if value.dtype != torch.float32:
            raise ValueError(f"Expected dtype of <{name}> to be torch.float32, got {value.dtype}.")
