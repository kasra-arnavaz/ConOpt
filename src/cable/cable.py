import torch
from cable.holes import Holes
from attrs import define, field


@define
class Cable:
    holes: Holes = field()
    pull_ratio: torch.Tensor = field(on_setattr=lambda self, attribute, value: torch.clip(value, min=0))
    stiffness: float = field(default=100.0)
    damping: float = field(default=0.01)

    def update(self):
        # a hack to force on_setattr to run when pull_ratio is changed by pointer through variables in optimizer step
        self.pull_ratio = self.pull_ratio

    @pull_ratio.default
    def _default_pull_ratio(self):
        return torch.tensor(0.0, dtype=torch.float32)

    @pull_ratio.validator
    def _check_type_and_shape(self, attribute, value):
        name = attribute.name
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Expected <{name}> to be of type torch.Tensor, got {type(value)}.")
        if len(value.shape) != 0:
            raise ValueError(f"Expected <{name}> to be a 0 dimensional tensor, got {len(value.shape)} dimensional.")
        if value.dtype != torch.float32:
            raise ValueError(f"Expected dtype of <{name}> to be torch.float32, got {value.dtype}.")
        if value < 0:
            raise ValueError(f"Expected the value of <{name}> to be greater than or equal to 0, got {value}")
