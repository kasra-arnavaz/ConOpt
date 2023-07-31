import torch
from typing import List
from attrs import define, field, validators


@define
class Cables:
    stiffness: float = field(default=100.0)
    damping: float = field(default=0.01)
    holes_position_per_cable: List[torch.Tensor] = field(default=None)
    num_holes_per_cable: List[int] = field(default=None)

    def __len__(self):
        return len(self.holes_position_per_cable) if self.holes_position_per_cable is not None else 0
    
    @holes_position_per_cable.validator
    def _check_type_and_shape_and_set_num_holes_per_cable(self, attribute, value):
        if value is not None:
            if not isinstance(value, list):
                raise TypeError(f"Expected <{attribute.name}> to be of type list, got {type(value)}.")
            if not all(isinstance(item, torch.Tensor) for item in value):
                raise TypeError(f"Expected list of torch tensors for <{attribute.name}>.")
            if any(len(tensor.shape) != 2 for tensor in value):
                raise ValueError(f"Expected <{attribute.name}> tensors to be 2 dimensional.")
            if any(tensor.shape[-1] != 3 for tensor in value):
                raise ValueError(f"Expected the last dimension of <{attribute.name}> tensors to be 3.")
            self.num_holes_per_cable = [item.shape[0] for item in value]