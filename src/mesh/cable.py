import torch
from typing import List
from attrs import define, field, validators


@define
class Cable:
    stiffness: float = field(default=100.0)
    damping: float = field(default=0.01)
    holes_position: List[torch.Tensor] = field(default=None)

    @holes_position.validator
    def _check_type_and_shape(self, attribute, value):
        if value is not None:
            if not isinstance(value, list):
                raise TypeError(f"Expected <{attribute.name}> to be of type list, got {type(value)}.")
            if not all(isinstance(item, torch.Tensor) for item in value):
                raise TypeError(f"Expected list of torch tensors for <{attribute.name}>.")
            if any(len(tensor.shape) != 2 for tensor in value):
                raise ValueError(f"Expected <{attribute.name}> tensors to be 2 dimensional.")
            if any(tensor.shape[-1] != 3 for tensor in value):
                raise ValueError(f"Expected the last dimension of <{attribute.name}> tensors to be 3.")
