import torch
from attrs import define, field

@define
class Holes:
    position: torch.Tensor = field()

    def __len__(self):
        return self.position.shape[0]

    @position.validator
    def _validate_type_and_shape(self, attribute, value):
        name = attribute.name
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Expected <{name}> to be of type torch.Tensor, got {type(value)}.")
        if len(value.shape) != 2:
            raise ValueError(f"Expected <{name}> to be a 2D tensor, got {len(value.shape)}D tensor.")
        if value.shape[-1] != 3:
            raise ValueError(f"Expected the last dimension of <{name}> to be of size 3, got {value.shape[-1]}.")
        if value.dtype != torch.float32:
            raise ValueError(f"Expected dtype of <{name}> to be torch.float32, got {value.dtype}.")