import torch
from attrs import define, field


@define
class Nodes:
    position: torch.Tensor = field()
    velocity: torch.Tensor = field(default=None)
    force: torch.Tensor = field(default=None)

    def __len__(self):
        return self.position.shape[0]

    @position.validator
    def _validate_position(self, attribute, value):
        self._validate_common_traits_of_attributes(name=attribute.name, value=value)

    @velocity.validator
    @force.validator
    def _validate_velocity(self, attribute, value):
        if value is not None:
            self._validate_common_traits_of_attributes(name=attribute.name, value=value)
            self._validate_equal_shape_with_position(name=attribute.name, value=value)

    def _validate_common_traits_of_attributes(self, name, value):
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Expected <{name}> to be of type torch.Tensor, got {type(value)}.")
        if len(value.shape) != 2:
            raise ValueError(f"Expected <{name}> to be a 2D tensor, got {len(value.shape)}D tensor.")
        if value.shape[-1] != 3:
            raise ValueError(f"Expected the last dimension of <{name}> to be of size 3, got {value.shape[-1]}.")
        if value.dtype != torch.float32:
            raise ValueError(f"Expected dtype of <{name}> to be torch.float32, got {value.dtype}.")

    def _validate_equal_shape_with_position(self, name, value):
        if value.shape != self.position.shape:
            raise ValueError(f"Expected <{name}> to have the same shape as <position>, got {value.shape})")
